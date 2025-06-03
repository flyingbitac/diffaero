from typing import Tuple, Dict, Union, Optional, List
import os

from omegaconf import DictConfig, OmegaConf
import torch
from torch import Tensor
import torch.nn as nn
from tensordict import TensorDict

from quaddif.network import build_network

class AgentBase(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: Union[int, Tuple[int, Tuple[int, int]]]
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.is_rnn_based = cfg.name.lower() == "rnn" or cfg.name.lower() == "rcnn"


class DeterministicActor(AgentBase):
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: Union[int, Tuple[int, Tuple[int, int]]],
        action_dim: int
    ):
        super().__init__(cfg, obs_dim)
        self.actor = build_network(cfg, obs_dim, action_dim, output_act=nn.Tanh())
        
    def forward(self, obs: Union[Tensor, Tuple[Tensor, Tensor]], hidden: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        return self.actor(obs, hidden=hidden)
    
    def save(self, path: str):
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))

    def load(self, path: str):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth")))

    def reset(self, indices: Tensor):
        self.actor.reset(indices)
    
    def detach(self):
        self.actor.detach()


class StochasticActor(AgentBase):
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: Union[int, Tuple[int, Tuple[int, int]]],
        action_dim: int
    ):
        super().__init__(cfg, obs_dim)
        self.actor_mean = build_network(cfg, obs_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, obs, sample=None, test=False, hidden=None):
        # type: (Union[Tensor, Tuple[Tensor, Tensor]], Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        action_mean = self.actor_mean(obs, hidden=hidden)
        LOG_STD_MAX = 2
        LOG_STD_MIN = -5
        action_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            torch.tanh(self.actor_logstd) + 1)  # From SpinUp / Denis Yarats
        action_std = torch.exp(action_logstd).expand_as(action_mean)
        probs = torch.distributions.Normal(action_mean, action_std)
        if sample is None and not test:
            sample = torch.randn_like(action_mean) * action_std + action_mean
        elif test:
            sample = action_mean.detach()
        action = torch.tanh(sample)
        logprob = probs.log_prob(sample) - torch.log(1. - action.pow(2) + 1e-8)
        # entropy = (-logprob * logprob.exp()).sum(-1)
        entropy = probs.entropy().sum(-1)
        return action, sample, logprob.sum(-1), entropy
    
    def save(self, path: str):
        torch.save(
            {"actor_mean": self.actor_mean.state_dict(),
             "actor_logstd": self.actor_logstd}, os.path.join(path, "actor.pth"))

    def load(self, path: str):
        actor = torch.load(os.path.join(path, "actor.pth"), weights_only=True)
        self.actor_mean.load_state_dict(actor["actor_mean"])
        self.actor_logstd.data.copy_(actor["actor_logstd"].to(self.actor_logstd.device))

    def reset(self, indices: Tensor):
        self.actor_mean.reset(indices)
    
    def detach(self):
        self.actor_mean.detach()


class CriticV(AgentBase):
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: Union[int, Tuple[int, Tuple[int, int]]]
    ):
        super().__init__(cfg, obs_dim)
        self.critic = build_network(cfg, obs_dim, 1)
    
    def forward(self, obs: Union[Tensor, Tuple[Tensor, Tensor]], hidden: Optional[Tensor] = None) -> Tensor:
        return self.critic(obs, hidden=hidden).squeeze(-1)
    
    def save(self, path: str):
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
    
    def load(self, path: str):
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth"), weights_only=True))

    def reset(self, indices: Tensor):
        self.critic.reset(indices)
    
    def detach(self):
        self.critic.detach()


class CriticQ(AgentBase):
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: Union[int, Tuple[int, Tuple[int, int]]],
        action_dim: int
    ):
        super().__init__(cfg, obs_dim)
        if not isinstance(obs_dim, int):
            input_dim = (obs_dim[0] + action_dim, obs_dim[1])
        else:
            input_dim = obs_dim + action_dim
        self.critic = build_network(cfg, input_dim, 1)
    
    def forward(self, obs: Union[Tensor, Tuple[Tensor, Tensor]], action: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        return self.critic(obs, action, hidden=hidden).squeeze(-1)
    
    def save(self, path: str):
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
    
    def load(self, path: str):
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth"), weights_only=True))

    def reset(self, indices: Tensor):
        self.critic.reset(indices)
    
    def detach(self):
        self.critic.detach()


class ActorCriticBase(AgentBase):
    actor: Union[DeterministicActor, StochasticActor]
    critic: Union[CriticV, CriticQ]
    
    def save(self, path: str):
        self.actor.save(path)
        self.critic.save(path)
    
    def load(self, path: str):
        self.actor.load(path)
        self.critic.load(path)

    def reset(self, indices: Tensor):
        self.actor.reset(indices)
        self.critic.reset(indices)
    
    def detach(self):
        self.actor.detach()
        self.critic.detach()
        

class StochasticActorCriticV(ActorCriticBase):
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: Union[int, Tuple[int, Tuple[int, int]]],
        action_dim: int
    ):
        super().__init__(cfg, obs_dim)
        self.critic = CriticV(cfg, obs_dim)
        self.actor = StochasticActor(cfg, obs_dim, action_dim)

    def get_value(self, obs: Union[Tensor, Tuple[Tensor, Tensor]], hidden: Optional[Tensor] = None) -> Tensor:
        return self.critic(obs, hidden=hidden)

    def get_action(self, obs, sample=None, test=False, hidden=None):
        # type: (Union[Tensor, Tuple[Tensor, Tensor]], Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        return self.actor(obs, sample, test, hidden=hidden)

    def get_action_and_value(self, obs, sample=None, test=False):
        # type: (Union[Tensor, Tuple[Tensor, Tensor]], Optional[Tensor], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        return *self.get_action(obs, sample=sample, test=test), self.get_value(obs)


class StochasticAsymmetricActorCriticV(ActorCriticBase):
    def __init__(
        self,
        actor_cfg: DictConfig,
        critic_cfg: DictConfig,
        obs_dim: Union[int, Tuple[int, Tuple[int, int]]],
        state_dim: int,
        action_dim: int
    ):
        super().__init__(actor_cfg, obs_dim)
        self.critic = CriticV(critic_cfg, state_dim)
        self.actor = StochasticActor(actor_cfg, obs_dim, action_dim)

    def get_value(self, state: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        return self.critic(state, hidden=hidden)

    def get_action(self, obs, sample=None, test=False, hidden=None):
        # type: (Union[Tensor, Tuple[Tensor, Tensor]], Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        return self.actor(obs, sample=sample, test=test, hidden=hidden)

    def get_action_and_value(self, obs, state, sample=None, test=False):
        # type: (Union[Tensor, Tuple[Tensor, Tensor]], Tensor, Optional[Tensor], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        return *self.get_action(obs, sample=sample, test=test), self.get_value(state)


class StochasticActorCriticQ(AgentBase):
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: Union[int, Tuple[int, Tuple[int, int]]],
        action_dim: int
    ):
        super().__init__(cfg, obs_dim)
        self.critic = CriticQ(cfg, obs_dim, action_dim)
        self.actor = StochasticActor(cfg, obs_dim, action_dim)

    def get_value(self, obs: Union[Tensor, Tuple[Tensor, Tensor]], action: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        return self.critic(obs, action, hidden=hidden)

    def get_action(self, obs, sample=None, test=False, hidden=None):
        # type: (Union[Tensor, Tuple[Tensor, Tensor]], Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        return self.actor(obs, sample, test, hidden=hidden)

    def get_action_and_value(self, obs, sample=None, test=False):
        # type: (Union[Tensor, Tuple[Tensor, Tensor]], Optional[Tensor], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        action, sample, logprob, entropy = self.get_action(obs, sample=sample, test=test)
        value = self.get_value(obs, action)
        return action, sample, logprob, entropy, value


class RPLActorCritic(StochasticActorCriticV):
    def __init__(
        self,
        cfg: DictConfig,
        anchor_ckpt: str,
        obs_dim: Tuple[int, Tuple[int, int]],
        anchor_obs_dim: int,
        action_dim: int,
        rpl_action: bool = True
    ):
        rpl_obs_dim = (obs_dim[0] + action_dim, obs_dim[1])
        super().__init__(
            cfg=cfg,
            obs_dim=rpl_obs_dim,
            action_dim=action_dim)
        
        torch.nn.init.zeros_(self.actor.actor_mean.net[-1].weight)
        torch.nn.init.zeros_(self.actor.actor_mean.net[-1].bias)
        
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(anchor_ckpt)), ".hydra", "config.yaml")
        ckpt_cfg = OmegaConf.load(cfg_path)
        self.anchor_agent = StochasticActorCriticV(ckpt_cfg.algo.network, anchor_obs_dim, action_dim)
        self.anchor_agent.load(anchor_ckpt)
        self.anchor_agent.eval()
        self.anchor_obs_dim = anchor_obs_dim
        self.rpl_action = rpl_action
    
    def rpl_obs(self, obs: Tuple[Tensor, Tensor], hidden: Optional[Tensor] = None) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        with torch.no_grad():
            anchor_action, _, _, _ = self.anchor_agent.get_action(
                obs["state"], test=True, hidden=hidden)
        rpl_obs = (torch.cat([obs[0], anchor_action], dim=-1), obs[1])
        return rpl_obs, anchor_action

    def get_value(self, obs: Tuple[Tensor, Tensor], hidden: Optional[Tensor] = None) -> Tensor:
        return super().get_value(self.rpl_obs(obs)[0], hidden=hidden)

    def get_action_and_value(self, obs, sample=None, test=False):
        # type: (Tuple[Tensor, Tensor], Optional[Tensor], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        rpl_obs, anchor_action = self.rpl_obs(obs)
        action, sample, logprob, entropy = super().get_action(rpl_obs, sample, test)
        if self.rpl_action:
            raw_rpl_action = action + anchor_action
            rpl_action = torch.where(raw_rpl_action >  1, action + (1-action).detach(), raw_rpl_action)
            rpl_action = torch.where(raw_rpl_action < -1, action - (1+action).detach(), rpl_action)
            # rpl_action = raw_rpl_action.clamp(min=-1, max=1) # numerically equalvalent, but gradient are stopped
        else:
            rpl_action = action
        return rpl_action, sample, logprob, entropy, super().get_value(rpl_obs)

    def save(self, path: str):
        super().save(path)
        self.anchor_agent.save(os.path.join(path, "anchor_agent"))
    
    def load(self, path: str):
        super().load(path)
        self.anchor_agent.load(os.path.join(path, "anchor_agent"))

    def reset(self, indices: Tensor):
        super().reset(indices)
        self.anchor_agent.reset(indices)
    
    def detach(self):
        super().detach()
        self.anchor_agent.detach()

def tensordict2tuple(state: Union[Tensor, TensorDict]) -> Union[Tuple[Tensor, Tensor], Tensor]:
    """
    Split TensorDict into Tuple of Tensors, in order to export the trained policy,
    since policy nets with TensorDict input are not supported by TorchScript.
    """
    if isinstance(state, TensorDict):
        return (state["state"], state["perception"])
    else:
        return state
