from typing import Tuple, Dict, Union, Optional, List
import os

from omegaconf import OmegaConf
import torch
from torch import Tensor
import torch.nn as nn
from tensordict import TensorDict

from quaddif.utils.nn import mlp

# Since the state may be a Tensor or a TensorDict containing some other perceptional information
# and this file only contains actor and critic networks based on MLP, we need to preprocess
# the state_dim and state such that the MLP can process it.
def mlp_state_dim(state_dim: Union[int, Tuple[int, Tuple[int, int]]]) -> int:
    if isinstance(state_dim, int):
        return state_dim
    elif isinstance(state_dim, tuple):
        return state_dim[0] + state_dim[1][0] * state_dim[1][1]

def preprocess_state(state: Union[Tensor, TensorDict]) -> Tensor:
    if isinstance(state, TensorDict):
        return torch.cat([state["state"], state["perception"].flatten(1)], dim=-1)
    else:
        return state

class DeterministicActorMLP(nn.Module):
    def __init__(
        self,
        state_dim: Union[int, Tuple[int, Tuple[int, int]]],
        hidden_dim: Union[int, List[int]],
        action_dim: int
    ):
        super().__init__()
        self.actor = mlp(mlp_state_dim(state_dim), hidden_dim, action_dim, hidden_act=nn.ELU(), output_act=nn.Tanh())
        
    def forward(self, obs: Union[Tensor, TensorDict]) -> Tensor:
        obs = preprocess_state(obs)
        return self.actor(obs)
    
    def save(self, path: str):
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))

    def load(self, path: str):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth")))

class StochasticActorMLP(nn.Module):
    def __init__(
        self,
        state_dim: Union[int, Tuple[int, Tuple[int, int]]],
        hidden_dim: Union[int, List[int]],
        action_dim: int
    ):
        super().__init__()
        self.actor_mean = mlp(mlp_state_dim(state_dim), hidden_dim, action_dim, hidden_act=nn.ELU())
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, obs, sample=None, test=False):
        # type: (Union[Tensor, TensorDict], Optional[Tensor], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        obs = preprocess_state(obs)
        action_mean: Tensor = self.actor_mean(obs)
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
        logprob = probs.log_prob(sample) - torch.log(1. - torch.tanh(sample).pow(2) + 1e-8)
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

class CriticMLP(nn.Module):
    def __init__(
        self,
        state_dim: Union[int, Tuple[int, Tuple[int, int]]],
        hidden_dim: Union[int, List[int]]
    ):
        super().__init__()
        state_dim = mlp_state_dim(state_dim)
        self.critic = mlp(state_dim, hidden_dim, 1, hidden_act=nn.ELU())
    
    def forward(self, obs: Union[Tensor, TensorDict]) -> Tensor:
        obs = preprocess_state(obs)
        return self.critic(obs).squeeze(-1)
    
    def save(self, path: str):
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
    
    def load(self, path: str):
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth"), weights_only=True))
        

class StochasticActorCriticMLP(nn.Module):
    def __init__(
        self,
        state_dim: Union[int, Tuple[int, Tuple[int, int]]],
        hidden_dim: Union[int, List[int]],
        action_dim: int
    ):
        super().__init__()
        self.critic = CriticMLP(state_dim, hidden_dim)
        self.actor = StochasticActorMLP(state_dim, hidden_dim, action_dim)

    def get_value(self, obs: Union[Tensor, TensorDict]) -> Tensor:
        return self.critic(obs)

    def get_action(self, obs, sample=None, test=False):
        # type: (Union[Tensor, TensorDict], Optional[Tensor], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        return self.actor(obs, sample, test)

    def get_action_and_value(self, obs, sample=None, test=False):
        # type: (Union[Tensor, TensorDict], Optional[Tensor], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        return *self.get_action(obs, sample, test), self.get_value(obs)
    
    def save(self, path: str):
        self.actor.save(path)
        self.critic.save(path)
    
    def load(self, path: str):
        self.actor.load(path)
        self.critic.load(path)


class RPLActorCriticMLP(StochasticActorCriticMLP):
    def __init__(
        self,
        anchor_ckpt: str,
        state_dim: Tuple[int, Tuple[int, int]],
        anchor_state_dim: int,
        hidden_dim: int,
        action_dim: int,
        rpl_action: bool = True
    ):
        super().__init__(
            state_dim=mlp_state_dim(state_dim)+action_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim)
        
        torch.nn.init.zeros_(self.actor.actor_mean[-1].weight)
        torch.nn.init.zeros_(self.actor.actor_mean[-1].bias)
        
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(anchor_ckpt)), ".hydra", "config.yaml")
        ckpt_cfg = OmegaConf.load(cfg_path)
        self.anchor_agent = StochasticActorCriticMLP(anchor_state_dim, list(ckpt_cfg.algo.hidden_dim), action_dim)
        self.anchor_agent.load(anchor_ckpt)
        self.anchor_agent.eval()
        self.anchor_state_dim = anchor_state_dim
        self.rpl_action = rpl_action
    
    def rpl_obs(self, obs: TensorDict) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            anchor_action, _, _, _ = self.anchor_agent.get_action(
                obs["state"], test=True)
        return torch.cat([preprocess_state(obs), anchor_action], dim=-1), anchor_action

    def get_value(self, obs: TensorDict) -> Tensor:
        return super().get_value(self.rpl_obs(obs)[0])

    def get_action_and_value(self, obs, sample=None, test=False):
        # type: (TensorDict, Optional[Tensor], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
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