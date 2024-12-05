from typing import Tuple, Dict, Union, Optional, List
import os

from omegaconf import OmegaConf
import torch
from torch import Tensor
import torch.nn as nn
from tensordict import TensorDict

from quaddif.utils.nn import mlp
from quaddif.network.mlp import StochasticActorCriticMLP

class CNN(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            # nn.LayerNorm(4),
            nn.ELU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            # nn.LayerNorm(8),
            nn.ELU(),
            nn.Conv2d(8, 4, kernel_size=1, stride=1, padding=0),
            # nn.LayerNorm(8),
            nn.ELU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            # nn.LayerNorm(8),
            nn.Flatten()
        )
        self.out_dim = 8 * (H // 4) * (W // 4)
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 3 and x.shape[0] != 1:
            x = x.unsqueeze(1)
        return self.cnn(x)

class DeterministicActorCNN(nn.Module):
    def __init__(
        self,
        state_dim: Tuple[int, Tuple[int, int]],
        hidden_dim: Union[int, List[int]],
        action_dim: int
    ):
        super().__init__()
        D, H, W = state_dim[0], state_dim[1][0], state_dim[1][1]
        self.cnn = CNN(H, W)
        self.actor = mlp(D + self.cnn.out_dim, hidden_dim, action_dim, hidden_act=nn.ELU(), output_act=nn.Tanh())
        
    def forward(self, obs: TensorDict) -> Tensor:
        return self.actor(torch.cat([obs["state"], self.cnn(obs["perception"])], dim=-1))
    
    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({
            "cnn": self.cnn.state_dict(),
            "mlp": self.actor.state_dict()}, os.path.join(path, "actor.pth"))

    def load(self, path: str):
        state_dicts = torch.load(os.path.join(path, "actor.pth"), weights_only=True)
        self.actor.load_state_dict(state_dicts["mlp"])
        self.cnn.load_state_dict(state_dicts["cnn"])

class StochasticActorCNN(nn.Module):
    def __init__(
        self,
        state_dim: Tuple[int, Tuple[int, int]],
        hidden_dim: Union[int, List[int]],
        action_dim: int
    ):
        super().__init__()
        D, H, W = state_dim[0], state_dim[1][0], state_dim[1][1]
        self.cnn = CNN(H, W)
        self.actor_mean = mlp(D + self.cnn.out_dim, hidden_dim, action_dim, hidden_act=nn.ELU(), output_act=nn.Tanh())
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, obs, sample=None, test=False):
        # type: (TensorDict, Optional[Tensor], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        action_mean: Tensor = self.actor_mean(torch.cat([obs["state"], self.cnn(obs["perception"])], dim=-1))
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
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({
            "cnn": self.cnn.state_dict(),
            "actor_mean": self.actor_mean.state_dict(),
            "actor_logstd": self.actor_logstd}, os.path.join(path, "actor.pth"))

    def load(self, path: str):
        state_dicts = torch.load(os.path.join(path, "actor.pth"), weights_only=True)
        self.cnn.load_state_dict(state_dicts["cnn"])
        self.actor_mean.load_state_dict(state_dicts["actor_mean"])
        self.actor_logstd.data.copy_(state_dicts["actor_logstd"].to(self.actor_logstd.device))

class CriticCNN(nn.Module):
    def __init__(
        self,
        state_dim: Tuple[int, Tuple[int, int]],
        hidden_dim: Union[int, List[int]]
    ):
        super().__init__()
        D, H, W = state_dim[0], state_dim[1][0], state_dim[1][1]
        self.cnn = CNN(H, W)
        self.critic = mlp(D + self.cnn.out_dim, hidden_dim, 1, hidden_act=nn.ELU())
    
    def forward(self, obs: TensorDict) -> Tensor:
        return self.critic(torch.cat([obs["state"], self.cnn(obs["perception"])], dim=-1)).squeeze(-1)
    
    def save(self, path: str):
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
    
    def load(self, path: str):
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth"), weights_only=True))
        

class StochasticActorCriticCNN(nn.Module):
    def __init__(
        self,
        state_dim: Tuple[int, Tuple[int, int]],
        hidden_dim: Union[int, List[int]],
        action_dim: int
    ):
        super().__init__()
        self.critic = CriticCNN(state_dim, hidden_dim)
        self.actor = StochasticActorCNN(state_dim, hidden_dim, action_dim)

    def get_value(self, obs: TensorDict) -> Tensor:
        return self.critic(obs)

    def get_action(self, obs, sample=None, test=False):
        # type: (TensorDict, Optional[Tensor], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        return self.actor(obs, sample, test)

    def get_action_and_value(self, obs, sample=None, test=False):
        # type: (TensorDict, Optional[Tensor], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        return *self.get_action(obs, sample, test), self.get_value(obs)
    
    def save(self, path: str):
        self.actor.save(path)
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
    
    def load(self, path: str):
        self.actor.load(path)
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth"), weights_only=True))


class RPLActorCriticCNN(StochasticActorCriticCNN):
    def __init__(
        self,
        anchor_ckpt: str,
        state_dim: Tuple[int, Tuple[int, int]],
        anchor_state_dim: int,
        hidden_dim: int,
        action_dim: int,
        rpl_action: bool = True
    ):
        rpl_state_dim = (state_dim[0] + action_dim, state_dim[1])
        super().__init__(
            state_dim=rpl_state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim)
        
        torch.nn.init.zeros_(self.actor.actor_mean[-2].weight)
        torch.nn.init.zeros_(self.actor.actor_mean[-2].bias)
        
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
        rpl_obs = TensorDict({
            "state": torch.cat([obs["state"], anchor_action], dim=-1),
            "perception": obs["perception"]
        }, batch_size=obs.batch_size)
        return rpl_obs, anchor_action

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