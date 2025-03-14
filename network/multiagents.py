from typing import Tuple, Dict, Union, Optional, List
import os

from omegaconf import DictConfig, OmegaConf
import torch
from torch import Tensor
import torch.nn as nn
from tensordict import TensorDict

from quaddif.network import build_network
from quaddif.network.agents import StochasticActor

class MAAgentBase(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: Union[int, Tuple[int, Tuple[int, int]]],
        global_state_dim: int
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.global_state_dim = global_state_dim
        self.is_rnn_based = cfg.name.lower() == "rnn" or cfg.name.lower() == "rcnn"


class MACriticV(MAAgentBase):
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: Union[int, Tuple[int, Tuple[int, int]]],
        global_state_dim: int
    ):
        super().__init__(cfg, obs_dim, global_state_dim)
        self.critic = build_network(cfg, global_state_dim, 1)
    
    def forward(self, global_state: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        return self.critic(global_state, hidden=hidden).squeeze(-1)
    
    def save(self, path: str):
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
    
    def load(self, path: str):
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth"), weights_only=True))

    def reset(self, indices: Tensor):
        self.critic.reset(indices)
    
    def detach(self):
        self.critic.detach()


class MAStochasticActorCriticV(MAAgentBase):
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: Union[int, Tuple[int, Tuple[int, int]]],
        global_state_dim: int,
        action_dim: int
    ):
        super().__init__(cfg, obs_dim, global_state_dim)
        self.critic = MACriticV(cfg, obs_dim, global_state_dim)
        self.actor = StochasticActor(cfg, obs_dim, action_dim)

    def get_value(self, global_state: Union[Tensor, Tuple[Tensor, Tensor]], hidden: Optional[Tensor] = None) -> Tensor:
        return self.critic(global_state, hidden=hidden)

    def get_action(self, obs, sample=None, test=False, hidden=None):
        # type: (Union[Tensor, Tuple[Tensor, Tensor]], Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        return self.actor(obs, sample, test, hidden=hidden)

    def get_action_and_value(self, obs, global_state, sample=None, test=False):
        # type: (Union[Tensor, Tuple[Tensor, Tensor]], Tensor, Optional[Tensor], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        return *self.get_action(obs, sample=sample, test=test), self.get_value(global_state)
    
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