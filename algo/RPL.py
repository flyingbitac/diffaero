from typing import Callable, Sequence, Tuple, Dict, Optional
import os

import torch
from torch import Tensor

from quaddif.utils.nn import StochasticActorCritic

class RPLActorCritic(StochasticActorCritic):
    def __init__(
        self,
        anchor_ckpt: str,
        state_dim: int,
        anchor_state_dim: int,
        hidden_dim: int,
        action_dim: int,
        rpl_action: bool = True
    ):
        super().__init__(
            state_dim=state_dim+action_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim)
        
        torch.nn.init.zeros_(self.actor.actor_mean[-1].weight)
        torch.nn.init.zeros_(self.actor.actor_mean[-1].bias)
        
        self.anchor_agent = StochasticActorCritic(anchor_state_dim, hidden_dim, action_dim)
        self.anchor_agent.load(anchor_ckpt)
        self.anchor_agent.eval()
        self.anchor_state_dim = anchor_state_dim
        self.rpl_action = rpl_action
    
    def rpl_obs(self, obs: Tensor) -> Tensor:
        with torch.no_grad():
            anchor_action, _, _, _ = self.anchor_agent.get_action(
                obs[..., :self.anchor_state_dim], test=True)
        return torch.cat([obs, anchor_action], dim=-1)

    def get_value(self, obs: Tensor) -> Tensor:
        return super().get_value(self.rpl_obs(obs))

    def get_action_and_value(self, obs, sample=None, test=False):
        # type: (Tensor, Optional[Tensor], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        with torch.no_grad():
            anchor_action, _, _, _ = self.anchor_agent.get_action(
                obs[..., :self.anchor_state_dim], test=True)
        rpl_obs = torch.cat([obs, anchor_action], dim=-1)
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