from typing import Sequence, Tuple, Dict, Union, Optional
import os

from omegaconf import DictConfig
import torch
from torch import Tensor
from tensordict import TensorDict

from quaddif.network.agents import DeterministicActor, StochasticActor, tensordict2tuple

class APG:
    def __init__(
        self,
        cfg: DictConfig,
        state_dim: int,
        action_dim: int,
        l_rollout: int,
        device: torch.device
    ):
        self.actor = DeterministicActor(cfg.network, state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.max_grad_norm: float = cfg.max_grad_norm
        self.l_rollout: int = l_rollout
        self.actor_loss = torch.zeros(1, device=device)
        self.device = device
    
    def act(self, state, test=False):
        # type: (Union[Tensor, TensorDict], bool) -> Tuple[Tensor, Dict[str, Tensor]]
        return self.actor(tensordict2tuple(state)), {}
    
    def record_loss(self, loss, policy_info, env_info):
        # type: (Tensor, Dict[str, Tensor], Dict[str, Tensor]) -> None
        self.actor_loss += loss.mean()
    
    def update_actor(self):
        # type: () -> Tuple[Dict[str, float], Dict[str, float]]
        self.actor_loss = self.actor_loss / self.l_rollout
        self.optimizer.zero_grad()
        self.actor_loss.backward()
        grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.actor.parameters()]) ** 0.5
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        actor_loss = self.actor_loss.item()
        self.actor_loss = torch.zeros(1, device=self.device)
        return {"actor_loss": actor_loss}, {"actor_grad_norm": grad_norm}

    def step(self, cfg, env, state, on_step_cb=None):
        for _ in range(cfg.l_rollout):
            action, policy_info = self.act(state)
            state, loss, terminated, env_info = env.step(action)
            self.reset(env_info["reset"])
            self.record_loss(loss, policy_info, env_info)
            if on_step_cb is not None:
                on_step_cb(
                    state=state,
                    action=action,
                    policy_info=policy_info,
                    env_info=env_info)
            
        losses, grad_norms = self.update_actor()
        self.detach()
        return state, policy_info, env_info, losses, grad_norms
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.actor.save(path)
    
    def load(self, path):
        self.actor.load(path)
    
    def reset(self, env_idx: Tensor):
        if self.actor.is_rnn_based:
            self.actor.reset(env_idx)
    
    def detach(self):
        if self.actor.is_rnn_based:
            self.actor.detach()
    
    @staticmethod
    def build(cfg, env, device):
        return APG(
            cfg=cfg.algo,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            l_rollout=cfg.l_rollout,
            device=device)


class APG_stochastic(APG):
    def __init__(
        self,
        cfg: DictConfig,
        state_dim: int,
        action_dim: int,
        l_rollout: int,
        device: torch.device
    ):
        super().__init__(cfg, state_dim, action_dim, l_rollout, device)
        del self.optimizer; del self.actor
        self.actor = StochasticActor(cfg.network, state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.entropy_loss = torch.zeros(1, device=device)
        self.entropy_weight: float = cfg.entropy_weight

    def act(self, state, test=False):
        # type: (Tensor, bool) -> Tuple[Tensor, Dict[str, Tensor]]
        action, sample, logprob, entropy = self.actor(tensordict2tuple(state), test=test)
        return action, {"sample": sample, "logprob": logprob, "entropy": entropy}
    
    def record_loss(self, loss, policy_info, env_info):
        # type: (Tensor, Dict[str, Tensor], Dict[str, Tensor]) -> None
        self.actor_loss += loss.mean()
        self.entropy_loss -= policy_info["entropy"].mean()
    
    def update_actor(self):
        # type: () -> Tuple[Dict[str, float], Dict[str, float]]
        actor_loss = self.actor_loss / self.l_rollout
        entropy_loss = self.entropy_loss / self.l_rollout
        total_loss = actor_loss + self.entropy_weight * entropy_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.actor.parameters()]) ** 0.5
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        self.actor_loss = torch.zeros(1, device=self.device)
        self.entropy_loss = torch.zeros(1, device=self.device)
        return {"actor_loss": actor_loss.mean().item(), "entropy_loss": entropy_loss.mean().item()}, {"actor_grad_norm": grad_norm}

    @staticmethod
    def build(cfg, env, device):
        return APG_stochastic(
            cfg=cfg.algo,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            l_rollout=cfg.l_rollout,
            device=device)