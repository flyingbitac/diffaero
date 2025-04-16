from typing import Sequence, Tuple, Dict, Union, Optional
import os
import math

from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor
from tensordict import TensorDict

from quaddif.utils.nn import mlp
from quaddif.network.networks import CNN, RCNN
from quaddif.network.agents import tensordict2tuple, StochasticActor
from quaddif.utils.runner import timeit


class GRID:
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: Tuple[int, Tuple[int, int]],
        latent_dim: int,
        action_dim: int,
        l_rollout: int,
        grid_cfg: DictConfig,
        device: torch.device
    ):
        # encoder
        self.encoder = RCNN(
            cfg=DictConfig({
                "hidden_dim": [128],
                "rnn_hidden_dim": 256,
                "rnn_n_layers": 1
            }),
            input_dim=obs_dim,
            output_dim=latent_dim
        ).to(device)
        
        # decoders
        self.n_grid_points = math.prod(grid_cfg.n_points)
        self.grid_decoder = mlp(latent_dim, [latent_dim * 2], self.n_grid_points).to(device)
        self.state_decoder = mlp(latent_dim, [latent_dim * 2], obs_dim[0]).to(device)
        
        # actor
        self.actor = StochasticActor(cfg.network, latent_dim, action_dim).to(device)
        
        # optimizers
        self.encdec_optimizer = torch.optim.Adam([
            {"params": self.encoder.parameters()},
            {"params": self.grid_decoder.parameters()},
            {"params": self.state_decoder.parameters()}
        ], lr=cfg.encdec_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        
        self.entropy_loss = torch.zeros(1, device=device)
        self.entropy_weight: float = cfg.entropy_weight
        self.max_grad_norm: float = cfg.max_grad_norm
        self.l_rollout: int = l_rollout
        self.actor_loss = torch.zeros(1, device=device)
        self.device = device
    
    def act(self, obs, test=False):
        # type: (Union[Tensor, TensorDict], bool) -> Tuple[Tensor, Dict[str, Tensor]]
        action, sample, logprob, entropy = self.actor(tensordict2tuple(obs), test=test)
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
        self.actor_optimizer.zero_grad()
        total_loss.backward()
        grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.actor.parameters()]) ** 0.5
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        self.actor_loss = torch.zeros(1, device=self.device)
        self.entropy_loss = torch.zeros(1, device=self.device)
        return {"actor_loss": actor_loss.mean().item(), "entropy_loss": entropy_loss.mean().item()}, {"actor_grad_norm": grad_norm}
    
    @timeit
    def step(self, cfg, env, obs, on_step_cb=None):
        for _ in range(cfg.l_rollout):
            action, policy_info = self.act(obs)
            obs, loss, terminated, env_info = env.step(env.rescale_action(action), need_obs_before_reset=False)
            self.reset(env_info["reset"])
            self.record_loss(loss, policy_info, env_info)
            if on_step_cb is not None:
                on_step_cb(
                    obs=obs,
                    action=action,
                    policy_info=policy_info,
                    env_info=env_info)
            
        losses, grad_norms = self.update_actor()
        self.detach()
        return obs, policy_info, env_info, losses, grad_norms
    
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
        return GRID(
            cfg=cfg,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            l_rollout=cfg.l_rollout,
            device=device)