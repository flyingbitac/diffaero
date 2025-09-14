from typing import Tuple, Optional, Dict, Callable
import os
import sys
sys.path.append('..')

import torch
from torch import Tensor
import torch.nn.functional as F
from tensordict import TensorDict
from pytorch3d import transforms as T
from omegaconf import DictConfig
import taichi as ti

from .functions import get_traj_points
from .YOPO import YOPO
from diffaero.env import ObstacleAvoidanceYOPO
from diffaero.network.agents import CriticV
from diffaero.utils.nn import clip_grad_norm
from diffaero.utils.math import mvp, rk4
from diffaero.utils.runner import timeit
from diffaero.utils.logger import Logger

class YOPOT(YOPO):
    def __init__(
        self,
        cfg: DictConfig,
        img_h: int,
        img_w: int,
        state_dim: int,
        device: torch.device
    ):
        super().__init__(cfg, img_h, img_w, device)
        self.lmbda: float = cfg.lmbda
        self.critic = CriticV(cfg.critic_network, state_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_network.lr)
        self.critic_grad_norm: float = cfg.critic_network.grad_norm
    
    @torch.no_grad()
    def bootstrap(
        self,
        next_values: Tensor,
        rewards: Tensor,
        dones: Tensor,
    ):
        N, HW, T_2 = next_values.shape
        dones = dones.clone().float()
        # value of the next obs should be zero if the next obs is a terminal obs
        next_values = next_values * (1. - dones)
        if self.lmbda == 0.:
            target_values = rewards + self.gamma * next_values
        else:
            target_values = torch.zeros_like(next_values).to(self.device)
            Ai = torch.zeros(N, HW, dtype=torch.float32, device=self.device)
            Bi = torch.zeros(N, HW, dtype=torch.float32, device=self.device)
            lam = torch.ones(N, HW, dtype=torch.float32, device=self.device)
            # dones[..., -1] = 1.
            for i in reversed(range(T_2)):
                lam = lam * self.lmbda * (1. - dones[..., i]) + dones[..., i]
                Ai = (1. - dones[..., i]) * (
                    self.gamma * (self.lmbda * Ai + next_values[..., i]) + \
                    (1. - lam) / (1. - self.lmbda) * rewards[..., i])
                Bi = self.gamma * (next_values[..., i] * dones[..., i] + Bi * (1. - dones[..., i])) + rewards[..., i]
                target_values[..., i] = (1.0 - self.lmbda) * Ai + lam * Bi
        return target_values

    def update_critic(
        self,
        states: Tensor,
        goal_rewards: Tensor,
        survive: Tensor
    ):
        self.critic.train()
        values = self.critic(states.detach()) # [N, HW, T-1]
        target_values = self.bootstrap(            # [N, HW, T-2]
            next_values=values[..., 1:],           # [N, HW, T-2]
            rewards=(goal_rewards * survive.float())[..., :-1], # [N, HW, T-2]
            dones=~survive[..., :-1]             # [N, HW, T-2]
        )
        state_values_alive = values[..., :-1][survive[..., :-1]]
        target_values_alive = target_values[survive[..., :-1]]
        critic_loss = F.mse_loss(state_values_alive, target_values_alive)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        grad_norm = clip_grad_norm(self.critic, self.critic_grad_norm)
        self.critic_optimizer.step()
        return {"critic_loss": critic_loss.item()}, {"critic_grad_norm": grad_norm}
    
    def update_backbone(
        self,
        states: Tensor,
        traj_rewards: Tensor,
        survive: Tensor,
        score: Tensor,
    ):
        N, HW, T_1, T_2 = states.size(0), self.n_pitch * self.n_yaw, self.n_points - 1, self.n_points - 2
        self.critic.eval()
        
        traj_rewards = traj_rewards.reshape(N, HW, T_1)[..., :-1] # [N, HW, T-2]
        
        discount = self.gamma ** torch.arange(T_2, device=self.device).reshape(1, 1, T_2)
        traj_reward_discounted = traj_rewards * discount * survive[..., :-1]
        terminal_value = self.critic(states[..., -1, :]) * survive[..., -1].float() * (self.gamma ** T_2)
        # Logger.debug(survive[0, 0], terminal_value[0, 0].item())
        assert terminal_value.requires_grad and states.requires_grad
        traj_value = torch.sum(traj_reward_discounted, dim=-1) + terminal_value
        traj_value = traj_value / survive.float().sum(dim=-1).clamp(min=1.)
        score_loss = F.mse_loss(score, traj_value.detach())
        total_loss = -traj_value.mean() + 0.01 * score_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = clip_grad_norm(self.net, self.grad_norm)
        self.optimizer.step()
        
        losses = {
            "traj_reward": traj_rewards.mean().item(),
            "score_loss": score_loss.item(),
            "total_loss": total_loss.item()}
        grad_norm = {"actor_grad_norm": grad_norm}
        
        return losses, grad_norm

    @timeit
    def step(self, cfg: DictConfig, env: ObstacleAvoidanceYOPO, logger: Logger, obs: TensorDict, on_step_cb=None):
        N, HW, T_1, T_2 = env.n_envs, self.n_pitch * self.n_yaw, self.n_points - 1, self.n_points - 2
        p_w, rotmat_b2w = env.p, env.dynamics.R
        
        for _ in range(cfg.algo.n_epochs):
            # traverse the trajectory and cumulate the loss
            coef_xyz, score = self.inference(obs) # [N, HW, 6, 3]
            goal_reward, differentiable_reward, survive, p_traj_w, v_traj_w, a_traj_w = self.eval_traj(coef_xyz, p_w, rotmat_b2w, env)
            states = env.get_state(p_traj_w, v_traj_w, a_traj_w).reshape(N, HW, T_1, -1) # [N, HW, T-1, state_dim]
            
            critic_losses, critic_grad_norms = self.update_critic(states, goal_reward, survive)
            actor_losses, actor_grad_norms = self.update_backbone(states, differentiable_reward, survive, score)
        
        losses = {**actor_losses, **critic_losses}
        grad_norms = {**actor_grad_norms, **critic_grad_norms}
        
        with torch.no_grad():
            action, policy_info = self.act(obs)
            self.render_trajectories(env, policy_info, p_w, rotmat_b2w)
            next_obs, (_, _), terminated, env_info = env.step(action)
            if on_step_cb is not None:
                on_step_cb(
                    obs=obs,
                    action=action,
                    policy_info=policy_info,
                    env_info=env_info)
        
        return next_obs, policy_info, env_info, losses, grad_norms

    @staticmethod
    def build(cfg: DictConfig, env: ObstacleAvoidanceYOPO, device: torch.device) -> "YOPOT":
        return YOPOT(
            cfg=cfg,
            img_h=env.sensor.H,
            img_w=env.sensor.W,
            state_dim=env.state_dim,
            device=device
        )