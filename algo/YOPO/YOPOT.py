from typing import Tuple, Optional, Dict, Callable
import os
import sys
sys.path.append('..')

import torch
from torch import Tensor
import torch.nn.functional as F
from pytorch3d import transforms as T
from omegaconf import DictConfig
import taichi as ti

from .functions import get_traj_points
from .YOPO import YOPO
from diffaero.env.obstacle_avoidance_yopo import ObstacleAvoidanceYOPO
from diffaero.network.agents import CriticV
from diffaero.utils.nn import clip_grad_norm
from diffaero.utils.math import mvp, rk4
from diffaero.utils.runner import timeit
from diffaero.utils.logger import Logger

class YOPOT(YOPO):
    def __init__(
        self,
        cfg: DictConfig,
        state_dim: int,
        n_envs: int,
        device: torch.device
    ):
        super().__init__(cfg, device)
        self.lmbda: float = cfg.lmbda
        self.critic = CriticV(cfg.critic_network, state_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_network.lr)
        self.critic_grad_norm: float = cfg.critic_network.grad_norm
        
        self.actor_loss = torch.tensor(0., device=self.device)
        self.rollout_gamma = torch.ones(n_envs, device=self.device)
        self.cumulated_loss = torch.zeros(n_envs, device=self.device)
        self.entropy_loss = torch.tensor(0., device=self.device)
    
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
        traj_rewards: Tensor,
        survive: Tensor
    ):
        self.critic.train()
        values = self.critic(states.detach()) # [N, HW, T-1]
        target_values = self.bootstrap(            # [N, HW, T-2]
            next_values=values[..., 1:],           # [N, HW, T-2]
            rewards=(traj_rewards * survive.float())[..., :-1], # [N, HW, T-2]
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
        traj_reward: Tensor,
        survive: Tensor,
        score: Tensor,
    ):
        N, HW, T_1, T_2 = states.size(0), self.n_pitch * self.n_yaw, self.n_points - 1, self.n_points - 2
        self.critic.eval()
        
        traj_reward = traj_reward.reshape(N, HW, T_1)[..., :-1] # [N, HW, T-2]
        
        discount = self.gamma ** torch.arange(T_2, device=self.device).reshape(1, 1, T_2)
        traj_reward_discounted = traj_reward * discount * survive[..., :-1]
        terminal_value = self.critic(states[..., -1, :]) * survive[..., -1].float() * (self.gamma ** T_2)
        # Logger.debug(survive[0, 0], terminal_value[0, 0].item())
        assert terminal_value.requires_grad and states.requires_grad
        traj_value = torch.sum(traj_reward_discounted, dim=-1) + terminal_value
        traj_value = traj_value / T_1
        score_loss = F.mse_loss(score, traj_value.detach())
        total_loss = -traj_value.mean() + 0.01 * score_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = clip_grad_norm(self.net, self.grad_norm)
        self.optimizer.step()
        
        losses = {
            "traj_reward": traj_reward.mean().item(),
            "score_loss": score_loss.item(),
            "total_loss": total_loss.item()}
        grad_norm = {"actor_grad_norm": grad_norm}
        
        return losses, grad_norm

    @timeit
    def step(self, cfg: DictConfig, env: ObstacleAvoidanceYOPO, logger: Logger, obs: Tuple[Tensor, ...], on_step_cb=None):
        N, HW, T_1, T_2 = env.n_envs, self.n_pitch * self.n_yaw, self.n_points - 1, self.n_points - 2
        p_w, rotmat_b2w, _, _, _, _ = obs
        
        for _ in range(cfg.algo.n_epochs):
            # traverse the trajectory and cumulate the loss
            score, coef_xyz = self.inference(obs) # [N, HW, 6, 3]
            
            p_traj_b, v_traj_b, a_traj_b = get_traj_points(self.coef_mats[1:], coef_xyz) # [N, HW, T-1, 3]
            p_traj_w = mvp(rotmat_b2w.unsqueeze(1), p_traj_b.reshape(N, HW*T_1, 3)) + p_w.unsqueeze(1)
            v_traj_w = mvp(rotmat_b2w.unsqueeze(1), v_traj_b.reshape(N, HW*T_1, 3))
            a_traj_w = mvp(rotmat_b2w.unsqueeze(1), a_traj_b.reshape(N, HW*T_1, 3)) + self.G.unsqueeze(1)
            
            states = env.get_state(p_traj_w, v_traj_w, a_traj_w).reshape(N, HW, T_1, -1) # [N, HW, T-1, state_dim]
            _, traj_reward, _, dead, _ = env.loss_and_reward(p_traj_w, v_traj_w, a_traj_w) # [N, HW*(T-1)]
            traj_reward = traj_reward.reshape(N, HW, T_1) # [N, HW, T-1]
            survive = dead.reshape(N, HW, T_1).int().cumsum(dim=2).eq(0) # [N, HW, T-1]
            
            critic_losses, critic_grad_norms = self.update_critic(states, traj_reward, survive)
            actor_losses, actor_grad_norms = self.update_backbone(states, traj_reward, survive, score)
        
        losses = {**actor_losses, **critic_losses}
        grad_norms = {**actor_grad_norms, **critic_grad_norms}
        
        with torch.no_grad():
            action, policy_info = self.act(obs, env=env)
            self.render_trajectories(env, policy_info, p_w, rotmat_b2w)
            next_obs, (loss, _), terminated, env_info = env.step(action)
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
            state_dim=env.state_dim,
            n_envs=env.n_envs,
            device=device
        )