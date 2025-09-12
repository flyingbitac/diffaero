from collections import defaultdict
from typing import Tuple, Dict, Union, Optional, List
import math
import os
from copy import deepcopy

from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import tensordict
from tensordict import TensorDict
import numpy as np

from diffaero.env.obstacle_avoidance_grid import ObstacleAvoidanceGridYOPO
from diffaero.algo.buffer import RolloutBufferGRID
from diffaero.network.networks import build_network
from diffaero.network.agents import CriticV
from diffaero.utils.math import mvp, rk4, quat_rotate
from diffaero.utils.nn import clip_grad_norm
from diffaero.utils.runner import timeit
from diffaero.utils.logger import Logger
from diffaero.utils.exporter import PolicyExporter

from .grid_wm.world.backbone import WorldModel, WorldModelTesttime
from .YOPO import YOPOBase
from .YOPO.functions import get_traj_point, get_traj_points

class YOPONet(nn.Module):
    def __init__(
        self,
        network_cfg: DictConfig,
        input_dim: int,
        h_out: int,
        w_out: int,
    ):
        super().__init__()
        self.h_out = h_out
        self.w_out = w_out
        self.net = build_network(network_cfg, input_dim, 10, output_act=None)
    
    def forward(self, token: Tensor, obs_p: Tensor) -> Tensor:
        N, HW = obs_p.size(0), self.h_out * self.w_out
        token = token.unsqueeze(1).expand(N, HW, -1) # [N, HW, feature_dim]
        token = torch.cat([token, obs_p], dim=-1) # [N, HW, feature_dim + obs_dim]
        return self.net(token) # [N, HW, out_dim]

class GRIDYOPO(YOPOBase):
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: Tuple[int, Tuple[int, int]],
        state_dim: int,
        action_dim: int,
        n_envs: int,
        grid_cfg: DictConfig,
        device: torch.device
    ):
        super().__init__(cfg.yopo, obs_dim[1][0], obs_dim[1][1], device)
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_envs = n_envs
        
        self.l_rollout: int = cfg.l_rollout
        self.lmbda: float = cfg.yopo.lmbda
        self.batch_size: int = cfg.wm.train.batch_size
        self.yopo_n_epochs: int = cfg.yopo.n_epochs
        self.wm_n_epochs: int = cfg.wm.train.n_epochs
        self.grid_cfg = grid_cfg
        self.grid_points: List[int] = grid_cfg.n_points
        self.n_grid_points = math.prod(self.grid_points)
        
        self.odom_free: bool = cfg.odom_free
        self.state_dim = obs_dim[0]
        
        # world model
        self.wm = WorldModel(obs_dim, cfg.wm, grid_cfg).to(device)
        if cfg.wm.compile:
            self.wm = torch.compile(self.wm, mode="reduce-overhead")
        # replay buffer
        self.buffer=RolloutBufferGRID(
            l_rollout=cfg.wm.l_rollout,
            buffer_size=int(cfg.wm.train.buffer_size),
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            grid_dim=self.n_grid_points,
            device=device
        )
        
        # actor
        actor_input_dim = self.wm.deter_dim + self.wm.latent_dim + (3 if self.odom_free else 9)
        self.yopo_net = YOPONet(cfg.actor, actor_input_dim, self.n_pitch, self.n_yaw).to(device)
        self.yopo_optimizer = torch.optim.Adam(self.yopo_net.parameters(), lr=cfg.actor.lr)
        self.yopo_grad_norm: float = cfg.actor.grad_norm
        
        # critic
        self.critic = CriticV(cfg.critic, state_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic.lr)
        self.critic_grad_norm: float = cfg.critic.grad_norm
    
    def make_state_input(self, obs: TensorDict) -> Tensor:
        target_vel_b, v_curr_b, a_curr_b = obs["state"][..., :3], obs["state"][..., 3:6], obs["state"][..., 6:9]
        rotmat_b2p = self.rotmat_b2p.unsqueeze(0)
        target_vel_p = mvp(rotmat_b2p, target_vel_b.unsqueeze(1)) # [N, HW, 3]
        inputs = [target_vel_p]
        if not self.odom_free:
            v_curr_p = mvp(rotmat_b2p, v_curr_b.unsqueeze(1)) # [N, HW, 3]
            a_curr_p = mvp(rotmat_b2p, a_curr_b.unsqueeze(1)) # [N, HW, 3]
            inputs += [v_curr_p, a_curr_p]
        return torch.cat(inputs, dim=-1) # [N, HW, 3] or [N, HW, 9]
    
    def inference(self, obs: TensorDict, test: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        if not hasattr(self, "deter"):
            self.deter = torch.zeros(obs['state'].shape[0], self.wm.deter_dim, device=self.device)
        with torch.no_grad():
            latent = self.wm.encode(obs['perception'], obs['state'], self.deter, test=test)
            token = torch.cat([latent, self.deter], dim=-1) # [N, deter_dim + latent_dim]
        net_output: Tensor = self.yopo_net(token, self.make_state_input(obs)) # [N, HW, 10]
        coef_xyz, score = self.post_process(net_output, v_curr_b=obs["state"][..., 3:6], a_curr_b=obs["state"][..., 6:9])
        return coef_xyz, score, latent
    
    def act(self, obs: TensorDict, test: bool = False) -> Tuple[Tensor, Dict[str, Tensor]]:
        self.eval()
        quat_xyzw = obs["state"][..., 9:13]
        N, HW = quat_xyzw.size(0), self.n_pitch * self.n_yaw
        coef_xyz, score, latent = self.inference(obs)
        best_idx = score.argmax(dim=-1) # [N, ]
        if not test:
            random_idx = torch.randint(0, HW, (N, ), device=self.device)
            use_random = torch.rand(N, device=self.device) < self.expl_prob
            patch_index = torch.where(use_random, random_idx, best_idx)
        else:
            patch_index = best_idx
        patch_index = patch_index.reshape(N, 1, 1, 1).expand(-1, -1, 6, 3)
        coef_best = torch.gather(coef_xyz, 1, patch_index).squeeze(1) # [N, 6, 3]

        p_next_b, v_next_b, a_next_b = get_traj_point(self.t_next, coef_best) # [N, 3]
        a_next_w = quat_rotate(quat_xyzw, a_next_b) + self.G
        policy_info = {
            "traj_coef": coef_xyz,
            "best_coef": coef_best,
            "best_idx": best_idx
        }
        self.deter = self.wm.recurrent(latent, self.deter, a_next_w)
        return a_next_w, policy_info
    
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
    
    @timeit
    def update_wm(self):
        if self.buffer.size < self.batch_size:
            return {}, {}, {}
        for _ in range(self.wm_n_epochs):
            observations, actions, terminated, rewards = self.buffer.sample4wm(self.batch_size)
            # find ground truth and visible grid
            ground_truth_occupancy = observations["occupancy"]
            visible_map = observations["visibility"]
            total_loss, grad_norms, predictions = self.wm.update(
                img=observations['perception'],
                state=observations['state'],
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                gt_occupancy=ground_truth_occupancy,
                visible_map=visible_map
            )
        return total_loss, grad_norms, predictions
    
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
        self.yopo_optimizer.zero_grad()
        total_loss.backward()
        grad_norm = clip_grad_norm(self.yopo_net, self.yopo_grad_norm)
        self.yopo_optimizer.step()
        
        losses = {
            "traj_reward": traj_rewards.mean().item(),
            "score_loss": score_loss.item(),
            "total_loss": total_loss.item()}
        grad_norm = {"actor_grad_norm": grad_norm}
        
        return losses, grad_norm
    
    def update_yopo(self, obs: TensorDict, env: ObstacleAvoidanceGridYOPO):
        p_w, rotmat_b2w = env.p, env.dynamics.R
        N, HW, T_1, T_2 = obs.shape[0], self.n_pitch * self.n_yaw, self.n_points - 1, self.n_points - 2
        for _ in range(self.yopo_n_epochs):
            coef_xyz, score, latent = self.inference(obs) # [N, HW, 6, 3]
            
            p_traj_b, v_traj_b, a_traj_b = get_traj_points(self.coef_mats[1:], coef_xyz) # [N, HW, T-1, 3]
            p_traj_w = mvp(rotmat_b2w.unsqueeze(1), p_traj_b.reshape(N, HW*T_1, 3)) + p_w.unsqueeze(1)
            v_traj_w = mvp(rotmat_b2w.unsqueeze(1), v_traj_b.reshape(N, HW*T_1, 3))
            a_traj_w = mvp(rotmat_b2w.unsqueeze(1), a_traj_b.reshape(N, HW*T_1, 3)) + self.G.unsqueeze(1)
            
            states = env.get_state(p_traj_w, v_traj_w, a_traj_w).reshape(N, HW, T_1, -1) # [N, HW, T-1, state_dim]
            goal_rewards, differentiable_rewards, _, dead, _ = env.reward_fn(p_traj_w, v_traj_w, a_traj_w) # [N, HW*(T-1)]
            goal_rewards = goal_rewards.reshape(N, HW, T_1) # [N, HW, T-1]
            differentiable_rewards = differentiable_rewards.reshape(N, HW, T_1) # [N, HW, T-1]
            survive = dead.reshape(N, HW, T_1).int().cumsum(dim=2).eq(0) # [N, HW, T-1]
            
            critic_losses, critic_grad_norms = self.update_critic(states, goal_rewards, survive)
            actor_losses, actor_grad_norms = self.update_backbone(states, differentiable_rewards, survive, score)
        
        return {**critic_losses, **actor_losses}, {**critic_grad_norms, **actor_grad_norms}
    
    @timeit
    def step_rollout(self, cfg: DictConfig, env: ObstacleAvoidanceGridYOPO, logger: Logger, obs: TensorDict, on_step_cb=None):
        # env.prev_visible_map.fill_(False) # clear the memory that wm shouldn't have
        p_w, rotmat_b2w = env.p, env.dynamics.R
        rollout_buffer_list = defaultdict(list)
        for t in range(self.l_rollout):
            
            yopo_losses, yopo_grad_norms = self.update_yopo(obs, env)
            
            with torch.no_grad():
                action, policy_info = self.act(obs)
                self.render_trajectories(env, policy_info, p_w, rotmat_b2w)
                next_obs, (goal_rew, diff_rew), terminated, env_info = env.step(action)
                self.reset(env_info['reset'])
            
            rollout_buffer_list["obs"].append(obs)
            rollout_buffer_list["action"].append(action)
            rollout_buffer_list["reward"].append(diff_rew)
            rollout_buffer_list["next_done"].append(env_info["reset"])
            rollout_buffer_list["next_terminated"].append(terminated)
            obs = next_obs
            if on_step_cb is not None:
                on_step_cb(
                    obs=obs,
                    action=action,
                    policy_info=policy_info,
                    env_info=env_info)
        
        self.buffer.add(
            obs=tensordict.stack(rollout_buffer_list["obs"], dim=1),
            action=torch.stack(rollout_buffer_list["action"], dim=1),
            reward=torch.stack(rollout_buffer_list["reward"], dim=1),
            next_done=torch.stack(rollout_buffer_list["next_done"], dim=1),
            next_terminated=torch.stack(rollout_buffer_list["next_terminated"], dim=1)
        )
        wm_losses, wm_grad_norm, predictions = self.update_wm()
        losses = {**wm_losses, **yopo_losses}
        grad_norms = {**wm_grad_norm, **yopo_grad_norms}
        
        self.log_predictions(logger, env, predictions)
        
        return next_obs, policy_info, env_info, losses, grad_norms

    @timeit
    def step_once(self, cfg: DictConfig, env: ObstacleAvoidanceGridYOPO, logger: Logger, obs: TensorDict, on_step_cb=None):
        p_w, rotmat_b2w = env.p, env.dynamics.R

        yopo_losses, yopo_grad_norms = self.update_yopo(obs, env)
        
        with torch.no_grad():
            action, policy_info = self.act(obs)
            self.render_trajectories(env, policy_info, p_w, rotmat_b2w)
            next_obs, (goal_rew, diff_rew), terminated, env_info = env.step(action)
            self.reset(env_info['reset'])
        obs = next_obs
        if on_step_cb is not None:
            on_step_cb(
                obs=obs,
                action=action,
                policy_info=policy_info,
                env_info=env_info)
        self.buffer.add_step(
            obs=obs,
            action=action,
            reward=diff_rew,
            next_done=env_info["reset"],
            next_terminated=terminated
        )
        wm_losses, wm_grad_norm, predictions = self.update_wm()
        losses = {**wm_losses, **yopo_losses}
        grad_norms = {**wm_grad_norm, **yopo_grad_norms}
        
        self.log_predictions(logger, env, predictions)
        
        return next_obs, policy_info, env_info, losses, grad_norms

    def step(self, cfg: DictConfig, env: ObstacleAvoidanceGridYOPO, logger: Logger, obs: TensorDict, on_step_cb=None):
        if self.l_rollout > 1:
            return self.step_rollout(cfg, env, logger, obs, on_step_cb)
        else:
            return self.step_once(cfg, env, logger, obs, on_step_cb)
    
    def log_predictions(self, logger: Logger, env: ObstacleAvoidanceGridYOPO, predictions: Dict[str, Tensor]):
        if logger.n % 100 == 0:
            if "occupancy_pred" in predictions.keys() and "occupancy_gt" in predictions.keys():
                # select the worst prediction and return to the logger for visualization
                visible_occupancy_gt_for_plot = predictions["occupancy_gt"] & predictions["visibility_gt"]
                visible_occupancy_pred_for_plot = predictions["occupancy_pred"] & predictions["visibility_gt"]
                n_missed_predictions = torch.sum(visible_occupancy_gt_for_plot != visible_occupancy_pred_for_plot, dim=-1) # [batch_size, l_rollout]
                env_idx, time_idx = torch.where(n_missed_predictions == n_missed_predictions.max())
                env_idx, time_idx = env_idx[0], time_idx[0]
                
                occupancy_gt = env.visualize_grid(visible_occupancy_gt_for_plot[env_idx, time_idx])
                occupancy_pred = env.visualize_grid(visible_occupancy_pred_for_plot[env_idx, time_idx])
                occupancy = np.concatenate([occupancy_gt, occupancy_pred], axis=1).transpose(2, 0, 1)
                logger.log_image("recon/occupancy", occupancy)
                
                if "visibility_pred" in predictions.keys():
                    visibility_gt = env.visualize_grid(predictions["visibility_gt"][env_idx, time_idx])
                    visibility_pred = env.visualize_grid(predictions["visibility_pred"][env_idx, time_idx])
                    visibility = np.concatenate([visibility_gt, visibility_pred], axis=1).transpose(2, 0, 1)
                    logger.log_image("recon/visibility", visibility)

            if "image_pred" in predictions.keys() and "image_gt" in predictions.keys():
                preprocess = lambda x: x.clamp(0., 1.).expand(3, -1, -1).cpu().numpy()
                img_gt = preprocess(predictions["image_gt"][env_idx, time_idx])
                img_pred = preprocess(predictions["image_pred"][env_idx, time_idx])
                img = np.concatenate([img_gt, img_pred], axis=-1)
                logger.log_image("recon/image", img)
                
                video_gt = predictions["image_gt"][:4].expand(-1, -1, 3, -1, -1)
                video_pred = predictions["image_pred"][:4].expand(-1, -1, 3, -1, -1)
                video = torch.concat([video_gt, video_pred], dim=-1).clamp(0., 1.)
                logger.log_video("recon/video", video.cpu().numpy(), fps=10)
    
    def reset(self, env_idx: Tensor):
        self.deter[env_idx] = 0.
    
    def save(self, path):
        self.wm.save(path)
        self.critic.save(path)
        torch.save(self.yopo_net.state_dict(), os.path.join(path, "backbone.pth"))
    
    def load(self, path):
        self.wm.load(path)
        self.critic.load(path)
        self.yopo_net.load_state_dict(torch.load(os.path.join(path, "backbone.pth")))
    
    @staticmethod
    def build(cfg: DictConfig, env: ObstacleAvoidanceGridYOPO, device: torch.device):
        return GRIDYOPO(
            cfg=cfg,
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_envs=env.n_envs,
            grid_cfg=env.cfg.grid,
            device=device
        )
    
    def export(
        self,
        path: str,
        export_cfg: DictConfig,
        verbose: bool = False,
    ):
        pass