from typing import Tuple, Dict, Union, Optional, List
import os
import math

from omegaconf import DictConfig
import torch
from torch import Tensor
import tensordict
from tensordict import TensorDict

from quaddif.env.obstacle_avoidance import ObstacleAvoidanceGrid
from quaddif.algo.buffer import RolloutBufferGRID
from quaddif.network.agents import  StochasticActor
from quaddif.utils.runner import timeit
from .world.backbone import WorldModel

class GRIDWM:
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: Tuple[int, Tuple[int, int]],
        action_dim: int,
        grid_cfg: DictConfig,
        device: torch.device
    ):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.l_rollout: int = cfg.l_rollout
        self.batch_size: int = cfg.batch_size
        self.n_epochs: int = cfg.n_epochs
        self.grid_points: List[int] = grid_cfg.n_points
        self.n_grid_points = math.prod(self.grid_points)
        
        # encoder
        self.wm_encoder = WorldModel(obs_dim, cfg, grid_cfg).to(device)
        # actor
        self.actor = StochasticActor(
            cfg.network,
            self.wm_encoder.deter_dim + self.wm_encoder.latent_dim + obs_dim[0],
            action_dim
        ).to(device)
        # optimizers
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        
        # replay buffer
        self.buffer=RolloutBufferGRID(
            l_rollout=self.l_rollout,
            buffer_size=int(cfg.buffer_size),
            obs_dim=obs_dim,
            action_dim=action_dim,
            grid_dim=self.n_grid_points,
            device=device
        )
        self.entropy_weight: float = cfg.entropy_weight
        self.max_grad_norm: float = cfg.max_grad_norm
        self.entropy_loss = torch.zeros(1, device=device)
        self.actor_loss = torch.zeros(1, device=device)
        self.device = device
        self.deter: Tensor = None
    
    @timeit
    def act(self, obs, test=False):
        # type: (TensorDict, bool) -> Tuple[Tensor, Dict[str, Tensor]]
        with torch.no_grad():
            if self.deter is None:
                self.deter = torch.zeros(obs['state'].shape[0], self.wm_encoder.deter_dim, device=obs['state'].device)
            latent = self.wm_encoder.encode(obs['perception'], obs['state'], self.deter)
            actor_input = torch.cat([latent, self.deter, obs["state"]], dim=-1)
        action, sample, logprob, entropy = self.actor(actor_input, test=test)
        self.deter = self.wm_encoder.recurrent(latent, self.deter, action)
        return action, {"latent": latent, "sample": sample, "logprob": logprob, "entropy": entropy}
    
    def record_loss(self, loss, policy_info, env_info):
        # type: (Tensor, Dict[str, Tensor], Dict[str, Tensor]) -> None
        self.actor_loss += loss.mean()
        self.entropy_loss -= policy_info["entropy"].mean()
    
    @timeit
    def update_wm(self):
        if not self.buffer.size >= self.batch_size:
            return {}, {}
        for _ in range(self.n_epochs):
            observations, actions, dones, rewards = self.buffer.sample(self.batch_size)
            # find ground truth and visible grid
            ground_truth_grid = observations["grid"]
            visible_map = observations["visible_map"]
            total_loss, grad_norms, grid_pred = self.wm_encoder.update(
                obs=observations['perception'],
                state=observations['state'],
                actions=actions,
                rewards=rewards,
                terminals=dones,
                gt_grids=ground_truth_grid,
                visible_map=visible_map
            )
        visible_grid_gt_for_plot = ground_truth_grid & visible_map
        visible_grid_pred_for_plot = grid_pred & visible_map
        
        n_missd_predictions = torch.sum(visible_grid_gt_for_plot != visible_grid_pred_for_plot, dim=-1) # [batch_size, l_rollout]
        env_idx, time_idx = torch.where(n_missd_predictions == n_missd_predictions.max())
        env_idx, time_idx = env_idx[0], time_idx[0]
        selected_grid_gt = visible_grid_gt_for_plot[env_idx, time_idx].reshape(*self.grid_points)
        selected_grid_pred = visible_grid_pred_for_plot[env_idx, time_idx].reshape(*self.grid_points)
        
        return total_loss, grad_norms, selected_grid_gt, selected_grid_pred
    
    @timeit
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
    
    @timeit
    def step(self, cfg, env, obs, on_step_cb=None):
        rollout_obs, rollout_dones, rollout_actions, rollout_rewards = [], [], [], []
        for _ in range(self.l_rollout):
            action, policy_info = self.act(obs)
            next_obs, loss, terminated, env_info = env.step(env.rescale_action(action), need_obs_before_reset=False)
            self.reset(env_info['reset'])
            self.record_loss(loss, policy_info, env_info)
            rollout_obs.append(obs)
            rollout_actions.append(action)
            rollout_dones.append(terminated)
            rollout_rewards.append(10.*(1. - 0.1*loss).detach())
            obs = next_obs
            if on_step_cb is not None:
                on_step_cb(
                    obs=obs,
                    action=action,
                    policy_info=policy_info,
                    env_info=env_info)
        self.buffer.add(
            obs=tensordict.stack(rollout_obs, dim=1),
            action=torch.stack(rollout_actions, dim=1),
            done=torch.stack(rollout_dones, dim=1),
            reward=torch.stack(rollout_rewards, dim=1)
        )
        wm_losses, wm_grad_norm, selected_grid_gt, selected_grid_pred = self.update_wm()
        actor_losses, actor_grad_norms = self.update_actor()
        losses = {**wm_losses, **actor_losses}
        grad_norms = {**wm_grad_norm, **actor_grad_norms}
        self.detach()
        policy_info.update({
            "grid_gt": selected_grid_gt,
            "grid_pred": selected_grid_pred
        })
        return obs, policy_info, env_info, losses, grad_norms
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.wm_encoder.state_dict(), os.path.join(path, "wm_encoder.pth"))
        self.actor.save(path)
    
    def load(self, path):
        self.wm_encoder.load_state_dict(torch.load(os.path.join(path, "wm_encoder.pth")))
        self.actor.load(path)
    
    def detach(self):
        self.deter.detach_()
    
    def reset(self, env_idx: Tensor):
        self.deter[env_idx] = 0.
    
    @staticmethod
    def build(cfg: DictConfig, env: ObstacleAvoidanceGrid, device: torch.device):
        return GRIDWM(
            cfg=cfg,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            grid_cfg=env.cfg.grid,
            device=device)