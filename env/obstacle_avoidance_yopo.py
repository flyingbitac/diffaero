from typing import Tuple, Dict, Union, List
import os

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tensordict import TensorDict

from diffaero.env.obstacle_avoidance import ObstacleAvoidance
from diffaero.dynamics.pointmass import point_mass_quat
from diffaero.utils.runner import timeit
from diffaero.utils.math import mvp
from diffaero.utils.logger import Logger

class ObstacleAvoidanceYOPO(ObstacleAvoidance):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        self.obs_dim = (13, (self.sensor.H, self.sensor.W))
        self.state_dim = 13 + self.n_obstacles * 3
    
    def get_observations(self):
        obs = torch.cat([
            self.world2body(self.target_vel),
            self.world2body(self.v),
            self.world2body(self.a),
            self.q
        ], dim=-1)
        obs = TensorDict({
            "state": obs, "perception": self.sensor_tensor.clone()}, batch_size=self.n_envs)
        return obs
    
    @timeit
    def get_state(
        self,
        _p: Tensor, # [n_envs, T, 3]
        _v: Tensor, # [n_envs, T, 3]
        _a: Tensor  # [n_envs, T, 3]
    ) -> Tensor:
        p, v, a = _p.detach(), _v.detach(), _a.detach()
        dist2obstacles, nearest_points2obstacles = self.obstacle_manager.nearest_distance_to_obstacles(p)
        state = torch.cat([
            (nearest_points2obstacles - p.unsqueeze(2)).flatten(start_dim=-2),
            self.target_pos.unsqueeze(1) - p,
            point_mass_quat(a, orientation=v),
            _v,
            _a,
        ], dim=-1)
        return state
    
    @timeit
    def step(self, action):
        # type: (Tensor) -> Tuple[TensorDict, Tuple[Tensor, Tensor], Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        terminated, truncated, success, avg_vel = super()._step(action)
        reset = terminated | truncated
        reset_indices = reset.nonzero().view(-1)
        goal_reward, differentiable_reward, loss_components, _, nearest_points2obstacles = self.reward_fn(self.p.unsqueeze(1), self.v.unsqueeze(1), self.a.unsqueeze(1))
        self.obstacle_nearest_points.copy_(nearest_points2obstacles.squeeze(1)) # [n_envs, n_obstacles, 3]
        self.update_sensor_data()
        extra = {
            "truncated": truncated,
            "l": self.progress.clone(),
            "reset": reset,
            "reset_indices": reset_indices,
            "success": success,
            "arrive_time": self.arrive_time.clone(),
            "loss_components": loss_components,
            "stats_raw": {
                "success_rate": success[reset],
                "survive_rate": truncated[reset],
                "l_episode": ((self.progress.clone() - 1) * self.dt)[reset],
                "avg_vel": avg_vel[success],
                "arrive_time": self.arrive_time.clone()[success],
            },
            "sensor": self.sensor_tensor.clone()
        }
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        return self.get_observations(), (goal_reward, differentiable_reward), terminated, extra
    
    @timeit
    def reward_fn(
        self,
        _p: Tensor, # [n_envs, T, 3]
        _v: Tensor, # [n_envs, T, 3]
        _a: Tensor  # [n_envs, T, 3]
    ) -> Tuple[Tensor, Tensor, Dict[str, float], Tensor, Tensor]:
        p, v, a = _p.detach(), _v.detach(), _a.detach()
        target_relpos = self.target_pos.unsqueeze(1) - p # [n_envs, T, 3]
        target_dist = target_relpos.norm(dim=-1) # [n_envs, T]
        target_vel = target_relpos / torch.max(target_dist / self.max_vel.unsqueeze(-1), torch.ones_like(target_dist)).unsqueeze(-1)
        inflation = 0.2
        # calculate the nearest points on the obstacles to the drone
        dist2obstacles, nearest_points2obstacles = self.obstacle_manager.nearest_distance_to_obstacles(p) # [n_envs, T, n_obstacles(, 3)]
        obstacle_reldirection = F.normalize(nearest_points2obstacles - p.unsqueeze(-2), dim=-1) # [n_envs, T, n_obstacles, 3]

        dist2surface_inflated = (dist2obstacles - (self.r_drone + inflation)).clamp(min=0) # [n_envs, T, n_obstacles]
        dangerous_factor = dist2surface_inflated.neg().exp() # [n_envs, T, n_obstacles]
        # calculate the obstacle avoidance loss
        approaching_vel = torch.sum(obstacle_reldirection * _v.unsqueeze(-2), dim=-1) # [n_envs, T, n_obstacles]
        approaching = approaching_vel > 0 # [n_envs, T, n_obstacles]
        avoiding_vel = torch.norm(_v.unsqueeze(-2) - approaching_vel.detach().unsqueeze(-1) * obstacle_reldirection, dim=-1) # [n_envs, T, n_obstacles]
        approaching_penalty, most_dangerous = (torch.where(approaching, approaching_vel, 0.) * dangerous_factor).max(dim=-1) # [n_envs, T]
        avoiding_reward = torch.where(approaching, avoiding_vel, 0.) * dangerous_factor # [n_envs, T, n_obstacles]
        avoiding_reward = torch.gather(input=avoiding_reward, dim=-1, index=most_dangerous.unsqueeze(-1)).squeeze(-1) # [n_envs, T]
        oa_loss = approaching_penalty - 0.5 * avoiding_reward # [n_envs, T]
        
        collision = (dist2obstacles - self.r_drone).lt(0.).any(dim=-1) # [n_envs, T]
        collision_loss = collision.float()
        
        arrive_loss = 1 - target_dist.lt(0.5).float()
        pos_loss = 1 - target_dist.neg().exp()
        
        vel_diff = torch.norm(_v - target_vel, dim=-1) # [n_envs, T]
        vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none") # [n_envs, T]
        z_diff = _p[..., 2] - self.target_pos[:, None, 2] # [n_envs, T]
        z_loss = 1 - z_diff.abs().neg().exp()
        goal_reward = (
            self.reward_weights.constant - 
            self.reward_weights.goal.vel * vel_loss -
            self.reward_weights.goal.z * z_loss -
            self.reward_weights.goal.oa * oa_loss -
            self.reward_weights.goal.pos * pos_loss -
            self.reward_weights.goal.collision * collision_loss
        ).detach()
        differentiable_reward = (
            self.reward_weights.constant - 
            self.reward_weights.differentiable.vel * vel_loss -
            self.reward_weights.differentiable.z * z_loss -
            self.reward_weights.differentiable.oa * oa_loss -
            self.reward_weights.differentiable.pos * pos_loss -
            self.reward_weights.differentiable.collision * collision_loss
        )
        loss_components = {
            "vel_loss": vel_loss.mean().item(),
            "z_loss": z_loss.mean().item(),
            "pos_loss": pos_loss.mean().item(),
            "arrive_loss": arrive_loss.mean().item(),
            "collision_loss": collision_loss.mean().item(),
            "oa_loss": oa_loss.mean().item(),
            "goal_reward": goal_reward.mean().item(),
            "total_loss": -differentiable_reward.mean().item(),
            "differentiable_reward": differentiable_reward.mean().item(),
            "goal_reward": goal_reward.mean().item(),
        }
        return goal_reward, differentiable_reward, loss_components, collision, nearest_points2obstacles

    def export_obs_fn(self, path):
        class ObsFn(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(
                self,
                target_vel_w: Tensor,
                v_w: Tensor,
                quat_xyzw: Tensor,
                Rz: Tensor,
                R: Tensor
            ) -> Tensor:
                rotmat_w2b = R.permute(0, 2, 1)
                target_vel_b = mvp(rotmat_w2b, target_vel_w)
                v_b = mvp(rotmat_w2b, v_w)
                a_b = torch.zeros_like(v_b)
                return torch.cat([target_vel_b, v_b, a_b, quat_xyzw], dim=-1)

        example_input = {
            "target_vel_w": torch.randn(1, 3),
            "v_w": torch.randn(1, 3),
            "quat_xyzw": torch.randn(1, 4),
            "Rz": torch.randn(1, 3, 3),
            "R": torch.randn(1, 3, 3),
        }

        torch.onnx.export(
            model=ObsFn(),
            args=(
                example_input["target_vel_w"],
                example_input["v_w"],
                example_input["quat_xyzw"],
                example_input["Rz"],
                example_input["R"],
            ),
            input_names=("target_vel_w", "v_w", "quat_xyzw", "Rz", "R"),
            f=os.path.join(path, "obs_fn.onnx"),
            output_names=("obs",)
        )