from typing import Tuple, Dict, Union, List

from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch import Tensor
from tensordict import TensorDict

from diffaero.env.obstacle_avoidance import ObstacleAvoidance
from diffaero.utils.runner import timeit
from diffaero.utils.logger import Logger

class ObstacleAvoidanceYOPO(ObstacleAvoidance):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
    
    def get_observations(self):
        return self.p, self.dynamics.R, self.v, self.a, self.target_vel, self.sensor_tensor.unsqueeze(1)
    
    @timeit
    def step(self, action):
        # type: (Tensor) -> Tuple[Tuple[Tensor, ...], Tensor, Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        terminated, truncated, success, avg_vel = super()._step(action)
        reset = terminated | truncated
        reset_indices = reset.nonzero().view(-1)
        loss, loss_components, _ = self.loss_fn(self.p.unsqueeze(1), self.v.unsqueeze(1), self.a.unsqueeze(1))
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
        return self.get_observations(), loss, terminated, extra
    
    @timeit
    def loss_fn(
        self,
        _p: Tensor, # [n_envs, T, 3]
        _v: Tensor, # [n_envs, T, 3]
        _a: Tensor  # [n_envs, T, 3]
    ) -> Tuple[Tensor, Dict[str, float], Tensor]:
        p, v, a = _p.detach(), _v.detach(), _a.detach()
        target_relpos = self.target_pos.unsqueeze(1) - p
        target_dist = target_relpos.norm(dim=-1)
        target_vel = target_relpos / torch.max(target_dist / self.max_vel.unsqueeze(-1), torch.ones_like(target_dist)).unsqueeze(-1)
        inflation = 0.2
        # calculate the nearest points on the obstacles to the drone
        dist2obstacles, nearest_points2obstacles = self.obstacle_manager.nearest_distance_to_obstacles(p)
        self.obstacle_nearest_points.copy_(nearest_points2obstacles.squeeze(1)) # [n_envs, n_obstacles, 3]
        obstacle_reldirection = F.normalize(nearest_points2obstacles.squeeze(1) - p, dim=-1) # [n_envs, n_obstacles, 3]
        
        dist2surface_inflated = (dist2obstacles.squeeze(1) - (self.r_drone + inflation)).clamp(min=0)
        dangerous_factor = dist2surface_inflated.neg().exp()
        # calculate the obstacle avoidance loss
        approaching_vel = torch.sum(obstacle_reldirection * _v, dim=-1) # [n_envs, n_obstacles]
        approaching = approaching_vel > 0
        avoiding_vel = torch.norm(_v - approaching_vel.detach().unsqueeze(-1) * obstacle_reldirection, dim=-1) # [n_envs, n_obstacles]
        approaching_penalty, most_dangerous = (torch.where(approaching, approaching_vel, 0.) * dangerous_factor).max(dim=-1) # [n_envs]
        avoiding_reward = torch.where(approaching, avoiding_vel, 0.) * dangerous_factor # [n_envs, n_obstacles]
        avoiding_reward = avoiding_reward[torch.arange(self.n_envs, device=self.device), most_dangerous] # [n_envs]
        oa_loss = approaching_penalty - 0.5 * avoiding_reward
        
        collision = (dist2obstacles.squeeze(1) - self.r_drone).lt(0.)
        collision_loss = collision.float()
        
        arrive_loss = 1 - target_dist.lt(0.5).float()
        pos_loss = 1 - target_dist.neg().exp()
        
        vel_diff = torch.norm(_v - target_vel, dim=-1)
        vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
        z_diff = _p[..., 2] - self.target_pos[:, None, 2]
        z_loss = 1 - z_diff.abs().neg().exp()

        total_loss = (
            self.loss_weights.pointmass.vel * vel_loss +
            self.loss_weights.pointmass.z * z_loss +
            self.loss_weights.pointmass.oa * oa_loss +
            self.loss_weights.pointmass.pos * pos_loss +
            self.loss_weights.pointmass.collision * collision_loss
        )
        loss_components = {
            "vel_loss": vel_loss.mean().item(),
            "z_loss": z_loss.mean().item(),
            "pos_loss": pos_loss.mean().item(),
            "arrive_loss": arrive_loss.mean().item(),
            "collision_loss": collision_loss.mean().item(),
            "oa_loss": oa_loss.mean().item(),
            "total_loss": total_loss.mean().item(),
            }
        return total_loss, loss_components, collision