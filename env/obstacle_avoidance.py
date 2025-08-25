from typing import Tuple, Dict, Union, List

from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch import Tensor
from tensordict import TensorDict

from diffaero.env.base_env import BaseEnv
from diffaero.utils.sensor import build_sensor
from diffaero.utils.render import ObstacleAvoidanceRenderer
from diffaero.utils.assets import ObstacleManager
from diffaero.utils.runner import timeit
from diffaero.utils.logger import Logger

class ObstacleAvoidance(BaseEnv):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super(ObstacleAvoidance, self).__init__(cfg, device)
        self.obstacle_manager = ObstacleManager(cfg.obstacles, self.n_envs, device)
        self.n_obstacles = self.obstacle_manager.n_obstacles
        self.height_scale: float = cfg.height_scale
        self.ground_plane: bool = cfg.ground_plane
        self.z_ground_plane = -self.height_scale*self.L if cfg.ground_plane else None
        
        self.sensor_type = cfg.sensor.name
        assert self.sensor_type in ["camera", "lidar", "relpos"]
        
        self.sensor = build_sensor(cfg.sensor, device=device)
        H, W = self.sensor.H, self.sensor.W
        
        self.last_action_in_obs: bool = cfg.last_action_in_obs
        if self.dynamic_type == "pointmass":
            if self.obs_frame == "local":
                state_dim = 9
            elif self.obs_frame == "world":
                state_dim = 10
        elif self.dynamic_type == "quadrotor":
            state_dim = 10
        if self.last_action_in_obs:
            state_dim += self.action_dim
        self.obs_dim: Tuple[int, Tuple[int, int]] = (state_dim, (H, W))
        self.state_dim = 13 + H * W + self.n_obstacles * 3
        self.sensor_tensor = torch.zeros((cfg.n_envs, H, W), device=device)
        
        record_video = hasattr(cfg.render, "record_video") and cfg.render.record_video
        need_renderer = (not cfg.render.headless) or record_video
        if need_renderer:
            self.renderer = ObstacleAvoidanceRenderer(
                cfg=cfg.render,
                device=device,
                obstacle_manager=self.obstacle_manager,
                height_scale=self.height_scale,
                headless=cfg.render.headless)
        else:
            self.renderer = None
        
        self.r_drone: float = cfg.r_drone
        self.obstacle_nearest_points = torch.empty(self.n_envs, self.n_obstacles, 3, device=device)
    
    @timeit
    def get_state(self, with_grad=False):
        dist2obstacles, nearest_points2obstacles = self.obstacle_manager.nearest_distance_to_obstacles(self.p.unsqueeze(1))
        state = torch.cat([
            self.sensor_tensor.flatten(start_dim=-2),
            (nearest_points2obstacles.squeeze(1) - self.p.unsqueeze(1)).flatten(start_dim=-2),
            self.target_pos - self.p,
            self.q,
            self._v,
            self._a if self.dynamic_type == "pointmass" else self._w,
        ], dim=-1)
        return state if with_grad else state.detach()

    @timeit
    def get_observations(self, with_grad=False):
        if self.obs_frame == "local":
            target_vel = self.dynamics.world2local(self.target_vel)
            _v = self.dynamics.world2local(self._v)
        elif self.obs_frame == "world":
            target_vel = self.target_vel
            _v = self._v
        
        if self.dynamic_type == "pointmass":
            obs = torch.cat([
                target_vel,
                self.dynamics.uz if self.obs_frame == "local" else self.q,
                _v,
            ], dim=-1)
        else:
            obs = torch.cat([target_vel, self._q, _v], dim=-1)
        if self.last_action_in_obs:
            obs = torch.cat([obs, self.last_action], dim=-1)
        obs = TensorDict({
            "state": obs, "perception": self.sensor_tensor.clone()}, batch_size=self.n_envs)
        obs = obs if with_grad else obs.detach()
        return obs
    
    @timeit
    def update_sensor_data(self):
        z_ground_plane = -self.height_scale*self.L if self.ground_plane else None
        sensory_data, contact_points = self.sensor(
            obstacle_manager=self.obstacle_manager,
            pos=self.p,
            quat_xyzw=self.q,
            z_ground_plane=z_ground_plane
        )
        self.sensor_tensor.copy_(sensory_data)
    
    @timeit
    def step(self, action, next_obs_before_reset=False, next_state_before_reset=False):
        # type: (Tensor, bool, bool) -> Tuple[TensorDict, Tuple[Tensor, Tensor], Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        terminated, truncated, success, avg_vel = super()._step(action)
        reset = terminated | truncated
        reset_indices = reset.nonzero().view(-1)
        loss, reward, loss_components = self.loss_and_reward(action)
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
        if next_obs_before_reset:
            extra["next_obs_before_reset"] = self.get_observations(with_grad=True)
        if next_state_before_reset:
            extra["next_state_before_reset"] = self.get_state(with_grad=True)
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        return self.get_observations(), (loss, reward), terminated, extra
    
    def states_for_render(self):
        pos = self.p.unsqueeze(1) if self.n_agents == 1 else self.p
        vel = self.v.unsqueeze(1) if self.n_agents == 1 else self.v
        quat_xyzw = self.q.unsqueeze(1) if self.n_agents == 1 else self.q
        target_pos = self.target_pos.unsqueeze(1) if self.n_agents == 1 else self.target_pos
        states_for_render = {
            "pos": pos,
            "vel": vel,
            "quat_xyzw": quat_xyzw,
            "target_pos": target_pos,
            "env_spacing": self.L.value,
            "nearest_points": self.obstacle_nearest_points,
        }
        return {k: v[:self.renderer.n_envs] for k, v in states_for_render.items()}
    
    @timeit
    def loss_and_reward(self, action):
        # type: (Tensor) -> Tuple[Tensor, Tensor, Dict[str, float]]
        inflation = 0.2
        # calculate the nearest points on the obstacles to the drone
        dist2obstacles, nearest_points2obstacles = self.obstacle_manager.nearest_distance_to_obstacles(self.p.unsqueeze(1))
        self.obstacle_nearest_points.copy_(nearest_points2obstacles.squeeze(1)) # [n_envs, n_obstacles, 3]
        obstacle_reldirection = F.normalize(nearest_points2obstacles.squeeze(1) - self.p.unsqueeze(1), dim=-1) # [n_envs, n_obstacles, 3]
        
        dist2surface_inflated = (dist2obstacles.squeeze(1) - (self.r_drone + inflation)).clamp(min=0)
        dangerous_factor = dist2surface_inflated.neg().exp()
        # calculate the obstacle avoidance loss
        approaching_vel = torch.sum(obstacle_reldirection * self._v.unsqueeze(1), dim=-1) # [n_envs, n_obstacles]
        approaching = approaching_vel > 0
        avoiding_vel = torch.norm(self._v.unsqueeze(1) - approaching_vel.detach().unsqueeze(-1) * obstacle_reldirection, dim=-1) # [n_envs, n_obstacles]
        approaching_penalty, most_dangerous = (torch.where(approaching, approaching_vel, 0.) * dangerous_factor).max(dim=-1) # [n_envs]
        avoiding_reward = torch.where(approaching, avoiding_vel, 0.) * dangerous_factor # [n_envs, n_obstacles]
        avoiding_reward = avoiding_reward[torch.arange(self.n_envs, device=self.device), most_dangerous] # [n_envs]
        oa_loss = approaching_penalty - 0.5 * avoiding_reward
        
        collision_loss = self.collision().float()
        arrive_loss = 1 - torch.norm(self.p - self.target_pos, dim=-1).lt(0.5).float()
        
        if self.dynamic_type == "pointmass":
            pos_loss = 1 - (-(self._p-self.target_pos).norm(dim=-1)).exp()
            
            vel_diff = torch.norm(self.dynamics._vel_ema - self.target_vel, dim=-1)
            vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
            z_loss = 1 - (-(self._p[..., 2]-self.target_pos[..., 2]).abs()).exp()

            if self.dynamics.action_frame == "local":
                action = self.dynamics.local2world(action)
            # jerk_loss = F.mse_loss(self.dynamics.a_thrust, action, reduction="none").sum(dim=-1) + \
            #             F.mse_loss(torch.norm(self.dynamics.a_thrust, dim=-1), torch.norm(action, dim=-1), reduction="none") * 5
            jerk_loss = F.mse_loss(self.dynamics.a_thrust, action, reduction="none").sum(dim=-1)
            total_loss = (
                self.loss_weights.pointmass.vel * vel_loss +
                self.loss_weights.pointmass.z * z_loss +
                self.loss_weights.pointmass.oa * oa_loss +
                self.loss_weights.pointmass.jerk * jerk_loss +
                self.loss_weights.pointmass.pos * pos_loss +
                self.loss_weights.pointmass.collision * collision_loss
            )
            total_reward = (
                self.reward_weights.constant - 
                self.reward_weights.pointmass.vel * vel_loss -
                self.reward_weights.pointmass.oa * oa_loss -
                self.reward_weights.pointmass.jerk * jerk_loss -
                self.reward_weights.pointmass.pos * pos_loss -
                self.reward_weights.pointmass.arrive * arrive_loss -
                self.reward_weights.pointmass.collision * collision_loss
            ).detach()
            loss_components = {
                "vel_loss": vel_loss.mean().item(),
                "z_loss": z_loss.mean().item(),
                "pos_loss": pos_loss.mean().item(),
                "arrive_loss": arrive_loss.mean().item(),
                "jerk_loss": jerk_loss.mean().item(),
                "collision_loss": collision_loss.mean().item(),
                "oa_loss": oa_loss.mean().item(),
                "total_loss": total_loss.mean().item(),
                "total_reward": total_reward.mean().item()
            }
        else:
            pos_loss = -(-(self._p-self.target_pos).norm(dim=-1)).exp()
            
            vel_diff = (self._v - self.target_vel).norm(dim=-1)
            vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
            
            jerk_loss = self._w.norm(dim=-1)
            
            total_loss = (
                self.loss_weights.quadrotor.vel * vel_loss +
                self.loss_weights.quadrotor.oa * oa_loss +
                self.loss_weights.quadrotor.jerk * jerk_loss +
                self.loss_weights.quadrotor.pos * pos_loss +
                self.loss_weights.quadrotor.collision * collision_loss
            )
            total_reward = (
                self.reward_weights.constant - 
                self.reward_weights.quadrotor.vel * vel_loss -
                self.reward_weights.quadrotor.oa * oa_loss -
                self.reward_weights.quadrotor.jerk * jerk_loss -
                self.reward_weights.quadrotor.pos * pos_loss -
                self.reward_weights.quadrotor.collision * collision_loss
            ).detach()
            loss_components = {
                "vel_loss": vel_loss.mean().item(),
                "pos_loss": pos_loss.mean().item(),
                "jerk_loss": jerk_loss.mean().item(),
                "collision_loss": collision_loss.mean().item(),
                "oa_loss": oa_loss.mean().item(),
                "total_loss": total_loss.mean().item(),
                "total_reward": total_reward.mean().item()
            }
        return total_loss, total_reward, loss_components

    @timeit
    def reset_idx(self, env_idx):
        self.randomizer.refresh(env_idx)
        self.imu.reset_idx(env_idx)
        n_resets = len(env_idx)
        state_mask = torch.zeros_like(self.dynamics._state, dtype=torch.bool)
        state_mask[env_idx] = True
        
        L = self.L.unsqueeze(-1)
        xy_min, xy_max = -L+0.5, L-0.5
        z_min, z_max = -self.height_scale*L+0.5, self.height_scale*L-0.5
        p_new = torch.cat([
            torch.rand((self.n_envs, 2), device=self.device) * (xy_max - xy_min) + xy_min,
            torch.rand((self.n_envs, 1), device=self.device) * (z_max - z_min) + z_min
        ], dim=-1)
        self.init_pos[env_idx] = p_new[env_idx]
        new_state = torch.cat([p_new, torch.zeros(self.n_envs, self.dynamics.state_dim-3, device=self.device)], dim=-1)
        if self.dynamic_type == "quadrotor":
            new_state[:, 6] = 1 # real part of the quaternion
        elif self.dynamic_type == "pointmass":
            new_state[:, -1] = 9.8
        self.dynamics._state = torch.where(state_mask, new_state, self.dynamics._state)
        self.dynamics.reset_idx(env_idx)
        
        # min_init_dist = 1.2 * self.L
        min_init_dist = ( # [n_envs, 1]
            ((xy_max - xy_min - 1) / 2) ** 2 + 
            ((xy_max - xy_min - 1) / 2) ** 2 + 
            ((z_max - z_min - 1) / 2) ** 2
        ) ** 0.5
        # randomly select a target position that meets the minimum distance constraint
        N = 10
        linspace = torch.linspace(0, 1, N, device=self.device).unsqueeze(0)
        x = y = (xy_max - xy_min) * linspace + xy_min
        z = (z_max - z_min) * linspace + z_min
        
        random_idx = torch.stack([torch.randperm(N**3, device=self.device) for _ in range(n_resets)], dim=0) # [n_resets, N**3]
        random_idx = random_idx.unsqueeze(-1).expand(-1, -1, 3) # [n_resets, N**3, 3]
        # indexing of meshgrid dosen't really matter here, explicitly setting to avoid warning
        xyz = torch.stack([
            x[env_idx].reshape(-1, N, 1, 1).expand(-1,-1, N, N),
            y[env_idx].reshape(-1, 1, N, 1).expand(-1, N,-1, N),
            z[env_idx].reshape(-1, 1, 1, N).expand(-1, N, N,-1)
        ], dim=-1).reshape(-1, N**3, 3).gather(dim=1, index=random_idx)
        valid = torch.gt((xyz - self.p[env_idx, None, :]).norm(dim=-1), min_init_dist[env_idx])
        
        valid_points = valid.nonzero()
        point_index = torch.tensor([(valid_points[:, 0] == i).sum() for i in range(n_resets)]).roll(1, dims=0)
        point_index[0] = 0
        point_index = torch.cumsum(point_index, dim=0)
        chosen_points = valid_points[point_index]
        self.target_pos[env_idx] = xyz[chosen_points[:, 0], chosen_points[:, 1]]
        # check that all regenerated initial and target positions meet the minimal distance contraint
        assert torch.all(((self.p - self.target_pos).norm(dim=-1) > min_init_dist.squeeze(-1))[env_idx]).item()
        
        # randomize obstacles sizes, poses and positions
        self.obstacle_manager.randomize_obstacles(
            env_spacing=self.L.value,
            env_idx=env_idx,
            drone_init_pos=self.p,
            target_pos=self.target_pos
        )
            
        self.progress[env_idx] = 0
        self.arrive_time[env_idx] = 0
        self.last_action[env_idx] = 0.
        self.max_vel[env_idx] = torch.rand(
            n_resets, device=self.device) * (self.max_target_vel - self.min_target_vel) + self.min_target_vel
    
    @timeit
    def collision(self) -> Tensor:
        dist2obstacles, nearest_points2obstacles = self.obstacle_manager.nearest_distance_to_obstacles(self.p.unsqueeze(1))
        min_dist2obstacle = dist2obstacles.squeeze(1).min(dim=-1).values
        collision = min_dist2obstacle < self.r_drone # [n_envs]
        
        if self.z_ground_plane is not None:
            collision = collision | (self.p[..., 2] - self.r_drone < self.z_ground_plane)
        
        return collision
    
    def terminated(self) -> Tensor:
        return self.collision()
    
    def truncated(self) -> torch.Tensor:
        x_range = 1.5 * self.L.value
        y_range = 1.5 * self.L.value
        z_range = self.L.value * self.height_scale
        range = torch.stack([x_range, y_range, z_range], dim=-1)
        out_of_bound = torch.any(self.p < -range, dim=-1) | torch.any(self.p > range, dim=-1)
        return (self.progress >= self.max_steps) | out_of_bound
