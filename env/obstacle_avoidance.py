from typing import Tuple, Dict, Union, List
import os
import math

from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch import Tensor
from tensordict import TensorDict
import open3d as o3d
import numpy as np

from diffaero.env.base_env import BaseEnv
from diffaero.dynamics import PointMassModelBase, QuadrotorModel
from diffaero.utils.sensor import build_sensor, RayCastingSensorBase
from diffaero.utils.render import ObstacleAvoidanceRenderer
from diffaero.utils.assets import ObstacleManager
from diffaero.utils.runner import timeit

class OccupancyGridMap3D:
    def __init__(self, 
                 grid_size=(100, 100, 50), 
                 resolution=0.1,
                 logit_occ=0.85, 
                 logit_free=0.4, 
                 device='cpu', 
                 batch_size=1):
        """
        初始化3D占据栅格地图
        """
        self.grid_size = grid_size
        self.resolution = resolution
        self.batch_size = batch_size
        self.device = device
        
        self.X, self.Y, self.Z = grid_size
        self.center = torch.tensor([self.X//2, self.Y//2, self.Z//2], device=device).view(1, 1, 1, 3)

        # 存储为 log-odds 形式
        self.log_odds = torch.zeros((batch_size, self.X, self.Y, self.Z), device=device)

        # 设置logit增量
        self.logit_occ = torch.logit(torch.tensor(logit_occ, device=device))
        self.logit_free = torch.logit(torch.tensor(logit_free, device=device))

    def world_to_grid(self, point_world, drone_pos):
        """
        将世界坐标转为voxel索引
        """
        relative = point_world - drone_pos[:, None, None, :]  # [N, H, W, 3]
        idx = (relative / self.resolution).long() + self.center  # 偏移至grid中心
        return idx

    def update(self, contact_points: torch.Tensor, drone_pos: torch.Tensor):
        """
        更新occupancy map,包含ray tracing free空间和终点occupied
        contact_points: [N, H, W, 3]
        drone_pos: [N, 3]
        """
        N, H, W, _ = contact_points.shape
        assert N == self.batch_size
        device = contact_points.device
        X, Y, Z = self.grid_size

        # --------------------------
        # Occupied 更新
        # --------------------------
        voxel_idx = self.world_to_grid(contact_points, drone_pos)  # [N, H, W, 3]

        # 合法mask
        valid = (
            (voxel_idx[..., 0] >= 0) & (voxel_idx[..., 0] < X) &
            (voxel_idx[..., 1] >= 0) & (voxel_idx[..., 1] < Y) &
            (voxel_idx[..., 2] >= 0) & (voxel_idx[..., 2] < Z)
        )

        # 向量化打occupied点
        b_idx, h_idx, w_idx = torch.nonzero(valid, as_tuple=True)
        occupied_voxel = voxel_idx[b_idx, h_idx, w_idx]
        x_occ, y_occ, z_occ = occupied_voxel.unbind(-1)

        self.log_odds.index_put_(
            (b_idx, x_occ, y_occ, z_occ),
            torch.full_like(x_occ, self.logit_occ, dtype=self.log_odds.dtype),
            accumulate=True
        )

        # --------------------------
        # Free 空间更新（ray tracing）
        # --------------------------
        p0 = drone_pos[b_idx]                     # [K, 3]
        p1 = contact_points[b_idx, h_idx, w_idx] # [K, 3]

        num_steps = torch.ceil((p1 - p0).norm(dim=-1) / self.resolution).long()  # [K]
        num_steps = torch.clamp(num_steps, min=1, max=100)  # 限制最大步数避免过多采样

        all_points = []
        for i in range(num_steps.max()):
            t = i / num_steps.float()
            t = t.unsqueeze(1)  # [K, 1]
            interp = p0 + (p1 - p0) * t  # [K, 3]
            all_points.append(interp)

        points = torch.stack(all_points, dim=1)  # [K, T, 3]

        # 转换为 voxel 坐标
        rel = points - drone_pos[b_idx].unsqueeze(1)
        voxel_coords = torch.floor(rel / self.resolution + self.center[0]).long()  # [K, T, 3]

        # Clip 越界
        x, y, z = voxel_coords.unbind(-1)
        in_bounds = (
            (x >= 0) & (x < X) &
            (y >= 0) & (y < Y) &
            (z >= 0) & (z < Z)
        )

        b_rep = b_idx.unsqueeze(1).expand(-1, num_steps.max())[in_bounds]  # [M]
        x = x[in_bounds]
        y = y[in_bounds]
        z = z[in_bounds]

        self.log_odds.index_put_(
            (b_rep, x, y, z),
            torch.full_like(x, -self.logit_free, dtype=self.log_odds.dtype),
            accumulate=True
        )

    def get_prob(self):
        return torch.sigmoid(self.log_odds)

    def get_binary(self, threshold=0.5):
        return (self.get_prob() > threshold).to(torch.uint8)

    def decay(self, amount=0.05):
        self.log_odds *= (1.0 - amount)

    def reset(self):
        self.log_odds.zero_()
    
    def reset_idx(self, env_idx):
        self.log_odds[env_idx].zero_()

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
        if isinstance(self.dynamics, PointMassModelBase):
            state_dim = 9
        elif isinstance(self.dynamics, QuadrotorModel):
            state_dim = 10
        if self.last_action_in_obs:
            state_dim += self.action_dim
        self.obs_dim = (state_dim, (H, W))
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
            self._a if isinstance(self.dynamics, PointMassModelBase) else self._w,
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
        self.sensor_tensor.copy_(self.sensor(
            obstacle_manager=self.obstacle_manager,
            pos=self.p,
            quat_xyzw=self.q,
            z_ground_plane=z_ground_plane
        ))
    
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
        
        if isinstance(self.dynamics, PointMassModelBase):
            pos_loss = 1 - (-(self._p-self.target_pos).norm(dim=-1)).exp()
            
            vel_diff = torch.norm(self.dynamics._vel_ema - self.target_vel, dim=-1)
            vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
            z_loss = 1 - (-(self._p[..., 2]-self.target_pos[..., 2]).abs()).exp()

            jerk_loss = F.mse_loss(self.dynamics.a_thrust, self.dynamics.local2world(action), reduction="none").sum(dim=-1)
            
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

class ObstacleAvoidanceGrid(ObstacleAvoidance):
    def __init__(self, cfg: DictConfig, device:torch.device, test:bool=False):
        super().__init__(cfg, device)
        assert isinstance(self.sensor, RayCastingSensorBase), "This environment only supports ray casting-based sensors."
        resolution = (cfg.grid.x_max - cfg.grid.x_min) / cfg.grid.n_points[0]
        self.ocp_map = OccupancyGridMap3D(grid_size=cfg.grid.n_points,
                                          resolution=resolution,
                                          device=self.device,
                                          batch_size=self.n_envs)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=270, height=270, left=50, top=350, visible=self.renderer is not None)
        self.cube_size = min(
            (cfg.grid.x_max - cfg.grid.x_min) / cfg.grid.n_points[0],
            (cfg.grid.y_max - cfg.grid.y_min) / cfg.grid.n_points[1],
            (cfg.grid.z_max - cfg.grid.z_min) / cfg.grid.n_points[2],
        )
        x_range = torch.linspace(cfg.grid.x_min, cfg.grid.x_max - self.cube_size, cfg.grid.n_points[0], device=self.device) + self.cube_size / 2
        y_range = torch.linspace(cfg.grid.y_min, cfg.grid.y_max - self.cube_size, cfg.grid.n_points[1], device=self.device) + self.cube_size / 2
        z_range = torch.linspace(cfg.grid.z_min, cfg.grid.z_max - self.cube_size, cfg.grid.n_points[2], device=self.device) + self.cube_size / 2
        grid_xyz_range = torch.stack(torch.meshgrid(x_range, y_range, z_range, indexing="ij"), dim=-1) # [x_points, y_points, z_points, 3]
        self.local_grid_centers = grid_xyz_range.reshape(1, -1, 3) # [1, n_points, 3]
    
    def visualize_grid(self, grid: torch.Tensor):
        xyz = self.local_grid_centers.squeeze(0)
        points = xyz[grid.flatten()].cpu().numpy()
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(points)
        
        self.vis.clear_geometries()
        self.vis.add_geometry(pcd_o3d)
        self.vis.poll_events()
        self.vis.update_renderer()
        
        return np.asarray(self.vis.capture_screen_float_buffer(do_render=self.renderer is not None))
    
    def get_ocp_map(self):
        N, H, W = self.sensor_tensor.shape
        start = self.p.unsqueeze(1).expand(-1, H*W, -1)
        contact_point = self.sensor.get_contact_point(
            depth=self.sensor_tensor,
            start=start,
            quat_xyzw=self.q
        ).reshape(N, H, W, 3)
        self.ocp_map.update(contact_point, self.p)
        return self.ocp_map.get_binary()
    
    def reset(self):
        super().reset()
        self.ocp_map.reset()
        return self.get_observations()
    
    def reset_idx(self, env_idx):
        super().reset_idx(env_idx)
        self.ocp_map.reset_idx(env_idx)
    
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
        # grid, visible_map = self.get_occupancy_map(), self.get_visibility_map()
        grid = self.get_ocp_map()
        visible_map = grid.clone()
        if self.renderer is not None:
            grid_tobe_visualized = visible_map
            self.visualize_grid(grid_tobe_visualized[self.renderer.gui_states["tracking_env_idx"]])

        obs = TensorDict({
            "state": obs, "perception": self.sensor_tensor.clone(), "grid": grid, "visible_map": visible_map}, batch_size=self.n_envs)
        obs = obs if with_grad else obs.detach()
        return obs
    
class ObstacleAvoidanceYOPO(ObstacleAvoidance):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
    
    def get_observations(self):
        return self.p, self.q, self.v, self.a, self.target_vel, self.sensor_tensor.unsqueeze(1)
    
    @timeit
    def step(self, action):
        # type: (Tensor) -> Tuple[TensorDict, Tensor, Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        self.dynamics.step(action)
        terminated, truncated = self.terminated(), self.truncated()
        self.progress += 1
        if self.renderer is not None:
            self.renderer.render(self.states_for_render())
            truncated = torch.full_like(truncated, self.renderer.gui_states["reset_all"]) | truncated
        arrived = (self.p - self.target_pos).norm(dim=-1) < 0.5
        self.arrive_time.copy_(torch.where(arrived & (self.arrive_time == 0), self.progress.float() * self.dt, self.arrive_time))
        truncated |= arrived & ((self.progress.float() * self.dt) > (self.arrive_time + self.wait_before_truncate))
        avg_vel = (self.init_pos - self.target_pos).norm(dim=-1) / self.arrive_time
        success = arrived & truncated
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
    def loss_fn(self, _p, _v, _a):
        # type: (ObstacleAvoidance, Tensor, Tensor, Tensor) -> Tuple[Tensor, Dict[str, float], Tensor]
        p, v, a = _p.detach(), _v.detach(), _a.detach()
        target_relpos = self.target_pos.unsqueeze(1) - _p
        target_dist = target_relpos.norm(dim=-1)
        target_vel = target_relpos / torch.max(target_dist / self.max_vel.unsqueeze(-1), torch.ones_like(target_dist)).unsqueeze(-1)
        target_vel.detach_()
        virtual_radius = 0.2
        # calculating the closest point on each sphere to the quadrotor
        sphere_relpos = self.obstacle_manager.p_spheres.unsqueeze(2) - p.unsqueeze(1) # [n_envs, n_spheres, 3]
        dist2surface_sphere = (sphere_relpos.norm(dim=-1) - self.obstacle_manager.r_spheres.unsqueeze(2)).clamp(min=0) # [n_envs, n_spheres]
        # calculating the closest point on each cube to the quadrotor
        nearest_point = p.unsqueeze(1).clamp(
            min=self.obstacle_manager.box_min.unsqueeze(2),
            max=self.obstacle_manager.box_max.unsqueeze(2)) # [n_envs, n_cubes, 3]
        cube_relpos = nearest_point - p.unsqueeze(1) # [n_envs, n_cubes, 3]
        dist2surface_cube = cube_relpos.norm(dim=-1).clamp(min=0) # [n_envs, n_cubes]
        # concatenate the relative direction and distance to the surface of both type of obstacles
        obstacle_reldirection = F.normalize(torch.cat([sphere_relpos, cube_relpos], dim=1), dim=-1) # [n_envs, n_obstacles, 3]
        dist2surface = torch.cat([dist2surface_sphere, dist2surface_cube], dim=1) # [n_envs, n_obstacles]
        dist2surface = (dist2surface - self.r_drone - virtual_radius).clamp(min=0)
        # calculate the obstacle avoidance loss
        approaching_vel = torch.sum(obstacle_reldirection * _v.unsqueeze(1), dim=-1) # [n_envs, n_obstacles]
        approaching = approaching_vel > 0
        avoiding_vel = torch.norm(_v.unsqueeze(1) - approaching_vel.detach().unsqueeze(-1) * obstacle_reldirection, dim=-1) # [n_envs, n_obstacles]
        approaching_penalty, most_dangerous = (torch.where(approaching, approaching_vel, 0.) * dist2surface.neg().exp()).max(dim=1) # [n_envs]
        avoiding_reward = torch.where(approaching, avoiding_vel, 0.) * dist2surface.neg().exp() # [n_envs, n_obstacles]
        avoiding_reward = avoiding_reward.gather(dim=1, index=most_dangerous.unsqueeze(1)).squeeze(1) # [n_envs]
        oa_loss = approaching_penalty - 0.2 * avoiding_reward
        # oa_loss = approaching_penalty
        
        pos_loss = 1 - target_relpos.norm(dim=-1).neg().exp()
        
        vel_diff = torch.norm(_v - target_vel, dim=-1)
        vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
        
        collision = torch.any(dist2surface < self.r_drone, dim=1) # [n_envs]
        collision = collision | (p[..., 2] - self.r_drone < self.z_ground_plane)
        # out_of_bound = torch.any(p < -1.5*self.L, dim=-1) | torch.any(p > 1.5*self.L, dim=-1)
        
        total_loss = (
            self.loss_weights.pointmass.vel * vel_loss +
            self.loss_weights.pointmass.oa * oa_loss +
            self.loss_weights.pointmass.pos * pos_loss +
            self.loss_weights.pointmass.collision * collision.float()
        )
        loss_components = {
            "vel_loss": vel_loss.mean().item(),
            "pos_loss": pos_loss.mean().item(),
            "oa_loss": oa_loss.mean().item(),
            "total_loss": total_loss.mean().item()
        }
        return total_loss, loss_components, collision
