from typing import Tuple, Dict, Union

from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch import Tensor
from tensordict import TensorDict

from quaddif.env.base_env import BaseEnv
from quaddif.utils.sensor import Camera, LiDAR
from quaddif.utils.render import ObstacleAvoidanceRenderer
from quaddif.utils.assets import ObstacleManager
from quaddif.utils.runner import timeit

class ObstacleAvoidance(BaseEnv):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super(ObstacleAvoidance, self).__init__(cfg, device)
        self.obstacle_manager = ObstacleManager(cfg.obstacles, self.n_envs, self.L, device)
        self.n_obstacles = self.obstacle_manager.n_obstacles
        self.height_scale: float = cfg.height_scale
        self.z_ground_plane = -self.height_scale*self.L if cfg.ground_plane else None
        
        self.sensor_type = cfg.sensor.name
        assert self.sensor_type in ["camera", "lidar", "relpos"]
        
        if self.sensor_type == "camera":
            self.camera = Camera(cfg.sensor, device=device)
            H, W = self.camera.H, self.camera.W
        elif self.sensor_type == "lidar":
            self.lidar = LiDAR(cfg.sensor, device=device)
            H, W = self.lidar.H, self.lidar.W
        elif self.sensor_type == "relpos":
            # relative position of obstacles as additional observation
            H, W = self.n_obstacles, 3
        
        self.obs_dim = (10, (H, W)) # flattened depth image as additional observation
        self.sensor_tensor = torch.zeros((cfg.n_envs, H, W), device=device)
        
        need_renderer = (not cfg.render.headless) or (hasattr(cfg.render, "record_video") and cfg.render.record_video)
        if need_renderer:
            self.renderer = ObstacleAvoidanceRenderer(
                cfg=cfg.render,
                device=device,
                obstacle_manager=self.obstacle_manager,
                z_ground_plane=self.z_ground_plane,
                headless=cfg.render.headless)
        else:
            self.renderer = None
        
        self.action_dim = self.dynamics.action_dim
        self.r_drone: float = cfg.r_drone
    
    @timeit
    def get_observations(self, with_grad=False):
        if self.dynamic_type == "pointmass":
            obs = torch.cat([self.target_vel, self.q, self._v], dim=-1)
        else:
            obs = torch.cat([self.target_vel, self._q, self._v], dim=-1)
        obs = TensorDict({
            "state": obs, "perception": self.sensor_tensor.clone()}, batch_size=self.n_envs)
        obs = obs if with_grad else obs.detach()
        return obs
    
    @timeit
    def update_sensor_data(self):
        if self.sensor_type == "camera" or self.sensor_type == "lidar":
            H, W = self.sensor_tensor.shape[1:]
            if self.sensor_type == "camera":
                sensor = self.camera
            else:
                sensor = self.lidar
            self.sensor_tensor.copy_(sensor(
                sphere_pos=self.obstacle_manager.p_spheres,
                sphere_r=self.obstacle_manager.r_spheres,
                box_min=self.obstacle_manager.box_min,
                box_max=self.obstacle_manager.box_max,
                start=self.p.unsqueeze(1).expand(-1, H*W, -1),
                quat_xyzw=self.q,
                z_ground_plane=self.z_ground_plane))
        else: # self.sensor_type == "relpos"
            obst_relpos = self.obstacle_manager.p_obstacles - self.p.unsqueeze(1)
            sorted_idx = obst_relpos.norm(dim=-1).argsort(dim=-1).unsqueeze(-1).expand(-1, -1, 3)
            self.sensor_tensor.copy_(obst_relpos.gather(dim=1, index=sorted_idx))
    
    @timeit
    def step(self, action, need_obs_before_reset=True):
        # type: (Tensor, bool) -> Tuple[TensorDict, Tensor, Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        self.dynamics.step(action)
        terminated, truncated = self.terminated(), self.truncated()
        self.progress += 1
        if self.renderer is not None:
            self.renderer.step(**self.state_for_render())
            self.renderer.render()
            truncated = torch.full_like(truncated, self.renderer.gui_states["reset_all"]) | truncated
        reset = terminated | truncated
        reset_indices = reset.nonzero().view(-1)
        arrived = (self.p - self.target_pos).norm(dim=-1) < 0.5
        self.arrive_time.copy_(torch.where(arrived & (self.arrive_time == 0), self.progress.float() * self.dt, self.arrive_time))
        success = arrived & truncated
        loss, loss_components = self.loss_fn(action)
        self.update_sensor_data()
        extra = {
            "truncated": truncated,
            "l": self.progress.clone(),
            "reset": reset,
            "reset_indicies": reset_indices,
            "success": success,
            "arrive_time": self.arrive_time.clone(),
            "loss_components": loss_components,
            "stats_raw": {
                "success_rate": success[reset],
                "survive_rate": truncated[reset],
                "arrive_time": self.arrive_time.clone()[reset],
                "l_episode": ((self.progress.clone() - 1) * self.dt)[reset],
                "arrive_time": self.arrive_time.clone()[success],
            },
            "sensor": self.sensor_tensor.clone()
        }
        if need_obs_before_reset:
            extra["next_obs_before_reset"] = self.get_observations(with_grad=True)
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        return self.get_observations(), loss, terminated, extra
    
    def state_for_render(self):
        return {"drone_pos": self.p.clone(), "drone_quat_xyzw": self.q.clone(), "target_pos": self.target_pos.clone()}
    
    @timeit
    def loss_fn(self, action):
        # type: (Tensor) -> Tuple[Tensor, Dict[str, float]]
        virtual_radius = 0.2
        # calculating the closest point on each sphere to the quadrotor
        sphere_relpos = self.obstacle_manager.p_spheres - self.p.unsqueeze(1) # [n_envs, n_spheres, 3]
        dist2surface_sphere = (sphere_relpos.norm(dim=-1) - self.obstacle_manager.r_spheres).clamp(min=0) # [n_envs, n_spheres]
        # calculating the closest point on each cube to the quadrotor
        nearest_point = self.p.unsqueeze(1).clamp(min=self.obstacle_manager.box_min, max=self.obstacle_manager.box_max) # [n_envs, n_cubes, 3]
        cube_relpos = nearest_point - self.p.unsqueeze(1) # [n_envs, n_cubes, 3]
        dist2surface_cube = cube_relpos.norm(dim=-1).clamp(min=0) # [n_envs, n_cubes]
        # concatenate the relative direction and distance to the surface of both type of obstacles
        obstacle_reldirection = F.normalize(torch.cat([sphere_relpos, cube_relpos], dim=1), dim=-1) # [n_envs, n_obstacles, 3]
        dist2surface = torch.cat([dist2surface_sphere, dist2surface_cube], dim=1) # [n_envs, n_obstacles]
        dist2surface = (dist2surface - self.r_drone - virtual_radius).clamp(min=0)
        # calculate the obstacle avoidance loss
        approaching_vel = torch.sum(obstacle_reldirection * self._v.unsqueeze(1), dim=-1) # [n_envs, n_obstacles]
        approaching = approaching_vel > 0
        avoiding_vel = torch.norm(self._v.unsqueeze(1) - approaching_vel.detach().unsqueeze(-1) * obstacle_reldirection, dim=-1) # [n_envs, n_obstacles]
        approaching_penalty, most_dangerous = (torch.where(approaching, approaching_vel, 0.) * dist2surface.neg().exp()).max(dim=-1) # [n_envs]
        avoiding_reward = torch.where(approaching, avoiding_vel, 0.) * dist2surface.neg().exp() # [n_envs, n_obstacles]
        avoiding_reward = avoiding_reward[torch.arange(self.n_envs, device=self.device), most_dangerous] # [n_envs]
        oa_loss = approaching_penalty - 0.5 * avoiding_reward
        
        collision_loss = self.collision().float() * 100.
        
        if self.dynamic_type == "pointmass":
            pos_loss = 1 - (-(self._p-self.target_pos).norm(dim=-1)).exp()
            
            vel_diff = (self.dynamics._vel_ema - self.target_vel).norm(dim=-1)
            vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
            
            jerk_loss = F.mse_loss(self.a, action, reduction="none").sum(dim=-1)
            
            total_loss = 0.5 * vel_loss + 4 * oa_loss + 0.003 * jerk_loss + 5 * pos_loss + collision_loss
            loss_components = {
                "vel_loss": vel_loss.mean().item(),
                "pos_loss": pos_loss.mean().item(),
                "jerk_loss": jerk_loss.mean().item(),
                "collision_loss": collision_loss.mean().item(),
                "oa_loss": oa_loss.mean().item(),
                "total_loss": total_loss.mean().item()
            }
        else:
            pos_loss = -(-(self._p-self.target_pos).norm(dim=-1)).exp()
            
            vel_diff = (self._v - self.target_vel).norm(dim=-1)
            vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
            
            jerk_loss = self._w.norm(dim=-1)
            
            total_loss = vel_loss + 3 * oa_loss + jerk_loss + 5 * pos_loss
            loss_components = {
                "vel_loss": vel_loss.mean().item(),
                "pos_loss": pos_loss.mean().item(),
                "jerk_loss": jerk_loss.mean().item(),
                "collision_loss": collision_loss.mean().item(),
                "oa_loss": oa_loss.mean().item(),
                "total_loss": total_loss.mean().item()
            }
        return total_loss, loss_components

    @timeit
    def reset_idx(self, env_idx):
        n_resets = len(env_idx)
        state_mask = torch.zeros_like(self.dynamics._state, dtype=torch.bool)
        state_mask[env_idx] = True
        
        xy_min, xy_max = -self.L+0.5, self.L-0.5
        z_min, z_max = -self.height_scale*self.L+0.5, self.height_scale*self.L-0.5
        p_new = torch.cat([
            torch.rand((self.n_envs, 2), device=self.device) * (xy_max - xy_min) + xy_min,
            torch.rand((self.n_envs, 1), device=self.device) * (z_max - z_min) + z_min
        ], dim=-1)
        new_state = torch.cat([p_new, torch.zeros(self.n_envs, self.dynamics.state_dim-3, device=self.device)], dim=-1)
        if self.dynamic_type == "quadrotor":
            new_state[:, 6] = 1 # real part of the quaternion
        elif self.dynamic_type == "pointmass":
            new_state[:, -1] = 9.8
        self.dynamics._state = torch.where(state_mask, new_state, self.dynamics._state)
        self.dynamics.reset_idx(env_idx)
        
        min_init_dist = 1.3 * self.L
        # randomly select a target position that meets the minimum distance constraint
        N = 10
        x = y = torch.linspace(xy_min, xy_max, N, device=self.device)
        z = torch.linspace(z_min, z_max, N, device=self.device)
        
        random_idx = torch.stack([torch.randperm(N**3, device=self.device) for _ in range(n_resets)], dim=0) # [n_resets, N**3]
        random_idx = random_idx.unsqueeze(-1).expand(-1, -1, 3) # [n_resets, N**3, 3]
        # indexing of meshgrid dosen't really matter here, explicitly setting to avoid warning
        xyz = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)
        xyz = xyz.reshape(1, N**3, 3).expand(n_resets, -1, -1).gather(dim=1, index=random_idx) # [n_resets, N**3, 3]
        valid = torch.gt((xyz - self.p[env_idx, None, :]).norm(dim=-1), min_init_dist)
        
        valid_points = valid.nonzero()
        point_index = torch.tensor([(valid_points[:, 0] == i).sum() for i in range(n_resets)]).roll(1, dims=0)
        point_index[0] = 0
        point_index = torch.cumsum(point_index, dim=0)
        chosen_points = valid_points[point_index]
        self.target_pos[env_idx] = xyz[chosen_points[:, 0], chosen_points[:, 1]]
        # check that all regenerated initial and target positions meet the minimal distance contraint
        assert torch.all((self.p[env_idx] - self.target_pos[env_idx]).norm(dim=-1) > min_init_dist).item()
        
        # obstacle position
        mask = self.obstacle_manager.randomize_asset_pose(
            env_idx=env_idx,
            drone_init_pos=self.p[env_idx],
            target_pos=self.target_pos[env_idx],
            safety_range=self.r_drone+0.3
        )
            
        self.progress[env_idx] = 0
        self.arrive_time[env_idx] = 0
        self.max_vel[env_idx] = torch.rand(
            n_resets, device=self.device) * (self.max_target_vel - self.min_target_vel) + self.min_target_vel
    
    def reset(self):
        super().reset()
        if self.renderer is not None:
            self.renderer.step(**self.state_for_render())
        return self.get_observations()
    
    @timeit
    def collision(self) -> Tensor:
        # check if the distance between the drone's mass center and the sphere's center is less than the sum of their radius
        dist2sphere = torch.norm(self.p.unsqueeze(1) - self.obstacle_manager.p_spheres, dim=-1) - self.obstacle_manager.r_spheres # [n_envs, n_spheres]
        collision_sphere = torch.any(dist2sphere < self.r_drone, dim=-1) # [n_envs]
        # check if the distance between the drone's mass center and the closest point on the cube is less than the drone's radius
        nearest_point2cube = self.p.unsqueeze(1).clamp(min=self.obstacle_manager.box_min, max=self.obstacle_manager.box_max) # [n_envs, n_cubes, 3]
        dist2cube = torch.norm(nearest_point2cube - self.p.unsqueeze(1), dim=-1) # [n_envs, n_cubes]
        collision_cube = torch.any(dist2cube < self.r_drone, dim=-1) # [n_envs]
        
        collision = collision_sphere | collision_cube
        
        if self.z_ground_plane is not None:
            collision = collision | (self.p[..., 2] - self.r_drone < self.z_ground_plane)
        
        return collision
    
    def terminated(self) -> Tensor:
        return self.collision()
    
    def truncated(self) -> torch.Tensor:
        out_of_bound = torch.any(self.p < -1.5*self.L, dim=-1) | \
                       torch.any(self.p >  1.5*self.L, dim=-1)
        return (self.progress >= self.max_steps) | out_of_bound


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
            self.renderer.step(**self.state_for_render())
            self.renderer.render()
            truncated = torch.full_like(truncated, self.renderer.gui_states["reset_all"]) | truncated
        reset = terminated | truncated
        reset_indices = reset.nonzero().view(-1)
        arrived = (self.p - self.target_pos).norm(dim=-1) < 0.5
        self.arrive_time.copy_(torch.where(arrived & (self.arrive_time == 0), self.progress.float() * self.dt, self.arrive_time))
        success = arrived & truncated
        loss, loss_components, _ = self.loss_fn(self.p.unsqueeze(1), self.v.unsqueeze(1), self.a.unsqueeze(1))
        self.update_sensor_data()
        extra = {
            "truncated": truncated,
            "l": self.progress.clone(),
            "reset": reset,
            "reset_indicies": reset_indices,
            "success": success,
            "arrive_time": self.arrive_time.clone(),
            "loss_components": loss_components,
            "stats_raw": {
                "success_rate": success[reset],
                "survive_rate": truncated[reset],
                "arrive_time": self.arrive_time.clone()[reset],
                "l_episode": ((self.progress.clone() - 1) * self.dt)[reset],
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
        target_vel = target_relpos / torch.max(target_dist / self.max_vel, torch.ones_like(target_dist)).unsqueeze(-1)
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
        
        total_loss = vel_loss + 4 * oa_loss + 0 * pos_loss + collision.float() * 0
        loss_components = {
            "vel_loss": vel_loss.mean().item(),
            "pos_loss": pos_loss.mean().item(),
            "oa_loss": oa_loss.mean().item(),
            "total_loss": total_loss.mean().item()
        }
        return total_loss, loss_components, collision
