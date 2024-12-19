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

class ObstacleAvoidance(BaseEnv):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super(ObstacleAvoidance, self).__init__(cfg, device)
        self.obstacle_manager = ObstacleManager(cfg.obstacles, self.n_envs, self.L, device)
        self.n_obstacles = self.obstacle_manager.n_obstacles
        self.height_scale = cfg.height_scale
        self.z_ground_plane = -self.height_scale*self.L if cfg.ground_plane else None
        
        self.sensor_type = cfg.sensor.name
        assert self.sensor_type in ["camera", "lidar", "relpos"]
        
        if self.sensor_type == "camera":
            self.camera_type = cfg.sensor.type
            if self.sensor_type == "camera" and self.camera_type == "raydist":
                self.camera = Camera(cfg.sensor, device=device)
            H, W = cfg.sensor.height, cfg.sensor.width
        elif self.sensor_type == "lidar":
            self.lidar = LiDAR(cfg.sensor, device=device)
            H, W = self.lidar.H, self.lidar.W
        elif self.sensor_type == "relpos":
            # relative position of obstacles as additional observation
            H, W = self.n_obstacles, 3
        
        self.state_dim = (13, (H, W)) # flattened depth image as additional observation
        self.sensor_tensor = torch.zeros((cfg.n_envs, H, W), device=device)
        
        use_isaacgym_camera = self.sensor_type == "camera" and self.camera_type == "isaacgym"
        need_renderer = (not cfg.render.headless) or cfg.render.record_video or use_isaacgym_camera
        if need_renderer:
            self.renderer = ObstacleAvoidanceRenderer(
                cfg=cfg.render,
                device=device.index,
                obstacle_manager=self.obstacle_manager,
                z_ground_plane=self.z_ground_plane,
                enable_camera=use_isaacgym_camera)
        else:
            self.renderer = None
        
        self.action_dim = self.model.action_dim
        self.r_drone = cfg.r_drone
    
    def state(self, with_grad=False):
        if self.dynamic_type == "pointmass":
            state = [self.target_vel, self.q, self._v, self._a]
        else:
            state = [self.target_vel, self._q, self._v, self._w]
        state = torch.cat(state, dim=-1)
        state = TensorDict({
            "state": state, "perception": self.sensor_tensor.clone()}, batch_size=self.n_envs)
        state = state if with_grad else state.detach()
        return state
    
    def update_sensor_data(self):
        if self.sensor_type == "camera":
            if self.camera_type == "isaacgym":
                self.sensor_tensor.copy_(self.renderer.render_camera())
            elif self.camera_type == "raydist":
                H, W = self.sensor_tensor.shape[1:]
                self.sensor_tensor.copy_(self.camera(
                    sphere_pos=self.obstacle_manager.p_spheres,
                    sphere_r=self.obstacle_manager.r_spheres,
                    box_min=self.obstacle_manager.box_min,
                    box_max=self.obstacle_manager.box_max,
                    start=self.p.unsqueeze(1).expand(-1, H*W, -1),
                    quat_xyzw=self.q,
                    z_ground_plane=self.z_ground_plane))
        elif self.sensor_type == "lidar":
            H, W = self.sensor_tensor.shape[1:]
            self.sensor_tensor.copy_(self.lidar(
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
    
    def step(self, action):
        # type: (Tensor) -> Tuple[TensorDict, Tensor, Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        action = self.rescale_action(action)
        self.model.step(action)
        terminated, truncated = self.terminated(), self.truncated()
        self.progress += 1
        if self.renderer is not None:
            self.renderer.step(*self.state_for_render())
            reset_all = self.renderer.render()
            truncated = torch.full_like(truncated, reset_all) | truncated
        reset = terminated | truncated
        reset_indices = reset.nonzero().squeeze(-1)
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
            "next_state_before_reset": self.state(with_grad=True),
            "loss_components": loss_components,
            "sensor": self.sensor_tensor.clone(),
        }
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        return self.state(), loss, terminated, extra
    
    def state_for_render(self):
        w = torch.zeros_like(self.v) if self.dynamic_type == "pointmass" else self.w
        drone_state = torch.concat([self.p, self.q, self.v, w], dim=-1)
        assets_state = torch.cat([
            self.obstacle_manager.p_obstacles,
            torch.zeros(self.n_envs, self.n_obstacles, 10, device=self.device)
        ], dim=-1)
        return torch.concat([drone_state.unsqueeze(1), assets_state], dim=1), self.target_pos
    
    def loss_fn(self, action):
        # type: (Tensor) -> Tuple[Tensor, Dict[str, float]]
        # calculating the closest point on each sphere to the quadrotor
        sphere_relpos = self.obstacle_manager.p_spheres - self.p.unsqueeze(1) # [n_envs, n_spheres, 3]
        dist2surface_sphere = (sphere_relpos.norm(dim=-1) - self.obstacle_manager.r_spheres).clamp(min=0) # [n_envs, n_spheres]
        # calculating the closest point on each cube to the quadrotor
        nearest_point = self.p.unsqueeze(1).clamp(min=self.obstacle_manager.box_min, max=self.obstacle_manager.box_max) # [n_envs, n_cubes, 3]
        cube_relpos = nearest_point - self.p.unsqueeze(1) # [n_envs, n_cubes, 3]
        dist2surface_cube = cube_relpos.norm(dim=-1).clamp(min=0) # [n_envs, n_cubes]
        # concatenate the relative direction and distance to the surface of both type of obstacles
        obstacle_reldirection = F.normalize(torch.cat([sphere_relpos, cube_relpos], dim=1), dim=-1)
        dist2surface = torch.cat([dist2surface_sphere, dist2surface_cube], dim=1) # [n_envs, n_obstacles, 3]
        # calculate the obstacle avoidance loss
        approaching_vel = torch.sum(obstacle_reldirection * self._v.unsqueeze(1), dim=-1)
        oa_loss = (approaching_vel.clamp(min=0) / dist2surface.exp()).max(dim=-1).values
        
        collision_loss = self.collision().float() * 10
        
        if self.dynamic_type == "pointmass":
            pos_loss = 1 - (-(self._p-self.target_pos).norm(dim=-1)).exp()
            
            vel_diff = (self.model._vel_ema - self.target_vel).norm(dim=-1)
            vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
            
            jerk_loss = F.mse_loss(self.a, action, reduction="none").sum(dim=-1)
            
            total_loss = vel_loss + 3 * oa_loss + 0.003 * jerk_loss + 5 * pos_loss + collision_loss
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

    def reset_idx(self, env_idx):
        n_resets = len(env_idx)
        state_mask = torch.zeros_like(self.model._state)
        state_mask[env_idx] = 1
        
        xy_min, xy_max = -self.L+0.5, self.L-0.5
        z_min, z_max = -self.height_scale*self.L+0.5, self.height_scale*self.L-0.5
        p_new = torch.cat([
            torch.rand((self.n_envs, 2), device=self.device) * (xy_max - xy_min) + xy_min,
            torch.rand((self.n_envs, 1), device=self.device) * (z_max - z_min) + z_min
        ], dim=-1)
        new_state = torch.cat([p_new, torch.zeros(self.n_envs, self.model.state_dim-3, device=self.device)], dim=-1)
        if self.dynamic_type == "quadrotor":
            new_state[:, 6] = 1 # real part of the quaternion
        self.model._state = torch.where(state_mask.bool(), new_state, self.model._state)
        
        min_init_dist = 1.3 * self.L
        # randomly select a target position that meets the minimum distance constraint
        N = 10
        x = y = torch.linspace(xy_min, xy_max, N, device=self.device)
        z = torch.linspace(z_min, z_max, N, device=self.device)
        random_idx = torch.randperm(N**3, device=self.device)
        # indexing of meshgrid dosen't really matter here, explicitly setting to avoid warning
        xyz = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1).reshape(N**3, 3)[random_idx]
        validility = torch.gt((xyz[None, ...] - self.p[env_idx, None, :]).norm(dim=-1), min_init_dist)
        sub_idx = validility.nonzero()
        env_sub_idx = torch.tensor([(sub_idx[:, 0] == i).sum() for i in range(n_resets)]).roll(1, dims=0)
        env_sub_idx[0] = 0
        env_sub_idx = torch.cumsum(env_sub_idx, dim=0)
        self.target_pos[env_idx] = xyz[sub_idx[env_sub_idx, 1]]
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
    
    def reset(self):
        super().reset()
        if self.renderer is not None:
            self.renderer.step(*self.state_for_render())
        return self.state()
    
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