from typing import Tuple, Dict, Union

from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch3d import transforms as T
from tensordict import TensorDict

from quaddif.env.base_env import BaseEnv
from quaddif.utils.render import ObstacleAvoidanceRenderer
from quaddif.utils.math import rand_range
from quaddif.utils.assets import ObstacleManager

@torch.jit.script
def raydist3d_sphere(
    obst_pos: Tensor, # [n_envs, n_spheres, 3]
    obst_r: Tensor, # [n_envs, n_spheres]
    start: Tensor, # [n_envs, n_rays, 3]
    direction: Tensor, # [n_envs, n_rays, 3]
    max_dist: float
) -> Tensor:
    """Compute the ray distance based on the start of the ray,
    the direction of the ray, and the position and radius of 
    the sphere obstacles.

    Args:
        obst_pos (torch.Tensor): The center position of the sphere obstacles.
        obst_r (torch.Tensor): The radius of the sphere obstacles.
        start (torch.Tensor): The start point of the ray.
        direction (torch.Tensor): The direction of the ray.
        max_dist (float): The maximum traveling distance of the ray.

    Returns:
        torch.Tensor: The distance of the ray to the nearest obstacle's surface.
    """    
    rel_pos = obst_pos.unsqueeze(1) - start.unsqueeze(2) # [n_envs, n_rays, n_spheres, 3]
    rel_dist = torch.norm(rel_pos, dim=-1) # [n_envs, n_agents, n_spheres]
    costheta = torch.cosine_similarity(rel_pos, direction.unsqueeze(2), dim=-1) # [n_envs, n_rays, n_spheres]
    sintheta = torch.where(costheta>0, torch.sqrt(1 - costheta**2), 0.9) # [n_envs, n_rays, n_spheres]
    dist_center2ray = rel_dist * sintheta # [n_envs, n_rays, n_spheres]
    obst_r = obst_r.unsqueeze(1) # [n_envs, 1, n_spheres]
    raydist = rel_dist * costheta - torch.sqrt(torch.pow(obst_r, 2) - torch.pow(dist_center2ray, 2)) # [n_envs, n_rays, n_spheres]
    valid = torch.logical_and(dist_center2ray < obst_r, costheta > 0)
    valid = torch.logical_and(valid, raydist < max_dist)
    # valid = (dist_center2ray < obst_r) & (costheta > 0) & (raydist < max_dist) # [n_envs, n_rays, n_spheres]
    raydist_valid = torch.where(valid, raydist, max_dist) # [n_envs, n_rays, n_spheres]
    return raydist_valid.min(dim=-1).values # [n_envs, n_rays]

@torch.jit.script
def raydist3d_cube(
    box_min: Tensor, # [n_envs, n_cubes, 3]
    box_max: Tensor, # [n_envs, n_cubes, 3]
    start: Tensor, # [n_envs, n_rays, 3]
    direction: Tensor, # [n_envs, n_rays, 3]
    max_dist: float
) -> Tensor:
    """Compute the ray distance based on the start of the ray,
    the direction of the ray, and the position and radius of 
    the cubic obstacles.

    Args:
        obst_pos (torch.Tensor): The center position of the sphcubicere obstacles.
        obst_lwh (torch.Tensor): The length, width and height of the cubic obstacles.
        start (torch.Tensor): The start point of the ray.
        direction (torch.Tensor): The direction of the ray.
        max_dist (float): The maximum traveling distance of the ray.

    Returns:
        torch.Tensor: The distance of the ray to the nearest obstacle's surface.
    """
    _tmin = (box_min.unsqueeze(1) - start.unsqueeze(2)) / direction.unsqueeze(2) # [n_envs, n_rays, n_cubes, 3]
    _tmax = (box_max.unsqueeze(1) - start.unsqueeze(2)) / direction.unsqueeze(2) # [n_envs, n_rays, n_cubes, 3]
    tmin = torch.where(direction.unsqueeze(2) < 0, _tmax, _tmin) # [n_envs, n_rays, n_cubes, 3]
    tmax = torch.where(direction.unsqueeze(2) < 0, _tmin, _tmax) # [n_envs, n_rays, n_cubes, 3]
    tentry = torch.max(tmin, dim=-1).values # [n_envs, n_rays, n_cubes]
    texit = torch.min(tmax, dim=-1).values # [n_envs, n_rays, n_cubes]
    valid = torch.logical_and(tentry <= texit, texit >= 0) # [n_envs, n_rays, n_cubes]
    raydist = torch.where(valid, tentry, max_dist) # [n_envs, n_rays, n_cubes]
    return raydist.min(dim=-1).values # [n_envs, n_rays]

class Camera:
    def __init__(self, cfg, device):
        self.H = cfg.height
        self.W = cfg.width
        self.hfov = cfg.horizontal_fov
        self.vfov = self.hfov * self.H / self.W
        self.max_dist = cfg.max_dist
        self.device = device
        self.ray_directions = self._get_ray_directions_plane()
        # self.ray_directions = self._get_ray_directions_sphere()
    
    def get_raydist(
        self,
        sphere_pos: Tensor, # [n_envs, n_spheres, 3]
        sphere_r: Tensor, # [n_envs, n_spheres]
        box_min: Tensor, # [n_envs, n_cubes, 3]
        box_max: Tensor, # [n_envs, n_cubes, 3]
        start: Tensor, # [n_envs, n_rays, 3]
        quat_xyzw: Tensor, # [n_envs, 4]
    ) -> Tensor: # [n_envs, n_rays]
        ray_directions = self.world2body(quat_xyzw) # [n_envs, n_rays, 3]
        raydist_sphere: Tensor = raydist3d_sphere(sphere_pos, sphere_r, start, ray_directions, self.max_dist) # [n_envs, n_rays]
        raydist_cube: Tensor = raydist3d_cube(box_min, box_max, start, ray_directions, self.max_dist) # [n_envs, n_rays]
        raydist = torch.minimum(raydist_sphere, raydist_cube) # [n_envs, n_rays]
        depth = 1. - raydist.reshape(-1, self.H, self.W) / self.max_dist # [n_envs, H, W]
        return depth
    
    def world2body(self, quat_xyzw: Tensor) -> Tensor:
        quat_wxyz = quat_xyzw.roll(1, dims=-1) # [n_envs, 4]
        quat_wxyz = quat_wxyz.unsqueeze(1).expand(-1, self.H*self.W, -1) # [n_envs, n_rays, 4]
        return T.quaternion_apply(quat_wxyz, self.ray_directions.view(1, -1, 3)) # [n_envs, n_rays, 3]
        
    def _get_ray_directions_sphere(self):
        pitch = torch.linspace(0.5*self.vfov, -0.5*self.vfov, self.H, device=self.device) * torch.pi / 180
        yaw = torch.linspace(-0.5*self.hfov, 0.5*self.hfov, self.W, device=self.device) * torch.pi / 180
        pitch, yaw = torch.meshgrid(pitch, yaw)
        roll = torch.zeros_like(pitch)
        ypr = torch.stack([yaw, pitch, roll], dim=-1)
        rotmat = T.euler_angles_to_matrix(ypr, convention="ZYX") # [H, W, 3, 3]
        forward = Tensor([[[1., 0., 0.]]], device=self.device).expand(self.H, self.W, -1).unsqueeze(-1) # [H, W, 3, 1]
        directions = rotmat.transpose(-1, -2) @ forward # [H, W, 3, 1]
        return directions.squeeze(-1) # [H, W, 3]

    def _get_ray_directions_plane(self):
        import math
        forward = torch.tensor([[[1., 0., 0.]]], device=self.device).expand(self.H, self.W, -1) # [H, W, 3]
        
        vangle = 0.5 * self.vfov * torch.pi / 180
        vertical_offset = torch.linspace(math.tan(vangle), -math.tan(vangle), self.H, device=self.device).reshape(-1, 1, 1) # [H, 1, 1]
        zero = torch.zeros_like(vertical_offset)
        vertical_offset = torch.concat([zero, zero, vertical_offset], dim=-1) # [H, 1, 3]
        
        hangle = 0.5 * self.hfov * torch.pi / 180
        horizontal_offset = torch.linspace(math.tan(hangle), -math.tan(hangle), self.W, device=self.device).reshape(1, -1, 1) # [1, W, 1]
        zero = torch.zeros_like(horizontal_offset)
        horizontal_offset = torch.concat([zero, horizontal_offset, zero], dim=-1) # [1, W, 3]
        
        return forward + vertical_offset + horizontal_offset # [H, W, 3]

class ObstacleAvoidance(BaseEnv):
    def __init__(self, env_cfg: DictConfig, model_cfg: DictConfig, device: torch.device):
        super(ObstacleAvoidance, self).__init__(env_cfg, model_cfg, device)
        self.camera_type = env_cfg.camera.type
        self.obstacle_manager = ObstacleManager(env_cfg, device)
        self.n_obstacles = self.obstacle_manager.n_obstacles
        
        if self.camera_type is not None:
            H, W = env_cfg.camera.height, env_cfg.camera.width
            self.state_dim = (13, (H, W)) # flattened depth image as additional observation
            self.camera_tensor = torch.zeros((env_cfg.n_envs, H, W), device=device)
            if self.camera_type == "raydist":
                self.camera = Camera(env_cfg.camera, device=device)
        else:
            # relative position of obstacles as additional observation
            self.state_dim = 13 + self.n_obstacles * 3
        
        if not env_cfg.render.headless or self.camera_type == "isaacgym":
            self.renderer = ObstacleAvoidanceRenderer(env_cfg.render, device.index, self.obstacle_manager)
        else:
            self.renderer = None # headless and camera_type == "raydist", then we don't need a renderer
        
        self.action_dim = self.model.action_dim
        self.r_drone = env_cfg.r_drone
    
    def state(self, with_grad=False):
        if self.dynamic_type == "pointmass":
            state = [self.target_vel, self.q, self._v, self._a]
        else:
            state = [self.target_vel, self._q, self._v, self._w]
        
        if self.camera_type is not None:
            state = torch.cat(state, dim=-1)
            state if with_grad else state.detach()
            state = TensorDict({
                "state": state, "perception": self.camera_tensor.clone()}, batch_size=self.n_envs)
        else:
            obst_relpos = self.obstacle_manager.p_obstacles - self._p.unsqueeze(1)
            sorted_idx = obst_relpos.norm(dim=-1).argsort(dim=-1).unsqueeze(-1).expand(-1, -1, 3)
            state.append(obst_relpos.gather(dim=1, index=sorted_idx).flatten(1))
            state = torch.cat(state, dim=-1)
            state if with_grad else state.detach()
        return state
    
    def render_camera(self):
        if self.renderer is not None and self.camera_type == "isaacgym":
            self.camera_tensor.copy_(self.renderer.render_camera())
        elif self.camera_type == "raydist":
            H, W = self.camera_tensor.shape[1:]
            self.camera_tensor.copy_(self.camera.get_raydist(
                sphere_pos=self.obstacle_manager.p_spheres,
                sphere_r=self.obstacle_manager.r_spheres,
                box_min=self.obstacle_manager.box_min,
                box_max=self.obstacle_manager.box_max,
                start=self.p.unsqueeze(1).expand(-1, H*W, -1),
                quat_xyzw=self.q))
    
    def step(self, action):
        # type: (Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        action = self.rescale_action(action)
        self.model.step(action)
        self.progress += 1
        terminated, truncated = self.terminated(), self.truncated()
        reset = terminated | truncated
        reset_indices = reset.nonzero().squeeze(-1)
        success = truncated & ((self.p - self.target_pos).norm(dim=-1) < 0.5)
        loss, loss_components = self.loss_fn(self.target_vel, action)
        extra = {
            "truncated": truncated,
            "l": self.progress.clone(),
            "reset": reset,
            "reset_indicies": reset_indices,
            "success": success,
            "next_state_before_reset": self.state(with_grad=True),
            "loss_components": loss_components,
        }
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        if self.renderer is not None:
            if self.renderer.enable_viewer_sync or self.camera_type == "isaacgym":
                self.renderer.step(*self.state_for_render())
            self.renderer.render()
        if self.camera_type is not None:
            self.render_camera()
            extra["camera"] = self.camera_tensor.clone()
        return self.state(), loss, terminated, extra
    
    def state_for_render(self):
        w = torch.zeros_like(self.v) if self.dynamic_type == "pointmass" else self.w
        drone_state = torch.concat([self.p, self.q, self.v, w], dim=-1)
        assets_state = torch.cat([
            # self.asset_poses,
            # torch.zeros(self.n_envs, self.asset_poses.size(1), 6, device=self.device)
            self.obstacle_manager.p_obstacles,
            torch.zeros(self.n_envs, self.n_obstacles, 10, device=self.device)
        ], dim=-1)
        return torch.concat([drone_state.unsqueeze(1), assets_state], dim=1), self.target_pos
    
    def loss_fn(self, target_vel, action):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Dict[str, float]]
        # calculating the closest point on each sphere to the quadrotor
        sphere_relpos = self.obstacle_manager.p_spheres - self.p.unsqueeze(1) # [n_envs, n_spheres, 3]
        dist2surface_sphere = (sphere_relpos.norm(dim=-1) - self.obstacle_manager.r_spheres).clamp(min=0.1) # [n_envs, n_spheres]
        # calculating the closest point on each cube to the quadrotor
        nearest_point = self.p.unsqueeze(1).clamp(min=self.obstacle_manager.box_min, max=self.obstacle_manager.box_max) # [n_envs, n_cubes, 3]
        cube_relpos = nearest_point - self.p.unsqueeze(1) # [n_envs, n_cubes, 3]
        dist2surface_cube = cube_relpos.norm(dim=-1).clamp(min=0.1) # [n_envs, n_cubes]
        # concatenate the relative direction and distance to the surface of both type of obstacles
        obstacle_reldirection = F.normalize(torch.cat([sphere_relpos, cube_relpos], dim=1), dim=-1)
        dist2surface = torch.cat([dist2surface_sphere, dist2surface_cube], dim=1) # [n_envs, n_obstacles, 3]
        # calculate the obstacle avoidance loss
        approaching_vel = torch.sum(obstacle_reldirection * self._v.unsqueeze(1), dim=-1)
        oa_loss = (approaching_vel.clamp(min=0) / dist2surface.exp()).max(dim=-1).values
        
        if self.dynamic_type == "pointmass":
            pos_loss = -(-(self._p-self.target_pos).norm(dim=-1)).exp()
            
            vel_diff = (self.model._vel_ema - target_vel).norm(dim=-1)
            vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
            
            jerk_loss = F.mse_loss(self.a, action, reduction="none").sum(dim=-1)
            
            total_loss = vel_loss + 3 * oa_loss + 0.003 * jerk_loss + 5 * pos_loss
            loss_components = {
                "vel_loss": vel_loss.mean().item(),
                "pos_loss": pos_loss.mean().item(),
                "jerk_loss": jerk_loss.mean().item(),
                "oa_loss": oa_loss.mean().item(),
                "total_loss": total_loss.mean().item()
            }
        else:
            pos_loss = -(-(self._p-self.target_pos).norm(dim=-1)).exp()
            
            vel_diff = (self._v - target_vel).norm(dim=-1)
            vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
            
            jerk_loss = self._w.norm(dim=-1)
            
            total_loss = vel_loss + 3 * oa_loss + jerk_loss + 5 * pos_loss
            loss_components = {
                "vel_loss": vel_loss.mean().item(),
                "pos_loss": pos_loss.mean().item(),
                "jerk_loss": jerk_loss.mean().item(),
                "oa_loss": oa_loss.mean().item(),
                "total_loss": total_loss.mean().item()
            }
        return total_loss, loss_components

    def reset_idx(self, env_idx):
        n_resets = len(env_idx)
        state_mask = torch.zeros_like(self.model._state)
        state_mask[env_idx] = 1
        p_new = rand_range(-self.L+0.5, self.L-0.5, size=(self.n_envs, 3), device=self.device)
        new_state = torch.cat([p_new, torch.zeros(self.n_envs, self.model.state_dim-3, device=self.device)], dim=-1)
        if self.dynamic_type == "quadrotor":
            new_state[:, 6] = 1 # real part of the quaternion
        self.model._state = torch.where(state_mask.bool(), new_state, self.model._state)
        
        min_init_dist = 1.5 * self.L
        # randomly select a target position that meets the minimum distance constraint
        N = 10
        x = y = z = torch.linspace(-self.L+1, self.L-1, N, device=self.device)
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
            safety_range=self.obstacle_manager.r_obstacles.max().item()+0.5
        )
            
        self.progress[env_idx] = 0
    
    def reset(self):
        super().reset()
        if self.renderer is not None and self.camera_type == "isaacgym":
            self.renderer.step(*self.state_for_render())
        return self.state()
    
    def terminated(self) -> Tensor:
        # check if the distance between the drone's mass center and the sphere's center is less than the sum of their radius
        dist2sphere = torch.norm(self.p.unsqueeze(1) - self.obstacle_manager.p_spheres, dim=-1) # [n_envs, n_spheres]
        collision_sphere = torch.any(dist2sphere < (self.obstacle_manager.r_spheres + self.r_drone), dim=-1) # [n_envs]
        # check if the distance between the drone's mass center and the closest point on the cube is less than the drone's radius
        nearest_point = self.p.unsqueeze(1).clamp(min=self.obstacle_manager.box_min, max=self.obstacle_manager.box_max) # [n_envs, n_cubes, 3]
        dist2cube = torch.norm(nearest_point - self.p.unsqueeze(1), dim=-1) # [n_envs, n_cubes]
        collision_cube = torch.any(dist2cube < self.r_drone, dim=-1) # [n_envs]
        
        collision = collision_sphere | collision_cube
        return collision
    
    def truncated(self) -> torch.Tensor:
        out_of_bound = torch.any(self.p < -1.5*self.L, dim=-1) | \
                       torch.any(self.p >  1.5*self.L, dim=-1)
        return (self.progress > self.max_steps) | out_of_bound