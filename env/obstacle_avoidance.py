from typing import Tuple, Dict, Union

from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch3d import transforms as T
from utils.assets import ObstacleManager

from quaddif.env.base_env import BaseEnv
from quaddif.model.quad import QuadrotorModel, PointMassModel
from quaddif.utils.render import ObstacleAvoidanceRenderer
from quaddif.utils.math import unitization, axis_rotmat, rand_range

@torch.jit.script
def raydist3d(
    obst_pos: Tensor, # [n_envs, n_obstacles, 3]
    obst_r: Tensor, # [n_envs, n_obstacles]
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
    rel_pos = obst_pos.unsqueeze(1) - start.unsqueeze(2) # [n_envs, n_rays, n_obstacles, 3]
    # rel_pos = rel_pos.unsqueeze(-2) # [n_envs, n_agents, n_obstacles, 1, 3]
    rel_dist = torch.norm(rel_pos, dim=-1) # [n_envs, n_agents, n_obstacles]
    costheta = torch.cosine_similarity(rel_pos, direction.unsqueeze(2), dim=-1) # [n_envs, n_rays, n_obstacles]
    sintheta = torch.where(costheta>0, torch.sqrt(1 - costheta**2), 0.9) # [n_envs, n_rays, n_obstacles]
    dist_center2ray = rel_dist * sintheta # [n_envs, n_rays, n_obstacles]
    obst_r = obst_r.unsqueeze(1) # [n_envs, 1, n_obstacles]
    raydist = rel_dist * costheta - torch.sqrt(torch.pow(obst_r, 2) - torch.pow(dist_center2ray, 2)) # [n_envs, n_rays, n_obstacles]
    valid = torch.logical_and(dist_center2ray < obst_r, costheta > 0)
    valid = torch.logical_and(valid, raydist < max_dist)
    # valid = (dist_center2ray < obst_r) & (costheta > 0) & (raydist < max_dist) # [n_envs, n_rays, n_obstacles]
    raydist_valid = torch.where(valid, raydist, max_dist) # [n_envs, n_rays, n_obstacles]
    return raydist_valid.min(dim=-1).values # [n_envs, n_rays]

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
        obst_pos: Tensor, # [n_envs, n_obstacles, 3]
        obst_r: Tensor, # [n_envs, n_obstacles]
        start: Tensor, # [n_envs, n_rays, 3]
        quat_xyzw: Tensor, # [n_envs, 4]
    ) -> Tensor: # [n_envs, n_rays]
        ray_directions = self.world2body(quat_xyzw) # [n_envs, n_rays, 3]
        raydist: Tensor = raydist3d(obst_pos, obst_r, start, ray_directions, self.max_dist) # [n_envs, n_rays, n_obstacles]
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

class PointMassObstacleAvoidance(BaseEnv):
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.model = PointMassModel(cfg.quad, cfg.dt, cfg.n_substeps, device)
        self.camera_type = cfg.camera.type
        self.obstacle_manager = ObstacleManager(cfg, device)
        self.r_obstacles = self.obstacle_manager.generate_obstacles(r_min=0.2, r_max=2.0, r_step=0.2)
        self.n_obstacles = self.obstacle_manager.n_obstacles
        
        if self.camera_type is not None:
            H, W = cfg.camera.height, cfg.camera.width
            self.state_dim = 13 + H * W # flattened depth image as additional observation
            self.camera_tensor = torch.zeros((cfg.n_envs, H, W), device=device)
            if self.camera_type == "raydist":
                self.camera = Camera(cfg.camera, device=device)
        else:
            # relative position of obstacles as additional observation
            self.state_dim = 13 + self.n_obstacles * 3
        
        if not cfg.render.headless or self.camera_type == "isaacgym":
            self.renderer = ObstacleAvoidanceRenderer(cfg.render, device.index, self.r_obstacles)
            self.asset_poses = torch.zeros(
                cfg.n_envs, self.renderer.asset_manager.assets_per_env, 7, device=device)
        else:
            self.renderer = None # headless and camera_type == "raydist"
        self.p_obstacles = torch.zeros(cfg.n_envs, self.n_obstacles, 3, device=device)
        
        self.action_dim = 3
        self.r_drone = cfg.r_drone
        super(PointMassObstacleAvoidance, self).__init__(cfg, device)
    
    def state(self, with_grad=False):
        state = [self.target_vel, self._v, self._a, self.q]
        if self.camera_type is not None:
            state.append(self.camera_tensor.flatten(1))
        elif self.camera_type is None:
            obst_relpos = self.asset_poses[:, :self.n_obstacles, :3] - self._p.unsqueeze(1)
            sorted_idx = obst_relpos.norm(dim=-1).argsort(dim=-1).unsqueeze(-1).expand(-1, -1, 3)
            state.append(obst_relpos.gather(dim=1, index=sorted_idx).flatten(1))
        else:
            raise Exception(f"Unknown camera type {self.camera_type}, expected 'isaacgym', 'raydist' or None")
        state = torch.cat(state, dim=-1)
        return state if with_grad else state.detach()
    
    def render_camera(self):
        if self.camera_type == "isaacgym":
            self.camera_tensor.copy_(self.renderer.render_camera())
        elif self.camera_type == "raydist":
            H, W = self.camera_tensor.shape[1:]
            self.camera_tensor.copy_(self.camera.get_raydist(
                obst_pos=self.p_obstacles,
                obst_r=self.r_obstacles,
                start=self.p.unsqueeze(1).expand(-1, H*W, -1),
                quat_xyzw=self.q))
    
    def step(self, action):
        # type: (Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        action = self.rescale_action(action)
        self._state = self.model(self._state, action)
        self._vel_ema = torch.lerp(self._vel_ema, self._v, self.vel_ema_factor)
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
            self.renderer.step(*self.state_for_render())
            self.renderer.render()
        if self.camera_type is not None:
            self.render_camera()
            extra["camera"] = self.camera_tensor.clone()
        return self.state(), loss, terminated, extra
    
    def state_for_render(self):
        w = torch.zeros_like(self.v)
        drone_state = torch.concat([self.p, self.q, self.v, w], dim=-1)
        assets_state = torch.cat([
            self.asset_poses,
            torch.zeros(self.n_envs, self.asset_poses.size(1), 6, device=self.device)
        ], dim=-1)
        return torch.concat([drone_state.unsqueeze(1), assets_state], dim=1), self.target_pos
    
    def loss_fn(self, target_vel, action):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Dict[str, float]]
        pos_loss = -(-(self._p-self.target_pos).norm(dim=-1)).exp()
        
        vel_diff = (self._vel_ema - target_vel).norm(dim=-1)
        vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
        
        obstacle_relpos = self.p_obstacles - self.p.unsqueeze(1) # [n_envs, n_obstacles, 3]
        dist2surface = (obstacle_relpos.norm(dim=-1) - self.r_obstacles).clamp(min=0.1)
        approaching_vel = torch.sum(F.normalize(obstacle_relpos, dim=-1) * self._v.unsqueeze(1), dim=-1)
        oa_loss = (approaching_vel.clamp(min=0) / dist2surface.exp()).max(dim=-1).values
    
        # obstacle_relpos = self.p_obstacles - self.p.unsqueeze(1) # [n_envs, n_obstacles, 3]
        # dist2surface = (obstacle_relpos.norm(dim=-1) - self.r_obstacles).clamp(min=0.1)
        # approaching_vel = (F.normalize(obstacle_relpos, dim=-1) * self._v.unsqueeze(1)).norm(dim=-1).clamp(min=0)
        # dangerous = dist2surface < 0.7
        # oa_loss = (approaching_vel * dangerous.float()).max(dim=-1).values
        
        jerk_loss = F.mse_loss(self._a, action, reduction="none").sum(dim=-1)
        
        total_loss = vel_loss + 3 * oa_loss + 0.003 * jerk_loss + 5 * pos_loss
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
        state_mask = torch.zeros_like(self._state)
        state_mask[env_idx] = 1
        p_new = rand_range(-self.L+1, self.L-1, size=(self.n_envs, 3), device=self.device)
        v_new = torch.zeros_like(self.v)
        a_new = torch.zeros_like(self.a)
        new_state = torch.cat([p_new, v_new, a_new], dim=-1)
        self._state = torch.where(state_mask.bool(), new_state, self._state)
        
        # target position
        min_init_dist = 1.5 * self.L
        N = 10
        x = y = z = torch.linspace(-self.L+1, self.L-1, N, device=self.device)
        random_idx = torch.randperm(N**3, device=self.device)
        xyz = torch.stack(torch.meshgrid(x, y, z), dim=-1).reshape(N**3, 3)[random_idx]
        validility: torch.BoolTensor = (xyz[None, ...] - self.p[env_idx, None, :]).norm(dim=-1) > min_init_dist
        sub_idx = validility.nonzero()
        env_sub_idx = torch.tensor([(sub_idx[:, 0] == i).sum() for i in range(n_resets)]).roll(1, dims=0)
        env_sub_idx[0] = 0
        env_sub_idx = torch.cumsum(env_sub_idx, dim=0)
        self.target_pos[env_idx] = xyz[sub_idx[env_sub_idx, 1]]
        # check that all regenerated initial and target positions meet the minimal distance contraint
        assert torch.all((self.p[env_idx] - self.target_pos[env_idx]).norm(dim=-1) > min_init_dist).item()
        
        # obstacle position
        self.p_obstacles[env_idx], mask = self.obstacle_manager.randomize_asset_pose(
            env_idx=env_idx,
            drone_init_pos=self.p[env_idx],
            target_pos=self.target_pos[env_idx],
            safety_range=self.r_obstacles.max().item()+0.5
        )
        if self.renderer is not None:
            self.asset_poses[env_idx, :self.n_obstacles, :3] = self.p_obstacles[env_idx]
            
        self.progress[env_idx] = 0
    
    def reset(self):
        super().reset()
        if self.camera_type == "isaacgym":
            self.renderer.step(*self.state_for_render())
        return self.state()
    
    def terminated(self) -> Tensor:
        # out_of_bound = torch.any(self.p < -1.5*self.L, dim=-1) | \
        #                torch.any(self.p >  1.5*self.L, dim=-1)
        dist2obst = torch.norm(self.p.unsqueeze(1) - self.p_obstacles[:, :, :3], dim=-1)
        collision = torch.any(dist2obst < (self.r_obstacles + self.r_drone), dim=-1)
        return collision
    
    def truncated(self) -> torch.Tensor:
        out_of_bound = torch.any(self.p < -1.5*self.L, dim=-1) | \
                       torch.any(self.p >  1.5*self.L, dim=-1)
        return (self.progress > self.max_steps) | out_of_bound