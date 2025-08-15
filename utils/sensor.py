from typing import Optional, Tuple, Union
import math

import torch
from torch import Tensor
import torch.nn.functional as F
from pytorch3d import transforms as T
from omegaconf import DictConfig

from diffaero.dynamics.base_dynamics import BaseDynamics
from diffaero.utils.math import (
    quaternion_apply,
    mvp,
    quat_rotate,
    quat_rotate_inverse,
    quat_mul,
    euler_to_quaternion
)
from diffaero.utils.assets import ObstacleManager
from diffaero.utils.logger import Logger

@torch.jit.script
def raydist3d_sphere(
    obst_pos: Tensor, # [m_spheres, 3]
    obst_r: Tensor, # [m_spheres]
    start: Tensor, # [m_spheres, n_rays, 3]
    direction: Tensor, # [m_spheres, n_rays, 3]
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
    rel_pos = obst_pos.unsqueeze(1) - start # [m_spheres, n_rays, 3]
    rel_dist = torch.norm(rel_pos, dim=-1) # [m_spheres, n_rays]
    costheta = torch.cosine_similarity(rel_pos, direction, dim=-1) # [m_spheres, n_rays]
    sintheta = torch.where(costheta>0, torch.sqrt(1 - costheta**2), 0.9) # [m_spheres, n_rays]
    dist_center2ray = rel_dist * sintheta # [m_spheres, n_rays]
    obst_r = obst_r.unsqueeze(1) # [m_spheres, 1]
    raydist = rel_dist * costheta - torch.sqrt(torch.pow(obst_r, 2) - torch.pow(dist_center2ray, 2)) # [m_spheres, n_rays]
    valid = torch.logical_and(dist_center2ray < obst_r, costheta > 0) # [m_spheres, n_rays]
    raydist_valid = torch.where(valid, raydist, max_dist) # [m_spheres, n_rays]
    return raydist_valid

@torch.jit.script
def raydist3d_cube(
    p_cubes: Tensor, # [m_cubes, 3]
    lwh_cubes: Tensor, # [m_cubes, 3]
    rpy_cubes: Tensor, # [m_cubes, 3]
    start: Tensor, # [m_cubes, n_rays, 3]
    direction: Tensor, # [m_cubes, n_rays, 3]
    max_dist: float
) -> Tensor:
    """Compute the ray distance based on the start of the ray,
    the direction of the ray, and the position and radius of 
    the cubic obstacles.

    Args:
        p_cubes (torch.Tensor): The center position of the cube obstacles.
        lwh_cubes (torch.Tensor): The length, width, and height of the cube obstacles.
        rpy_cubes (torch.Tensor): The roll, pitch, and yaw angles of the cube obstacles.
        start (torch.Tensor): The start point of the ray.
        direction (torch.Tensor): The direction of the ray.
        max_dist (float): The maximum traveling distance of the ray.

    Returns:
        torch.Tensor: The distance of the ray to the nearest obstacle's surface.
    """
    if not torch.all(rpy_cubes == 0):
        rotmat = T.euler_angles_to_matrix(rpy_cubes, convention='XYZ').transpose(-1, -2) # [m_cubes, 3, 3]
        start = mvp(rotmat.unsqueeze(1), (start-p_cubes.unsqueeze(1))) # [m_cubes, n_rays, 3]
        direction = mvp(rotmat.unsqueeze(1), direction) # [m_cubes, n_rays, 3]
        box_min = -lwh_cubes / 2. # [m_cubes, 3]
        box_max =  lwh_cubes / 2. # [m_cubes, 3]
    else: # yay, no rotation!
        box_min = (p_cubes - lwh_cubes / 2.) # [m_cubes, 3]
        box_max = (p_cubes + lwh_cubes / 2.) # [m_cubes, 3]
    _tmin = (box_min.unsqueeze(1) - start) / direction # [m_cubes, n_rays, 3]
    _tmax = (box_max.unsqueeze(1) - start) / direction # [m_cubes, n_rays, 3]
    tmin = torch.where(direction < 0, _tmax, _tmin) # [m_cubes, n_rays, 3]
    tmax = torch.where(direction < 0, _tmin, _tmax) # [m_cubes, n_rays, 3]
    tentry = torch.max(tmin, dim=-1).values # [m_cubes, n_rays]
    texit = torch.min(tmax, dim=-1).values # [m_cubes, n_rays]
    valid = torch.logical_and(tentry <= texit, texit >= 0) # [m_cubes, n_rays]
    raydist = torch.where(valid, tentry, max_dist) # [m_cubes, n_rays]
    return raydist

@torch.jit.script
def raydist3d_ground_plane(
    z_ground_plane: Tensor, # [n_envs]
    start: Tensor, # [n_envs, n_rays, 3]
    direction: Tensor, # [n_envs, n_rays, 3]
    max_dist: float
) -> Tensor:
    """Compute the ray distance based on the start of the ray,
    the direction of the ray, and the position of the ground plane.

    Args:
        z_ground_plane (float): The absolute height of the ground plane in world frame.
        start (torch.Tensor): The start point of the ray.
        direction (torch.Tensor): The direction of the ray.
        max_dist (float): The maximum traveling distance of the ray.

    Returns:
        torch.Tensor: The distance of the ray to the ground plane.
    """
    z_ground_plane = z_ground_plane.unsqueeze(-1) # [n_envs, 1]
    valid = (start[..., 2] - z_ground_plane) * direction[..., 2] < 0 # [n_envs, n_rays]
    raydist = torch.where(valid, (z_ground_plane - start[..., 2]) / direction[..., 2], max_dist) # [n_envs, n_rays]
    return raydist

@torch.jit.script
def ray_directions_world2body(
    ray_directions: Tensor,
    quat_xyzw: Tensor,
    H: int,
    W: int
) -> Tensor: # [n_envs, n_rays, 3]
    quat_wxyz = quat_xyzw.roll(1, dims=-1) # [n_envs, 4]
    quat_wxyz = quat_wxyz.unsqueeze(1).expand(-1, H*W, -1) # [n_envs, n_rays, 4]
    return quaternion_apply(quat_wxyz, ray_directions.view(1, -1, 3)) # [n_envs, n_rays, 3]

@torch.jit.script
def get_ray_dist(
    sphere_ray_dists: Tensor, # [n_envs, n_spheres, n_rays]
    sphere_env_ids: Tensor,   # [m_spheres]
    sphere_ids: Tensor,       # [m_spheres]
    p_spheres: Tensor,        # [n_envs, n_spheres, 3]
    r_spheres: Tensor,        # [n_envs, n_spheres]
    cube_ray_dists: Tensor,   # [n_envs, n_cubes, n_rays]
    cube_env_ids: Tensor,     # [m_cubes]
    cube_ids: Tensor,         # [m_cubes]
    p_cubes: Tensor,          # [n_envs, n_cubes, 3]
    lwh_cubes: Tensor,        # [n_envs, n_cubes, 3]
    rpy_cubes: Tensor,        # [n_envs, n_cubes, 3]
    start: Tensor,            # [n_envs, n_rays, 3]
    ray_directions_b: Tensor, # [H, W, 3]
    quat_xyzw: Tensor,        # [n_envs, 4]
    max_dist: float,
    z_ground_plane: Optional[Tensor] = None, # [n_envs]
) -> Tuple[Tensor, Tensor]: # [n_envs, H, W], [n_envs, n_rays, 3]
    H, W = ray_directions_b.shape[:2]
    ray_directions_w = ray_directions_world2body(ray_directions_b, quat_xyzw, H, W) # [n_envs, n_rays, 3]
    
    n_spheres = p_spheres.shape[1]
    if n_spheres > 0:
        sphere_ray_starts = start[sphere_env_ids] # [m_spheres, n_rays, 3]
        sphere_ray_directions_w = ray_directions_w[sphere_env_ids] # [m_spheres, n_rays, 3]
        p_spheres = p_spheres[sphere_env_ids, sphere_ids] # [m_spheres, 3]
        r_spheres = r_spheres[sphere_env_ids, sphere_ids] # [m_spheres]
        raydist_sphere = raydist3d_sphere(p_spheres, r_spheres, sphere_ray_starts, sphere_ray_directions_w, max_dist)
        sphere_ray_dists[sphere_env_ids, sphere_ids] = raydist_sphere
    
    n_cubes = p_cubes.shape[1]
    if n_cubes > 0:
        cube_ray_starts = start[cube_env_ids] # [m_cubes, n_rays, 3]
        cube_ray_directions_w = ray_directions_w[cube_env_ids] # [m_cubes, n_rays, 3]
        p_cubes = p_cubes[cube_env_ids, cube_ids] # [m_cubes, 3]
        lwh_cubes = lwh_cubes[cube_env_ids, cube_ids] # [m_cubes, 3]
        rpy_cubes = rpy_cubes[cube_env_ids, cube_ids] # [m_cubes, 3]
        raydist_cube = raydist3d_cube(p_cubes, lwh_cubes, rpy_cubes, cube_ray_starts, cube_ray_directions_w, max_dist)
        cube_ray_dists[cube_env_ids, cube_ids] = raydist_cube
    
    raydist = torch.concat([sphere_ray_dists, cube_ray_dists], dim=1).min(dim=1).values # [n_envs, n_rays]
    if z_ground_plane is not None:
        raydist_ground_plane: Tensor = raydist3d_ground_plane(z_ground_plane, start, ray_directions_w, max_dist) # [n_envs, n_rays]
        raydist = torch.minimum(raydist, raydist_ground_plane) # [n_envs, n_rays]
    raydist.clamp_(max=max_dist)
    contact_points = ray_directions_w * raydist.unsqueeze(-1) + start # [n_envs, n_rays, 3]
    depth = 1. - raydist.reshape(-1, H, W) / max_dist # [n_envs, H, W]
    return depth, contact_points # [n_envs, H, W], [n_envs, n_rays, 3]

@torch.jit.script
def get_contact_point(
    depth: Tensor, # [n_envs, H, W]
    ray_directions_w: Tensor, # [n_envs, n_rays, 3]
    start: Tensor, # [n_envs, n_rays, 3]
    max_dist: float,
    H: int,
    W: int
):
    raydist = (1. - depth.reshape(-1, H*W, 1)) * max_dist
    return ray_directions_w * raydist + start # [n_envs, n_rays, 3]


class RayCastingSensorBase:
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.H: int
        self.W: int
        self.max_dist: float = cfg.max_dist
        self.device = device
        self.ray_directions: Tensor # [H, W, 3]
    
    def __call__(
        self,
        obstacle_manager: ObstacleManager,
        pos: Tensor, # [n_envs, 3]
        quat_xyzw: Tensor, # [n_envs, 4]
        z_ground_plane: Optional[Tensor] = None
    ) -> Tensor: # [n_envs, H, W]
        ray_starts = pos.unsqueeze(1).expand(-1, self.H * self.W, -1) # [n_envs, n_rays, 3]
        sphere_ray_dists = torch.full( # [n_envs, n_obstacles, n_rays]
            (pos.shape[0], obstacle_manager.n_spheres, self.H*self.W),
            fill_value=self.max_dist, dtype=torch.float, device=self.device)
        cube_ray_dists = torch.full( # [n_envs, n_obstacles, n_rays]
            (pos.shape[0], obstacle_manager.n_cubes, self.H*self.W),
            fill_value=self.max_dist, dtype=torch.float, device=self.device)
        dist2obstacles, nearest_points2obstacles = obstacle_manager.nearest_distance_to_obstacles(pos.unsqueeze(1))
        dist2obstacles = dist2obstacles.squeeze(1) # [n_envs, n_obstacles]
        env_ids, obstacle_ids = torch.where(dist2obstacles.le(self.max_dist))
        sphere_mask = obstacle_ids < obstacle_manager.n_spheres
        sphere_env_ids, sphere_ids = env_ids[sphere_mask], obstacle_ids[sphere_mask]
        cube_env_ids, cube_ids = env_ids[~sphere_mask], obstacle_ids[~sphere_mask] - obstacle_manager.n_spheres
        
        return get_ray_dist(
            sphere_ray_dists,
            sphere_env_ids, sphere_ids,
            obstacle_manager.p_spheres, # [n_envs, n_spheres, 3]
            obstacle_manager.r_spheres, # [n_envs, n_spheres]
            
            cube_ray_dists,
            cube_env_ids, cube_ids,
            obstacle_manager.p_cubes, # [n_envs, n_cubes, 3]
            obstacle_manager.lwh_cubes, # [n_envs, n_cubes, 3]
            obstacle_manager.rpy_cubes, # [n_envs, n_cubes, 3]
            
            ray_starts,
            self.ray_directions, # [H, W, 3]
            quat_xyzw,
            self.max_dist,
            z_ground_plane=z_ground_plane
        )
    
    def get_contact_point(
        self,
        depth: Tensor, # [n_envs, H, W]
        start: Tensor, # [n_envs, n_rays, 3]
        quat_xyzw: Tensor # [n_envs, 4]
    ) -> Tensor: # [n_envs, n_rays, 3]
        ray_directions_w = ray_directions_world2body(self.ray_directions, quat_xyzw, self.H, self.W) # [n_envs, n_rays, 3]
        return get_contact_point(depth, ray_directions_w, start, self.max_dist, self.H, self.W)


class Camera(RayCastingSensorBase):
    def __init__(self, cfg: DictConfig, device: torch.device):
        assert cfg.name == "camera"
        super().__init__(cfg, device)
        self.H: int = cfg.height
        self.W: int = cfg.width
        self.hfov: float = cfg.horizontal_fov
        self.vfov: float = self.hfov * self.H / self.W
        self.max_dist: float = cfg.max_dist
        self.device = device
        self.ray_directions = F.normalize(self._get_ray_directions_plane(), dim=-1) # [H, W, 3]
        # self.ray_directions = F.normalize(self._get_ray_directions_sphere(), dim=-1) # [H, W, 3]
        
    def _get_ray_directions_sphere(self):
        forward = torch.tensor([[[1., 0., 0.]]], device=self.device).expand(self.H, self.W, -1) # [H, W, 3]
        
        pitch = torch.linspace(0.5*self.vfov, -0.5*self.vfov, self.H, device=self.device) * torch.pi / 180
        yaw = torch.linspace(-0.5*self.hfov, 0.5*self.hfov, self.W, device=self.device) * torch.pi / 180
        pitch, yaw = torch.meshgrid(pitch, yaw, indexing="ij")
        roll = torch.zeros_like(pitch)
        euler_angles = torch.stack([yaw, pitch, roll], dim=-1)
        rotmat = T.euler_angles_to_matrix(euler_angles, convention='ZYX') # [H, W, 3, 3]
        directions = rotmat.transpose(-1, -2) @ forward.unsqueeze(-1) # [H, W, 3, 1]
        return directions.squeeze(-1) # [H, W, 3]

    def _get_ray_directions_plane(self):
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


class LiDAR(RayCastingSensorBase):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        self.H: int = cfg.n_rays_vertical
        self.W: int = cfg.n_rays_horizontal
        self.dep_angle_rad: float = cfg.depression_angle * torch.pi / 180
        self.ele_angle_rad: float = cfg.elevation_angle * torch.pi / 180
        self.ray_directions = F.normalize(self._get_ray_directions(), dim=-1) # [H, W, 3]
    
    def _get_ray_directions(self):
        forward = torch.tensor([[[1., 0., 0.]]], device=self.device).expand(self.H, self.W, -1) # [H, W, 3]
        
        yaw = torch.arange(0, self.W, device=self.device) / self.W * 2 * torch.pi
        pitch = torch.linspace(self.ele_angle_rad, self.dep_angle_rad, self.H, device=self.device)
        pitch, yaw = torch.meshgrid(pitch, yaw, indexing="ij")
        roll = torch.zeros_like(pitch)
        rpy = torch.stack([roll, pitch, yaw], dim=-1)
        rotmat = T.euler_angles_to_matrix(rpy, convention='XYZ') # [H, W, 3, 3]
        directions = rotmat.transpose(-1, -2) @ forward.unsqueeze(-1) # [H, W, 3, 1]
        return directions.squeeze(-1) # [H, W, 3]


class RelativePositionSensor:
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.H: int = cfg.n_obstacles + int(cfg.ceiling) + 4 * int(cfg.walls)
        self.W: int = 3
        self.device = device
    
    def __call__(
        self,
        obstacle_manager: ObstacleManager,
        pos: Tensor, # [n_envs, n_rays, 3]
        quat_xyzw: Tensor, # [n_envs, 4]
        z_ground_plane: Optional[float] = None
    ) -> Tensor: # [n_envs, H, W]
        dist2obstacles, nearest_points2obstacles = obstacle_manager.nearest_distance_to_obstacles(pos.unsqueeze(1))
        dist2obstacles, nearest_points2obstacles = dist2obstacles.squeeze(1), nearest_points2obstacles.squeeze(1)
        obst_relpos = nearest_points2obstacles - pos.unsqueeze(1)
        sorted_idx = dist2obstacles.argsort(dim=-1).unsqueeze(-1).expand(-1, -1, 3)
        sorted_obst_relpos = obst_relpos.gather(dim=1, index=sorted_idx) # [n_envs, n_obstacles, 3]
        return sorted_obst_relpos


def build_sensor(cfg: DictConfig, device: torch.device) -> Union[Camera, LiDAR, RelativePositionSensor]:
    sensor_alias = {
        "camera": Camera,
        "lidar": LiDAR,
        "relpos": RelativePositionSensor,
    }
    return sensor_alias[cfg.name](cfg, device)


class IMU:
    def __init__(self, cfg: DictConfig, dynamics: BaseDynamics):
        self.dynamics = dynamics
        self.n_envs = dynamics.n_envs
        self.n_agents = dynamics.n_agents
        self.device = dynamics.device
        factory_kwargs = {
            "device": self.device,
            "dtype": torch.float32,
        }
        self.dt = torch.tensor(dynamics.dt, **factory_kwargs)
        self.sqrt_dt = torch.sqrt(self.dt)
        self.mounting_range_rad = cfg.imu_mounting_error_range_deg * torch.pi / 180.
        self.mounting_quat_xyzw = torch.zeros((self.n_envs, self.n_agents, 4), **factory_kwargs)
        
        self.acc_drift_std: Tensor = cfg.acc_drift_std * self.sqrt_dt
        self.acc_noise_std: Tensor = cfg.acc_noise_std / self.sqrt_dt
        self.acc_drift_b = torch.zeros((self.n_envs, self.n_agents, 3), **factory_kwargs)
        self.acc_noise_b = torch.zeros((self.n_envs, self.n_agents, 3), **factory_kwargs)
        self.vel_drift_w = torch.zeros((self.n_envs, self.n_agents, 3), **factory_kwargs)
        self.pos_drift_w = torch.zeros((self.n_envs, self.n_agents, 3), **factory_kwargs)        
        
        self.gyro_drift_std: Tensor = cfg.gyro_drift_std * self.sqrt_dt
        self.gyro_noise_std: Tensor = cfg.gyro_noise_std / self.sqrt_dt
        self.gyro_drift_b = torch.zeros((self.n_envs, self.n_agents, 3), **factory_kwargs)
        self.gyro_noise_b = torch.zeros((self.n_envs, self.n_agents, 3), **factory_kwargs)
        self.pose_drift_b = torch.zeros((self.n_envs, self.n_agents, 3), **factory_kwargs)
        
        if self.n_agents == 1:
            self.mounting_quat_xyzw.squeeze_(1)
            self.acc_drift_b.squeeze_(1)
            self.acc_noise_b.squeeze_(1)
            self.vel_drift_w.squeeze_(1)
            self.pos_drift_w.squeeze_(1)
            self.gyro_drift_b.squeeze_(1)
            self.gyro_noise_b.squeeze_(1)
            self.pose_drift_b.squeeze_(1)
        self.enable_drift = int(cfg.enable_drift)
        self.enable_noise = int(cfg.enable_noise)
    
    def sensor2body(self, vec_s: Tensor) -> Tensor:
        return quat_rotate(self.mounting_quat_xyzw, vec_s)
    
    def body2sensor(self, vec_b: Tensor) -> Tensor:
        return quat_rotate_inverse(self.mounting_quat_xyzw, vec_b)
    
    def sensor2world(self, vec_s: Tensor) -> Tensor:
        return self.dynamics.body2world(self.sensor2body(vec_s))

    def world2sensor(self, vec_w: Tensor) -> Tensor:
        return self.body2sensor(self.dynamics.world2body(vec_w))
    
    def step(self):
        # Drift and noise are generated in sensor frame
        self.gyro_drift_b += self.sensor2body(torch.randn_like(self.gyro_drift_b) * self.gyro_drift_std)
        self.gyro_noise_b  = self.sensor2body(torch.randn_like(self.gyro_noise_b) * self.gyro_noise_std)
        self.pose_drift_b += (
            self.enable_drift * self.gyro_drift_b + 
            self.enable_noise * self.gyro_noise_b
        ) * self.dt

        self.acc_drift_b += self.sensor2body(torch.randn_like( self.acc_drift_b) * self.acc_drift_std)
        self.acc_noise_b  = self.sensor2body(torch.randn_like( self.acc_noise_b) * self.acc_noise_std)
        self.vel_drift_w += self.dt * self.dynamics.body2world(
            self.enable_drift * self.acc_drift_b + 
            self.enable_noise * self.acc_noise_b
        )
        self.pos_drift_w += self.vel_drift_w * self.dt
        # Logger.debug(self.pos_drift_w[0].norm(dim=0))
    
    def reset_idx(self, env_idx: Tensor):
        self.mounting_quat_xyzw[env_idx] = \
            torch.rand_like(self.mounting_quat_xyzw[env_idx]) * 2 * self.mounting_range_rad - self.mounting_range_rad
        self.acc_drift_b[env_idx] = torch.zeros_like(self.acc_drift_b[env_idx])
        self.acc_noise_b[env_idx] = torch.zeros_like(self.acc_noise_b[env_idx])
        self.gyro_drift_b[env_idx] = torch.zeros_like(self.gyro_drift_b[env_idx])
        self.gyro_noise_b[env_idx] = torch.zeros_like(self.gyro_noise_b[env_idx])
        self.pose_drift_b[env_idx] = torch.zeros_like(self.pose_drift_b[env_idx])
        self.vel_drift_w[env_idx] = torch.zeros_like(self.vel_drift_w[env_idx])
        self.pos_drift_w[env_idx] = torch.zeros_like(self.pos_drift_w[env_idx])
    
    @property
    def a_w(self):
        """Acceleration measurement in world frame"""
        return self.dynamics.body2world(self.a_b)
    
    @property
    def a_b(self):
        """Acceleration measurement in body frame"""
        a_b_true = self.dynamics.world2body(self.dynamics.a)
        a_b_measured = (
            a_b_true +
            self.enable_drift * self.acc_drift_b +
            self.enable_noise * self.acc_noise_b
        )
        return a_b_measured

    @property
    def v_w(self):
        """Velocity measurement in world frame"""
        v_w_true = self.dynamics.v
        v_w_measured = v_w_true + self.vel_drift_w
        return v_w_measured
    
    @property
    def v_b(self):
        """Velocity measurement in body frame"""
        return self.dynamics.world2body(self.v_w)
    
    @property
    def p_w(self):
        """Position measurement in world frame"""
        p_w_true = self.dynamics.p
        p_w_measured = p_w_true + self.pos_drift_w
        return p_w_measured
    
    @property
    def p_b(self):
        """Position measurement in body frame"""
        return self.dynamics.world2body(self.p_w)
    
    @property
    def q(self):
        """Quaternion measurement in world frame"""
        q_true = self.dynamics.q
        q_measured = quat_mul(q_true, euler_to_quaternion(*self.pose_drift_b.unbind(dim=-1)))
        return q_measured