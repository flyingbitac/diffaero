from typing import Optional, Tuple, List

import numpy as np
from omegaconf import DictConfig
import torch
from torch import Tensor
from torch.nn import functional as F
from pytorch3d import transforms as T

from quaddif.utils.math import mat_vec_mul, rand_range

@torch.jit.script
def are_points_inside_spheres(
    points: Tensor, # [n_envs, n_points, 3]
    p_spheres: Tensor, # [n_envs, n_spheres, 3]
    r_spheres: Tensor # [n_envs, n_spheres]
) -> Tensor:
    assert points.shape[-1] == 3, "Points should have shape (n_envs, n_points, 3)"
    dist2spheres = torch.norm(points.unsqueeze(2) - p_spheres.unsqueeze(1), dim=-1)  # [n_envs, n_points, n_spheres]
    inside = torch.any(dist2spheres < r_spheres.unsqueeze(1), dim=-1)  # [n_envs, n_points]
    return inside

@torch.jit.script
def are_points_inside_cubes(
    points: Tensor, # [n_envs, n_points, 3]
    p_cubes: Tensor, # [n_envs, n_cubes, 3]
    lwh_cubes: Tensor, # [n_envs, n_cubes, 3]
    rpy_cubes: Optional[Tensor] = None # [n_envs, n_cubes, 3]
) -> Tensor:
    if rpy_cubes is not None:
        rotmat = T.euler_angles_to_matrix(rpy_cubes, convention='XYZ').transpose(-1, -2)  # [n_envs, n_cubes, 3, 3]
        points_cube_frame = mat_vec_mul(rotmat.unsqueeze(1), (points.unsqueeze(2) - p_cubes.unsqueeze(1)))  # [n_envs, n_points, n_cubes, 3]
        box_min, box_max = -lwh_cubes / 2, lwh_cubes / 2  # [n_envs, n_cubes, 3]
    else:
        box_min, box_max = p_cubes - lwh_cubes / 2, p_cubes + lwh_cubes / 2  # [n_envs, n_cubes, 3]]
        points_cube_frame = points.unsqueeze(2)  # [n_envs, n_points, n_cubes, 3]
    inside = torch.any(torch.logical_and(
        torch.all(points_cube_frame >= box_min.unsqueeze(1), dim=-1),  # [n_envs, n_points, n_cubes]
        torch.all(points_cube_frame <= box_max.unsqueeze(1), dim=-1)   # [n_envs, n_points, n_cubes]
    ), dim=-1)  # [n_envs, n_points]
    return inside

@torch.jit.script
def nearest_distance_to_spheres(
    points: Tensor, # [n_envs, n_points, 3]
    p_spheres: Tensor, # [n_envs, n_spheres, 3]
    r_spheres: Tensor # [n_envs, n_spheres]
) -> Tuple[Tensor, Tensor]:
    relpos = p_spheres.unsqueeze(1) - points.unsqueeze(2)  # [n_envs, n_points, n_spheres, 3]
    vector_center2surface = F.normalize(-relpos, dim=-1) * r_spheres.reshape(p_spheres.shape[0], 1, p_spheres.shape[1], 1)  # [n_envs, n_points, n_spheres, 3]
    nearest_points = p_spheres.unsqueeze(1) + vector_center2surface  # [n_envs, n_points, n_spheres, 3]
    dist2sphere_center = torch.norm(relpos, dim=-1)  # [n_envs, n_points, n_spheres]
    dist2surface = dist2sphere_center - r_spheres.unsqueeze(1)  # [n_envs, n_points, n_spheres]
    return dist2surface, nearest_points

@torch.jit.script
def nearest_distance_to_cubes(
    points: Tensor, # [n_envs, n_points, 3]
    p_cubes: Tensor, # [n_envs, n_cubes, 3]
    lwh_cubes: Tensor, # [n_envs, n_cubes, 3]
    rpy_cubes: Optional[Tensor] = None # [n_envs, n_cubes, 3]
) -> Tuple[Tensor, Tensor]:
    if rpy_cubes is not None:
        rotmat = T.euler_angles_to_matrix(rpy_cubes, convention='XYZ').transpose(-1, -2)  # [n_envs, n_cubes, 3, 3]
        points_cube_frame = mat_vec_mul(rotmat.unsqueeze(1), (points.unsqueeze(2) - p_cubes.unsqueeze(1)))  # [n_envs, n_points, n_cubes, 3]
        box_min, box_max = -lwh_cubes / 2, lwh_cubes / 2  # [n_envs, n_cubes, 3]
        nearest_points_cube_frame = torch.clamp(points_cube_frame, box_min.unsqueeze(1), box_max.unsqueeze(1))  # [n_envs, n_points, n_cubes, 3]
        nearest_points = mat_vec_mul(rotmat.unsqueeze(1).transpose(-1, -2), nearest_points_cube_frame) + p_cubes.unsqueeze(1)  # [n_envs, n_points, n_cubes, 3]
    else:
        box_min, box_max = p_cubes - lwh_cubes / 2, p_cubes + lwh_cubes / 2  # [n_envs, n_cubes, 3]
        nearest_points = torch.clamp(points.unsqueeze(2), box_min.unsqueeze(1), box_max.unsqueeze(1)) # [n_envs, n_points, n_cubes, 3]
    dist2surface = torch.norm(nearest_points - points.unsqueeze(2), dim=-1)  # [n_envs, n_points, n_cubes]
    return dist2surface, nearest_points

class ObstacleManager:
    def __init__(self, cfg: DictConfig, n_envs: int, device: torch.device):
        self.n_envs = n_envs
        self.walls: bool = cfg.walls
        self.ceiling: bool = cfg.ceiling
        self.height_scale: float = cfg.height_scale
        n_obstacles: int = cfg.n_obstacles
        self.n_spheres: int = int(n_obstacles * cfg.sphere_percentage)
        self.n_cubes: int = n_obstacles - self.n_spheres + 4 * int(self.walls) + int(self.ceiling)
        self.n_obstacles = self.n_cubes + self.n_spheres
        self.sphere_rmin, self.sphere_rmax, self.sphere_rstep = list(cfg.sphere_radius_range)
        # lwh for Length(along x axis), Width(along y axis) and Height(along z axis)
        self.cube_lwmin, self.cube_lwmax, self.cube_lwstep = list(cfg.cube_lw_range)
        self.cube_hmin,  self.cube_hmax,  self.cube_hstep  = list(cfg.cube_h_range)
        self.randomize_cube_pose: bool = cfg.randomize_cube_pose
        self.cube_roll_pitch_range = cfg.cube_roll_pitch_range * torch.pi / 180.0
        
        self.randpos_minstd: float = cfg.randpos_std_min
        self.randpos_maxstd: float = cfg.randpos_std_max
        self.safety_range: float = cfg.safety_range
        self.device = device
        
        self.r_obstacles = torch.empty(self.n_envs, self.n_obstacles, device=self.device)
        self.p_obstacles = torch.zeros(self.n_envs, self.n_obstacles, 3, device=device)
        self.lwh_cubes = torch.empty(self.n_envs, self.n_cubes, 3, device=device)
        self.rpy_cubes = torch.zeros(self.n_envs, self.n_cubes, 3, device=device)
        
    @property
    def p_spheres(self): return self.p_obstacles[:, :self.n_spheres]
    @property
    def r_spheres(self): return self.r_obstacles[:, :self.n_spheres]
    @property
    def p_cubes(self): return self.p_obstacles[:, self.n_spheres:]
    
    def are_points_inside_obstacles(self, points: Tensor) -> Tensor: # [n_envs, n_points]
        rpy_cubes = self.rpy_cubes if self.randomize_cube_pose else None
        inside_spheres = are_points_inside_spheres(points, self.p_spheres, self.r_spheres) # [n_envs, n_points]
        inside_cubes = are_points_inside_cubes(points, self.p_cubes, self.lwh_cubes, rpy_cubes)  # [n_envs, n_points]
        return torch.logical_or(inside_spheres, inside_cubes)  # [n_envs, n_points]
    
    def nearest_distance_to_obstacles(self, points: Tensor) -> Tuple[Tensor, Tensor]: # [n_envs, n_points, n_obstacle(, 3)]
        rpy_cubes = self.rpy_cubes if self.randomize_cube_pose else None
        nearest_dist_spheres, nearest_points_spheres = nearest_distance_to_spheres(points, self.p_spheres, self.r_spheres)    # [n_envs, n_points, n_spheres]
        nearest_dist_cubes, nearest_points_cubes = nearest_distance_to_cubes(points, self.p_cubes, self.lwh_cubes, rpy_cubes) # [n_envs, n_points, n_cubes]
        nearest_dist = torch.cat([nearest_dist_spheres, nearest_dist_cubes], dim=-1)       # [n_envs, n_points, n_obstacles]
        nearest_points = torch.cat([nearest_points_spheres, nearest_points_cubes], dim=-2) # [n_envs, n_points, n_obstacles, 3]
        return nearest_dist, nearest_points

    def randomize_obstacles_sizes(
        self, 
        env_spacing: Tensor, # [n_resets]
        env_idx: Tensor, # [n_resets]
    ):
        n_resets = len(env_idx)
        # randomly generate spheral obstacles
        radius = torch.arange(self.sphere_rmin, self.sphere_rmax, self.sphere_rstep, device=self.device)
        idx_radius = torch.randint(0, len(radius), (n_resets, self.n_spheres), device=self.device)
        selected_radius = radius[idx_radius]
        self.r_obstacles[env_idx, :self.n_spheres] = selected_radius

        # randomly generate cubical obstacles
        cube_lw = torch.arange(self.cube_lwmin, self.cube_lwmax, self.cube_lwstep, device=self.device)
        cube_h  = torch.arange(self.cube_hmin,  self.cube_hmax,  self.cube_hstep, device=self.device)
        idx_lw = torch.randint(0, len(cube_lw), (n_resets, self.n_cubes, 2), device=self.device)
        idx_h  = torch.randint(0, len(cube_h),  (n_resets, self.n_cubes, 1), device=self.device)
        randomized_cube_lw = cube_lw[idx_lw]
        randomized_cube_h  = cube_h[idx_h]
        randomized_cube_lwh = torch.cat([randomized_cube_lw, randomized_cube_h], dim=-1)
        self.lwh_cubes[env_idx] = randomized_cube_lwh
        self.r_obstacles[env_idx, self.n_spheres:] = randomized_cube_lwh.div(2).norm(dim=-1)
        
        L, H = env_spacing, self.height_scale * env_spacing
        ones = torch.ones_like(L)
        if self.walls:
            self.lwh_cubes[env_idx, :4] = torch.stack([
                torch.stack([2*L, 0.1*ones, 2*H], dim=-1),
                torch.stack([2*L, 0.1*ones, 2*H], dim=-1),
                torch.stack([0.1*ones, 2*L, 2*H], dim=-1),
                torch.stack([0.1*ones, 2*L, 2*H], dim=-1)
            ], dim=-2)
        if self.ceiling:
            self.lwh_cubes[env_idx, int(self.walls)*4] = torch.stack([2*L, 2*L, 0.1*ones], dim=-1)
    
    def randomize_obstacles_poses(self, env_idx: Tensor):
        n_resets = len(env_idx)
        if self.randomize_cube_pose:
            self.rpy_cubes[env_idx, :, :2] = rand_range(
                -self.cube_roll_pitch_range, self.cube_roll_pitch_range, (n_resets, self.n_cubes, 2), device=self.device)
            self.rpy_cubes[env_idx, :, 2:] = rand_range(
                -torch.pi, torch.pi, (n_resets, self.n_cubes, 1), device=self.device)
        else:
            self.rpy_cubes.fill_(0.)
    
    def randomize_obstacles_positions(
        self,
        env_spacing: Tensor, # [num_resets]
        env_idx: Tensor, # [num_resets]
        drone_init_pos: Tensor, # [num_resets, 3]
        target_pos: Tensor, # [num_resets, 3]
        n_enabled_obstacles: Optional[Tensor] = None, # [num_resets]
    ):
        if self.n_obstacles == 0:
            return self.p_obstacles[env_idx]
        
        n_resets = len(env_idx)
        
        safety_range: torch.Tensor = self.r_obstacles[env_idx] + self.safety_range # [n_resets, n_obstacles]
        
        rel_pos = target_pos - drone_init_pos
        # target_axis: unit vector in the direction of the target's relative position
        target_axis = F.normalize(rel_pos, dim=-1)
        # horizontal_axis: unit vector in the horizontal plane, perpendicular to the target_axis
        horizontal_axis = F.normalize(torch.stack([
            -rel_pos[:, 1],
            rel_pos[:, 0],
            torch.zeros(n_resets, device=self.device)], dim=-1), dim=-1)
        # third_axis: unit vector perpendicular to two other vectors
        third_axis = torch.cross(target_axis, horizontal_axis, dim=-1)
        
        # sample uniformally along the target axis
        target_axis_ratio = torch.rand(n_resets, self.n_obstacles, 1, device=self.device)
        target_axis_pos = target_axis_ratio * rel_pos.unsqueeze(1)
        
        # sample from gaussian distribution
        std = torch.abs(target_axis_ratio - 0.5) * 2 * (self.randpos_maxstd - self.randpos_minstd) + self.randpos_minstd
        horizontal_axis_ratio = torch.randn(
            n_resets, self.n_obstacles, 1, device=self.device) * std
        third_axis_ratio = torch.randn(
            n_resets, self.n_obstacles, 1, device=self.device) * std * self.height_scale
        
        horizontal_axis_pos = horizontal_axis_ratio * horizontal_axis.unsqueeze(1)
        third_axis_pos = third_axis_ratio * third_axis.unsqueeze(1)
        
        relpos2target_axis = horizontal_axis_pos + third_axis_pos # [n_resets, n_obstacles, 3]
        relpos2drone = target_axis_pos + relpos2target_axis # [n_resets, n_obstacles, 3]
        relpos2target = relpos2drone - rel_pos.unsqueeze(1) # [n_resets, n_obstacles, 3]
        
        # whether the sampled point is too close to the drone's initial position
        # or to the target position
        dist2drone, dist2target = relpos2drone.norm(dim=-1), relpos2target.norm(dim=-1) # [n_resets, n_obstacles]
        tooclose2drone = torch.lt(dist2drone, safety_range) # [n_resets, n_obstacles, 3]
        tooclose2target = torch.lt(dist2target, safety_range) # [n_resets, n_obstacles, 3]
        tooclose = torch.logical_or(tooclose2drone, tooclose2target) # [n_resets, n_obstacles, 3]
        
        # push obstacles away from the line from drone's initial position to the target position
        if torch.any(tooclose):
            idx = tooclose.nonzero(as_tuple=True)
            relpos2drone[idx] += F.normalize(relpos2target_axis[idx], dim=-1) * self.safety_range
            assert torch.all(torch.logical_and(relpos2drone.norm(dim=-1) >= self.safety_range, (relpos2drone - rel_pos.unsqueeze(1)).norm(dim=-1) >= self.safety_range))
        
        # nearest_dist2drone_sphere, nearest_point2drone_sphere = nearest_distance_to_spheres( # [n_resets, 1, n_spheres(, 3)]
        #     drone_init_pos.unsqueeze(1), self.p_spheres[env_idx], self.r_spheres[env_idx])
        # nearest_dist2drone_cube, nearest_point2drone_cube = nearest_distance_to_cubes( # [n_resets, 1, n_cubes(, 3)]
        #     drone_init_pos.unsqueeze(1), self.p_cubes[env_idx], self.lwh_cubes[env_idx], self.rpy_cubes[env_idx])
        # nearest_dist2drone = torch.cat([nearest_dist2drone_sphere, nearest_dist2drone_cube], dim=-1).squeeze(1) # [n_resets, n_obstacles]
        # nearest_point2drone = torch.cat([nearest_point2drone_sphere, nearest_point2drone_cube], dim=-2).squeeze(1) # [n_resets, n_obstacles, 3]
        # tooclose2drone = nearest_dist2drone.lt(self.safety_range) # [n_resets, n_obstacles]

        # nearest_dist2target_sphere, nearest_point2target_sphere = nearest_distance_to_spheres( # [n_resets, 1, n_spheres(, 3)]
        #     target_pos.unsqueeze(1), self.p_spheres[env_idx], self.r_spheres[env_idx])
        # nearest_dist2target_cube, nearest_point2target_cube = nearest_distance_to_cubes( # [n_resets, 1, n_cubes(, 3)]
        #     target_pos.unsqueeze(1), self.p_cubes[env_idx], self.lwh_cubes[env_idx], self.rpy_cubes[env_idx])
        # nearest_dist2target = torch.cat([nearest_dist2target_sphere, nearest_dist2target_cube], dim=-1).squeeze(1) # [n_resets, n_obstacles]
        # nearest_point2target = torch.cat([nearest_point2target_sphere, nearest_point2target_cube], dim=-2).squeeze(1) # [n_resets, n_obstacles, 3]
        # tooclose2target = nearest_dist2target.lt(self.safety_range) # [n_resets, n_obstacles]
        
        # if torch.any(tooclose2drone):
        #     idx = tooclose2drone.nonzero(as_tuple=True)
        #     obstacle_pos[idx] += F.normalize(nearest_point2drone - drone_init_pos.unsqueeze(1), dim=-1)[idx] * self.safety_range
        # if torch.any(tooclose2target):
        #     idx = tooclose2target.nonzero(as_tuple=True)
        #     obstacle_pos[idx] += F.normalize(nearest_point2target - target_pos.unsqueeze(1), dim=-1)[idx] * self.safety_range

        obstacle_pos = drone_init_pos.unsqueeze(1) + relpos2drone # [n_resets, n_obstacles, 3]
        self.p_obstacles[env_idx] = obstacle_pos
        
        # assert torch.all(torch.norm(self.p_obstacles[env_idx] - target_pos.unsqueeze(1), dim=-1) >= safety_range)
        # assert torch.all(torch.norm(self.p_obstacles[env_idx] - drone_init_pos.unsqueeze(1), dim=-1) >= safety_range)
        
        L, H = env_spacing, self.height_scale * env_spacing
        if self.walls:
            zeros = torch.zeros_like(L)
            self.p_obstacles[env_idx, self.n_spheres:self.n_spheres+4] = torch.stack([
                torch.stack([zeros,  L, zeros], dim=-1),
                torch.stack([zeros, -L, zeros], dim=-1),
                torch.stack([ L, zeros, zeros], dim=-1),
                torch.stack([-L, zeros, zeros], dim=-1),
            ], dim=-2)
            self.rpy_cubes[env_idx, :4] = 0
        if self.ceiling:
            self.p_obstacles[env_idx, self.n_spheres+4*int(self.walls)] = torch.stack([zeros, zeros, H], dim=-1)
            self.rpy_cubes[env_idx, 4*int(self.walls)] = 0
        
        # # move disabled obstacles under the ground plane
        # if n_enabled_obstacles is not None:
        #     enabled_obstacles_idx = torch.randperm(self.n_obstacles, device=self.device)
        #     indices = torch.arange(self.n_obstacles, device=self.device).expand(n_resets, -1)
        #     mask = (indices >= n_enabled_obstacles.unsqueeze(-1))[:, enabled_obstacles_idx]
        #     mask = mask.unsqueeze(-1).expand(-1, -1, 3).clone()
        #     mask[:, :, :2] = False
        #     temp = self.p_obstacles[env_idx]
        #     temp[mask] = -100 * self.env_spacing
        #     self.p_obstacles[env_idx] = temp
        #     mask = ~mask
        # else:
        #     mask = torch.ones(self.n_obstacles, device=self.device, dtype=torch.bool)
        
        # return mask

    def randomize_obstacles(
        self,
        env_spacing: Tensor, # [n_envs]
        env_idx: Tensor, # [n_resets]
        drone_init_pos: Tensor, # [n_envs, 3]
        target_pos: Tensor, # [n_envs, 3]
        n_enabled_obstacles: Optional[Tensor] = None, # [n_envs]
    ):
        env_spacing = env_spacing[env_idx]
        drone_init_pos = drone_init_pos[env_idx]
        target_pos = target_pos[env_idx]
        n_enabled_obstacles = n_enabled_obstacles[env_idx] if n_enabled_obstacles is not None else None
        self.randomize_obstacles_sizes(env_spacing, env_idx)
        self.randomize_obstacles_poses(env_idx)
        self.randomize_obstacles_positions(env_spacing, env_idx, drone_init_pos, target_pos, n_enabled_obstacles)