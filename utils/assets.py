from typing import Optional, Tuple, List

import numpy as np
from omegaconf import DictConfig
import torch
from torch.nn import functional as F

class ObstacleManager:
    def __init__(self, cfg: DictConfig, n_envs: int, env_spacing: float, device: torch.device):
        self.cfg = cfg
        self.n_envs = n_envs
        self.env_spacing = env_spacing
        self.obst_cfg = cfg
        self.n_obstacles: int = self.obst_cfg.n_obstacles
        self.n_spheres: int = int(self.n_obstacles * self.obst_cfg.sphere_percentage)
        self.n_cubes: int = self.n_obstacles - self.n_spheres
        self.sphere_rmin, self.sphere_rmax, self.sphere_rstep = list(self.obst_cfg.sphere_radius_range)
        # lwh for Length(along x axis), Width(along y axis) and Height(along z axis)
        self.cube_lwhmin, self.cube_lwhmax, self.cube_lwhstep = list(self.obst_cfg.cube_lwh_range)
        self.randpos_minstd: float = self.obst_cfg.randpos_std_min
        self.randpos_maxstd: float = self.obst_cfg.randpos_std_max
        self.device = device
        
        self.r_obstacles = torch.empty(self.n_envs, self.n_obstacles, device=self.device)
        self.p_obstacles = torch.zeros(self.n_envs, self.n_obstacles, 3, device=device)
        self.lwh_cubes = torch.empty(self.n_envs, self.n_cubes, 3, device=device)
        self.box_min = torch.zeros(self.n_envs, self.n_cubes, 3, device=device)
        self.box_max = torch.zeros(self.n_envs, self.n_cubes, 3, device=device)
        self.generate_obstacles()
        
    @property
    def p_spheres(self): return self.p_obstacles[:, :self.n_spheres]
    @property
    def r_spheres(self): return self.r_obstacles[:, :self.n_spheres]
    @property
    def p_cubes(self): return self.p_obstacles[:, self.n_spheres:]
        
    def generate_obstacles(self):
        # randomly generate spheral obstacles
        radius = np.arange(self.sphere_rmin, self.sphere_rmax, self.sphere_rstep)
        selected_radius = np.random.choice(radius, size=(self.n_envs, self.n_spheres), replace=True)
        self.r_obstacles[:, :self.n_spheres] = torch.from_numpy(selected_radius).to(self.device)
        # randomly generate cubical obstacles
        lwh = np.arange(self.cube_lwhmin, self.cube_lwhmax, self.cube_lwhstep)
        selected_lwh = np.random.choice(lwh, size=(self.n_envs, self.n_cubes, 3), replace=True)
        self.lwh_cubes.copy_(torch.from_numpy(selected_lwh).to(self.device))
        self.r_obstacles[:, self.n_spheres:] = self.lwh_cubes.pow(2).sum(dim=-1).sqrt()
    
    def randomize_asset_pose(
        self,
        env_idx: torch.Tensor, # [num_resets]
        drone_init_pos: torch.Tensor, # [num_resets, 3]
        target_pos: torch.Tensor, # [num_resets, 3]
        n_enabled_obstacles: Optional[torch.Tensor] = None, # [num_resets]
        safety_range: float = 0.5
    ) -> torch.Tensor:
        if self.n_obstacles == 0:
            return self.p_obstacles[env_idx]
        
        safety_range: torch.Tensor = self.r_obstacles[env_idx] + safety_range # [n_resets, n_obstacles]
        
        rel_pos = target_pos - drone_init_pos
        # target_axis: unit vector in the direction of the target's relative position
        target_axis = F.normalize(rel_pos, dim=-1)
        # horizontal_axis: unit vector in the horizontal plane, perpendicular to the target_axis
        horizontal_axis = F.normalize(torch.stack([
            -rel_pos[:, 1],
            rel_pos[:, 0],
            torch.zeros(len(env_idx), device=self.device)], dim=-1), dim=-1)
        # third_axis: unit vector perpendicular to two other vectors
        third_axis = torch.cross(target_axis, horizontal_axis, dim=-1)
        
        # sample uniformally along the target axis
        target_axis_ratio = torch.rand(len(env_idx), self.n_obstacles, 1, device=self.device)
        target_axis_pos = target_axis_ratio * rel_pos.unsqueeze(1)
        
        # sample from gaussian distribution
        std = torch.abs(target_axis_ratio - 0.5) * 2 * (self.randpos_maxstd - self.randpos_minstd) + self.randpos_minstd
        horizontal_axis_ratio = torch.randn(
            len(env_idx), self.n_obstacles, 1, device=self.device) * std
        third_axis_ratio = torch.randn(
            len(env_idx), self.n_obstacles, 1, device=self.device) * std
        
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
            relpos2drone[idx] += F.normalize(relpos2target_axis[idx], dim=-1) * safety_range[idx].unsqueeze(-1)
            assert torch.all(torch.logical_and(relpos2drone.norm(dim=-1) >= safety_range, (relpos2drone - rel_pos.unsqueeze(1)).norm(dim=-1) >= safety_range))
        
        self.p_obstacles[env_idx] = drone_init_pos.unsqueeze(1) + relpos2drone
        
        # assert torch.all(torch.norm(self.p_obstacles[env_idx] - target_pos.unsqueeze(1), dim=-1) >= safety_range)
        # assert torch.all(torch.norm(self.p_obstacles[env_idx] - drone_init_pos.unsqueeze(1), dim=-1) >= safety_range)
        
        self.box_min[env_idx] = self.p_cubes[env_idx] - self.lwh_cubes[env_idx] / 2.
        self.box_max[env_idx] = self.p_cubes[env_idx] + self.lwh_cubes[env_idx] / 2.
        
        # move disabled obstacles under the ground plane
        if n_enabled_obstacles is not None:
            enabled_obstacles_idx = torch.randperm(self.n_obstacles, device=self.device)
            indices = torch.arange(self.n_obstacles, device=self.device).expand(len(env_idx), -1)
            mask = (indices >= n_enabled_obstacles.unsqueeze(-1))[:, enabled_obstacles_idx]
            mask = mask.unsqueeze(-1).expand(-1, -1, 3).clone()
            mask[:, :, :2] = False
            temp = self.p_obstacles[env_idx]
            temp[mask] = -100 * self.env_spacing
            self.p_obstacles[env_idx] = temp
            mask = ~mask
        else:
            mask = torch.ones(self.n_obstacles, device=self.device, dtype=torch.bool)
        
        return mask
