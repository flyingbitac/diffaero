import os
from typing import Optional, Tuple, List

import numpy as np
from omegaconf import DictConfig
import torch
from torch.nn import functional as F

from quaddif import QUADDIF_ROOT_DIR
from quaddif.utils.math import quat_from_euler_xyz


class ObstacleManager:
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.n_envs: int = cfg.n_envs
        self.env_spacing: float = cfg.length
        self.n_obstacles: int = cfg.n_obstacles
        self.n_spheres: int = self.n_obstacles // 2
        self.n_cubes: int = self.n_obstacles - self.n_spheres
        self.device = device
        
        self.r_obstacles = torch.empty(self.n_envs, self.n_obstacles, device=self.device)
        self.p_obstacles = torch.zeros(self.n_envs, self.n_obstacles, 3, device=device)
        self.lwh_cubes = torch.empty(self.n_envs, self.n_cubes, 3, device=device)
        
    @property
    def p_spheres(self): return self.p_obstacles[:, :self.n_spheres]
    @property
    def r_spheres(self): return self.r_obstacles[:, :self.n_spheres]
    @property
    def p_cubes(self): return self.p_obstacles[:, self.n_spheres:]
        
    def generate_obstacles(self, r_min=0.2, r_max=2.0, r_step=0.2):
        # type: (float, float, float) -> torch.Tensor
        # randomly generate spheral obstacles
        radius = [r_min + i * r_step for i in range(int((r_max - r_min) / r_step))]
        selected_radius = list(np.random.choice(radius, size=(self.n_envs*self.n_spheres), replace=True))
        self.r_obstacles[:, :self.n_spheres] = torch.tensor(selected_radius, device=self.device).reshape(self.n_envs, self.n_spheres)
        # randomly generate cubical obstacles
        lwh_min, lwh_max, lwh_step = 0.4, 1.6, 0.2
        lwh = [lwh_min + i * lwh_step for i in range(int((lwh_max - lwh_min) / r_step))]
        selected_lwh = list(np.random.choice(lwh, size=(self.n_envs*self.n_cubes*3), replace=True))
        self.lwh_cubes.copy_(torch.tensor(selected_lwh, device=self.device).reshape(self.n_envs, self.n_cubes, 3))
        self.r_obstacles[:, self.n_spheres:] = self.lwh_cubes.pow(2).sum(dim=-1).sqrt()
        return self.r_obstacles
    
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
        
        max_std = 1.
        min_std = 1.
        std = 4.
        
        # sample uniformally along the target axis
        target_axis_ratio = torch.rand(
            len(env_idx), self.n_obstacles, 1, device=self.device) * 1.4 - 0.2
        target_axis_pos = target_axis_ratio * rel_pos.unsqueeze(1)
        
        # whether the sampled point is too close to the drone's initial position
        # or to the target position
        violations = (
            (target_axis_pos.norm(dim=-1) < safety_range) |
            ((target_axis_pos - rel_pos.unsqueeze(1)).norm(dim=-1) < safety_range))
        
        # sample from gaussian distribution
        std = (torch.abs(target_axis_ratio - 0.5) / 0.7) * (max_std - min_std) + min_std
        horizontal_axis_ratio = torch.randn(
            len(env_idx), self.n_obstacles, 1, device=self.device) * std
        third_axis_ratio = torch.randn(
            len(env_idx), self.n_obstacles, 1, device=self.device) * std
        
        # move obstacles around the drone's initial position and target a little bit further
        env_index, asset_index = violations.nonzero(as_tuple=True)
        horizontal_axis_ratio[env_index, asset_index] += torch.sign(horizontal_axis_ratio[env_index, asset_index]) * safety_range
        third_axis_ratio[env_index, asset_index] += torch.sign(third_axis_ratio[env_index, asset_index]) * safety_range
        
        horizontal_axis_pos = horizontal_axis_ratio * horizontal_axis.unsqueeze(1)
        third_axis_pos = third_axis_ratio * third_axis.unsqueeze(1)
        
        self.p_obstacles[env_idx] = (
            drone_init_pos.unsqueeze(1) +
            target_axis_pos +
            horizontal_axis_pos +
            third_axis_pos).clamp(-self.env_spacing, self.env_spacing)
        
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


def create_cube(x: float, y: float, z: float) -> str:
    xml = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        '<robot name="box">',
        '  <link name="base_link">',
        '    <inertial>',
        '      <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>',
        '      <mass value="1.0"/>',
        '      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>',
        '    </inertial>',
        '    <visual name="base_link_visual">',
        '      <geometry>',
        f'        <box size="{x} {y} {z}"/>',
        '      </geometry>',
        '      <origin xyz="0 0 0" rpy="0 0 0"/>',
        '    </visual>',
        '    <collision name="base_link_collision">',
        '      <geometry>',
        f'        <box size="{x} {y} {z}"/>',
        '      </geometry>',    
        '      <origin xyz="0 0 0" rpy="0 0 0"/>',
        '    </collision>',
        '  </link>',
        '</robot>']
    file = f"cube_{x:.1f}__{y:.1f}__{z:.1f}".replace(".", "_") + ".urdf"
    root = os.path.join(QUADDIF_ROOT_DIR, 'resources', 'environment_assets', 'cubes')
    if not os.path.exists(root):
        os.makedirs(root)
    path = os.path.join(root, file)
    with open(path, 'w') as f:
        for line in xml:
            f.write(line+"\n")
    return path
    
def create_ball(r: float) -> str:
    xml = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        '<robot name="sphere">',
        '  <link name="base_link">',
        '    <inertial>',
        '      <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>',
        '      <mass value="1.0"/>',
        '      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>',
        '    </inertial>',
        '    <visual name="base_link_visual">',
        '      <geometry>',
        f'        <sphere radius="{float(r):.1f}"/>',
        '      </geometry>',
        '      <origin xyz="0 0 0" rpy="0 0 0"/>',
        '    </visual>',
        '    <collision name="base_link_collision">',
        '      <geometry>',
        f'        <sphere radius="{float(r):.1f}"/>',
        '      </geometry>',    
        '      <origin xyz="0 0 0" rpy="0 0 0"/>',
        '    </collision>',
        '  </link>',
        '</robot>']
    file = f"sphere_{r:.2f}".replace(".", "_") + ".urdf"
    root = os.path.join(QUADDIF_ROOT_DIR, 'resources', 'environment_assets', 'spheres')
    if not os.path.exists(root):
        os.makedirs(root)
    path = os.path.join(root, file)
    if not os.path.exists(path):    
        with open(path, 'w') as f:
            for line in xml:
                f.write(line+"\n")
    return path