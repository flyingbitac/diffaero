from typing import Tuple, Dict, Union, List
import math

from omegaconf import DictConfig
import torch
from torch import Tensor
from tensordict import TensorDict, merge_tensordicts
import open3d as o3d
import numpy as np

from diffaero.env.obstacle_avoidance import ObstacleAvoidance
from diffaero.utils.sensor import RayCastingSensorBase
from diffaero.utils.runner import timeit
from diffaero.utils.logger import Logger

@torch.jit.script
def check_if_valid(
    point_list: Tensor,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float
) -> Tensor:
    return (
        (point_list[..., 0] >= x_min) & (point_list[..., 0] < x_max) &
        (point_list[..., 1] >= y_min) & (point_list[..., 1] < y_max) &
        (point_list[..., 2] >= z_min) & (point_list[..., 2] < z_max)
    )

@torch.jit.script
def get_linear_idx(
    point_list: Tensor,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    n_x: int,
    n_y: int,
    n_z: int,
    cube_size: float
) -> Tensor:
    x_idx = ((point_list[..., 0] - x_min).clamp(max=x_max-x_min-1e-5) / cube_size).long()
    y_idx = ((point_list[..., 1] - y_min).clamp(max=y_max-y_min-1e-5) / cube_size).long()
    z_idx = ((point_list[..., 2] - z_min).clamp(max=z_max-z_min-1e-5) / cube_size).long()
    return x_idx * (n_y * n_z) + y_idx * n_z + z_idx

@torch.jit.script
def get_visibility_map(
    p: Tensor,
    sensor_tensor: Tensor,
    contact_points: Tensor,
    ray_segment_weight: Tensor,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    n_x: int,
    n_y: int,
    n_z: int,
    cube_size: float,
    prev_visible_map: Tensor,
    prev_points: Tensor,
    grid_centers_local: Tensor,
    Rz: Tensor,
    grid_in_local_frame: bool
) -> Tuple[Tensor, Tensor]:
    # get visiability map
    N, H, W = sensor_tensor.shape
    n_grid_points = n_x * n_y * n_z
    start = p.unsqueeze(1).expand(-1, H*W, -1) # [n_envs, n_rays, 3]
    
    # 1. check if previous visible points are inside the current range of the grid
    # if so, mark them as visible and update their coordinate
    curr_visible_map_prev = torch.zeros_like(prev_visible_map) # [n_envs, n_points]
    
    # 1.1 get the local coordinate of previous visible points
    prev_points_rel = prev_points - p.unsqueeze(1)
    if grid_in_local_frame:
        # world frame -> local frame
        prev_points_rel = torch.matmul(Rz.transpose(-1, -2), prev_points_rel.transpose(-1, -2)).transpose(-1, -2)
    env_ids, prev_point_ids = torch.where(prev_visible_map)
    prev_visible_points_local = prev_points_rel[env_ids, prev_point_ids]
    # 1.2 check if previous visible points are inside the current range of the grid
    valid_mask = check_if_valid(prev_visible_points_local, x_min, x_max, y_min, y_max, z_min, z_max)
    # 1.3 mark the previous visible points as visible in the current occupancy map
    valid_env_ids = env_ids[valid_mask]
    valid_prev_id = prev_point_ids[valid_mask] # indices in previous local frame for points that are visible in previous timestep
    valid_current_local = prev_visible_points_local[valid_mask]
    linear_ids = get_linear_idx(valid_current_local, x_min, x_max, y_min, y_max, z_min, z_max, n_x, n_y, n_z, cube_size)
    curr_visible_map_prev[valid_env_ids, linear_ids] = True
    # 1.4 update the position of previous visible points by its current relative position
    curr_points = torch.zeros_like(prev_points)
    curr_points[valid_env_ids, linear_ids] = prev_points[valid_env_ids, valid_prev_id]
    
    # 2. get the current visible points
    # 2.1 sample visible points on the ray segments
    curr_visible_points = torch.lerp( # [n_envs, n_segments * n_rays, 3]
        input=start.unsqueeze(1),                           # [n_envs, 1, n_rays, 3]
        end=contact_points.unsqueeze(1),                    # [n_envs, 1, n_rays, 3]
        weight=ray_segment_weight.reshape(1, -1, 1, 1)      # [1, n_segments, 1, 1]
    ).reshape(N, -1, 3)
    
    # 2.2 get the local coordinate of the current visible points
    curr_visible_points_rel = curr_visible_points - p.unsqueeze(1) # [n_envs, n_segments * n_rays, 3]
    if grid_in_local_frame:
        # world frame -> local frame
        curr_visible_points_rel = torch.matmul(Rz.transpose(-1, -2), curr_visible_points_rel.transpose(-1, -2)).transpose(-1, -2)
    valid_mask = check_if_valid(curr_visible_points_rel, x_min, x_max, y_min, y_max, z_min, z_max)
    env_ids, prev_point_ids = torch.where(valid_mask) # [n_valid_points, ]
    local_valid_visible_points = curr_visible_points_rel[env_ids, prev_point_ids] # [n_valid_points, 3]
    linear_ids = get_linear_idx(local_valid_visible_points, x_min, x_max, y_min, y_max, z_min, z_max, n_x, n_y, n_z, cube_size)
    
    # 3. mark the current visible points as visible in the occupancy map
    # flatten env and voxel index into one combined index
    combined = env_ids * n_grid_points + linear_ids # [n_valid_points, ]
    # collect one 3D point per visible voxel by picking the first ray‐segment hit (vectorized)
    # find unique combined indices in input order, get their first occurrence positions
    _, first_idx = torch.unique(combined, sorted=False, return_inverse=True) # [n_valid_points, ]
    first_idx_unique = torch.unique(first_idx, sorted=False) # [n_unique_points]
    # select corresponding env, voxel for each first hit
    env_sel = env_ids[first_idx_unique]    # [n_unique_points]
    lin_sel = linear_ids[first_idx_unique] # [n_unique_points]
    
    voxel_centers = p.unsqueeze(1) + grid_centers_local # [n_envs, n_points, 3]
    # fill the visible points tensor with the global coordinate of the first hit points
    curr_points[env_sel, lin_sel] = voxel_centers[env_sel, lin_sel]
    diff = curr_points - voxel_centers # [n_envs, n_points, 3]
    # mark voxels where the first hit point actually falls inside the voxel
    visible_map = (diff.abs() <= (cube_size / 2)).all(dim=-1)  # [n_envs, n_points]
    
    # current visible map should also include points that are visible in the previous steps
    visible_map |= curr_visible_map_prev
    
    return visible_map, curr_points # [n_envs, n_points], [n_envs, n_points, 3]


class ObstacleAvoidanceGrid(ObstacleAvoidance):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        assert cfg.grid.name != "none", "ObstacleAvoidanceGrid requires a grid configuration."
        assert isinstance(self.sensor, RayCastingSensorBase), "This environment only supports ray casting-based sensors."
        self.n_grid_points = math.prod(cfg.grid.n_points)
        self.grid_frame = cfg.grid.frame
        assert self.grid_frame in ["local", "world"], "Grid frame must be either 'local' or 'world'."
        self.x_min, self.x_max = self.cfg.grid.x_min, self.cfg.grid.x_max
        self.y_min, self.y_max = self.cfg.grid.y_min, self.cfg.grid.y_max
        self.z_min, self.z_max = self.cfg.grid.z_min, self.cfg.grid.z_max
        xyz_cube_size = (
            ((self.x_max - self.x_min) / cfg.grid.n_points[0]),
            ((self.y_max - self.y_min) / cfg.grid.n_points[1]),
            ((self.z_max - self.z_min) / cfg.grid.n_points[2])
        )
        assert min(xyz_cube_size) == max(xyz_cube_size), "Grid cube size must be equal in all dimensions."
        self.cube_size = min(xyz_cube_size)

        x_range = torch.linspace(self.x_min, self.x_max - self.cube_size, cfg.grid.n_points[0], device=self.device) + self.cube_size / 2
        y_range = torch.linspace(self.y_min, self.y_max - self.cube_size, cfg.grid.n_points[1], device=self.device) + self.cube_size / 2
        z_range = torch.linspace(self.z_min, self.z_max - self.cube_size, cfg.grid.n_points[2], device=self.device) + self.cube_size / 2
        grid_xyz_range = torch.stack(torch.meshgrid(x_range, y_range, z_range, indexing="ij"), dim=-1) # [x_points, y_points, z_points, 3]
        self.local_grid_centers = grid_xyz_range.reshape(1, -1, 3) # [1, n_points, 3]
        n_segments = math.ceil(self.sensor.max_dist / self.cube_size)
        self.ray_segment_weight = torch.linspace(0, 1, n_segments, device=self.device)
        
        self.prev_visible_map = torch.zeros(self.n_envs, self.n_grid_points, dtype=torch.bool, device=self.device)
        self.grid_points = torch.zeros(self.n_envs, self.n_grid_points, 3, device=self.device)
        self.contact_points = torch.zeros(self.n_envs, self.sensor.H * self.sensor.W, 3, device=self.device)
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=200, height=200, left=50, top=350, visible=True)
    
    def visualize_grid(self, grid, do_render=False):
        xyz = self.local_grid_centers.squeeze(0)
        points = xyz[grid.flatten()].cpu().numpy()
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([pcd_o3d])

        self.vis.clear_geometries()
        self.vis.add_geometry(pcd_o3d)
        self.vis.poll_events()
        self.vis.update_renderer()
        
        return np.asarray(self.vis.capture_screen_float_buffer(do_render=do_render))

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
        self.contact_points.copy_(contact_points)
    
    @timeit
    def get_occupancy_map(self): 
        # get occupancy map
        if self.grid_frame == "world":
            local_grid_centers = self.local_grid_centers
        else:
            local_grid_centers = torch.matmul(self.dynamics.Rz, self.local_grid_centers.transpose(-1, -2)).transpose(-1, -2)
        grid_xyz = self.p.unsqueeze(1) + local_grid_centers # [n_envs, n_points, 3]
        occupancy_map = self.obstacle_manager.are_points_inside_obstacles(grid_xyz) # [n_envs, n_points]
        if self.z_ground_plane is not None:
            occupancy_ground_plane = ((grid_xyz[..., 2] - self.r_drone) < self.z_ground_plane.unsqueeze(1))
            occupancy_map = torch.logical_or(occupancy_map, occupancy_ground_plane)
        return occupancy_map
    
    @timeit
    def get_visibility_map(self):
        visible_map, grid_points = get_visibility_map(
            self.p, self.sensor_tensor, self.contact_points, self.ray_segment_weight,
            self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max,
            self.cfg.grid.n_points[0], self.cfg.grid.n_points[1], self.cfg.grid.n_points[2],
            self.cube_size, self.prev_visible_map, self.grid_points, self.local_grid_centers,
            self.dynamics.Rz, grid_in_local_frame=self.grid_frame=="local"
        )
        self.prev_visible_map.copy_(visible_map)
        self.grid_points.copy_(grid_points)
        return self.prev_visible_map.clone()
    
    @timeit
    def get_observations(self, with_grad=False):
        obs = super().get_observations(with_grad=with_grad)
        grid, visible_map = self.get_occupancy_map(), self.get_visibility_map()
        grid_info = TensorDict({
            "grid": grid, "visible_map": visible_map}, batch_size=self.n_envs)
        obs = merge_tensordicts(obs, grid_info)
        # if self.renderer is not None:
        #     grid_tobe_visualized = visible_map
        #     self.visualize_grid(grid_tobe_visualized[self.renderer.gui_states["tracking_env_idx"]])
        # grid_tobe_visualized = grid
        # self.visualize_grid(grid_tobe_visualized[0], do_render=True)
        return obs
    
    @timeit
    def reset_idx(self, env_idx):
        super().reset_idx(env_idx)
        self.prev_visible_map[env_idx] = False
        self.grid_points[env_idx] = 0.