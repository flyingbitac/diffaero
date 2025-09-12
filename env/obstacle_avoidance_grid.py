from typing import Tuple, Dict, Union, List
import math

from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn.functional as F
from tensordict import TensorDict, merge_tensordicts
import open3d as o3d
import numpy as np
from tqdm import tqdm

from diffaero.env.obstacle_avoidance import ObstacleAvoidance
from diffaero.dynamics.pointmass import point_mass_quat
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

# @timeit
@torch.jit.script
def get_visibility_map(
    p: Tensor,
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
    grid_centers_rel: Tensor,
    Rz: Tensor,
    grid_in_local_frame: bool
) -> Tuple[Tensor, Tensor]:
    N = contact_points.size(0)
    n_grid_points = n_x * n_y * n_z
    start = p.unsqueeze(1).expand_as(contact_points) # [n_envs, n_rays, 3]
    
    # 1. check if previous visible points are inside the current range of the grid
    # if so, mark them as visible and update their coordinate
    visible_map = torch.zeros_like(prev_visible_map) # [n_envs, n_points]
    curr_points = torch.zeros_like(prev_points)
    Rz_T = Rz.transpose(-1, -2)
    
    # 1.1 get the local coordinate of previous visible points
    prev_points_rel = prev_points - p.unsqueeze(1)
    if grid_in_local_frame:
        # world frame -> local frame
        prev_points_rel = torch.matmul(Rz_T, prev_points_rel.transpose(-1, -2)).transpose(-1, -2)
    # # 1.2 check if previous visible points are inside the current range of the grid
    valid_mask = check_if_valid(prev_points_rel, x_min, x_max, y_min, y_max, z_min, z_max)
    valid_and_visible = valid_mask & prev_visible_map
    env_ids, point_ids = torch.where(valid_and_visible)
    # points that are previously visible and still in the grid range currently
    prev_visible_points_local = prev_points_rel[valid_and_visible]
    linear_ids = get_linear_idx(prev_visible_points_local, x_min, x_max, y_min, y_max, z_min, z_max, n_x, n_y, n_z, cube_size)
    # # 1.3 mark the previous visible points as visible in the current occupancy map
    visible_map[env_ids, linear_ids] = True
    # 1.4 update the position of previous visible points by its current relative position
    curr_points[env_ids, linear_ids] = prev_points[env_ids, point_ids]
    
    # 2. get the current visible points
    # 2.1 sample visible points on the ray segments
    curr_visible_points = torch.lerp( # [n_envs, n_segments * n_rays, 3]
        input=start.unsqueeze(1),                           # [n_envs, 1, n_rays, 3]
        end=contact_points.unsqueeze(1),                    # [n_envs, 1, n_rays, 3]
        weight=ray_segment_weight                           # [1, n_segments, 1, 1]
    ).reshape(N, -1, 3)
    
    # 2.2 get the local coordinate of the current visible points
    curr_visible_points_rel = curr_visible_points - p.unsqueeze(1) # [n_envs, n_segments * n_rays, 3]
    if grid_in_local_frame:
        # world frame -> local frame
        curr_visible_points_rel = torch.matmul(Rz_T, curr_visible_points_rel.transpose(-1, -2)).transpose(-1, -2)
    valid_mask = check_if_valid(curr_visible_points_rel, x_min, x_max, y_min, y_max, z_min, z_max)
    env_ids, point_ids = torch.where(valid_mask) # [n_valid_points, ]
    local_valid_visible_points = curr_visible_points_rel[env_ids, point_ids] # [n_valid_points, 3]
    linear_ids = get_linear_idx(local_valid_visible_points, x_min, x_max, y_min, y_max, z_min, z_max, n_x, n_y, n_z, cube_size)
    
    # 3. mark the current visible points as visible in the map
    # flatten env and voxel index into one combined index
    combined = env_ids * n_grid_points + linear_ids # [n_valid_points, ]
    # collect one 3D point per visible voxel by picking the first ray‐segment hit (vectorized)
    # find unique combined indices in input order, get their first occurrence positions
    _, first_idx = torch.unique(combined, sorted=False, return_inverse=True) # [n_valid_points, ]
    first_idx_unique = torch.unique(first_idx, sorted=False) # [n_unique_points]
    # select corresponding env, voxel for each first hit
    env_sel = env_ids[first_idx_unique]    # [n_unique_points]
    lin_sel = linear_ids[first_idx_unique] # [n_unique_points]
    
    voxel_centers = p.unsqueeze(1) + grid_centers_rel # [n_envs, n_points, 3]
    # mark voxels where the first hit point actually falls inside the voxel
    visible_map[env_sel, lin_sel] = True
    # fill the visible points tensor with the global coordinate of the first hit points
    curr_points[env_sel, lin_sel] = voxel_centers[env_sel, lin_sel]
    
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
        self.ray_segment_weight = torch.linspace(0, 1, n_segments, device=self.device).reshape(1, n_segments, 1, 1)
        
        self.prev_visible_map = torch.zeros(self.n_envs, self.n_grid_points, dtype=torch.bool, device=self.device)
        Logger.debug(f"Space allocated for self.prev_visible_map: {self.prev_visible_map.dtype.itemsize * self.prev_visible_map.numel() / 1024 / 1024:.1f} MB")
        self.grid_points = torch.zeros(self.n_envs, self.n_grid_points, 3, device=self.device)
        Logger.debug(f"Space allocated for self.grid_points: {self.grid_points.dtype.itemsize * self.grid_points.numel() / 1024 / 1024:.1f} MB")
        self.contact_points = torch.zeros(self.n_envs, self.sensor.H * self.sensor.W, 3, device=self.device)
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=200, height=200, left=50, top=350, visible=True)
    
    def visualize_grid(self, grid, do_render=False):
        with tqdm.external_write_mode():
            xyz = self.local_grid_centers.squeeze(0)
            points = xyz[grid.flatten()].cpu().numpy()
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(points)
            # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_o3d, voxel_size=self.cube_size)

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
            grid_centers_rel = self.local_grid_centers
        else:
            grid_centers_rel = torch.matmul(self.dynamics.Rz, self.local_grid_centers.transpose(-1, -2)).transpose(-1, -2)
        grid_xyz = self.p.unsqueeze(1) + grid_centers_rel # [n_envs, n_points, 3]
        occupancy_map = self.obstacle_manager.are_points_inside_obstacles(grid_xyz) # [n_envs, n_points]
        if self.z_ground_plane is not None:
            occupancy_ground_plane = ((grid_xyz[..., 2] - self.r_drone) < self.z_ground_plane.unsqueeze(1))
            occupancy_map = torch.logical_or(occupancy_map, occupancy_ground_plane)
        return occupancy_map
    
    @timeit
    def get_visibility_map(self):
        if self.grid_frame == "world":
            grid_centers_rel = self.local_grid_centers
        else:
            grid_centers_rel = torch.matmul(self.dynamics.Rz, self.local_grid_centers.transpose(-1, -2)).transpose(-1, -2)
        visible_map, grid_points = get_visibility_map(
            # ray casting information
            self.p, self.contact_points, self.ray_segment_weight,
            # grid configurations
            self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max,
            self.cfg.grid.n_points[0], self.cfg.grid.n_points[1], self.cfg.grid.n_points[2],
            self.cube_size, self.prev_visible_map, self.grid_points, grid_centers_rel,
            # whether in local frame or in world frame
            self.dynamics.Rz, grid_in_local_frame=self.grid_frame=="local"
        )
        self.prev_visible_map.copy_(visible_map)
        self.grid_points.copy_(grid_points)
        return self.prev_visible_map.clone()
    
    @timeit
    def get_observations(self, with_grad=False):
        obs = super().get_observations(with_grad=with_grad)
        occupancy, visibility = self.get_occupancy_map(), self.get_visibility_map()
        grid_info = TensorDict({
            "occupancy": occupancy, "visibility": visibility}, batch_size=self.n_envs)
        obs = merge_tensordicts(obs, grid_info)
        return obs
    
    @timeit
    def reset_idx(self, env_idx):
        super().reset_idx(env_idx)
        self.prev_visible_map[env_idx] = False
        self.grid_points[env_idx] = 0.


class ObstacleAvoidanceGridYOPO(ObstacleAvoidanceGrid):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        self.obs_dim = (13, (self.sensor.H, self.sensor.W))
        self.state_dim = 13 + self.n_obstacles * 3
    
    def get_observations(self):
        obs = torch.cat([
            self.world2body(self.target_vel),
            self.world2body(self.v),
            self.world2body(self.a),
            self.q
        ], dim=-1)
        obs = TensorDict(
            {
                "state": obs,
                "perception": self.sensor_tensor.clone(),
                "occupancy": self.get_occupancy_map(),
                "visibility": self.get_visibility_map()
            },
            batch_size=self.n_envs
        )
        return obs
    
    @timeit
    def get_state(
        self,
        _p: Tensor, # [n_envs, T, 3]
        _v: Tensor, # [n_envs, T, 3]
        _a: Tensor  # [n_envs, T, 3]
    ) -> Tensor:
        p, v, a = _p.detach(), _v.detach(), _a.detach()
        dist2obstacles, nearest_points2obstacles = self.obstacle_manager.nearest_distance_to_obstacles(p)
        state = torch.cat([
            (nearest_points2obstacles - p.unsqueeze(2)).flatten(start_dim=-2),
            self.target_pos.unsqueeze(1) - p,
            point_mass_quat(a, orientation=v),
            _v,
            _a,
        ], dim=-1)
        return state
    
    @timeit
    def step(self, action):
        # type: (Tensor) -> Tuple[TensorDict, Tuple[Tensor, Tensor], Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        terminated, truncated, success, avg_vel = super()._step(action)
        reset = terminated | truncated
        reset_indices = reset.nonzero().view(-1)
        goal_reward, differentiable_reward, loss_components, _, nearest_points2obstacles = self.reward_fn(self.p.unsqueeze(1), self.v.unsqueeze(1), self.a.unsqueeze(1))
        self.obstacle_nearest_points.copy_(nearest_points2obstacles.squeeze(1)) # [n_envs, n_obstacles, 3]
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
        return self.get_observations(), (goal_reward.squeeze(-1), differentiable_reward.squeeze(-1)), terminated, extra
    
    @timeit
    def reward_fn(
        self,
        _p: Tensor, # [n_envs, T, 3]
        _v: Tensor, # [n_envs, T, 3]
        _a: Tensor  # [n_envs, T, 3]
    ) -> Tuple[Tensor, Tensor, Dict[str, float], Tensor, Tensor]:
        p, v, a = _p.detach(), _v.detach(), _a.detach()
        target_relpos = self.target_pos.unsqueeze(1) - p # [n_envs, T, 3]
        target_dist = target_relpos.norm(dim=-1) # [n_envs, T]
        target_vel = target_relpos / torch.max(target_dist / self.max_vel.unsqueeze(-1), torch.ones_like(target_dist)).unsqueeze(-1)
        inflation = 0.2
        # calculate the nearest points on the obstacles to the drone
        dist2obstacles, nearest_points2obstacles = self.obstacle_manager.nearest_distance_to_obstacles(p) # [n_envs, T, n_obstacles(, 3)]
        obstacle_reldirection = F.normalize(nearest_points2obstacles - p.unsqueeze(-2), dim=-1) # [n_envs, T, n_obstacles, 3]

        dist2surface_inflated = (dist2obstacles - (self.r_drone + inflation)).clamp(min=0) # [n_envs, T, n_obstacles]
        dangerous_factor = dist2surface_inflated.neg().exp() # [n_envs, T, n_obstacles]
        # calculate the obstacle avoidance loss
        approaching_vel = torch.sum(obstacle_reldirection * _v.unsqueeze(-2), dim=-1) # [n_envs, T, n_obstacles]
        approaching = approaching_vel > 0 # [n_envs, T, n_obstacles]
        avoiding_vel = torch.norm(_v.unsqueeze(-2) - approaching_vel.detach().unsqueeze(-1) * obstacle_reldirection, dim=-1) # [n_envs, T, n_obstacles]
        approaching_penalty, most_dangerous = (torch.where(approaching, approaching_vel, 0.) * dangerous_factor).max(dim=-1) # [n_envs, T]
        avoiding_reward = torch.where(approaching, avoiding_vel, 0.) * dangerous_factor # [n_envs, T, n_obstacles]
        avoiding_reward = torch.gather(input=avoiding_reward, dim=-1, index=most_dangerous.unsqueeze(-1)).squeeze(-1) # [n_envs, T]
        oa_loss = approaching_penalty - 0.5 * avoiding_reward # [n_envs, T]
        
        collision = (dist2obstacles - self.r_drone).lt(0.).any(dim=-1) # [n_envs, T]
        collision_loss = collision.float()
        
        arrive_loss = 1 - target_dist.lt(0.5).float()
        pos_loss = 1 - target_dist.neg().exp()
        
        vel_diff = torch.norm(_v - target_vel, dim=-1) # [n_envs, T]
        vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none") # [n_envs, T]
        z_diff = _p[..., 2] - self.target_pos[:, None, 2] # [n_envs, T]
        z_loss = 1 - z_diff.abs().neg().exp()
        goal_reward = (
            self.reward_weights.constant - 
            self.reward_weights.goal.vel * vel_loss -
            self.reward_weights.goal.z * z_loss -
            self.reward_weights.goal.oa * oa_loss -
            self.reward_weights.goal.pos * pos_loss -
            self.reward_weights.goal.collision * collision_loss
        ).detach()
        differentiable_reward = (
            self.reward_weights.constant - 
            self.reward_weights.differentiable.vel * vel_loss -
            self.reward_weights.differentiable.z * z_loss -
            self.reward_weights.differentiable.oa * oa_loss -
            self.reward_weights.differentiable.pos * pos_loss -
            self.reward_weights.differentiable.collision * collision_loss
        )
        loss_components = {
            "vel_loss": vel_loss.mean().item(),
            "z_loss": z_loss.mean().item(),
            "pos_loss": pos_loss.mean().item(),
            "arrive_loss": arrive_loss.mean().item(),
            "collision_loss": collision_loss.mean().item(),
            "oa_loss": oa_loss.mean().item(),
            "goal_reward": goal_reward.mean().item(),
            "total_loss": -differentiable_reward.mean().item(),
            "differentiable_reward": differentiable_reward.mean().item(),
            "goal_reward": goal_reward.mean().item(),
        }
        return goal_reward, differentiable_reward, loss_components, collision, nearest_points2obstacles
