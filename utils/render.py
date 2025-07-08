from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch import Tensor
from pytorch3d import transforms as T
import taichi as ti
from omegaconf import DictConfig

from quaddif.utils.assets import ObstacleManager
from quaddif.utils.math import quaternion_to_euler, axis_rotmat
from quaddif.utils.logger import Logger

@torch.jit.script
def torch2ti(tensor_from_torch: Tensor):
    # assert tensor_from_torch.size(-1) == 3
    x, y, z = tensor_from_torch.unbind(dim=-1)
    return torch.stack([x, z, -y], dim=-1)

@torch.jit.script
def ti2torch(tensor_from_ti: Tensor):
    # assert tensor_from_ti.size(-1) == 3
    x, y, z = tensor_from_ti.unbind(dim=-1)
    return torch.stack([x, -z, y], dim=-1)

faces = torch.tensor([
    [0, 1, 2, 3],  # front
    [4, 5, 6, 7],  # back
    [0, 1, 5, 4],  # bottom
    [2, 3, 7, 6],  # top
    [1, 2, 6, 5],  # right
    [0, 3, 7, 4],  # left
], dtype = torch.int32)

INDICES_TORCH = torch.stack([
    faces[:, [0, 1, 2]],
    faces[:, [2, 3, 0]],
], dim=-2).flatten()

def add_box(xyz: Tensor, lwh: Tensor, rpy: Tensor, color: Tensor):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
    l, w, h = lwh.unbind(dim=-1)
    n_boxes = torch.cumprod(torch.tensor(xyz.shape[:-1]), dim=0)[-1].item()
    n_boxes = xyz[..., 0].numel()
    centered_vertices_tensor = torch.stack([
        torch.stack([-l/2, -w/2, -h/2], dim=-1),
        torch.stack([+l/2, -w/2, -h/2], dim=-1),
        torch.stack([+l/2, +w/2, -h/2], dim=-1),
        torch.stack([-l/2, +w/2, -h/2], dim=-1),
        torch.stack([-l/2, -w/2, +h/2], dim=-1),
        torch.stack([+l/2, -w/2, +h/2], dim=-1),
        torch.stack([+l/2, +w/2, +h/2], dim=-1),
        torch.stack([-l/2, +w/2, +h/2], dim=-1),
    ], dim=-2)
    rotation_matrix = T.euler_angles_to_matrix(rpy, "XYZ")
    rotated_centered_vertices_tensor = torch.matmul(centered_vertices_tensor, rotation_matrix.transpose(-2, -1))
    rotated_vertices_tensor = rotated_centered_vertices_tensor + xyz.unsqueeze(-2)
    indices_torch = INDICES_TORCH.clone()
    while indices_torch.dim() < rotated_vertices_tensor.dim() - 1:
        indices_torch.unsqueeze_(0)
    indices_torch = indices_torch.expand(*rotated_vertices_tensor.shape[:-2], -1)
    indices_torch = indices_torch + torch.arange(0, 8 * n_boxes, 8, dtype=torch.int32).reshape(*xyz.shape[:-1], 1)
    while color.dim() < rotated_vertices_tensor.dim() - 1:
        color = color.unsqueeze(0)
    color = color.unsqueeze(-2).expand_as(rotated_vertices_tensor)
    return rotated_vertices_tensor, indices_torch, color

def add_sphere(xyz: Tensor, radius: Tensor, color: Tensor, lat_segments: int = 8, lon_segments: int = 16):
    # type: (Tensor, Tensor, Tensor, int, int) -> Tuple[Tensor, Tensor, Tensor]
    # xyz: (..., 3), radius: (...), color: (..., 3)
    # 生成单位球体顶点（单个球）
    device = xyz.device
    pi = torch.pi
    # 顶点数量：北极+南极+(纬度-1)*经度数
    n_int = (lat_segments - 1) * lon_segments
    n_v = 2 + n_int

    # 生成中间纬度角 theta（去除两极）
    theta = torch.linspace(0, pi, steps=lat_segments + 1, device=device)[1:-1]  # (lat_segments-1,)
    phi = torch.linspace(0, 2*pi, steps=lon_segments + 1, device=device)[:-1]  # (lon_segments,)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")  # shape: (lat_segments-1, lon_segments)

    # 计算中间顶点，采用球面坐标：(x,y,z) = (sinθ*cosφ, sinθ*sinφ, cosθ)
    inter_x = torch.sin(theta_grid) * torch.cos(phi_grid)  # (lat_segments-1, lon_segments)
    inter_y = torch.sin(theta_grid) * torch.sin(phi_grid)
    inter_z = torch.cos(theta_grid)
    # 将中间顶点展平，形状 (n_int, 3)
    inter_vertices = torch.stack([inter_x, inter_y, inter_z], dim=-1).reshape(-1, 3)

    # 北极与南极顶点
    north_vertex = torch.tensor([[0.0, 0.0, 1.0]], device=device)  # shape (1,3)
    south_vertex = torch.tensor([[0.0, 0.0, -1.0]], device=device)  # shape (1,3)

    # 单个球的单位顶点集合，形状 (n_v, 3)
    unit_vertices = torch.cat([north_vertex, inter_vertices, south_vertex], dim=0)

    # 生成 indices（针对单个球）
    indices_list = []
    # 顶点编号：北极：0, 中间：1 .. 1+n_int-1, 南极： n_v-1
    # 顶部三角形：连接北极与第一排中间顶点
    for j in range(lon_segments):
        next_j = (j + 1) % lon_segments
        indices_list.append([0, 1 + j, 1 + next_j])
    # 中间区域，i 从 0 到 (lat_segments-2)-1，即 i=0,...,lat_segments-2
    for i in range(lat_segments - 2):
        for j in range(lon_segments):
            next_j = (j + 1) % lon_segments
            v1 = 1 + i * lon_segments + j
            v2 = 1 + (i + 1) * lon_segments + j
            v3 = 1 + i * lon_segments + next_j
            v4 = 1 + (i + 1) * lon_segments + next_j
            indices_list.append([v1, v2, v3])
            indices_list.append([v3, v2, v4])
    # 底部三角形：连接最后一排中间顶点与南极点
    for j in range(lon_segments):
        next_j = (j + 1) % lon_segments
        vj = 1 + (lat_segments - 2) * lon_segments + j
        vnj = 1 + (lat_segments - 2) * lon_segments + next_j
        indices_list.append([n_v - 1, vj, vnj])
    
    n_triangles = len(indices_list)
    
    indices_single = torch.tensor(indices_list, dtype=torch.int32, device=device).flatten()  # shape (num_indices,)

    # 处理批量：令 batch_size = xyz[...,0].numel()
    batch_shape = xyz.shape[:-1]
    batch_size = xyz.numel() // 3

    # 扩展 unit_vertices 至 (batch_size, n_v, 3)
    unit_vertices = unit_vertices.unsqueeze(0).expand(batch_size, -1, -1)  # (B, n_v, 3)
    # 调整 radius 和 xyz，形状：(B, 1)
    radius = radius.reshape(batch_size, 1)
    xyz = xyz.reshape(batch_size, 3)
    # 计算所有球体顶点
    vertices = unit_vertices * radius.unsqueeze(-1) + xyz.unsqueeze(1)  # (B, n_v, 3)

    # 生成颜色：复制每个顶点的颜色，颜色形状 (B, n_v, 3)
    colors = color.reshape(batch_size, 1, 3).expand(-1, n_v, -1)

    # 生成 indices 批量: 对于每个球体需要加上偏移量 k * n_v
    indices_single = indices_single.unsqueeze(0)  # (1, num_indices)
    indices = indices_single + (torch.arange(batch_size, device=device, dtype=torch.int32).unsqueeze(1) * n_v)
    return vertices.reshape(*batch_shape, n_v, 3), indices.reshape(*batch_shape, n_triangles * 3), colors.reshape(*batch_shape, n_v, 3)

@ti.data_oriented
class BaseRenderer:
    def __init__(self, cfg: DictConfig, device: torch.device, height_scale: Optional[float] = None, headless: bool = False):
        self.n_envs: int = min(cfg.n_envs, cfg.render_n_envs)
        self.n_agents: int = cfg.n_agents
        self.L: int = cfg.env_spacing
        self.dt: float = cfg.dt
        self.ground_plane: bool = cfg.ground_plane
        self.height_scale = height_scale if height_scale is not None else 1.
        self.record_video: bool = cfg.record_video
        self.enable_rendering: bool = True
        self.headless = headless
        self.device = device
        
        if self.record_video:
            assert str(self.device) in ["cpu", "cuda:0"], "Video recording is only supported on cpu and cuda:0."
        
        if "cpu" in str(self.device):
            Logger.info("Using CPU to render GUI.")
            ti.init(arch=ti.cpu)
        else:
            Logger.info("Using GPU to render GUI.")
            ti.init(arch=ti.gpu)
        
        N = torch.ceil(torch.sqrt(torch.tensor(self.n_envs, device=self.device))).int()
        assert N * N >= self.n_envs
        x = y = torch.arange(N, device=self.device, dtype=torch.float32) * self.L * 2
        xy = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1).reshape(-1, 2)
        xy -= (N-1) * self.L
        xyz = torch.cat([xy, torch.zeros_like(xy[:, :1])], dim=-1)
        self.env_origin = xyz[:self.n_envs] # [n_envs, 3]
        
        n_boxes_per_drone = 4 # use 4 boxes to represent a drone simply
        self.drone_mesh_dict = {
            "vertices":         ti.Vector.field(3, ti.f32, shape=(self.n_envs * self.n_agents *  8*n_boxes_per_drone)),
            "indices":                 ti.field(   ti.i32, shape=(self.n_envs * self.n_agents * 36*n_boxes_per_drone)),
            "per_vertex_color": ti.Vector.field(3, ti.f32, shape=(self.n_envs * self.n_agents *  8*n_boxes_per_drone))
        }
        self.drone_mesh_dict_one_env = {
            "vertices":         ti.Vector.field(3, ti.f32, shape=(self.n_agents *  8*n_boxes_per_drone, )),
            "indices":                 ti.field(   ti.i32, shape=(self.n_agents * 36*n_boxes_per_drone, )),
            "per_vertex_color": ti.Vector.field(3, ti.f32, shape=(self.n_agents *  8*n_boxes_per_drone, ))
        }
        self.drone_vertices_tensor = torch.empty(self.n_envs, self.n_agents, 32, 3, device=self.device)
        self._init_drone_model()
        
        if self.ground_plane:
            edge_length = 5
            ground_plane_size = max(edge_length*2, int(N.item() * self.L * 2 + self.L))
            self.n_plane: int = max(2, int(ground_plane_size / edge_length + 1))
            n_ground_faces = self.n_plane * self.n_plane
            n_ground_vertices = n_ground_faces * 4
            n_ground_indices = n_ground_faces * 6
            self.ground_plane_mesh_dict = {
                "vertices":         ti.Vector.field(3, ti.f32, shape=n_ground_vertices),
                "indices":                 ti.field(   ti.i32, shape=n_ground_indices),
                "per_vertex_color": ti.Vector.field(3, ti.f32, shape=n_ground_vertices)
            }
            z_ground_plane = -self.L - 0.1 if height_scale is None else -self.L * height_scale - 0.1
            self._init_ground_plane(z_ground_plane=z_ground_plane, edge_length=edge_length)
        
        self.gui_states = {
            "reset_all": False,
            "enable_lightsource": True,
            "brightness": 1.0,
            "display_basis": False,
            "display_target_line": True,
            "display_groundplane": True,
            "prev_tracking_view": False,
            "tracking_view": False,
            "render_one_env": False,
            "fpp_view": False,
            "lookat_target": False,
            "tracking_env_idx": 0,
            "tracking_agent_idx": 0
        }
        if self.record_video:
            self.video_H, self.video_W, self.video_fov = cfg.video_camera.height, cfg.video_camera.width, cfg.video_camera.fov
            self.video_frame = np.empty((self.n_envs, self.video_H, self.video_W, 3), dtype=np.uint8)
            self.video_cam_pos    = torch.empty(self.n_envs, 3, device=self.device)
            self.video_cam_lookat = torch.empty(self.n_envs, 3, device=self.device)
            self.video_cam_up     = torch.empty(self.n_envs, 3, device=self.device)
        self._init_viewer()
        self.camera_state = {
            "position": self.gui_camera.curr_position,
            "lookat": self.gui_camera.curr_lookat,
            "up": self.gui_camera.curr_up,
        }
        
        end_points = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.axis_lines = [(ti.Vector.field(3, ti.f32, shape=2), tuple(end_points[i])) for i in range(3)]
        for i, end_point in enumerate(end_points):
            self.axis_lines[i][0].from_torch(torch2ti(torch.tensor([
                [0, 0, 0],
                end_point
            ], dtype=torch.float32, device=self.device)))
        self.target_line_vertices         = ti.Vector.field(3, ti.f32, shape=(self.n_envs * self.n_agents * 2))
        self.target_line_colors           = ti.Vector.field(3, ti.f32, shape=(self.n_envs * self.n_agents * 2))
        self.target_line_vertices_one_env = ti.Vector.field(3, ti.f32, shape=(self.n_agents * 2))
        self.target_line_colors_one_env   = ti.Vector.field(3, ti.f32, shape=(self.n_agents * 2))
        self.target_line_color_tensor = torch.zeros(self.n_envs, self.n_agents, 3, device=self.device)
    
    def _init_viewer(self):
        self.gui_window = ti.ui.Window(
            name='Renderer Running at',
            res=(1280, 900),
            fps_limit=2*int(1/self.dt),
            pos=(150, 150), 
            show_window=not self.headless
        )
        self.gui_handle = self.gui_window.get_gui()
        self.gui_scene = self.gui_window.get_scene()
        self.gui_camera = ti.ui.make_camera()
        env_bound = max(5, self.env_origin.max().item())
        self.gui_camera.position(-1.5*env_bound, 0.5*env_bound, -1.8*env_bound)  # x, y, z
        self.gui_camera.lookat(0, -0.1*env_bound, 0)
        self.gui_camera.up(0, 1, 0) 
        # self.gui_camera.z_far(200)
        self.gui_camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        self.gui_canvas = self.gui_window.get_canvas()
        
        if self.record_video:
            self.video_window = ti.ui.Window(name=f"Env", res=(self.video_W, self.video_H), fps_limit=int(1/self.dt), show_window=False)
            self.video_gui    = self.video_window.get_gui()
            self.video_scene  = self.video_window.get_scene()
            self.video_camera = ti.ui.make_camera()
            self.video_canvas = self.video_window.get_canvas()
    
    def _set_camera_state(self, camera_handle, camera_state, fov=90.):
        # type: (ti.ui.Camera, Dict[str, List[ti.Vector]], float) -> None
        camera_handle.position(*camera_state["position"])
        camera_handle.lookat(*camera_state["lookat"])
        camera_handle.up(*camera_state["up"])
        camera_handle.fov(fov=fov)
    
    @ti.kernel
    def _init_ground_plane(self, z_ground_plane: float, edge_length: float):
        # Initialize the ground plane mesh by creating a grid of quadrilateral faces
        for i, j in ti.ndrange(self.n_plane, self.n_plane):
            idx = (i * self.n_plane + j)
            vertex_base = idx * 4  # 4 vertices per face

            # Calculate the coordinates for the current face
            x0 = (i - self.n_plane / 2) * edge_length
            y0 = (j - self.n_plane / 2) * edge_length
            x1 = x0 + edge_length
            y1 = y0 + edge_length

            # Define the 4 vertices of the face
            self.ground_plane_mesh_dict["vertices"][vertex_base + 0] = ti.Vector([x0, z_ground_plane, y0])
            self.ground_plane_mesh_dict["vertices"][vertex_base + 1] = ti.Vector([x1, z_ground_plane, y0])
            self.ground_plane_mesh_dict["vertices"][vertex_base + 2] = ti.Vector([x0, z_ground_plane, y1])
            self.ground_plane_mesh_dict["vertices"][vertex_base + 3] = ti.Vector([x1, z_ground_plane, y1])

            # Assign color to the vertices of the face, alternating colors for a checkerboard pattern
            color = 0.8 if (i + j) % 2 == 0 else 0.2
            for k in range(4):
                self.ground_plane_mesh_dict["per_vertex_color"][vertex_base + k] = ti.Vector([color, color, color])

            # Define the two triangles that make up the face by specifying vertex indices
            index_base = idx * 6
            self.ground_plane_mesh_dict["indices"][index_base + 0] = vertex_base + 0
            self.ground_plane_mesh_dict["indices"][index_base + 1] = vertex_base + 1
            self.ground_plane_mesh_dict["indices"][index_base + 2] = vertex_base + 2
            self.ground_plane_mesh_dict["indices"][index_base + 3] = vertex_base + 1
            self.ground_plane_mesh_dict["indices"][index_base + 4] = vertex_base + 3
            self.ground_plane_mesh_dict["indices"][index_base + 5] = vertex_base + 2
    
    def _create_envs(self):
        raise NotImplementedError
    
    def _init_drone_model(self):
        l, L = 0.02, 0.1
        D = (L + l) / 2 / 2**0.5
        vertices_tensor, indices_tensor, color_tensor = add_box(
            xyz=torch.tensor([
                [ D,  D, 0],
                [-D,  D, 0],
                [ D, -D, 0],
                [-D, -D, 0]], device=self.device
            ).reshape(1, 1, 4, 3).expand(self.n_envs, self.n_agents, -1, -1),
            lwh=torch.tensor([
                [L, l, l],
                [l, L, l],
                [l, L, l],
                [L, l, l]], device=self.device
            ).reshape(1, 1, 4, 3).expand(self.n_envs, self.n_agents, -1, -1),
            rpy=torch.tensor([
                [0, 0, torch.pi/4]
            ], device=self.device).reshape(1, 1, 1, 3).expand(self.n_envs, self.n_agents, 4, -1),
            color=torch.tensor([
                [0.8867, 0.9219, 0.1641],
                [0.5156, 0.1016, 0.5391],
                [0.8867, 0.9219, 0.1641],
                [0.5156, 0.1016, 0.5391]], device=self.device
            ).reshape(1, 1, 4, 3).expand(self.n_envs, self.n_agents, -1, -1)
        )
        self.drone_vertices_tensor.copy_(vertices_tensor.reshape(self.n_envs, self.n_agents, -1, 3))
        self.drone_mesh_dict["vertices"].from_torch(torch2ti(self.drone_vertices_tensor.flatten(end_dim=-2)))
        self.drone_mesh_dict["indices"].from_torch(indices_tensor.flatten())
        self.drone_mesh_dict["per_vertex_color"].from_torch(color_tensor.flatten(end_dim=-2))
        self.drone_mesh_dict_one_env["vertices"].from_torch(torch2ti(self.drone_vertices_tensor[0]))
        self.drone_mesh_dict_one_env["indices"].from_torch(indices_tensor[0].flatten())
        self.drone_mesh_dict_one_env["per_vertex_color"].from_torch(color_tensor[0].flatten(end_dim=-2))
    
    def _update_drone_pose(
        self,
        pos: Tensor,      # [n_envs, n_agents, 3]
        quat_xyzw: Tensor # [n_envs, n_agents, 4]
    ):
        rotation_matrix = T.quaternion_to_matrix(quat_xyzw.roll(1, dims=-1))
        drone_vertices_tensor = torch.matmul(self.drone_vertices_tensor, rotation_matrix.transpose(-2, -1))
        absolute_pos = pos + self.env_origin.unsqueeze(1)
        drone_vertices_tensor = drone_vertices_tensor + absolute_pos.unsqueeze(-2) # [n_envs, n_agents, 32, 3]
        if self.enable_rendering:
            self.drone_mesh_dict["vertices"].from_torch(torch2ti(drone_vertices_tensor.flatten(end_dim=-2)))
        
        idx = self.gui_states["tracking_env_idx"]
        self.drone_mesh_dict_one_env["vertices"].from_torch(torch2ti(drone_vertices_tensor[idx].flatten(end_dim=-2)))

    def _update_camera_pose(
        self,
        pos: Tensor,        # [n_envs, n_agents, 3]
        quat_xyzw: Tensor,  # [n_envs, n_agents, 4]
        target_pos: Tensor, # [n_envs, n_agents, 3]
    ):
        rotation_matrix = T.quaternion_to_matrix(quat_xyzw.roll(1, dims=-1))
        absolute_pos = pos + self.env_origin.unsqueeze(1)
        unbind = lambda xyz: tuple(map(lambda x: x.item(), torch2ti(xyz).unbind(dim=-1)))
        
        if self.gui_states["tracking_view"]:
            env_idx = self.gui_states["tracking_env_idx"]
            agent_idx = self.gui_states["tracking_agent_idx"]
            pos_tracked = pos[env_idx, agent_idx]
            absolute_pos_tracked = absolute_pos[env_idx, agent_idx]
            rotation_matrix_tracked = rotation_matrix[env_idx, agent_idx]
            target_pos_tracked = target_pos[env_idx, agent_idx]
            quat_xyzw_tracked = quat_xyzw[env_idx, agent_idx]
            
            if self.gui_states["fpp_view"]:
                campos_w = absolute_pos_tracked
                lookat = torch.mm(rotation_matrix_tracked, torch.tensor([[1., 0., 0.]], device=self.device).T).T + absolute_pos_tracked
                up_b = torch.tensor([[0., 0., 1.]], device=self.device)
                up_w = torch.mm(rotation_matrix_tracked, up_b.T).T
            else:
                if self.gui_states["lookat_target"]:
                    target_relpos = target_pos_tracked - pos_tracked
                    yaw_rotmat = axis_rotmat("Z", torch.atan2(target_relpos[1], target_relpos[0]))
                else:
                    yaw_rotmat = axis_rotmat("Z", quaternion_to_euler(quat_xyzw_tracked)[..., -1])
                campos_b = torch.tensor([[-1., 0., 0.5]], device=self.device)
                campos_w = torch.mm(yaw_rotmat, campos_b.T).T + absolute_pos_tracked
                lookat = absolute_pos_tracked
                up_w = torch.tensor([[0., 0., 1.]], device=self.device)
            cam_state = {
                "position": unbind(campos_w),
                "lookat": unbind(lookat),
                "up": unbind(up_w)}
            self._set_camera_state(self.gui_camera, cam_state, fov=90.)
        
        if self.record_video:
            self.video_cam_pos.copy_(absolute_pos[:, 0])
            lookat_b = torch.tensor([[[1., 0., 0.]]], device=self.device).expand(self.n_envs, -1, -1)
            lookat_w = torch.bmm(lookat_b, rotation_matrix[:, 0].transpose(-2, -1)).squeeze(1) + absolute_pos[:, 0]
            self.video_cam_lookat.copy_(lookat_w)
            up_b = torch.tensor([[0., 0., 1.]], device=self.device).expand(self.n_envs, -1, -1)
            up_w = torch.bmm(up_b, rotation_matrix[:, 0].transpose(-2, -1)).squeeze(1)
            self.video_cam_up.copy_(up_w)
    
    def _update_lines(self, drone_pos: Tensor, target_pos: Tensor):
        idx = self.gui_states["tracking_env_idx"]
        target_line_vertices_tensor = torch.stack([drone_pos, target_pos], dim=-2) + self.env_origin.reshape(self.n_envs, 1, 1, 3)
        self.target_line_vertices.from_torch(torch2ti(target_line_vertices_tensor.flatten(end_dim=-2)))
        self.target_line_vertices_one_env.from_torch(torch2ti(target_line_vertices_tensor[idx].flatten(end_dim=-2)))
        factory_kwargs = {"dtype": torch.float32, "device": self.device}
        near_target = (drone_pos-target_pos).norm(dim=-1).lt(0.5).unsqueeze(-1).expand(-1, -1, 3)
        white = torch.tensor([[[1., 1., 1.]]], **factory_kwargs).expand(self.n_envs, self.n_agents, -1)
        # red = torch.tensor([[1., 0., 0.]], **factory_kwargs).expand_as(self.target_line_color_tensor)
        # yellow = torch.tensor([[0.7, 0.7, 0.2]], **factory_kwargs).expand_as(self.target_line_color_tensor)
        green = torch.tensor([[[0., 1., 0.]]], **factory_kwargs).expand(self.n_envs, self.n_agents, -1)
        self.target_line_color_tensor = torch.where(near_target, green, white)
        target_line_color_tensor = self.target_line_color_tensor.unsqueeze(-2).expand(-1, -1, 2, -1)
        self.target_line_colors.from_torch(target_line_color_tensor.flatten(end_dim=-2))
        self.target_line_colors_one_env.from_torch(target_line_color_tensor[idx].flatten(end_dim=-2))
    
    def _update_state(self, states_for_rendering: Dict[str, Tensor]):
        if self.enable_rendering:
            pos = states_for_rendering["pos"]
            quat_xyzw = states_for_rendering["quat_xyzw"]
            target_pos = states_for_rendering["target_pos"]
            self._update_drone_pose(pos, quat_xyzw)
            self._update_camera_pose(pos, quat_xyzw, target_pos)
            self._update_lines(pos, target_pos)

    def _render_subwindows(self, states_for_rendering: Dict[str, Tensor]):
        self.gui_states["reset_all"] = False
        with self.gui_handle.sub_window("Simulation Settings", x=0.02, y=0.02, height=0.1, width=0.35) as sub_window:
            self.gui_states["reset_all"] = sub_window.button("(R) Reset All")
            if sub_window.button("(ESC) Exit"): raise KeyboardInterrupt
        
        with self.gui_handle.sub_window("Render Settings", x=0.02, y=0.14, height=0.3, width=0.35) as sub_window:
            if sub_window.button("(V) Pause Rendering"):
                self.enable_rendering = False
                sub_window.text("Rendering paused. Press \"V\" to resume.")
                Logger.info("Rendering paused. Press \"V\" to resume.")
            self.gui_states["enable_lightsource"] = sub_window.checkbox("Enable Light Source", self.gui_states["enable_lightsource"])
            if self.gui_states["enable_lightsource"]:
                self.gui_states["brightness"] = sub_window.slider_float("Brightness", self.gui_states["brightness"], minimum=0, maximum=1)
            self.gui_states["display_basis"] = sub_window.checkbox("(B) Display Axis Basis", self.gui_states["display_basis"])
            self.gui_states["display_target_line"] = sub_window.checkbox("(L) Display Target Line", self.gui_states["display_target_line"])
            if self.ground_plane:
                self.gui_states["display_groundplane"] = sub_window.checkbox("(G) Display Ground Plane", self.gui_states["display_groundplane"])
            self.gui_states["render_one_env"] = sub_window.checkbox("(O) Render selected env only", self.gui_states["render_one_env"])
            if self.gui_states["render_one_env"]:
                self.gui_states["tracking_env_idx"] = sub_window.slider_int(
                    "Tracking Env Index",
                    self.gui_states["tracking_env_idx"],
                    minimum=0,
                    maximum=self.n_envs-1)
                self.gui_states["tracking_agent_idx"] = sub_window.slider_int(
                    "Tracking Agent Index",
                    self.gui_states["tracking_agent_idx"],
                    minimum=0,
                    maximum=self.n_agents-1)
            self.gui_states["prev_tracking_view"] = self.gui_states["tracking_view"]
            self.gui_states["tracking_view"] = sub_window.checkbox("(T) Tracking View", self.gui_states["tracking_view"])
            if self.gui_states["tracking_view"]:
                if not self.gui_states["render_one_env"]:
                    self.gui_states["render_one_env"] = True
                self.gui_states["fpp_view"] = sub_window.checkbox("(F) First Person View", self.gui_states["fpp_view"])
                self.gui_states["lookat_target"] = sub_window.checkbox("Look at Target", self.gui_states["lookat_target"])
            
        self._render_agent_states(states_for_rendering, handle=self.gui_handle)

    def _render_agent_states(self, states_for_rendering, handle, env_idx=None):
        # type: (Dict[str, Tensor], ti.ui.Gui, Optional[int]) -> None
        env_idx = self.gui_states['tracking_env_idx'] if env_idx is None else env_idx
        agent_idx = self.gui_states['tracking_agent_idx']
        tensor2list = lambda x: list(map(lambda y: f"{y.item():6.2f}", x.unbind(dim=-1)))
        pos = states_for_rendering['pos'][env_idx, agent_idx]
        quat_xyzw = states_for_rendering['quat_xyzw'][env_idx, agent_idx]
        euler_angles = quaternion_to_euler(quat_xyzw) * 180 / torch.pi
        vel = states_for_rendering['vel'][env_idx, agent_idx]
        rotmat_w2b = T.quaternion_to_matrix(quat_xyzw.roll(1, dims=-1)).transpose(-2, -1)
        vel_b = torch.matmul(rotmat_w2b, vel.unsqueeze(-1)).squeeze(-1)
        tgt_pos = states_for_rendering['target_pos'][env_idx, agent_idx] - pos
        tgt_pos_b = torch.matmul(rotmat_w2b, tgt_pos.unsqueeze(-1)).squeeze(-1)
        
        if handle == self.gui_handle:
            with handle.sub_window("Agent States", x=0.4, y=0.02, height=0.18, width=0.2) as sub_window:
                sub_window.text(f"pos:   [" + ", ".join(tensor2list(pos)) + "]m")
                sub_window.text(f"tgt_b: [" + ", ".join(tensor2list(tgt_pos_b)) + "]m")
                sub_window.text(f"rpy:   [" + ", ".join(tensor2list(euler_angles)) + "]°")
                sub_window.text(f"vel_w: [" + ", ".join(tensor2list(vel)) + "]m/s")
                sub_window.text(f"vel_b: [" + ", ".join(tensor2list(vel_b)) + "]m/s")
        elif hasattr(self, "video_gui") and handle == self.video_gui:
            pos_str = f"pos:   [" + ", ".join(tensor2list(pos)) + "]m   " + f"tgt_b: [" + ", ".join(tensor2list(tgt_pos_b)) + "]m"
            vel_str = f"vel_w: [" + ", ".join(tensor2list(vel)) + "]m/s " + f"vel_b: [" + ", ".join(tensor2list(vel_b)) + "]m/s"
            with handle.sub_window(pos_str, x=0.0, y=0.83, height=0.17, width=1.) as sub_window:
                sub_window.text("  "+vel_str)

    def _track_user_inputs(self):
        # Track user inputs
        self.gui_camera.track_user_inputs(
            self.gui_window,
            movement_speed=0.25,
            pitch_speed=4,
            yaw_speed=4,
            hold_key=ti.ui.RMB)
        
        if self.gui_window.get_event(ti.ui.PRESS):
            if self.gui_window.event.key == 'v':
                self.enable_rendering = not self.enable_rendering
                if not self.enable_rendering:
                    Logger.info("Rendering paused. Press \"V\" to resume.")
            if self.gui_window.event.key == 'r':
                self.gui_states["reset_all"] = True
            if self.gui_window.event.key == 'l':
                self.gui_states["display_target_line"] = not self.gui_states["display_target_line"]
            if self.gui_window.event.key == 'b':
                self.gui_states["display_basis"] = not self.gui_states["display_basis"]
            if self.gui_window.event.key == 'g':
                self.gui_states["display_groundplane"] = not self.gui_states["display_groundplane"]
            if self.gui_window.event.key == 'o':
                self.gui_states["render_one_env"] = not self.gui_states["render_one_env"]
            if self.gui_window.event.key == 't':
                self.gui_states["tracking_view"] = not self.gui_states["tracking_view"]
            if self.gui_window.event.key == 'f':
                self.gui_states["fpp_view"] = not self.gui_states["fpp_view"]
            if self.gui_states["render_one_env"]:
                if self.gui_window.is_pressed(ti.ui.RIGHT):
                    self.gui_states["tracking_env_idx"] = (self.gui_states["tracking_env_idx"] + 1) % self.n_envs
                elif self.gui_window.is_pressed(ti.ui.LEFT):
                    self.gui_states["tracking_env_idx"] = (self.gui_states["tracking_env_idx"] + self.n_envs - 1) % self.n_envs
                elif self.gui_window.is_pressed(ti.ui.UP):
                    self.gui_states["tracking_agent_idx"] = (self.gui_states["tracking_agent_idx"] + 1) % self.n_agents
                elif self.gui_window.is_pressed(ti.ui.DOWN):
                    self.gui_states["tracking_agent_idx"] = (self.gui_states["tracking_agent_idx"] + self.n_agents - 1) % self.n_agents
        
        if self.gui_states["prev_tracking_view"] != self.gui_states["tracking_view"]:
            if not self.gui_states["tracking_view"]:
                # restore camera state
                self._set_camera_state(self.gui_camera, self.camera_state, fov=45.)
            else:
                # backup current camera state
                self.camera_state = {
                    "position": self.gui_camera.curr_position,
                    "lookat": self.gui_camera.curr_lookat,
                    "up": self.gui_camera.curr_up
                }
    
    def render_fpp(self, states_for_rendering: Dict[str, Tensor]):
        assert self.record_video
        unbind = lambda xyz: tuple(map(lambda x: x.item(), torch2ti(xyz).unbind(dim=-1)))
        for i in range(self.n_envs):
            self.video_scene.mesh(**self.ground_plane_mesh_dict)
            self.video_scene.point_light(pos=(0, 50, 0), color=(1., 1., 1.))
            self.video_scene.ambient_light(color=(0.5, 0.5, 0.5))
            self._render_agent_states(states_for_rendering, handle=self.video_gui, env_idx=i)
            video_camera_state = {
                "position": unbind(self.video_cam_pos[i]),
                "lookat": unbind(self.video_cam_lookat[i]),
                "up": unbind(self.video_cam_up[i])
            }
            self._set_camera_state(self.video_camera, video_camera_state, fov=self.video_fov)
            self.video_scene.set_camera(self.video_camera)
            self.video_canvas.scene(self.video_scene)
            buffer: np.ndarray = (self.video_window.get_image_buffer_as_numpy() * 255).astype(np.uint8)
            self.video_frame[i] = np.flip(buffer[..., :3].transpose(1, 0, 2), axis=0)
        return self.video_frame

    def render(self, states_for_rendering: Dict[str, Tensor]):
        env_spacing = states_for_rendering["env_spacing"]
        self.env_origin[:, 2] = (env_spacing - self.L) * self.height_scale
        self._update_state(states_for_rendering)
        if self.headless:
            return
        if self.enable_rendering:
            self._render_subwindows(states_for_rendering)
            self._track_user_inputs()
        
            # render ground plane
            if self.ground_plane and self.gui_states["display_groundplane"]:
                self.gui_scene.mesh(**self.ground_plane_mesh_dict)
            
            # render drones
            if self.gui_states["render_one_env"]:
                self.gui_scene.mesh(**self.drone_mesh_dict_one_env)
            else:
                self.gui_scene.mesh(**self.drone_mesh_dict)
            
            # render external obstacles
            self._render_obstacles()
            
            # render lines
            self._render_lines()
            
            # set illumination
            if self.gui_states["enable_lightsource"]:
                self.gui_scene.point_light(pos=(0, 50, 0), color=(self.gui_states["brightness"] for _ in range(3)))
            # self.gui_scene.ambient_light(color=(self.gui_states["brightness"]*0.5 for _ in range(3)))
            self.gui_scene.ambient_light(color=(0.5, 0.5, 0.5))
            
            self.gui_scene.set_camera(self.gui_camera)
            self.gui_canvas.scene(self.gui_scene)
            self.gui_window.show()
        
        v_pressed = self.gui_window.get_event(ti.ui.PRESS) and self.gui_window.event.key == 'v'
        if not self.enable_rendering and v_pressed:
            self.enable_rendering = True
        if self.gui_window.is_pressed(ti.ui.ESCAPE):
            raise KeyboardInterrupt
    
    def _render_lines(self):
        if self.gui_states["display_basis"]:
            for line, color in self.axis_lines:
                self.gui_scene.lines(line, color=color, width=5.0)
        if self.gui_states["display_target_line"]:
            if self.gui_states["render_one_env"]:
                vertices, colors = self.target_line_vertices_one_env, self.target_line_colors_one_env
            else:
                vertices, colors = self.target_line_vertices, self.target_line_colors
            self.gui_scene.lines(vertices=vertices, per_vertex_color=colors, width=1.0)
    
    def _render_obstacles(self):
        pass
    
    def close(self):
        if not self.headless:
            self.gui_window.destroy()
        if self.record_video:
            self.video_window.destroy()


class PositionControlRenderer(BaseRenderer):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
            

class ObstacleAvoidanceRenderer(BaseRenderer):
    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device,
        obstacle_manager: ObstacleManager,
        height_scale: Optional[float] = None,
        headless: bool = False
    ):
        super().__init__(cfg, device, height_scale=height_scale, headless=headless)
        self.obstacle_manager = obstacle_manager
        self.cube_color = [0.8, 0.3, 0.1]
        self.sphere_color = [0.8, 0.1, 0.3]
        self.sphere_n_segments: int = cfg.sphere_n_segments
        self.sphere_n_vertices = (self.sphere_n_segments - 1) * self.sphere_n_segments * 2 + 2
        self.sphere_n_triangles = 4 * self.sphere_n_segments * (self.sphere_n_segments - 1)
        
        n_cubes = self.obstacle_manager.n_cubes
        self.cube_mesh_dict = {
            "vertices":          ti.Vector.field(3, ti.f32, shape=(self.n_envs * n_cubes *  8)),
            "indices":                  ti.field(   ti.i32, shape=(self.n_envs * n_cubes * 36)),
            "per_vertex_color":  ti.Vector.field(3, ti.f32, shape=(self.n_envs * n_cubes *  8))
        }
        self.cube_mesh_dict_one_env = {
            "vertices":          ti.Vector.field(3, ti.f32, shape=(n_cubes *  8, )),
            "indices":                  ti.field(   ti.i32, shape=(n_cubes * 36, )),
            "per_vertex_color":  ti.Vector.field(3, ti.f32, shape=(n_cubes *  8, ))
        }
        self.cube_vertices_tensor = torch.empty(self.n_envs, n_cubes, 8, 3, device=self.device)
        
        n_spheres = self.obstacle_manager.n_spheres
        self.sphere_mesh_dict = {
            "vertices":          ti.Vector.field(3, ti.f32, shape=(self.n_envs * n_spheres * self.sphere_n_vertices)),
            "indices":                  ti.field(   ti.i32, shape=(self.n_envs * n_spheres * self.sphere_n_triangles * 3)),
            "per_vertex_color":  ti.Vector.field(3, ti.f32, shape=(self.n_envs * n_spheres * self.sphere_n_vertices))
        }
        self.sphere_mesh_dict_one_env = {
            "vertices":          ti.Vector.field(3, ti.f32, shape=(n_spheres * self.sphere_n_vertices,)),
            "indices":                  ti.field(   ti.i32, shape=(n_spheres * self.sphere_n_triangles * 3,)),
            "per_vertex_color":  ti.Vector.field(3, ti.f32, shape=(n_spheres * self.sphere_n_vertices,))
        }
        self.sphere_vertices_tensor = torch.empty(self.n_envs, n_spheres, self.sphere_n_vertices, 3, device=self.device)
        
        self._init_obstacles()
        
        self.target_line_vertices         = ti.Vector.field(3, ti.f32, shape=(self.n_envs * 2))
        self.target_line_colors           = ti.Vector.field(3, ti.f32, shape=(self.n_envs * 2))
        self.target_line_vertices_one_env = ti.Vector.field(3, ti.f32, shape=(2, ))
        self.target_line_colors_one_env   = ti.Vector.field(3, ti.f32, shape=(2, ))
        self.target_line_color_tensor = torch.empty(self.n_envs, 3, device=self.device)
        
        self.nearest_points_field = ti.Vector.field(3, ti.f32, shape=(self.n_envs * self.obstacle_manager.n_obstacles))
        self.nearest_points_field_one_env = ti.Vector.field(3, ti.f32, shape=(self.obstacle_manager.n_obstacles))
    
    def _init_obstacles(self):
        # initialize meshes of the cubes
        lwh = self.obstacle_manager.lwh_cubes[:self.n_envs]
        xyz = torch.zeros_like(lwh)
        vertices_tensor, indices_tensor, color_tensor = add_box(
            xyz=xyz,
            lwh=self.obstacle_manager.lwh_cubes[:self.n_envs],
            rpy=self.obstacle_manager.rpy_cubes[:self.n_envs],
            color=torch.tensor([[self.cube_color]], device=self.device).expand_as(lwh)
        )
        self.cube_vertices_tensor.copy_(vertices_tensor)
        self.cube_mesh_dict["indices"].from_torch(indices_tensor.flatten())
        self.cube_mesh_dict["per_vertex_color"].from_torch(color_tensor.flatten(end_dim=-2))
        self.cube_mesh_dict_one_env["indices"].from_torch(indices_tensor[0].flatten())
        self.cube_mesh_dict_one_env["per_vertex_color"].from_torch(color_tensor[0].flatten(end_dim=-2))
        
        # initialize meshes of the spheres
        xyz = torch.zeros_like(self.obstacle_manager.p_spheres[:self.n_envs])
        vertices_tensor, indices_tensor, color_tensor = add_sphere(
            xyz=xyz,
            radius=self.obstacle_manager.r_spheres[:self.n_envs],
            lat_segments=self.sphere_n_segments,
            lon_segments=self.sphere_n_segments * 2,
            color=torch.tensor([[self.sphere_color]], device=self.device).expand_as(xyz)
        )
        self.sphere_vertices_tensor.copy_(vertices_tensor)
        self.sphere_mesh_dict["indices"].from_torch(indices_tensor.flatten())
        self.sphere_mesh_dict["per_vertex_color"].from_torch(color_tensor.flatten(end_dim=-2))
        self.sphere_mesh_dict_one_env["indices"].from_torch(indices_tensor[0].flatten())
        self.sphere_mesh_dict_one_env["per_vertex_color"].from_torch(color_tensor[0].flatten(end_dim=-2))
    
    def _update_state(self, states_for_rendering: Dict[str, Tensor]):
        super()._update_state(states_for_rendering)
        if self.enable_rendering:
            self._update_obstacles()
        nearest_points = states_for_rendering.get("nearest_points", None)
        if nearest_points is not None:
            nearest_points_tensor = nearest_points[:self.n_envs] + self.env_origin.unsqueeze(-2) # [n_envs, n_obstacles, 3]
            self.nearest_points_field.from_torch(torch2ti(nearest_points_tensor.flatten(end_dim=-2)))
            self.nearest_points_field_one_env.from_torch(torch2ti(nearest_points_tensor[self.gui_states["tracking_env_idx"]].flatten(end_dim=-2)))

    def _update_obstacles(self):
        # update the pose of the cubes
        cube_vertices_tensor, cube_indices_tensor, cube_color_tensor = add_box(
            xyz=torch.zeros(self.n_envs, self.obstacle_manager.n_cubes, 3, device=self.device),
            lwh=self.obstacle_manager.lwh_cubes[:self.n_envs],
            rpy=self.obstacle_manager.rpy_cubes[:self.n_envs],
            color=torch.tensor([[self.cube_color]], device=self.device).expand(self.n_envs, self.obstacle_manager.n_cubes, -1)
        )
        self.cube_vertices_tensor.copy_(cube_vertices_tensor)
        
        sphere_vertices_tensor, sphere_indices_tensor, sphere_color_tensor = add_sphere(
            xyz=torch.zeros_like(self.obstacle_manager.p_spheres[:self.n_envs]),
            radius=self.obstacle_manager.r_spheres[:self.n_envs],
            lat_segments=self.sphere_n_segments,
            lon_segments=self.sphere_n_segments * 2,
            color=torch.tensor([[self.sphere_color]], device=self.device).expand(self.n_envs, self.obstacle_manager.n_spheres, -1)
        )
        self.sphere_vertices_tensor.copy_(sphere_vertices_tensor)
        
        idx = self.gui_states["tracking_env_idx"]
        cube_vertices_tensor = (
            self.cube_vertices_tensor + 
            self.obstacle_manager.p_cubes[:self.n_envs].unsqueeze(-2) + 
            self.env_origin.unsqueeze(-2).unsqueeze(-2))
        sphere_vertices_tensor = (
            self.sphere_vertices_tensor +
            self.obstacle_manager.p_spheres[:self.n_envs].unsqueeze(-2) + 
            self.env_origin.unsqueeze(-2).unsqueeze(-2)
        )
        
        if self.enable_rendering:
            self.cube_mesh_dict["vertices"].from_torch(torch2ti(cube_vertices_tensor.flatten(end_dim=-2)))
            self.sphere_mesh_dict["vertices"].from_torch(torch2ti(sphere_vertices_tensor.flatten(end_dim=-2)))
        
        self.cube_mesh_dict_one_env["vertices"].from_torch(torch2ti(cube_vertices_tensor[idx].flatten(end_dim=-2)))
        self.sphere_mesh_dict_one_env["vertices"].from_torch(torch2ti(sphere_vertices_tensor[idx].flatten(end_dim=-2)))
    
    def _render_obstacles(self):
        if self.gui_states["tracking_view"]:
            self.gui_scene.mesh(**self.cube_mesh_dict_one_env)
            self.gui_scene.mesh(**self.sphere_mesh_dict_one_env)
            self.gui_scene.particles(
                centers=self.nearest_points_field_one_env,
                radius=0.1,
                color=(0.2, 0.8, 0.2),
            )
        else:
            self.gui_scene.mesh(**self.cube_mesh_dict)
            self.gui_scene.mesh(**self.sphere_mesh_dict)
            self.gui_scene.particles(
                centers=self.nearest_points_field,
                radius=0.1,
                color=(0.2, 0.8, 0.2),
            )
    
    def render_fpp(self, states_for_rendering: Dict[str, Tensor]):
        assert self.record_video
        unbind = lambda xyz: tuple(map(lambda x: x.item(), torch2ti(xyz).unbind(dim=-1)))
        for i in range(self.n_envs):
            cube_vertices_tensor = (
                self.cube_vertices_tensor + 
                self.obstacle_manager.p_cubes[:self.n_envs].unsqueeze(-2) + 
                self.env_origin.unsqueeze(-2).unsqueeze(-2))
            self.cube_mesh_dict_one_env["vertices"].from_torch(torch2ti(cube_vertices_tensor[i].flatten(end_dim=-2)))
            self.video_scene.mesh(**self.cube_mesh_dict_one_env)
            
            sphere_vertices_tensor = (
                self.sphere_vertices_tensor + 
                self.obstacle_manager.p_spheres[:self.n_envs].unsqueeze(-2) + 
                self.env_origin.unsqueeze(-2).unsqueeze(-2))
            self.sphere_mesh_dict_one_env["vertices"].from_torch(torch2ti(sphere_vertices_tensor[i].flatten(end_dim=-2)))
            self.video_scene.mesh(**self.sphere_mesh_dict_one_env)
            
            self.video_scene.mesh(**self.ground_plane_mesh_dict)
            self.video_scene.point_light(pos=(0, 50, 0), color=(1., 1., 1.))
            self.video_scene.ambient_light(color=(0.5, 0.5, 0.5))
            self._render_agent_states(states_for_rendering, handle=self.video_gui, env_idx=i)
            video_camera_state = {
                "position": unbind(self.video_cam_pos[i]),
                "lookat": unbind(self.video_cam_lookat[i]),
                "up": unbind(self.video_cam_up[i])
            }
            self._set_camera_state(self.video_camera, video_camera_state, fov=self.video_fov)
            self.video_scene.set_camera(self.video_camera)
            self.video_canvas.scene(self.video_scene)
            buffer: np.ndarray = (self.video_window.get_image_buffer_as_numpy() * 255).astype(np.uint8)
            self.video_frame[i] = np.flip(buffer[..., :3].transpose(1, 0, 2), axis=0)
        return self.video_frame
    