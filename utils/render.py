from typing import Optional, Tuple

import torch
from torch import Tensor
from pytorch3d import transforms as T
import taichi as ti
from omegaconf import DictConfig
from tqdm import tqdm

from quaddif.utils.assets import ObstacleManager

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
    indices_torch = INDICES_TORCH.unsqueeze(0).expand(n_boxes, -1)
    indices_torch = indices_torch + torch.arange(0, 8 * n_boxes, 8, dtype=torch.int32).unsqueeze(-1)
    while color.dim() < rotated_vertices_tensor.dim() - 1:
        color = color.unsqueeze(0)
    color = color.unsqueeze(-2).expand_as(rotated_vertices_tensor)
    return rotated_vertices_tensor, indices_torch, color

class BaseRenderer:
    def __init__(self, cfg: DictConfig, device: torch.device, z_ground_plane: Optional[float] = None):
        # ti.init(arch=ti.vulkan)
        if "cpu" in str(device):
            print("Using CPU to render the GUI.")
            ti.init(arch=ti.cpu)
        else:
            print("Using GPU to render the GUI.")
            ti.init(arch=ti.gpu)
        self.n_envs: int = min(cfg.n_envs, cfg.render_n_envs)
        self.L: int = cfg.env_spacing
        self.ground_plane: bool = cfg.ground_plane
        self.dt = cfg.dt
        self.enable_rendering: bool = True
        self.device = device
        
        N = torch.ceil(torch.sqrt(torch.tensor(self.n_envs, device=self.device))).int()
        assert N * N >= self.n_envs
        x = y = torch.arange(N, device=self.device, dtype=torch.float32) * self.L
        xy = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1).reshape(-1, 2)
        xy -= (N-1) * self.L / 2
        xyz = torch.cat([xy, torch.zeros_like(xy[:, :1])], dim=-1)
        self.env_origin = xyz[:self.n_envs]
        
        n_boxes_per_drone = 4 # use 4 boxes to represent a drone simply
        self.drone_mesh_dict = {
            "vertices":         ti.Vector.field(3, ti.f32, shape=(self.n_envs *  8*n_boxes_per_drone)),
            "indices":                 ti.field(   ti.i32, shape=(self.n_envs * 36*n_boxes_per_drone)),
            "per_vertex_color": ti.Vector.field(3, ti.f32, shape=(self.n_envs *  8*n_boxes_per_drone))
        }
        self.drone_vertices_tensor = torch.empty(self.n_envs, 32, 3, device=self.device)
        self._init_drone_model()
        
        if self.ground_plane:
            self.ground_plane_size: float = N.item() * self.L
            n_ground_faces = self.ground_plane_size * self.ground_plane_size
            n_ground_vertices = n_ground_faces * 4
            n_ground_indices = n_ground_faces * 6
            self.ground_plane_mesh_dict = {
                "vertices":         ti.Vector.field(3, ti.f32, shape=n_ground_vertices),
                "indices":                 ti.field(   ti.i32, shape=n_ground_indices),
                "per_vertex_color": ti.Vector.field(3, ti.f32, shape=n_ground_vertices)
            }
            z_ground_plane = -self.L - 0.1 if z_ground_plane is None else z_ground_plane
            self._init_ground_plane(z_ground_plane=z_ground_plane)
        
        self.gui_states = {
            "reset_all": False,
            "enable_lightsource": True,
            "brightness": 1.0,
            "display_basis": False,
            "display_groundplane": True,
        }
        self._init_viewer()
    
    def _init_viewer(self):
        end_points = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.axis_lines = [(ti.Vector.field(3, ti.f32, shape=2), tuple(end_points[i])) for i in range(3)]
        for i, end_point in enumerate(end_points):
            self.axis_lines[i][0].from_torch(torch2ti(torch.tensor([
                [0, 0, 0],
                end_point
            ], dtype=torch.float32, device=self.device)))
        
        self.gui_window = ti.ui.Window(name='Renderer Running at', res=(1280, 900), fps_limit=2*int(1/self.dt), pos=(150, 150))
        self.gui_handle = self.gui_window.get_gui()
        self.gui_scene = self.gui_window.get_scene()
        self.gui_camera = ti.ui.make_camera()
        env_bound = self.env_origin.min().item()
        self.gui_camera.position(env_bound-3*self.L, 5*self.L, env_bound-5*self.L)  # x, y, z
        self.gui_camera.lookat(0, -2*self.L, 0)
        self.gui_camera.up(0, 1, 0)
        self.gui_camera.projection_mode(ti.ui.ProjectionMode.Perspective)
    
    def _init_ground_plane(self, z_ground_plane: float):
        for i, j in ti.ndrange(self.ground_plane_size, self.ground_plane_size):
            self.ground_plane_size = self.ground_plane_size
            idx = (i * self.ground_plane_size + j)
            vertex_base = idx * 4  # 4 vertices per face

            x0 = i - self.ground_plane_size / 2
            z0 = j - self.ground_plane_size / 2
            x1 = x0 + 1
            z1 = z0 + 1

            # coordinates of the 4 vertices of the face
            self.ground_plane_mesh_dict["vertices"][vertex_base + 0] = ti.Vector([x0, z_ground_plane, z0])
            self.ground_plane_mesh_dict["vertices"][vertex_base + 1] = ti.Vector([x1, z_ground_plane, z0])
            self.ground_plane_mesh_dict["vertices"][vertex_base + 2] = ti.Vector([x0, z_ground_plane, z1])
            self.ground_plane_mesh_dict["vertices"][vertex_base + 3] = ti.Vector([x1, z_ground_plane, z1])

            # assign color to the vertices of the face
            color = 0.8 if (i + j) % 2 == 0 else 0.2
            for k in range(4):
                self.ground_plane_mesh_dict["per_vertex_color"][vertex_base + k] = ti.Vector([color, color, color])

            # vertex indices of the two triangles of the face
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
        l, L = 0.05, 0.2
        D = (L + l) / 2 / 2**0.5
        vertices_tensor, indices_tensor, color_tensor = add_box(
            xyz=torch.tensor([
                [ D,  D, 0],
                [-D,  D, 0],
                [ D, -D, 0],
                [-D, -D, 0]], device=self.device
            ).unsqueeze(0).expand(self.n_envs, -1, -1),
            lwh=torch.tensor([
                [L, l, l],
                [l, L, l],
                [l, L, l],
                [L, l, l]], device=self.device
            ).unsqueeze(0).expand(self.n_envs, -1, -1),
            rpy=torch.tensor([
                [0, 0, torch.pi/4]
            ], device=self.device).unsqueeze(0).expand(self.n_envs, 4, -1),
            color=torch.tensor([
                [0.8867, 0.9219, 0.1641],
                [0.5156, 0.1016, 0.5391],
                [0.8867, 0.9219, 0.1641],
                [0.5156, 0.1016, 0.5391]], device=self.device
            ).unsqueeze(0).expand(self.n_envs, -1, -1)
        )
        self.drone_vertices_tensor.copy_(vertices_tensor.reshape(self.n_envs, -1, 3))
        self.drone_mesh_dict["vertices"].from_torch(torch2ti(self.drone_vertices_tensor.flatten(end_dim=-2)))
        self.drone_mesh_dict["indices"].from_torch(indices_tensor.flatten())
        self.drone_mesh_dict["per_vertex_color"].from_torch(color_tensor.flatten(end_dim=-2))
    
    def _update_drone_model(
        self,
        pos: Tensor,      # [n_envs, 3]
        quat_xyzw: Tensor # [n_envs, 4]
    ):
        rotation_matrix = T.quaternion_to_matrix(quat_xyzw.roll(1, dims=-1))
        drone_vertices_tensor = torch.bmm(self.drone_vertices_tensor, rotation_matrix.transpose(-2, -1))
        drone_vertices_tensor = drone_vertices_tensor + pos.unsqueeze(-2) + self.env_origin.unsqueeze(-2)
        self.drone_mesh_dict["vertices"].from_torch(torch2ti(drone_vertices_tensor.flatten(end_dim=-2)))
    
    def step(self, state: Tensor):
        return NotImplementedError
    
    def _render_subwindow(self):
        self.gui_states["reset_all"] = False
        with self.gui_handle.sub_window("Simulation Settings", x=0.02, y=0.02, height=0.1, width=0.25) as sub_window:
            self.gui_states["reset_all"] = sub_window.button("Reset All")
            if sub_window.button("Exit"): raise KeyboardInterrupt
        
        color = (0, 0, 1)
        with self.gui_handle.sub_window("Render Settings", x=0.02, y=0.14, height=0.25, width=0.25) as sub_window:
            if sub_window.button("Pause Rendering"):
                self.enable_rendering = False
                sub_window.text("Rendering paused. Press \"V\" to resume.")
                tqdm.write("Rendering paused. Press \"V\" to resume.")
            self.gui_states["enable_lightsource"] = sub_window.checkbox("Enable Light Source", self.gui_states["enable_lightsource"])
            if self.gui_states["enable_lightsource"]:
                self.gui_states["brightness"] = sub_window.slider_float("Brightness", self.gui_states["brightness"], minimum=0, maximum=1)
            self.gui_states["display_basis"] = sub_window.checkbox("Display Axis Basis", self.gui_states["display_basis"])
            if self.ground_plane:
                self.gui_states["display_groundplane"] = sub_window.checkbox("Display Ground Plane", self.gui_states["display_groundplane"])
            # color = sub_window.color_edit_3("name2", color)
    
    def render(self):
        if self.enable_rendering:
            
            self.gui_camera.track_user_inputs(
                self.gui_window,
                movement_speed=0.1,
                pitch_speed=4,
                yaw_speed=4,
                hold_key=ti.ui.RMB)
            
            self._render_subwindow()
        
            # render ground plane
            if self.ground_plane and self.gui_states["display_groundplane"]:
                self.gui_scene.mesh(**self.ground_plane_mesh_dict)
            
            # render drones
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
            canvas = self.gui_window.get_canvas()
            canvas.scene(self.gui_scene)
            self.gui_window.show()
        
        if self.gui_window.get_event(ti.ui.PRESS):
            if self.gui_window.event.key == 'v':
                self.enable_rendering = not self.enable_rendering
                if not self.enable_rendering:
                    tqdm.write("Rendering paused. Press \"V\" to resume.")
            if self.gui_window.event.key == 'r':
                self.gui_states["reset_all"] = True
        if self.gui_window.is_pressed(ti.ui.ESCAPE):
            raise KeyboardInterrupt
    
    def _render_lines(self):
        if self.gui_states["display_basis"]:
            for line, color in self.axis_lines:
                self.gui_scene.lines(line, color=color, width=5.0)
    
    def _render_obstacles(self):
        pass


class PositionControlRenderer(BaseRenderer):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
    
    def step(self, drone_pos: Tensor, drone_quat_xyzw: Tensor):
        if self.enable_rendering:
            self._update_drone_model(drone_pos[:self.n_envs], drone_quat_xyzw[:self.n_envs])

class ObstacleAvoidanceRenderer(BaseRenderer):
    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device,
        obstacle_manager: ObstacleManager,
        z_ground_plane: float
    ):
        super().__init__(cfg, device, z_ground_plane=z_ground_plane)
        self.obstacle_manager = obstacle_manager
        self.cube_color = [0.8, 0.3, 0.1]
        self.sphere_color = [0.8, 0.1, 0.3]
        
        self.n_cubes = self.obstacle_manager.n_cubes
        self.cube_mesh_dict = {
            "vertices":          ti.Vector.field(3, ti.f32, shape=(self.n_envs * self.n_cubes *  8)),
            "indices":                  ti.field(   ti.i32, shape=(self.n_envs * self.n_cubes * 36)),
            "per_vertex_color":  ti.Vector.field(3, ti.f32, shape=(self.n_envs * self.n_cubes *  8))
        }
        self.cube_vertices_tensor = torch.empty(self.n_envs, self.n_cubes, 8, 3, device=self.device)
        
        self.n_spheres = self.obstacle_manager.n_spheres
        self.sphere_mesh_dict = {
            "centers":           ti.Vector.field(3, ti.f32, shape=(self.n_envs * self.n_spheres)),
            "per_vertex_radius":        ti.field(   ti.f32, shape=(self.n_envs * self.n_spheres)),
            "radius": 0.,
            "color": tuple(self.sphere_color)
        }
        self.sphere_center_tensor = torch.empty(self.n_envs, self.n_spheres, 3, device=self.device)
        
        self._init_obstacles()
        
        self.target_line_vertices = ti.Vector.field(3, ti.f32, shape=(self.n_envs * 2))
        self.target_line_colors   = ti.Vector.field(3, ti.f32, shape=(self.n_envs * 2))
        self.target_line_color_tensor = torch.empty(self.n_envs, 3, device=self.device)
    
    def _init_obstacles(self):
        vertices_tensor, indices_tensor, color_tensor = add_box(
            xyz=torch.zeros(self.n_envs, self.n_cubes, 3, device=self.device),
            lwh=self.obstacle_manager.lwh_cubes[:self.n_envs],
            rpy=torch.zeros(self.n_envs, self.n_cubes, 3, device=self.device),
            color=torch.tensor([[self.cube_color]], device=self.device).expand(self.n_envs, self.n_cubes, -1)
        )
        self.cube_vertices_tensor.copy_(vertices_tensor)
        self.cube_mesh_dict["vertices"].from_torch(torch2ti(self.cube_vertices_tensor.flatten(end_dim=-2)))
        self.cube_mesh_dict["indices"].from_torch(indices_tensor.flatten())
        self.cube_mesh_dict["per_vertex_color"].from_torch(color_tensor.flatten(end_dim=-2))
        
        self.sphere_center_tensor.copy_(self.obstacle_manager.p_spheres[:self.n_envs])
        self.sphere_mesh_dict["centers"].from_torch(torch2ti(self.sphere_center_tensor.flatten(end_dim=-2)))
        self.sphere_mesh_dict["per_vertex_radius"].from_torch(self.obstacle_manager.r_spheres[:self.n_envs].flatten())
    
    def step(self, drone_pos: Tensor, drone_quat_xyzw: Tensor, target_pos: Tensor):
        if self.enable_rendering:
            self._update_drone_model(drone_pos[:self.n_envs], drone_quat_xyzw[:self.n_envs])
            self._update_obstacles()
            self._update_lines(drone_pos[:self.n_envs], target_pos[:self.n_envs])
    
    def _update_lines(self, drone_pos: Tensor, target_pos: Tensor):
        target_line_vertices_tensor = torch.stack([drone_pos, target_pos], dim=-2) + self.env_origin.unsqueeze(-2)
        self.target_line_vertices.from_torch(torch2ti(target_line_vertices_tensor.flatten(end_dim=-2)))
        factory_kwargs = {"dtype": torch.float32, "device": self.device}
        white = torch.tensor([[1., 1., 1.]], **factory_kwargs).expand_as(self.target_line_color_tensor)
        red = torch.tensor([[1., 0., 0.]], **factory_kwargs).expand_as(self.target_line_color_tensor)
        yellow = torch.tensor([[0.7, 0.7, 0.2]], **factory_kwargs).expand_as(self.target_line_color_tensor)
        green = torch.tensor([[0., 1., 0.]], **factory_kwargs).expand_as(self.target_line_color_tensor)
        near_target = (drone_pos-target_pos).norm(dim=-1).lt(0.5).unsqueeze(-1).expand(-1, 3)
        self.target_line_color_tensor = torch.where(near_target, green, white)
        target_line_color_tensor = self.target_line_color_tensor.unsqueeze(-2).expand(-1, 2, -1)
        self.target_line_colors.from_torch(target_line_color_tensor.flatten(end_dim=-2))
    
    def _update_obstacles(self):
        cube_vertices_tensor = (
            self.cube_vertices_tensor + 
            self.obstacle_manager.p_cubes[:self.n_envs].unsqueeze(-2) + 
            self.env_origin.unsqueeze(-2).unsqueeze(-2))
        self.cube_mesh_dict["vertices"].from_torch(torch2ti(cube_vertices_tensor.flatten(end_dim=-2)))
        
        self.sphere_center_tensor.copy_(self.obstacle_manager.p_spheres[:self.n_envs])
        sphere_center_tensor = self.sphere_center_tensor + self.env_origin.unsqueeze(-2)
        self.sphere_mesh_dict["centers"].from_torch(torch2ti(sphere_center_tensor.flatten(end_dim=-2)))
    
    def _render_obstacles(self):
        self.gui_scene.mesh(**self.cube_mesh_dict)
        self.gui_scene.particles(**self.sphere_mesh_dict)
    
    def _render_lines(self):
        super()._render_lines()
        self.gui_scene.lines(self.target_line_vertices, per_vertex_color=self.target_line_colors, width=3.0)