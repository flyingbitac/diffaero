from typing import Optional, Tuple, List
import sys
import os
from collections import defaultdict

import torch
from isaacgym import gymapi, gymtorch
from omegaconf import DictConfig
from tqdm import tqdm

from quaddif import QUADDIF_ROOT_DIR
from quaddif.utils.assets import ObstacleManager, create_ball, create_cube

class BaseRenderer:
    def __init__(self, cfg: DictConfig, device):
        """Initialize the simulation."""
        self.cfg = cfg
        self.gym = gymapi.acquire_gym()
        # create gymapi.simParams struct and fill its attributes
        self.sim_params = get_sim_params(cfg.sim)
        self.dt = self.sim_params.dt
        
        physics_engines = {'physx': gymapi.SIM_PHYSX, 'flex': gymapi.SIM_FLEX}
        assert cfg.physics_engine.lower() in physics_engines.keys(), "Invalid physics engine"
        self.physics_engine = physics_engines[cfg.physics_engine.lower()]
        
        self.headless = cfg.headless
        self.n_envs = cfg.n_envs

        self.sim_device_id = device
        # graphics device for rendering viewer and cameras, -1 for no rendering
        enable_camera_sensors = 'camera' in dict(cfg).keys()
        if self.headless and not enable_camera_sensors:
            self.graphics_device_id = -1
        else:
            self.graphics_device_id = device
        self.device = torch.device(f"cuda:{device}")
        
        # create simulation handle
        self.actor_handles = []
        self.env_handles = []
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        
        # create simulation environment
        self.create_envs()
        
        # prepare simulation handle ready to run
        self.gym.prepare_sim(self.sim)

        # allocate buffers        
        self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self._root_tensor).view(self.n_envs, -1, 13)
        self.drone_states = self.root_states[:, 0]
        self.env_assets_states = self.root_states[:, 1:]

        self.drone_positions = self.drone_states[...,  0: 3]
        self.drone_quats     = self.drone_states[...,  3: 7]
        self.drone_linvels   = self.drone_states[...,  7:10]
        self.drone_angvels   = self.drone_states[..., 10:13]
        
        self._contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces: torch.Tensor = gymtorch.wrap_tensor(self._contact_force_tensor).view(self.n_envs, -1, 3)
        # self.drone_contact_forces = self.contact_forces[:, :self.robot_num_bodies].sum(dim=1)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        robot_body_property = self.gym.get_actor_rigid_body_properties(self.env_handles[0], self.actor_handles[0])
        # calculate mass and inertia
        robot_mass = sum([prop.mass for prop in robot_body_property])
        # ignore the four propellers when calculating inertia matrix
        self.mass_inertia = torch.tensor([
            robot_mass,
            robot_body_property[0].inertia.x.x,
            robot_body_property[0].inertia.y.y,
            robot_body_property[0].inertia.z.z
        ], device=self.device)
        print("Total robot mass: ", robot_mass, "kg")
        print("Inertia matrix: diag", self.mass_inertia[1:].cpu().numpy())
        print("Successfully created environment.")

        self.enable_viewer_sync = not self.headless
        self.viewer = None

        # create viewer
        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard events
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            cam_pos_x, cam_pos_y, cam_pos_z = self.cfg.viewer.pos
            cam_target_x, cam_target_y, cam_target_z = self.cfg.viewer.lookat
            cam_pos = gymapi.Vec3(cam_pos_x, cam_pos_y, cam_pos_z)
            cam_target = gymapi.Vec3(cam_target_x, cam_target_y, cam_target_z)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "exit")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "reset_all")

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
    
    def create_envs(self):
        """Override this function to create simulation environment."""
        raise NotImplementedError

    def create_ground_plane(self, height=0.):
        plane_params = gymapi.PlaneParams()
        plane_params.distance = height
        plane_params.dynamic_friction = 1.
        plane_params.static_friction = 1.
        plane_params.restitution = 0.8
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def reset_idx(self, env_ids: torch.Tensor):
        """Reset selected environments"""
        raise NotImplementedError
    
    def step(self, state: torch.Tensor):
        return NotImplementedError
    
    def simulation_step(self):
        self.gym.simulate(self.sim)
        # NOTE: as per the isaacgym docs, self.gym.fetch_results must be called after self.gym.simulate,
        # but not having it here seems to work fine it is called in the render function.
        # Fetch results
        self.gym.fetch_results(self.sim, True)
        # Copy the state tensors from physics engine to buffers in vram
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def render(self, sync_frame_time=True):
        # self.gym.fetch_results(self.sim, True) # use only when device is not "cpu"
        # Step graphics. Skipping this causes the onboard robot camera tensors to not be updated
        self.gym.step_graphics(self.sim)
        reset_all = False
        if self.headless: return reset_all
        
        # if viewer exists update it based on requirement
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                self.gym.destroy_viewer(self.viewer)
                self.gym.destroy_sim(self.sim)
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "exit" and evt.value > 0:
                    self.gym.destroy_viewer(self.viewer)
                    self.gym.destroy_sim(self.sim)
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "reset_all" and evt.value > 0:
                    reset_all = True

            # update viewer based on requirement
            if self.enable_viewer_sync:
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
        return reset_all
    
    def close(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


class PositionControlRenderer(BaseRenderer):
    def __init__(self, cfg: DictConfig, device):
        self.env_spacing = cfg.env_spacing
        super().__init__(cfg, device)
        self.target_positions = torch.zeros_like(self.drone_positions)
        self.total_dists = torch.zeros_like(self.drone_positions[..., 0])
        self.dist2target = torch.zeros(self.n_envs, device=self.device)
    
    def create_envs(self):
        print("Creating environment...")
        asset_path = os.path.join(QUADDIF_ROOT_DIR, self.cfg.robot_asset.file)
        drone_asset_path = os.path.join(QUADDIF_ROOT_DIR, self.cfg.robot_asset.file)
        print("Loading asset:", drone_asset_path)
        drone_asset_root = os.path.dirname(drone_asset_path)
        drone_asset_file = os.path.basename(drone_asset_path)

        drone_asset_options = get_asset_options(self.cfg.robot_asset)

        robot_asset = self.gym.load_asset(
            self.sim, drone_asset_root, drone_asset_file, drone_asset_options)

        start_pose = gymapi.Transform()
        pos = torch.tensor([0, 0, 0], device=self.device)
        start_pose.p = gymapi.Vec3(*pos)
        
        env_lower = gymapi.Vec3(
            -self.env_spacing,
            -self.env_spacing,
            -self.env_spacing)
        env_upper = gymapi.Vec3(
            self.env_spacing,
            self.env_spacing,
            self.env_spacing)
        
        pbar = tqdm(range(self.n_envs), unit="env")
        for i in pbar:
            pbar.set_description(f"Creating env {i+1}")
            
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(self.n_envs ** 0.5))
            self.env_handles.append(env_handle)
            
            # create drone instance
            actor_handle = self.gym.create_actor(
                env_handle, # env
                robot_asset, # asset
                start_pose, # pose
                self.cfg.robot_asset.name, # name
                i, # group
                self.cfg.robot_asset.collision_mask, # filter
                0 # segmentation ID
            )
            self.actor_handles.append(actor_handle)
            
    def step(self, drone_state: torch.Tensor):
        if self.enable_viewer_sync:
            self.drone_states.copy_(drone_state)
            self.gym.set_actor_root_state_tensor(self.sim, self._root_tensor)
            self.simulation_step()


class ObstacleAvoidanceRenderer(BaseRenderer):
    def __init__(
        self,
        cfg: DictConfig,
        device: int,
        obstacle_manager: ObstacleManager,
        z_ground_plane: Optional[float] = None,
        enable_camera: bool = False
    ):
        self.env_spacing = cfg.env_spacing
        self.enable_camera = enable_camera
        if self.enable_camera:
            self.camera_cfg = cfg.camera
            self.camera_handles = []
            self.camera_tensor_list = []
        self.record_video = cfg.record_video
        if self.record_video:
            self.rgb_camera_cfg = cfg.rgb_camera
            self.rgb_camera_handles = []
            self.rgb_camera_tensor_list = []
        self.env_asset_handles = defaultdict(list)
        self.n_obstacles = cfg.env_asset.n_assets
        self.z_ground_plane = z_ground_plane
        self.obstacle_manager = obstacle_manager
        super().__init__(cfg, device)
        self.target_pos = torch.zeros_like(self.drone_positions)
        self.asset_positions = torch.empty(self.n_envs, self.n_obstacles, 3, device=self.device)
        self.asset_quats = torch.empty(self.n_envs, self.n_obstacles, 4, device=self.device)
        
    @staticmethod
    def generate_env_assets(r_spheres, lwh_cubes):
        # type: (torch.Tensor, torch.Tensor) -> List[str]
        Ls, Ws, Hs = lwh_cubes.unbind(dim=-1)
        selected_files = [create_ball(r.item()) for r in r_spheres] + \
                         [create_cube(l.item(), w.item(), h.item()) for l, w, h in zip(Ls, Ws, Hs)]
        return selected_files
    
    def create_envs(self):
        print("Creating environment...")
        if self.z_ground_plane is not None:
            self.create_ground_plane(-self.z_ground_plane)
            
        drone_asset_path = os.path.join(QUADDIF_ROOT_DIR, self.cfg.robot_asset.file)
        print("Loading asset:", drone_asset_path)
        drone_asset_root = os.path.dirname(drone_asset_path)
        drone_asset_file = os.path.basename(drone_asset_path)
        drone_asset_options = get_asset_options(self.cfg.robot_asset)
        robot_asset = self.gym.load_asset(
            self.sim, drone_asset_root, drone_asset_file, drone_asset_options)
        self.robot_num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.bodies_per_env = self.robot_num_bodies
        
        env_asset_options = get_asset_options(self.cfg.env_asset)

        start_pose = gymapi.Transform()
        pos = torch.tensor([0, 0, 0], device=self.device)
        start_pose.p = gymapi.Vec3(*pos)
        
        env_lower = gymapi.Vec3(
            -self.env_spacing,
            -self.env_spacing,
            -self.env_spacing)
        env_upper = gymapi.Vec3(
            self.env_spacing,
            self.env_spacing,
            self.env_spacing)
        
        if self.enable_camera:
            camera_props, local_transform = get_camera_properties(self.camera_cfg)
        if self.record_video:
            rgb_camera_props, rgb_local_transform = get_camera_properties(self.rgb_camera_cfg)
        
        pbar = tqdm(range(self.n_envs), unit="env")
        for i in pbar:
            pbar.set_description(f"Creating env {i+1}")
            
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(self.n_envs ** 0.5))
            self.env_handles.append(env_handle)
            
            # create drone instance
            actor_handle = self.gym.create_actor(
                env_handle, # env
                robot_asset, # asset
                start_pose, # pose
                self.cfg.robot_asset.name, # name
                i, # group
                self.cfg.robot_asset.collision_mask, # filter
                0 # segmentation ID
            )
            self.actor_handles.append(actor_handle)
            
            if self.enable_camera:
                # create camera
                cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                self.gym.attach_camera_to_body(cam_handle, env_handle, actor_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
                self.camera_handles.append(cam_handle)
                cam_type = gymapi.IMAGE_DEPTH # gymapi.IMAGE_DEPTH or gymapi.IMAGE_COLOR
                _camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, cam_handle, cam_type)
                camera_tensor: torch.Tensor = gymtorch.wrap_tensor(_camera_tensor)
                if camera_tensor.ndim < 3:
                    camera_tensor = camera_tensor.unsqueeze(0)
                self.camera_tensor_list.append(camera_tensor)
            
            if self.record_video:
                # create camera
                rgb_cam_handle = self.gym.create_camera_sensor(env_handle, rgb_camera_props)
                self.gym.attach_camera_to_body(rgb_cam_handle, env_handle, actor_handle, rgb_local_transform, gymapi.FOLLOW_TRANSFORM)
                self.rgb_camera_handles.append(rgb_cam_handle)
                cam_type = gymapi.IMAGE_COLOR # gymapi.IMAGE_DEPTH or gymapi.IMAGE_COLOR
                _rgb_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, rgb_cam_handle, cam_type)
                rgb_camera_tensor: torch.Tensor = gymtorch.wrap_tensor(_rgb_camera_tensor)
                self.rgb_camera_tensor_list.append(rgb_camera_tensor)
            
            # create environment assets
            env_asset_list = self.generate_env_assets(
                r_spheres=self.obstacle_manager.r_spheres[i],
                lwh_cubes=self.obstacle_manager.lwh_cubes[i])
            for j, path in enumerate(env_asset_list):
                env_asset_root = os.path.dirname(path)
                env_asset_file = os.path.basename(path)
                env_asset = self.gym.load_asset(self.sim, env_asset_root, env_asset_file, env_asset_options)
                if i == 0:
                    self.bodies_per_env += self.gym.get_asset_rigid_body_count(env_asset)
                env_asset_handle = self.gym.create_actor(
                    env_handle,
                    env_asset,
                    start_pose, 
                    "env_asset_"+str(i*self.n_obstacles+j),
                    i,
                    self.cfg.env_asset.collision_mask,
                    self.cfg.env_asset.segmentation_id
                )
                self.env_asset_handles[i].append(env_asset_handle)
                color = [1.0, 0.35, 0.35] if j < self.n_obstacles else [0.5] * 3
                self.gym.set_rigid_body_color(env_handle, env_asset_handle, 0, gymapi.MESH_VISUAL,
                    gymapi.Vec3(*color))
    
    def check_collisions(self):
        return self.contact_forces[:, :self.robot_num_bodies].abs().sum(dim=-1).norm(dim=-1) > 0.1
    
    def step(self, state: torch.Tensor, target_pos: torch.Tensor):
        if self.enable_viewer_sync or self.enable_camera or self.record_video:
            self.root_states.copy_(state)
            self.gym.set_actor_root_state_tensor(self.sim, self._root_tensor)
            self.target_pos.copy_(target_pos)
            self.simulation_step()
    
    def render_camera(self) -> torch.Tensor:
        if not self.enable_camera:
            raise ValueError("Camera is not initialized")
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        new_camera = 1 - (-torch.concat(self.camera_tensor_list, dim=0) / self.camera_cfg.far_plane).clamp(0, 1)
        # new_camera = torch.stack(self.camera_tensor_list, dim=0)[..., :3].permute(0, 3, 1, 2).float() / 255 # for rgb camera
        self.gym.end_access_image_tensors(self.sim)
        return new_camera
    
    def render_rgb_camera(self) -> torch.Tensor:
        if not self.record_video:
            raise ValueError("Camera is not initialized")
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        # new_camera = torch.stack(self.rgb_camera_tensor_list, dim=0)[..., :3].permute(0, 3, 1, 2).float() / 255 # for rgb camera
        new_camera = torch.stack(self.rgb_camera_tensor_list, dim=0)[..., :3].permute(0, 3, 1, 2) # for rgb camera
        self.gym.end_access_image_tensors(self.sim)
        return new_camera
    
    def render(self, add_lines=True, sync_frame_time=True):
        add_lines = add_lines and self.viewer is not None
        if add_lines:
            vel = torch.zeros(self.n_envs, 3, device=self.device)
            vel[:, 0] = 1
            # vel = self.drone_positions + T.quaternion_apply(self.drone_quats.roll(1, dims=-1), vel)
            vel = self.drone_positions + self.drone_linvels
            vel = torch.concat([self.drone_positions, vel], dim=-1).cpu().numpy()
            lines = torch.concat(
                [self.drone_positions, self.target_pos], dim=-1).cpu().numpy()
            factory_kwargs = {"dtype": torch.float32, "device": self.device}
            colors = torch.zeros(self.n_envs, 3, **factory_kwargs)
            white = torch.tensor([[1., 1., 1.]], **factory_kwargs)
            red = torch.tensor([[1., 0., 0.]], **factory_kwargs)
            yellow = torch.tensor([[0.7, 0.7, 0.2]], **factory_kwargs)
            green = torch.tensor([[0., 1., 0.]], **factory_kwargs)
            colors[:] = white
            colors[(self.drone_positions-self.target_pos).norm(dim=-1)<0.5] = green
            for i, env in enumerate(self.env_handles):
                self.gym.add_lines(self.viewer, env, 1, lines[i:i+1].T, colors[i:i+1].T.cpu().numpy())
                self.gym.add_lines(self.viewer, env, 1, vel[i:i+1].T, yellow.T.cpu().numpy())
        reset_all = super().render(sync_frame_time=sync_frame_time)
        if add_lines:
            self.gym.clear_lines(self.viewer)
        return reset_all


def get_sim_params(sim_cfg):
    sim_params = gymapi.SimParams()
    sim_params.substeps = sim_cfg.substeps
    up_axises = {0: gymapi.UP_AXIS_Y, 1: gymapi.UP_AXIS_Z}
    sim_params.up_axis = up_axises[sim_cfg.up_axis]
    sim_params.gravity = gymapi.Vec3(
        sim_cfg.gravity[0],
        sim_cfg.gravity[1],
        sim_cfg.gravity[2]
    )
    sim_params.use_gpu_pipeline = sim_cfg.use_gpu_pipeline
    
    exclude_keys = []
    
    physx_param = sim_cfg.physx
    for k, v in dict(physx_param).items():
        if hasattr(sim_params.physx, k) and v is not None:
            setattr(sim_params.physx, k, v)
        elif k not in exclude_keys:
            print(f'\033[31mWarning: {k} is not a valid physx param.\033[0m')
    return sim_params

def get_asset_options(asset_cfg):
    asset_options = gymapi.AssetOptions()
    exclude_keys = [
        'file', 'name', 'base_link_name', 'foot_name', 'penalize_contacts_on',
        'terminate_after_contacts_on', 'collision_mask', 'assets_per_env', 'segmentation_id',
        'walls', 'ground_plane', 'n_assets']
    for k, v in dict(asset_cfg).items():
        if hasattr(asset_options, k) and v is not None:
            setattr(asset_options, k, v)
        elif k not in exclude_keys:
            print(f'\033[31mWarning: {k} is not a valid asset param.\033[0m')
    return asset_options

def get_camera_properties(camera_cfg):
    # Set Camera Properties
    camera_props = gymapi.CameraProperties()
    exclude_keys = ['max_dist', 'type', 'name', 'onboard_position', 'onboard_attitude']
    for k, v in dict(camera_cfg).items():
        if hasattr(camera_props, k) and v is not None:
            setattr(camera_props, k, v)
        elif k not in exclude_keys:
            print(f'\033[31mWarning: {k} is not a valid asset param.\033[0m')
    # local camera transform
    local_transform = gymapi.Transform()
    # position of the camera relative to the body
    local_transform.p = gymapi.Vec3(*camera_cfg.onboard_position)
    # orientation of the camera relative to the body
    local_transform.r = gymapi.Quat(*camera_cfg.onboard_attitude)
    return camera_props, local_transform