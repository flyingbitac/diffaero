import warnings

import torch
from torch import Tensor
from torch.nn import functional as F
from pytorch3d import transforms as T
from omegaconf import DictConfig

from diffaero.dynamics.base_dynamics import BaseDynamics
from diffaero.utils.math import EulerIntegral, rk4, axis_rotmat, mvp
from diffaero.utils.randomizer import build_randomizer

class PointMassModelBase(BaseDynamics):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        self.type = "pointmass"
        self.action_frame: str = cfg.action_frame
        assert self.action_frame in ["world", "local"], f"Invalid action frame: {self.action_frame}. Must be 'world' or 'local'."
        self.state_dim = 9
        self.action_dim = 3
        self._state = torch.zeros(self.n_envs, self.n_agents, self.state_dim, device=device)
        self._vel_ema = torch.zeros(self.n_envs, self.n_agents, 3, device=device)
        self._acc = torch.zeros(self.n_envs, self.n_agents, 3, device=device)
        xyz = torch.zeros(self.n_envs, self.n_agents, 3, device=device)
        w = torch.ones(self.n_envs, self.n_agents, 1, device=device)
        self.quat_xyzw = torch.cat([xyz, w], dim=-1)
        self.quat_xyzw_init = self.quat_xyzw.clone()
        if self.n_agents == 1:
            self._state.squeeze_(1)
            self._vel_ema.squeeze_(1)
            self._acc.squeeze_(1)
            self.quat_xyzw.squeeze_(1)
            self.quat_xyzw_init.squeeze_(1)
        self.align_yaw_with_target_direction: bool = cfg.align_yaw_with_target_direction
        self.align_yaw_with_vel_ema: bool = cfg.align_yaw_with_vel_ema
    
        self.vel_ema_factor = build_randomizer(cfg.vel_ema_factor, [self.n_envs, self.n_agents, 1], device=device)
        self._D = build_randomizer(cfg.D, [self.n_envs, self.n_agents, 1], device=device)
        self.lmbda = build_randomizer(cfg.lmbda, [self.n_envs, self.n_agents, 1], device=device)
        if self.n_agents == 1:
            self.vel_ema_factor.value.squeeze_(1)
            self._D.value.squeeze_(1)
            self.lmbda.value.squeeze_(1)
        self.max_acc_xy = build_randomizer(cfg.max_acc.xy, [self.n_envs, self.n_agents], device=device)
        self.max_acc_z = build_randomizer(cfg.max_acc.z, [self.n_envs, self.n_agents], device=device)
        
    @property
    def min_action(self) -> Tensor:
        zero = torch.zeros_like(self.max_acc_xy.value)
        min_action = torch.stack([-self.max_acc_xy.value, -self.max_acc_xy.value, zero], dim=-1)
        if self.n_agents == 1:
            min_action.squeeze_(1)
        return min_action

    @property
    def max_action(self) -> Tensor:
        max_action = torch.stack([self.max_acc_xy.value, self.max_acc_xy.value, self.max_acc_z.value], dim=-1)
        if self.n_agents == 1:
            max_action.squeeze_(1)
        return max_action
    
    def detach(self):
        super().detach()
        self._vel_ema.detach_()
        self._acc.detach_()
    
    def reset_idx(self, env_idx: Tensor) -> None:
        mask = torch.zeros(*self._vel_ema.shape[:-1], dtype=torch.bool, device=self.device)
        mask[env_idx] = True
        mask3 = mask.unsqueeze(-1).expand_as(self._vel_ema)
        self._vel_ema = torch.where(mask3, 0., self._vel_ema)
        self._acc = torch.where(mask3, 0., self._acc)
        mask4 = mask.unsqueeze(-1).expand_as(self.quat_xyzw)
        self.quat_xyzw = torch.where(mask4, self.quat_xyzw_init, self.quat_xyzw)
    
    @property
    def q(self) -> Tensor: return self.quat_xyzw
    @property
    def w(self) -> Tensor:
        warnings.warn("Access of angular velocity in point mass model is not supported. Returning zero tensor instead.")
        return torch.zeros_like(self.p)
    @property
    def _p(self) -> Tensor: return self._state[..., 0:3]
    @property
    def _v(self) -> Tensor: return self._state[..., 3:6]
    @property
    def _a(self) -> Tensor: return self._acc
    @property
    def _a_thrust(self) -> Tensor: return self._state[..., 6:9]
    @property
    def a_thrust(self) -> Tensor: return self._a_thrust.detach()
    @property
    def _q(self) -> Tensor:
        warnings.warn("Direct access of quaternion with gradient in point mass model is not supported. Returning detached version instead.")
        return self.q
    @property
    def _w(self) -> Tensor:
        warnings.warn("Access of angular velocity with gradient in point mass model is not supported. Returning zero tensor instead.")
        return torch.zeros_like(self.p)
    
    def update_state(self, next_state: Tensor) -> None:
        self._state = self.grad_decay(next_state)
        self._vel_ema = torch.lerp(self._vel_ema, self._v, self.vel_ema_factor.value)
        self._acc = self._a_thrust + self._G_vec - self._D.value * self._v
        with torch.no_grad():
            orientation = self._vel_ema if self.align_yaw_with_vel_ema else self.v
            self.quat_xyzw = point_mass_quat(self.a_thrust, orientation=orientation)

@torch.jit.script
def continuous_point_mass_dynamics_local(
    X: Tensor,
    U: Tensor,
    dt: float,
    Rz: Tensor,
    G_vec: Tensor,
    D: Tensor,
    lmbda: Tensor,
):
    """Dynamics function for continuous point mass model in local frame."""
    p, v, a_thrust = X[..., :3], X[..., 3:6], X[..., 6:9]
    p_dot = v
    fdrag = -D * v
    v_dot = a_thrust + G_vec + fdrag
    control_delay_factor = (1 - torch.exp(-lmbda * dt)) / dt
    a_thrust_cmd_local = U
    a_thrust_cmd = torch.matmul(Rz, a_thrust_cmd_local.unsqueeze(-1)).squeeze(-1)
    a_dot = control_delay_factor * (a_thrust_cmd - a_thrust)
    
    X_dot = torch.concat([p_dot, v_dot, a_dot], dim=-1)
    return X_dot

@torch.jit.script
def continuous_point_mass_dynamics_world(
    X: Tensor,
    U: Tensor,
    dt: float,
    G_vec: Tensor,
    D: Tensor,
    lmbda: Tensor,
):
    """Dynamics function for continuous point mass model in local frame."""
    p, v, a_thrust = X[..., :3], X[..., 3:6], X[..., 6:9]
    p_dot = v
    fdrag = -D * v
    v_dot = a_thrust + G_vec + fdrag
    control_delay_factor = (1 - torch.exp(-lmbda * dt)) / dt
    a_thrust_cmd = U
    a_dot = control_delay_factor * (a_thrust_cmd - a_thrust)
    
    X_dot = torch.concat([p_dot, v_dot, a_dot], dim=-1)
    return X_dot

class ContinuousPointMassModel(PointMassModelBase):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        self.n_substeps: int = cfg.n_substeps
        assert cfg.solver_type in ["euler", "rk4"]
        if cfg.solver_type == "euler":
            self.solver = EulerIntegral
        elif cfg.solver_type == "rk4":
            self.solver = rk4
        self.Rz_temp: Tensor
    
    def dynamics(self, X: Tensor, U: Tensor) -> Tensor:
        if self.action_frame == "local":
            X_dot = continuous_point_mass_dynamics_local(
                X, U, self.dt, self.Rz_temp, self._G_vec, self._D.value, self.lmbda.value
            )
        elif self.action_frame == "world":
            X_dot = continuous_point_mass_dynamics_world(
                X, U, self.dt, self._G_vec, self._D.value, self.lmbda.value
            )
        return X_dot

    def step(self, U: Tensor) -> None:
        if self.action_frame == "local":
            self.Rz_temp = self.Rz.clone()
        next_state = self.solver(self.dynamics, self._state, U, dt=self.dt, M=self.n_substeps)
        self.update_state(next_state)


@torch.jit.script
def discrete_point_mass_dynamics_local(
    X: Tensor,
    U: Tensor,
    dt: float,
    Rz: Tensor,
    G_vec: Tensor,
    D: Tensor,
    lmbda: Tensor,
):
    """Dynamics function for discrete point mass model in local frame."""
    p, v, a_thrust = X[..., :3], X[..., 3:6], X[..., 6:9]
    next_p = p + dt * (v + 0.5 * (a_thrust + G_vec) * dt)
    control_delay_factor = 1 - torch.exp(-lmbda*dt)
    a_thrust_cmd_local = U
    a_thrust_cmd = mvp(Rz, a_thrust_cmd_local)
    next_a = torch.lerp(a_thrust, a_thrust_cmd, control_delay_factor) - D * v
    next_v = v + dt * (0.5 * (a_thrust + next_a) + G_vec)
    
    next_state = torch.cat([next_p, next_v, next_a], dim=-1)
    return next_state

@torch.jit.script
def discrete_point_mass_dynamics_world(
    X: Tensor,
    U: Tensor,
    dt: float,
    G_vec: Tensor,
    D: Tensor,
    lmbda: Tensor,
):
    """Dynamics function for discrete point mass model in world frame."""
    p, v, a_thrust = X[..., :3], X[..., 3:6], X[..., 6:9]
    next_p = p + dt * (v + 0.5 * (a_thrust + G_vec) * dt)
    control_delay_factor = 1 - torch.exp(-lmbda*dt)
    a_thrust_cmd = U
    next_a = torch.lerp(a_thrust, a_thrust_cmd, control_delay_factor) - D * v
    next_v = v + dt * (0.5 * (a_thrust + next_a) + G_vec)
    
    next_state = torch.cat([next_p, next_v, next_a], dim=-1)
    return next_state

class DiscretePointMassModel(PointMassModelBase):

    def step(self, U: Tensor) -> None:
        if self.action_frame == "local":
            next_state = discrete_point_mass_dynamics_local(
                self._state, U, self.dt, self.Rz, self._G_vec, self._D.value, self.lmbda.value
            )
        elif self.action_frame == "world":
            next_state = discrete_point_mass_dynamics_world(
                self._state, U, self.dt, self._G_vec, self._D.value, self.lmbda.value
            )
        self.update_state(next_state)


@torch.jit.script
def point_mass_quat(a: Tensor, orientation: Tensor) -> Tensor:
    """Compute the drone pose using target direction and thrust acceleration direction.

    Args:
        a (Tensor): the acceleration of the drone in world frame.
        orientation (Tensor): at which direction(yaw) the drone should be facing.

    Returns:
        Tensor: attitude quaternion of the drone with real part last.
    """
    up: Tensor = F.normalize(a, dim=-1)
    yaw = torch.atan2(orientation[..., 1], orientation[..., 0])
    mat_yaw = axis_rotmat("Z", yaw)
    new_up = (mat_yaw.transpose(-2, -1) @ up.unsqueeze(-1)).squeeze(-1)
    z = torch.zeros_like(new_up)
    z[..., -1] = 1.
    quat_axis = F.normalize(torch.cross(z, new_up, dim=-1), dim=-1)
    cos = torch.cosine_similarity(new_up, z, dim=-1)
    sin = torch.norm(new_up[..., :2], dim=-1) / (torch.norm(new_up, dim=-1) + 1e-7)
    quat_angle = torch.atan2(sin, cos)
    quat_pitch_roll_xyz = quat_axis * torch.sin(0.5 * quat_angle).unsqueeze(-1)
    quat_pitch_roll_w = torch.cos(0.5 * quat_angle).unsqueeze(-1)
    quat_pitch_roll = T.standardize_quaternion(torch.cat([quat_pitch_roll_w, quat_pitch_roll_xyz], dim=-1))
    yaw_half = yaw.unsqueeze(-1) / 2
    quat_yaw = torch.concat([torch.cos(yaw_half), torch.sin(yaw_half) * z], dim=-1) # T.matrix_to_quaternion(mat_yaw)
    quat_wxyz = T.quaternion_multiply(quat_yaw, quat_pitch_roll)
    quat_xyzw = quat_wxyz.roll(-1, dims=-1)
    
    # ori = torch.stack([orientation[..., 0], orientation[..., 1], torch.zeros_like(orientation[..., 2])], dim=-1)
    # print(F.normalize(quaternion_apply(quaternion_invert(quat_yaw), ori), dim=-1)[..., 0]) # 1
    # assert torch.max(torch.abs(quaternion_apply(quat_wxyz, z) - up)) < 1e-6
    # assert torch.max(torch.abs(quaternion_apply(quaternion_invert(quat_wxyz), up) - z)) < 1e-6
    # assert torch.max(torch.abs(quaternion_apply(quat_pitch_roll, z) - new_up)) < 1e-6
    
    # mat = T.quaternion_to_matrix(quat_wxyz)
    # print(((mat @ z.unsqueeze(-1)).squeeze(-1) - up).norm(dim=-1).max())
    
    # euler = quaternion_to_euler(quat_xyzw)
    # mat_roll, mat_pitch, mat_yaw = axis_rotmat("X", euler[..., 0]), axis_rotmat("Y", euler[..., 1]), axis_rotmat("Z", euler[..., 2])
    # mat_rot = mat_roll @ mat_pitch @ mat_yaw
    # print((mat_rot @ z.unsqueeze(-1)).squeeze(-1) - up)
    
    return quat_xyzw
