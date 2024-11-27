import warnings

import torch
from torch import Tensor
from torch.nn import functional as F
from pytorch3d import transforms as T
from omegaconf import DictConfig

from quaddif.utils.math import *

class PointMassModel:
    def __init__(
        self,
        cfg: DictConfig,
        n_envs: int,
        dt: float,
        n_substeps: int,
        device: torch.device
    ):
        self.device = device
        self.state_dim = 9
        self.action_dim = 3
        self._state = torch.zeros(n_envs, self.state_dim, device=device)
        self._vel_ema = torch.zeros(n_envs, 3, device=device)
        self.vel_ema_factor = cfg.vel_ema_factor
        self.dt = dt
        self.n_substeps = n_substeps
        self.align_yaw_with_vel_direction = cfg.align_yaw_with_vel_direction
        self.aligh_yaw_with_vel_ema = cfg.aligh_yaw_with_vel_ema
        self.min_action = torch.tensor([list(cfg.min_action)], device=device)
        self.max_action = torch.tensor([list(cfg.max_action)], device=device)
        
        if cfg.solver_type == "euler":
            self.solver = EulerIntegral
        elif cfg.solver_type == "rk4":
            self.solver = rk4
        
        wrap = lambda x: torch.tensor(x, device=device, dtype=torch.float32)
        
        self._G = wrap(cfg.g)
        self._G_vec = wrap([0.0, 0.0, self._G])
        self.lmbda = cfg.lmbda # soft control latency
    
    def detach(self):
        self._state = self._state.detach()
        self._vel_ema = self._vel_ema.detach()
    
    def dynamics(self, X: Tensor, U: Tensor) -> Tensor:
        # Unpacking state and input variables
        p, v, a = X[:, :3], X[:, 3:6], X[:, 6:9]
        
        a_dot = self.lmbda * (U - a) / self.dt
        
        # State derivatives
        X_dot = torch.concat([v, a - self._G_vec, a_dot], dim=-1)
        
        return X_dot

    def step(self, U: Tensor) -> None:
        new_state = self.solver(self.dynamics, self._state, U, dt=self.dt, M=self.n_substeps)
        self._state = new_state
        self._vel_ema = torch.lerp(self._vel_ema, self._v, self.vel_ema_factor)
    
    @property
    def p(self) -> Tensor: return self._state[:, 0:3].detach()
    @property
    def v(self) -> Tensor: return self._state[:, 3:6].detach()
    @property
    def a(self) -> Tensor: return self._state[:, 6:9].detach()
    @property
    def q(self) -> Tensor:
        orientation = self._vel_ema.detach() if self.aligh_yaw_with_vel_ema else self.v
        return point_mass_quat(self.a, orientation=orientation)
    @property
    def w(self) -> Tensor:
        warnings.warn("Access of angular velocity in point mass model is not supported. Returning zero tensor instead.")
        return torch.zeros_like(self.p)
    @property
    def _p(self) -> Tensor: return self._state[:, 0:3]
    @property
    def _v(self) -> Tensor: return self._state[:, 3:6]
    @property
    def _a(self) -> Tensor: return self._state[:, 6:9]
    @property
    def _q(self) -> Tensor:
        warnings.warn("Direct access of quaternion with gradient in point mass model is strongly not recommanded. Please consider using the detached version instead.")
        orientation = self._vel_ema if self.aligh_yaw_with_vel_ema else self._v
        return point_mass_quat(self._a, orientation=orientation)
    @property
    def _w(self) -> Tensor:
        warnings.warn("Access of angular velocity with gradient in point mass model is not supported. Returning zero tensor instead.")
        return torch.zeros_like(self.p)

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
    yaw = torch.atan2(orientation[:, 1], orientation[:, 0])
    mat_yaw = axis_rotmat("Z", yaw)
    new_up = (mat_yaw.transpose(1, 2) @ up.unsqueeze(-1)).squeeze(-1)
    z = torch.zeros_like(new_up)
    z[..., -1] = 1.
    quat_axis = F.normalize(torch.cross(z, new_up, dim=-1), dim=-1)
    cos = torch.cosine_similarity(new_up, z, dim=-1)
    sin = torch.norm(new_up[:, :2], dim=-1) / (torch.norm(new_up, dim=-1) + 1e-7)
    quat_angle = torch.atan2(sin, cos)
    quat_pitch_roll_xyz = quat_axis * torch.sin(0.5 * quat_angle).unsqueeze(-1)
    quat_pitch_roll_w = torch.cos(0.5 * quat_angle).unsqueeze(-1)
    quat_pitch_roll = T.standardize_quaternion(torch.cat([quat_pitch_roll_w, quat_pitch_roll_xyz], dim=-1))
    yaw_2 = yaw.unsqueeze(-1) / 2
    quat_yaw = torch.concat([torch.cos(yaw_2), torch.sin(yaw_2) * z], dim=-1) # T.matrix_to_quaternion(mat_yaw)
    quat_wxyz = T.quaternion_multiply(quat_yaw, quat_pitch_roll)
    return quat_wxyz.roll(-1, dims=-1)