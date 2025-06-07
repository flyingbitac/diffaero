import warnings

import torch
from torch import Tensor
import torch.autograd as autograd
from torch.nn import functional as F
from pytorch3d import transforms as T
from omegaconf import DictConfig

from quaddif.utils.math import *
from quaddif.utils.randomizer import build_randomizer

class GradientDecay(autograd.Function):
    @staticmethod
    def forward(ctx, state: Tensor, alpha: float, dt: float):
        ctx.save_for_backward(torch.tensor(-alpha * dt, device=state.device).exp())
        return state
    
    @staticmethod
    def backward(ctx, grad_state: Tensor):
        decay_factor = ctx.saved_tensors[0]
        if ctx.needs_input_grad[0]:
            grad_state = grad_state * decay_factor
        return grad_state, None, None


class PointMassModelBase:
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.type = "pointmass"
        self.device = device
        self.state_dim = 9
        self.action_dim = 3
        self.n_agents: int = cfg.n_agents
        self.n_envs: int = cfg.n_envs
        self._state = torch.zeros(self.n_envs, self.n_agents, self.state_dim, device=device)
        self._vel_ema = torch.zeros(self.n_envs, self.n_agents, 3, device=device)
        if self.n_agents == 1:
            self._state.squeeze_(1)
            self._vel_ema.squeeze_(1)
        self.dt: float = cfg.dt
        self.alpha: float = cfg.alpha
        self.align_yaw_with_target_direction: bool = cfg.align_yaw_with_target_direction
        self.align_yaw_with_vel_ema: bool = cfg.align_yaw_with_vel_ema
        self._G = torch.tensor(cfg.g, device=device, dtype=torch.float32)
        self._G_vec = torch.tensor([0.0, 0.0, -self._G], device=device, dtype=torch.float32)
        if self.n_agents > 1:
            self._G_vec.unsqueeze_(0)
    
        # self.vel_ema_factor: float = cfg.vel_ema_factor
        self.vel_ema_factor = build_randomizer(cfg.vel_ema_factor, [self.n_envs, self.n_agents, 1], device=device)
        # self._D = torch.tensor(cfg.D, device=device, dtype=torch.float32)
        self._D = build_randomizer(cfg.D, [self.n_envs, self.n_agents, 1], device=device)
        # self.lmbda: float = cfg.lmbda # soft control latency
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
        self._state = self._state.detach()
        self._vel_ema.detach_()
    
    def reset_idx(self, env_idx: Tensor) -> None:
        mask = torch.zeros_like(self._vel_ema, dtype=torch.bool)
        mask[env_idx] = True
        self._vel_ema = torch.where(mask, 0., self._vel_ema)
    
    def grad_decay(self, state: Tensor) -> Tensor:
        if self.alpha > 0:
            state = GradientDecay.apply(state, self.alpha, self.dt)
        return state
    
    @property
    def p(self) -> Tensor: return self._state[..., 0:3].detach()
    @property
    def v(self) -> Tensor: return self._state[..., 3:6].detach()
    @property
    def a(self) -> Tensor: return self._state[..., 6:9].detach()
    @property
    def q(self) -> Tensor:
        orientation = self._vel_ema.detach() if self.align_yaw_with_vel_ema else self.v
        return point_mass_quat(self.a, orientation=orientation)
    @property
    def w(self) -> Tensor:
        warnings.warn("Access of angular velocity in point mass model is not supported. Returning zero tensor instead.")
        return torch.zeros_like(self.p)
    @property
    def _p(self) -> Tensor: return self._state[..., 0:3]
    @property
    def _v(self) -> Tensor: return self._state[..., 3:6]
    @property
    def _a(self) -> Tensor: return self._state[..., 6:9]
    @property
    def _q(self) -> Tensor:
        warnings.warn("Direct access of quaternion with gradient in point mass model is not supported. Returning detached version instead.")
        return self.q
    @property
    def _w(self) -> Tensor:
        warnings.warn("Access of angular velocity with gradient in point mass model is not supported. Returning zero tensor instead.")
        return torch.zeros_like(self.p)
    
    def step(self, U: Tensor) -> None:
        """Step the model with the given action U.

        Args:
            U (Tensor): The action tensor of shape (n_envs, n_agents, 3).
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


class ContinuousPointMassModel(PointMassModelBase):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        self.n_substeps: int = cfg.n_substeps
        assert cfg.solver_type in ["euler", "rk4"]
        if cfg.solver_type == "euler":
            self.solver = EulerIntegral
        elif cfg.solver_type == "rk4":
            self.solver = rk4
    
    def dynamics(self, X: Tensor, U: Tensor) -> Tensor:
        # Unpacking state and input variables
        p, v, a = X[..., :3], X[..., 3:6], X[..., 6:9]
        
        fdrag = -self._D.value * v
        v_dot = a + self._G_vec + fdrag
        
        a_dot = self.lmbda.value * (U - a)
        
        # State derivatives
        X_dot = torch.concat([v, v_dot, a_dot], dim=-1)
        
        return X_dot

    def step(self, U: Tensor) -> None:
        next_state = self.solver(self.dynamics, self._state, U, dt=self.dt, M=self.n_substeps)
        next_state = self.grad_decay(next_state)
        self._state = next_state
        self._vel_ema = torch.lerp(self._vel_ema, self._v, self.vel_ema_factor.value)


class DiscretePointMassModel(PointMassModelBase):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)

    def step(self, U: Tensor) -> None:
        p, v, a = self._state.chunk(3, dim=-1)
        
        next_p = p + self.dt * (v + 0.5 * (a + self._G_vec) * self.dt)
        next_a = torch.lerp(a, U, self.lmbda.value)
        next_v = v + self.dt * (0.5 * (a + next_a) + self._G_vec)
        next_state = torch.cat([next_p, next_v, next_a], dim=-1)
        next_state = self.grad_decay(next_state)
        self._state = next_state
        self._vel_ema = torch.lerp(self._vel_ema, self._v, self.vel_ema_factor.value)


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
