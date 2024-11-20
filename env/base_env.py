from typing import Tuple, Dict, Union

from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch3d import transforms as T

from quaddif.utils.math import axis_rotmat, rand_range

# @torch.jit.script
def point_mass_quat(a: Tensor, orientation: Tensor) -> Tensor:
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

class BaseEnv:
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.min_action = torch.tensor([list(cfg.min_action)], device=device)
        self.max_action = torch.tensor([list(cfg.max_action)], device=device)
        self._state = torch.zeros(cfg.n_envs, 9, device=device)
        self._vel_ema = torch.zeros(cfg.n_envs, 3, device=device)
        self.vel_ema_factor = cfg.vel_ema_factor
        self.dt = cfg.dt
        self.L = cfg.length
        self.n_envs = cfg.n_envs
        self.target_pos = torch.zeros(self.n_envs, 3, device=device)
        self.progress = torch.zeros(self.n_envs, device=device, dtype=torch.int)
        self.max_steps = cfg.max_time / cfg.dt
        self.max_vel = cfg.max_vel
        self.reset_indices = None
        self.align_yaw_with_vel_direction = cfg.align_yaw_with_vel_direction
        self.device = device
    
    def state(self, with_grad=False):
        raise NotImplementedError
    
    def detach(self):
        self._state = self._state.detach()
        self._vel_ema = self._vel_ema.detach()
    
    @property
    def p(self): return self._state[:, 0:3].detach()
    @property
    def v(self): return self._state[:, 3:6].detach()
    @property
    def a(self): return self._state[:, 6:9].detach()
    @property
    def _p(self): return self._state[:, 0:3]
    @property
    def _v(self): return self._state[:, 3:6]
    @property
    def _a(self): return self._state[:, 6:9]
    @property
    def q(self) -> Tensor:
        """Compute the drone pose using target direction and thrust acceleration direction.

        Returns:
            Tensor: attitude quaternion of the drone with real part last.
        """
        if self.align_yaw_with_vel_direction:
            return point_mass_quat(self.a, orientation=self.v)
        else:
            target_relpos = self.target_pos - self.p
            return point_mass_quat(self.a, orientation=target_relpos)
    @property
    def target_vel(self):
        target_relpos = self.target_pos - self.p
        target_dist = target_relpos.norm(dim=-1)
        return target_relpos / torch.max(target_dist / self.max_vel, torch.ones_like(target_dist)).unsqueeze(-1)

    def step(self, action):
        # type: (Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        raise NotImplementedError
    
    def state_for_render(self):
        # type: () -> Tensor
        raise NotImplementedError
    
    def loss_fn(self, target_vel, action):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Dict[str, float]]
        raise NotImplementedError

    def reset_idx(self, env_idx):
        # type: (Tensor) -> None
        raise NotImplementedError
    
    def reset(self):
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
    
    def terminated(self) -> torch.Tensor:
        raise NotImplementedError
    
    def truncated(self) -> torch.Tensor:
        return self.progress > self.max_steps
    
    def rescale_action(self, action: Tensor) -> Tensor:
        return self.min_action + (self.max_action - self.min_action) * (action + 1) / 2