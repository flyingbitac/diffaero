from typing import Tuple, Dict, Union

from omegaconf import DictConfig
import torch
from torch import Tensor

from quaddif.model.quad import PointMassModel, QuadrotorModel, point_mass_quat

class BaseEnv:
    def __init__(self, env_cfg: DictConfig, model_cfg: DictConfig, device: torch.device):
        self.dynamic_type = model_cfg.name
        assert self.dynamic_type in ["pointmass", "quadrotor"]
        self.model: Union[PointMassModel, QuadrotorModel] = {
            "pointmass": PointMassModel,
            "quadrotor": QuadrotorModel
        }[self.dynamic_type](model_cfg, env_cfg.n_envs, env_cfg.dt, env_cfg.n_substeps, device)
        self.dt: float = env_cfg.dt
        self.L: float = env_cfg.length
        self.n_envs: int = env_cfg.n_envs
        self.target_pos = torch.zeros(self.n_envs, 3, device=device)
        self.progress = torch.zeros(self.n_envs, device=device, dtype=torch.int)
        self.max_steps: float = env_cfg.max_time / env_cfg.dt
        self.max_vel: float = env_cfg.max_vel
        self.reset_indices = None
        self.device = device
    
    def state(self, with_grad=False):
        raise NotImplementedError
    
    def detach(self):
        self.model.detach()
    
    @property
    def p(self): return self.model.p
    @property
    def v(self): return self.model.v
    @property
    def a(self): return self.model.a
    @property
    def w(self): return self.model.w
    @property
    def q(self) -> Tensor:
        if isinstance(self.model, PointMassModel): # Ugly implementation of quaternion for point mass
            if self.model.align_yaw_with_vel_direction:
                return self.model.q
            else:
                target_relpos = self.target_pos - self.p
                return point_mass_quat(self.a, orientation=target_relpos)
        else:
            return self.model.q
    @property
    def _p(self): return self.model._p
    @property
    def _v(self): return self.model._v
    @property
    def _a(self): return self.model._a
    @property
    def _w(self): return self.model._w
    @property
    def _q(self): return self.model._q
    
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

    def reset_idx(self, env_idx: Tensor):
        raise NotImplementedError
    
    def reset(self):
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
    
    def terminated(self) -> Tensor:
        raise NotImplementedError
    
    def truncated(self) -> Tensor:
        return self.progress > self.max_steps
    
    def rescale_action(self, action: Tensor) -> Tensor:
        return self.model.min_action + (self.model.max_action - self.model.min_action) * (action + 1) / 2