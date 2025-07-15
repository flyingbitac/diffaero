from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.autograd as autograd
import torch.nn.functional as F
import pytorch3d.transforms as T
from omegaconf import DictConfig

from quaddif.utils.math import quat_rotate, quat_rotate_inverse, mvp, axis_rotmat, quaternion_to_euler
from quaddif.utils.logger import Logger

class BaseDynamics(ABC):
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.type: str
        self.state_dim: int
        self.action_dim: int
        self.device = device
        self.n_agents: int = cfg.n_agents
        self.n_envs: int = cfg.n_envs
        self.dt: float = cfg.dt
        self.alpha: float = cfg.alpha
        
        self._G = torch.tensor(cfg.g, device=device, dtype=torch.float32)
        self._G_vec = torch.tensor([0.0, 0.0, -self._G], device=device, dtype=torch.float32)
        if self.n_agents > 1:
            self._G_vec.unsqueeze_(0)

    def detach(self):
        """Detach the state to prevent backpropagation through released computation graphs."""
        self._state = self._state.detach()
    
    def grad_decay(self, state: Tensor) -> Tensor:
        if self.alpha > 0:
            state = GradientDecay.apply(state, self.alpha, self.dt)
        return state
    
    @abstractmethod
    def step(self, U: Tensor) -> None:
        """Step the model with the given action U.

        Args:
            U (Tensor): The action tensor of shape (n_envs, n_agents, 3).
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    # Action ranges
    @property
    @abstractmethod
    def min_action(self) -> Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def max_action(self) -> Tensor:
        raise NotImplementedError

    # Properties of agents, requires_grad=True if stepped with undetached action inputs
    @property
    @abstractmethod
    def _p(self) -> Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def _v(self) -> Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def _a(self) -> Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def _w(self) -> Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def _q(self) -> Tensor:
        """Quaternion representing the orientation of the body frame in world frame, with real part last."""
        raise NotImplementedError
    
    # Detached versions of properties
    @property
    def p(self) -> Tensor:
        return self._p.detach()

    @property
    def v(self) -> Tensor:
        return self._v.detach()

    @property
    def a(self) -> Tensor:
        return self._a.detach()

    @property
    def w(self) -> Tensor:
        return self._w.detach()

    @property
    def q(self) -> Tensor:
        return self._q.detach()
    
    # Rotation utilities
    @property
    def R(self) -> Tensor:
        "Rotation matrix with columns being coordinate of axis unit vectors of body frame in world frame."
        return T.quaternion_to_matrix(self.q.roll(1, -1))
    
    # Rotation utilities
    @property
    def Rz(self) -> Tensor:
        "Rotation matrix with columns being coordinate of axis unit vectors of local frame in world frame."
        # Rz = self.R.clone()
        # fwd = Rz[..., 0]
        # fwd[..., 2] = 0.
        # fwd = F.normalize(fwd, dim=-1)
        # up = torch.zeros_like(fwd)
        # up[..., 2] = 1.
        # left = torch.cross(up, fwd, dim=-1)
        # return torch.stack([fwd, left, up], dim=-1)
        return axis_rotmat("Z", quaternion_to_euler(self.q)[..., 2])
    
    @property
    def ux(self) -> Tensor:
        "Unit vector along the x-axis of the body frame in world frame."
        return self.R[..., 0]
    
    @property
    def uy(self) -> Tensor:
        "Unit vector along the y-axis of the body frame in world frame."
        return self.R[..., 1]
    
    @property
    def uz(self) -> Tensor:
        "Unit vector along the z-axis of the body frame in world frame."
        return self.R[..., 2]
    
    def world2body(self, vec_w: Tensor) -> Tensor:
        """
        Convert vector from world frame to body frame.
        Args:
            vec_w (Tensor): vector in world frame
        Returns:
            Tensor: vector in body frame
        """
        return quat_rotate_inverse(self.q, vec_w)
    
    def body2world(self, vec_b: Tensor) -> Tensor:
        """
        Convert vector from body frame to world frame.
        Args:
            vec_b (Tensor): vector in body frame
        Returns:
            Tensor: vector in world frame
        """
        return quat_rotate(self.q, vec_b)

    def world2local(self, vec_w: Tensor) -> Tensor:
        """
        Convert vector from world frame to local frame.
        Args:
            vec_w (Tensor): vector in world frame
        Returns:
            Tensor: vector in local frame
        """
        # Logger.debug(mvp(self.Rz.transpose(-1, -2), self.ux)[0][..., 1].cpu(), "should be around 0")
        return mvp(self.Rz.transpose(-1, -2), vec_w)
    
    def local2world(self, vec_l: Tensor) -> Tensor:
        """
        Convert vector from local frame to world frame.
        Args:
            vec_l (Tensor): vector in local frame
        Returns:
            Tensor: vector in world frame
        """
        return mvp(self.Rz, vec_l)

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