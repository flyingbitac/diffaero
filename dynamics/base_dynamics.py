from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.autograd as autograd
from omegaconf import DictConfig

from quaddif.utils.math import quat_rotate, quat_rotate_inverse

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
        self._state = self._state.detach()
    
    def grad_decay(self, state: Tensor) -> Tensor:
        if self.alpha > 0:
            state = GradientDecay.apply(state, self.alpha, self.dt)
        return state

    @property
    @abstractmethod
    def min_action(self) -> Tensor: raise NotImplementedError
    
    @property
    @abstractmethod
    def max_action(self) -> Tensor: raise NotImplementedError
    
    @property
    @abstractmethod
    def _p(self) -> Tensor: raise NotImplementedError
    
    @property
    @abstractmethod
    def _v(self) -> Tensor: raise NotImplementedError
    
    @property
    @abstractmethod
    def _a(self) -> Tensor: raise NotImplementedError
    
    @property
    @abstractmethod
    def _w(self) -> Tensor: raise NotImplementedError
    
    @property
    @abstractmethod
    def _q(self) -> Tensor: raise NotImplementedError
    
    @property
    def p(self) -> Tensor: return self._p.detach()
    @property
    def v(self) -> Tensor: return self._v.detach()
    @property
    def a(self) -> Tensor: return self._a.detach()
    @property
    def w(self) -> Tensor: return self._w.detach()
    @property
    def q(self) -> Tensor: return self._q.detach()
    
    @abstractmethod
    def step(self, U: Tensor) -> None:
        """Step the model with the given action U.

        Args:
            U (Tensor): The action tensor of shape (n_envs, n_agents, 3).
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
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