from typing import Tuple, Dict, Union

from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch3d import transforms as T

from quaddif.env.base_env import BaseEnv
from quaddif.model.quad import QuadrotorModel, PointMassModel
from quaddif.utils.render import PositionControlRenderer
from quaddif.utils.math import unitization, axis_rotmat, rand_range


class QuadrotorDynamicsPositionControl:
    def __init__(self, cfg, device):
        self.model = QuadrotorModel(cfg.quad, cfg.dt, cfg.n_substeps, device)
        self.state_dim = 13
        self.action_dim = 4
        self._state = torch.zeros(cfg.n_envs, self.state_dim, device=device)
        self.dt = cfg.dt
        self.L = cfg.length
        self.n_envs = cfg.n_envs
        self.target_pos = torch.zeros(self.n_envs, 3, device=device)
        self.progress = torch.zeros(self.n_envs, device=device, dtype=torch.int)
        self.max_steps = cfg.max_time / cfg.dt
        self.max_vel = cfg.max_vel
        self.reset_indices = None
        
        self.device = device
    
    def state(self, with_grad=False):
        state = [
            self.target_vel(),
            self._state[:, 3:]]
        state = torch.cat(state, dim=-1)
        return state if with_grad else state.detach()

    def debug(self):
        return self._state
    
    def cut_grad(self):
        self._state = self._state.detach()
    
    def p(self): return self._state[:,  0: 3].detach()
    def v(self): return self._state[:,  3: 6].detach()
    def q(self): return self._state[:,  6:10].detach()
    def w(self): return self._state[:, 10:13].detach()
        
    def step(self, action):
        # type: (Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        self._state = self.model(self._state, action)
        self.progress += 1
        terminated, truncated = self.terminated(), self.truncated()
        reset = terminated | truncated
        reset_indices = reset.nonzero().squeeze(-1)
        success = truncated & ((self.p - self.target_pos).norm(dim=-1) < 0.5)
        target_vel = self.target_vel()
        loss = self.loss_fn(target_vel)
        extra = {
            "terminated": terminated,
            "truncated": truncated,
            "reset": reset,
            "l": self.progress.clone(),
            "reset_indicies": reset_indices,
            "success": success,
            "state_before_reset": self.state(with_grad=True)
        }
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        return self.state(), loss, terminated, extra
    
    def target_vel(self):
        target_relpos = self.target_pos - self._state[:, :3].detach()
        target_dist = target_relpos.norm(dim=-1)
        return target_relpos / torch.max(target_dist / self.max_vel, torch.ones_like(target_dist)).unsqueeze(-1)
        
    def loss_fn(self, target_vel):
        vel_diff = (self._state[:, 3:6] - target_vel).norm(dim=-1)
        vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
        return vel_loss

    def reset_idx(self, env_idx):
        state_mask = torch.zeros_like(self._state)
        state_mask[env_idx] = 1
        p_new = (torch.rand_like(self.p) - 0.5) * 5
        v_new = torch.zeros_like(self.v)
        q_new = torch.zeros_like(self.q())
        q_new[:, 3] = 1
        w_new = torch.zeros_like(self.w())
        new_state = torch.cat([p_new, v_new, q_new, w_new], dim=-1)
        self._state = torch.where(state_mask.bool(), new_state, self._state)
        self.target_pos.fill_(0.)
        self.progress[env_idx] = 0
    
    def reset(self):
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
        return self.state()
    
    def terminated(self) -> torch.Tensor:
        out_of_bound = torch.any(self.p < -self.L, dim=-1) | \
                       torch.any(self.p >  self.L, dim=-1)
        return out_of_bound
    
    def truncated(self) -> torch.Tensor:
        return self.progress > self.max_steps


class PointMassPositionControl(BaseEnv):
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.model = PointMassModel(cfg.quad, cfg.dt, cfg.n_substeps, device)
        self.state_dim = 9
        self.action_dim = 3
        if cfg.render.headless:
            self.renderer = None
        else:
            self.renderer = PositionControlRenderer(cfg.render, device.index)
        super(PointMassPositionControl, self).__init__(cfg, device)
    
    def state(self, with_grad=False):
        state = [self.target_vel(), self._v, self._a]
        state = torch.cat(state, dim=-1)
        return state if with_grad else state.detach()
    
    def step(self, action):
        # type: (Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        action = self.rescale_action(action)
        self._state = self.model(self._state, action)
        self._vel_ema.lerp_(self._v, self.vel_ema_factor)
        self.progress += 1
        terminated, truncated = self.terminated(), self.truncated()
        reset = terminated | truncated
        reset_indices = reset.nonzero().squeeze(-1)
        success = truncated & ((self.p - self.target_pos).norm(dim=-1) < 0.5)
        target_vel = self.target_vel()
        loss, loss_components = self.loss_fn(target_vel, action)
        extra = {
            "truncated": truncated,
            "l": self.progress.clone(),
            "reset": reset,
            "reset_indicies": reset_indices,
            "success": success,
            "state_before_reset": self.state(with_grad=True),
            "loss_components": loss_components
        }
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        if self.renderer is not None:
            self.renderer.step(self.state_for_render())
            self.renderer.render()
        return self.state(), loss, terminated, extra
    
    def state_for_render(self) -> Tensor:
        w = torch.zeros_like(self.v)
        state = torch.concat([self.p, self.q, self.v, w], dim=-1)
        return state
    
    def loss_fn(self, target_vel, action):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Dict[str, float]]
        # vel_diff = (self._state[:, 3:6] - target_vel).norm(dim=-1)
        vel_diff = (self._vel_ema - target_vel).norm(dim=-1)
        vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
        
        stable_loss = F.mse_loss(self._a, action, reduction="none").sum(dim=-1)
        
        # total_loss = pos_diff * 0.2 + vel_loss + 0.01 * stable_loss
        total_loss = vel_loss + 0.003 * stable_loss
        loss_components = {
            "vel_loss": vel_loss.mean().item(),
            "stable_loss": stable_loss.mean().item(),
            "total_loss": total_loss.mean().item()
        }
        return total_loss, loss_components

    def reset_idx(self, env_idx):
        state_mask = torch.zeros_like(self._state)
        state_mask[env_idx] = 1
        p_new = rand_range(-self.L+0.5, self.L-0.5, size=(self.n_envs, 3), device=self.device)
        v_new = torch.zeros_like(self.v)
        a_new = torch.zeros_like(self.a)
        new_state = torch.cat([p_new, v_new, a_new], dim=-1)
        self._state = torch.where(state_mask.bool(), new_state, self._state)
        self.target_pos.fill_(0.)
        self.progress[env_idx] = 0
    
    def reset(self):
        super().reset()
        return self.state()
    
    def terminated(self) -> Tensor:
        out_of_bound = torch.any(self.p < -self.L, dim=-1) | \
                       torch.any(self.p >  self.L, dim=-1)
        return out_of_bound