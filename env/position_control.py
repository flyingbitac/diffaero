from typing import Tuple, Dict, Union

from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch3d import transforms as T

from quaddif.env.base_env import BaseEnv
from quaddif.utils.render import PositionControlRenderer
from quaddif.utils.math import rand_range

class PositionControl(BaseEnv):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super(PositionControl, self).__init__(cfg, device)
        self.state_dim = 10
        self.action_dim = self.model.action_dim
        if cfg.render.headless:
            self.renderer = None
        else:
            self.renderer = PositionControlRenderer(cfg.render, device)
    
    def state(self, with_grad=False):
        if self.dynamic_type == "pointmass":
            state = torch.cat([self.target_vel, self.q, self._v], dim=-1)
        else:
            state = torch.cat([self.target_vel, self._q, self._v], dim=-1)
        return state if with_grad else state.detach()
    
    def step(self, action):
        # type: (Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Union[Dict[str, float], Tensor]]]
        self.model.step(action)
        terminated, truncated = self.terminated(), self.truncated()
        self.progress += 1
        if self.renderer is not None:
            self.renderer.step(**self.state_for_render())
            self.renderer.render()
            truncated = torch.full_like(truncated, self.renderer.gui_states["reset_all"]) | truncated
        reset = terminated | truncated
        reset_indices = reset.nonzero().view(-1)
        arrived = (self.p - self.target_pos).norm(dim=-1) < 0.5
        self.arrive_time.copy_(torch.where(arrived & (self.arrive_time == 0), self.progress.float() * self.dt, self.arrive_time))
        success = arrived & truncated
        loss, loss_components = self.loss_fn(action)
        extra = {
            "truncated": truncated,
            "l": self.progress.clone(),
            "reset": reset,
            "reset_indicies": reset_indices,
            "success": success,
            "arrive_time": self.arrive_time.clone(),
            "next_state_before_reset": self.state(with_grad=True),
            "loss_components": loss_components
        }
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        return self.state(), loss, terminated, extra
    
    def state_for_render(self) -> Tensor:
        return {"drone_pos": self.p.clone(), "drone_quat_xyzw": self.q.clone(), "target_pos": self.target_pos.clone()}
    
    def loss_fn(self, action):
        # type: (Tensor) -> Tuple[Tensor, Dict[str, float]]
        if self.dynamic_type == "pointmass":
            vel_diff = (self.model._vel_ema - self.target_vel).norm(dim=-1)
            vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
            
            pos_loss = 1 - (-(self._p-self.target_pos).norm(dim=-1)).exp()
            
            jerk_loss = F.mse_loss(self.a, action, reduction="none").sum(dim=-1)
            
            total_loss = vel_loss + 0.005 * jerk_loss + pos_loss
            loss_components = {
                "vel_loss": vel_loss.mean().item(),
                "pos_loss": pos_loss.mean().item(),
                "jerk_loss": jerk_loss.mean().item(),
                "total_loss": total_loss.mean().item()
            }
        else:
            rotation_matrix_b2i = T.quaternion_to_matrix(self._q.roll(1, dims=-1)).clamp_(min=-1.0+1e-6, max=1.0-1e-6)
            yaw, pitch, roll = T.matrix_to_euler_angles(rotation_matrix_b2i, "ZYX").unbind(dim=-1)
            attitute_loss = roll**2 + pitch**2
            
            vel_diff = (self._v - self.target_vel).norm(dim=-1)
            vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
            
            jerk_loss = self._w.norm(dim=-1)
            
            pos_loss = 1 - (-(self._p-self.target_pos).norm(dim=-1)).exp()
            
            total_loss = vel_loss + 0.2 * jerk_loss + pos_loss + 0.1 * attitute_loss
            loss_components = {
                "vel_loss": vel_loss.mean().item(),
                "jerk_loss": jerk_loss.mean().item(),
                "attitute_loss": attitute_loss.mean().item(),
                "pos_loss": pos_loss.mean().item(),
                "total_loss": total_loss.mean().item()
            }
        return total_loss, loss_components

    def reset_idx(self, env_idx):
        state_mask = torch.zeros_like(self.model._state, dtype=torch.bool)
        state_mask[env_idx] = True
        p_new = rand_range(-self.L+0.5, self.L-0.5, size=(self.n_envs, 3), device=self.device)
        new_state = torch.cat([p_new, torch.zeros(self.n_envs, self.model.state_dim-3, device=self.device)], dim=-1)
        if self.dynamic_type == "quadrotor":
            new_state[:, 6] = 1 # real part of the quaternion
        elif self.dynamic_type == "pointmass":
            new_state[:, 8] = 9.8
        self.model._state = torch.where(state_mask, new_state, self.model._state)
        self.model.reset_idx(env_idx)
        self.target_pos.fill_(0.)
        self.progress[env_idx] = 0
        self.arrive_time[env_idx] = 0
    
    def reset(self):
        super().reset()
        return self.state()
    
    def terminated(self) -> Tensor:
        out_of_bound = torch.any(self.p < -self.L, dim=-1) | \
                       torch.any(self.p >  self.L, dim=-1)
        return out_of_bound