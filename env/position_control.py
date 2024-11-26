from typing import Tuple, Dict, Union

from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch3d import transforms as T

from quaddif.model.quad import QuadrotorModel, PointMassModel
from quaddif.env.base_env import BaseEnv
from quaddif.utils.render import PositionControlRenderer
from quaddif.utils.math import rand_range

class PositionControl(BaseEnv):
    def __init__(self, env_cfg: DictConfig, model_cfg: DictConfig, device: torch.device):
        super(PositionControl, self).__init__(env_cfg, model_cfg, device)
        self.state_dim = self.model.state_dim
        self.action_dim = self.model.action_dim
        if env_cfg.render.headless:
            self.renderer = None
        else:
            self.renderer = PositionControlRenderer(env_cfg.render, device.index)
    
    def state(self, with_grad=False):
        if self.dynamic_type == "pointmass":
            state = [self.target_vel, self._v, self._a]
        else:
            state = [self.target_vel, self._q, self._v, self._a]
        state = torch.cat(state, dim=-1)
        return state if with_grad else state.detach()
    
    def step(self, action):
        # type: (Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Union[Dict[str, float], Tensor]]]
        action = self.rescale_action(action)
        self.model.step(action)
        self.progress += 1
        terminated, truncated = self.terminated(), self.truncated()
        reset = terminated | truncated
        reset_indices = reset.nonzero().squeeze(-1)
        success = truncated & torch.lt((self.p - self.target_pos).norm(dim=-1), 0.5)
        target_vel = self.target_vel
        loss, loss_components = self.loss_fn(target_vel, action)
        extra = {
            "truncated": truncated,
            "l": self.progress.clone(),
            "reset": reset,
            "reset_indicies": reset_indices,
            "success": success,
            "next_state_before_reset": self.state(with_grad=True),
            "loss_components": loss_components
        }
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        if self.renderer is not None:
            if self.renderer.enable_viewer_sync:
                self.renderer.step(self.state_for_render())
            self.renderer.render()
        return self.state(), loss, terminated, extra
    
    def state_for_render(self) -> Tensor:
        w = torch.zeros_like(self.v) if self.dynamic_type == "pointmass" else self.w
        state = torch.concat([self.p, self.q, self.v, w], dim=-1)
        return state
    
    def loss_fn(self, target_vel, action):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Dict[str, float]]
        if self.dynamic_type == "pointmass":
            vel_diff = (self.model._vel_ema - target_vel).norm(dim=-1)
            vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
            
            jerk_loss = F.mse_loss(self._a, action, reduction="none").sum(dim=-1)
            
            total_loss = vel_loss + 0.003 * jerk_loss
            loss_components = {
                "vel_loss": vel_loss.mean().item(),
                "jerk_loss": jerk_loss.mean().item(),
                "total_loss": total_loss.mean().item()
            }
        else:
            rotation_matrix_b2i = T.quaternion_to_matrix(self._q.roll(1, dims=-1)).clamp_(min=-1.0+1e-6, max=1.0-1e-6)
            yaw, pitch, roll = T.matrix_to_euler_angles(rotation_matrix_b2i, "ZYX").unbind(dim=-1)
            attitute_loss = roll**2 + pitch**2
            
            vel_diff = (self._v - target_vel).norm(dim=-1)
            vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
            
            jerk_loss = self._w.norm(dim=-1)
            
            pos_loss = -(-(self._p-self.target_pos).norm(dim=-1)).exp()
            
            total_loss = vel_loss + jerk_loss + pos_loss
            loss_components = {
                "vel_loss": vel_loss.mean().item(),
                "jerk_loss": jerk_loss.mean().item(),
                "attitute_loss": attitute_loss.mean().item(),
                "pos_loss": pos_loss.mean().item(),
                "total_loss": total_loss.mean().item()
            }
        return total_loss, loss_components

    def reset_idx(self, env_idx):
        state_mask = torch.zeros_like(self.model._state)
        state_mask[env_idx] = 1
        p_new = rand_range(-self.L+0.5, self.L-0.5, size=(self.n_envs, 3), device=self.device)
        new_state = torch.cat([p_new, torch.zeros(self.n_envs, self.model.state_dim-3, device=self.device)], dim=-1)
        if isinstance(self.model, QuadrotorModel):
            new_state[:, 6] = 1 # real part of the quaternion
        self.model._state = torch.where(state_mask.bool(), new_state, self.model._state)
        self.target_pos.fill_(0.)
        self.progress[env_idx] = 0
    
    def reset(self):
        super().reset()
        return self.state()
    
    def terminated(self) -> Tensor:
        out_of_bound = torch.any(self.p < -self.L, dim=-1) | \
                       torch.any(self.p >  self.L, dim=-1)
        return out_of_bound