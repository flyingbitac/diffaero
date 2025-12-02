from typing import Tuple, Dict, Union
import os

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pytorch3d import transforms as T

from diffaero.env.base_env import BaseEnv
from diffaero.utils.math import mvp
from diffaero.utils.render import PositionControlRenderer
from diffaero.utils.runner import timeit

class PositionControl(BaseEnv):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        self.last_action_in_obs: bool = cfg.last_action_in_obs
        if self.dynamic_type == "pointmass":
            if self.obs_frame == "local":
                self.obs_dim = 9
            elif self.obs_frame == "world":
                self.obs_dim = 10
        elif self.dynamic_type == "quadrotor":
            self.obs_dim = 10
        if self.last_action_in_obs:
            self.obs_dim += self.action_dim
        self.state_dim = 13
        self.renderer = None if cfg.render.headless else PositionControlRenderer(cfg.render, device)
        self.check_dims()
    
    @timeit
    def get_state(self, with_grad=False):
        state = torch.cat([
            self.target_pos - self.p,
            self.q,
            self._v,
            self._a if self.dynamic_type == "pointmass" else self._w,
        ], dim=-1)
        return state if with_grad else state.detach()
    
    @timeit
    def get_observations(self, with_grad=False):
        if self.obs_frame == "local":
            target_vel = self.dynamics.world2local(self.target_vel)
            _v = self.dynamics.world2local(self._v)
        elif self.obs_frame == "body":
            target_vel = self.world2body(self.target_vel)
            vel = self.world2body(self._v)
        elif self.obs_frame == "world":
            target_vel = self.target_vel
            _v = self._v
        
        if self.dynamic_type == "pointmass":
            if self.obs_frame == "local":
                orient = self.dynamics.uz
            else:
                orient = self.q
            obs = torch.cat([
                target_vel,
                self.dynamics.uz if self.obs_frame == "local" else self.q,
                _v,
            ], dim=-1)
        else:
            obs = torch.cat([target_vel, self._q, _v], dim=-1)
        if self.last_action_in_obs:
            obs = torch.cat([obs, self.last_action], dim=-1)
        return obs if with_grad else obs.detach()
    
    @timeit
    def step(self, action, next_obs_before_reset=False, next_state_before_reset=False):
        # type: (Tensor, bool, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor, Dict[str, Union[Dict[str, Tensor], Dict[str, float], Tensor]]]
        terminated, truncated, success, avg_vel = super()._step(action)
        reset = terminated | truncated
        reset_indices = reset.nonzero().view(-1)
        loss, reward, loss_components = self.loss_and_reward(action)
        extra = {
            "truncated": truncated,
            "l": self.progress.clone(),
            "reset": reset,
            "reset_indices": reset_indices,
            "success": success,
            "arrive_time": self.arrive_time.clone(),
            "loss_components": loss_components,
            # Ddata dictionary that contains all the statistical metrics
            # need to be calculated and logged in a sliding-window manner
            # Note: all items in dictionary "stats_raw" should have ndim=1
            "stats_raw": {
                "success_rate": success[reset],
                "survive_rate": truncated[reset],
                "l_episode": ((self.progress.clone() - 1) * self.dt)[reset],
                "avg_vel": avg_vel[success],
                "arrive_time": self.arrive_time.clone()[success]
            },
        }
        if next_obs_before_reset:
            extra["next_obs_before_reset"] = self.get_observations(with_grad=True)
        if next_state_before_reset:
            extra["next_state_before_reset"] = self.get_state(with_grad=True)
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        return self.get_observations(), (loss, reward), terminated, extra
    
    def states_for_render(self) -> Dict[str, Tensor]:
        pos = self.p.unsqueeze(1) if self.n_agents == 1 else self.p
        vel = self.v.unsqueeze(1) if self.n_agents == 1 else self.v
        quat_xyzw = self.q.unsqueeze(1) if self.n_agents == 1 else self.q
        target_pos = self.target_pos.unsqueeze(1) if self.n_agents == 1 else self.target_pos
        states_for_render = {
            "pos": pos,
            "vel": vel,
            "quat_xyzw": quat_xyzw,
            "target_pos": target_pos,
            "env_spacing": torch.ones_like(self.L.value),
        }
        return {k: v[:self.renderer.n_envs] for k, v in states_for_render.items()}
    
    @timeit
    def loss_and_reward(self, action):
        # type: (Tensor) -> Tuple[Tensor, Tensor, Dict[str, float]]
        if self.dynamic_type == "pointmass":
            vel_diff = (self.dynamics._vel_ema - self.target_vel).norm(dim=-1)
            vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
            pos_loss = 1 - (-(self._p-self.target_pos).norm(dim=-1)).exp()
            if self.dynamics.action_frame == "local":
                action = self.dynamics.local2world(action)
            jerk_loss = F.mse_loss(self.dynamics.a_thrust, action, reduction="none").sum(dim=-1)
            total_loss = (
                self.loss_weights.pointmass.vel * vel_loss +
                self.loss_weights.pointmass.jerk * jerk_loss +
                self.loss_weights.pointmass.pos * pos_loss
            )
            total_reward = (
                self.reward_weights.constant - 
                self.reward_weights.pointmass.vel * vel_loss -
                self.reward_weights.pointmass.jerk * jerk_loss -
                self.reward_weights.pointmass.pos * pos_loss
            ).detach()
            
            loss_components = {
                "vel_loss": vel_loss.mean().item(),
                "pos_loss": pos_loss.mean().item(),
                "jerk_loss": jerk_loss.mean().item(),
                "total_loss": total_loss.mean().item(),
                "total_reward": total_reward.mean().item()
            }
        else:
            rotation_matrix_b2i = T.quaternion_to_matrix(self._q.roll(1, dims=-1)).clamp_(min=-1.0+1e-6, max=1.0-1e-6)
            yaw, pitch, roll = T.matrix_to_euler_angles(rotation_matrix_b2i, "ZYX").unbind(dim=-1)
            attitude_loss = roll**2 + pitch**2
            vel_diff = (self._v - self.target_vel).norm(dim=-1)
            vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
            jerk_loss = self._w.norm(dim=-1)
            pos_loss = 1 - (-(self._p-self.target_pos).norm(dim=-1)).exp()
            
            total_loss = (
                self.loss_weights.quadrotor.vel * vel_loss +
                self.loss_weights.quadrotor.jerk * jerk_loss +
                self.loss_weights.quadrotor.pos * pos_loss +
                self.loss_weights.quadrotor.attitude * attitude_loss
            )
            total_reward = (
                self.reward_weights.constant - 
                self.reward_weights.quadrotor.vel * vel_loss -
                self.reward_weights.quadrotor.jerk * jerk_loss -
                self.reward_weights.quadrotor.pos * pos_loss -
                self.reward_weights.quadrotor.attitude * attitude_loss
            ).detach()
            
            loss_components = {
                "vel_loss": vel_loss.mean().item(),
                "jerk_loss": jerk_loss.mean().item(),
                "attitute_loss": attitude_loss.mean().item(),
                "pos_loss": pos_loss.mean().item(),
                "total_loss": total_loss.mean().item(),
                "total_reward": total_reward.mean().item()
            }
        return total_loss, total_reward, loss_components

    @timeit
    def reset_idx(self, env_idx):
        self.randomizer.refresh(env_idx)
        self.imu.reset_idx(env_idx)
        n_resets = len(env_idx)
        state_mask = torch.zeros_like(self.dynamics._state, dtype=torch.bool)
        state_mask[env_idx] = True
        
        L = self.L.unsqueeze(-1) # [n_envs, 1]
        p_min, p_max = -L+0.5, L-0.5
        p_new = torch.rand((self.n_envs, 3), device=self.device) * (p_max - p_min) + p_min
        self.init_pos[env_idx] = p_new[env_idx]
        new_state = torch.cat([p_new, torch.zeros(self.n_envs, self.dynamics.state_dim-3, device=self.device)], dim=-1)
        if self.dynamic_type == "pointmass":
            new_state[:, 8] = 9.8
        elif self.dynamic_type == "quadrotor":
            new_state[:, 6] = 1 # real part of the quaternion
        self.dynamics._state = torch.where(state_mask, new_state, self.dynamics._state)
        self.dynamics.reset_idx(env_idx)
        self.target_pos.fill_(0.)
        self.progress[env_idx] = 0
        self.arrive_time[env_idx] = 0
        self.last_action[env_idx] = 0.
        self.max_vel[env_idx] = torch.rand(
            n_resets, device=self.device) * (self.max_target_vel - self.min_target_vel) + self.min_target_vel
    
    def terminated(self) -> Tensor:
        p_range = self.L.value.unsqueeze(-1)
        out_of_bound = torch.any(self.p < -p_range, dim=-1) | torch.any(self.p > p_range, dim=-1)
        return out_of_bound

    def export_obs_fn(self, path):
        class ObsFn(nn.Module):
            def __init__(self, obs_in_local_frame: bool):
                super().__init__()
                # this boolean becomes a constant attribute in the scripted module
                self.obs_in_local_frame = obs_in_local_frame

            def forward(
                self,
                target_vel_w: Tensor,
                v_w: Tensor,
                quat_xyzw: Tensor,
                Rz: Tensor,
                R: Tensor
            ) -> Tensor:
                if self.obs_in_local_frame:
                    v_l = mvp(Rz.permute(0, 2, 1), v_w)
                    target_vel_l = mvp(Rz.permute(0, 2, 1), target_vel_w)
                    uz = R[:, :, 2]
                    return torch.cat([target_vel_l, uz, v_l], dim=-1)
                else:
                    return torch.cat([target_vel_w, quat_xyzw, v_w], dim=-1)

        example_input = {
            "target_vel_w": torch.randn(1, 3),
            "v_w": torch.randn(1, 3),
            "quat_xyzw": torch.randn(1, 4),
            "Rz": torch.randn(1, 3, 3),
            "R": torch.randn(1, 3, 3),
        }

        model = ObsFn(obs_in_local_frame=self.obs_frame=="local")
        torch.onnx.export(
            model=model,
            args=(
                example_input["target_vel_w"],
                example_input["v_w"],
                example_input["quat_xyzw"],
                example_input["Rz"],
                example_input["R"],
            ),
            input_names=("target_vel_w", "v_w", "quat_xyzw", "Rz", "R"),
            f=os.path.join(path, "obs_fn.onnx"),
            output_names=("obs",)
        )
        # import onnxruntime as ort
        # ort_session = ort.InferenceSession(os.path.join(path, "obs_fn.onnx"))
        # print({input.name: input.shape for input in ort_session.get_inputs()})


class Sim2RealPositionControl(PositionControl):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super(Sim2RealPositionControl, self).__init__(cfg, device)
        self.square_size: float = cfg.square_size
        self.square_positions = torch.tensor([
            [ self.square_size, -self.square_size, 0],
            [-self.square_size, -self.square_size, 0],
            [-self.square_size,  self.square_size, 0],
            [ self.square_size,  self.square_size, 0]
        ], device=self.device, dtype=torch.float32)
        self.switch_time: float = cfg.switch_time
    
    def update_target(self):
        t = self.progress.float() * self.dt
        target_index = torch.floor(t / self.switch_time).long() % self.square_positions.shape[0]
        self.target_pos = self.square_positions[target_index]
    
    @timeit
    def step(self, action, next_obs_before_reset=False, next_state_before_reset=False):
        # type: (Tensor, bool, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor, Dict[str, Union[Dict[str, Tensor], Dict[str, float], Tensor]]]
        self.update_target()
        terminated, truncated, success, avg_vel = super()._step(action)
        reset = terminated | truncated
        reset_indices = reset.nonzero().view(-1)
        extra = {
            "truncated": truncated,
            "l": self.progress.clone(),
            "reset": reset,
            "reset_indices": reset_indices,
            "success": success,
            "arrive_time": self.arrive_time.clone(),
            "stats_raw": {
                "success_rate": success[reset],
                "survive_rate": truncated[reset],
                "l_episode": ((self.progress.clone() - 1) * self.dt)[reset],
                "avg_vel": avg_vel[success],
                "arrive_time": self.arrive_time.clone()[success]
            },
        }
        if next_obs_before_reset:
            extra["next_obs_before_reset"] = self.get_observations(with_grad=True)
        if next_state_before_reset:
            extra["next_state_before_reset"] = self.get_state(with_grad=True)
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        return self.get_observations(), (None, None), terminated, extra

    def terminated(self) -> Tensor:
        p_range = torch.full_like(self.L.value.unsqueeze(-1), fill_value=self.square_size*2)
        out_of_bound = torch.any(self.p < -p_range, dim=-1) | torch.any(self.p > p_range, dim=-1)
        return out_of_bound
