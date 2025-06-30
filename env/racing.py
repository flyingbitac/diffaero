from typing import Tuple, Dict, Union, List
import os
import math

from omegaconf import DictConfig
import torch
import torch.nn.functional as F
import pytorch3d.transforms as T
from torch import Tensor
from tensordict import TensorDict
import open3d as o3d
import numpy as np

from quaddif.env.base_env import BaseEnv
from quaddif.dynamics.pointmass import PointMassModelBase
from quaddif.utils.sensor import build_sensor
from quaddif.utils.render import PositionControlRenderer
from quaddif.utils.assets import ObstacleManager
from quaddif.utils.runner import timeit
from quaddif.utils.math import mat_vec_mul

@torch.jit.script
def get_gate_rotmat_w2g(gate_yaw: Tensor) -> Tensor:
    zero, one = torch.zeros_like(gate_yaw), torch.ones_like(gate_yaw)
    sin, cos = torch.sin(gate_yaw), torch.cos(gate_yaw)
    rotmat_w2g = torch.stack([
        torch.stack([ cos,  sin, zero], dim=-1),
        torch.stack([-sin,  cos, zero], dim=-1),
        torch.stack([zero, zero,  one], dim=-1)
    ], dim=-2)
    return rotmat_w2g

class Racing(BaseEnv):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        g_r = cfg.gates.radius
        g_h = cfg.gates.height
        #number 8
        self.gate_pos = torch.tensor([
            [ g_r,  - g_r, g_h],
            [  0,       0, g_h],
            [-g_r,    g_r, g_h],
            [   0,  2*g_r, g_h],
            [ g_r,    g_r, g_h],
            [   0,      0, g_h],
            [-g_r,   -g_r, g_h],
            [   0, -2*g_r, g_h],
        ], device=device)
        self.gate_yaw = torch.tensor([1, 2, 1, 0, -1, -2, -1, 0], device=device)*torch.pi/2
        if(cfg.ref_path is not None):
            self.ref_pos = torch.from_numpy(np.load(cfg.ref_path + "/position_ned.npy")).float().to(device)
            self.ref_vel = torch.from_numpy(np.load(cfg.ref_path + "/velocity_ned.npy")).float().to(device)
            self.ref_pos = torch.roll(self.ref_pos, shifts=1, dims=0)
            self.ref_vel = torch.roll(self.ref_vel, shifts=1, dims=0)
            self.ref_pos = self.ref_pos.reshape(-1, 7, 3) # [8, 7, 3]
            self.ref_vel = self.ref_vel.reshape(-1, 7, 3) # [8, 7, 3]
        self.target_gates = torch.zeros(self.n_envs, dtype=torch.int, device=device)
        self.n_gates = self.gate_pos.shape[0]
        
        self.obs_dim = 13
        self.state_dim = 4 + 3 + 3 * (3 * 3)
        
        # Calculate relative gates
        self.gate_rel_pos = torch.zeros(self.n_gates, 3, device=device)
        self.gate_yaw_rel = torch.zeros(self.n_gates, device=device)
        for i in range(0, self.n_gates):
            self.gate_rel_pos[i] = self.gate_pos[i] - self.gate_pos[i-1]
            # Rotation matrix
            rotmat = get_gate_rotmat_w2g(self.gate_yaw[i-1])
            self.gate_rel_pos[i] = rotmat @ self.gate_rel_pos[i]
            # wrap yaw
            yaw_diff = self.gate_yaw[i] - self.gate_yaw[i-1]
            self.gate_yaw_rel[i] = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
        
        self.episode_length = torch.zeros((self.n_envs,), dtype=torch.int, device=device)
        self.renderer = None if cfg.render.headless else PositionControlRenderer(cfg.render, device)
    
    
    @timeit
    def get_observations(self, with_grad=False):
        gate_pos = self.gate_pos[self.target_gates]
        gate_yaw = self.gate_yaw[self.target_gates]
        rotmat_w2g = get_gate_rotmat_w2g(gate_yaw)
        
        pos_g = mat_vec_mul(rotmat_w2g, gate_pos - self._p) # 3
        vel_g = mat_vec_mul(rotmat_w2g, self._v) # 3
        
        rotmat_b2w = T.quaternion_to_matrix(self.q.roll(1, dims=-1))
        rotmat_b2g = torch.matmul(rotmat_w2g, rotmat_b2w)
        rpy_g = T.matrix_to_euler_angles(rotmat_b2g, "ZYX")[..., [2, 1, 0]] # 3
        
        next_gate_idx = (self.target_gates + 1) % self.n_gates
        
        obs = torch.cat([
            pos_g,
            vel_g,
            rpy_g,
            self.gate_rel_pos[next_gate_idx],
            self.gate_yaw_rel[next_gate_idx].unsqueeze(-1)
        ], dim=-1)

        return obs if with_grad else obs.detach()
    
    @timeit
    def get_state(self, with_grad=False):
        states = [
            self._v,
            self.q
        ]
        for i in range(3):
            gate_pos = self.gate_pos[(self.target_gates + i) % self.n_gates]
            gate_yaw = self.gate_yaw[(self.target_gates + i) % self.n_gates]
            rotmat_w2g = get_gate_rotmat_w2g(gate_yaw)
            
            pos_g = mat_vec_mul(rotmat_w2g, gate_pos - self._p)
            vel_g = mat_vec_mul(rotmat_w2g, self._v)
            
            rotmat_b2w = T.quaternion_to_matrix(self.q.roll(1, dims=-1))
            rotmat_b2g = torch.matmul(rotmat_w2g, rotmat_b2w)
            rpy_g = T.matrix_to_euler_angles(rotmat_b2g, "ZYX")[..., [2, 1, 0]]
            
            states.append(pos_g) # 3
            states.append(vel_g) # 3
            states.append(rpy_g) # 3
        states = torch.cat(states, dim=-1)
        return states if with_grad else states.detach()
    
    @timeit
    def step(self, action, need_obs_before_reset=True):
        # type: (Tensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        prev_pos = self._p.clone()
        # simulation step
        self.dynamics.step(action)
        
        gate_passed, gate_collision = self.is_passed(prev_pos)
        self.target_gates[gate_passed] = (self.target_gates[gate_passed] + 1) % self.n_gates
        self.target_pos.copy_(self.gate_pos[self.target_gates])
        
        # termination and truncation logic
        terminated, truncated = gate_collision, self.truncated()
        self.progress += 1
        if self.renderer is not None:
            self.renderer.render(self.states_for_render())
            # truncate if `reset_all` is commanded by the user from GUI
            truncated = torch.full_like(truncated, self.renderer.gui_states["reset_all"]) | truncated
        # average velocity of the agents
        avg_vel = (self.init_pos - self.target_pos).norm(dim=-1) / self.arrive_time
        # success flag denoting whether the agent has reached the target position at the end of the episode
        success = truncated & (self.progress >= self.max_steps)
        # update last action
        self.last_action.copy_(action.detach())
        
        reset = terminated | truncated
        reset_indices = reset.nonzero().view(-1)
        loss, reward, loss_components = self.loss_and_reward(action, prev_pos, gate_passed, gate_collision)
        extra = {
            "truncated": truncated,
            "l": self.progress.clone(),
            "reset": reset,
            "reset_indicies": reset_indices,
            "success": success,
            "loss_components": loss_components,
            "stats_raw": {
                "success_rate": success[reset],
                "survive_rate": truncated[reset],
                "l_episode": ((self.progress.clone() - 1) * self.dt)[reset],
                # "avg_vel": avg_vel[success],
                "avg_vel": self.v.norm(dim=-1)
            },
        }
        if need_obs_before_reset:
            extra["next_obs_before_reset"] = self.get_observations(with_grad=True)
            extra["next_state_before_reset"] = self.get_state(with_grad=True)
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        return self.get_observations(), (loss, reward), terminated, extra
    
    @timeit
    def reset_idx(self, env_idx: torch.Tensor):
        self.randomizer.refresh(env_idx)
        n_resets = len(env_idx)
        state_mask = torch.zeros_like(self.dynamics._state, dtype=torch.bool)
        state_mask[env_idx] = True
        
        # set target gates to random gates
        self.target_gates[env_idx] = torch.randint(0, self.n_gates, (n_resets,), device=self.device, dtype=torch.int32)
        # set position to 1m in front of the target gate
        # gate_pos + [cos(gate_yaw), sin(gate_yaw), 0]
        pos = self.gate_pos[self.target_gates]
        yaw = self.gate_yaw[self.target_gates]
        p_new = pos - torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)]).T
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
    
    def is_passed(self, prev_pos):
        # gain previous and current position in world frame
        curr_pos = self.p
        gate_pos = self.gate_pos[self.target_gates]
        gate_yaw = self.gate_yaw[self.target_gates]

        # Gate passing/collision
        rotmat = get_gate_rotmat_w2g(gate_yaw)
        prev_rel_pos = mat_vec_mul(rotmat, prev_pos - gate_pos)
        curr_rel_pos = mat_vec_mul(rotmat, curr_pos - gate_pos)
        prev_behind = prev_rel_pos[:, 0] < 0
        curr_infront = curr_rel_pos[:, 0] > 0
        pass_through = prev_behind & curr_infront
        gate_size = 3
        inside_gate = torch.norm(curr_rel_pos[..., 1:], dim=-1, p=1) < gate_size/2
        gate_passed = pass_through & inside_gate
        gate_collision = pass_through & ~inside_gate
        
        return gate_passed, gate_collision

    def cal_ref_vel(self,):
        if hasattr(self, "ref_vel"):
            ref_pos = self.ref_pos[self.target_gates] # [n_envs, 7, 3]
            ref_vel = self.ref_vel[self.target_gates] # [n_envs, 7, 3]
            ref_pos_next = self.ref_pos[(self.target_gates+1) % self.n_gates] # [n_envs, 7, 3]
            start_pts = ref_pos         
            end_pts = torch.cat([ref_pos[:, 1:], ref_pos_next[:, :1]], dim=1) # [n_envs, 7, 3]        
            segment_vec = end_pts - start_pts                      
            # (n_envs, 3)
            agent_pos = self.p                                     
            vec_agent_to_start = agent_pos.unsqueeze(1) - start_pts # (n_envs, 7, 3)
            # (n_envs, 7)
            t = torch.sum(vec_agent_to_start * segment_vec, dim=-1) / \
                torch.sum(segment_vec * segment_vec, dim=-1)       
            closest_pts = start_pts + t.unsqueeze(-1) * segment_vec  # (n_envs, 7, 3)
            vec_agent_to_closest = closest_pts - agent_pos.unsqueeze(1)  # (n_envs, 7, 3)
            min_dist, indices = torch.min(vec_agent_to_closest.norm(dim=-1), dim=-1)
            
            # vert_ref_vel = F.normalize(vec_agent_to_closest[torch.arange(self.n_envs), indices], dim=-1) * 1. * (1. - torch.exp(-min_dist.unsqueeze(-1)))
            vert_ref_vel = torch.zeros_like(self.v)
            vert_ref_vel[:, 2] = (1.57 - self.p[:, 2]).clamp(min=-1., max=1.) * 0.1
            hori_ref_vel = ref_vel[torch.arange(self.n_envs), indices] 
            target_vel = torch.cat([hori_ref_vel[..., :-1], vert_ref_vel[..., -1:]], dim=-1).detach()
            return target_vel
        else:
            return self.target_vel
            
    @timeit
    def loss_and_reward(self, action, prev_pos, gate_passed, gate_collision):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Dict[str, float]]
        # gain previous and current position in world frame
        curr_pos = self._p.clone()
        gate_pos = self.gate_pos[self.target_gates]
        gate_yaw = self.gate_yaw[self.target_gates]
        
        d2g_old = torch.norm(prev_pos - gate_pos, dim=1)
        d2g_new = torch.norm(curr_pos - gate_pos, dim=1)
        # rate_penalty = 0.005 * torch.norm(self._w, dim=1)
        progress_loss = d2g_new - d2g_old.detach()
        
        out_of_bounds = torch.any(torch.abs(self.p[:, 0:2]) > 5, dim=1) # edges of the grid
        out_of_bounds |= self.p[:, 2] > 7                               # height of the grid
        oob_loss = out_of_bounds.float()
        pass_loss = -gate_passed.float()
        collision_loss = gate_collision.float()
        
        rotmat_w2g = get_gate_rotmat_w2g(gate_yaw)
        pos_g = mat_vec_mul(rotmat_w2g, gate_pos - self._p) # 3
        vel_g = mat_vec_mul(rotmat_w2g, self._v) # 3
        
        if isinstance(self.dynamics, PointMassModelBase):
            target_vel = self.cal_ref_vel()
            vel_diff = (self.dynamics._vel_ema - target_vel).norm(dim=-1)
            vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
            # print(vel_g.mean(dim=0))
            
            # forward = torch.tensor([[-1., 0., 0.]], device=self.device)
            # vel_loss = torch.sum(F.normalize(vel_g, dim=-1) * forward, dim=-1)
            
            jerk_loss = F.mse_loss(self.a, action, reduction="none").sum(dim=-1)
            total_loss = (
                self.loss_weights.pointmass.vel * vel_loss +
                self.loss_weights.pointmass.jerk * jerk_loss +
                self.loss_weights.pointmass.progress * progress_loss
            )
            total_reward = (
                self.reward_weights.constant - 
                self.reward_weights.pointmass.vel * vel_loss -
                self.reward_weights.pointmass.jerk * jerk_loss -
                self.reward_weights.pointmass.passed * pass_loss -
                self.reward_weights.pointmass.oob * oob_loss -
                self.reward_weights.pointmass.progress * progress_loss -
                self.reward_weights.pointmass.collision * collision_loss
            ).detach()

            loss_components = {
                "vel_loss": vel_loss.mean().item(),
                "jerk_loss": jerk_loss.mean().item(),
                "pass_loss": pass_loss.mean().item(),
                "progress_loss": progress_loss.mean().item(),
                "collision_loss": collision_loss.mean().item(),
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
                self.reward_weights.quadrotor.attitude * attitude_loss - 
                self.reward_weights.quadrotor.progress * progress_loss - 
                self.reward_weights.quadrotor.collision * collision_loss 
            ).detach()
            
            loss_components = {
                "vel_loss": vel_loss.mean().item(),
                "jerk_loss": jerk_loss.mean().item(),
                "attitute_loss": attitude_loss.mean().item(),
                "pos_loss": pos_loss.mean().item(),
                "progress_loss": progress_loss.mean().item(),
                "collision_loss": collision_loss.mean().item(),
                "total_loss": total_loss.mean().item(),
                "total_reward": total_reward.mean().item()
            }
        return total_loss, total_reward, loss_components

    def truncated(self):
        out_of_bounds = torch.any(torch.abs(self.p[:, 0:2]) > 5, dim=1) # edges of the grid
        out_of_bounds |= self.p[:, 2] > 7                               # height of the grid
        return out_of_bounds | super().truncated()
    
    def states_for_render(self) -> Dict[str, Tensor]:
        pos = self.p.unsqueeze(1) if self.n_agents == 1 else self.p
        vel = self.v.unsqueeze(1) if self.n_agents == 1 else self.v
        quat_xyzw = self.q.unsqueeze(1) if self.n_agents == 1 else self.q
        target_pos = self.target_pos.unsqueeze(1) if self.n_agents == 1 else self.target_pos
        states_for_render = {
            "pos": pos,
            "vel": vel,
            "quat_xyzw": quat_xyzw,
            "env_spacing": self.L.value,
            "target_pos": target_pos,
        }
        return {k: v[:self.renderer.n_envs] for k, v in states_for_render.items()}