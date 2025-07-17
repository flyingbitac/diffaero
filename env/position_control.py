from typing import Tuple, Dict, Union

from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch3d import transforms as T

from quaddif.env.base_env import BaseEnv, BaseEnvMultiAgent
from quaddif.dynamics import PointMassModelBase, QuadrotorModel
from quaddif.utils.math import mvp
from quaddif.utils.render import PositionControlRenderer
from quaddif.utils.runner import timeit

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
            self._a if isinstance(self.dynamics, PointMassModelBase) else self._w,
        ], dim=-1)
        return state if with_grad else state.detach()
    
    @timeit
    def get_observations(self, with_grad=False):
        if self.obs_frame == "local":
            target_vel = self.dynamics.world2local(self.target_vel)
            _v = self.dynamics.world2local(self._v)
        elif self.obs_frame == "world":
            target_vel = self.target_vel
            _v = self._v
        
        if self.dynamic_type == "pointmass":
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
    def step(self, action, need_obs_before_reset=True):
        # type: (Tensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor, Dict[str, Union[Dict[str, Tensor], Dict[str, float], Tensor]]]
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
        if need_obs_before_reset:
            extra["next_obs_before_reset"] = self.get_observations(with_grad=True)
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
            "env_spacing": self.L.value,
        }
        return {k: v[:self.renderer.n_envs] for k, v in states_for_render.items()}
    
    @timeit
    def loss_and_reward(self, action):
        # type: (Tensor) -> Tuple[Tensor, Tensor, Dict[str, float]]
        if isinstance(self.dynamics, PointMassModelBase):
            vel_diff = (self.dynamics._vel_ema - self.target_vel).norm(dim=-1)
            vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
            pos_loss = 1 - (-(self._p-self.target_pos).norm(dim=-1)).exp()
            jerk_loss = F.mse_loss(self.dynamics.a_thrust, self.dynamics.local2world(action), reduction="none").sum(dim=-1)
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
    def step(self, action, need_obs_before_reset=True):
        # type: (Tensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor, Dict[str, Union[Dict[str, Tensor], Dict[str, float], Tensor]]]
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
        if need_obs_before_reset:
            extra["next_obs_before_reset"] = self.get_observations(with_grad=True)
            extra["next_state_before_reset"] = self.get_state(with_grad=True)
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        return self.get_observations(), (None, None), terminated, extra

    def terminated(self) -> Tensor:
        p_range = torch.full_like(self.L.value.unsqueeze(-1), fill_value=self.square_size*2)
        out_of_bound = torch.any(self.p < -p_range, dim=-1) | torch.any(self.p > p_range, dim=-1)
        return out_of_bound

class MultiAgentPositionControl(BaseEnvMultiAgent):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super(MultiAgentPositionControl, self).__init__(cfg, device)
        self.last_action_in_obs: bool = cfg.last_action_in_obs
        self.obs_dim = (
            3 * self.n_agents + # target velocities of all agents
            4 + # quaternion of its own
            3 + # velocity of its own
            (3 + 3) * (self.n_agents - 1) + # relative positions and velocities of all OTHER agents
            3 * self.n_agents + # targets' relative positions of all agents
            self.action_dim * int(self.last_action_in_obs) # last action
        )
        self.global_state_dim = (
            (3 + 4 + 3) * self.n_agents + # positions, quaternions, velocities of all agents
            3 * self.n_agents + # target positions of all agents
            6 # box size
        )
        self.box_size = torch.tensor([[self.L, -self.L, self.L, -self.L, self.L, -self.L]], device=device).expand(self.n_envs, -1)
        self.action_dim = self.dynamics.action_dim
        self.renderer = None if cfg.render.headless else PositionControlRenderer(cfg.render, device)
        self.collision_distance = 0.5
    
    @timeit
    def get_observations(self, with_grad=False):
        if self.dynamic_type == "pointmass":
            # agent观测 每个agent的观测为 自身状态(目标方向单位向量+自身姿态四元数+自身速度向量)+所有其他agent相对状态(p+v)+目标位置
            target_vel_all = self.target_vel.reshape(self.n_envs, self.n_agents*3).unsqueeze(1).expand(-1, self.n_agents, -1) # [n_envs, n_agents, n_agents*3]
            quat_self = self.q # [n_envs, n_agents, 4]
            vel_self = self._v # [n_envs, n_agents, 3]
            
            rel_pos = self.p[:, None, :] - self._p[:, :, None] # [n_envs, n_agents, n_agents, 3]
            rel_pos_all_others = torch.stack(
                [torch.cat([rel_pos[:, i, :i, :], rel_pos[:, i, i+1:, :]], dim=-2) for i in range(self.n_agents)],
                dim=1).reshape(self.n_envs, self.n_agents, -1) # [n_envs, n_agents, (n_agents-1)*3]
            
            rel_vel = self.v[:, None, :] - self._v[:, :, None] # [n_envs, n_agents, n_agents, 3]
            rel_vel_all_others = torch.stack(
                [torch.cat([rel_vel[:, i, :i, :], rel_vel[:, i, i+1:, :]], dim=-2) for i in range(self.n_agents)],
                dim=1).reshape(self.n_envs, self.n_agents, -1) # [n_envs, n_agents, (n_agents-1)*3]
            
            related_pos = (self.target_pos[:, None, :] - self.p[:, :, None]).reshape(self.n_envs, self.n_agents, -1) # [n_envs, n_agents, n_agents*3]
            
            obs = torch.cat([
                target_vel_all,     # [n_envs, n_agents, n_agents*3]
                quat_self,          # [n_envs, n_agents, 4]
                vel_self,           # [n_envs, n_agents, 3]
                rel_pos_all_others, # [n_envs, n_agents, (n_agents-1)*3]
                rel_vel_all_others, # [n_envs, n_agents, (n_agents-1)*3]
                related_pos         # [n_envs, n_agents, n_agents*3]
            ], dim=-1)
            if self.last_action_in_obs:
                obs = torch.cat([obs, self.last_action], dim=-1)
        else:
            raise NotImplementedError("Observations for quadrotor dynamics are not implemented yet")
        return obs if with_grad else obs.detach()

    @timeit
    def get_global_state(self, with_grad=False):
        if self.dynamic_type == "pointmass":   
            # 全局状态 为所有agent自身状态(p+q+v)+目标位置
            drone_states = torch.cat([self._p, self.q, self._v], dim=-1) # [n_envs, n_agents, 10]
            global_state = torch.cat([
                drone_states.reshape(self.n_envs, -1),    # [n_envs, n_agents*10]
                self.target_pos.reshape(self.n_envs, -1), # [n_envs, n_agents*3]
                self.box_size                             # [n_envs, 6]
            ], dim=-1)
        else:
            raise NotImplementedError("Global states for quadrotor dynamics are not implemented yet")
        return global_state if with_grad else global_state.detach()
    
    @timeit
    def step(self, action, need_global_state_before_reset=True):
        # type: (Tensor, bool) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor, Dict[str, Union[Dict[str, float], Tensor]]]
        self.dynamics.step(action)
        (terminated, collision, out_of_bound), truncated = self.terminated(), self.truncated()
        self.progress += 1
        if self.renderer is not None:
            self.renderer.render(self.states_for_render())
            truncated = torch.full_like(truncated, self.renderer.gui_states["reset_all"]) | truncated
        arrived = torch.norm(self.p - self.target_pos, dim=-1).lt(0.5).all(dim=-1) # [n_envs, ]
        self.arrive_time.copy_(torch.where(arrived & (self.arrive_time == 0), self.progress.float() * self.dt, self.arrive_time))
        truncated |= arrived & ((self.progress.float() * self.dt) > (self.arrive_time + 5))
        success = arrived & truncated
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
                "collision_rate": collision[reset],
                "out_of_bound_rate": out_of_bound[reset],
                "arrive_time": self.arrive_time.clone()[reset],
                "l_episode": ((self.progress.clone() - 1) * self.dt)[reset],
                "arrive_time": self.arrive_time.clone()[success],
            },
        }
        if need_global_state_before_reset:
            extra["next_global_state_before_reset"] = self.get_global_state(with_grad=True)
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        return self.get_obs_and_state(), (loss, reward), terminated, extra
    
    @timeit
    def reset_idx(self, env_idx):
        self.randomizer.refresh(env_idx)
        n_resets = len(env_idx)
        state_mask = torch.zeros_like(self.dynamics._state, dtype=torch.bool)
        state_mask[env_idx] = True
        
        self.target_pos_base[env_idx] = 0.

        self.target_pos_rel[env_idx] = 0.
        self.target_pos_rel[env_idx, 1, 0] = 1.
        self.target_pos_rel[env_idx, 2, 1] = 1.
        # self.target_pos_rel[env_idx, 3, :2] = 1.
        # 随机初始化新的位置
        N = 5
        assert N**3 > self.n_agents
        assert (2 * self.L - 1) / N > self.collision_distance
        x = y = z = torch.linspace(-self.L+0.5, self.L-0.5, N, device=self.device)
        xyz = torch.stack(torch.meshgrid([x, y, z], indexing="xy"), dim=-1).reshape(-1, 3) # [N*N*N, 3]
        xyz = xyz.unsqueeze(0).expand(n_resets, -1, -1) # [n_envs, N*N*N, 3]
        random_idx = torch.stack([torch.randperm(N**3, device=self.device) for _ in range(n_resets)], dim=0) # [n_resets, N**3]
        random_idx = random_idx[:, :self.n_agents, None].expand(-1, -1, 3) # [n_resets, n_agents, 3]
        new_pos = torch.zeros(self.n_envs, self.n_agents, 3, device=self.device)
        new_pos[env_idx] = xyz.gather(dim=1, index=random_idx)
        new_state = torch.cat([
            new_pos,
            torch.zeros(self.n_envs, self.n_agents, self.dynamics.state_dim-3, device=self.device)
        ], dim=-1)
        
        if self.dynamic_type == "pointmass":
            new_state[:, :, 8] = 9.8
        elif self.dynamic_type == "quadrotor":
            raise NotImplementedError
        self.dynamics._state = torch.where(state_mask, new_state, self.dynamics._state)
        self.dynamics.reset_idx(env_idx)

        self.progress[env_idx] = 0
        self.arrive_time[env_idx] = 0
        self.max_vel[env_idx] = torch.rand(
            n_resets, device=self.device) * (self.max_target_vel - self.min_target_vel) + self.min_target_vel
    
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
        }
        return {k: v[:self.renderer.n_envs] for k, v in states_for_render.items()}

    @timeit
    def loss_and_reward(self, action):
        # type: (Tensor) -> Tuple[Tensor, Tensor, Dict[str, float]]
        if isinstance(self.dynamics, PointMassModelBase):

            vel_diff = (self.dynamics._vel_ema - self.target_vel).norm(dim=-1)
            vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
        
            mean_distance_to_nearest_target = torch.norm(self.allocated_target_pos - self.p, dim=-1) # [n_envs, n_agents]

            # pos_loss = 1 - mean_distance_to_nearest_target.neg().exp() # 每架飞机距离最近的目标点的距离作为position loss
            pos_loss = 1 - (-(self._p-self.target_pos).norm(dim=-1)).exp()

            jerk_loss = F.mse_loss(self.a, action, reduction="none").sum(dim=-1)

            collide_loss = ( -10 * (self.internal_min_distance-0.5) ).exp()

            total_loss = (
                self.loss_weights.pointmass.vel * vel_loss +
                self.loss_weights.pointmass.jerk * jerk_loss +
                self.loss_weights.pointmass.pos * pos_loss +
                self.loss_weights.pointmass.collision * collide_loss
            ).sum(dim=-1)
            
            total_reward = (
                self.reward_weights.constant - 
                self.reward_weights.pointmass.vel * vel_loss -
                self.reward_weights.pointmass.jerk * jerk_loss -
                self.reward_weights.pointmass.pos * pos_loss -
                self.reward_weights.pointmass.collision * collide_loss
            ).sum(dim=-1).detach()

            loss_components = {
                "vel_loss": vel_loss.mean().item(),
                "pos_loss": pos_loss.mean().item(),
                "jerk_loss": jerk_loss.mean().item(),
                "collide_loss": collide_loss.mean().item(),
                "total_loss": total_loss.mean().item(),
                "mean_distance_to_nearest_target": mean_distance_to_nearest_target.mean().item()
            }
        else:
            raise NotImplementedError
        return total_loss, total_reward, loss_components

    @timeit
    def terminated(self) -> Tensor:
        out_of_bound = torch.logical_or(
            torch.any(self.p < -self.L, dim=-1),
            torch.any(self.p >  self.L, dim=-1)
        ).any(dim=-1) # [n_envs, n_agents, 3] -> [n_envs, n_agents] -> [n_envs, ]
        
        diag = torch.diag(torch.full((self.n_agents, ), float("inf"), device=self.device)).unsqueeze(0).expand(self.n_envs, -1, -1)
        distances = torch.norm(self.p[:, :, None] - self.p[:, None, :], dim=-1).add(diag) # [n_envs, n_agents, n_agents]
        collision = distances.lt(self.collision_distance).any(dim=-1).any(dim=-1) # [n_envs, ]

        terminated = collision | out_of_bound
        return terminated, collision, out_of_bound