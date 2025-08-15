from typing import Tuple, Dict, Union
import math

from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch import Tensor

from diffaero.env.base_env import BaseEnvMultiAgent
from diffaero.utils.render import PositionControlRenderer
from diffaero.utils.runner import timeit

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
        self.box_size = torch.stack([
             self.L.value,
            -self.L.value,
             self.L.value,
            -self.L.value,
             self.L.value,
            -self.L.value
        ], dim=-1)
        self.action_dim = self.dynamics.action_dim
        self.renderer = None if cfg.render.headless else PositionControlRenderer(cfg.render, device)
        self.collision_distance = 0.3
    
    @timeit
    def get_observations(self, with_grad=False):
        if self.dynamic_type == "pointmass":
            # target velocities of all agents
            target_vel_all = self.target_vel.reshape(self.n_envs, self.n_agents*3).unsqueeze(1).expand(-1, self.n_agents, -1) # [n_envs, n_agents, n_agents*3]
            # quaternion of its own
            quat_self = self.q # [n_envs, n_agents, 4]
            # velocity of its own
            vel_self = self._v # [n_envs, n_agents, 3]
            
            # relative positions and velocities of all OTHER agents
            rel_pos = self.p[:, None, :] - self._p[:, :, None] # [n_envs, n_agents, n_agents, 3]
            rel_pos_all_others = torch.stack(
                [torch.cat([rel_pos[:, i, :i, :], rel_pos[:, i, i+1:, :]], dim=-2) for i in range(self.n_agents)],
                dim=1).reshape(self.n_envs, self.n_agents, -1) # [n_envs, n_agents, (n_agents-1)*3]
            
            # targets' relative positions of all agents
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
    def get_state(self, with_grad=False):
        if self.dynamic_type == "pointmass":   
            drone_states = torch.cat([self._p, self.q, self._v], dim=-1) # [n_envs, n_agents, 10]
            global_state = torch.cat([
                # positions, quaternions, velocities of all agents
                drone_states.reshape(self.n_envs, -1),    # [n_envs, n_agents*10]
                # target positions of all agents
                self.target_pos.reshape(self.n_envs, -1), # [n_envs, n_agents*3]
                self.box_size                             # [n_envs, 6]
            ], dim=-1)
        else:
            raise NotImplementedError("Global states for quadrotor dynamics are not implemented yet")
        return global_state if with_grad else global_state.detach()
    
    @timeit
    def step(self, action, next_obs_before_reset=False, next_state_before_reset=False):
        # type: (Tensor, bool, bool) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor, Dict[str, Union[Dict[str, float], Tensor]]]
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
        if next_obs_before_reset:
            extra["next_obs_before_reset"] = self.get_observations(with_grad=True)
        if next_state_before_reset:
            extra["next_state_before_reset"] = self.get_state(with_grad=True)
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

        # formation
        edge_length = self.collision_distance * 4
        radius = edge_length / (2 * math.sin(math.pi / self.n_agents))
        angles = torch.linspace(0, 2 * math.pi, self.n_agents + 1, device=self.device)[:-1] # [n_agents]
        angles = angles[None, :].expand(n_resets, -1) + torch.rand(n_resets, 1, device=self.device) * (2 * math.pi / self.n_agents) # [n_resets, n_agents]
        self.target_pos_rel[env_idx] = torch.stack([
            radius * torch.cos(angles),
            radius * torch.sin(angles),
            torch.zeros_like(angles)
        ], dim=-1)
        
        # init positions
        N = 5
        L = self.L.unsqueeze(-1) # [n_envs, 1]
        p_min, p_max = -L+0.5, L-0.5
        linspace = torch.linspace(0, 1, N, device=self.device).unsqueeze(0)
        x = y = z = (p_max - p_min) * linspace + p_min
        assert N**3 > self.n_agents
        assert torch.all((2 * (self.L.value - 0.5)) / N > self.collision_distance)
        xyz = torch.stack([
            x[env_idx].reshape(-1, N, 1, 1).expand(-1,-1, N, N),
            y[env_idx].reshape(-1, 1, N, 1).expand(-1, N,-1, N),
            z[env_idx].reshape(-1, 1, 1, N).expand(-1, N, N,-1)
        ], dim=-1).reshape(-1, N**3, 3)
        random_idx = torch.stack([torch.randperm(N**3, device=self.device) for _ in range(n_resets)], dim=0) # [n_resets, N**3]
        random_idx = random_idx[:, :self.n_agents, None].expand(-1, -1, 3) # [n_resets, n_agents, 3]
        p_new = torch.zeros(self.n_envs, self.n_agents, 3, device=self.device)
        p_new[env_idx] = xyz.gather(dim=1, index=random_idx)
        new_state = torch.cat([
            p_new,
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
            "env_spacing": self.L.value,
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

            collide_loss = self.collision().float()

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

    def collision(self) -> Tensor:
        return self.internal_min_distance < self.collision_distance

    @timeit
    def terminated(self) -> Tensor:
        prange = self.L.value.reshape(self.n_envs, 1, 1).expand(-1, self.n_agents, 3) # [n_envs, n_agents, 3]
        out_of_bound = torch.logical_or(
            torch.any(self.p < -prange, dim=-1),
            torch.any(self.p >  prange, dim=-1)
        ).any(dim=-1) # [n_envs, n_agents, 3] -> [n_envs, n_agents] -> [n_envs, ]
        
        collision = self.collision().any(dim=-1) # [n_envs, ]

        terminated = collision | out_of_bound
        return terminated, collision, out_of_bound