import pathlib
import sys
from dataclasses import dataclass
from typing import List,Dict,Callable
from copy import deepcopy
import os
folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))

import torch
import torch.nn as nn
import numpy as np
import hydra
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from diffaero.dynamics import build_dynamics

class Trajectory:
    def __init__(self, cfg, device: torch.device):
        p2 = np.load(os.path.join(cfg.ref_path, "position_ned.npy"))
        v2 = np.load(os.path.join(cfg.ref_path, "velocity_ned.npy"))
        dt = np.load(os.path.join(cfg.ref_path, "dt.npy"))
        dt = torch.from_numpy(dt).to(device).float()  # [num_ref_points]
        p2 = torch.from_numpy(p2).to(device).float() # [num_ref_points, 3]
        v2 = torch.from_numpy(v2).to(device).float() # [num_ref_points, 3]
        p1 = torch.roll(p2, shifts=1, dims=0)  # [num_ref_points, 3]
        v1 = torch.roll(v2, shifts=1, dims=0)  # [num_ref_points, 3]
        self.polynomials = self._poly_traj(p1, v1, p2, v2, dt)
        self.ref_pos = p1.unsqueeze(0).repeat(cfg.n_envs, 1, 1)  # [n_envs, num_ref_points, 3]
        self._traj_N = cfg._traj_N
        self._dt = dt
            
    def _poly_traj(self, p1:torch.Tensor, v1:torch.Tensor, p2:torch.Tensor, v2:torch.Tensor, dt:float):
        dt = dt.unsqueeze(-1)  # [num_ref_points, 1]
        a0 = p1
        a1 = v1
        a2 = (3*p2-3*p1-2*v1*dt-v2*dt)/dt/dt
        a3 = (2*p1-2*p2+v2*dt+v1*dt)/dt/dt/dt
        polynomials = torch.stack([a0, a1, a2, a3], dim=1)
        return polynomials
    
    def get_ref_traj(self, cur_pos:torch.Tensor):
        """
        cur_pos: [n_envs, 3]
        return: [n_envs, _traj_N, 3], [n_envs, _traj_N]
        """
        n_envs = cur_pos.shape[0]
        idx = torch.argmin(torch.sum((self.ref_pos - cur_pos.unsqueeze(1))**2, dim=-1), dim=1) - 1 # [n_envs]
        idx = torch.where(idx < 0, torch.ones_like(idx) * (self.ref_pos.shape[1] - 1), idx)  # Ensure idx is non-negative
        polynomials = self.polynomials.unsqueeze(0).repeat(n_envs, 1, 1, 1)  # [n_envs, num_ref_points, 4, 3]
        dt = self._dt.unsqueeze(0).repeat(n_envs, 1)  # [n_envs, num_ref_points]
        traj = []
        traj_dt = []
        for i in range(self._traj_N):
            traj.append(
                polynomials[torch.arange(n_envs), (idx + i)%self.ref_pos.shape[1]]
            )
            traj_dt.append(
                dt[torch.arange(n_envs), (idx + i)%self.ref_pos.shape[1]]
            )
        traj = torch.stack(traj, dim=1) # [n_envs, _traj_N, 4, 3]
        traj_dt = torch.stack(traj_dt, dim=1) # [n_envs, _traj_N]
        traj_t = torch.cumsum(traj_dt, dim=1)  # [n_envs, _traj_N]
        traj_t = torch.cat([torch.zeros(n_envs, 1, device=traj_t.device), traj_t[:, :-1]], dim=1) # [n_envs, _traj_N]
        return traj, traj_t
    
    def generate_ref_pt(self, traj: torch.Tensor, traj_t: torch.Tensor, t0:torch.Tensor, horizon:int):
        ref_pos = []
        ref_vel = []
        traj_t = traj_t.unsqueeze(1).expand(-1, t0.shape[1], -1) # [n_envs, n_parallel, n_rollouts]
        traj = traj.unsqueeze(1).expand(-1, t0.shape[1], -1, -1, -1) # [n_envs, n_parallel, n_rollouts, 4, 3] 
        for i in range(horizon):
            t = (t0 + i * 0.1).unsqueeze(-1) # [n_envs, n_parallel, 1]
            idx = torch.searchsorted(traj_t, t, right=False).squeeze(-1) - 1 # [n_envs, n_parallel]
            batch_env = torch.arange(traj.shape[0]).view(-1, 1).expand(traj.shape[0], traj.shape[1])
            batch_parallel = torch.arange(traj.shape[1]).view(1, -1).expand(traj.shape[0], traj.shape[1])
            # polynomial = traj[torch.arange(traj.shape[0]), idx] # [n_envs, n_parallel, 4, 3]
            polynomial = traj[batch_env, batch_parallel, idx]
            dt = (t.squeeze(-1) - traj_t[batch_env, batch_parallel, idx]).unsqueeze(-1)
            ref_pos.append(
                polynomial[:, :, 0] + polynomial[:, :, 1] * dt + polynomial[:, :, 2] * dt**2 + polynomial[:, :, 3] * dt**3
            )
            ref_vel.append(
                polynomial[:, :, 1] + 2 * polynomial[:, :, 2] * dt + 3 * polynomial[:, :, 3] * dt**2
            )
        ref_pos = torch.stack(ref_pos, dim=2)  # [n_envs, horizon, 3]
        ref_vel = torch.stack(ref_vel, dim=2)  # [n_envs, horizon, 3]
        return ref_pos, ref_vel

def build_track_dynamics(cfg, parallel_traj:int, device: torch.device):
    new_cfg = deepcopy(cfg)
    new_cfg.n_envs = cfg.n_envs * parallel_traj
    return build_dynamics(new_cfg, device=device)

class Tracker:
    def __init__(self, cfg, device: torch.device):
        self.dynamics = build_track_dynamics(cfg.dynamics,cfg.tracker.parallel_traj, device)
        print(self.dynamics.n_agents)
        self.action_dim = self.dynamics.action_dim
        track_cfg = cfg.tracker
        self.n_envs = track_cfg.n_envs
        self.optim_steps = track_cfg.optim_steps
        self.device = device
        self.noise_scale = track_cfg.noise_scale
        self.prev_controller = torch.zeros(self.n_envs, track_cfg.rollout_steps, self.action_dim, device=device)
        self.controlled_action = nn.Parameter(torch.zeros(self.n_envs, track_cfg.parallel_traj, track_cfg.rollout_steps, self.action_dim, device=device))
        self.t0 = nn.Parameter(torch.zeros(self.n_envs, track_cfg.parallel_traj, device=device))
        self.optim = torch.optim.Adam([self.controlled_action, self.t0], lr=track_cfg.lr)
        self.max_action = self.dynamics.max_action
        self.min_action = self.dynamics.min_action
        self.cfg = track_cfg
        self.trajectory = Trajectory(cfg.trajectory, device)

    def rescale_action(self, action: torch.tensor):
        scale = (self.max_action - self.min_action) / 2
        shift = (self.max_action + self.min_action) / 2
        return torch.tanh(action) * scale + shift

    def sample_gaussion(self):
        noise = self.noise_scale * torch.randn(self.n_envs, self.cfg.parallel_traj, self.cfg.rollout_steps, self.action_dim, device=self.device)
        controlled_action = self.prev_controller.unsqueeze(1) + noise
        return controlled_action

    def reset(self, env_idx:torch.Tensor):
        self.dynamics.reset_idx(env_idx)
        self.prev_controller[env_idx] = torch.zeros_like(self.prev_controller[env_idx])
    
    def reset_state(self, state:torch.Tensor, reset_controller:bool=False):
        assert state.ndim == 2 and state.shape[0] == self.n_envs, "State must be of shape [n_envs, state_dim]"
        rollout_init_state = state.unsqueeze(1).repeat(1, self.cfg.parallel_traj, 1).reshape(self.n_envs * self.cfg.parallel_traj, -1)  # [n_envs * parallel_traj, state_dim]
        self.dynamics._state = rollout_init_state
        self.dynamics._vel_ema = torch.zeros_like(self.dynamics._vel_ema)
        if reset_controller:
            self.prev_controller = torch.zeros_like(self.prev_controller)
    
    def track_loss(self, ref_traj: torch.Tensor, traj: torch.Tensor, track_vel:bool=False):
        """
            ref_traj: [n_envs, n_parallel, rollout_steps, state_dim],
            traj: [n_envs, n_parallel, rollout_steps, state_dim]
        """
        if track_vel:
            position_loss = torch.mean(torch.sum((ref_traj[..., :3] - traj[..., :3])**2, dim=-1), dim=-1)
            velocity_loss = torch.mean(torch.sum((ref_traj[..., 3:6] - traj[..., 3:6])**2, dim=-1), dim=-1)
            loss = position_loss + velocity_loss # TODO: add weight
        else:
            loss = torch.mean(torch.sum((ref_traj[..., :3] - traj[..., :3])**2, dim=-1), dim=-1)
        return loss # [n_envs, n_parallel]
    
    def rollout(self, controlled_action: torch.Tensor, cur_state: torch.Tensor):
        assert controlled_action.ndim == 4 and controlled_action.shape[0] == self.n_envs, "Controlled action must be of shape [n_envs, parallel_traj, rollout_steps, action_dim]"
        self.reset_state(cur_state.detach())
        trajectory = []
        for i in range(self.cfg.rollout_steps):
            action = controlled_action[:, :, i, :].reshape(self.n_envs * self.cfg.parallel_traj, -1)  # [n_envs * parallel_traj, action_dim]
            for _ in range(3):
                self.dynamics.step(self.rescale_action(action)) # rollout for 0.1s
            trajectory.append(self.dynamics._state)
        trajectory = torch.stack(trajectory, dim=2).reshape(self.n_envs, self.cfg.parallel_traj, self.cfg.rollout_steps, -1)  # [n_envs, parallel_traj, rollout_steps, state_dim]     
        return trajectory   
    
    def step(self, state: torch.Tensor, track_vel:bool=False):
        """
        state: [n_envs, state_dim]
        ref_traj: [n_envs, rollout_steps, state_dim]
        """
        traj, traj_t = self.trajectory.get_ref_traj(state[:, :3]) # [n_envs, rollout_steps, 4, 3], [n_envs, rollout_steps]
        traj, traj_t = traj.detach(), traj_t.detach()  # Detach to avoid gradients through trajectory generation
        t0 = traj_t[:, :1] + torch.rand(self.n_envs, self.cfg.parallel_traj, device=traj.device) * (traj_t[:, 2:3] - traj_t[:, 0:1]) # [n_envs, parallel_traj]
        with torch.no_grad():
            self.t0.copy_(t0)
            self.controlled_action.copy_(self.sample_gaussion()) 
        self.optim_steps = 50
        
        for _ in range(self.optim_steps):
            ref_traj, ref_vel = self.trajectory.generate_ref_pt(traj, traj_t, self.t0, self.cfg.rollout_steps)
            ref_traj = torch.cat([ref_traj, ref_vel], dim=-1)  # [n_envs, n_parallel, rollout_steps, 6]
            trajectory = self.rollout(self.controlled_action, state)
            loss = self.track_loss(ref_traj, trajectory, track_vel=track_vel).sum()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        
        ref_traj, ref_vel = self.trajectory.generate_ref_pt(traj, traj_t, self.t0, self.cfg.rollout_steps)
        print('ref traj:', ref_traj[:, 0])
        trajectory = self.rollout(self.controlled_action, state)
        track_loss = self.track_loss(ref_traj, trajectory, track_vel=track_vel)
        min_loss, min_idx = torch.min(track_loss, dim=1)
        best_controller = self.controlled_action[torch.arange(self.n_envs), min_idx].detach()
        # self.prev_controller = best_controller.roll(shifts=-1, dims=1)
        self.prev_controller = torch.zeros_like(self.prev_controller)
        best_action = best_controller[:, 0, :].detach()  # [n_envs, action_dim]
        extra_info = {
            "track_loss": min_loss.mean().item(),
        }
        return best_action.detach(), extra_info

class TimeOptimalRacing:
    def __init__(self, cfg, device: torch.device):
        self.dynamics = build_dynamics(cfg.dynamics, device=device)
    
    def get_observation(self,):
        return self.dynamics._state.detach()
    
    def rescale_action(self, action: torch.Tensor):
        scale = (self.dynamics.max_action - self.dynamics.min_action) / 2
        shift = (self.dynamics.max_action + self.dynamics.min_action) / 2
        return torch.tanh(action) * scale + shift
    
    def reset(self,):
        random_pos = torch.ones(self.dynamics.n_envs, 3, device=self.dynamics.device)
        random_pos = random_pos + 0.2*(torch.rand_like(random_pos) * 2 - 1)  # Random position in [-1, 1]
        self.dynamics._state[:, :3] = random_pos
        self.dynamics._state[:, 3:6] = torch.zeros_like(self.dynamics._state[:, 3:6])
        self.dynamics._state[:, 6:9] = torch.zeros_like(self.dynamics._state[:, 6:9]) 
        return self.get_observation()
    
    def step(self, action: torch.Tensor):
        self.dynamics.step(self.rescale_action(action))
        return self.get_observation()

@hydra.main(version_base=None, config_path="/home/zxh/ws/wrqws/diffaero/cfg", config_name="config_track")
def main(cfg: DictConfig):
    env = TimeOptimalRacing(cfg, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    tracker = Tracker(cfg.algo, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    state = env.reset()
    for i in range(1000):
        action, extra_info = tracker.step(state)
        state = env.step(action)
        print('cur pos:', state[:, :3])
        print(f"Step {i}, Track Loss: {extra_info['track_loss']:.4f}")
    
    
if __name__ == "__main__":
    main()
