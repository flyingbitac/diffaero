import torch
import numpy as np

from quaddif.dynamics import build_dynamics

class Tracker:
    def __init__(self, cfg, device: torch.device):
        self.dynamics = build_dynamics(cfg.dynamics, device)
        self.action_dim = self.dynamics.action_dim
        self.n_envs = cfg.n_envs
        self.optim_steps = cfg.optim_steps
        self.device = device
        self.noise_scale = cfg.noise_scale
        self.prev_controller = torch.zeros(self.n_envs, cfg.rollout_steps, self.action_dim, device=device)
        self.max_action = self.dynamics.max_action
        self.min_action = self.dynamics.min_action
        self.cfg = cfg

    def sample_gaussion(self):
        noise = self.noise_scale * torch.randn(self.n_envs, self.cfg.parallel_traj, self.cfg.rollout_steps, self.action_dim, device=self.device)
        controlled_action = self.prev_controller.unsqueeze(1) + noise
        controlled_action = torch.tanh(controlled_action) * (self.max_action - self.min_action) / 2 + (self.max_action + self.min_action) / 2
        return controlled_action

    def reset(self, env_idx:torch.Tensor):
        self.dynamics.reset_idx(env_idx)
        self.prev_controller[env_idx] = torch.zeros_like(self.prev_controller[env_idx])
    
    def reset_state(self, state:torch.Tensor, reset_controller:bool=False):
        assert state.ndim() == 2 and state.shape[0] == self.n_envs, "State must be of shape [n_envs, state_dim]"
        rollout_init_state = state.unsqueeze(1).repeat(1, self.cfg.parallel_traj, 1)
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
    
    def rollout(self, controlled_action: torch.Tensor):
        assert controlled_action.ndim == 4 and controlled_action.shape[0] == self.n_envs, "Controlled action must be of shape [n_envs, parallel_traj, rollout_steps, action_dim]"
        trajectory = []
        for i in range(self.cfg.rollout_steps):
            action = controlled_action[:, :, i, :]
            self.dynamics.step(action)
            trajectory.append(self.dynamics._state)
        trajectory = torch.stack(trajectory, dim=2)     
        return trajectory   
    
    def step(self, state: torch.Tensor, ref_traj: torch.Tensor, track_vel:bool=False):
        """
        state: [n_envs, state_dim]
        ref_traj: [n_envs, rollout_steps, state_dim]
        """
        self.reset_state(state)
        ref_traj = ref_traj.unsqueeze(1).repeat(1, self.cfg.parallel_traj, 1, 1).detach()  # [n_envs, n_parallel, rollout_steps, state_dim]
        controlled_action = self.sample_gaussion()
        controlled_action.requires_grad__(True)
        optim = torch.optim.Adam([controlled_action], lr=self.cfg.lr)
        for _ in range(self.optim_steps):
            optim.zero_grad()
            trajectory = self.rollout(controlled_action)
            loss = self.track_loss(ref_traj, trajectory, track_vel=track_vel)
            loss.backward()
            optim.step()
        trajectory = self.rollout(controlled_action)
        track_loss = self.track_loss(ref_traj, trajectory, track_vel=track_vel)
        min_loss, min_idx = torch.min(track_loss, dim=1)
        best_controller = controlled_action[torch.arange(self.n_envs), min_idx].detach()
        self.prev_controller = best_controller
        best_action = best_controller[:, 0, :].detach()  # [n_envs, action_dim]
        extra_info = {
            "track_loss": min_loss.mean().item(),
        }
        return best_action, extra_info

if __name__ == "__main__":
    # Example usage
    pass