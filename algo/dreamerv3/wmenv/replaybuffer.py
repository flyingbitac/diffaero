from dataclasses import dataclass

import numpy as np
import torch

from diffaero.utils.runner import timeit

@dataclass
class buffercfg:
    perception_width: int
    perception_height: int
    state_dim: int
    action_dim: int
    num_envs: int
    max_length: int
    warmup_length: int
    store_on_gpu: bool
    device: str
    use_perception: bool

class ReplayBuffer():
    def __init__(self, cfg:buffercfg) -> None:
        self.store_on_gpu = cfg.store_on_gpu
        device = torch.device(cfg.device)
        if cfg.store_on_gpu:
            self.state_buffer = torch.empty((cfg.max_length//cfg.num_envs, cfg.num_envs, cfg.state_dim), dtype=torch.float32, device=device, requires_grad=False)
            self.perception_buffer = torch.empty((cfg.max_length//cfg.num_envs, cfg.num_envs, 1, cfg.perception_height, cfg.perception_width), dtype=torch.float32, device=device, requires_grad=False)
            self.action_buffer = torch.empty((cfg.max_length//cfg.num_envs, cfg.num_envs,cfg.action_dim), dtype=torch.float32, device=device, requires_grad=False)
            self.reward_buffer = torch.empty((cfg.max_length//cfg.num_envs, cfg.num_envs), dtype=torch.float32, device=device, requires_grad=False)
            self.termination_buffer = torch.empty((cfg.max_length//cfg.num_envs, cfg.num_envs), dtype=torch.float32, device=device, requires_grad=False)
        else:
            raise ValueError("Only support gpu!!!")

        self.length = 0
        self.num_envs = cfg.num_envs
        self.last_pointer = -1
        self.max_length = cfg.max_length
        self.warmup_length = cfg.warmup_length
        self.use_perception = cfg.use_perception

    def ready(self):
        return self.length * self.num_envs > self.warmup_length and self.length > 64

    @torch.no_grad()
    @timeit
    def sample(self, batch_size, batch_length):
        if batch_size < self.num_envs:
            batch_size = self.num_envs
        if self.store_on_gpu:
            indexes = torch.randint(0, self.length - batch_length, (batch_size,), device=self.state_buffer.device)
            arange = torch.arange(batch_length, device=self.state_buffer.device)
            idxs = torch.flatten(indexes.unsqueeze(1) + arange.unsqueeze(0)) # shape: (batch_size * batch_length)
            env_idx = torch.randint(0, self.num_envs, (batch_size, 1), device=self.state_buffer.device).expand(-1, batch_length).reshape(-1)
            state = self.state_buffer[idxs, env_idx].reshape(batch_size, batch_length, -1)
            action = self.action_buffer[idxs, env_idx].reshape(batch_size, batch_length, -1)
            reward = self.reward_buffer[idxs, env_idx].reshape(batch_size, batch_length)
            termination = self.termination_buffer[idxs, env_idx].reshape(batch_size, batch_length)
            if self.use_perception:
                perception = self.perception_buffer[idxs, env_idx].reshape(batch_size, batch_length, *self.perception_buffer.shape[2:])
            else:
                perception = None
            
        else:
            raise ValueError("Only support gpu!!!")

        return state, action, reward, termination, perception

    def append(self, state, action, reward, termination, perception=None, visible_map=None):
        self.last_pointer = (self.last_pointer + 1) % (self.max_length//self.num_envs)
        if self.store_on_gpu:
            self.state_buffer[self.last_pointer] = state
            self.action_buffer[self.last_pointer] = action
            self.reward_buffer[self.last_pointer] = reward
            self.termination_buffer[self.last_pointer] = termination
            if self.use_perception and perception is not None:
                self.perception_buffer[self.last_pointer] = perception
        else:
            raise ValueError("Only support gpu!!!")

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length * self.num_envs
