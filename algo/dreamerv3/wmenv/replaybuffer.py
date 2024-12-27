from dataclasses import dataclass

import numpy as np
import torch

@dataclass
class buffercfg:
    perception_dim: int
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
            self.perception_buffer = torch.empty((cfg.max_length//cfg.num_envs, cfg.num_envs, 1, cfg.perception_dim, cfg.perception_dim), dtype=torch.float32, device=device, requires_grad=False)
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
        self.use_perception = True

    def ready(self):
        return self.length * self.num_envs > self.warmup_length and self.length > 64

    @torch.no_grad()
    def sample(self, batch_size, batch_length):
        if batch_size < self.num_envs:
            batch_size = self.num_envs
        if self.store_on_gpu:
            state, action, reward, termination, perception = [], [], [], [], []
            if batch_size > 0:
                for i in range(self.num_envs):
                    indexes = np.random.randint(0, self.length+1-batch_length, size=batch_size//self.num_envs)
                    state.append(torch.stack([self.state_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    action.append(torch.stack([self.action_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    reward.append(torch.stack([self.reward_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    termination.append(torch.stack([self.termination_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    if self.use_perception:
                        perception.append(torch.stack([self.perception_buffer[idx:idx+batch_length, i] for idx in indexes]))

            state = torch.cat(state, dim=0)
            action = torch.cat(action, dim=0)
            reward = torch.cat(reward, dim=0)
            termination = torch.cat(termination, dim=0)
            if self.use_perception:
                perception = torch.cat(perception, dim=0)
        else:
            raise ValueError("Only support gpu!!!")

        return state, action, reward, termination, perception

    def append(self, state, action, reward, termination,perception=None):
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
