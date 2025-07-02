from dataclasses import dataclass

import numpy as np
import torch

from quaddif.utils.runner import timeit

@dataclass
class buffercfg:
    perception_width: int
    perception_height: int
    state_dim: int
    action_dim: int
    grid_dim: int
    num_envs: int
    max_length: int
    warmup_length: int
    store_on_gpu: bool
    device: str
    use_perception: bool
    use_grid: bool

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
            self.grid_buffer = torch.empty((cfg.max_length//cfg.num_envs, cfg.num_envs, cfg.grid_dim), dtype=torch.bool, device=device, requires_grad=False) if cfg.use_grid else None
            self.visible_map_buffer = torch.empty((cfg.max_length//cfg.num_envs, cfg.num_envs, cfg.grid_dim), dtype=torch.bool, device=device, requires_grad=False) if cfg.use_grid else None
        else:
            raise ValueError("Only support gpu!!!")

        self.length = 0
        self.num_envs = cfg.num_envs
        self.last_pointer = -1
        self.max_length = cfg.max_length
        self.warmup_length = cfg.warmup_length
        self.use_perception = cfg.use_perception
        self.use_grid = cfg.use_grid

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
            # state, action, reward, termination, perception, grid, visible_map = [], [], [], [], [], [], []
            # for i in range(self.num_envs):
            #     indexes = np.random.randint(0, self.length+1-batch_length, size=batch_size//self.num_envs)
            #     state.append(torch.stack([self.state_buffer[idx:idx+batch_length, i] for idx in indexes]))
            #     action.append(torch.stack([self.action_buffer[idx:idx+batch_length, i] for idx in indexes]))
            #     reward.append(torch.stack([self.reward_buffer[idx:idx+batch_length, i] for idx in indexes]))
            #     termination.append(torch.stack([self.termination_buffer[idx:idx+batch_length, i] for idx in indexes]))
            #     if self.use_perception:
            #         perception.append(torch.stack([self.perception_buffer[idx:idx+batch_length, i] for idx in indexes]))
            #     if self.use_grid:
            #         grid.append(torch.stack([self.grid_buffer[idx:idx+batch_length, i] for idx in indexes]))
            #         visible_map.append(torch.stack([self.visible_map_buffer[idx:idx+batch_length, i] for idx in indexes]))

            # state = torch.cat(state, dim=0)
            # action = torch.cat(action, dim=0)
            # reward = torch.cat(reward, dim=0)
            # termination = torch.cat(termination, dim=0)
            # print("state shape:", state.shape, "action shape:", action.shape, "reward shape:", reward.shape, "termination shape:", termination.shape)
            if self.use_perception:
                perception = self.perception_buffer[idxs, env_idx].reshape(batch_size, batch_length, *self.perception_buffer.shape[2:])
                # perception = torch.cat(perception, dim=0)
            else:
                perception = None
            if self.use_grid:
                grid = self.grid_buffer[idxs, env_idx].reshape(batch_size, batch_length, -1)
                visible_map = self.visible_map_buffer[idxs, env_idx].reshape(batch_size, batch_length, -1)
                # grid = torch.cat(grid, dim=0)
                # visible_map = torch.cat(visible_map, dim=0)
            else:
                grid = None
                visible_map = None
            
        else:
            raise ValueError("Only support gpu!!!")

        return state, action, reward, termination, perception, grid, visible_map

    def append(self, state, action, reward, termination, perception=None, grid=None, visible_map=None):
        self.last_pointer = (self.last_pointer + 1) % (self.max_length//self.num_envs)
        if self.store_on_gpu:
            self.state_buffer[self.last_pointer] = state
            self.action_buffer[self.last_pointer] = action
            self.reward_buffer[self.last_pointer] = reward
            self.termination_buffer[self.last_pointer] = termination
            if self.use_perception and perception is not None:
                self.perception_buffer[self.last_pointer] = perception
            if self.use_grid and grid is not None:
                self.grid_buffer[self.last_pointer] = grid
                self.visible_map_buffer[self.last_pointer] = visible_map
        else:
            raise ValueError("Only support gpu!!!")

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length * self.num_envs
