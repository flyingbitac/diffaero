from dataclasses import dataclass

import numpy as np
import torch

# from diffaero.utils.runner import timeit

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
            self.action_buffer = torch.empty((cfg.max_length//cfg.num_envs, cfg.num_envs,cfg.action_dim), dtype=torch.float32, device=device, requires_grad=False)
            self.reward_buffer = torch.empty((cfg.max_length//cfg.num_envs, cfg.num_envs), dtype=torch.float32, device=device, requires_grad=False)
            self.termination_buffer = torch.empty((cfg.max_length//cfg.num_envs, cfg.num_envs), dtype=torch.float32, device=device, requires_grad=False)
            if cfg.use_perception:
                self.perception_buffer = torch.empty((cfg.max_length//cfg.num_envs, cfg.num_envs, 1, cfg.perception_height, cfg.perception_width), dtype=torch.float32, device=device, requires_grad=False)
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
    # @timeit
    def sample(self, batch_size, batch_length):
        perception = None
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
            raise ValueError("Only support gpu!!!")

        return state, action, reward, termination, perception

    def append(self, state, action, reward, termination, perception=None):
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
        
    def load_external(self, path:str, max_action:torch.Tensor=None, min_action:torch.Tensor=None):
        if min_action == None:
            min_action = torch.tensor([[-20, -20, 0]]).to(self.state_buffer.device)
            max_action = torch.tensor([[20, 20, 40]]).to(self.state_buffer.device)
        with np.load(path) as data:
            state = np.squeeze(data["state"], axis=1) # [length, 9]
            perception = data["perception"] # [length, 1, 9, 16]
            action = data["action"] # [length, 3]
        self.extern_action_buff = (torch.from_numpy(action).float().to(self.state_buffer.device) - min_action) / (max_action - min_action) * 2.0 - 1.0
        self.extern_state_buff = torch.from_numpy(state).float().to(self.state_buffer.device)
        self.extern_perception_buff = torch.from_numpy(perception).float().to(self.state_buffer.device)
    
    def sample_extern(self, batch_size:int, batch_length:int):
        assert hasattr(self, "extern_action_buff"), "Please load external data first!!!"
        index = torch.randint(0, self.extern_action_buff.shape[0] - batch_length, (batch_size,), device=self.state_buffer.device)
        state = torch.stack([self.extern_state_buff[i:i + batch_length] for i in index], dim=0)
        action = torch.stack([self.extern_action_buff[i:i + batch_length] for i in index], dim=0)
        perception = torch.stack([self.extern_perception_buff[i:i + batch_length] for i in index], dim=0)
        return state, action, perception

    def __len__(self):
        return self.length * self.num_envs

if __name__ == "__main__":
    cfg = buffercfg(
        perception_width=16,
        perception_height=9,
        state_dim=9,
        action_dim=3,
        num_envs=16,
        max_length=10000,
        warmup_length=1000,
        store_on_gpu=True,
        device="cuda:0",
        use_perception=True,
    )
    rplb = ReplayBuffer(cfg)
    rplb.load_external("/home/zxh/ws/wrqws/diff/traj/all_trajs.npz")
    s, a, p = rplb.sample_extern(4, 32)
    print(s.shape, a.shape, p.shape)