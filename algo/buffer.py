from typing import Union, Optional, Tuple

import torch
from torch import Tensor
from tensordict import TensorDict

from diffaero.utils.logger import Logger
from diffaero.utils.runner import timeit

class RNNStateBuffer:
    def __init__(self, l_rollout, n_envs, rnn_hidden_dim, rnn_n_layers, device):
        # type: (int, int, int, int, torch.device) -> None
        factory_kwargs = {"dtype": torch.float32, "device": device}
        self.actor_rnn_state  = torch.zeros((l_rollout, n_envs, rnn_n_layers, rnn_hidden_dim), **factory_kwargs)
        self.critic_rnn_state = torch.zeros((l_rollout, n_envs, rnn_n_layers, rnn_hidden_dim), **factory_kwargs)
    
    def clear(self):
        self.step = 0
    
    @torch.no_grad()
    def add(self, actor_hidden_state: Optional[Tensor], critic_hidden_state: Optional[Tensor] = None):
        if actor_hidden_state is not None:
            self.actor_rnn_state[self.step]  = actor_hidden_state.permute(1, 0, 2)
        if critic_hidden_state is not None:
            self.critic_rnn_state[self.step] = critic_hidden_state.permute(1, 0, 2)
        self.step += 1

class RolloutBufferSHAC:
    def __init__(self, l_rollout, n_envs, obs_dim, action_dim, device):
        # type: (int, int, Union[int, Tuple[int, Tuple[int, int]]], int, torch.device) -> None
        factory_kwargs = {"dtype": torch.float32, "device": device}
        
        assert isinstance(obs_dim, tuple) or isinstance(obs_dim, int)
        if isinstance(obs_dim, tuple):
            self.obs = TensorDict({
                "state": torch.zeros((l_rollout, n_envs, obs_dim[0]), **factory_kwargs),
                "perception": torch.zeros((l_rollout, n_envs, obs_dim[1][0], obs_dim[1][1]), **factory_kwargs)
            }, batch_size=(l_rollout, n_envs))
        else:
            self.obs = torch.zeros((l_rollout, n_envs, obs_dim), **factory_kwargs)
        self.samples = torch.zeros((l_rollout, n_envs, action_dim), **factory_kwargs)
        self.logprobs = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.losses = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.values = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.next_dones = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.next_terminated = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.next_values = torch.zeros((l_rollout, n_envs), **factory_kwargs)
    
    def clear(self):
        self.step = 0
    
    @torch.no_grad()
    def add(self, obs, sample, logprob, loss, value, next_done, next_terminated, next_value):
        # type: (Union[Tensor, TensorDict], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> None
        self.obs[self.step] = obs
        self.samples[self.step] = sample
        self.logprobs[self.step] = logprob
        self.losses[self.step] = loss
        self.values[self.step] = value
        self.next_dones[self.step] = next_done.float()
        self.next_terminated[self.step] = next_terminated.float()
        self.next_values[self.step] = next_value
        self.step += 1

class RolloutBufferMASHAC:
    def __init__(self, l_rollout, n_envs, obs_dim, global_state_dim, n_agents, device):
        # type: (int, int, Union[int, Tuple[int, Tuple[int, int]]], int, int, torch.device) -> None
        factory_kwargs = {"dtype": torch.float32, "device": device}
        
        assert isinstance(obs_dim, tuple) or isinstance(obs_dim, int)
        assert isinstance(global_state_dim, int)
        if isinstance(obs_dim, tuple):
            self.obs = TensorDict({
                "state": torch.zeros((l_rollout, n_envs, n_agents, obs_dim[0]), **factory_kwargs),
                "perception": torch.zeros((l_rollout, n_envs, obs_dim[1][0], obs_dim[1][1]), **factory_kwargs)
            }, batch_size=(l_rollout, n_envs))
        else:
            self.obs = torch.zeros((l_rollout, n_envs, n_agents, obs_dim), **factory_kwargs)
        self.global_states = torch.zeros((l_rollout, n_envs, global_state_dim), **factory_kwargs)
        self.rewards = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.values = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.next_dones = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.next_terminated = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.next_values = torch.zeros((l_rollout, n_envs), **factory_kwargs)
    
    def clear(self):
        self.step = 0
    
    @torch.no_grad()
    def add(self, obs, global_state, reward, value, next_done, next_terminated, next_value):
        # type: (Union[Tensor, TensorDict], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> None
        self.obs[self.step] = obs
        self.global_states[self.step] = global_state
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.next_dones[self.step] = next_done.float()
        self.next_terminated[self.step] = next_terminated.float()
        self.next_values[self.step] = next_value
        self.step += 1


class RolloutBufferSHACQ:
    def __init__(self, l_rollout, n_envs, obs_dim, action_dim, device):
        # type: (int, int, Union[int, Tuple[int, Tuple[int, int]]], int, torch.device) -> None
        factory_kwargs = {"dtype": torch.float32, "device": device}
        
        assert isinstance(obs_dim, tuple) or isinstance(obs_dim, int)
        if isinstance(obs_dim, tuple):
            self.obs = TensorDict({
                "state": torch.zeros((l_rollout, n_envs, obs_dim[0]), **factory_kwargs),
                "perception": torch.zeros((l_rollout, n_envs, obs_dim[1][0], obs_dim[1][1]), **factory_kwargs)
            }, batch_size=(l_rollout, n_envs))
        else:
            self.obs = torch.zeros((l_rollout, n_envs, obs_dim), **factory_kwargs)
        self.next_obs = self.obs.clone()
        self.actions = torch.zeros((l_rollout, n_envs, action_dim), **factory_kwargs)
        self.rewards = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.next_terminated = torch.zeros((l_rollout, n_envs), **factory_kwargs)
    
    def clear(self):
        self.step = 0
    
    @torch.no_grad()
    def add(self, state, action, reward, next_state, next_terminated):
        # type: (Union[Tensor, TensorDict], Tensor, Tensor, Union[Tensor, TensorDict], Tensor) -> None
        self.obs[self.step] = state
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.next_obs[self.step] = next_state
        self.next_terminated[self.step] = next_terminated.float()
        self.step += 1


class RolloutBufferPPO:
    def __init__(self, l_rollout, n_envs, obs_dim, action_dim, device):
        # type: (int, int, Union[int, Tuple[int, Tuple[int, int]]], int, torch.device) -> None
        factory_kwargs = {"dtype": torch.float32, "device": device}
        
        assert isinstance(obs_dim, tuple) or isinstance(obs_dim, int)
        if isinstance(obs_dim, tuple):
            self.obs = TensorDict({
                "state": torch.zeros((l_rollout, n_envs, obs_dim[0]), **factory_kwargs),
                "perception": torch.zeros((l_rollout, n_envs, obs_dim[1][0], obs_dim[1][1]), **factory_kwargs)
            }, batch_size=(l_rollout, n_envs))
        else:
            self.obs = torch.zeros((l_rollout, n_envs, obs_dim), **factory_kwargs)
        self.samples = torch.zeros((l_rollout, n_envs, action_dim), **factory_kwargs)
        self.logprobs = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.rewards = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.next_dones = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.values = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.next_values = torch.zeros((l_rollout, n_envs), **factory_kwargs)
    
    def clear(self):
        self.step = 0
    
    @torch.no_grad()
    def add(self, obs, sample, logprob, reward, next_done, value, next_value):
        # type: (Union[Tensor, TensorDict], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> None
        self.obs[self.step] = obs
        self.samples[self.step] = sample
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.next_dones[self.step] = next_done.float()
        self.values[self.step] = value
        self.next_values[self.step] = next_value
        self.step += 1


class RolloutBufferAPPO(RolloutBufferPPO):
    def __init__(self, l_rollout, n_envs, obs_dim, state_dim, action_dim, device):
        # type: (int, int, Union[int, Tuple[int, Tuple[int, int]]], int, int, torch.device) -> None
        super().__init__(l_rollout, n_envs, obs_dim, action_dim, device)
        factory_kwargs = {"dtype": torch.float32, "device": device}
        self.states = torch.zeros((l_rollout, n_envs, state_dim), **factory_kwargs)
    
    @torch.no_grad()
    def add(self, obs, state, sample, logprob, reward, next_done, value, next_value):
        # type: (Union[Tensor, TensorDict], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> None
        self.states[self.step] = state
        super().add(obs, sample, logprob, reward, next_done, value, next_value)


class RolloutBufferGRID:
    def __init__(
        self,
        l_rollout: int,
        buffer_size: int,
        obs_dim: Tuple[int, Tuple[int, int]],
        state_dim: int,
        action_dim: int,
        grid_dim: int,
        device: torch.device
    ):
        factory_kwargs = {"dtype": torch.float32, "device": device}
        assert l_rollout % 8 == 0
        self.n_bytes = l_rollout // 8
        self.bitmask = torch.tensor([2**i for i in range(8)], dtype=torch.uint8, device=device)
        self.obs = TensorDict({
            "state":       torch.zeros((buffer_size, l_rollout, obs_dim[0]), **factory_kwargs),
            "perception":  torch.zeros((buffer_size, l_rollout, obs_dim[1][0], obs_dim[1][1]), **factory_kwargs),
            "occupancy":   torch.zeros((buffer_size, self.n_bytes, grid_dim), device=device, dtype=torch.uint8),
            "visibility":  torch.zeros((buffer_size, self.n_bytes, grid_dim), device=device, dtype=torch.uint8),
        }, batch_size=(buffer_size, ))
        self.states = torch.zeros((buffer_size, l_rollout, state_dim), **factory_kwargs)
        self.actions = torch.zeros((buffer_size, l_rollout, action_dim), **factory_kwargs)
        self.rewards = torch.zeros((buffer_size, l_rollout), **factory_kwargs)
        self.values = torch.zeros((buffer_size, l_rollout), **factory_kwargs)
        self.dones = torch.zeros((buffer_size, l_rollout), device=device, dtype=torch.bool)
        self.terminated = torch.zeros((buffer_size, l_rollout), device=device, dtype=torch.bool)
        self.next_values = torch.zeros((buffer_size, l_rollout), **factory_kwargs)
        for k, v in self.obs.items():
            mb = v.dtype.itemsize * v.numel() / 1024 / 1024
            Logger.debug(f"Space allocated for buffer.obs.{k}: {mb:.1f} MB")
        for k, v in [("states", self.states), ("actions", self.actions), ("rewards", self.rewards), 
                    ("values", self.values), ("dones", self.dones), ("terminated", self.terminated), 
                    ("next_values", self.next_values)]:
            mb = v.dtype.itemsize * v.numel() / 1024 / 1024
            Logger.debug(f"Space allocated for buffer.{k}: {mb:.1f} MB")
        self.l_rollout = l_rollout
        self.device = device
        self.max_size = buffer_size
        self.size = 0
        self.ptr = 0
        self.time = 0
    
    @timeit
    def compress_obs(self, obs: TensorDict):
        occupied = obs["occupancy"].to(torch.uint8)
        visible = obs["visibility"].to(torch.uint8)
        occupied_compressed = torch.zeros((occupied.shape[0], self.n_bytes, occupied.shape[2]), dtype=torch.uint8, device=occupied.device)
        visible_compressed = torch.zeros((visible.shape[0], self.n_bytes, visible.shape[2]), dtype=torch.uint8, device=visible.device)
        for i in range(self.n_bytes):
            o, v = occupied[:, i*8:(i+1)*8], visible[:, i*8:(i+1)*8]
            o, v = o * self.bitmask[None, :, None], v * self.bitmask[None, :, None]
            o = o.sum(dim=1)
            v = v.sum(dim=1)
            occupied_compressed[:, i] = o
            visible_compressed[:, i] = v
        compressed = TensorDict({
            "state": obs["state"],
            "perception": obs["perception"],
            "occupancy": occupied_compressed,
            "visibility": visible_compressed
        }, batch_size=obs.batch_size[0])
        return compressed

    @timeit
    def expand_obs(self, obs: TensorDict):
        occupied_compressed = obs["occupancy"]
        visible_compressed = obs["visibility"]
        occupied = torch.zeros((occupied_compressed.shape[0], self.l_rollout, occupied_compressed.shape[2]), dtype=torch.bool, device=occupied_compressed.device)
        visible = torch.zeros((visible_compressed.shape[0], self.l_rollout, visible_compressed.shape[2]), dtype=torch.bool, device=visible_compressed.device)
        for i in range(self.n_bytes):
            o = occupied_compressed[:, i:i+1]
            v = visible_compressed[:, i:i+1]
            o = torch.bitwise_and(o, self.bitmask[None, :, None]).ne(0)
            v = torch.bitwise_and(v, self.bitmask[None, :, None]).ne(0)
            occupied[:, i*8:(i+1)*8] = o
            visible[:, i*8:(i+1)*8] = v
        expanded = TensorDict({
            "state": obs["state"],
            "perception": obs["perception"],
            "occupancy": occupied,
            "visibility": visible
        }, batch_size=(obs.batch_size[0], self.l_rollout))
        return expanded

    @torch.no_grad()
    def add(
        self,
        obs: TensorDict,
        state: Optional[Tensor] = None,
        action: Optional[Tensor] = None,
        reward: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        next_done: Optional[Tensor] = None,
        next_terminated: Optional[Tensor] = None,
        next_value: Optional[Tensor] = None
    ):
        n = obs.shape[0]
        start1, end1 = self.ptr, min(self.max_size, self.ptr + n)
        start2, end2 = 0, max(0, self.ptr + n - self.max_size)
        n1, n2 = end1 - start1, end2 - start2
        
        obs = self.compress_obs(obs)
        self.obs[start1:end1] = obs[:n1]
        if n2 > 0:
            self.obs[start2:end2] = obs[n1:]
        
        buffers = [self.states, self.actions, self.rewards, self.values, self.dones, self.terminated, self.next_values]
        data = [state, action, reward, value, next_done, next_terminated, next_value]
        for i, (buf, dat) in enumerate(zip(buffers, data)):
            if dat is None:
                continue
            buf[start1:end1] = dat[:n1]
            if n2 > 0:
                buf[start2:end2] = dat[n1:]
        
        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)
    
    @torch.no_grad()
    def add_step(
        self,
        obs: TensorDict,
        state: Optional[Tensor] = None,
        action: Optional[Tensor] = None,
        reward: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        next_done: Optional[Tensor] = None,
        next_terminated: Optional[Tensor] = None,
        next_value: Optional[Tensor] = None
    ):
        # compress occupancy and visibility
        byte_idx, bit_idx = divmod(self.time, 8)
        base = self.bitmask[bit_idx]
        occupied_compressed = obs["occupancy"].to(torch.uint8) * base
        visible_compressed = obs["visibility"].to(torch.uint8) * base
        
        n = obs.shape[0]
        start1, end1 = self.ptr, min(self.max_size, self.ptr + n)
        start2, end2 = 0, max(0, self.ptr + n - self.max_size)
        n1, n2 = end1 - start1, end2 - start2
        
        if self.time == 0:
            self.obs["occupancy"][start1:end1] = 0
            self.obs["visibility"][start1:end1] = 0
            if n2 > 0:
                self.obs["occupancy"][start2:end2] = 0
                self.obs["visibility"][start2:end2] = 0
        self.obs["occupancy"][start1:end1, byte_idx] += occupied_compressed[:n1]
        self.obs["visibility"][start1:end1, byte_idx] += visible_compressed[:n1]
        self.obs["state"][start1:end1, self.time] = obs["state"][:n1]
        self.obs["perception"][start1:end1, self.time] = obs["perception"][:n1]
        if n2 > 0:
            self.obs["occupancy"][start2:end2, byte_idx] += occupied_compressed[n1:]
            self.obs["visibility"][start2:end2, byte_idx] += visible_compressed[n1:]
            self.obs["state"][start2:end2, self.time] = obs["state"][n1:]
            self.obs["perception"][start2:end2, self.time] = obs["perception"][n1:]
        
        buffers = [self.states, self.actions, self.rewards, self.values, self.dones, self.terminated, self.next_values]
        data = [state, action, reward, value, next_done, next_terminated, next_value]
        for i, (buf, dat) in enumerate(zip(buffers, data)):
            if dat is None:
                continue
            buf[start1:end1, self.time] = dat[:n1]
            if n2 > 0:
                buf[start2:end2] = dat[n1:]
        
        self.time += 1
        if self.time == self.l_rollout:
            self.time = 0
            self.ptr = (self.ptr + n) % self.max_size
            self.size = min(self.size + n, self.max_size)
    
    def sample4wm(self, batch_size):
        # type: (int) -> Tuple[TensorDict, Tensor, Tensor, Tensor]
        ind = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        return self.expand_obs(self.obs[ind]), self.actions[ind], self.terminated[ind], self.rewards[ind]
    
    def sample4critic(self, batch_size):
        # type: (int) -> Tuple[Tensor, ...]
        ind = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        return tuple(map(lambda x: x[ind].transpose(0, 1).contiguous().clone().float(), 
            [self.states, self.next_values, self.rewards, self.dones, self.terminated]))

if __name__ == "__main__":
    B, L = 2, 16
    buffer = RolloutBufferGRID(L, 128, (3, (9, 16)), 10, 4, 10, torch.device("cpu"))
    obs = TensorDict({
        "state": torch.randn(B, L, 10),
        "perception": torch.randn(B, L, 9, 16),
        "occupancy": torch.randint(0, 2, (B, L, 10)).bool(),
        "visibility": torch.randint(0, 2, (B, L, 10)).bool()
    }, batch_size=(B, L))
    obs_restored = buffer.expand_obs(buffer.compress_obs(obs))
    assert torch.equal(obs["occupancy"], obs_restored["occupancy"])
    assert torch.equal(obs["visibility"], obs_restored["visibility"])