from typing import Union, Optional, Tuple

import torch
from torch import Tensor
from tensordict import TensorDict

class RNNStateBuffer:
    def __init__(self, l_rollout, n_envs, rnn_hidden_dim, rnn_n_layers, device):
        # type: (int, int, int, int, torch.device) -> None
        factory_kwargs = {"dtype": torch.float32, "device": device}
        self.actor_rnn_state  = torch.zeros((l_rollout, n_envs, rnn_n_layers, rnn_hidden_dim), **factory_kwargs)
        self.critic_rnn_state = torch.zeros((l_rollout, n_envs, rnn_n_layers, rnn_hidden_dim), **factory_kwargs)
    
    def clear(self):
        self.step = 0
    
    @torch.no_grad()
    def add(self, actor_hidden_state: Optional[Tensor], critic_hidden_state: Optional[Tensor]):
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
        # type: (int, int, int, int, torch.device) -> None
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
    def add(self, state, sample, logprob, reward, next_done, value, next_value):
        # type: (Union[Tensor, TensorDict], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> None
        self.obs[self.step] = state
        self.samples[self.step] = sample
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.next_dones[self.step] = next_done.float()
        self.values[self.step] = value
        self.next_values[self.step] = next_value
        self.step += 1


class RolloutBufferGRID:
    def __init__(
        self,
        l_rollout: int,
        buffer_size: int,
        obs_dim: Tuple[int, Tuple[int, int]],
        action_dim: int,
        grid_dim: int,
        device: torch.device
    ):
        factory_kwargs = {"dtype": torch.float32, "device": device}
        self.obs = TensorDict({
            "state":       torch.zeros((buffer_size, l_rollout, obs_dim[0]), **factory_kwargs),
            "perception":  torch.zeros((buffer_size, l_rollout, obs_dim[1][0], obs_dim[1][1]), **factory_kwargs),
            "grid":        torch.zeros((buffer_size, l_rollout, grid_dim), device=device, dtype=torch.bool),
            "visible_map": torch.zeros((buffer_size, l_rollout, grid_dim), device=device, dtype=torch.bool),
        }, batch_size=(buffer_size, l_rollout))
        self.dones = torch.zeros((buffer_size, l_rollout), device=device, dtype=torch.bool)
        self.actions = torch.zeros((buffer_size, l_rollout, action_dim), **factory_kwargs)
        self.rewards = torch.zeros((buffer_size, l_rollout), **factory_kwargs)
        self.device = device
        self.max_size = buffer_size
        self.size = 0
        self.ptr = 0
    
    @torch.no_grad()
    def add(self, obs, action, done, reward):
        # type: (TensorDict, Tensor, Tensor, Tensor) -> None
        n = obs.shape[0]
        start1, end1 = self.ptr, min(self.max_size, self.ptr + n)
        start2, end2 = 0, max(0, self.ptr + n - self.max_size)
        n1, n2 = end1 - start1, end2 - start2
        self.obs[start1:end1] = obs[:n1]
        self.dones[start1:end1] = done[:n1]
        self.rewards[start1:end1] = reward[:n1]
        self.actions[start1:end1] = action[:n1]
        if n2 > 0:
            self.obs[start2:end2] = obs[n1:]
            self.dones[start2:end2] = done[n1:]
            self.actions[start2:end2] = action[n1:]
            self.rewards[start2:end2] = reward[n1:]
        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)
    
    def sample(self, batch_size):
        # type: (int) -> Tuple[Tensor]
        ind = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        return self.obs[ind], self.actions[ind], self.dones[ind], self.rewards[ind]
