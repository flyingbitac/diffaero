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
    def __init__(self, l_rollout, n_envs, obs_dim, device):
        # type: (int, int, Union[int, Tuple[int, Tuple[int, int]]], torch.device) -> None
        factory_kwargs = {"dtype": torch.float32, "device": device}
        
        assert isinstance(obs_dim, tuple) or isinstance(obs_dim, int)
        if isinstance(obs_dim, tuple):
            self.obs = TensorDict({
                "state": torch.zeros((l_rollout, n_envs, obs_dim[0]), **factory_kwargs),
                "perception": torch.zeros((l_rollout, n_envs, obs_dim[1][0], obs_dim[1][1]), **factory_kwargs)
            }, batch_size=(l_rollout, n_envs))
        else:
            self.obs = torch.zeros((l_rollout, n_envs, obs_dim), **factory_kwargs)
        self.rewards = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.values = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.next_dones = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.next_terminated = torch.zeros((l_rollout, n_envs), **factory_kwargs)
        self.next_values = torch.zeros((l_rollout, n_envs), **factory_kwargs)
    
    def clear(self):
        self.step = 0
    
    @torch.no_grad()
    def add(self, state, reward, value, next_done, next_terminated, next_value):
        # type: (Union[Tensor, TensorDict], Tensor, Tensor, Tensor, Tensor, Tensor) -> None
        self.obs[self.step] = state
        self.rewards[self.step] = reward
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