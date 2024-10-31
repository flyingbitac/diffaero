from typing import Tuple, Dict, Union, Optional
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import Tensor

from quaddif.utils.nn import PPOAgent


class SHACRolloutBuffer:
    def __init__(self, num_steps, num_envs, state_dim, device):
        factory_kwargs = {"dtype": torch.float32, "device": device}
        self.states = torch.zeros((num_steps, num_envs, state_dim), **factory_kwargs)
        self.rewards = torch.zeros((num_steps, num_envs), **factory_kwargs)
        self.dones = torch.zeros((num_steps, num_envs), **factory_kwargs)
        self.values = torch.zeros((num_steps, num_envs), **factory_kwargs)
        self.clear()
    
    def clear(self):
        self.step = 0
    
    def add(self, state, reward, done, value):
        self.states[self.step] = state
        self.rewards[self.step] = reward
        self.dones[self.step] = done.float()
        self.values[self.step] = value
        self.step += 1


class SHAC:
    def __init__(
        self,
        cfg,
        state_dim,
        action_dim,
        min_action,
        max_action,
        n_envs,
        l_rollout,
        device
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent = PPOAgent(
            state_dim, action_dim, min_action, max_action).to(device)
        self.actor_optim = torch.optim.Adam([
            {"params": self.agent.actor_mean.parameters()},
            {"params": self.agent.actor_logstd}], lr=cfg.actor_lr)
        self.value_optim = torch.optim.Adam(self.agent.critic.parameters(), lr=cfg.critic_lr)
        self.buffer = SHACRolloutBuffer(l_rollout, n_envs, state_dim, device)
        self._critic_target = deepcopy(self.agent.critic)
        for p in self._critic_target.parameters():
            p.requires_grad_(False)
        
        self.discount = cfg.gamma
        self.lmbda = cfg.lmbda
        self.entropy_weight = cfg.entropy_weight
        self.actor_grad_norm = cfg.actor_grad_norm
        self.critic_grad_norm = cfg.critic_grad_norm
        self.n_minibatch = cfg.n_minibatch
        self.n_envs = n_envs
        self.l_rollout = l_rollout
        self.device = device
        self.clear_loss()
    
    def act(self, state):
        # type: (Tensor) -> Tuple[Tensor, Dict[str, Tensor]]
        action, sample, logprob, entropy, value = self.agent.get_action_value(state)
        return action, {"sample": sample, "logprob": logprob, "entropy": entropy, "value": value}
    
    @torch.no_grad
    def bootstrap1(self):
        """GAE with lambda"""
        if self.lmbda == 0.:
            target_values = self.buffer.rewards + self.discount * (1 - self.buffer.dones) * self.buffer.values
        else:
            target_values = torch.zeros_like(self.buffer.values).to(self.device)
            Ai = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)
            Bi = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)
            lam = torch.ones(self.n_envs, dtype=torch.float32, device=self.device)
            self.buffer.dones[-1] = 1.
            for i in reversed(range(self.l_rollout)):
                lam = lam * self.lmbda * (1. - self.buffer.dones[i]) + self.buffer.dones[i]
                Ai = (1. - self.buffer.dones[i]) * (
                    self.discount * (self.lmbda * Ai + self.buffer.values[i]) + \
                    (1. - lam) / (1. - self.lmbda) * self.buffer.rewards[i])
                Bi = self.discount * (self.buffer.values[i] * self.buffer.dones[i] + Bi * (1.0 - self.buffer.dones[i])) + self.buffer.rewards[i]
                target_values[i] = (1.0 - self.lmbda) * Ai + lam * Bi
        return target_values
    
    @torch.no_grad
    def bootstrap2(self, final_state, final_terminated):
        """lambda return"""
        final_value = self.agent.get_value(final_state)
        advantages = torch.zeros_like(self.buffer.rewards)
        lastgaelam = 0
        for t in reversed(range(self.l_rollout)):
            if t == self.l_rollout - 1:
                nextnonterminal = 1.0 - final_terminated.float()
                nextvalues = final_value
            else:
                nextnonterminal = 1.0 - self.buffer.dones[t + 1]
                nextvalues = self.buffer.values[t + 1]
            # TD-error / vanilla advantage function.
            delta = self.buffer.rewards[t] + self.discount * nextvalues * nextnonterminal - self.buffer.values[t]
            # Generalized Advantage Estimation bootstraping formula.
            advantages[t] = lastgaelam = delta + self.discount * self.lmbda * nextnonterminal * lastgaelam
        target_values = advantages + self.buffer.values
        return target_values.view(-1)
    
    def clear_loss(self):
        self.actor_loss = torch.tensor(0., device=self.device)
        self.rollout_gamma = torch.ones(self.n_envs, device=self.device)
        self.cumulated_loss = torch.zeros(self.n_envs, device=self.device)
    
    def record_loss(self, loss, info, extra, last_step=False):
        self.cumulated_loss = self.cumulated_loss + self.rollout_gamma * loss
        self.rollout_gamma = self.rollout_gamma * self.discount
        reset = torch.ones_like(extra["reset"]) if last_step else extra["reset"]
        terminal_value = self.rollout_gamma * self.agent.get_value(extra["state_before_reset"])
        terminal_value = terminal_value[reset].sum() if last_step else terminal_value[extra["truncated"]].sum()
        cumulated_loss = self.cumulated_loss[reset].sum()
        self.actor_loss = self.actor_loss + cumulated_loss + terminal_value - self.entropy_weight * info["entropy"].sum()
        # self.actor_loss = self.actor_loss + cumulated_loss - self.entropy_weight * info["entropy"].sum()
        self.rollout_gamma = torch.where(reset, 1, self.rollout_gamma)
        self.cumulated_loss = torch.where(reset, 0, self.cumulated_loss)
    
    def update_actor(self):
        actor_loss = self.actor_loss / (self.n_envs * self.l_rollout)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.agent.actor_mean.parameters()]) ** 0.5
        if self.actor_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.agent.actor_mean.parameters(), max_norm=self.actor_grad_norm)
        self.actor_optim.step()
        return actor_loss.item(), grad_norm
    
    def update_critic(self, target_values):
        T, N, D = self.buffer.states.shape
        batch_indices = torch.randperm(T*N, device=self.device)
        mb_size = T*N // self.n_minibatch
        states = self.buffer.states.reshape(T*N, D)
        for start in range(0, T*N, mb_size):
            end = start + mb_size
            mb_indices = batch_indices[start:end]
            values = self.agent.get_value(states[mb_indices])
            critic_loss = F.mse_loss(values, target_values[mb_indices])
            self.value_optim.zero_grad()
            critic_loss.backward()
            grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.agent.critic.parameters()]) ** 0.5
            if self.critic_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_norm=self.critic_grad_norm)
            self.value_optim.step()
        for p, p_t in zip(self.agent.critic.parameters(), self._critic_target.parameters()):
            p_t.data.lerp_(p.data, 5e-3)
        return critic_loss.item(), grad_norm
