from typing import List, Tuple, Dict, Union, Optional

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from welford_torch import Welford

from quaddif.algo.PPO import PPOAgent
from quaddif.utils.nn import mlp

class PPORolloutBuffer:
    def __init__(self, num_steps, num_envs, state_dim, action_dim, device):
        factory_kwargs = {"dtype": torch.float32, "device": device}
        
        self.states = torch.zeros((num_steps, num_envs, state_dim), **factory_kwargs)
        self.samples = torch.zeros((num_steps, num_envs, action_dim), **factory_kwargs)
        self.logprobs = torch.zeros((num_steps, num_envs), **factory_kwargs)
        self.rewards = torch.zeros((num_steps, num_envs), **factory_kwargs)
        self.dones = torch.zeros((num_steps, num_envs), **factory_kwargs)
        self.values = torch.zeros((num_steps, num_envs), **factory_kwargs)
        self.clear()
    
    def clear(self):
        self.step = 0
        
    def add(self, state, sample, logprob, loss, done, value):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> None
        for buf, input in zip(
            [self.states, self.samples, self.logprobs, self.rewards, self.dones, self.values],
            [state, sample, logprob, loss, done, value]):
            buf[self.step] = input.detach().to(buf.dtype)
        self.step += 1

class APG:
    def __init__(
        self,
        cfg: DictConfig,
        state_dim: int,
        hidden_dim: List[int],
        action_dim: int,
        min_action: torch.Tensor,
        max_action: torch.Tensor,
        l_rollout: int,
        device: torch.device
    ):
        self.actor = mlp(state_dim, hidden_dim, action_dim, hidden_act=nn.ELU()).to(device)
        
        self.optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr)
        self.discount = cfg.gamma
        self.max_grad_norm = cfg.max_grad_norm
        self.min_action = min_action
        self.max_action = max_action
        self.l_rollout = l_rollout
        self.actor_loss = torch.zeros(1, device=device)
        self.device = device
    
    def act(self, state):
        # type: (Tensor) -> Tuple[Tensor, Dict[str, Tensor]]
        action = self.actor(state) * (self.max_action - self.min_action) + self.min_action
        return action, {}
    
    def record_loss(self, loss, info, extra):
        self.actor_loss += loss.mean()
    
    def update_actor(self):
        self.actor_loss = self.actor_loss / self.l_rollout
        self.optimizer.zero_grad()
        self.actor_loss.backward()
        grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.actor.parameters()]) ** 0.5
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        actor_loss = self.actor_loss.item()
        self.actor_loss = torch.zeros(1, device=self.device)
        return actor_loss, grad_norm

    @staticmethod
    def build(cfg, env, device):
        return APG(
            cfg=cfg.algo,
            state_dim=env.state_dim,
            hidden_dim=list(cfg.algo.hidden_dim),
            action_dim=env.action_dim,
            min_action=env.min_action,
            max_action=env.max_action,
            l_rollout=cfg.l_rollout,
            device=device)


class APG_stocastic:
    def __init__(
        self,
        cfg: DictConfig,
        state_dim: int,
        hidden_dim: List[int],
        action_dim: int,
        min_action: torch.Tensor,
        max_action: torch.Tensor,
        l_rollout: int,
        device: torch.device
    ):
        self.actor = mlp(state_dim, hidden_dim, action_dim, hidden_act=nn.ELU()).to(device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim, device=device))
        
        self.optimizer = torch.optim.Adam([
            {"params": self.actor.parameters()},
            {"params": self.actor_logstd}], lr=cfg.actor_lr)
        self.discount = cfg.gamma
        self.max_grad_norm = cfg.max_grad_norm
        self.entropy_weight = cfg.entropy_weight
        self.min_action = min_action
        self.max_action = max_action
        self.l_rollout = l_rollout
        self.actor_loss = torch.zeros(1, device=device)
        self.device = device

    def act(self, state, sample=None):
        # type: (Tensor, Optional[Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]
        action_mean = self.actor(state.view(-1, state.size(-1)))
        LOG_STD_MAX = 2
        LOG_STD_MIN = -5
        action_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            torch.tanh(self.actor_logstd) + 1)  # From SpinUp / Denis Yarats
        action_std = torch.exp(action_logstd).expand_as(action_mean)
        probs = torch.distributions.Normal(action_mean, action_std)
        if sample is None:
            sample = torch.randn_like(action_mean) * action_std + action_mean
        action = self.min_action + 0.5 * (self.max_action - self.min_action) * (torch.tanh(sample) + 1)
        logprob = probs.log_prob(sample) - torch.log(1. - torch.tanh(sample).pow(2) + 1e-8)
        entropy = probs.entropy().sum(-1)
        return action, {"sample": sample, "logprob": logprob.sum(-1), "entropy": entropy}
    
    def record_loss(self, loss, info, extra):
        self.actor_loss += (loss - self.entropy_weight * info["entropy"]).mean()
    
    def update_actor(self):
        self.actor_loss = self.actor_loss / self.l_rollout
        self.optimizer.zero_grad()
        self.actor_loss.backward()
        grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.actor.parameters()]) ** 0.5
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        actor_loss = self.actor_loss.item()
        self.actor_loss = torch.zeros(1, device=self.device)
        return actor_loss, grad_norm

    @staticmethod
    def build(cfg, env, device):
        return APG_stocastic(
            cfg=cfg.algo,
            state_dim=env.state_dim,
            hidden_dim=list(cfg.algo.hidden_dim),
            action_dim=env.action_dim,
            min_action=env.min_action,
            max_action=env.max_action,
            l_rollout=cfg.l_rollout,
            device=device)


class APG_PPO:
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
        self.buffer = PPORolloutBuffer(l_rollout, n_envs, state_dim, action_dim, device)
        
        self.var_tracker = Welford()
        
        self.discount = cfg.gamma
        self.lmbda = cfg.lmbda
        self.order_weight = cfg.order_weight
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
    def bootstrap(self, final_state, final_terminated):
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
        return advantages.view(-1), target_values.view(-1)
    
    def clear_loss(self):
        self.actor_loss = torch.tensor(0., device=self.device)
        self.rollout_gamma = torch.ones(self.n_envs, device=self.device)
    
    def record_loss(self, loss, info, extra):
        self.actor_loss += (self.rollout_gamma * loss).sum()
        self.rollout_gamma = torch.where(extra["reset"], 1, self.rollout_gamma * self.discount)
    
    def update_actor(self, advantages):
        states = self.buffer.states.view(-1, self.state_dim)
        samples = self.buffer.samples.view(-1, self.action_dim)
        logprobs = self.buffer.logprobs.view(-1)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        _, _, newlogprob, entropy, _ = self.agent.get_action_value(states, samples)
        logratio = newlogprob - logprobs
        ratio = logratio.exp()
        
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        T, N = self.l_rollout, self.n_envs
        actor_loss = torch.lerp(pg_loss, self.actor_loss / (T * N), self.order_weight)
        # actor_loss = self.actor_loss / (T * N)
        actor_loss = actor_loss - self.entropy_weight * entropy.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        grad = torch.cat([p.grad.data.flatten() for p in self.agent.actor_mean.parameters()], dim=0)
        self.var_tracker.add(grad)
        grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.agent.actor_mean.parameters()]) ** 0.5
        if self.actor_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.agent.actor_mean.parameters(), max_norm=self.actor_grad_norm)
        self.actor_optim.step()
        self.rollout_gamma[:] = 1.
        return actor_loss.item(), grad_norm

    def update_critic(self, target_values, clipped=False):
        T, N, D = self.l_rollout, self.n_envs, self.state_dim
        batch_indices = torch.randperm(T*N, device=self.device)
        mb_size = T*N // self.n_minibatch
        states = self.buffer.states.reshape(T*N, D)
        values = self.buffer.values.reshape(-1)
        for start in range(0, T*N, mb_size):
            end = start + mb_size
            mb_indices = batch_indices[start:end]
            newvalue = self.agent.get_value(states[mb_indices])
            if clipped:
                v_loss_unclipped = (newvalue - target_values[mb_indices]) ** 2
                v_clipped = values[mb_indices] + torch.clamp(
                    newvalue - values[mb_indices], -0.2, 0.2)
                v_loss_clipped = (v_clipped - target_values[mb_indices]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                critic_loss = 0.5 * v_loss_max.mean()
            else:
                critic_loss = F.mse_loss(newvalue, target_values[mb_indices])
            self.value_optim.zero_grad()
            critic_loss.backward()
            grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.agent.critic.parameters()]) ** 0.5
            if self.critic_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_norm=self.critic_grad_norm)
            self.value_optim.step()
        return critic_loss.item(), grad_norm
