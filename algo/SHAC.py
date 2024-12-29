from typing import Union, Sequence, Tuple, Dict, Optional
from copy import deepcopy
import os

from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn.functional as F
from tensordict import TensorDict

from quaddif.algo.buffer import RolloutBufferSHAC, RolloutBufferSHACQ, RNNStateBuffer
from quaddif.network.agents import (
    tensordict2tuple,
    StochasticActorCriticV,
    RPLActorCritic,
    StochasticActorCriticQ,
    PolicyExporter)

class SHAC:
    def __init__(
        self,
        cfg: DictConfig,
        state_dim: int,
        action_dim: int,
        n_envs: int,
        l_rollout: int,
        device: torch.device
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent = StochasticActorCriticV(cfg.network, state_dim, action_dim).to(device)
        if self.agent.is_rnn_based:
            self.rnn_state_buffer = RNNStateBuffer(l_rollout, n_envs, cfg.network.rnn_hidden_dim, cfg.network.rnn_n_layers, device)
        self.actor_optim = torch.optim.Adam(self.agent.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.agent.critic.parameters(), lr=cfg.critic_lr)
        self.buffer = RolloutBufferSHAC(l_rollout, n_envs, state_dim, device)
        self._critic_target = deepcopy(self.agent.critic)
        for p in self._critic_target.parameters():
            p.requires_grad_(False)
        
        self.discount: float = cfg.gamma
        self.lmbda: float = cfg.lmbda
        self.entropy_weight: float = cfg.entropy_weight
        self.actor_grad_norm: float = cfg.actor_grad_norm
        self.critic_grad_norm: float = cfg.critic_grad_norm
        self.n_minibatch: int = cfg.n_minibatch
        self.n_envs: int = n_envs
        self.l_rollout: int = l_rollout
        self.device = device
        
        self.actor_loss = torch.tensor(0., device=self.device)
        self.rollout_gamma = torch.ones(self.n_envs, device=self.device)
        self.cumulated_loss = torch.zeros(self.n_envs, device=self.device)
        self.entropy_loss = torch.tensor(0., device=self.device)
    
    def act(self, state, test=False):
        # type: (Union[Tensor, TensorDict], bool) -> Tuple[Tensor, Dict[str, Tensor]]
        if self.agent.is_rnn_based:
            self.rnn_state_buffer.add(self.agent.actor.actor_mean.hidden_state, self.agent.critic.critic.hidden_state)
        action, sample, logprob, entropy, value = self.agent.get_action_and_value(tensordict2tuple(state), test=test)
        return action, {"sample": sample, "logprob": logprob, "entropy": entropy, "value": value}
    
    def value_target(self, state):
        # type: (Tensor) -> Tensor
        return self._critic_target(state).squeeze(-1)
    
    @torch.no_grad()
    def bootstrap_tdlambda(self):
        # value of the next state should be zero if the next state is a terminal state
        next_values = self.buffer.next_values * (1 - self.buffer.next_terminated)
        if self.lmbda == 0.:
            target_values = self.buffer.rewards + self.discount * next_values
        else:
            target_values = torch.zeros_like(next_values).to(self.device)
            Ai = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)
            Bi = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)
            lam = torch.ones(self.n_envs, dtype=torch.float32, device=self.device)
            self.buffer.next_dones[-1] = 1.
            for i in reversed(range(self.l_rollout)):
                lam = lam * self.lmbda * (1. - self.buffer.next_dones[i]) + self.buffer.next_dones[i]
                Ai = (1. - self.buffer.next_dones[i]) * (
                    self.discount * (self.lmbda * Ai + next_values[i]) + \
                    (1. - lam) / (1. - self.lmbda) * self.buffer.rewards[i])
                Bi = self.discount * (next_values[i] * self.buffer.next_dones[i] + Bi * (1. - self.buffer.next_dones[i])) + \
                     self.buffer.rewards[i]
                # Bi = self.discount * torch.where(self.buffer.next_dones[i], next_values[i], Bi) + self.buffer.rewards[i]
                target_values[i] = (1.0 - self.lmbda) * Ai + lam * Bi
        return target_values.view(-1)
    
    @torch.no_grad()
    def bootstrap_gae(self):
        advantages = torch.zeros_like(self.buffer.rewards)
        lastgaelam = 0
        for t in reversed(range(self.l_rollout)):
            nextnonterminal = 1.0 - self.buffer.next_terminated[t]
            nextnonreset = 1.0 - self.buffer.next_dones[t]
            # nextnonterminal = 1.0 - self.buffer.next_dones[t]
            nextvalues = self.buffer.next_values[t]
            # TD-error / vanilla advantage function.
            delta = self.buffer.rewards[t] + self.discount * nextvalues * nextnonterminal - self.buffer.values[t]
            # Generalized Advantage Estimation bootstraping formula.
            advantages[t] = lastgaelam = delta + self.discount * self.lmbda * nextnonreset * lastgaelam
        target_values = advantages + self.buffer.values
        return target_values.view(-1)
    
    def record_loss(self, loss, policy_info, env_info, last_step=False):
        # type: (Tensor, Dict[str, Tensor], Dict[str, Tensor], Optional[bool]) -> Tensor
        reset = torch.ones_like(env_info["reset"]) if last_step else env_info["reset"]
        truncated = torch.ones_like(env_info["reset"]) if last_step else env_info["truncated"]
        # add cumulated loss if rollout ends or trajectory ends (terminated or truncated)
        self.cumulated_loss = self.cumulated_loss + self.rollout_gamma * loss
        cumulated_loss = self.cumulated_loss[reset].sum()
        # add terminal value if rollout ends or truncated
        next_value = self.value_target(tensordict2tuple(env_info["next_state_before_reset"]))
        terminal_value = (self.rollout_gamma * self.discount * next_value)[truncated].sum()
        assert terminal_value.requires_grad == True
        # add up the discounted cumulated loss, the terminal value and the entropy loss
        self.actor_loss = self.actor_loss + cumulated_loss + terminal_value
        self.entropy_loss = self.entropy_loss - policy_info["entropy"].sum()
        # reset the discount factor, clear the cumulated loss if trajectory ends
        self.rollout_gamma = torch.where(reset, 1, self.rollout_gamma * self.discount)
        self.cumulated_loss = torch.where(reset, 0, self.cumulated_loss)
        return next_value.detach()

    def clear_loss(self):
        self.rollout_gamma.fill_(1.)
        self.actor_loss.detach_().fill_(0.)
        self.cumulated_loss.detach_().fill_(0.)
        self.entropy_loss.detach_().fill_(0.)
        
    def update_actor(self) -> Dict[str, float]:
        actor_loss = self.actor_loss / (self.n_envs * self.l_rollout)
        entropy_loss = self.entropy_loss / (self.n_envs * self.l_rollout)
        total_loss = actor_loss + self.entropy_weight * entropy_loss
        self.actor_optim.zero_grad()
        total_loss.backward()
        grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.agent.actor.parameters()]) ** 0.5
        if self.actor_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), max_norm=self.actor_grad_norm)
        self.actor_optim.step()
        return {"actor_loss": actor_loss.item(), "entropy_loss": entropy_loss.item()}, {"actor_grad_norm": grad_norm}
    
    def update_critic(self, target_values: Tensor) -> Dict[str, float]:
        T, N = self.l_rollout, self.n_envs
        batch_indices = torch.randperm(T*N, device=self.device)
        mb_size = T*N // self.n_minibatch
        states = self.buffer.states.flatten(0, 1)
        if self.agent.is_rnn_based:
            critic_hidden_state = self.rnn_state_buffer.critic_rnn_state.flatten(0, 1)
        for start in range(0, T*N, mb_size):
            end = start + mb_size
            mb_indices = batch_indices[start:end]
            if self.agent.is_rnn_based:
                values = self.agent.get_value(tensordict2tuple(states[mb_indices]),
                                              critic_hidden_state[mb_indices].permute(1, 0, 2))
            else:
                values = self.agent.get_value(tensordict2tuple(states[mb_indices]))
            critic_loss = F.mse_loss(values, target_values[mb_indices])
            self.critic_optim.zero_grad()
            critic_loss.backward()
            grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.agent.critic.parameters()]) ** 0.5
            if self.critic_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_norm=self.critic_grad_norm)
            self.critic_optim.step()
        for p, p_t in zip(self.agent.critic.parameters(), self._critic_target.parameters()):
            p_t.data.lerp_(p.data, 5e-3)
        return {"critic_loss": critic_loss.item()}, {"critic_grad_norm": grad_norm}
    
    def step(self, cfg, env, state, on_step_cb=None):
        self.buffer.clear()
        if self.agent.is_rnn_based:
            self.rnn_state_buffer.clear()
        self.clear_loss()
        for t in range(cfg.l_rollout):
            action, policy_info = self.act(state)
            next_state, loss, terminated, env_info = env.step(action)
            next_value = self.record_loss(loss, policy_info, env_info, last_step=(t==cfg.l_rollout-1))
            # divide by 10 to avoid disstability
            self.buffer.add(state, loss/10, policy_info["value"], env_info["reset"], terminated, next_value)
            self.reset(env_info["reset"])
            state = next_state
            if on_step_cb is not None:
                on_step_cb(
                    state=state,
                    action=action,
                    policy_info=policy_info,
                    env_info=env_info)
        target_values = self.bootstrap_gae()
        actor_losses, actor_grad_norms = self.update_actor()
        critic_losses, critic_grad_norms = self.update_critic(target_values)
        self.detach()
        losses = {**actor_losses, **critic_losses}
        grad_norms = {**actor_grad_norms, **critic_grad_norms}
        return state, policy_info, env_info, losses, grad_norms

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.agent.save(path)
    
    def load(self, path):
        self.agent.load(path)
    
    def reset(self, env_idx: Tensor):
        if self.agent.is_rnn_based:
            self.agent.reset(env_idx)
    
    def detach(self):
        if self.agent.is_rnn_based:
            self.agent.detach()
            self._critic_target.detach()
        
    @staticmethod
    def build(cfg, env, device):
        return SHAC(
            cfg=cfg.algo,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_envs=env.n_envs,
            l_rollout=cfg.l_rollout,
            device=device)
    
    def export(self, path: str, verbose: bool = False):
        PolicyExporter(self.agent.actor).export(path, verbose=verbose)


class SHAC_RPL(SHAC):
    def __init__(
        self,
        cfg: DictConfig,
        state_dim: int,
        action_dim: int,
        n_envs: int,
        l_rollout: int,
        device: torch.device
    ):
        super().__init__(cfg, state_dim, action_dim, n_envs, l_rollout, device)
        del self.agent, self.actor_optim, self.critic_optim, self._critic_target
        self.agent = RPLActorCritic(cfg.network, cfg.anchor_ckpt, state_dim, cfg.anchor_state_dim, action_dim, cfg.rpl_action).to(device)
        self.actor_optim = torch.optim.Adam(self.agent.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.agent.critic.parameters(), lr=cfg.critic_lr)
        self._critic_target = deepcopy(self.agent.critic)
        for p in self._critic_target.parameters():
            p.requires_grad_(False)
    
    def value_target(self, state):
        # type: (Tensor) -> Tensor
        return self._critic_target(self.agent.rpl_obs(state)[0]).squeeze(-1)
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(os.path.join(path, "anchor_agent")):
            os.makedirs(os.path.join(path, "anchor_agent"))
        self.agent.save(path)
    
    @staticmethod
    def build(cfg, env, device):
        return SHAC_RPL(
            cfg=cfg.algo,
            state_dim=env.state_dim,
            hidden_dim=list(cfg.algo.network.hidden_dim),
            action_dim=env.action_dim,
            n_envs=env.n_envs,
            l_rollout=cfg.l_rollout,
            device=device)

class SHAC_Q:
    def __init__(
        self,
        cfg: DictConfig,
        state_dim: int,
        action_dim: int,
        n_envs: int,
        l_rollout: int,
        device: torch.device
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent = StochasticActorCriticQ(cfg.network, state_dim, action_dim).to(device)
        if self.agent.is_rnn_based:
            self.rnn_state_buffer = RNNStateBuffer(l_rollout, n_envs, cfg.network.rnn_hidden_dim, cfg.network.rnn_n_layers, device)
        self.actor_optim = torch.optim.Adam(self.agent.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.agent.critic.parameters(), lr=cfg.critic_lr)
        self.buffer = RolloutBufferSHACQ(l_rollout, n_envs, state_dim, action_dim, device)
        self.agent_target = deepcopy(self.agent)
        for p in self.agent_target.parameters():
            p.requires_grad_(False)
        
        self.discount: float = cfg.gamma
        self.entropy_weight: float = cfg.entropy_weight
        self.actor_grad_norm: float = cfg.actor_grad_norm
        self.critic_grad_norm: float = cfg.critic_grad_norm
        self.n_minibatch: int = cfg.n_minibatch
        self.n_envs: int = n_envs
        self.l_rollout: int = l_rollout
        self.device = device
        
        self.one_step_loss = torch.zeros(self.n_envs, device=self.device)
        self.next_values = torch.zeros(self.n_envs, device=self.device)
        self.entropy_loss = torch.zeros(self.n_envs, device=self.device)
    
    def act(self, state, test=False):
        # type: (Tensor, bool) -> Tuple[Tensor, Dict[str, Tensor]]
        if self.agent.is_rnn_based:
            self.rnn_state_buffer.add(self.agent.actor.actor_mean.hidden_state, self.agent.critic.critic.hidden_state)
        action, sample, logprob, entropy, value = self.agent.get_action_and_value(tensordict2tuple(state), test=test)
        return action, {"sample": sample, "logprob": logprob, "entropy": entropy, "value": value}
    
    def value_target(self, state, action):
        # type: (Union[Tensor, TensorDict], Tensor) -> Tensor
        return self.agent_target.get_value(tensordict2tuple(state), action)

    def record_loss(self, loss, policy_info, env_info, terminated):
        # type: (Tensor, Dict[str, Tensor], Dict[str, Tensor], Tensor) -> Tensor
        self.one_step_loss = self.one_step_loss + loss
        next_state = env_info["next_state_before_reset"].detach()
        next_values = self.agent.get_action_and_value(tensordict2tuple(next_state))[-1] * (1. - terminated.float())
        self.next_values = self.next_values + next_values
        self.entropy_loss = self.entropy_loss - policy_info["entropy"]

    def clear_loss(self):
        self.one_step_loss.detach_().fill_(0.)
        self.next_values.detach_().fill_(0.)
        self.entropy_loss.detach_().fill_(0.)
        
    def update_actor(self) -> Dict[str, float]:
        actor_loss = (self.one_step_loss + self.discount * self.next_values).sum() / (self.n_envs * self.l_rollout)
        entropy_loss = self.entropy_loss.sum() / (self.n_envs * self.l_rollout)
        total_loss = actor_loss + self.entropy_weight * entropy_loss
        self.actor_optim.zero_grad()
        total_loss.backward()
        grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.agent.actor.parameters()]) ** 0.5
        if self.actor_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), max_norm=self.actor_grad_norm)
        self.actor_optim.step()
        for p, p_t in zip(self.agent.actor.parameters(), self.agent_target.actor.parameters()):
            p_t.data.lerp_(p.data, 5e-3)
        return {"actor_loss": actor_loss.item(), "entropy_loss": entropy_loss.item()}, {"actor_grad_norm": grad_norm}
    
    def update_critic(self) -> Dict[str, float]:
        T, N = self.l_rollout, self.n_envs
        batch_indices = torch.randperm(T*N, device=self.device)
        mb_size = T*N // self.n_minibatch
        states = self.buffer.states.flatten(0, 1)
        actions = self.buffer.actions.flatten(0, 1)
        rewards = self.buffer.rewards.flatten(0, 1)
        next_states = self.buffer.next_states.flatten(0, 1)
        terminated = self.buffer.next_terminated.flatten(0, 1)
        if self.agent.is_rnn_based:
            critic_hidden_state = self.rnn_state_buffer.critic_rnn_state.flatten(0, 1)
        with torch.no_grad():
            next_values = self.agent_target.get_action_and_value(tensordict2tuple(next_states))[-1]
            target_values = rewards + (1 - terminated) * self.discount * next_values
        
        for start in range(0, T*N, mb_size):
            end = start + mb_size
            mb_indices = batch_indices[start:end]
            if self.agent.is_rnn_based:
                values = self.agent.get_value(
                    tensordict2tuple(states[mb_indices]),
                    actions[mb_indices],
                    critic_hidden_state[mb_indices].permute(1, 0, 2))
            else:
                values = self.agent.get_value(
                    tensordict2tuple(states[mb_indices]),
                    actions[mb_indices])
            critic_loss = F.mse_loss(values, target_values[mb_indices])
            self.critic_optim.zero_grad()
            critic_loss.backward()
            grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.agent.critic.parameters()]) ** 0.5
            if self.critic_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_norm=self.critic_grad_norm)
            self.critic_optim.step()
        for p, p_t in zip(self.agent.critic.parameters(), self.agent_target.critic.parameters()):
            p_t.data.lerp_(p.data, 5e-3)
        return {"critic_loss": critic_loss.item()}, {"critic_grad_norm": grad_norm}
    
    def step(self, cfg, env, state, on_step_cb=None):
        self.buffer.clear()
        if self.agent.is_rnn_based:
            self.rnn_state_buffer.clear()
        self.clear_loss()
        for t in range(cfg.l_rollout):
            action, policy_info = self.act(state)
            next_state, loss, terminated, env_info = env.step(action)
            self.record_loss(loss, policy_info, env_info, terminated)
            # divide by 10 to avoid disstability
            self.buffer.add(state, action, loss/10, env_info["next_state_before_reset"], terminated)
            self.reset(env_info["reset"])
            state = next_state
            if on_step_cb is not None:
                on_step_cb(
                    state=state,
                    action=action,
                    policy_info=policy_info,
                    env_info=env_info)
        actor_losses, actor_grad_norms = self.update_actor()
        critic_losses, critic_grad_norms = self.update_critic()
        self.detach()
        losses = {**actor_losses, **critic_losses}
        grad_norms = {**actor_grad_norms, **critic_grad_norms}
        return state, policy_info, env_info, losses, grad_norms

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.agent.save(path)
    
    def load(self, path):
        self.agent.load(path)
    
    def reset(self, env_idx: Tensor):
        if self.agent.is_rnn_based:
            self.agent.reset(env_idx)
    
    def detach(self):
        if self.agent.is_rnn_based:
            self.agent.detach()
            self.agent_target.detach()
        
    @staticmethod
    def build(cfg, env, device):
        return SHAC_Q(
            cfg=cfg.algo,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_envs=env.n_envs,
            l_rollout=cfg.l_rollout,
            device=device)
    
    def export(self, path: str, verbose: bool = False):
        PolicyExporter(self.agent.actor).export(path, verbose=verbose)