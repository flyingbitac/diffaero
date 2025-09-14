from termios import N_SLIP
from typing import Union, Sequence, Tuple, Dict, Optional
from copy import deepcopy
import os

from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn.functional as F
from tensordict import TensorDict

from diffaero.algo.buffer import RolloutBufferMASHAC, RNNStateBuffer
from diffaero.network.agents import tensordict2tuple
from diffaero.network.multiagents import MAStochasticActorCriticV
from diffaero.utils.runner import timeit
from diffaero.utils.exporter import PolicyExporter


class MASHAC:
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: int,
        global_state_dim: int,
        n_agents: int,
        action_dim: int,
        n_envs: int,
        l_rollout: int,
        device: torch.device
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.agent = MAStochasticActorCriticV(cfg.network, obs_dim, global_state_dim, action_dim).to(device)
        if self.agent.is_rnn_based:
            self.rnn_state_buffer = RNNStateBuffer(l_rollout, n_envs, cfg.network.rnn_hidden_dim, cfg.network.rnn_n_layers, device)
        self.actor_optim = torch.optim.Adam(self.agent.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.agent.critic.parameters(), lr=cfg.critic_lr)
        self.buffer = RolloutBufferMASHAC(l_rollout, n_envs, obs_dim, global_state_dim, n_agents, device)
        self._critic_target = deepcopy(self.agent.critic)
        for p in self._critic_target.parameters():
            p.requires_grad_(False)
        
        self.discount: float = cfg.gamma
        self.lmbda: float = cfg.lmbda
        self.entropy_weight: float = cfg.entropy_weight
        self.actor_grad_norm: float = cfg.actor_grad_norm
        self.critic_grad_norm: float = cfg.critic_grad_norm
        self.target_update_rate: float = cfg.target_update_rate
        self.n_minibatch: int = cfg.n_minibatch
        self.n_envs: int = n_envs
        self.l_rollout: int = l_rollout
        self.device = device
        
        self.actor_loss = torch.tensor(0., device=self.device)
        self.rollout_gamma = torch.ones(self.n_envs, device=self.device)
        self.cumulated_loss = torch.zeros(self.n_envs, device=self.device)
        self.entropy_loss = torch.tensor(0., device=self.device)
    
    def act(self, obs, global_state, test=False):
        # type: (Union[Tensor, TensorDict], Tensor, bool) -> Tuple[Tensor, Dict[str, Tensor]]
        if self.agent.is_rnn_based:
            self.rnn_state_buffer.add(self.agent.actor.actor_mean.hidden_state, self.agent.critic.critic.hidden_state)
        action, sample, logprob, entropy, value = self.agent.get_action_and_value(tensordict2tuple(obs), global_state, test=test)
        return action, {"sample": sample, "logprob": logprob, "entropy": entropy, "value": value}
    
    def value_target(self, global_state):
        # type: (Tensor) -> Tensor
        return self._critic_target(global_state).squeeze(-1)
    
    @torch.no_grad()
    def bootstrap_tdlambda(self):
        # value of the next obs should be zero if the next obs is a terminal obs
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
        global_states = self.buffer.global_states.flatten(0, 1)
        if self.agent.is_rnn_based:
            critic_hidden_state = self.rnn_state_buffer.critic_rnn_state.flatten(0, 1)
        for start in range(0, T*N, mb_size):
            end = start + mb_size
            mb_indices = batch_indices[start:end]
            if self.agent.is_rnn_based:
                values = self.agent.get_value(global_states[mb_indices],
                                              critic_hidden_state[mb_indices].permute(1, 0, 2))
            else:
                values = self.agent.get_value(global_states[mb_indices])
            critic_loss = F.mse_loss(values, target_values[mb_indices])
            self.critic_optim.zero_grad()
            critic_loss.backward()
            grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.agent.critic.parameters()]) ** 0.5
            if self.critic_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_norm=self.critic_grad_norm)
            self.critic_optim.step()
        for p, p_t in zip(self.agent.critic.parameters(), self._critic_target.parameters()):
            p_t.data.lerp_(p.data, self.target_update_rate)
        return {"critic_loss": critic_loss.item()}, {"critic_grad_norm": grad_norm}
    
    @timeit
    def step(self, cfg, env, logger, obs, on_step_cb=None):
        obs, global_state = obs
        self.buffer.clear()
        if self.agent.is_rnn_based:
            self.rnn_state_buffer.clear()
        self.clear_loss()
        for t in range(cfg.l_rollout):
            action, policy_info = self.act(obs, global_state)
            (next_obs, next_global_state), (loss, reward), terminated, env_info = env.step(env.rescale_action(action), next_state_before_reset=True)
            next_value = self.record_loss(loss, policy_info, env_info, last_step=(t==cfg.l_rollout-1))
            # divide by 10 to avoid disstability
            self.buffer.add(
                obs=obs,
                global_state=global_state,
                reward=loss/10,
                value=policy_info["value"],
                next_done=env_info["reset"],
                next_terminated=terminated,
                next_value=next_value
            )
            self.reset(env_info["reset"])
            obs = next_obs
            global_state = next_global_state
            if on_step_cb is not None:
                on_step_cb(
                    obs=obs,
                    action=action,
                    policy_info=policy_info,
                    env_info=env_info)
        target_values = self.bootstrap_gae()
        actor_losses, actor_grad_norms = self.update_actor()
        critic_losses, critic_grad_norms = self.update_critic(target_values)
        self.detach()
        losses = {**actor_losses, **critic_losses}
        grad_norms = {**actor_grad_norms, **critic_grad_norms}
        return (obs, global_state), policy_info, env_info, losses, grad_norms

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
        return MASHAC(
            cfg=cfg,
            obs_dim=env.obs_dim,
            global_state_dim=env.global_state_dim,
            n_agents=env.n_agents,
            action_dim=env.action_dim,
            n_envs=env.n_envs,
            l_rollout=cfg.l_rollout,
            device=device)
    
    def export(
        self,
        path: str,
        export_cfg: DictConfig,
        verbose: bool = False,
    ):
        PolicyExporter(self.agent.actor).export(path, export_cfg, verbose)