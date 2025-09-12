from typing import Union, List, Tuple, Dict
from collections import defaultdict
import os

from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn.functional as F
from tensordict import TensorDict

from diffaero.algo.buffer import RolloutBufferPPO, RolloutBufferAPPO, RNNStateBuffer
from diffaero.network.agents import (
    tensordict2tuple,
    StochasticActorCriticV,
    StochasticAsymmetricActorCriticV,
    RPLActorCritic)
from diffaero.utils.runner import timeit
from diffaero.utils.exporter import PolicyExporter

class PPO:
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: int,
        action_dim: int,
        n_envs: int,
        l_rollout: int,
        device: torch.device
    ):
        self.agent = StochasticActorCriticV(cfg.network, obs_dim, action_dim).to(device)
        self.optim = torch.optim.Adam(self.agent.parameters(), lr=cfg.lr, eps=cfg.eps)
        self.buffer = RolloutBufferPPO(l_rollout, n_envs, obs_dim, action_dim, device)
        if self.agent.is_rnn_based:
            self.rnn_state_buffer = RNNStateBuffer(l_rollout, n_envs, cfg.network.rnn_hidden_dim, cfg.network.rnn_n_layers, device)
        
        self.discount = cfg.gamma
        self.lmbda = cfg.lmbda
        self.entropy_weight = cfg.entropy_weight
        self.value_weight = cfg.value_weight
        self.actor_grad_norm = cfg.actor_grad_norm
        self.critic_grad_norm = cfg.critic_grad_norm
        self.clip_coef = cfg.clip_coef
        self.clip_value_loss = cfg.clip_value_loss
        self.norm_adv = cfg.norm_adv
        self.n_minibatch = cfg.n_minibatch
        self.n_envs = n_envs
        self.l_rollout = l_rollout
        self.device = device
    
    def act(self, obs, test=False):
        # type: (Union[Tensor, TensorDict], bool) -> Tuple[Tensor, Dict[str, Tensor]]
        if self.agent.is_rnn_based and not test:
            self.rnn_state_buffer.add(self.agent.actor.actor_mean.hidden_state, self.agent.critic.critic.hidden_state)
        action, sample, logprob, entropy, value = self.agent.get_action_and_value(tensordict2tuple(obs), test=test)
        return action, {"sample": sample, "logprob": logprob, "entropy": entropy, "value": value}
    
    @torch.no_grad()
    def bootstrap(self):
        advantages = torch.zeros_like(self.buffer.rewards)
        lastgaelam = 0
        for t in reversed(range(self.l_rollout)):
            nextnonterminal = 1.0 - self.buffer.next_dones[t]
            nextvalues = self.buffer.next_values[t]
            # TD-error / vanilla advantage function.
            delta = self.buffer.rewards[t] + self.discount * nextvalues * nextnonterminal - self.buffer.values[t]
            # Generalized Advantage Estimation bootstraping formula.
            advantages[t] = lastgaelam = delta + self.discount * self.lmbda * nextnonterminal * lastgaelam
        target_values = advantages + self.buffer.values
        return advantages.view(-1), target_values.view(-1)
    
    @timeit
    def train(self, advantages, target_values):
        # type: (Tensor, Tensor) -> Tuple[Dict[str, float], Dict[str, float]]
        T, N = self.l_rollout, self.n_envs
        obs = self.buffer.obs.flatten(0, 1)
        samples = self.buffer.samples.flatten(0, 1)
        logprobs = self.buffer.logprobs.flatten(0, 1)
        values = self.buffer.values.flatten(0, 1)
        if self.agent.is_rnn_based:
            actor_hidden_state = self.rnn_state_buffer.actor_rnn_state.flatten(0, 1)
            critic_hidden_state = self.rnn_state_buffer.critic_rnn_state.flatten(0, 1)
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch_indices = torch.randperm(T*N, device=self.device)
        mb_size = T*N // self.n_minibatch
        losses = defaultdict(list)
        grad_norms = defaultdict(list)
        
        for start in range(0, T*N, mb_size):
            end = start + mb_size
            mb_indices = batch_indices[start:end]
            # policy loss
            if self.agent.is_rnn_based:
                _, _, newlogprob, entropy = self.agent.get_action(
                    tensordict2tuple(obs[mb_indices]),
                    samples[mb_indices],
                    hidden=actor_hidden_state[mb_indices].permute(1, 0, 2))
            else:
                _, _, newlogprob, entropy = self.agent.get_action(
                    tensordict2tuple(obs[mb_indices]),
                    samples[mb_indices])
            
            logratio = newlogprob - logprobs[mb_indices]
            ratio = logratio.exp()
            advantages_mb = advantages[mb_indices]
            pg_loss1 = -advantages_mb * ratio
            pg_loss2 = -advantages_mb * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            # entropy loss
            entropy_loss = -entropy.mean()
            # value loss
            hidden = critic_hidden_state[mb_indices].permute(1, 0, 2) if self.agent.is_rnn_based else None
            newvalue = self.agent.get_value(tensordict2tuple(obs[mb_indices]), hidden=hidden)
            if self.clip_value_loss:
                v_loss_unclipped = (newvalue - target_values[mb_indices]) ** 2
                v_clipped = values[mb_indices] + torch.clamp(
                    newvalue - values[mb_indices], -self.clip_coef, self.clip_coef)
                v_loss_clipped = (v_clipped - target_values[mb_indices]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = F.mse_loss(newvalue, target_values[mb_indices])
            # total loss
            loss = pg_loss + self.value_weight * v_loss + self.entropy_weight * entropy_loss
            self.optim.zero_grad()
            loss.backward()
            
            actor_grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.agent.actor.parameters()]) ** 0.5
            if self.actor_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), max_norm=self.actor_grad_norm)
            critic_grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.agent.critic.parameters()]) ** 0.5
            if self.critic_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_norm=self.critic_grad_norm)
            
            self.optim.step()
            
            losses["actor_loss"].append(pg_loss.item())
            losses["entropy_loss"].append(entropy_loss.item())
            losses["critic_loss"].append(v_loss.item())
            grad_norms["actor_grad_norm"].append(actor_grad_norm)
            grad_norms["critic_grad_norm"].append(critic_grad_norm)
        losses = {k: sum(v) / len(v) for k, v in losses.items()}
        grad_norms = {k: sum(v) / len(v) for k, v in grad_norms.items()}
        return losses, grad_norms
    
    @timeit
    def step(self, cfg, env, logger, obs, on_step_cb=None):
        self.buffer.clear()
        if self.agent.is_rnn_based:
            self.rnn_state_buffer.clear()
        with torch.no_grad():
            for t in range(cfg.l_rollout):
                action, policy_info = self.act(obs)
                next_obs, (loss, reward), terminated, env_info = env.step(env.rescale_action(action), next_obs_before_reset=True)
                self.buffer.add(
                    obs=obs,
                    sample=policy_info["sample"],
                    logprob=policy_info["logprob"],
                    reward=reward,
                    next_done=terminated,
                    value=policy_info["value"],
                    next_value=self.agent.get_value(tensordict2tuple(env_info["next_obs_before_reset"])))
                obs = next_obs
                self.reset(env_info["reset"])
                if on_step_cb is not None:
                    on_step_cb(
                        obs=obs,
                        action=action,
                        policy_info=policy_info,
                        env_info=env_info)
            
        advantages, target_values = self.bootstrap()
        for _ in range(cfg.algo.n_epoch):
            losses, grad_norms = self.train(advantages, target_values)
        self.agent.detach()
        return obs, policy_info, env_info, losses, grad_norms
    
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

    @staticmethod
    def build(cfg, env, device):
        return PPO(
            cfg=cfg,
            obs_dim=env.obs_dim,
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

class AsymmetricPPO(PPO):
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        n_envs: int,
        l_rollout: int,
        device: torch.device
    ):
        self.agent = StochasticAsymmetricActorCriticV(cfg.network, cfg.critic_network, obs_dim, state_dim, action_dim).to(device)
        self.optim = torch.optim.Adam(self.agent.parameters(), lr=cfg.lr, eps=cfg.eps)
        self.buffer = RolloutBufferAPPO(l_rollout, n_envs, obs_dim, state_dim, action_dim, device)
        if self.agent.is_rnn_based:
            self.rnn_state_buffer = RNNStateBuffer(l_rollout, n_envs, cfg.network.rnn_hidden_dim, cfg.network.rnn_n_layers, device)
        
        self.discount = cfg.gamma
        self.lmbda = cfg.lmbda
        self.entropy_weight = cfg.entropy_weight
        self.value_weight = cfg.value_weight
        self.actor_grad_norm = cfg.actor_grad_norm
        self.critic_grad_norm = cfg.critic_grad_norm
        self.clip_coef = cfg.clip_coef
        self.clip_value_loss = cfg.clip_value_loss
        self.norm_adv = cfg.norm_adv
        self.n_minibatch = cfg.n_minibatch
        self.n_envs = n_envs
        self.l_rollout = l_rollout
        self.device = device
    
    def act(self, obs, test=False):
        # type: (Union[Tensor, TensorDict], bool) -> Tuple[Tensor, Dict[str, Tensor]]
        if self.agent.is_rnn_based and not test:
            self.rnn_state_buffer.add(self.agent.actor.actor_mean.hidden_state)
        action, sample, logprob, entropy = self.agent.get_action(tensordict2tuple(obs), test=test)
        return action, {"sample": sample, "logprob": logprob, "entropy": entropy}
    
    @timeit
    def train(self, advantages, target_values):
        # type: (Tensor, Tensor) -> Tuple[Dict[str, float], Dict[str, float]]
        T, N = self.l_rollout, self.n_envs
        obs = self.buffer.obs.flatten(0, 1)
        state = self.buffer.states.flatten(0, 1)
        samples = self.buffer.samples.flatten(0, 1)
        logprobs = self.buffer.logprobs.flatten(0, 1)
        values = self.buffer.values.flatten(0, 1)
        if self.agent.is_rnn_based:
            actor_hidden_state = self.rnn_state_buffer.actor_rnn_state.flatten(0, 1)
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch_indices = torch.randperm(T*N, device=self.device)
        mb_size = T*N // self.n_minibatch
        losses = defaultdict(list)
        grad_norms = defaultdict(list)
        
        for start in range(0, T*N, mb_size):
            end = start + mb_size
            mb_indices = batch_indices[start:end]
            # policy loss
            if self.agent.is_rnn_based:
                _, _, newlogprob, entropy = self.agent.get_action(
                    tensordict2tuple(obs[mb_indices]),
                    samples[mb_indices],
                    hidden=actor_hidden_state[mb_indices].permute(1, 0, 2))
            else:
                _, _, newlogprob, entropy = self.agent.get_action(
                    tensordict2tuple(obs[mb_indices]),
                    samples[mb_indices])
            
            logratio = newlogprob - logprobs[mb_indices]
            ratio = logratio.exp()
            advantages_mb = advantages[mb_indices]
            pg_loss1 = -advantages_mb * ratio
            pg_loss2 = -advantages_mb * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            # entropy loss
            entropy_loss = -entropy.mean()
            # value loss
            newvalue = self.agent.get_value(state[mb_indices])
            if self.clip_value_loss:
                v_loss_unclipped = (newvalue - target_values[mb_indices]) ** 2
                v_clipped = values[mb_indices] + torch.clamp(
                    newvalue - values[mb_indices], -self.clip_coef, self.clip_coef)
                v_loss_clipped = (v_clipped - target_values[mb_indices]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = F.mse_loss(newvalue, target_values[mb_indices])
            # total loss
            loss = pg_loss + self.value_weight * v_loss + self.entropy_weight * entropy_loss
            self.optim.zero_grad()
            loss.backward()
            
            actor_grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.agent.actor.parameters()]) ** 0.5
            if self.actor_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), max_norm=self.actor_grad_norm)
            critic_grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.agent.critic.parameters()]) ** 0.5
            if self.critic_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_norm=self.critic_grad_norm)
            
            self.optim.step()
            
            losses["actor_loss"].append(pg_loss.item())
            losses["entropy_loss"].append(entropy_loss.item())
            losses["critic_loss"].append(v_loss.item())
            grad_norms["actor_grad_norm"].append(actor_grad_norm)
            grad_norms["critic_grad_norm"].append(critic_grad_norm)
        losses = {k: sum(v) / len(v) for k, v in losses.items()}
        grad_norms = {k: sum(v) / len(v) for k, v in grad_norms.items()}
        return losses, grad_norms
    
    @timeit
    def step(self, cfg, env, logger, obs, on_step_cb=None):
        self.buffer.clear()
        if self.agent.is_rnn_based:
            self.rnn_state_buffer.clear()
        with torch.no_grad():
            for t in range(cfg.l_rollout):
                action, policy_info = self.act(obs)
                state = env.get_state()
                with torch.no_grad():
                    value = self.agent.get_value(state)
                next_obs, (loss, reward), terminated, env_info = env.step(env.rescale_action(action), next_state_before_reset=True)
                self.buffer.add(
                    obs=obs,
                    state=state,
                    sample=policy_info["sample"],
                    logprob=policy_info["logprob"],
                    reward=reward,
                    next_done=terminated,
                    value=value,
                    next_value=self.agent.get_value(env_info["next_state_before_reset"]))
                obs = next_obs
                self.reset(env_info["reset"])
                if on_step_cb is not None:
                    on_step_cb(
                        obs=obs,
                        action=action,
                        policy_info=policy_info,
                        env_info=env_info)
            
        logger.log_scalar("value", value.mean().item())
        advantages, target_values = self.bootstrap()
        for _ in range(cfg.algo.n_epoch):
            losses, grad_norms = self.train(advantages, target_values)
        self.agent.detach()
        return obs, policy_info, env_info, losses, grad_norms

    @staticmethod
    def build(cfg, env, device):
        return AsymmetricPPO(
            cfg=cfg,
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_envs=env.n_envs,
            l_rollout=cfg.l_rollout,
            device=device)
