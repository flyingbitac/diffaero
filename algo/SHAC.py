from typing import Callable, Sequence, Tuple, Dict, Optional
from copy import deepcopy

from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm

from quaddif.env.base_env import BaseEnv
from quaddif.utils.nn import StochasticActorCritic
from quaddif.utils.logger import Logger

class SHACRolloutBuffer:
    def __init__(self, num_steps, num_envs, state_dim, device):
        factory_kwargs = {"dtype": torch.float32, "device": device}
        self.states = torch.zeros((num_steps, num_envs, state_dim), **factory_kwargs)
        self.rewards = torch.zeros((num_steps, num_envs), **factory_kwargs)
        self.values = torch.zeros((num_steps, num_envs), **factory_kwargs)
        self.next_dones = torch.zeros((num_steps, num_envs), **factory_kwargs)
        self.next_terminated = torch.zeros((num_steps, num_envs), **factory_kwargs)
        self.next_values = torch.zeros((num_steps, num_envs), **factory_kwargs)
    
    def clear(self):
        self.step = 0
    
    @torch.no_grad()
    def add(self, state, reward, value, next_done, next_terminated, next_value):
        self.states[self.step] = state
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.next_dones[self.step] = next_done.float()
        self.next_terminated[self.step] = next_terminated.float()
        self.next_values[self.step] = next_value
        self.step += 1


class SHAC:
    def __init__(
        self,
        cfg: DictConfig,
        state_dim: int,
        hidden_dim: Sequence[int],
        action_dim: int,
        n_envs: int,
        l_rollout: int,
        device: torch.device
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent = StochasticActorCritic(state_dim, hidden_dim, action_dim).to(device)
        self.optim = torch.optim.Adam([
            {"params": self.agent.actor_mean.parameters(), "lr": cfg.actor_lr},
            {"params": self.agent.actor_logstd, "lr": cfg.actor_lr},
            {"params": self.agent.critic.parameters(), "lr": cfg.critic_lr}])
        self.buffer = SHACRolloutBuffer(l_rollout, n_envs, state_dim, device)
        self._critic_target = deepcopy(self.agent.critic)
        for p in self._critic_target.parameters():
            p.requires_grad_(False)
        
        self.discount: float = cfg.gamma
        self.lmbda: float = cfg.lmbda
        self.entropy_weight: float = cfg.entropy_weight
        self.value_weight: float = cfg.value_weight
        self.actor_grad_norm: float = cfg.actor_grad_norm
        self.critic_grad_norm: float = cfg.critic_grad_norm
        self.n_minibatch: int = cfg.n_minibatch
        self.n_envs: int = n_envs
        self.l_rollout: int = l_rollout
        self.device = device
    
    def act(self, state):
        # type: (Tensor) -> Tuple[Tensor, Dict[str, Tensor]]
        action, sample, logprob, entropy, value = self.agent.get_action_and_value(state)
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
    
    def record_loss(self, loss, info, extra, last_step=False):
        # type: (Tensor, Dict[str, Tensor], Dict[str, Tensor], Optional[bool]) -> Tensor
        reset = torch.ones_like(extra["reset"]) if last_step else extra["reset"]
        truncated = torch.ones_like(extra["reset"]) if last_step else extra["truncated"]
        # add cumulated loss if rollout ends or trajectory ends (terminated or truncated)
        self.cumulated_loss = self.cumulated_loss + self.rollout_gamma * loss
        cumulated_loss = self.cumulated_loss[reset].sum()
        # add terminal value if rollout ends or truncated
        next_value = self.value_target(extra["next_state_before_reset"])
        terminal_value = (self.rollout_gamma * self.discount * next_value)[truncated].sum()
        assert terminal_value.requires_grad == True
        # add up the discounted cumulated loss, the terminal value and the entropy loss
        self.actor_loss = self.actor_loss + cumulated_loss + terminal_value
        self.entropy_loss = self.entropy_loss - info["entropy"].sum()
        # reset the discount factor, clear the cumulated loss if trajectory ends
        self.rollout_gamma = torch.where(reset, 1, self.rollout_gamma * self.discount)
        self.cumulated_loss = torch.where(reset, 0, self.cumulated_loss)
        return next_value.detach()

    def clear_loss(self):
        self.actor_loss = torch.tensor(0., device=self.device)
        self.rollout_gamma = torch.ones(self.n_envs, device=self.device)
        self.cumulated_loss = torch.zeros(self.n_envs, device=self.device)
        self.entropy_loss = torch.tensor(0., device=self.device)
    
    def train(self, target_values: Tensor) -> Dict[str, float]:
        actor_loss = self.actor_loss / (self.n_envs * self.l_rollout)
        entropy_loss = self.entropy_loss / (self.n_envs * self.l_rollout)
        T, N, D = self.buffer.states.shape
        states = self.buffer.states.reshape(T*N, D)
        values = self.agent.get_value(states)
        critic_loss = F.mse_loss(values, target_values)
        
        total_loss = actor_loss + self.entropy_weight * entropy_loss + self.value_weight * critic_loss
        self.optim.zero_grad()
        total_loss.backward()
        actor_grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.agent.actor_mean.parameters()]) ** 0.5
        if self.actor_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.agent.actor_mean.parameters(), max_norm=self.actor_grad_norm)
        critic_grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.agent.critic.parameters()]) ** 0.5
        if self.critic_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_norm=self.critic_grad_norm)
        self.optim.step()
        
        for p, p_t in zip(self.agent.critic.parameters(), self._critic_target.parameters()):
            p_t.data.lerp_(p.data, 5e-3)
        losses = {
            "actor_loss": actor_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "critic_loss": critic_loss.item()}
        grad_norms = {
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm}
        return losses, grad_norms

    @staticmethod
    def build(cfg, env, device):
        return SHAC(
            cfg=cfg.algo,
            state_dim=env.state_dim,
            hidden_dim=list(cfg.algo.hidden_dim),
            action_dim=env.action_dim,
            n_envs=env.n_envs,
            l_rollout=cfg.l_rollout,
            device=device)

    @staticmethod
    def learn(cfg, agent, env, logger, on_step_cb=None, on_update_cb=None):
        # type: (DictConfig, SHAC, BaseEnv, Logger, Optional[Callable], Optional[Callable]) -> None
        state = env.reset()
        pbar = tqdm(range(cfg.n_updates))
        for i in pbar:
            t1 = pbar._time()
            env.detach()
            agent.buffer.clear()
            agent.clear_loss()
            for t in range(cfg.l_rollout):
                action, policy_info = agent.act(state)
                next_state, loss, terminated, env_info = env.step(action)
                next_value = agent.record_loss(loss, policy_info, env_info, last_step=(t==cfg.l_rollout-1))
                agent.buffer.add(state, loss, policy_info["value"], env_info["reset"], terminated, next_value)
                state = next_state
                if on_step_cb is not None:
                    on_step_cb(
                        state=state,
                        action=action,
                        policy_info=policy_info,
                        env_info=env_info)

            target_values = agent.bootstrap_gae()
            losses, grad_norms = agent.train(target_values)
            l_episode = env_info["stats"]["l"].float().mean().item()
            success_rate = env_info['stats']['success_rate']
            pbar.set_postfix({
                "param_norm": f"{grad_norms['actor_grad_norm']:.3f}",
                "loss": f"{loss.mean():.3f}",
                "l_episode": f"{l_episode:.1f}",
                "success_rate": f"{success_rate:.2f}",
                "fps": f"{(cfg.l_rollout*cfg.env.n_envs)/(pbar._time()-t1):,.0f}"})
            log_info = {
                "value": policy_info["value"].mean().item(),
                "env_loss": env_info["loss_components"],
                "agent_loss": losses,
                "agent_grad_norm": grad_norms,
                "metrics": {"l_episode": l_episode, "success_rate": success_rate}
            }
            logger.log_scalars(log_info, i+1)

            if on_update_cb is not None:
                on_update_cb(log_info=log_info)
        