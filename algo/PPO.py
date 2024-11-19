from typing import Callable, List, Tuple, Dict, Optional
from collections import defaultdict
import os

from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm

from quaddif.env.base_env import BaseEnv
from quaddif.utils.nn import StochasticActorCritic
from quaddif.utils.logger import Logger

class PPORPLAgent(StochasticActorCritic):
    def __init__(
        self,
        anchor_ckpt: str,
        state_dim: int,
        anchor_state_dim: int,
        hidden_dim: int,
        action_dim: int,
    ):
        super().__init__(
            state_dim=state_dim+action_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim)
        
        torch.nn.init.zeros_(self.actor_mean[-1].weight)
        torch.nn.init.zeros_(self.actor_mean[-1].bias)
        
        self.anchor_agent = StochasticActorCritic(anchor_state_dim, hidden_dim, action_dim)
        self.anchor_agent.load_state_dict(torch.load(os.path.join(anchor_ckpt, "agent.pt")))
        self.anchor_agent.eval()
        self.anchor_state_dim = anchor_state_dim

    def get_value(self, obs):
        with torch.no_grad():
            anchor_action, _, _, _, _ = self.anchor_agent.get_action_and_value(
                obs[..., :self.anchor_state_dim], test=True)
        return super().get_value(torch.cat([obs, anchor_action], dim=-1))

    def get_action_and_value(self, obs, sample=None, test=False):
        with torch.no_grad():
            anchor_action, _, _, _, _ = self.anchor_agent.get_action_and_value(
                obs[..., :self.anchor_state_dim], test=True)
        obs = torch.cat([obs, anchor_action], dim=-1)
        action, sample, logprob, entropy, value = super().get_action_and_value(obs, sample, test)
        action = (action + anchor_action).clamp(min=-1, max=1)
        return action, sample, logprob, entropy, value


class RolloutBuffer:
    def __init__(self, l_rollout, num_envs, state_dim, action_dim, device):
        factory_kwargs = {"dtype": torch.float32, "device": device}
        
        self.l_rollout = l_rollout
        self.states = torch.zeros((l_rollout, num_envs, state_dim), **factory_kwargs)
        self.samples = torch.zeros((l_rollout, num_envs, action_dim), **factory_kwargs)
        self.logprobs = torch.zeros((l_rollout, num_envs), **factory_kwargs)
        self.rewards = torch.zeros((l_rollout, num_envs), **factory_kwargs)
        self.next_dones = torch.zeros((l_rollout, num_envs), **factory_kwargs)
        self.values = torch.zeros((l_rollout, num_envs), **factory_kwargs)
        self.next_values = torch.zeros((l_rollout, num_envs), **factory_kwargs)
    
    def clear(self):
        self.step = 0
        
    def add(self, state, sample, logprob, reward, next_done, value, next_value):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> None
        for buf, input in zip(
            [self.states, self.samples, self.logprobs, self.rewards, self.next_dones, self.values, self.next_values],
            [state, sample, logprob, reward, next_done, value, next_value]):
            buf[self.step] = input.detach().to(buf.dtype)
        self.step += 1


class PPO:
    def __init__(
        self,
        cfg: DictConfig,
        state_dim: int,
        hidden_dim: List[int],
        action_dim: int,
        n_envs: int,
        l_rollout: int,
        device: torch.device
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent = StochasticActorCritic(state_dim, hidden_dim, action_dim).to(device)
        self.optim = torch.optim.Adam(self.agent.parameters(), lr=cfg.lr, eps=cfg.eps)
        self.buffer = RolloutBuffer(l_rollout, n_envs, state_dim, action_dim, device)
        
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
    
    def act(self, state):
        # type: (torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        action, sample, logprob, entropy, value = self.agent.get_action_and_value(state)
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
    
    def train(self, advantages, target_values):
        # type: (Tensor, Tensor) -> Tuple[Dict[str, float], Dict[str, float]]
        T, N, Ds, Da = self.l_rollout, self.n_envs, self.state_dim, self.action_dim
        states = self.buffer.states.view(T*N, Ds)
        samples = self.buffer.samples.view(T*N, Da)
        logprobs = self.buffer.logprobs.view(T*N)
        values = self.buffer.values.reshape(T*N)
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
            _, _, newlogprob, entropy, _ = self.agent.get_action_and_value(states, samples)
            logratio = newlogprob - logprobs
            ratio = logratio.exp()
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            # entropy loss
            entropy_loss = -entropy.mean()
            # value loss
            newvalue = self.agent.get_value(states[mb_indices])
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
            
            actor_grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.agent.actor_mean.parameters()]) ** 0.5
            if self.actor_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.agent.actor_mean.parameters(), max_norm=self.actor_grad_norm)
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
    
    def add(self, state, sample, logprob, reward, done, value, next_value):
        self.buffer.add(state, sample, logprob, reward, done, value, next_value)

    @staticmethod
    def build(cfg, env, device):
        return PPO(
            cfg=cfg.algo,
            state_dim=env.state_dim,
            hidden_dim=list(cfg.algo.hidden_dim),
            action_dim=env.action_dim,
            n_envs=env.n_envs,
            l_rollout=cfg.l_rollout,
            device=device)

    @staticmethod
    def learn(cfg, agent, env, logger, on_step_cb=None, on_update_cb=None):
        # type: (DictConfig, PPO, BaseEnv, Logger, Optional[Callable], Optional[Callable]) -> None
        state = env.reset()
        pbar = tqdm(range(cfg.n_updates))
        for i in pbar:
            t1 = pbar._time()
            # 超级重要，为了后续轨迹的loss梯度不反向传播到此前的状态，要先把梯度截断
            env.detach()
            agent.buffer.clear()
            with torch.no_grad():
                for t in range(cfg.l_rollout):
                    action, policy_info = agent.act(state)
                    next_state, loss, terminated, env_info = env.step(action)
                    agent.add(
                        state=state,
                        sample=policy_info["sample"],
                        logprob=policy_info["logprob"],
                        reward=1-loss*0.2,
                        done=terminated,
                        value=policy_info["value"],
                        next_value=agent.agent.get_value(env_info["next_state_before_reset"]))
                    # terminated = next_terminated
                    state = next_state
                    if on_step_cb is not None:
                        on_step_cb(
                            state=state,
                            action=action,
                            policy_info=policy_info,
                            env_info=env_info)
                
            advantages, target_values = agent.bootstrap()
            for _ in range(cfg.algo.n_epoch):
                losses, grad_norms = agent.train(advantages, target_values)
            
            # log data
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
            logger.log_scalars(log_info, i)

            if on_update_cb is not None:
                on_update_cb(log_info=log_info)


class PPO_RPL(PPO):
    def __init__(
        self,
        anchor_ckpt: str,
        state_dim: int,
        anchor_state_dim: int,
        hidden_dim: int,
        action_dim: int,
        discount: float,
        gae_lambda: float,
        lr: float,
        eps: float,
        max_grad_norm: float,
        l_rollout: int,
        clip_coef: float,
        clip_vloss: bool,
        entropy_coef: float,
        value_coef: float,
        norm_adv: bool,
        device: torch.device,
    ):
        super().__init__(
            state_dim, hidden_dim, action_dim,
            discount, gae_lambda, lr, eps, max_grad_norm, l_rollout,
            clip_coef, clip_vloss, entropy_coef, value_coef, norm_adv, device)
        self.agent = PPORPLAgent(
            anchor_ckpt, state_dim, anchor_state_dim, hidden_dim,
            action_dim).to(device)
        self.optim = torch.optim.Adam([
            {"params": self.agent.actor_mean.parameters()},
            {"params": self.agent.actor_logstd},
            {"params": self.agent.critic.parameters()},
        ], lr=lr, eps=eps)