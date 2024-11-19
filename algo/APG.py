from typing import Callable, Sequence, Tuple, Dict, Union, Optional

from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from quaddif.env.base_env import BaseEnv
from quaddif.utils.nn import mlp
from quaddif.utils.logger import Logger


class APG:
    def __init__(
        self,
        cfg: DictConfig,
        state_dim: int,
        hidden_dim: Sequence[int],
        action_dim: int,
        l_rollout: int,
        device: torch.device
    ):
        self.actor = mlp(state_dim, hidden_dim, action_dim, hidden_act=nn.ELU()).to(device)
        self.optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr)
        self.discount: float = cfg.gamma
        self.max_grad_norm: float = cfg.max_grad_norm
        self.l_rollout: int = l_rollout
        self.actor_loss = torch.zeros(1, device=device)
        self.device = device
    
    def act(self, state):
        # type: (Tensor) -> Tuple[Tensor, Dict[str, Tensor]]
        return torch.tanh(self.actor(state)), {}
    
    def record_loss(self, loss, policy_info, env_info):
        # type: (Tensor, Dict[str, Tensor], Dict[str, Tensor]) -> None
        self.actor_loss += loss.mean()
    
    def update_actor(self):
        # type: () -> Tuple[Dict[str, float], Dict[str, float]]
        self.actor_loss = self.actor_loss / self.l_rollout
        self.optimizer.zero_grad()
        self.actor_loss.backward()
        grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.actor.parameters()]) ** 0.5
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        actor_loss = self.actor_loss.item()
        self.actor_loss = torch.zeros(1, device=self.device)
        return {"actor_loss": actor_loss}, {"actor_grad_norm": grad_norm}

    @staticmethod
    def build(cfg, env, device):
        return APG(
            cfg=cfg.algo,
            state_dim=env.state_dim,
            hidden_dim=list(cfg.algo.hidden_dim),
            action_dim=env.action_dim,
            l_rollout=cfg.l_rollout,
            device=device)

    @staticmethod
    def learn(cfg, agent, env, logger, on_step_cb=None, on_update_cb=None):
        # type: (DictConfig, Union[APG, APG_stochastic], BaseEnv, Logger, Optional[Callable], Optional[Callable]) -> None
        state = env.reset()
        pbar = tqdm(range(cfg.n_updates))
        for i in pbar:
            t1 = pbar._time()
            env.detach()
            for _ in range(cfg.l_rollout):
                action, policy_info = agent.act(state)
                state, loss, terminated, env_info = env.step(action)
                agent.record_loss(loss, policy_info, env_info)
                if on_step_cb is not None:
                    on_step_cb(
                        state=state,
                        action=action,
                        policy_info=policy_info,
                        env_info=env_info)
                
            losses, grad_norms = agent.update_actor()
            # log data
            l_episode = env_info["stats"]["l"].float().mean().item()
            success_rate = env_info['stats']['success_rate']
            pbar.set_postfix({
                "param_norm": f"{grad_norms['actor_grad_norm']:.3f}",
                "loss": f"{loss.mean().item():.3f}",
                "l_episode": f"{l_episode:.1f}",
                "success_rate": f"{success_rate:.2f}",
                "fps": f"{(cfg.l_rollout*cfg.env.n_envs)/(pbar._time()-t1):,.0f}"})
            log_info = {
                "env_loss": env_info["loss_components"],
                "agent_loss": losses,
                "agent_grad_norm": grad_norms,
                "metrics": {"l_episode": l_episode, "success_rate": success_rate}
            }
            logger.log_scalars(log_info, i)

            if on_update_cb is not None:
                on_update_cb(log_info=log_info)


class APG_stochastic(APG):
    def __init__(
        self,
        cfg: DictConfig,
        state_dim: int,
        hidden_dim: Sequence[int],
        action_dim: int,
        l_rollout: int,
        device: torch.device
    ):
        super().__init__(cfg, state_dim, hidden_dim, action_dim, l_rollout, device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim, device=device))
        del self.optimizer
        self.optimizer = torch.optim.Adam([
            {"params": self.actor.parameters()},
            {"params": self.actor_logstd}], lr=cfg.actor_lr)
        self.entropy_loss = torch.zeros(1, device=device)
        self.entropy_weight: float = cfg.entropy_weight

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
        action = torch.tanh(sample)
        logprob = probs.log_prob(sample) - torch.log(1. - torch.tanh(sample).pow(2) + 1e-8)
        entropy = probs.entropy().sum(-1)
        return action, {"sample": sample, "logprob": logprob.sum(-1), "entropy": entropy}
    
    def record_loss(self, loss, policy_info, env_info):
        # type: (Tensor, Dict[str, Tensor], Dict[str, Tensor]) -> None
        self.actor_loss += loss.mean()
        self.entropy_loss -= policy_info["entropy"].mean()
    
    def update_actor(self):
        # type: () -> Tuple[Dict[str, float], Dict[str, float]]
        actor_loss = self.actor_loss / self.l_rollout
        entropy_loss = self.entropy_loss / self.l_rollout
        total_loss = actor_loss + self.entropy_weight * entropy_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.actor.parameters()]) ** 0.5
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        self.actor_loss = torch.zeros(1, device=self.device)
        self.entropy_loss = torch.zeros(1, device=self.device)
        return {"actor_loss": actor_loss, "entropy_loss": entropy_loss}, {"actor_grad_norm": grad_norm}

    @staticmethod
    def build(cfg, env, device):
        return APG_stochastic(
            cfg=cfg.algo,
            state_dim=env.state_dim,
            hidden_dim=list(cfg.algo.hidden_dim),
            action_dim=env.action_dim,
            l_rollout=cfg.l_rollout,
            device=device)
