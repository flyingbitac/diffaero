from typing import List, Tuple, Dict, Union, Optional

from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from quaddif.env.base_env import BaseEnv
from quaddif.utils.nn import mlp
from quaddif.utils.logger import Logger, CallBack


class APG:
    def __init__(
        self,
        cfg: DictConfig,
        state_dim: int,
        hidden_dim: List[int],
        action_dim: int,
        l_rollout: int,
        device: torch.device
    ):
        self.actor = mlp(state_dim, hidden_dim, action_dim, hidden_act=nn.ELU()).to(device)
        self.optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr)
        self.discount = cfg.gamma
        self.max_grad_norm = cfg.max_grad_norm
        self.l_rollout = l_rollout
        self.actor_loss = torch.zeros(1, device=device)
        self.device = device
    
    def act(self, state):
        # type: (Tensor) -> Tuple[Tensor, Dict[str, Tensor]]
        return torch.tanh(self.actor(state)), {}
    
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
            l_rollout=cfg.l_rollout,
            device=device)


class APG_stocastic(APG):
    def __init__(
        self,
        cfg: DictConfig,
        state_dim: int,
        hidden_dim: List[int],
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
        self.entropy_weight = cfg.entropy_weight

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
    
    def record_loss(self, loss, info, extra):
        self.actor_loss += (loss - self.entropy_weight * info["entropy"]).mean()

    @staticmethod
    def build(cfg, env, device):
        return APG_stocastic(
            cfg=cfg.algo,
            state_dim=env.state_dim,
            hidden_dim=list(cfg.algo.hidden_dim),
            action_dim=env.action_dim,
            l_rollout=cfg.l_rollout,
            device=device)


def learn(cfg, agent, env, logger, callback=None):
    # type: (DictConfig, Union[APG, APG_stocastic], BaseEnv, Logger, Optional[CallBack]) -> None
    state = env.reset()
    pbar = tqdm(range(cfg.n_updates))
    for i in pbar:
        t1 = pbar._time()
        # 超级重要，为了后续轨迹的loss梯度不反向传播到此前的状态，要先把梯度截断
        env.detach()
        for _ in range(cfg.l_rollout):
            action, info = agent.act(state)
            state, loss, terminated, extra = env.step(action)
            agent.record_loss(loss, info, extra)
            if callback is not None:
                callback.on_step(
                    state=state,
                    action=action,
                    loss=loss,
                    extra=extra)
            
        actor_loss, grad_norm = agent.update_actor()
        
        # log data
        l_episode = extra["stats"]["l"].float().mean().item()
        success_rate = extra['stats']['success_rate']
        pbar.set_postfix({
            "param_norm": f"{grad_norm:.3f}",
            "loss": f"{loss.mean().item():.3f}",
            "l_episode": f"{l_episode:.3f}",
            "success_rate": f"{success_rate:.2f}",
            "fps": f"{(cfg.l_rollout*cfg.env.n_envs)/(pbar._time()-t1):.2f}"})
        log_info = {
            "env_loss": extra["loss_components"],
            "agent_loss": {"actor_loss": actor_loss, "actor_grad_norm": grad_norm},
            "metrics": {"l_episode": l_episode, "success_rate": success_rate}
        }
        logger.log_scalars(log_info, i)

        if callback is not None:
            callback.on_update(log_info=log_info)

