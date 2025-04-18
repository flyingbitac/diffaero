from typing import Tuple, Dict, Union, Optional, List
import os
import math

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import tensordict
from tensordict import TensorDict

from quaddif.env.obstacle_avoidance import ObstacleAvoidanceGrid
from quaddif.algo.buffer import RolloutBufferGRID, RNNStateBuffer
from quaddif.network.networks import CNNBackbone
from quaddif.network.agents import tensordict2tuple, StochasticActor
from quaddif.utils.runner import timeit
from quaddif.utils.nn import mlp
from quaddif.algo.percption.world.backbone import WorldModel


class RCNN(nn.Module):
    def __init__(
        self,
        input_dim: Tuple[int, Tuple[int, int]],
        hidden_dim: Union[int, List[int]],
        rnn_n_layers: int,
        rnn_hidden_dim: int,
        output_dim: int,
        output_act: Optional[nn.Module] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cnn = CNNBackbone(input_dim)
        
        self.rnn_n_layers = rnn_n_layers
        self.rnn_hidden_dim = rnn_hidden_dim
        self.gru = torch.nn.GRU(
            input_size=self.cnn.out_dim,
            hidden_size=self.rnn_hidden_dim,
            num_layers=self.rnn_n_layers,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
            dtype=torch.float
        )
        self.head = mlp(self.rnn_hidden_dim, hidden_dim, output_dim, output_act=output_act)
        self.hidden_state: Tensor = None
    
    def forward(
        self,
        obs: Tuple[Tensor, Tensor], # ([N, D_state], [N, H, W])
        action: Optional[Tensor] = None, # [N, D_action]
        hidden: Optional[Tensor] = None, # [n_layers, N, D_hidden]
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # self.gru.flatten_parameters()
        
        perception = obs[1]
        if perception.ndim == 3:
            perception = perception.unsqueeze(1)
        rnn_input = torch.cat([obs[0], self.cnn(perception)] + ([] if action is None else [action]), dim=-1)
        
        use_own_hidden = hidden is None
        if use_own_hidden:
            if self.hidden_state is None:
                hidden = torch.zeros(self.rnn_n_layers, rnn_input.size(0), self.rnn_hidden_dim, dtype=rnn_input.dtype, device=rnn_input.device)
            else:
                hidden = self.hidden_state
        
        rnn_out, hidden = self.gru(rnn_input.unsqueeze(1), hidden)
        if use_own_hidden:
            self.hidden_state = hidden
        return self.head(rnn_out.squeeze(1)), hidden
    
    def forward_export(
        self,
        state: Tensor, # [N, D_state]
        perception: Tensor, # [N, H, W]
        hidden: Tensor, # [n_layers, N, D_hidden]
        action: Optional[Tensor] = None, # [N, D_action]
    ) -> Tuple[Tensor, Tensor]:
        if perception.ndim == 3:
            perception = perception.unsqueeze(1)
        rnn_input = torch.cat([state, self.cnn(perception)] + ([] if action is None else [action]), dim=-1)
        rnn_out, hidden = self.gru(rnn_input.unsqueeze(1), hidden)
        return self.head(rnn_out.squeeze(1)), hidden

    def reset(self, indices: Tensor):
        self.hidden_state[:, indices, :] = 0
    
    def detach(self):
        self.hidden_state.detach_()


class GRID:
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: Tuple[int, Tuple[int, int]],
        action_dim: int,
        grid_cfg: DictConfig,
        device: torch.device
    ):
        self.l_rollout: int = cfg.l_rollout
        self.batch_size: int = cfg.batch_size
        self.latent_dim: int = cfg.latent_dim
        
        # encoder
        self.wm_encoder = WorldModel(cfg.wmperc)
        # actor
        self.actor = StochasticActor(cfg.network, self.wm_encoder.deter_dim + self.wm_encoder.latent_dim, action_dim).to(device)
        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        
        # replay buffer
        self.buffer = RolloutBufferGRID(
            self.l_rollout,
            int(cfg.buffer_size),
            obs_dim,
            self.latent_dim,
            action_dim,
            self.n_grid_points,
            device
        )
        self.entropy_loss = torch.zeros(1, device=device)
        self.entropy_weight: float = cfg.entropy_weight
        self.max_grad_norm: float = cfg.max_grad_norm
        self.actor_loss = torch.zeros(1, device=device)
        self.device = device
        self.deter: Tensor = None
    
    @timeit
    def act(self, obs, test=False):
        # type: (TensorDict, bool) -> Tuple[Tensor, Dict[str, Tensor]]
        with torch.no_grad():
            if self.deter is None:
                self.deter = torch.zeros(obs['state'].shape[0], self.wm_encoder.deter_dim, device=obs['state'].device)
            latent = self.wm_encoder.encode(obs['perception'], obs['state'], self.deter)
            actor_input = torch.cat([latent, self.deter], dim=-1)
        action, sample, logprob, entropy = self.actor(actor_input, test=test)
        return action, {"latent": latent, "sample": sample, "logprob": logprob, "entropy": entropy}, latent
    
    def record_loss(self, loss, policy_info, env_info):
        # type: (Tensor, Dict[str, Tensor], Dict[str, Tensor]) -> None
        self.actor_loss += loss.mean()
        self.entropy_loss -= policy_info["entropy"].mean()
    
    @timeit
    def update_encdec(self):
        if not self.buffer.size >= self.batch_size:
            return {}, {}
        
        observations, actions, dones, rewards = self.buffer.sample(self.batch_size)
        # find ground truth and visible grid
        ground_truth_grid = observations["grid"]
        visible_map = observations["visible_map"]
        visible_ground_truth_grid = ground_truth_grid[visible_map]
        total_loss, metrics = self.wm_encoder.update(observations['perception'], observations['state'], 
                                actions, rewards, dones, visible_ground_truth_grid, visible_map)
        return total_loss, metrics
            
    @timeit
    def update_actor(self):
        # type: () -> Tuple[Dict[str, float], Dict[str, float]]
        actor_loss = self.actor_loss / self.l_rollout
        entropy_loss = self.entropy_loss / self.l_rollout
        total_loss = actor_loss + self.entropy_weight * entropy_loss
        self.actor_optimizer.zero_grad()
        total_loss.backward()
        grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.actor.parameters()]) ** 0.5
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
        self.actor_optimizer.step()
        self.actor_loss = torch.zeros(1, device=self.device)
        self.entropy_loss = torch.zeros(1, device=self.device)
        return {"actor_loss": actor_loss.mean().item(), "entropy_loss": entropy_loss.mean().item()}, {"actor_grad_norm": grad_norm}
    
    @timeit
    def step(self, cfg, env, obs, on_step_cb=None):
        rollout_obs, rollout_dones, rollout_actions, rollout_rewards = [], [], [], []
        for _ in range(cfg.l_rollout):
            action, policy_info, latent = self.act(obs)
            next_obs, loss, terminated, env_info = env.step(env.rescale_action(action), need_obs_before_reset=False)
            self.deter = self.wm_encoder.recurrent(latent, self.deter, action, terminated)
            self.record_loss(loss, policy_info, env_info)
            rollout_obs.append(obs)
            rollout_actions.append(action)
            rollout_dones.append(terminated)
            rollout_rewards.append(10.*(1. - 0.1*loss).detach())
            self.reset(env_info["reset"])
            obs = next_obs
            if on_step_cb is not None:
                on_step_cb(
                    obs=obs,
                    action=action,
                    policy_info=policy_info,
                    env_info=env_info)
        self.buffer.add(
            obs=tensordict.stack(rollout_obs, dim=1),
            action=torch.stack(rollout_actions, dim=1),
            done=torch.stack(rollout_dones, dim=1),
            reward=torch.stack(rollout_rewards, dim=1)
        )
        _ , metrics = self.update_encdec()
        actor_losses, actor_grad_norms = self.update_actor()
        losses = {**metrics, **actor_losses}
        grad_norms = {**actor_grad_norms}
        self.detach()
        return obs, policy_info, env_info, losses, grad_norms
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.wm_encoder.state_dict(), os.path.join(path, "wm_encoder.pth"))
        self.actor.save(path)
    
    def load(self, path):
        self.wm_encoder.load_state_dict(torch.load(os.path.join(path, "wm_encoder.pth")))
        self.actor.load(path)
    
    def detach(self):
        self.wm_encoder.detach()
    
    @staticmethod
    def build(cfg: DictConfig, env: ObstacleAvoidanceGrid, device: torch.device):
        return GRID(
            cfg=cfg,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            grid_cfg=env.cfg.grid,
            device=device)