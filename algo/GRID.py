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
        self.encoder = RCNN(
            input_dim=obs_dim,
            hidden_dim=cfg.encoder_hidden_dim,
            rnn_n_layers=cfg.encoder_rnn_n_layer,
            rnn_hidden_dim=cfg.encoder_rnn_hidden_dim,
            output_dim=self.latent_dim
        ).to(device)
        
        # decoders
        self.grid_points: List[int] = grid_cfg.n_points
        self.n_grid_points = math.prod(self.grid_points)
        self.grid_decoder = mlp(self.latent_dim, [self.latent_dim * 2], self.n_grid_points).to(device)
        self.state_decoder = mlp(self.latent_dim, [self.latent_dim * 2], obs_dim[0]).to(device)
        self.image_decoder = mlp(self.latent_dim, [self.latent_dim * 2], obs_dim[1][0] * obs_dim[1][1], output_act=nn.Sigmoid()).to(device)
        self.dynamic_predictor = mlp(self.latent_dim+action_dim, [self.latent_dim * 2], self.latent_dim).to(device)
        
        # actor
        self.actor = StochasticActor(cfg.network, self.latent_dim + obs_dim[0], action_dim).to(device)
        
        # optimizers
        self.encdec_optimizer = torch.optim.Adam([
            {"params": self.encoder.parameters()},
            {"params": self.grid_decoder.parameters()},
            {"params": self.state_decoder.parameters()},
            {"params": self.image_decoder.parameters()},
            {"params": self.dynamic_predictor.parameters()}
        ], lr=cfg.encdec_lr)
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
        self.grid_recon_weight: float = cfg.grid_recon_weight
        self.image_recon_weight: float = cfg.image_recon_weight
        self.dynamics_weight: float = cfg.dynamics_weight
        self.representation_weight: float = cfg.representation_weight
        self.grid_pos_weight: Tensor = torch.tensor(cfg.grid_pos_weight, device=device)
        self.actor_loss = torch.zeros(1, device=device)
        self.device = device
    
    @timeit
    def act(self, obs, test=False):
        # type: (TensorDict, bool) -> Tuple[Tensor, Dict[str, Tensor]]
        with torch.no_grad():
            latent, hidden = self.encoder((obs["state"], obs["perception"]))
            actor_input = torch.cat([latent, obs["state"]], dim=-1)
        action, sample, logprob, entropy = self.actor(actor_input, test=test)
        return action, {"latent": latent, "sample": sample, "logprob": logprob, "entropy": entropy}
    
    def record_loss(self, loss, policy_info, env_info):
        # type: (Tensor, Dict[str, Tensor], Dict[str, Tensor]) -> None
        self.actor_loss += loss.mean()
        self.entropy_loss -= policy_info["entropy"].mean()
    
    @timeit
    def update_encdec(self):
        if not self.buffer.size >= self.batch_size:
            return {}, {}
        
        hidden = torch.zeros(self.encoder.rnn_n_layers, self.batch_size, self.encoder.rnn_hidden_dim, device=self.device)
        zero_hidden = hidden.clone()
        observations, actions, dones = self.buffer.sample(self.batch_size)
        decoded_grid_logits, decoded_state, decoded_image, decoded_next_latent, latents = [], [], [], [], []
        for l in range(self.l_rollout):
            obs, action, done = observations[:, l], actions[:, l], dones[:, l]
            latent, hidden = self.encoder((obs["state"], obs["perception"]), hidden=hidden)
            # reset hidden state when episode termination occurs
            hidden = hidden.where(done.reshape(1, -1, 1).expand(
                self.encoder.rnn_n_layers, -1, self.encoder.rnn_hidden_dim), zero_hidden)
            decoded_grid_logits.append(self.grid_decoder(latent))
            decoded_state.append(self.state_decoder(latent))
            decoded_image.append(self.image_decoder(latent))
            decoded_next_latent.append(self.dynamic_predictor(torch.cat([latent, action], dim=-1)))
            latents.append(latent)
        
        decoded_grid_logits = torch.stack(decoded_grid_logits, dim=1) # [batch_size, l_rollout, n_grid_points]
        decoded_state = torch.stack(decoded_state, dim=1) # [batch_size, l_rollout, obs_dim[0]]
        decoded_image = torch.stack(decoded_image, dim=1) # [batch_size, l_rollout, H, W]
        decoded_next_latent = torch.stack(decoded_next_latent, dim=1) # [batch_size, l_rollout, latent_dim]
        latents = torch.stack(latents, dim=1) # [batch_size, l_rollout, latent_dim]
        
        grid_pred = decoded_grid_logits > 0
        ground_truth_grid = observations["grid"]
        
        visible_map = observations["visible_map"]
        visible_grid_logits = decoded_grid_logits[visible_map]
        visible_grid_pred = grid_pred[visible_map]
        visible_ground_truth_grid = ground_truth_grid[visible_map]
        
        grid_recon_loss = F.binary_cross_entropy_with_logits(
            visible_grid_logits,
            visible_ground_truth_grid.float(),
            pos_weight=self.grid_pos_weight
        )
        image_recon_loss = F.mse_loss(decoded_image, observations["perception"].flatten(-2))
        state_recon_loss = F.mse_loss(decoded_state, observations["state"])
        dynamics_loss = F.mse_loss(decoded_next_latent[:, :-1], latents[:, 1:].detach())
        representation_loss = F.mse_loss(latents[:, 1:], decoded_next_latent[:, :-1].detach())
        encdec_loss = (
            state_recon_loss + 
            self.grid_recon_weight * grid_recon_loss + 
            self.image_recon_weight * image_recon_loss + 
            self.dynamics_weight * dynamics_loss +
            self.representation_weight * representation_loss
        )
        
        self.encdec_optimizer.zero_grad()
        encdec_loss.backward()
        grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.encoder.parameters()]) ** 0.5
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=self.max_grad_norm)
        self.encdec_optimizer.step()
        
        losses = {
            "grid_recon_loss": grid_recon_loss.item(),
            "grid_acc": (visible_grid_pred == visible_ground_truth_grid).float().mean().item(),
            "grid_precision": visible_grid_pred[visible_ground_truth_grid].float().mean().item(),
            "image_recon_loss": image_recon_loss.item(),
            "state_recon_loss": state_recon_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
            "representation_loss": representation_loss.item(),
            "encdec_loss": encdec_loss.item()
        }
        grad_norms = {"encdec_grid_norm": grad_norm}
        
        visible_grid_gt_for_plot = ground_truth_grid & visible_map
        visible_grid_pred_for_plot = grid_pred & visible_map
        
        n_missd_predictions = torch.sum(visible_grid_gt_for_plot != visible_grid_pred_for_plot, dim=-1) # [batch_size, l_rollout]
        env_idx, time_idx = torch.where(n_missd_predictions == n_missd_predictions.max())
        env_idx, time_idx = env_idx[0], time_idx[0]
        selected_grid_gt = visible_grid_gt_for_plot[env_idx, time_idx].reshape(*self.grid_points)
        selected_grid_pred = visible_grid_pred_for_plot[env_idx, time_idx].reshape(*self.grid_points)
        
        return losses, grad_norms, selected_grid_gt, selected_grid_pred
            
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
        rollout_obs, rollout_dones, rollout_actions = [], [], []
        for _ in range(cfg.l_rollout):
            action, policy_info = self.act(obs)
            next_obs, loss, terminated, env_info = env.step(env.rescale_action(action), need_obs_before_reset=False)
            self.record_loss(loss, policy_info, env_info)
            rollout_obs.append(obs)
            rollout_actions.append(action)
            rollout_dones.append(terminated)
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
        )
        encdec_losses, encdec_grad_norms, selected_grid_gt, selected_grid_pred = self.update_encdec()
        actor_losses, actor_grad_norms = self.update_actor()
        losses = {**encdec_losses, **actor_losses}
        grad_norms = {**encdec_grad_norms, **actor_grad_norms}
        self.detach()
        policy_info.update({
            "grid_gt": selected_grid_gt,
            "grid_pred": selected_grid_pred
        })
        return obs, policy_info, env_info, losses, grad_norms
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.encoder.state_dict(), os.path.join(path, "encoder.pth"))
        torch.save({
            "grid_decoder": self.grid_decoder.state_dict(),
            "state_decoder": self.state_decoder.state_dict(),
            "image_decoder": self.image_decoder.state_dict()
        }, os.path.join(path, "decoders.pth"))
        self.actor.save(path)
    
    def load(self, path):
        self.encoder.load_state_dict(torch.load(os.path.join(path, "encoder.pth")))
        decoders = torch.load(os.path.join(path, "decoders.pth"))
        self.grid_decoder.load_state_dict(decoders["grid_decoder"])
        self.state_decoder.load_state_dict(decoders["state_decoder"])
        self.image_decoder.load_state_dict(decoders["image_decoder"])
        self.actor.load(path)
    
    def reset(self, env_idx: Tensor):
        self.encoder.reset(env_idx)
    
    def detach(self):
        self.encoder.detach()
    
    @staticmethod
    def build(cfg: DictConfig, env: ObstacleAvoidanceGrid, device: torch.device):
        return GRID(
            cfg=cfg,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            grid_cfg=env.cfg.grid,
            device=device)