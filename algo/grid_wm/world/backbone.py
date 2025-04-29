from typing import Optional, Tuple
import math
import os

from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .module import *
from .function import *
from quaddif.utils.runner import timeit

class RSSM(nn.Module):
    def __init__(
        self,
        token_dim: int,
        action_dim: int,
        stoch: int,
        classes: int,
        deter: int,
        hidden: int,
        act: str,
        norm: str
    ):
        super().__init__()
        self.post_proj = nn.Sequential(
            MLP(token_dim + deter, hidden, [hidden], norm, act),
            nn.Linear(hidden, stoch*classes)
        )
        self.prior_proj = nn.Sequential(
            MLP(deter, hidden, [hidden], norm, act),
            nn.Linear(hidden, stoch*classes)
        )
        self.seq_model = MiniGru(deter, stoch*classes, action_dim, hidden, act, norm)
        self.stoch = stoch
    
    def straight_through_gradient(self, x:torch.Tensor):
        x = rearrange(x, '... (S C) -> ... S C', S = self.stoch)
        logits = get_unimix_logits(x)
        probs= F.softmax(logits, dim=-1)
        onehot = torch.distributions.OneHotCategorical(probs = probs).sample()
        straight_sample = onehot - probs.detach() + probs
        straight_sample = rearrange(straight_sample, '... S C -> ... (S C)')
        return straight_sample, probs
    
    def _post(self, token:torch.Tensor, deter:torch.Tensor):
        x = torch.cat([token, deter], dim=-1)
        post_logits = self.post_proj(x)
        post_sample, post_probs = self.straight_through_gradient(post_logits)
        return post_sample, post_probs
    
    def _prior(self, deter:torch.Tensor, stoch:torch.Tensor, action:torch.Tensor):
        deter = self.seq_model(deter, stoch, action)
        prior_logits = self.prior_proj(deter)
        prior_sample, prior_probs = self.straight_through_gradient(prior_logits)
        return prior_sample, prior_probs, deter
    
    def recurrent(self, stoch:torch.Tensor, deter:torch.Tensor, action:torch.Tensor):
        return self.seq_model(deter, stoch, action)
    
    @staticmethod
    def build(token_dim: int, rssm_cfg: DictConfig):
        return RSSM(
            token_dim=token_dim,
            action_dim=rssm_cfg.action_dim,
            stoch=rssm_cfg.stoch,
            classes=rssm_cfg.classes,
            deter=rssm_cfg.deter,
            hidden=rssm_cfg.hidden,
            act=rssm_cfg.act,
            norm=rssm_cfg.norm
        )

class GridDecoder(nn.Module):
    def __init__(self, rssm_cfg: DictConfig, grid_cfg: DictConfig):
        super().__init__()
        input_dim = rssm_cfg.deter + rssm_cfg.stoch * rssm_cfg.classes
        ds_rate = 4
        self.n_grids = math.prod(grid_cfg.n_points)
        self.fmap_shape = [i // ds_rate for i in grid_cfg.n_points]
        self.fmap_channels = 8
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, self.fmap_channels * math.prod(self.fmap_shape)),
            nn.SiLU())
        self.up_convs = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=self.fmap_channels,
                out_channels=self.fmap_channels//2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.SiLU(),
            nn.Conv3d(
                in_channels=self.fmap_channels//2,
                out_channels=self.fmap_channels//2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.SiLU(),
            nn.ConvTranspose3d(
                in_channels=self.fmap_channels//2,
                out_channels=self.fmap_channels//4,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.SiLU(),
            nn.Conv3d(
                in_channels=self.fmap_channels//4,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
    
    def forward(self, x):
        flattened_fmap = self.input_layer(x)
        reshaped_fmap = rearrange(
            flattened_fmap,
            '... (c l w h) -> ... c l w h',
            c=self.fmap_channels,
            l=self.fmap_shape[0],
            w=self.fmap_shape[1],
            h=self.fmap_shape[2])
        flag = reshaped_fmap.ndim == 6
        if flag:
            B, T, C, L, W, H = reshaped_fmap.shape
            reshaped_fmap = reshaped_fmap.reshape(B*T, C, L, W, H)
        else:
            B, C, L, W, H = reshaped_fmap.shape
        out = self.up_convs(reshaped_fmap)
        if flag:
            out = out.reshape(B, T, self.n_grids)
        else:
            out = out.reshape(B, self.n_grids)
        return out
        

class WorldModel(nn.Module):
    @staticmethod
    def _build_mlp(rssm_cfg: DictConfig, output_dim: int) -> MLP:
        return nn.Sequential(
            MLP(
                input_dim=rssm_cfg.deter + rssm_cfg.stoch * rssm_cfg.classes,
                hidden_units=[rssm_cfg.hidden],
                act=rssm_cfg.act,
                norm=rssm_cfg.norm,
                output_dim=rssm_cfg.hidden
            ),
            nn.Linear(rssm_cfg.hidden, output_dim)
        )
    @staticmethod
    def _build_image_encoder(image_shape: Tuple[int, int], img_enc_cfg: DictConfig) -> ImageEncoder:
        return ImageEncoder(
            image_shape=[1, image_shape[0], image_shape[1]],
            channels=img_enc_cfg.channels, 
            stride=img_enc_cfg.stride,
            kernel_size=img_enc_cfg.kernel_size,
            act=img_enc_cfg.act,
            norm=img_enc_cfg.norm
        )
    @staticmethod
    def _build_state_encoder(state_enc_cfg: DictConfig) -> nn.Sequential:
        return StateEncoder(state_enc_cfg)
    
    def __init__(self, obs_dim: Tuple[int, Tuple[int, int]], cfg: DictConfig, grid_cfg: DictConfig):
        super().__init__()
        self.l_rollout: int = cfg.l_rollout
        self.recon_image: bool = cfg.decoder.image.enable
        self.recon_state: bool = cfg.decoder.state.enable
        self.recon_grid: bool = cfg.decoder.grid.enable
        image_enc_cfg = cfg.encoder.image
        state_enc_cfg = cfg.encoder.state
        rssm_cfg = cfg.rssm
        image_dec_cfg = cfg.decoder.image
        grid_dec_cfg = cfg.decoder.grid
        
        # image encoder and decoder
        self.image_encoder = self._build_image_encoder(obs_dim[1], image_enc_cfg)
        fmap_final_shape = self.image_encoder.final_shape
        
        # state encoder
        state_embed_dim = state_enc_cfg.embedding_dim
        self.state_encoder = self._build_state_encoder(state_enc_cfg)
        
        # sequence model
        self.rssm = RSSM.build(token_dim=math.prod(fmap_final_shape) + state_embed_dim, rssm_cfg=rssm_cfg)
        
        # image decoder
        if self.recon_image:
            if image_dec_cfg.use_mlp:
                self.img_decoder = ImageDecoderMLP(
                    final_image_shape=[1, obs_dim[1][0], obs_dim[1][1]],
                    feat_dim=rssm_cfg.stoch*rssm_cfg.classes+rssm_cfg.deter,
                    hidden_units=image_dec_cfg.mlpdecoder.hidden_units,
                    act=image_dec_cfg.mlpdecoder.act,
                    norm=image_dec_cfg.mlpdecoder.norm)
            else:
                self.img_decoder = ImageDecoder(
                    final_image_shape=fmap_final_shape,
                    feat_dim=rssm_cfg.stoch*rssm_cfg.classes+rssm_cfg.deter,
                    channels=image_dec_cfg.channels,
                    stride=image_dec_cfg.stride,
                    kernel_size=image_dec_cfg.kernel_size,
                    act=image_dec_cfg.act,
                    norm=image_dec_cfg.norm)
        # state decoder
        if self.recon_state:
            self.state_decoder = self._build_mlp(rssm_cfg, output_dim=obs_dim[0])
        # grid deocder
        if self.recon_grid:
            if grid_dec_cfg.use_mlp:
                self.grid_decoder = self._build_mlp(rssm_cfg, output_dim=math.prod(grid_cfg.n_points))
            else:
                self.grid_decoder = GridDecoder(rssm_cfg, grid_cfg)
        
        # reward deocder
        self.reward_decoder = self._build_mlp(rssm_cfg, output_dim=255)
        # termination deocder
        self.termination_decoder = self._build_mlp(rssm_cfg, output_dim=1)
        
        self.optim = torch.optim.Adam(self.parameters(), lr=cfg.train.lr)
        self.grad_norm = cfg.train.grad_norm
        self.symlogtwohot = SymLogTwoHotLoss(255, -20, 20)
        self.term_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.train.term_pos_weight]))
        self.grid_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.train.grid_pos_weight]))
        self.kl_loss = CategoricalLossWithFreeBits(free_bits=cfg.train.free_bits)
    
        self.deter_dim = rssm_cfg.deter
        self.latent_dim = rssm_cfg.stoch*rssm_cfg.classes
        
        self.rec_img_weight: float = cfg.train.rec_img_weight
        self.rec_state_weight: float = cfg.train.rec_state_weight
        self.dyn_weight: float = cfg.train.dyn_weight
        self.rep_weight: float = cfg.train.rep_weight
        self.rew_weight: float = cfg.train.rew_weight
        self.ter_weight: float = cfg.train.ter_weight
        self.grid_weight: float = cfg.train.grid_weight
        self.soft_label: float = cfg.train.soft_label
    
    @torch.jit.export
    def encode(self, obs, state, deter):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        tokens = torch.cat([self.image_encoder(obs), self.state_encoder(state)], dim=-1)
        post_sample, _ = self.rssm._post(tokens, deter)
        return post_sample
    
    @torch.no_grad()
    @torch.jit.export
    def recurrent(self, stoch, deter, action):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        deter = self.rssm.recurrent(stoch, deter, action)
        return deter
    
    def forward(self, tokens, deter, actions):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        post_samples, post_prob = self.rssm._post(tokens, deter)
        prior_samples, prior_prob, deter = self.rssm._prior(deter, post_samples, actions)
        return post_samples, post_prob, prior_samples, prior_prob, deter

    @timeit    
    def update(
        self,
        obs: Tensor,        # [B L C H W]
        state: Tensor,      # [B L S]
        actions: Tensor,    # [B L D]
        rewards: Tensor,    # [B L]
        terminals: Tensor,  # [B L]
        gt_grids: Tensor,   # [B L N_grids]
        visible_map: Tensor # [B L N_grids]
    ):
        deter = torch.zeros(obs.size(0), self.deter_dim, device=obs.device)
        tokens = torch.cat([self.image_encoder(obs), self.state_encoder(state)], dim=-1)   
        
        post_probs_list, prior_probs_list, feat_list = [], [], []
        for i in range(self.l_rollout):
            post_samples, post_prob, prior_samples, prior_prob, deter = self(tokens[:, i], deter, actions[:, i])
            post_probs_list.append(post_prob)
            prior_probs_list.append(prior_prob)
            feat_list.append(torch.cat([deter, prior_samples], dim=-1))
        
        post_probs = torch.stack(post_probs_list, dim=1)
        prior_probs = torch.stack(prior_probs_list, dim=1)
        feats = torch.stack(feat_list, dim=1)
        reward_logits = self.reward_decoder(feats)
        ter_logits = self.termination_decoder(feats).squeeze(-1)
        
        rec_img_loss = torch.tensor(0)
        if self.recon_image:
            rec_images = self.img_decoder(feats)
            rec_img_loss = mse(rec_images, obs.unsqueeze(2))
        
        rec_state_loss = torch.tensor(0)
        if self.recon_state:
            rec_states = self.state_decoder(feats)
            rec_state_loss = mse(rec_states, state)
        
        grid_loss, grid_acc, grid_precision = torch.tensor(0), torch.tensor(0), torch.tensor(0)
        if self.recon_grid:
            grid_logits = self.grid_decoder(feats)
            visible_grid_logits = grid_logits[visible_map]
            visible_gt_grid = gt_grids[visible_map]
            target = visible_gt_grid.float() * (1 - self.soft_label) + self.soft_label
            grid_loss = self.grid_loss(visible_grid_logits, target)
            
            visible_pred_grid = visible_grid_logits > 0
            grid_acc = (visible_pred_grid == visible_gt_grid).float().mean()
            grid_precision = visible_pred_grid[visible_gt_grid].float().mean()
        grid_pred = grid_logits > 0 if self.recon_grid else None
        
        dyn_loss = self.kl_loss.kl_loss(post_probs.detach(), prior_probs)
        rep_loss = self.kl_loss.kl_loss(post_probs, prior_probs.detach())
        rew_loss = self.symlogtwohot(reward_logits, rewards)
        term_loss = self.term_loss(ter_logits, terminals.float())
        term_pred = ter_logits > 0
        term_acc = (term_pred == terminals).float().mean()
        term_precision = term_pred[terminals].float().mean() if torch.any(terminals) else torch.tensor(1.0)
        
        total_loss = (
            self.rec_img_weight * rec_img_loss +
            self.rec_state_weight * rec_state_loss +
            self.dyn_weight * dyn_loss +
            self.rep_weight * rep_loss +
            self.rew_weight * rew_loss +
            self.ter_weight * term_loss +
            self.grid_weight * grid_loss
        )
        
        self.optim.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm)
        self.optim.step()
        
        losses = {
            'wm/dyn_loss': dyn_loss.item(),
            'wm/rep_loss': rep_loss.item(),
            'wm/rew_loss': rew_loss.item(),
            'wm/term_loss': term_loss.item(),
            'wm/term_acc': term_acc.item(),
            'wm/term_precision': term_precision.item(),
            'wm/total_loss': total_loss.item(),
        }
        
        if self.recon_image:
            losses['wm/image_recon'] = rec_img_loss.item()
        if self.recon_state:
            losses['wm/state_recon'] = rec_state_loss.item()
        if self.recon_grid:
            losses['wm/grid_recon'] = grid_loss.item()
            losses['wm/grid_acc'] = grid_acc.item()
            losses['wm/grid_precision'] = grid_precision.item()
        
        grad_norms = {
            'wm/grad_norm': grad_norm.item()
        }
        
        return losses, grad_norms, grid_pred
    
    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        state_dicts = {
            "rssm": self.rssm.state_dict(),
            "image_encoder": self.image_encoder.state_dict(),
            "state_encoder": self.state_encoder.state_dict(),
            "reward_decoder": self.reward_decoder.state_dict(),
            "termination_decoder": self.termination_decoder.state_dict()
        }
        if self.recon_image:
            state_dicts["image_decoder"] = self.img_decoder.state_dict()
        if self.recon_state:
            state_dicts["state_decoder"] = self.state_decoder.state_dict()
        if self.recon_grid:
            state_dicts["grid_decoder"] = self.grid_decoder.state_dict()
        torch.save(state_dicts, os.path.join(path, "world_model.pth"))
    
    def load(self, path: str):
        state_dicts = torch.load(os.path.join(path, "world_model.pth"))
        self.rssm.load_state_dict(state_dicts["rssm"])
        self.image_encoder.load_state_dict(state_dicts["image_encoder"])
        self.state_encoder.load_state_dict(state_dicts["state_encoder"])
        self.grid_decoder.load_state_dict(state_dicts["grid_decoder"])
        self.reward_decoder.load_state_dict(state_dicts["reward_decoder"])
        self.termination_decoder.load_state_dict(state_dicts["termination_decoder"])
        if self.recon_image:
            self.img_decoder.load_state_dict(state_dicts["image_decoder"])
        if self.recon_state:
            self.state_decoder.load_state_dict(state_dicts["state_decoder"])
        if self.recon_grid:
            self.grid_decoder.load_state_dict(state_dicts["grid_decoder"])


class WorldModelTesttime(nn.Module):
    def __init__(self, obs_dim: Tuple[int, Tuple[int, int]], cfg: DictConfig):
        super().__init__()
        img_enc_cfg = cfg.encoder.image
        state_enc_cfg = cfg.encoder.state
        rssm_cfg = cfg.rssm

        self.deter_dim = rssm_cfg.deter
        self.latent_dim = rssm_cfg.stoch * rssm_cfg.classes
        
        # image encoder and decoder
        self.img_encoder = WorldModel._build_image_encoder(obs_dim[1], img_enc_cfg)
        fmap_final_shape = self.img_encoder.final_shape
        
        # state encoder
        state_embed_dim = state_enc_cfg.embedding_dim
        self.state_encoder = WorldModel._build_state_encoder(state_enc_cfg)
        
        # sequence model
        self.rssm = RSSM.build(token_dim=math.prod(fmap_final_shape) + state_embed_dim, rssm_cfg=rssm_cfg)
    
    def encode(self, obs, state, deter):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        tokens = torch.cat([self.img_encoder(obs), self.state_encoder(state)], dim=-1)
        post_sample, _ = self.rssm._post(tokens, deter)
        return post_sample
    
    @torch.no_grad()
    def recurrent(self, stoch, deter, action):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        deter = self.rssm.recurrent(stoch, deter, action)
        return deter

    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        state_dicts = {
            "rssm": self.rssm.state_dict(),
            "image_encoder": self.img_encoder.state_dict(),
            "state_encoder": self.state_encoder.state_dict()
        }
        torch.save(state_dicts, os.path.join(path, "world_model.pth"))
        
    def load(self, path: str):
        state_dicts = torch.load(os.path.join(path, "world_model.pth"))
        self.rssm.load_state_dict(state_dicts["rssm"])
        self.img_encoder.load_state_dict(state_dicts["image_encoder"])
        self.state_encoder.load_state_dict(state_dicts["state_encoder"])