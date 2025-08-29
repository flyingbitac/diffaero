from typing import Optional, Tuple, Dict
import math
import os

from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .module import (
    MiniGru,
    ImageEncoder,
    ImageDecoder,
    ImageDecoderMLP,
    StateEncoder,
    StateDecoder,
    GridDecoder
)
from .function import (
    get_unimix_logits,
    one_hot_sample,
    SymLogTwoHotLoss,
    CategoricalLossWithFreeBits
)
from diffaero.utils.nn import mlp
from diffaero.utils.runner import timeit
from diffaero.utils.logger import Logger

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
        self.post_proj = mlp(token_dim + deter, [hidden, hidden], stoch*classes)
        self.prior_proj = mlp(deter, [hidden, hidden], stoch*classes)
        self.seq_model = MiniGru(deter, stoch*classes, action_dim, hidden, act, norm)
        self.stoch = stoch
        self.classes = classes
    
    def straight_through_gradient(self, x: Tensor, test: bool = False):
        x_chunked = torch.stack(x.chunk(self.stoch, dim=-1), dim=-2)
        logits = get_unimix_logits(x_chunked)
        probs = F.softmax(logits, dim=-1)
        onehot = one_hot_sample(probs, test=test)
        straight_sample = onehot - probs.detach() + probs
        straight_sample = straight_sample.reshape_as(x)
        return straight_sample, logits
    
    def _post(self, token: Tensor, deter: Tensor, test: bool = False):
        x = torch.cat([token, deter], dim=-1)
        post_logits = self.post_proj(x)
        post_sample, post_logits = self.straight_through_gradient(post_logits, test)
        return post_sample, post_logits
       
    def _prior(self, deter: Tensor, stoch: Tensor, action: Tensor):
        deter = self.recurrent(stoch, deter, action)
        prior_logits = self.prior_proj(deter)
        prior_sample, prior_logits = self.straight_through_gradient(prior_logits)
        return prior_sample, prior_logits, deter
    
    def recurrent(self, stoch: Tensor, deter:Tensor, action: Tensor):
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

class WorldModelTesttime(nn.Module):
    @staticmethod
    def _build_mlp(rssm_cfg: DictConfig, output_dim: int) -> nn.Sequential:
        return mlp(
            in_dim=rssm_cfg.deter + rssm_cfg.stoch * rssm_cfg.classes,
            mlp_dims=[rssm_cfg.hidden, rssm_cfg.hidden],
            out_dim=output_dim
        )
    def __init__(self, obs_dim: Tuple[int, Tuple[int, int]], cfg: DictConfig, grid_cfg: Optional[DictConfig] = None):
        super().__init__()
        state_enc_cfg = cfg.encoder.state
        self.recon_state: bool = cfg.decoder.state.enable
        self.recon_grid: bool = cfg.decoder.grid.enable
        self.rssm_feature_dim: int = cfg.rssm.deter + cfg.rssm.stoch * cfg.rssm.classes
        rssm_cfg = cfg.rssm
        self.deter_dim = rssm_cfg.deter
        self.latent_dim = rssm_cfg.stoch * rssm_cfg.classes
        
        # image encoder
        self.image_encoder = ImageEncoder(obs_dim[1])
        self.fmap_final_shape = self.image_encoder.final_shape
        # state encoder
        self.encode_state: bool = not cfg.odom_free
        if self.encode_state:
            self.state_embed_dim = state_enc_cfg.embedding_dim
            self.state_encoder = StateEncoder(obs_dim[0]-3, state_enc_cfg)
        else:
            self.state_encoder = nn.Identity()
            self.state_embed_dim = 0
        
        # state decoder
        if self.recon_state:
            self.state_decoder = StateDecoder(obs_dim[0], rssm_cfg)
        else:
            self.state_decoder = nn.Identity()
        # grid decoder
        if grid_cfg is not None:
            if self.recon_grid:
                if cfg.decoder.grid.use_mlp:
                    self.grid_decoder = self._build_mlp(rssm_cfg, output_dim=math.prod(grid_cfg.n_points))
                else:
                    self.grid_decoder = GridDecoder(rssm_cfg, grid_cfg)
        else:
            self.grid_decoder = nn.Identity()
        
        # sequence model
        self.rssm = RSSM.build(token_dim=self.fmap_final_shape + self.state_embed_dim, rssm_cfg=rssm_cfg)
    
    def encode(self, obs, state, deter, test=False):
        # type: (Tensor, Tensor, Tensor, bool) -> Tensor
        tokens = [self.image_encoder(obs)]
        if self.encode_state:
            tokens.append(self.state_encoder(state))
        tokens = torch.cat(tokens, dim=-1)
        post_sample, _ = self.rssm._post(tokens, deter, test)
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
            "image_encoder": self.image_encoder.state_dict()
        }
        if self.encode_state:
            state_dicts["state_encoder"] = self.state_encoder.state_dict()
        torch.save(state_dicts, os.path.join(path, "world_model.pth"))
        
    def load(self, path: str):
        state_dicts = torch.load(os.path.join(path, "world_model.pth"))
        self.rssm.load_state_dict(state_dicts["rssm"])
        self.image_encoder.load_state_dict(state_dicts["image_encoder"])
        if self.encode_state:
            self.state_encoder.load_state_dict(state_dicts["state_encoder"])


class WorldModel(WorldModelTesttime):
    def __init__(self, obs_dim: Tuple[int, Tuple[int, int]], cfg: DictConfig, grid_cfg: DictConfig):
        assert grid_cfg is not None, "Grid configuration must be provided for WorldModel."
        super().__init__(obs_dim, cfg, grid_cfg)
        self.l_rollout: int = cfg.l_rollout
        self.recon_image: bool = cfg.decoder.image.enable
        self.recon_reward: bool = cfg.decoder.reward.enable
        self.rssm_feature_dim: int = cfg.rssm.deter + cfg.rssm.stoch * cfg.rssm.classes
        rssm_cfg = cfg.rssm
        image_dec_cfg = cfg.decoder.image
        
        # image decoder
        if image_dec_cfg.use_mlp:
            self.image_decoder = ImageDecoderMLP(
                final_image_shape=[1, obs_dim[1][0], obs_dim[1][1]],
                feat_dim=self.rssm_feature_dim,
                hidden_units=image_dec_cfg.mlpdecoder.hidden_units,
                act=image_dec_cfg.mlpdecoder.act,
                norm=image_dec_cfg.mlpdecoder.norm)
        else:
            self.image_decoder = ImageDecoder(
                final_image_shape=self.fmap_final_shape,
                feat_dim=self.rssm_feature_dim,
                channels=image_dec_cfg.channels,
                stride=image_dec_cfg.stride,
                kernel_size=image_dec_cfg.kernel_size,
                act=image_dec_cfg.act,
                norm=image_dec_cfg.norm)
        # reward deocder
        if self.recon_reward:
            self.reward_decoder = self._build_mlp(rssm_cfg, output_dim=255)
        # termination deocder
        self.termination_decoder = self._build_mlp(rssm_cfg, output_dim=1)
        
        self.optim = torch.optim.Adam(self.parameters(), lr=cfg.train.lr)
        self.grad_norm = cfg.train.grad_norm
        self.symlogtwohot = SymLogTwoHotLoss(255, -20, 20)
        self.term_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.train.term_pos_weight]))
        self.grid_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.train.grid_pos_weight]))
        self.kl_loss = CategoricalLossWithFreeBits(free_bits=cfg.train.free_bits)
        
        self.rec_img_weight: float = cfg.train.rec_img_weight
        self.rec_state_weight: float = cfg.train.rec_state_weight
        self.dyn_weight: float = cfg.train.dyn_weight
        self.rep_weight: float = cfg.train.rep_weight
        self.rew_weight: float = cfg.train.rew_weight
        self.ter_weight: float = cfg.train.ter_weight
        self.grid_weight: float = cfg.train.grid_weight
        self.soft_label: float = cfg.train.soft_label
    
    def forward(self, tokens, deter, actions):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        post_samples, post_logits = self.rssm._post(tokens, deter)
        prior_samples, prior_logits, deter = self.rssm._prior(deter, post_samples, actions)
        return post_samples, post_logits, prior_samples, prior_logits, deter

    @timeit    
    def update(
        self,
        img: Tensor,        # [B T H W]
        state: Tensor,      # [B T S]
        actions: Tensor,    # [B T D]
        rewards: Tensor,    # [B T]
        terminated: Tensor, # [B T]
        gt_grids: Tensor,   # [B T N_grids]
        visible_map: Tensor # [B T N_grids]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Tensor]]:
        deter = torch.zeros(img.size(0), self.deter_dim, device=img.device)
        tokens = [self.image_encoder(img)]
        if self.encode_state:
            tokens.append(self.state_encoder(state))
        tokens = torch.cat(tokens, dim=-1)  
        
        post_logits_list, prior_logits_list, post_feat_list = [], [], []
        for i in range(self.l_rollout):
        #   z_i                       \hat{z_{i+1}}               h_{i+1}           x_i           h_i    a_i
            post_samples, post_logit, prior_samples, prior_logit, deter_next = self(tokens[:, i], deter, actions[:, i])
            post_logits_list.append(post_logit)
            prior_logits_list.append(prior_logit)
            post_feat_list.append(torch.cat([deter, post_samples], dim=-1)) # h_i, z_i
            deter = deter_next
        
        post_logits = torch.stack(post_logits_list, dim=1)[:, 1:]
        prior_logits = torch.stack(prior_logits_list, dim=1)[:, :-1]
        post_feats = torch.stack(post_feat_list, dim=1) # h_i, z_i
        
        rewards, terminated = rewards[:, :-1], terminated[:, :-1]

        if self.recon_reward:
            reward_logits = self.reward_decoder(post_feats[:, 1:]) # \hat{r_{i+1}} = reward_decoder(h_{i+1}, \hat{z_{i+1}})
            rew_loss = self.symlogtwohot(reward_logits, rewards)
        else:
            rew_loss = torch.tensor(0)
        
        if self.recon_image:
            rec_img, rec_img_loss = self.image_decoder.compute_loss(post_feats, img)
        else:
            rec_img, rec_img_loss = self.image_decoder.compute_loss(post_feats.detach(), img)
        
        if self.recon_state:
            rec_state_loss = self.state_decoder.compute_loss(post_feats, state) # type: ignore
        else:
            rec_state_loss = torch.tensor(0)
        
        if self.recon_grid:
            grid_logits = self.grid_decoder(post_feats)
            visible_grid_logits = grid_logits[visible_map]
            visible_gt_grid = gt_grids[visible_map]
            target = visible_gt_grid.float() * (1 - self.soft_label) + self.soft_label
            grid_loss = self.grid_loss(visible_grid_logits, target)
            
            visible_pred_grid = visible_grid_logits > 0
            grid_acc = (visible_pred_grid == visible_gt_grid).float().mean()
            grid_precision = visible_pred_grid[visible_gt_grid].float().mean()
        else:
            grid_loss, grid_acc, grid_precision = torch.tensor(0), torch.tensor(0), torch.tensor(0)
        
        dyn_loss = self.kl_loss(post_logits.detach(), prior_logits)
        rep_loss = self.kl_loss(post_logits, prior_logits.detach())
        
        ter_logits = self.termination_decoder(post_feats[:, 1:]).squeeze(-1)
        term_loss = self.term_loss(ter_logits, terminated.float())
        term_pred = ter_logits > 0
        term_acc = (term_pred == terminated).float().mean()
        term_precision = term_pred[terminated].float().mean() if torch.any(terminated) else torch.tensor(1.0)
        
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
        if Logger.logging.level == 10:
            for n, m in self.named_children():
                if len(list(m.parameters())) != 0:
                    grads = [p.grad for p in m.parameters() if p.grad is not None]
                    Logger.debug(n, "\t", nn.utils.get_total_norm(grads).item())
            for n, m in self.rssm.named_children():
                if len(list(m.parameters())) != 0:
                    grads = [p.grad for p in m.parameters() if p.grad is not None]
                    Logger.debug(n, "\t", nn.utils.get_total_norm(grads).item())
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm)
        self.optim.step()
        
        losses = {
            'wm/dyn_loss': dyn_loss.item(),
            'wm/rep_loss': rep_loss.item(),
            'wm/term_loss': term_loss.item(),
            'wm/term_acc': term_acc.item(),
            'wm/term_precision': term_precision.item(),
            'wm/total_loss': total_loss.item(),
        }
        
        losses['wm/image_recon'] = rec_img_loss.item()
        if self.recon_state:
            losses['wm/state_recon'] = rec_state_loss.item()
        if self.recon_grid:
            losses['wm/grid_recon'] = grid_loss.item()
            losses['wm/grid_acc'] = grid_acc.item()
            losses['wm/grid_precision'] = grid_precision.item()
        if self.recon_reward:
            losses['wm/rew_loss'] = rew_loss.item()
        
        grad_norms = {
            'wm/grad_norm': grad_norm.item()
        }
        
        predictions = {}
        if self.recon_grid:
            predictions["occupancy_pred"] = grid_logits > 0
            predictions["occupancy_gt"] = gt_grids
            predictions["visible_map"] = visible_map
        predictions["image_pred"] = rec_img
        predictions["image_gt"] = img.reshape_as(rec_img)
        
        return losses, grad_norms, predictions
    
    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        state_dicts = {
            "rssm": self.rssm.state_dict(),
            "image_encoder": self.image_encoder.state_dict(),
            "image_decoder": self.image_decoder.state_dict(),
            "termination_decoder": self.termination_decoder.state_dict()
        }
        if self.encode_state:
            state_dicts["state_encoder"] = self.state_encoder.state_dict()
        if self.recon_state:
            state_dicts["state_decoder"] = self.state_decoder.state_dict()
        if self.recon_grid:
            state_dicts["grid_decoder"] = self.grid_decoder.state_dict()
        if self.recon_reward:
            state_dicts["reward_decoder"] = self.reward_decoder.state_dict()

        torch.save(state_dicts, os.path.join(path, "world_model.pth"))
    
    def load(self, path: str):
        state_dicts = torch.load(os.path.join(path, "world_model.pth"))
        self.rssm.load_state_dict(state_dicts["rssm"])
        self.image_encoder.load_state_dict(state_dicts["image_encoder"])
        self.image_decoder.load_state_dict(state_dicts["image_deocder"])
        self.termination_decoder.load_state_dict(state_dicts["termination_decoder"])
        if self.encode_state:
            self.state_encoder.load_state_dict(state_dicts["state_encoder"])
        if self.recon_state:
            self.state_decoder.load_state_dict(state_dicts["state_decoder"])
        if self.recon_grid:
            self.grid_decoder.load_state_dict(state_dicts["grid_decoder"])
        if self.recon_reward:
            self.reward_decoder.load_state_dict(state_dicts["reward_decoder"])
