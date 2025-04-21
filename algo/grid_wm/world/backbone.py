import math

from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .module import *
from .function import *

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

class WorldModel(nn.Module):
    def __init__(self, cfg: DictConfig, grid_cfg: DictConfig):
        super().__init__()
        img_enc_cfg = cfg.encoder.img_encoder
        state_enc_cfg = cfg.encoder.state_encoder
        rssm_cfg = cfg.rssm
        dec_cfg = cfg.decoder
        embed_dim, final_shape = 0, [0]
        common_kwargs = {'input_dim':rssm_cfg.deter + rssm_cfg.stoch*rssm_cfg.classes, 'hidden_units':[rssm_cfg.hidden], 
                         'norm':rssm_cfg.norm,'act':rssm_cfg.act,'output_dim':rssm_cfg.hidden}
        if cfg.encoder.use_image:
            self.img_encoder = ImageEncoder(
                image_shape=img_enc_cfg.image_shape,
                channels=img_enc_cfg.channels, 
                stride=img_enc_cfg.stride,
                kernel_size=img_enc_cfg.kernel_size,
                act=img_enc_cfg.act,
                norm=img_enc_cfg.norm
            )
            final_shape = self.img_encoder.final_shape
            if not cfg.decoder.use_mlp:
                self.img_decoder = ImageDecoder(
                    final_image_shape=final_shape,
                    feat_dim=rssm_cfg.stoch*rssm_cfg.classes+rssm_cfg.deter,
                    channels=dec_cfg.channels,
                    stride=dec_cfg.stride,
                    kernel_size=dec_cfg.kernel_size,
                    act=dec_cfg.act,
                    norm=dec_cfg.norm
                )
            else:
                self.img_decoder = ImageDecoderMLP(
                    img_enc_cfg.image_shape,
                    rssm_cfg.stoch*rssm_cfg.classes+rssm_cfg.deter,
                    cfg.decoder.mlpdecoder.hidden_units,
                    cfg.decoder.mlpdecoder.act,
                    cfg.decoder.mlpdecoder.norm
                )
        if cfg.encoder.use_state:
            self.state_encoder = MLP(
                input_dim=state_enc_cfg.state_dim,
                output_dim=state_enc_cfg.embedding_dim,
                hidden_units=state_enc_cfg.hidden_units,
                act=state_enc_cfg.act,
                norm=state_enc_cfg.norm
            )
            self.state_decoder = nn.Sequential(
                MLP(**common_kwargs),
                nn.Linear(rssm_cfg.hidden, state_enc_cfg.state_dim)
            )
            embed_dim = state_enc_cfg.embedding_dim
        self.rssm = RSSM(
            token_dim=math.prod(final_shape) + embed_dim,
            action_dim=rssm_cfg.action_dim,
            stoch=rssm_cfg.stoch,
            classes=rssm_cfg.classes,
            deter=rssm_cfg.deter,
            hidden=rssm_cfg.hidden,
            act=rssm_cfg.act,
            norm=rssm_cfg.norm
        )
        self.grid_decoder = nn.Sequential(
            MLP(**common_kwargs),
            nn.Linear(rssm_cfg.hidden, math.prod(grid_cfg.n_points))
        )
        self.rew = nn.Sequential(
            MLP(**common_kwargs),
            nn.Linear(rssm_cfg.hidden, 255)
        )
        self.ter = nn.Sequential(
            MLP(**common_kwargs),
            nn.Linear(rssm_cfg.hidden, 1)
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=cfg.train.lr)
        self.grad_norm = cfg.train.grad_norm
        self.symlogtwohot = SymLogTwoHotLoss(255, -20, 20)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.train.grid_pos_weight]))
        self.kl_loss = CategoricalLossWithFreeBits()
    
        self.deter_dim = rssm_cfg.deter
        self.latent_dim = rssm_cfg.stoch*rssm_cfg.classes
        
        self.rec_img_weight: float = cfg.train.rec_img_weight
        self.rec_state_weight: float = cfg.train.rec_state_weight
        self.dyn_weight: float = cfg.train.dyn_weight
        self.rep_weight: float = cfg.train.rep_weight
        self.rew_weight: float = cfg.train.rew_weight
        self.ter_weight: float = cfg.train.ter_weight
        self.grid_loss: float = cfg.train.grid_loss
    
    def encode(self, obs, state, deter):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        tokens = []
        if hasattr(self, 'img_encoder'):
            tokens.append(self.img_encoder(obs))
        if hasattr(self, 'state_encoder'):
            tokens.append(self.state_encoder(state))
        tokens = torch.cat(tokens, dim=-1)
        post_sample, _ = self.rssm._post(tokens, deter)
        return post_sample
    
    def recurrent(self, stoch, deter, action, terminal):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
        deter = self.rssm.recurrent(stoch, deter, action)
        deter = torch.where(terminal.unsqueeze(-1), torch.zeros_like(deter), deter)
        return deter
        
    def update(
        self,
        obs: Tensor,       # [B L C H W]
        state: Tensor,     # [B L S]
        actions: Tensor,   # [B L D]
        rewards: Tensor,   # [B L]
        terminals: Tensor, # [B L]
        gt_grids: Tensor,
        visible_map: Tensor
    ):
        deter = torch.zeros(obs.size(0), self.deter_dim, device=obs.device)
        tokens = []
        if hasattr(self, 'img_encoder'):
            tokens.append(self.img_encoder(obs))
        if hasattr(self, 'state_encoder'):
            tokens.append(self.state_encoder(state))
        tokens = torch.cat(tokens, dim=-1)   
        post_probs_list, prior_probs_list, feat_list = [], [], []
        for i in range(obs.size(1)):
            post_samples, post_prob = self.rssm._post(tokens[:, i], deter)
            prior_samples, prior_prob, deter = self.rssm._prior(deter, post_samples, actions[:, i])
            post_probs_list.append(post_prob)
            prior_probs_list.append(prior_prob)
            feat_list.append(torch.cat([deter, prior_samples], dim=-1))
        
        post_probs = torch.stack(post_probs_list, dim=1)
        prior_probs = torch.stack(prior_probs_list, dim=1)
        feats = torch.stack(feat_list, dim=1)
        reward_logits = self.rew(feats)
        ter_logits = self.ter(feats)
        
        rec_img_loss = 0
        if hasattr(self, 'img_decoder'):
            rec_images = self.img_decoder(feats)
            rec_img_loss = mse(rec_images, obs.unsqueeze(2).detach())
        
        rec_state_loss = 0
        if hasattr(self, 'state_decoder'):
            rec_states = self.state_decoder(feats)
            rec_state_loss = mse(rec_states, state.detach())
        
        grid_loss, grid_acc, grid_precision = torch.zeros((), device=obs.device), 0, 0
        if hasattr(self, 'grid_decoder'):
            grid_logits = self.grid_decoder(feats)
            visible_grid_logits = grid_logits[visible_map]
            visible_gt_grid = gt_grids[visible_map]
            grid_loss = self.bce_loss(visible_grid_logits, visible_gt_grid.float())
            
            visible_pred_grid = visible_grid_logits > 0
            grid_acc = (visible_pred_grid == visible_gt_grid).float().mean()
            grid_precision = visible_pred_grid[visible_gt_grid].float().mean()
        
        dyn_loss = self.kl_loss.kl_loss(post_probs.detach(), prior_probs)
        rep_loss = self.kl_loss.kl_loss(post_probs, prior_probs.detach())
        rew_loss = self.symlogtwohot(reward_logits, rewards)
        term_loss = self.bce_loss(ter_logits.squeeze(-1), terminals.float())
        
        total_loss = (
            self.rec_img_weight * rec_img_loss +
            self.rec_state_weight * rec_state_loss +
            self.dyn_weight * dyn_loss +
            self.rep_weight * rep_loss +
            self.rew_weight * rew_loss +
            self.ter_weight * term_loss +
            self.grid_loss * grid_loss
        )
        
        self.optim.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm)
        self.optim.step()
        
        losses = {
            'wm/image_recon': rec_img_loss.item(),
            'wm/state_recon': rec_state_loss.item(),
            'wm/grid_recon': grid_loss.item(),
            'wm/grid_acc': grid_acc.item(),
            'wm/grid_precision': grid_precision.item(),
            'wm/dyn_loss': dyn_loss.item(),
            'wm/rep_loss': rep_loss.item(),
            'wm/rew_loss': rew_loss.item(),
            'wm/term_loss': term_loss.item(),
            'wm/total_loss': total_loss.item(),
        }
        grad_norms = {
            'wm/grad_norm': grad_norm.item()
        }
        
        return losses, grad_norms, grid_logits > 0