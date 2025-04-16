from typing import List
from dataclasses import dataclass
from omegaconf import DictConfig
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra

from .module import *
from .function import *

class RSSM(nn.Module):
    def __init__(self, token_dim:int, action_dim:int, stoch:int, classes:int, deter:int, hidden:int, act:str, norm:str):
        super().__init__()
        self.post_proj = nn.Sequential(MLP(token_dim + deter, hidden, [hidden], norm, act),
                                       nn.Linear(hidden, stoch*classes))
        self.prior_proj = nn.Sequential(MLP(deter, hidden, [hidden], norm, act),
                                        nn.Linear(hidden, stoch*classes))
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
    def __init__(self, cfg:DictConfig):
        super().__init__()
        enc_cfg = cfg.encoder
        rssm_cfg = cfg.rssm
        dec_cfg = cfg.decoder
        self.encoder = ImageEncoder(image_shape=enc_cfg.image_shape, channels=enc_cfg.channels, 
                                    stride=enc_cfg.stride, kernel_size=enc_cfg.kernel_size, 
                                    act=enc_cfg.act, norm=enc_cfg.norm)
        final_shape = self.encoder.final_shape
        self.rssm = RSSM(token_dim=math.prod(final_shape), action_dim=rssm_cfg.action_dim,
                         stoch=rssm_cfg.stoch, classes=rssm_cfg.classes, deter=rssm_cfg.deter,
                         hidden=rssm_cfg.hidden, act=rssm_cfg.act, norm=rssm_cfg.norm)
        self.decoder = ImageDecoder(final_image_shape=final_shape, feat_dim=rssm_cfg.stoch*rssm_cfg.classes+rssm_cfg.deter,
                                    channels=dec_cfg.channels, stride=dec_cfg.stride, kernel_size=dec_cfg.kernel_size,
                                    act=dec_cfg.act, norm=dec_cfg.norm)
        common_kwargs = {'input_dim':rssm_cfg.deter + rssm_cfg.stoch*rssm_cfg.classes, 'hidden_units':[rssm_cfg.hidden], 
                         'norm':rssm_cfg.norm,'act':rssm_cfg.act,'output_dim':rssm_cfg.hidden}
        self.rew = nn.Sequential(MLP(**common_kwargs), nn.Linear(rssm_cfg.hidden, 255),)
        self.ter = nn.Sequential(MLP(**common_kwargs), nn.Linear(rssm_cfg.hidden, 1), nn.Sigmoid())
        self.symlogtwohot = SymLogTwoHotLoss(255, -20, 20)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.kl_loss = CategoricalLossWithFreeBits()
    
        self.deter_dim = rssm_cfg.deter
    
    def encode(self, obs:torch.Tensor, deter:torch.Tensor):
        tokens = self.encoder(obs)
        post_sample, _ = self.rssm._post(tokens, deter)
        return post_sample
    
    def recurrent(self, stoch:torch.Tensor, deter:torch.Tensor, action:torch.Tensor, terminal:torch.Tensor):
        deter = self.rssm.recurrent(stoch, deter, action)
        deter = torch.where(terminal.unsqueeze(-1), torch.zeros_like(deter), deter)
        return deter
        
    def compute_loss(self, obs:torch.Tensor, actions:torch.Tensor, rewards:torch.Tensor, terminals:torch.Tensor):
        # obs: B L C H W, actions:B L D, rewards:B L, terminals:B L
        deter = torch.zeros(obs.size(0), self.deter_dim, device=obs.device)
        tokens = self.encoder(obs)
        
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
        rec_images = self.decoder(feats)
        
        dyn_loss = self.kl_loss.kl_loss(post_probs.detach(), prior_probs)
        rep_loss = self.kl_loss.kl_loss(post_probs, prior_probs.detach())
        rew_loss = self.symlogtwohot(reward_logits, rewards)
        ter_loss = self.bce_loss(ter_logits.squeeze(-1), terminals)
        rec_loss = mse(rec_images, obs.detach())
        
        total_loss = rec_loss + 0.5*dyn_loss + 0.1*rep_loss + rew_loss + ter_loss
        
        metrics = {'WorldModel/Reconstruction:', rec_loss.item(),
                   'WorldModel/Dynamics:', dyn_loss.item(),
                   'WorldModel/Representation:', rep_loss.item(),
                   'WorldModel/Reward:', rew_loss.item(),
                   'WorldModel/Termination:', ter_loss.item(),
                   'WorldModel/Total:', total_loss.item()}
        
        return total_loss, metrics