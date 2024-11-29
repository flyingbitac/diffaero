import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch import Tensor
from torch.distributions import OneHotCategorical
from typing import Optional
from einops import rearrange,reduce
from .blocks import SymLogTwoHotLoss,SymLogTwoHotLossMulti

@dataclass
class StateModelCfg:
    state_dim: int
    hidden_dim: int
    action_dim: int
    latent_dim: int
    categoricals: int
    obstacle_relpos_dim: int
    num_classes: int
    use_simnorm: bool=False
    use_multirew: bool=False

@dataclass
class Batch:
    obs:torch.ByteTensor
    act:torch.LongTensor
    rew:torch.FloatTensor
    end:torch.LongTensor
    mask_padding:torch.BoolTensor
    drone_state:torch.FloatTensor
    obstacle_relpos:torch.FloatTensor

class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        return kl_div, real_kl_div

class RewardDecoder(nn.Module):
    def __init__(self, num_classes, hidden_dim, latent_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim+hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, feat, hidden):
        feat = self.backbone(torch.cat([feat,hidden],dim=-1))
        reward = self.head(feat)
        return reward

class EndDecoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(hidden_dim+latent_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Linear(hidden_dim, 1)
    
    def forward(self, feat, hidden):
        feat = self.backbone(torch.cat([feat,hidden],dim=-1))
        end = self.head(feat)
        return end.squeeze(-1)


class StateModel(nn.Module):
    def __init__(self, cfg:StateModelCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_simnorm = cfg.use_simnorm
        self.seq_model = nn.GRUCell(cfg.hidden_dim,cfg.hidden_dim)
        self.categoricals = cfg.categoricals
        self.kl_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        if not cfg.use_multirew:
            self.symlogtwohotloss = SymLogTwoHotLoss(cfg.num_classes,-20,20)
        else:
            self.symlogtwohotloss = SymLogTwoHotLossMulti(cfg.num_classes,-20,20,5)
        self.endloss = nn.BCEWithLogitsLoss()

        self.inp_proj = nn.Sequential(
            nn.Linear(cfg.state_dim + cfg.hidden_dim,cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
            nn.SiLU(),
            nn.Linear(cfg.latent_dim,cfg.latent_dim)
        )
        self.act_state_proj = nn.Sequential(
            nn.Linear(cfg.latent_dim+cfg.action_dim,cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim,cfg.hidden_dim)
        )
        self.state_decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim+cfg.hidden_dim,cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
            nn.SiLU(),
            nn.Linear(cfg.latent_dim,cfg.state_dim)
        )
        self.prior_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim,cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
            nn.SiLU(),
            nn.Linear(cfg.latent_dim,cfg.latent_dim)
        )

        self.reward_predictor = RewardDecoder(cfg.num_classes,cfg.hidden_dim,cfg.latent_dim)
        self.end_predictor = EndDecoder(cfg.hidden_dim,cfg.latent_dim)
    
    def straight_with_gradient(self,logits:Tensor):
        probs = F.softmax(logits,dim=-1)
        dist = OneHotCategorical(probs=probs)
        sample = dist.sample()
        sample_with_gradient = sample + probs - probs.detach()
        return sample_with_gradient

    def decode(self,latent:Tensor,hidden:Optional[Tensor]=None):
        if hidden==None:
            hidden = torch.zeros(latent.shape[0],self.cfg.hidden_dim,device=latent.device)
        return self.state_decoder(torch.cat([latent,hidden],dim=-1))
    
    def sample_with_prior(self,latent:Tensor,act:Tensor,hidden:Optional[Tensor]=None):
        assert latent.ndim==act.ndim==2
        state_act = self.act_state_proj(torch.cat([latent,act],dim=-1))
        if hidden==None:
            hidden = torch.zeros(state_act.shape[0],self.cfg.hidden_dim).to(state_act.device)
        hidden = self.seq_model(state_act,hidden)
        prior_logits = self.prior_proj(hidden)
        prior_logits = prior_logits.view(*prior_logits.shape[:-1],self.categoricals,-1)
        if self.use_simnorm:
            prior_probs = prior_logits.softmax(dim=-1)
            return prior_probs,prior_logits,hidden
        else:
            prior_sample = self.straight_with_gradient(prior_logits)
            return prior_sample,prior_logits,hidden

    def flatten(self,categorical_sample:Tensor):
        return categorical_sample.view(*categorical_sample.shape[:-2],-1)

    def sample_with_post(self,state:Tensor,hidden:Optional[Tensor]=None):
        if hidden==None:
            hidden = torch.zeros(state.shape[0],self.cfg.hidden_dim,device=state.device)
        post_logits = self.inp_proj(torch.cat([state,hidden],dim=-1))
        post_logits = post_logits.view(*post_logits.shape[:-1],self.categoricals,-1) # b l d -> b l c k
        if self.use_simnorm:
            post_probs = post_logits.softmax(dim=-1)
            return post_probs,post_logits
        else:
            post_sample = self.straight_with_gradient(post_logits) #b l k c
            return post_sample,post_logits
    
    @torch.no_grad()
    def predict_next(self,latent:Tensor,act:Tensor,hidden:Optional[Tensor]=None):
        assert latent.ndim==act.ndim==2
        prior_sample,_,hidden = self.sample_with_prior(latent,act,hidden)
        flattend_prior_sample = self.flatten(prior_sample)
        next_state = self.decode(flattend_prior_sample,hidden)
        reward_logit = self.reward_predictor(flattend_prior_sample,hidden)
        end_logit = self.end_predictor(flattend_prior_sample,hidden)
        pred_reward = self.symlogtwohotloss.decode(reward_logit)
        pred_end = end_logit>0
        return next_state,prior_sample,pred_reward,pred_end,hidden

    def compute_loss(self,states:Tensor, actions:Tensor, rewards:Tensor, terminations:Tensor):
        b,l,d = states.shape

        hidden = torch.zeros(b,self.cfg.hidden_dim,device=states.device)
        # post_samples,post_logits = self.sample_with_post(states,hidden)
        # flattened_post_samples = self.flatten(post_samples)
        # rec_states = self.decode(flattened_post_samples)
        # rec_loss = torch.sum((rec_states-states)**2,dim=-1).mean()

        post_logits = []
        prior_logits = []
        reward_logits = []
        end_logits = []
        rec_states = []

        for i in range(l):
            post_sample,post_logit = self.sample_with_post(states[:,i],hidden)
            flattend_post_sample = self.flatten(post_sample)
            rec_state = self.decode(flattend_post_sample,hidden)
            action = actions[:,i]
            prior_sample,prior_logit,hidden = self.sample_with_prior(flattend_post_sample,action,hidden)
            flattened_prior_sample = self.flatten(prior_sample)
            reward_logit = self.reward_predictor(flattened_prior_sample,hidden)
            end_logit = self.end_predictor(flattened_prior_sample,hidden)

            rec_states.append(rec_state) 
            post_logits.append(post_logit)
            prior_logits.append(prior_logit)
            reward_logits.append(reward_logit)
            end_logits.append(end_logit)

        rec_states = torch.stack(rec_states,dim=1)
        post_logits = torch.stack(post_logits,dim=1)
        prior_logits = torch.stack(prior_logits,dim=1)
        reward_logits = torch.stack(reward_logits,dim=1)
        end_logits = torch.stack(end_logits,dim=1)

        rep_loss,_ = self.kl_loss(post_logits[:,1:],prior_logits[:,:-1].detach())
        dyn_loss,_ = self.kl_loss(post_logits[:,1:].detach(),prior_logits[:,:-1])
        rew_loss = self.symlogtwohotloss(reward_logits,rewards)
        end_loss = self.endloss(end_logits,terminations)
        rec_loss = torch.sum((rec_states-states)**2,dim=-1).mean()
        total_loss = rec_loss + 0.5*dyn_loss + 0.1*rep_loss + rew_loss + end_loss
        return total_loss,rep_loss,dyn_loss,rec_loss,rew_loss,end_loss


if __name__=='__main__':

    cfg = StateModelCfg
    cfg.action_dim = 4
    cfg.categoricals = 16
    cfg.hidden_dim = 256
    cfg.latent_dim = 256
    cfg.state_dim = 13
    state_predictor = StateModel(cfg)

    batch = Batch(None,torch.randn(5,10,4),None,None,None,torch.randn(5,10,13),None)
    state_loss = state_predictor.compute_loss(batch)
    print(state_loss)
