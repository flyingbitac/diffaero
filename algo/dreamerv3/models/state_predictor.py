import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch import Tensor
from torch.distributions import OneHotCategorical
from typing import Optional
from einops import rearrange,reduce
from einops.layers.torch import Rearrange

from .blocks import SymLogTwoHotLoss,SymLogTwoHotLossMulti,proj

@dataclass
class DepthStateModelCfg:
    state_dim: int
    image_width: int
    hidden_dim: int
    action_dim: int
    latent_dim: int
    categoricals: int
    num_classes: int
    use_simnorm: bool=False

@dataclass
class PercModelCfg:
    state_dim: int
    hidden_dim: int
    action_dim: int
    latent_dim: int
    categoricals: int
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

class ImageEncoder(nn.Module):
    def __init__(self, in_channels:int, stem_channels:int, image_width:int):
        super().__init__()
        backbone = []
        
        backbone.append(nn.Conv2d(in_channels, stem_channels, kernel_size=4,stride=2,padding=1,bias=False))
        backbone.append(nn.BatchNorm2d(stem_channels))
        backbone.append(nn.ReLU(inplace=True))
        image_width //=2
        
        while image_width>4:
            backbone.append(nn.Conv2d(stem_channels, 2*stem_channels, kernel_size=4,stride=2,padding=1,bias=False))
            backbone.append(nn.BatchNorm2d(2*stem_channels))
            backbone.append(nn.ReLU(inplace=True))
            stem_channels*=2
            image_width //=2
        
        self.backbone = nn.Sequential(*backbone)
        self.last_channels = stem_channels
        
    def forward(self, depth_image:Tensor):
        depth_image = self.backbone(depth_image)
        depth_image = rearrange(depth_image, "B C H W -> B (C H W)")
        return depth_image

class ImageDecoder(nn.Module):
    def __init__(self, feat_dim:int, stem_channels:int, last_channels:int, final_image_width:int):
        super().__init__()
        backbone = []
        backbone.append(nn.Linear(feat_dim, last_channels*final_image_width*final_image_width, bias=False))
        backbone.append(Rearrange("B (C H W) -> B C H W", C=last_channels, H=final_image_width, W=final_image_width))
        backbone.append(nn.BatchNorm2d(last_channels))
        backbone.append(nn.ReLU(inplace=True))
        
        channels = last_channels
        while channels>stem_channels:
            backbone.append(nn.ConvTranspose2d(channels, channels//2, kernel_size=4,stride=2,padding=1,bias=False))
            backbone.append(nn.BatchNorm2d(channels//2))
            backbone.append(nn.ReLU(inplace=True))
            channels //= 2
        
        backbone.append(nn.ConvTranspose2d(channels, 1, kernel_size=4,stride=2,padding=1,bias=False))
        self.backbone = nn.Sequential(*backbone)
    
    def forward(self, feat:Tensor):
        feat = self.backbone(feat)
        return feat

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

class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        loss = reduce(loss, "B L C H W -> B L", "sum")
        return loss.mean()

class DepthStateModel(nn.Module):
    def __init__(self, cfg:DepthStateModelCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_simnorm = cfg.use_simnorm
        self.categoricals = cfg.categoricals
        self.kl_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        self.mse_loss = MSELoss()
        self.symlogtwohotloss = SymLogTwoHotLoss(cfg.num_classes,-20,20)
        self.endloss = nn.BCEWithLogitsLoss()

        self.seq_model = nn.GRUCell(cfg.hidden_dim,cfg.hidden_dim)
        self.image_encoder = ImageEncoder(in_channels=1, stem_channels=16, image_width=cfg.image_width)
        self.state_encoder = nn.Sequential(nn.Linear(cfg.state_dim,64,bias=False),
                                           nn.LayerNorm(64),nn.SiLU())
        self.image_decoder = ImageDecoder(feat_dim=cfg.latent_dim+cfg.hidden_dim,stem_channels=16,
                                          last_channels=self.image_encoder.last_channels,final_image_width=4)

        self.inp_proj = nn.Sequential(
            nn.Linear(64 + self.image_encoder.last_channels*4*4 + cfg.hidden_dim,cfg.latent_dim),
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
        feat = torch.cat([latent,hidden],dim=-1)
        return self.state_decoder(feat),self.image_decoder(feat)
    
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

    def sample_with_post(self,state:Tensor,depth_image:Tensor,hidden:Optional[Tensor]=None):
        if hidden==None:
            hidden = torch.zeros(state.shape[0],self.cfg.hidden_dim,device=state.device)
        state_feat = self.state_encoder(state)
        depth_feat = self.image_encoder(depth_image)
        feat = torch.cat([state_feat,depth_feat],dim=-1)
        post_logits = self.inp_proj(torch.cat([feat,hidden],dim=-1))
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
        next_state,next_perception = self.decode(flattend_prior_sample,hidden)
        reward_logit = self.reward_predictor(flattend_prior_sample,hidden)
        end_logit = self.end_predictor(flattend_prior_sample,hidden)
        pred_reward = self.symlogtwohotloss.decode(reward_logit)
        pred_end = end_logit>0
        return next_state,prior_sample,pred_reward,pred_end,hidden

    def compute_loss(self,states:Tensor, depth_images:Tensor, actions:Tensor, rewards:Tensor, terminations:Tensor):
        b,l,d = states.shape

        hidden = torch.zeros(b,self.cfg.hidden_dim,device=states.device)

        post_logits = []
        prior_logits = []
        reward_logits = []
        end_logits = []
        rec_states = []
        rec_images = []

        for i in range(l):
            post_sample,post_logit = self.sample_with_post(states[:,i],depth_images[:,i],hidden)
            flattend_post_sample = self.flatten(post_sample)
            rec_state,rec_image = self.decode(flattend_post_sample,hidden)
            action = actions[:,i]
            prior_sample,prior_logit,hidden = self.sample_with_prior(flattend_post_sample,action,hidden)
            flattened_prior_sample = self.flatten(prior_sample)
            reward_logit = self.reward_predictor(flattened_prior_sample,hidden)
            end_logit = self.end_predictor(flattened_prior_sample,hidden)

            rec_states.append(rec_state) 
            rec_images.append(rec_image)
            post_logits.append(post_logit)
            prior_logits.append(prior_logit)
            reward_logits.append(reward_logit)
            end_logits.append(end_logit)

        rec_states = torch.stack(rec_states,dim=1)
        rec_images = torch.stack(rec_images,dim=1)
        post_logits = torch.stack(post_logits,dim=1)
        prior_logits = torch.stack(prior_logits,dim=1)
        reward_logits = torch.stack(reward_logits,dim=1)
        end_logits = torch.stack(end_logits,dim=1)

        rep_loss,_ = self.kl_loss(post_logits[:,1:],prior_logits[:,:-1].detach())
        dyn_loss,_ = self.kl_loss(post_logits[:,1:].detach(),prior_logits[:,:-1])
        rew_loss = self.symlogtwohotloss(reward_logits,rewards)
        end_loss = self.endloss(end_logits,terminations)
        rec_loss = torch.sum((rec_states-states)**2,dim=-1).mean()+self.mse_loss(rec_images,depth_images)
        total_loss = rec_loss + 0.5*dyn_loss + 0.1*rep_loss + rew_loss + end_loss
        return total_loss,rep_loss,dyn_loss,rec_loss,rew_loss,end_loss

class PercStateModel(nn.Module):
    def __init__(self, statecfg: DepthStateModelCfg, perccfg: PercModelCfg) -> None:
        super().__init__()
        self.statecfg = statecfg
        self.perccfg = perccfg
        self.use_simnorm = statecfg.use_simnorm
        self.state_categoricals = statecfg.categoricals
        self.perc_categoricals = perccfg.categoricals

        # Loss functions
        self.kl_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        loss_cls = SymLogTwoHotLossMulti if statecfg.use_multirew else SymLogTwoHotLoss
        self.symlogtwohotloss = loss_cls(statecfg.num_classes,-20,20)
        self.endloss = nn.BCEWithLogitsLoss()

        # Models
        self.state_model = nn.ModuleDict(self._create_components(statecfg))
        self.perc_model = nn.ModuleDict(self._create_components(perccfg))
        self.reward_predictor = RewardDecoder(statecfg.num_classes, perccfg.hidden_dim, statecfg.hidden_dim)
        self.end_predictor = EndDecoder(perccfg.hidden_dim, statecfg.hidden_dim)

    @staticmethod
    def _create_components(cfg):
        return {
            "seq_model": nn.GRUCell(cfg.hidden_dim, cfg.hidden_dim),
            "inp_proj": proj(cfg.state_dim + cfg.hidden_dim, cfg.latent_dim),
            "state_act_proj": proj(cfg.latent_dim + cfg.action_dim, cfg.hidden_dim),
            "decoder": proj(cfg.latent_dim + cfg.hidden_dim, cfg.state_dim, cfg.latent_dim),
            "prior_proj": proj(cfg.hidden_dim, cfg.latent_dim),
        }

    def _init_hidden(self, batch_size, hidden_dim, device):
        return torch.zeros(batch_size, hidden_dim, device=device)

    def straight_with_gradient(self, logits: Tensor):
        probs = F.softmax(logits, dim=-1)
        sample = OneHotCategorical(probs=probs).sample()
        return sample + probs - probs.detach()

    def decode(self, state_latent: Tensor, perc_latent: Tensor, state_hidden=None, perc_hidden=None):
        if state_hidden is None or perc_hidden is None:
            state_hidden = self._init_hidden(state_latent.size(0), self.statecfg.hidden_dim, state_latent.device)
            perc_hidden = self._init_hidden(perc_latent.size(0), self.perccfg.hidden_dim, perc_latent.device)
        return (
            self.state_model["decoder"](torch.cat([state_latent, state_hidden], dim=-1)),
            self.perc_model["decoder"](torch.cat([perc_latent, perc_hidden], dim=-1)),
        )

    def sample_with_prior(self, state_latent, perc_latent, act, state_hidden=None, perc_hidden=None):
        assert state_latent.ndim == perc_latent.ndim == act.ndim == 2
        state_act = self.state_model["state_act_proj"](torch.cat([state_latent, act], dim=-1))
        perc_act = self.perc_model["state_act_proj"](torch.cat([perc_latent, act], dim=-1))
        if state_hidden is None or perc_hidden is None:
            state_hidden = self._init_hidden(state_latent.size(0), self.statecfg.hidden_dim, state_latent.device)
            perc_hidden = self._init_hidden(perc_latent.size(0), self.perccfg.hidden_dim, perc_latent.device)

        state_hidden = self.state_model["seq_model"](state_act, state_hidden)
        perc_hidden = self.perc_model["seq_model"](perc_act, perc_hidden)

        state_prior_logits = self.state_model["prior_proj"](state_hidden)
        perc_prior_logits = self.perc_model["prior_proj"](perc_hidden)
        state_prior_logits = state_prior_logits.view(*state_prior_logits.shape[:-1], self.state_categoricals, -1)
        perc_prior_logits = perc_prior_logits.view(*perc_prior_logits.shape[:-1], self.perc_categoricals, -1)

        if self.use_simnorm:
            return (
                state_prior_logits.softmax(dim=-1),
                state_prior_logits,
                state_hidden,
                perc_prior_logits.softmax(dim=-1),
                perc_prior_logits,
                perc_hidden,
            )
        else:
            return (
                self.straight_with_gradient(state_prior_logits),
                state_prior_logits,
                state_hidden,
                self.straight_with_gradient(perc_prior_logits),
                perc_prior_logits,
                perc_hidden,
            )

    def flatten(self, categorical_sample: Tensor):
        return categorical_sample.view(*categorical_sample.shape[:-2], -1)

    def sample_with_post(self, state, perc, state_hidden=None, perc_hidden=None):
        if state_hidden is None or perc_hidden is None:
            state_hidden = self._init_hidden(state.size(0), self.statecfg.hidden_dim, state.device)
            perc_hidden = self._init_hidden(perc.size(0), self.perccfg.hidden_dim, perc.device)

        state_post_logits = self.state_model["inp_proj"](torch.cat([state, state_hidden], dim=-1))
        state_post_logits = state_post_logits.view(*state_post_logits.shape[:-1], self.state_categoricals, -1)
        perc_post_logits = self.perc_model["inp_proj"](torch.cat([perc, perc_hidden], dim=-1))
        perc_post_logits = perc_post_logits.view(*perc_post_logits.shape[:-1], self.perc_categoricals, -1)

        if self.use_simnorm:
            return state_post_logits.softmax(dim=-1), state_post_logits, perc_post_logits.softmax(dim=-1), perc_post_logits
        else:
            return (
                self.straight_with_gradient(state_post_logits),
                state_post_logits,
                self.straight_with_gradient(perc_post_logits),
                perc_post_logits,
            )

    @torch.no_grad()
    def predict_next(self, state_latent, perc_latent, act, state_hidden=None, perc_hidden=None):
        state_prior_sample, _, state_hidden, perc_prior_sample, _, perc_hidden = self.sample_with_prior(
            state_latent, perc_latent, act, state_hidden, perc_hidden
        )
        state_flat_prior = self.flatten(state_prior_sample)
        perc_flat_prior = self.flatten(perc_prior_sample)

        reward_logit = self.reward_predictor(state_hidden, perc_hidden)
        end_logit = self.end_predictor(state_hidden, perc_hidden)
        return (
            state_prior_sample,
            perc_prior_sample,
            self.symlogtwohotloss.decode(reward_logit),
            end_logit > 0,
            state_hidden,
            perc_hidden,
        )

    def compute_loss(self, states, percs, actions, rewards, terminations):
        b, l, _ = states.shape
        state_hidden = self._init_hidden(b, self.statecfg.hidden_dim, states.device)
        perc_hidden = self._init_hidden(b, self.perccfg.hidden_dim, percs.device)

        rec_states, rec_percs, reward_logits, end_logits = [], [], [], []
        state_post_logits, state_prior_logits = [], []
        perc_post_logits, perc_prior_logits = [], []

        for i in range(l):
            state_post_sample, state_post_logit, perc_post_sample, perc_post_logit = self.sample_with_post(
                states[:, i], percs[:, i], state_hidden, perc_hidden
            )
            state_flat_post = self.flatten(state_post_sample)
            perc_flat_post = self.flatten(perc_post_sample)

            rec_state, rec_perc = self.decode(state_flat_post, perc_flat_post, state_hidden, perc_hidden)
            rec_states.append(rec_state)
            rec_percs.append(rec_perc)

            action = actions[:, i]
            state_prior_sample, state_prior_logit, state_hidden, perc_prior_sample, perc_prior_logit, perc_hidden = self.sample_with_prior(
                state_flat_post, perc_flat_post, action, state_hidden, perc_hidden
            )
            state_post_logits.append(state_post_logit)
            state_prior_logits.append(state_prior_logit)
            perc_post_logits.append(perc_post_logit)
            perc_prior_logits.append(perc_prior_logit)

            reward_logits.append(self.reward_predictor(state_hidden, perc_hidden))
            end_logits.append(self.end_predictor(state_hidden, perc_hidden))

        # Stack tensors
        rec_states, rec_percs = map(lambda x: torch.stack(x, dim=1), (rec_states, rec_percs))
        state_post_logits, state_prior_logits = map(lambda x: torch.stack(x, dim=1), (state_post_logits, state_prior_logits))
        perc_post_logits, perc_prior_logits = map(lambda x: torch.stack(x, dim=1), (perc_post_logits, perc_prior_logits))
        reward_logits, end_logits = map(lambda x: torch.stack(x, dim=1), (reward_logits, end_logits))

        # Loss calculation
        state_rep_loss, _ = self.kl_loss(state_post_logits[:, 1:], state_prior_logits[:, :-1].detach())
        state_dyn_loss, _ = self.kl_loss(state_post_logits[:, 1:].detach(), state_prior_logits[:, :-1])
        perc_rep_loss, _ = self.kl_loss(perc_post_logits[:, 1:], perc_prior_logits[:, :-1].detach())
        perc_dyn_loss, _ = self.kl_loss(perc_post_logits[:, 1:].detach(), perc_prior_logits[:, :-1])

        rew_loss = self.symlogtwohotloss(reward_logits, rewards)
        end_loss = self.endloss(end_logits, terminations)

        state_rec_loss = F.mse_loss(rec_states, states)
        perc_rec_loss = F.mse_loss(rec_percs, percs)

        total_loss = (
            state_rec_loss
            + perc_rec_loss
            + 0.5 * (state_dyn_loss + perc_dyn_loss)
            + 0.1 * (state_rep_loss + perc_rep_loss)
            + rew_loss
            + end_loss
        )

        return total_loss, state_rep_loss, state_dyn_loss, state_rec_loss, perc_rep_loss, perc_dyn_loss, perc_rec_loss, rew_loss, end_loss

    
        
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
