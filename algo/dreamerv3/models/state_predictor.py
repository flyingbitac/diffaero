from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch import Tensor
from torch.distributions import OneHotCategorical
from einops import rearrange,reduce
from einops.layers.torch import Rearrange

from .blocks import SymLogTwoHotLoss, MLP

from diffaero.utils.runner import timeit

@dataclass
class DepthStateModelCfg:
    state_dim: int
    image_width: int
    image_height: int
    hidden_dim: int
    action_dim: int
    latent_dim: int
    categoricals: int
    num_classes: int
    end_loss_pos_weight: float
    img_recon_loss_weight: float
    use_simnorm: bool=False
    use_grid: bool=False
    grid_dim: int=4000
    only_state: bool=False
    enable_rec: bool=True
    rec_coef: float=1.0
    rew_coef: float=1.0
    end_coef: float=1.0
    rep_coef: float=0.1
    dyn_coef: float=0.5
    ocp_coef: float=0.5

@dataclass
class Batch:
    obs:torch.ByteTensor
    act:torch.LongTensor
    rew:torch.FloatTensor
    end:torch.LongTensor
    mask_padding:torch.BoolTensor
    drone_state:torch.FloatTensor
    obstacle_relpos:torch.FloatTensor

def cal_image_width(image_width:int,kernel_size:int,stride:int,padding:int):
    return (image_width-kernel_size+2*padding)//stride + 1

class ImageEncoder(nn.Module):
    def __init__(self, in_channels:int, stem_channels:int, image_width:int, image_height:int):
        super().__init__()
        backbone = []
        
        backbone.append(nn.Conv2d(in_channels, stem_channels, kernel_size=4,stride=2,padding=1,bias=False))
        backbone.append(nn.BatchNorm2d(stem_channels))
        backbone.append(nn.ReLU(inplace=True))
        image_width  = cal_image_width(image_width,4,2,1)
        image_height = cal_image_width(image_height,4,2,1)
        
        while image_width>4:
            backbone.append(nn.Conv2d(stem_channels, 2*stem_channels, kernel_size=4,stride=2,padding=1,bias=False))
            backbone.append(nn.BatchNorm2d(2*stem_channels))
            backbone.append(nn.ReLU(inplace=True))
            stem_channels*=2
            image_width = cal_image_width(image_width,4,2,1)
            image_height = cal_image_width(image_height,4,2,1)
        
        self.backbone = nn.Sequential(*backbone)
        self.last_channels = stem_channels
        self.flatten_dim = stem_channels*image_width*image_height
        
    def forward(self, depth_image:Tensor):
        depth_image = self.backbone(depth_image)
        # depth_image = rearrange(depth_image, "B C H W -> B (C H W)")
        depth_image = depth_image.view(depth_image.shape[0],-1)
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

class ImageDecoderMLP(nn.Module):
    def __init__(self, feat_dim:int, hidden_dim:int, image_width:int, image_height:int):
        super().__init__()
        self.backbone = MLP(feat_dim, hidden_dim, hidden_dim, 2, 'SiLU', 'LayerNorm', bias=False)
        self.head = nn.Linear(hidden_dim,image_height*image_width)
        self.image_height = image_height

    def forward(self,feat:torch.Tensor):
        flatten_image = self.head(self.backbone(feat))
        rec_image = rearrange(flatten_image,'B (H W) -> B H W',H=self.image_height)
        return rec_image
        
@torch.jit.script
def onehotsample(probs:torch.Tensor):
    B,K,C = probs.shape
    
    flatten_probs = probs.view(-1,C)
    
    sample_indices = torch.multinomial(flatten_probs,1).squeeze()
    
    one_hot_samples = torch.zeros(B*K,C,device=probs.device)
    
    one_hot_samples.scatter_(1,sample_indices.unsqueeze(1),1)
    
    one_hot_samples = one_hot_samples.view(B,K,C)
    
    return one_hot_samples

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
    def __init__(self, num_classes:int, hidden_dim:int, latent_dim:int) -> None:
        super().__init__()
        self.backbone = MLP(latent_dim+hidden_dim, hidden_dim, hidden_dim, 2, 'SiLU', 'LayerNorm', bias=False)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, feat:Tensor, hidden:Tensor) -> torch.Tensor:
        feat = self.backbone(torch.cat([feat,hidden],dim=-1))
        reward = self.head(feat)
        return reward

class EndDecoder(nn.Module):
    def __init__(self, hidden_dim:int, latent_dim:int) -> None:
        super().__init__()
        self.backbone = MLP(hidden_dim+latent_dim, hidden_dim, hidden_dim, 2, 'SiLU', 'LayerNorm', bias=False)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, feat:Tensor, hidden:Tensor) -> torch.Tensor:
        feat = self.backbone(torch.cat([feat,hidden],dim=-1))
        end = self.head(feat)
        return end.squeeze(-1)

class GridDecoder(nn.Module):
    def __init__(self, hidden_dim:int, latent_dim:int, grid_dim:int) -> None:
        super().__init__()
        self.backbone = MLP(hidden_dim+latent_dim, hidden_dim, hidden_dim, 2, 'SiLU', 'LayerNorm', bias=False)
        self.head = nn.Linear(hidden_dim, grid_dim)

    def forward(self, feat:Tensor, hidden:Tensor) -> torch.Tensor:
        feat = self.backbone(torch.cat([feat,hidden],dim=-1))
        grid = self.head(feat)
        return grid

class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        loss = reduce(loss, "B L C H W -> B L", "sum")
        return loss.mean()

class DepthStateModel(nn.Module):
    def __init__(self, cfg: DepthStateModelCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_simnorm = cfg.use_simnorm
        self.categoricals = cfg.categoricals
        self.kl_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        self.mse_loss = MSELoss()
        self.symlogtwohotloss = SymLogTwoHotLoss(cfg.num_classes,-20,20)
        self.endloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.end_loss_pos_weight))

        self.seq_model = nn.GRUCell(cfg.hidden_dim,cfg.hidden_dim)
        if not cfg.only_state:
            self.image_encoder = ImageEncoder(in_channels=1, stem_channels=16, image_width=cfg.image_width,
                                              image_height=cfg.image_height)
            if cfg.enable_rec:
                self.image_decoder = ImageDecoderMLP(feat_dim=cfg.latent_dim+cfg.hidden_dim,hidden_dim=cfg.hidden_dim,
                                                    image_width=cfg.image_width,image_height=cfg.image_height)
            state_emb_dim = 64
            self.state_encoder = MLP(cfg.state_dim, state_emb_dim, state_emb_dim, 1, 'SiLU', 'LayerNorm', bias=False)
            depth_flatten_dim = self.image_encoder.flatten_dim
        else:
            self.state_encoder = nn.Identity()
            state_emb_dim = self.cfg.state_dim
            depth_flatten_dim = 0
        if cfg.use_grid:
            self.grid_decoder = GridDecoder(cfg.hidden_dim, cfg.latent_dim, cfg.grid_dim)

        self.inp_proj = nn.Sequential(MLP(state_emb_dim + depth_flatten_dim + cfg.hidden_dim, cfg.latent_dim, 
                                          cfg.latent_dim, 1, 'SiLU', 'LayerNorm', bias=False),
                                      nn.Linear(cfg.latent_dim, cfg.latent_dim))
        
        self.act_state_proj = nn.Sequential(
            MLP(cfg.latent_dim+cfg.action_dim, cfg.hidden_dim, cfg.hidden_dim, 1, 'SiLU', 'LayerNorm', bias=False),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        )
        
        self.state_decoder = nn.Sequential(
            MLP(cfg.latent_dim+cfg.hidden_dim, cfg.latent_dim, cfg.hidden_dim, 1, 'SiLU', 'LayerNorm', bias=False),
            nn.Linear(cfg.latent_dim,cfg.state_dim)
        )
        
        self.prior_proj = nn.Sequential(
            MLP(cfg.hidden_dim, cfg.latent_dim, cfg.hidden_dim, 1, 'SiLU', 'LayerNorm', bias=False),
            nn.Linear(cfg.latent_dim, cfg.latent_dim)
        )

        self.reward_predictor = RewardDecoder(cfg.num_classes,cfg.hidden_dim,cfg.latent_dim)
        self.end_predictor = EndDecoder(cfg.hidden_dim,cfg.latent_dim)

    def straight_with_gradient(self,logits:Tensor):
        probs = F.softmax(logits,dim=-1)
        dist = OneHotCategorical(probs=probs)
        sample = dist.sample()
        sample_with_gradient = sample + probs - probs.detach()
        return sample_with_gradient
    
    def sample_for_deploy(self,logits:Tensor):
        probs = F.softmax(logits,dim=-1)
        return onehotsample(probs)
    
    def decode(self,latent:Tensor,hidden:Optional[Tensor]=None):
        if hidden is None:
            hidden = torch.zeros(latent.shape[0],self.cfg.hidden_dim,device=latent.device)
        feat = torch.cat([latent,hidden],dim=-1)
        if self.cfg.only_state or not self.cfg.enable_rec:
            return self.state_decoder(feat),None
        else:
            if feat.ndim==3:
                bz, sq = feat.shape[0], feat.shape[1]
                feat = rearrange(feat, "B L D -> (B L) D")
                states, videos = self.state_decoder(feat), self.image_decoder(feat)
                states = rearrange(states, "(B L) D -> B L D", B=bz, L=sq)
                videos = rearrange(videos, "(B L) H W -> B L H W", B=bz, L=sq)
                return states, videos
            else:
                return self.state_decoder(feat),self.image_decoder(feat)
    
    def sample_with_prior(self,latent:Tensor,act:Tensor,hidden:Optional[Tensor]=None,is_for_deploy:bool=False):
        assert latent.ndim==act.ndim==2
        state_act = self.act_state_proj(torch.cat([latent,act],dim=-1))
        if hidden is None:
            hidden = torch.zeros(state_act.shape[0],self.cfg.hidden_dim).to(state_act.device)
        hidden = self.seq_model(state_act,hidden)
        prior_logits = self.prior_proj(hidden)
        prior_logits = prior_logits.view(*prior_logits.shape[:-1],self.categoricals,-1)
        
        if is_for_deploy:
            prior_sample = self.sample_for_deploy(prior_logits)
            return prior_sample,prior_logits,hidden

        if self.use_simnorm:
            prior_probs = prior_logits.softmax(dim=-1)
            return prior_probs,prior_logits,hidden
        else:
            prior_sample = self.straight_with_gradient(prior_logits)
            return prior_sample,prior_logits,hidden

    def flatten(self,categorical_sample:Tensor):
        return categorical_sample.view(*categorical_sample.shape[:-2],-1)

    def sample_with_post(self,state:Tensor,depth_image:Tensor=None,hidden:Optional[Tensor]=None,is_for_deploy:bool=False):
        if hidden is None:
            hidden = torch.zeros(state.shape[0],self.cfg.hidden_dim,device=state.device)
        
        if depth_image is not None:
            state_feat = self.state_encoder(state)
            depth_feat = self.image_encoder(depth_image)
            feat = torch.cat([state_feat,depth_feat],dim=-1)
        else:
            feat = self.state_encoder(state)
        
        post_logits = self.inp_proj(torch.cat([feat,hidden],dim=-1))
        post_logits = post_logits.view(*post_logits.shape[:-1],self.categoricals,-1) # b l d -> b l c k
        
        if is_for_deploy:
            post_sample = self.sample_for_deploy(post_logits)
            return post_sample,post_logits
        
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
        reward_logit = self.reward_predictor(flattend_prior_sample,hidden)
        end_logit = self.end_predictor(flattend_prior_sample,hidden)
        pred_reward = self.symlogtwohotloss.decode(reward_logit)
        pred_end = end_logit>0
        if hasattr(self, 'grid_decoder'):
            grid_logits = self.grid_decoder(flattend_prior_sample, hidden)
            grid_logits = grid_logits>0
            grid_logits = grid_logits.float()
        else:
            grid_logits = None
        return prior_sample,pred_reward,pred_end,hidden,grid_logits

    @timeit
    def compute_loss(
        self,
        states: Tensor,
        depth_images: Tensor,
        actions: Tensor,
        rewards: Tensor,
        terminations: Tensor,
        grids: Optional[Tensor] = None,
    ):
        b, l, d = states.shape

        hidden = torch.zeros(b,self.cfg.hidden_dim,device=states.device)
        post_logits, prior_logits, reward_logits, end_logits = [], [], [], []
        grid_logits, rec_states, rec_images = [], [], []

        for i in range(l):
            if depth_images is not None:
                post_sample,post_logit = self.sample_with_post(states[:, i], depth_images[:, i], hidden)
            else:
                post_sample,post_logit = self.sample_with_post(states[:, i], None, hidden)
            flattend_post_sample = self.flatten(post_sample)
            rec_state,rec_image = self.decode(flattend_post_sample,hidden)
            action = actions[:, i]
            prior_sample, prior_logit, hidden = self.sample_with_prior(flattend_post_sample, action, hidden)
            flattened_prior_sample = self.flatten(prior_sample)
            reward_logit = self.reward_predictor(flattened_prior_sample,hidden)
            end_logit = self.end_predictor(flattened_prior_sample,hidden)
            if hasattr(self, 'grid_decoder'):
                grid_logit = self.grid_decoder(flattened_prior_sample, hidden)
                grid_logits.append(grid_logit)

            rec_states.append(rec_state)
            rec_images.append(rec_image)
            post_logits.append(post_logit)
            prior_logits.append(prior_logit)
            reward_logits.append(reward_logit)
            end_logits.append(end_logit)

        rec_states = torch.stack(rec_states,dim=1)
        if rec_image is not None:
            rec_images = torch.stack(rec_images,dim=1).unsqueeze(2)
        post_logits = torch.stack(post_logits,dim=1)
        prior_logits = torch.stack(prior_logits,dim=1)
        reward_logits = torch.stack(reward_logits,dim=1)
        end_logits = torch.stack(end_logits,dim=1)

        rep_loss,_ = self.kl_loss(post_logits[:,1:],prior_logits[:,:-1].detach())
        dyn_loss,_ = self.kl_loss(post_logits[:,1:].detach(),prior_logits[:,:-1])
        rew_loss = self.symlogtwohotloss(reward_logits,rewards)
        end_loss = self.endloss(end_logits,terminations)
        if hasattr(self, 'grid_decoder'):
            grid_logits = torch.stack(grid_logits, dim=1)
            grid_loss = self.endloss(grid_logits[:, :-1],grids[:, 1:].float())
        else:
            grid_loss = torch.zeros(())

        if rec_image!=None:
            rec_loss = torch.sum((rec_states-states)**2,dim=-1).mean() + self.mse_loss(rec_images,depth_images)
        else:
            rec_loss = torch.sum((rec_states-states)**2,dim=-1).mean()
        # total_loss = rec_loss + 0.5*dyn_loss + 0.1*rep_loss + rew_loss + end_loss + grid_loss
        total_loss = self.cfg.rec_coef * rec_loss + self.cfg.dyn_coef * dyn_loss + self.cfg.rep_coef * rep_loss \
                     + self.cfg.rew_coef * rew_loss + self.cfg.end_coef * end_loss + self.cfg.ocp_coef * grid_loss

        return total_loss, rep_loss, dyn_loss, rec_loss, rew_loss, end_loss, grid_loss

if __name__=='__main__':

    cfg = DepthStateModelCfg()
    cfg.action_dim = 4
    cfg.categoricals = 16
    cfg.hidden_dim = 256
    cfg.latent_dim = 256
    cfg.state_dim = 13
    state_predictor = DepthStateModel(cfg)

    batch = Batch(None,torch.randn(5,10,4),None,None,None,torch.randn(5,10,13),None)
    state_loss = state_predictor.compute_loss(batch)
    print(state_loss)
