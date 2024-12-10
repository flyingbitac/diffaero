from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import Conv3x3, FourierFeatures, GroupNorm, UNet,CrossAttn1d


@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    drone_states_dim: int
    obstacles_num: int
    d_model: int
    num_actions: Optional[int] = None


class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()
        self.noise_emb = FourierFeatures(cfg.cond_channels) #cond_channels=256
        self.act_state_emb = nn.Sequential(
            nn.Linear(cfg.num_actions + cfg.d_model,cfg.cond_channels // cfg.num_steps_conditioning),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels // cfg.num_steps_conditioning,cfg.cond_channels // cfg.num_steps_conditioning), #num_steps_conditioning=4
            nn.Flatten(),  # b t e -> b (t e)
        )
        self.drone_states_emb = nn.Sequential(
            nn.Linear(cfg.drone_states_dim,cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.SiLU(),
        )
        self.obstacles_emb = nn.Sequential(
            nn.Linear(3,cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.SiLU(),
        )
        self.states_obstacles_attn = CrossAttn1d(d_model=cfg.d_model,num_heads=4)
        # self.attn_proj = nn.Sequential(
        #     nn.Linear(cfg.d_model,cfg.cond_channels // cfg.num_steps_conditioning),
        #     nn.LayerNorm(cfg.cond_channels // cfg.num_steps_conditioning),
        #     nn.SiLU(),
        #     nn.Linear(cfg.cond_channels // cfg.num_steps_conditioning,cfg.cond_channels // cfg.num_steps_conditioning),
        #     nn.Flatten(), # b t e -> b (t e)
        # )

        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, cfg.channels[0]) #img_channels=3

        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths) #256,[2,2,2,2],[64,64,64,64],[0,0,0,0]

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, obs: Tensor, 
                act: Tensor,drone_states:Tensor,obstacles:Tensor) -> Tensor:
        assert len(drone_states.shape)==3 and len(obstacles.shape)==4
        b,l,n,d = obstacles.shape
        drone_states = drone_states.unsqueeze(-2)
        obstacles_z = obstacles[:,:,:,-1:0]
        mask = torch.ones(b,l,n,1).bool().to(obs.device)
        mask.masked_fill(obstacles_z<=-1000,0)
        mask = mask.transpose(-1,-2)

        drone_obstacles_attn,_ = self.states_obstacles_attn(self.drone_states_emb(drone_states),self.obstacles_emb(obstacles),mask) #b,l,1,d_model
        drone_obstacles_attn = drone_obstacles_attn.squeeze(-2) #b,l,d_model
        
        # cond = self.cond_proj(self.noise_emb(c_noise) + self.act_emb(act) + self.attn_proj(drone_obstacles_attn))
        cond = self.cond_proj(self.noise_emb(c_noise) + self.act_state_emb(torch.cat([act,drone_obstacles_attn],dim=-1)))
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1)) #b (t+1)*c h w
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x

def main():
    inner_cfg = InnerModelConfig(
        img_channels=3,
        num_steps_conditioning=4,
        cond_channels=256,
        depths=[2,2,2,2],
        channels=[64,64,64,64],
        attn_depths=[False,False,False,False],
        num_actions=5
    )
    innermodel = InnerModel(inner_cfg)
    noisy_next_obs = torch.randn(16,3,64,64)
    obs = torch.randn(16,3*4,64,64)
    c_noise = torch.randn(16,)
    act = torch.randint(0,4,(16,4))
    x = innermodel(noisy_next_obs,c_noise,obs,act)
    print(x.shape)




