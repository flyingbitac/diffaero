from typing import Optional, List, Tuple
from copy import deepcopy
import sys
sys.path.append('..')

import torch
from torch import Tensor
import torch.nn as nn
from omegaconf import DictConfig

from diffaero.network.networks import CNNBackbone
from diffaero.utils.logger import Logger

class YOPONet(nn.Module):
    def __init__(
        self,
        layers: List[Tuple[int, int, int]],
        img_h: int,
        img_w: int,
        h_out: int,
        w_out: int,
        head_hidden_dim: int,
        out_dim: int
    ):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.h_out = h_out
        self.w_out = w_out
        self.head_hidden_dim = head_hidden_dim
        self.out_dim = out_dim
        feature_dim = layers[-1][1]
        self.net = CNNBackbone(layers, input_dim=(0, (img_h, img_w)))
        assert self.net.h_out % h_out == 0 and self.net.w_out % w_out == 0, \
            f"feature map size ({self.net.h_out}, {self.net.w_out}) must be divisible by (h_out, w_out) ({h_out}, {w_out})"
        self.net.pop(-1)
        self.net.append(nn.AdaptiveAvgPool2d((h_out, w_out)))
        self.head = nn.Sequential(
            nn.Conv2d(feature_dim+9, head_hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.Conv2d(head_hidden_dim, head_hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.Conv2d(head_hidden_dim, out_dim, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(
        self,
        depth_image: Tensor,
        obs_p: Tensor, # [N, H_out*W_out, 9]
    ):
        N, C, HW = obs_p.size(0), self.out_dim, self.h_out * self.w_out
        feat = self.net(depth_image.unsqueeze(1))
        obs_p = obs_p.reshape(N, self.h_out, self.w_out, 9).permute(0, 3, 1, 2) # [N, 9, H_out, W_out]
        feat = torch.cat([feat, obs_p], dim=1) # [N, feature_dim + 9, H_out, W_out]
        return self.head(feat).reshape(N, C, HW).permute(0, 2, 1) # [N, H_out*W_out, out_dim]
