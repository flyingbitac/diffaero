import sys
sys.path.append('..')

import torch
from torch import Tensor
import torch.nn as nn

from diffaero.utils.logger import Logger

class YOPONet(nn.Module):
    def __init__(
        self,
        H_out: int,
        W_out: int,
        feature_dim: int,
        head_hidden_dim: int,
        out_dim: int
    ):
        super().__init__()
        self.H_out = H_out
        self.W_out = W_out
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.Conv2d(16, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((H_out, W_out))
        )
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
        obs_p: Tensor # []
    ):
        N, C, HW = obs_p.size(0), self.out_dim, self.H_out * self.W_out
        feat = self.net(depth_image)
        obs_p = obs_p.reshape(N, self.H_out, self.W_out, 9).permute(0, 3, 1, 2) # [N, 9, H_out, W_out]
        feat = torch.cat([feat, obs_p], dim=1) # [N, feature_dim + 9, H_out, W_out]
        return self.head(feat).reshape(N, C, HW).permute(0, 2, 1) # [N, H_out*W_out, out_dim]