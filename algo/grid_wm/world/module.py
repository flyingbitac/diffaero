import math
from typing import List, Dict
from copy import deepcopy

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from quaddif.network.networks import CNNBackbone, mlp

class MLP(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, hidden_units: List, norm:str=None, act:str=None):
        super().__init__()
        module_list = nn.ModuleList()
        dims = [input_dim] + hidden_units + [output_dim]
        for inp_dim, out_dim in zip(dims[:-1], dims[1:]):
            module_list.append(nn.Linear(inp_dim, out_dim))
            if norm != None:
                module_list.append(getattr(nn, norm)(out_dim))
            if act != None:
                module_list.append(getattr(nn, act)())
        self.mlp = nn.Sequential(*module_list)
    
    def forward(self, x):
        return self.mlp(x)

class MiniGru(nn.Module):
    def __init__(self, deter:int, stoch:int, action_dim:int, hidden:int, act:str, norm:str):
        super().__init__()
        self.stoch_action_proj = mlp(stoch+action_dim, [], hidden)
        self._core = nn.Linear(hidden + deter, 3 * deter, bias = False)
        self._core_norm = nn.LayerNorm(3*deter)
    
    def forward(self, deter:torch.Tensor, stoch:torch.Tensor, action:torch.Tensor):
        tokens = torch.cat([stoch, action], dim = -1)
        tokens = self.stoch_action_proj(tokens)
        parts = self._core_norm(self._core(torch.cat([tokens, deter], dim = -1)))
        reset, cand, update = torch.split(parts, parts.size(-1)//3, dim = -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.)
        deter = update * cand + (1 - update) * deter
        return deter


class ImageEncoder(nn.Module):
    def __init__(self, image_shape:List):
        super().__init__()
        self.encoder = CNNBackbone([0, image_shape])
        self.final_shape = self.encoder.out_dim
    
    def forward(self, x:torch.Tensor):
        if x.ndim == 3:
            x = self.encoder(x.unsqueeze(1))
        elif x.ndim == 4:
            B, L, H, W = x.shape
            x = self.encoder(x.reshape(B*L, 1, H, W))
            x = x.reshape(B, L, -1)
        else:
            raise ValueError(f'Invalid input dimension {x.ndim}')
        return x


class ImageDecoder(nn.Module):
    def __init__(self, final_image_shape:List, feat_dim:int, channels:List[int], stride:int, kernel_size:int, act:str, norm:str):
        super().__init__()
        dconv_list = nn.ModuleList()
        self.proj = nn.Linear(feat_dim, math.prod(final_image_shape), bias=False)
        self.final_shape = final_image_shape
        channels = [final_image_shape[0]] + channels 
        for in_channel, out_channel in zip(channels[:-2], channels[1:-1]):
            dconv_list.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding = stride // 2))
            dconv_list.append(getattr(nn, norm)(out_channel))
            dconv_list.append(getattr(nn, act)())
        dconv_list.append(nn.ConvTranspose2d(channels[-2], channels[-1], kernel_size, stride, padding = stride // 2))
        self.decoder = nn.Sequential(*dconv_list)
    
    def forward(self, x:torch.Tensor):
        x = self.proj(x)
        x = rearrange(x, '... (c h w) -> ... c h w', c=self.final_shape[0], h=self.final_shape[1])
        if x.ndim == 5:
            batch_size = x.size(0)
            x = rearrange(x, 'b t ... -> (b t) ...')
            x = F.sigmoid(self.decoder(x))
            x = rearrange(x, '(b t) c h w -> b t c h w', b = batch_size)
        else:
            x = F.sigmoid(self.decoder(x))
        return x

class ImageDecoderMLP(nn.Module):
    def __init__(self, final_image_shape:List, feat_dim:int, hidden_units:List[int], act:str, norm:str):
        super().__init__()
        # self.backbone = MLP(feat_dim, hidden_units[-1], hidden_units[:-1], norm, act)
        self.backbone = mlp(feat_dim, hidden_units[:-1], hidden_units[-1])
        self.proj = nn.Linear(hidden_units[-1], math.prod(final_image_shape))
        self.final_shape = final_image_shape
    
    def forward(self, x:torch.Tensor):
        x = self.proj(self.backbone(x))
        x = rearrange(x, '... (c h w) -> ... c h w', c=self.final_shape[0], h=self.final_shape[1])
        x = F.sigmoid(x)
        return x


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
        reshaped_fmap = flattened_fmap.reshape(
            *flattened_fmap.shape[:-1], self.fmap_channels, *self.fmap_shape)
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


class StateEncoder(nn.Module):
    def __init__(self, state_enc_cfg: DictConfig):
        super().__init__()
        self.encode_target_vel: bool = state_enc_cfg.encode_target_vel
        self.encode_quat: bool = state_enc_cfg.encode_quat
        self.encode_vel: bool = state_enc_cfg.encode_vel
        assert self.encode_target_vel or self.encode_quat or self.encode_vel
        input_dim = (
            self.encode_target_vel * 3 +
            self.encode_quat * 4 +
            self.encode_vel * 3
        )
        self.state_encoder = mlp(input_dim, state_enc_cfg.hidden_units, state_enc_cfg.embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_vel, quat, vel = x[..., 0:3], x[..., 3:7], x[..., 7:10]
        inputs = []
        if self.encode_target_vel:
            inputs.append(target_vel)
        if self.encode_quat:
            inputs.append(quat)
        if self.encode_vel:
            inputs.append(vel)
        return self.state_encoder(torch.cat(inputs, dim=-1))


class Encoder(nn.Module):
    def __init__(self, obs_space:Dict, channels:List[int], stride:int, kernel_size:int, 
                 embed:int, hidden:int, layers:int, act:str, norm:str):
        super().__init__()
        for key, value in obs_space.items():
            if key == 'perception' and value != None:
                self.img_encoder = ImageEncoder(value, channels, stride, kernel_size, act, norm)
            elif key == 'state' and value != None:
                self.state_encoder = MLP(value[0], embed, layers*[hidden], 'LayerNorm', act)
            else:
                raise NotImplementedError(f'Unknown observation space key: {key}')
        final_shape = self.img_encoder.final_shape
        self.token_dim = math.prod(final_shape) + embed
        self.final_shape = final_shape
    
    def forward(self, x:Dict):
        token = []
        if hasattr(self, 'img_encoder'):
            token.append(self.img_encoder(x['perception']))
        if hasattr(self, 'state_encoder'):
            token.append(self.state_encoder(x['state']))
        token = torch.cat(token, dim = -1)
        return token

class Decoder(nn.Module):
    def __init__(self, obs_space:Dict, final_shape:List, feat_dim:int, channels:List[int], stride:int, 
                 kernel_size:int, hidden:int, act:str, norm:str, use_dconv:bool=False):
        super().__init__()
        self.use_dconv = use_dconv
        for key, value in obs_space.items():
            if key == 'perception' and value != None:
                self.img_shape = value
                if use_dconv:
                    self.img_decoder = ImageDecoder(final_shape, feat_dim, channels,stride, kernel_size, act, norm)
                else:
                    self.img_decoder = nn.Sequential(MLP(feat_dim, hidden, 2*[hidden], 'LayerNorm', act),
                                                     nn.Linear(hidden, math.prod(value)),
                                                     nn.Sigmoid())
            elif key == 'state' and value != None:
                self.state_decoder = nn.Sequential(MLP(feat_dim, hidden, [hidden], 'LayerNorm', act),
                                                   nn.Linear(hidden, value[0]))
            else:
                raise NotImplementedError
    
    def forward(self, feats:torch.Tensor):
        rec = {}
        if hasattr(self, 'img_decoder'):
            image = self.img_decoder(feats)
            if not self.use_dconv:
                image = rearrange(image, '... (c h w) -> ... c h w', c=self.img_shape[0], h=self.img_shape[1])
            rec['perception'] = image
        if hasattr(self, 'state_decoder'):
            state = self.state_decoder(feats)
            rec['state'] = state
        return rec

if __name__ == '__main__': 
    obs_space = {'perception':[3, 64, 64], 'state':[9]}
    enc = Encoder(obs_space, [16, 32, 64, 128], 2, 4, 64, 512, 1, 'ReLU', 'BatchNorm2d')
    inp = {'perception':torch.randn(3, 3, 64, 64), 'state':torch.randn(3, 9)}
    print(enc(inp).shape)
    print(enc.token_dim)
    dec = Decoder(obs_space, enc.final_shape, 1536, [64, 32, 16, 3], 2, 4, 512, 'ReLU', 'BatchNorm2d', True)
    rec=dec(torch.randn(10, 1536))
    for key, value in rec.items():
        print(key, value.shape)