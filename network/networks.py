from typing import Tuple, Dict, Union, Optional, List
from math import ceil

from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn
from diffaero.utils.nn import mlp

def obs_action_concat(state: Union[Tensor, Tuple[Tensor, Tensor]], action: Optional[Tensor] = None) -> Tensor:
    if isinstance(state, Tensor):
        return torch.cat([state, action], dim=-1) if action is not None else state
    else:
        return torch.cat([state[0], state[1].flatten(-2)] + ([] if action is None else [action]), dim=-1)

class BaseNetwork(nn.Module):
    def __init__(
        self,
        input_dim: Union[int, Tuple[int, Tuple[int, int]]],
        rnn_n_layers: int = 0,
        rnn_hidden_dim: int = 0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.rnn_n_layers = rnn_n_layers
        self.rnn_hidden_dim = rnn_hidden_dim
    
    def reset(self, indices: Tensor) -> None:
        pass
    
    def detach(self) -> None:
        pass

class MLP(BaseNetwork):
    def __init__(
        self,
        cfg: DictConfig,
        input_dim: Union[int, Tuple[int, Tuple[int, int]]],
        output_dim: int,
        output_act: Optional[nn.Module] = None
    ):
        super().__init__(input_dim)
        if not isinstance(input_dim, int):
            D, (H, W) = input_dim
            input_dim = D + H * W
        self.head = mlp(input_dim, cfg.hidden_dim, output_dim, output_act=output_act)
    
    def forward(
        self,
        obs: Union[Tensor, Tuple[Tensor, Tensor]], # [N, D_state] or ([N, D_state], [N, H, W])
        action: Optional[Tensor] = None, # [N, D_action]
        hidden: Optional[Tensor] = None
    ) -> Tensor:
        return self.head(obs_action_concat(obs, action))
    
    def forward_export(
        self,
        obs: Union[Tensor, Tuple[Tensor, Tensor]], # [N, D_obs]
        action: Optional[Tensor] = None, # [N, D_action]
    ) -> Tensor:
        return self.forward(obs=obs, action=action)


class CNNBackbone(nn.Sequential):
    def __init__(self, input_dim: Tuple[int, Tuple[int, int]]):
        D, (H, W) = input_dim
        layers = [
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.Flatten(start_dim=-3)
        ]
        ds_rate = 4
        if any([H % ds_rate != 0, W % ds_rate != 0]):
            hpad = ceil(H / ds_rate) * ds_rate - H
            wpad = ceil(W / ds_rate) * ds_rate - W
            top, left = hpad // 2, wpad // 2
            bottom, right = hpad - top, wpad - left
            layers.insert(0, nn.ZeroPad2d((left, right, top, bottom)))
        hout, wout = ceil(H / ds_rate), ceil(W / ds_rate)
        super().__init__(*layers)
        self.out_dim = D + 8 * hout * wout

class CNN(BaseNetwork):
    def __init__(
        self,
        cfg: DictConfig,
        input_dim: Tuple[int, Tuple[int, int]],
        output_dim: int,
        output_act: Optional[nn.Module] = None
    ):
        super().__init__(input_dim)
        self.cnn = CNNBackbone(input_dim)
        self.head = mlp(self.cnn.out_dim, cfg.hidden_dim, output_dim, output_act=output_act)
    
    def forward(
        self,
        obs: Tuple[Tensor, Tensor], # ([N, D_state], [N, H, W])
        action: Optional[Tensor] = None, # [N, D_action]
        hidden: Optional[Tensor] = None
    ) -> Tensor:
        perception = obs[1]
        if perception.ndim == 3:
            perception = perception.unsqueeze(1)
        input = [obs[0], self.cnn(perception)] + ([] if action is None else [action])
        return self.head(torch.cat(input, dim=-1))
    
    def forward_export(
        self,
        state: Tensor, # [N, D_state]
        perception: Tensor, # [N, H, W]
        action: Optional[Tensor] = None, # [N, D_action]
    ) -> Tensor:
        if perception.ndim == 3:
            perception = perception.unsqueeze(1)
        input = [state, self.cnn(perception)] + ([] if action is None else [action])
        return self.head(torch.cat(input, dim=-1))


class RNN(BaseNetwork):
    def __init__(
        self,
        cfg: DictConfig,
        input_dim: Union[int, Tuple[int, Tuple[int, int]]],
        output_dim: int,
        output_act: Optional[nn.Module] = None
    ):
        super().__init__(input_dim, cfg.rnn_n_layers, cfg.rnn_hidden_dim)
        if not isinstance(input_dim, int):
            D, (H, W) = input_dim
            input_dim = D + H * W
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=self.rnn_hidden_dim,
            num_layers=self.rnn_n_layers,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
            dtype=torch.float
        )
        self.head = mlp(self.rnn_hidden_dim, cfg.hidden_dim, output_dim, output_act=output_act)
        self.hidden_state: Optional[Tensor] = None
    
    def forward(
        self,
        obs: Union[Tensor, Tuple[Tensor, Tensor]], # [N, D_state] or ([N, D_state], [N, H, W])
        action: Optional[Tensor] = None, # [N, D_action]
        hidden: Optional[Tensor] = None, # [n_layers, N, D_hidden]
    ) -> Tensor:
        # self.gru.flatten_parameters()
        rnn_input = obs_action_concat(obs, action)
        
        use_own_hidden = hidden is None
        if use_own_hidden:
            if self.hidden_state is None:
                hidden = torch.zeros(self.rnn_n_layers, rnn_input.size(0), self.rnn_hidden_dim, dtype=rnn_input.dtype, device=rnn_input.device)
            else:
                hidden = self.hidden_state
        
        rnn_out, hidden = self.gru(rnn_input.unsqueeze(1), hidden)
        if use_own_hidden:
            self.hidden_state = hidden
        return self.head(rnn_out.squeeze(1))
    
    def forward_export(
        self,
        obs: Union[Tensor, Tuple[Tensor, Tensor]], # [N, D_state] or ([N, D_state], [N, H, W])
        hidden: Tensor, # [n_layers, N, D_hidden]
        action: Optional[Tensor] = None, # [N, D_action]
    ) -> Tuple[Tensor, Tensor]:
        rnn_input = obs_action_concat(obs, action)
        rnn_out, hidden = self.gru(rnn_input.unsqueeze(1), hidden)
        return self.head(rnn_out.squeeze(1)), hidden

    def reset(self, indices: Tensor):
        self.hidden_state[:, indices, :] = 0
    
    def detach(self):
        self.hidden_state.detach_()


class RCNN(BaseNetwork):
    def __init__(
        self,
        cfg: DictConfig,
        input_dim: Tuple[int, Tuple[int, int]],
        output_dim: int,
        output_act: Optional[nn.Module] = None
    ):
        super().__init__(input_dim, cfg.rnn_n_layers, cfg.rnn_hidden_dim)
        self.cnn = CNNBackbone(input_dim)
        self.gru = torch.nn.GRU(
            input_size=self.cnn.out_dim,
            hidden_size=self.rnn_hidden_dim,
            num_layers=self.rnn_n_layers,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
            dtype=torch.float
        )
        self.head = mlp(self.rnn_hidden_dim, cfg.hidden_dim, output_dim, output_act=output_act)
        self.hidden_state: Tensor = None
    
    def forward(
        self,
        obs: Tuple[Tensor, Tensor], # ([N, D_state], [N, H, W])
        action: Optional[Tensor] = None, # [N, D_action]
        hidden: Optional[Tensor] = None, # [n_layers, N, D_hidden]
    ) -> Tensor:
        # self.gru.flatten_parameters()
        
        perception = obs[1]
        if perception.ndim == 3:
            perception = perception.unsqueeze(1)
        rnn_input = torch.cat([obs[0], self.cnn(perception)] + ([] if action is None else [action]), dim=-1)
        
        use_own_hidden = hidden is None
        if use_own_hidden:
            if self.hidden_state is None:
                hidden = torch.zeros(self.rnn_n_layers, rnn_input.size(0), self.rnn_hidden_dim, dtype=rnn_input.dtype, device=rnn_input.device)
            else:
                hidden = self.hidden_state
        
        rnn_out, hidden = self.gru(rnn_input.unsqueeze(1), hidden)
        if use_own_hidden:
            self.hidden_state = hidden
        return self.head(rnn_out.squeeze(1))
    
    def forward_export(
        self,
        state: Tensor, # [N, D_state]
        perception: Tensor, # [N, H, W]
        hidden: Tensor, # [n_layers, N, D_hidden]
        action: Optional[Tensor] = None, # [N, D_action]
    ) -> Tuple[Tensor, Tensor]:
        if perception.ndim == 3:
            perception = perception.unsqueeze(1)
        rnn_input = torch.cat([state, self.cnn(perception)] + ([] if action is None else [action]), dim=-1)
        rnn_out, hidden = self.gru(rnn_input.unsqueeze(1), hidden)
        return self.head(rnn_out.squeeze(1)), hidden

    def reset(self, indices: Tensor):
        self.hidden_state[:, indices, :] = 0
    
    def detach(self):
        self.hidden_state.detach_()