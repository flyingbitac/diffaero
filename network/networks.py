from typing import Tuple, Dict, Union, Optional, List

from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn
from quaddif.utils.nn import mlp

def state_action_concat(state: Union[Tensor, Tuple[Tensor, Tensor]], action: Optional[Tensor] = None) -> Tensor:
    if isinstance(state, Tensor):
        return torch.cat([state, action], dim=-1) if action is not None else state
    else:
        return torch.cat([state[0], state[1].flatten(-2)] + ([] if action is None else [action]), dim=-1)

class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    
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
        super().__init__()
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
        return self.head(state_action_concat(obs, action))
    
    def forward_export(
        self,
        obs: Union[Tensor, Tuple[Tensor, Tensor]], # [N, D_obs]
        action: Optional[Tensor] = None, # [N, D_action]
    ) -> Tensor:
        return self.forward(obs=obs, action=action)


class CNNBackbone(nn.Sequential):
    def __init__(self, input_dim: Tuple[int, Tuple[int, int]]):
        super().__init__(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.Flatten(start_dim=-3)
        )
        D, (H, W) = input_dim
        self.out_dim = D + 8 * (H // 4) * (W // 4)

class CNN(BaseNetwork):
    def __init__(
        self,
        cfg: DictConfig,
        input_dim: Tuple[int, Tuple[int, int]],
        output_dim: int,
        output_act: Optional[nn.Module] = None
    ):
        super().__init__()
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
        obs: Tuple[Tensor, Tensor], # ([N, D_state], [N, H, W])
        action: Optional[Tensor] = None, # [N, D_action]
    ) -> Tensor:
        return self.forward(obs=obs, action=action)


class RNN(BaseNetwork):
    def __init__(
        self,
        cfg: DictConfig,
        input_dim: Union[int, Tuple[int, Tuple[int, int]]],
        output_dim: int,
        output_act: Optional[nn.Module] = None
    ):
        super().__init__()
        if not isinstance(input_dim, int):
            D, (H, W) = input_dim
            input_dim = D + H * W
        self.rnn_hidden_dim = cfg.rnn_hidden_dim
        self.n_layers = cfg.rnn_n_layers
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=self.rnn_hidden_dim,
            num_layers=self.n_layers,
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
        rnn_input = state_action_concat(obs, action)
        
        use_own_hidden = hidden is None
        if use_own_hidden:
            if self.hidden_state is None:
                hidden = torch.zeros(self.n_layers, rnn_input.size(0), self.rnn_hidden_dim, dtype=rnn_input.dtype, device=rnn_input.device)
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
        rnn_input = state_action_concat(obs, action)
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
        super().__init__()
        self.rnn_hidden_dim = cfg.rnn_hidden_dim
        self.n_layers = cfg.rnn_n_layers
        self.cnn = CNNBackbone(input_dim)
        self.gru = torch.nn.GRU(
            input_size=self.cnn.out_dim,
            hidden_size=self.rnn_hidden_dim,
            num_layers=self.n_layers,
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
                hidden = torch.zeros(self.n_layers, rnn_input.size(0), self.rnn_hidden_dim, dtype=rnn_input.dtype, device=rnn_input.device)
            else:
                hidden = self.hidden_state
        
        rnn_out, hidden = self.gru(rnn_input.unsqueeze(1), hidden)
        if use_own_hidden:
            self.hidden_state = hidden
        return self.head(rnn_out.squeeze(1))
    
    def forward_export(
        self,
        obs: Tuple[Tensor, Tensor], # ([N, D_state], [N, H, W])
        hidden: Tensor, # [n_layers, N, D_hidden]
        action: Optional[Tensor] = None, # [N, D_action]
    ) -> Tuple[Tensor, Tensor]:
        perception = obs[1]
        if perception.ndim == 3:
            perception = perception.unsqueeze(1)
        rnn_input = torch.cat([obs[0], self.cnn(perception)] + ([] if action is None else [action]), dim=-1)
        rnn_out, hidden = self.gru(rnn_input.unsqueeze(1), hidden)
        return self.head(rnn_out.squeeze(1)), hidden

    def reset(self, indices: Tensor):
        self.hidden_state[:, indices, :] = 0
    
    def detach(self):
        self.hidden_state.detach_()