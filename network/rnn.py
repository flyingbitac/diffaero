from typing import Tuple, Dict, Union, Optional, List
from abc import ABC
import os

import torch
from torch import Tensor
import torch.nn as nn
from tensordict import TensorDict

from quaddif.utils.nn import mlp

class RNNBasedAgent(ABC):
    pass

class RNN(torch.nn.GRU):
    def __init__(
        self,
        input_dim: Union[int, Tuple[int, Tuple[int, int]]],
        rnn_hidden_dim: int,
        hidden_dim: Union[int, List[int]],
        output_dim: int,
        n_layers: int,
        output_act: Optional[nn.Module] = None
    ):
        if not isinstance(input_dim, int):
            D, (H, W) = input_dim
            input_dim = D + H * W
        super().__init__(
            input_size=input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
            dtype=torch.float
        )
        self.head = mlp(rnn_hidden_dim, hidden_dim, output_dim, output_act=output_act)
        self.n_layers = n_layers
        self.rnn_hidden_dim = rnn_hidden_dim
        self.hidden_state: Optional[Tensor] = None
    
    def forward(
        self,
        obs: Union[Tensor, TensorDict], # [N, D_in]
        hidden: Optional[Tensor] = None, # [n_layers, N, D_hidden]
        action: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        self.flatten_parameters()
        if isinstance(obs, TensorDict):
            obs = torch.cat([obs["state"], obs["perception"].flatten(1)] + ([] if action is None else [action]), dim=-1)
        use_own_hidden = hidden is None
        if use_own_hidden:
            if self.hidden_state is None:
                self.hidden_state = torch.zeros(self.n_layers, obs.size(0), self.rnn_hidden_dim, dtype=obs.dtype, device=obs.device)
            hidden = self.hidden_state
        else:
            assert hidden.size(1) == obs.size(0)
        rnn_out, hidden = super().forward(obs.unsqueeze(1), hidden)
        if use_own_hidden:
            self.hidden_state = hidden
        return self.head(rnn_out.squeeze(1))

    def reset(self, indices: Tensor):
        self.hidden_state[:, indices, :] = 0
    
    def detach(self):
        self.hidden_state.detach_()

class DeterministicActorRNN(nn.Module, RNNBasedAgent):
    def __init__(
        self,
        state_dim: Union[int, Tuple[int, Tuple[int, int]]],
        hidden_dim: Union[int, List[int]],
        action_dim: int,
        rnn_hidden_dim: int,
        n_layers: int
    ):
        super().__init__()
        self.actor = RNN(state_dim, rnn_hidden_dim, hidden_dim, action_dim, n_layers, output_act=nn.Tanh())
        
    def forward(self, obs: Union[Tensor, TensorDict], hidden: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        return self.actor(obs, hidden)
    
    def save(self, path: str):
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))

    def load(self, path: str):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth")))

    def reset(self, indices: Tensor):
        self.actor.reset(indices)
    
    def detach(self):
        self.actor.detach()


class StochasticActorRNN(nn.Module, RNNBasedAgent):
    def __init__(
        self,
        state_dim: Union[int, Tuple[int, Tuple[int, int]]],
        hidden_dim: Union[int, List[int]],
        action_dim: int,
        rnn_hidden_dim: int,
        n_layers: int
    ):
        super().__init__()
        self.actor_mean = RNN(state_dim, rnn_hidden_dim, hidden_dim, action_dim, n_layers)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, obs, sample=None, test=False, hidden=None):
        # type: (Union[Tensor, TensorDict], Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        action_mean = self.actor_mean(obs, hidden)
        LOG_STD_MAX = 2
        LOG_STD_MIN = -5
        action_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            torch.tanh(self.actor_logstd) + 1)  # From SpinUp / Denis Yarats
        action_std = torch.exp(action_logstd).expand_as(action_mean)
        probs = torch.distributions.Normal(action_mean, action_std)
        if sample is None and not test:
            sample = torch.randn_like(action_mean) * action_std + action_mean
        elif test:
            sample = action_mean.detach()
        action = torch.tanh(sample)
        logprob = probs.log_prob(sample) - torch.log(1. - action.pow(2) + 1e-8)
        # entropy = (-logprob * logprob.exp()).sum(-1)
        entropy = probs.entropy().sum(-1)
        return action, sample, logprob.sum(-1), entropy
    
    def save(self, path: str):
        torch.save(
            {"actor_mean": self.actor_mean.state_dict(),
             "actor_logstd": self.actor_logstd}, os.path.join(path, "actor.pth"))

    def load(self, path: str):
        actor = torch.load(os.path.join(path, "actor.pth"), weights_only=True)
        self.actor_mean.load_state_dict(actor["actor_mean"])
        self.actor_logstd.data.copy_(actor["actor_logstd"].to(self.actor_logstd.device))

    def reset(self, indices: Tensor):
        self.actor_mean.reset(indices)
    
    def detach(self):
        self.actor_mean.detach()


class CriticVRNN(nn.Module, RNNBasedAgent):
    def __init__(
        self,
        state_dim: Union[int, Tuple[int, Tuple[int, int]]],
        hidden_dim: Union[int, List[int]],
        rnn_hidden_dim: int,
        n_layers: int
    ):
        super().__init__()
        self.critic = RNN(state_dim, rnn_hidden_dim, hidden_dim, 1, n_layers)
    
    def forward(self, obs: Union[Tensor, TensorDict], hidden: Optional[Tensor] = None) -> Tensor:
        return self.critic(obs, hidden).squeeze(-1)
    
    def save(self, path: str):
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
    
    def load(self, path: str):
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth"), weights_only=True))

    def reset(self, indices: Tensor):
        self.critic.reset(indices)
    
    def detach(self):
        self.critic.detach()


class CriticQRNN(nn.Module, RNNBasedAgent):
    def __init__(
        self,
        state_dim: Union[int, Tuple[int, Tuple[int, int]]],
        hidden_dim: Union[int, List[int]],
        action_dim: int,
        rnn_hidden_dim: int,
        n_layers: int
    ):
        super().__init__()
        if not isinstance(state_dim, int):
            input_dim = (state_dim[0] + action_dim, state_dim[1])
        self.critic = RNN(input_dim, rnn_hidden_dim, hidden_dim, 1, n_layers)
    
    def forward(self, obs: Union[Tensor, TensorDict], action: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        return self.critic(obs, hidden, action=action).squeeze(-1)
    
    def save(self, path: str):
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
    
    def load(self, path: str):
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth"), weights_only=True))

    def reset(self, indices: Tensor):
        self.critic.reset(indices)
    
    def detach(self):
        self.critic.detach()
        

class StochasticActorCriticVRNN(nn.Module, RNNBasedAgent):
    def __init__(
        self,
        state_dim: Union[int, Tuple[int, Tuple[int, int]]],
        hidden_dim: Union[int, List[int]],
        action_dim: int,
        rnn_hidden_dim: int,
        n_layers: int
    ):
        super().__init__()
        self.critic = CriticVRNN(state_dim, hidden_dim, rnn_hidden_dim, n_layers)
        self.actor = StochasticActorRNN(state_dim, hidden_dim, action_dim, rnn_hidden_dim, n_layers)

    def get_value(self, obs: Union[Tensor, TensorDict], hidden: Optional[Tensor] = None) -> Tensor:
        return self.critic(obs, hidden)

    def get_action(self, obs, sample=None, test=False, hidden=None):
        # type: (Union[Tensor, TensorDict], Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        return self.actor(obs, sample, test, hidden)

    def get_action_and_value(self, obs, sample=None, test=False):
        # type: (Union[Tensor, TensorDict], Optional[Tensor], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        return *self.get_action(obs, sample=sample, test=test), self.get_value(obs)
    
    def save(self, path: str):
        self.actor.save(path)
        self.critic.save(path)
    
    def load(self, path: str):
        self.actor.load(path)
        self.critic.load(path)

    def reset(self, indices: Tensor):
        self.actor.reset(indices)
        self.critic.reset(indices)
    
    def detach(self):
        self.actor.detach()
        self.critic.detach()


class StochasticActorCriticQRNN(nn.Module, RNNBasedAgent):
    def __init__(
        self,
        state_dim: Union[int, Tuple[int, Tuple[int, int]]],
        hidden_dim: Union[int, List[int]],
        action_dim: int,
        rnn_hidden_dim: int,
        n_layers: int
    ):
        super().__init__()
        self.critic = CriticQRNN(state_dim, hidden_dim, action_dim, rnn_hidden_dim, n_layers)
        self.actor = StochasticActorRNN(state_dim, hidden_dim, action_dim, rnn_hidden_dim, n_layers)

    def get_value(self, obs: Union[Tensor, TensorDict], action: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        return self.critic(obs, action, hidden)

    def get_action(self, obs, sample=None, test=False, hidden=None):
        # type: (Union[Tensor, TensorDict], Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        return self.actor(obs, sample, test, hidden)

    def get_action_and_value(self, obs, sample=None, test=False):
        # type: (Union[Tensor, TensorDict], Optional[Tensor], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        action, sample, logprob, entropy = self.get_action(obs, sample=sample, test=test)
        value = self.get_value(obs, action)
        return action, sample, logprob, entropy, value
    
    def save(self, path: str):
        self.actor.save(path)
        self.critic.save(path)
    
    def load(self, path: str):
        self.actor.load(path)
        self.critic.load(path)

    def reset(self, indices: Tensor):
        self.actor.reset(indices)
        self.critic.reset(indices)
    
    def detach(self):
        self.actor.detach()
        self.critic.detach()