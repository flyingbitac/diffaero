from typing import Tuple, Dict, Union, Optional, List
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

def num_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def layer_init(layer, std=2.**0.5, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def weight_init(m):
	"""Custom weight initialization for TD-MPC2."""
	if isinstance(m, nn.Linear):
		nn.init.trunc_normal_(m.weight, std=0.02)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.Embedding):
		nn.init.uniform_(m.weight, -0.02, 0.02)
	elif isinstance(m, nn.ParameterList):
		for i,p in enumerate(m):
			if p.dim() == 3: # Linear
				nn.init.trunc_normal_(p, std=0.02) # Weight
				nn.init.constant_(m[i+1], 0) # Bias

def zero_(params):
	"""Initialize parameters to zero."""
	for p in params:
		p.data.fill_(0)

class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation.
    """
    def __init__(self, *args, act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
    def forward(self, x):
        x = super().forward(x)
        return self.act(self.ln(x))
    def __repr__(self):
        return f"NormedLinear(in_features={self.in_features}, "\
            f"out_features={self.out_features}, "\
            f"bias={self.bias is not None}, "\
            f"act={self.act.__class__.__name__})"

def mlp(
    in_dim: int,
    mlp_dims: Union[int, List[int]],
    out_dim: int,
    hidden_act: nn.Module = nn.Mish(inplace=True),
    output_act: Optional[nn.Module] = None):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(layer_init(NormedLinear(dims[i], dims[i+1], act=hidden_act)))
    if output_act is not None:
        mlp.append(layer_init(NormedLinear(dims[-2], dims[-1], act=output_act), std=0.01))
    else:
        mlp.append(layer_init(nn.Linear(dims[-2], dims[-1]), std=0.01))
    return nn.Sequential(*mlp)

class StochasticActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: Union[int, List[int]],
        action_dim: int
    ):
        super().__init__()
        self.critic = mlp(state_dim, hidden_dim, 1, hidden_act=nn.ELU())
        self.actor_mean = mlp(state_dim, hidden_dim, action_dim, hidden_act=nn.ELU())
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)

    def get_action(self, obs, sample=None, test=False):
        action_mean = self.actor_mean(obs)
        LOG_STD_MAX = 2
        LOG_STD_MIN = -5
        action_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            torch.tanh(self.actor_logstd) + 1)  # From SpinUp / Denis Yarats
        action_std = torch.exp(action_logstd).expand_as(action_mean)
        probs = torch.distributions.Normal(action_mean, action_std)
        if sample is None and not test:
            sample = torch.randn_like(action_mean) * action_std + action_mean
            # sample = probs.sample()
        elif test:
            sample = action_mean.detach()
        action = torch.tanh(sample)
        logprob = probs.log_prob(sample) - torch.log(1. - torch.tanh(sample).pow(2) + 1e-8)
        # entropy = (-logprob * logprob.exp()).sum(-1)
        entropy = probs.entropy().sum(-1)
        return action, sample, logprob.sum(-1), entropy

    def get_action_and_value(self, obs, sample=None, test=False):
        return *self.get_action(obs, sample, test), self.get_value(obs)