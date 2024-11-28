from typing import Tuple, Dict, Union, Optional, List

import torch
import torch.nn as nn

def num_params(model: nn.Module) -> int:
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
    mlp.append(layer_init(nn.Linear(dims[-2], dims[-1]), std=0.01))
    if output_act is not None:
        mlp.append(output_act)
    return nn.Sequential(*mlp)