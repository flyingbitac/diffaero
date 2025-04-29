import sys
import pathlib
folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
sys.path.insert(2, str(folder.parent.parent.parent))

import torch
import torch.nn as nn

from sub_models.world_models import WorldModel
from dreamerv3.models.agent import ActorCriticAgent, ActorCriticConfig

if __name__ == "__main__":
    pass
    model = WorldModel(
        in_channels=3,
        action_dim=4,
        transformer_max_length=100,
        transformer_hidden_dim=512,
        transformer_num_layers=2,
        transformer_num_heads=4
    )
    cfg = ActorCriticConfig(1536, 2, 512, 4, 0.99, 0.95, 0.01, torch.device('cpu'))
    agent = ActorCriticAgent(cfg, None)
    perception = torch.randn(16, 64, 3, 64, 64)
    state = torch.randn(16, 64, 10)
    action = torch.randn(16, 64, 4)
    reward = torch.randn(16, 64, )
    done = torch.randn(16, 64, )
    model.update(perception, state, action, reward, done)