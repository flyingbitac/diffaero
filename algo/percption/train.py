from omegaconf import DictConfig    

import hydra
import torch

from world.backbone import WorldModel

@hydra.main(config_path='./config', config_name='config')
def main(cfg:DictConfig):
    wm = WorldModel(cfg)
    obs = torch.randn(16,64,3,64,64)
    action = torch.randn(16,64,3)
    rewards = torch.randn(16,64)
    terminals = torch.randn(16,64)
    loss, metrics =wm.compute_loss(obs, action, rewards, terminals)

main()