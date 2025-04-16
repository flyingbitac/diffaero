from omegaconf import DictConfig    

import hydra
import torch

from world.backbone import WorldModel

@hydra.main(config_path='./config', config_name='config')
def main(cfg:DictConfig):
    wm = WorldModel(cfg)
    # cfg.encoder.use_state = False
    cfg.encoder.use_image = False
    obs = torch.randn(16,64,3,64,64)
    state = torch.randn(16, 64, 10)
    action = torch.randn(16,64,3)
    rewards = torch.randn(16,64)
    terminals = torch.randn(16,64)
    loss, metrics = wm.compute_loss(obs, state, action, rewards, terminals)
    print(f"loss {loss}")
   
if __name__ == '__main__': 
    main()