from omegaconf import DictConfig    

import hydra
import torch

from world.backbone import WorldModel

@hydra.main(config_path='./config', config_name='config')
def main(cfg:DictConfig):
    wm = WorldModel(cfg)
    # cfg.encoder.use_state = False
    grid = torch.randn(16, 64, 4000)
    cfg.encoder.use_image = True
    obs = torch.randn(16,64,1,16,9)
    state = torch.randn(16, 64, 10)
    action = torch.randn(16,64,3)
    rewards = torch.randn(16,64)
    terminals = torch.randn(16,64)
    loss, metrics = wm.update(obs, state, action, rewards, terminals, grid)
    print(f"loss {loss}")
    print(f"metrics {metrics}")
   
if __name__ == '__main__': 
    main()