from typing import *
import sys
sys.path.append('..')

import isaacgym
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from quaddif.env import ENV_ALIAS
from quaddif.algo import AGENT_ALIAS
from quaddif.utils.exporter import PolicyExporter
from quaddif.utils.logger import RecordEpisodeStatistics, Logger
from quaddif.utils.device import idle_device

@hydra.main(config_path="../cfg", config_name="config")
def main(cfg: DictConfig):
    device_idx = f"{idle_device()}" if cfg.device is None else f"{cfg.device}"
    device = f"cuda:{device_idx}" if torch.cuda.is_available() and device_idx != "-1" else "cpu"
    print(f"Using device {device}.")
    device = torch.device(device)
    
    env_class = ENV_ALIAS[cfg.env.name]
    env = RecordEpisodeStatistics(env_class(cfg.env, device=device))
    
    agent_class = AGENT_ALIAS[cfg.algo.name]
    agent = agent_class.build(cfg, env, device)
    agent.load(cfg.checkpoint)
    
    PolicyExporter(agent.policy_net).export(path=cfg.checkpoint, verbose=True, export_pnnx=False)
    
    if env.renderer is not None:
        env.renderer.close()

if __name__ == "__main__":
    main()