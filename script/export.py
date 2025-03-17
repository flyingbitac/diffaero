from typing import *
import os
import sys
sys.path.append('..')

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from quaddif.env import build_env
from quaddif.algo import build_agent
from quaddif.utils.exporter import PolicyExporter

@hydra.main(config_path="../cfg", config_name="test_config")
def main(cfg: DictConfig):
    print(f"Using device cpu.")
    device = torch.device("cpu")
    
    assert cfg.checkpoint is not None
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(cfg.checkpoint)), ".hydra", "config.yaml")
    ckpt_cfg = OmegaConf.load(cfg_path)
    cfg.algo = ckpt_cfg.algo
    # cfg.dynamics = ckpt_cfg.dynamics
    if cfg.algo.name != 'world':
        cfg.network = ckpt_cfg.network
    cfg.env.render.headless = True
    
    env = build_env(cfg.env, device=device)
    agent = build_agent(cfg.algo, env, device)
    agent.load(cfg.checkpoint)
    PolicyExporter(agent.policy_net).export(path=cfg.checkpoint, verbose=True, export_onnx=False, export_pnnx=False)

if __name__ == "__main__":
    main()