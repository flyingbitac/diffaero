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

@hydra.main(config_path="../cfg", config_name="config_test", version_base="1.3")
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
    ckpt_cfg.env.render.headless = True
    cfg.dynamics = ckpt_cfg.dynamics
    cfg.sensor = ckpt_cfg.sensor
    ckpt_cfg.env.max_target_vel = cfg.env.max_target_vel
    ckpt_cfg.env.min_target_vel = cfg.env.min_target_vel
    ckpt_cfg.env.n_envs = cfg.env.n_envs
    cfg.env = ckpt_cfg.env
    
    env = build_env(cfg.env, device=device)
    agent = build_agent(cfg.algo, env, device)
    agent.load(cfg.checkpoint)
    PolicyExporter(agent.policy_net).export(path=cfg.checkpoint, verbose=True, export_onnx=False, export_pnnx=False)

if __name__ == "__main__":
    main()