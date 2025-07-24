import sys
sys.path.append('..')
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from diffaero.env import build_env
from diffaero.algo import build_agent

@hydra.main(config_path=str(Path(__file__).parent.parent.joinpath("cfg")), config_name="config_test", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Using device cpu.")
    device = torch.device("cpu")
    
    assert cfg.checkpoint is not None
    ckpt_path = Path(cfg.checkpoint).resolve()
    cfg_path = ckpt_path.parent.joinpath(".hydra", "config.yaml")
    ckpt_cfg = OmegaConf.load(cfg_path)
    cfg.algo = ckpt_cfg.algo
    # cfg.dynamics = ckpt_cfg.dynamics
    if cfg.algo.name != 'world':
        cfg.network = ckpt_cfg.network
    ckpt_cfg.env.render.headless = True
    cfg.dynamics = ckpt_cfg.dynamics
    cfg.sensor = ckpt_cfg.sensor
    cfg.env.n_envs = cfg.n_envs = 1
    ckpt_cfg.env.max_target_vel = cfg.env.max_target_vel
    ckpt_cfg.env.min_target_vel = cfg.env.min_target_vel
    ckpt_cfg.env.n_envs = cfg.env.n_envs
    cfg.env = ckpt_cfg.env
    
    env = build_env(cfg.env, device=device)
    agent = build_agent(cfg.algo, env, device)
    agent.load(ckpt_path)
    assert any(dict(cfg.export).values())
    agent.export(
        path=ckpt_path,
        export_cfg=cfg.export,
        verbose=True
    )

if __name__ == "__main__":
    main()