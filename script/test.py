import random
import sys
sys.path.append('..')
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path=str(Path(__file__).parent.parent.joinpath("cfg")), config_name="config_test", version_base="1.3")
def main(cfg: DictConfig):
    
    import torch
    import numpy as np

    from diffaero.env import build_env
    from diffaero.algo import build_agent
    from diffaero.utils.logger import Logger
    from diffaero.utils.runner import TestRunner

    logger = Logger(cfg, run_name=cfg.runname)

    device_idx = cfg.device
    device = f"cuda:{device_idx}" if torch.cuda.is_available() and device_idx != -1 else "cpu"
    Logger.info(f"Using device {device}.")
    device = torch.device(device)
    
    if cfg.seed != -1:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = cfg.torch_deterministic
    
    ckpt_path = Path(cfg.checkpoint).resolve()
    cfg_path = ckpt_path.parent.joinpath(".hydra", "config.yaml")
    ckpt_cfg = OmegaConf.load(cfg_path)
    cfg.algo = ckpt_cfg.algo
    if cfg.algo.name != 'world':
        cfg.network = ckpt_cfg.network
    else:
        cfg.algo.common.is_test = True
    if cfg.use_training_cfg:
        cfg.dynamics = ckpt_cfg.dynamics
        cfg.sensor = ckpt_cfg.sensor
        ckpt_cfg.env.max_target_vel = cfg.env.max_target_vel
        ckpt_cfg.env.min_target_vel = cfg.env.min_target_vel
        ckpt_cfg.env.n_envs = cfg.env.n_envs
        cfg.env = ckpt_cfg.env
    
    env = build_env(cfg.env, device=device)
    
    agent = build_agent(cfg.algo, env, device)
    agent.load(ckpt_path)
    
    runner = TestRunner(cfg, logger, env, agent)
    
    try:
        runner.run()
    except KeyboardInterrupt:
        Logger.warning("Interrupted.")
    
    success_rate = runner.close()
    
    return success_rate

if __name__ == "__main__":
    main()