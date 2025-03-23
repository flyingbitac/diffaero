import os
from time import sleep
import random
import sys
sys.path.append('..')

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../cfg", config_name="config_test", version_base="1.3")
def main(cfg: DictConfig):
    
    import torch
    import numpy as np

    from quaddif.env import build_env
    from quaddif.algo import build_agent
    from quaddif.utils.device import get_idle_device
    from quaddif.utils.logger import RecordEpisodeStatistics, Logger
    from quaddif.utils.runner import TestRunner

    if cfg.device is None and cfg.n_jobs > 1:
        sleep(random.random() * 3)
    device_idx = get_idle_device() if cfg.device is None else cfg.device
    device = f"cuda:{device_idx}" if torch.cuda.is_available() and device_idx != "-1" else "cpu"
    print(f"Using device {device}.")
    device = torch.device(device)
    
    if cfg.seed != -1:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = cfg.torch_deterministic
    
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(cfg.checkpoint)), ".hydra", "config.yaml")
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

    runname = f"__{cfg.runname}" if len(cfg.runname) > 0 else ""
    logger = Logger(cfg, run_name=runname)
    
    env = RecordEpisodeStatistics(build_env(cfg.env, device=device))
    
    agent = build_agent(cfg.algo, env, device)
    agent.load(cfg.checkpoint)
    
    runner = TestRunner(cfg, logger, env, agent)
    
    try:
        runner.run()
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        success_rate = runner.close()
    
    return success_rate

if __name__ == "__main__":
    main()