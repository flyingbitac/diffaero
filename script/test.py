import os
import random
import sys
sys.path.append('..')

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../cfg", config_name="config_test", version_base="1.3")
def main(cfg: DictConfig):
    
    import torch
    import numpy as np
    from tqdm import tqdm

    from quaddif import QUADDIF_ROOT_DIR
    from quaddif.env import build_env
    from quaddif.algo import build_agent
    from quaddif.utils.device import get_idle_device
    from quaddif.utils.logger import RecordEpisodeStatistics, Logger
    from quaddif.utils.runner import TestRunner

    runname = f"__{cfg.runname}" if len(cfg.runname) > 0 else ""
    logger = Logger(cfg, run_name=f"__test{runname}")

    device_idx = f"{get_idle_device()}" if cfg.device is None else f"{cfg.device}"
    device = f"cuda:{device_idx}" if torch.cuda.is_available() and device_idx != "-1" else "cpu"
    print(f"Using device {device}.")
    device = torch.device(device)
    
    if cfg.seed != -1:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = cfg.torch_deterministic
    
    if cfg.checkpoint is None:
        cfg.checkpoint = os.path.join(QUADDIF_ROOT_DIR, "outputs", "latest", "checkpoints")
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
        cfg.env = ckpt_cfg.env
    
    env = RecordEpisodeStatistics(build_env(cfg.env, device=device))
    
    agent = build_agent(cfg.algo, env, device)
    agent.load(cfg.checkpoint)
    
    runner = TestRunner(cfg, logger, env, agent)
    
    try:
        runner.run()
    except KeyboardInterrupt:
        tqdm.write("Interrupted.")
    finally:
        runner.close()

if __name__ == "__main__":
    main()