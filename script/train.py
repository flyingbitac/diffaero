import random
from time import sleep
import sys
sys.path.append('..')

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../cfg", config_name="config_train", version_base="1.3")
def main(cfg: DictConfig):
    
    import torch
    import numpy as np
    
    from quaddif.env import build_env
    from quaddif.algo import build_agent
    from quaddif.utils.device import get_idle_device
    from quaddif.utils.logger import RecordEpisodeStatistics, Logger
    from quaddif.utils.runner import TrainRunner

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

    env = RecordEpisodeStatistics(build_env(cfg.env, device=device))
    
    agent = build_agent(cfg.algo, env, device)
    
    runname = f"__{cfg.runname}" if len(cfg.runname) > 0 else ""
    logger = Logger(cfg, run_name=f"__train{runname}")
    
    runner = TrainRunner(cfg, logger, env, agent)
    
    try:
        runner.run()
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        max_success_rate = runner.close()
    
    return max_success_rate

if __name__ == "__main__":
    main()