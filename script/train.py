import random
import sys
sys.path.append('..')
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

def allocate_device(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    multirun = hydra_cfg.mode == hydra.types.RunMode.MULTIRUN
    use_multiple_devices = isinstance(cfg.device, str) and len(cfg.device) > 0
    if multirun and use_multiple_devices:
        available_devices = list(map(int, list(cfg.device)))
        n_devices = len(available_devices)
        job_id = hydra_cfg.job.num
        job_device = available_devices[job_id % n_devices]
        cfg.device = 0
    else:
        job_device = int(cfg.device) if isinstance(cfg.device, int) else 0
    return job_device

@hydra.main(config_path=str(Path(__file__).parent.parent.joinpath("cfg")), config_name="config_train", version_base="1.3")
def main(cfg: DictConfig):
    
    job_device = allocate_device(cfg)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(job_device)
    
    import torch
    import numpy as np
    
    from quaddif.env import build_env
    from quaddif.algo import build_agent
    from quaddif.utils.logger import Logger
    from quaddif.utils.runner import TrainRunner
    
    device = f"cuda:{cfg.device}" if torch.cuda.is_available() and cfg.device != -1 else "cpu"
    print(f"Using device {job_device}.")
    device = torch.device(device)
    
    if cfg.seed != -1:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = cfg.torch_deterministic
    
    if cfg.checkpoint is not None and len(cfg.checkpoint) > 0:
        ckpt_path = Path(cfg.checkpoint).resolve()
        cfg_path = ckpt_path.parent.joinpath(".hydra", "config.yaml")
        ckpt_cfg = OmegaConf.load(cfg_path)
        cfg.sensor = ckpt_cfg.sensor
        train_from_checkpoint = True
    else:
        ckpt_path = ''
        train_from_checkpoint = False

    env = build_env(cfg.env, device=device)
    
    agent = build_agent(cfg.algo, env, device)
    if train_from_checkpoint:
        agent.load(ckpt_path)
    
    runname = f"__{cfg.runname}" if len(cfg.runname) > 0 else ""
    logger = Logger(cfg, run_name=runname)
    
    runner = TrainRunner(cfg, logger, env, agent)
    
    try:
        runner.run()
    except KeyboardInterrupt:
        print("Interrupted.")
    
    max_success_rate = runner.close()
    
    return max_success_rate

if __name__ == "__main__":
    main()