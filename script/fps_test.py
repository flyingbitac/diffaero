import random
import sys
sys.path.append('..')
from pathlib import Path

import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path=str(Path(__file__).parent.parent.joinpath("cfg")), config_name="config_test", version_base="1.3")
def main(cfg: DictConfig):
    
    import torch
    import numpy as np

    from quaddif.env import build_env
    from gpustat import new_query as gpu_query

    device_idx = cfg.device
    device = f"cuda:{device_idx}" if torch.cuda.is_available() and device_idx != -1 else "cpu"
    print(f"Using device {device}.")
    device = torch.device(device)
    
    if cfg.seed != -1:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = cfg.torch_deterministic
    
    env = build_env(cfg.env, device=device)
    
    pbar = tqdm(range(cfg.n_steps))
    try:
        with torch.no_grad():
            obs = env.reset()
            start = pbar._time()
            for i in pbar:
                action = torch.zeros(env.n_envs, env.action_dim, device=device)
                env.step(action)
                pbar.set_postfix({"FPS": f"{int(cfg.n_envs * pbar.n / (pbar._time() - start)):,d}"})
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        end = pbar._time()
        fps = int(cfg.n_envs * pbar.n / (end - start))
        processes = []
        for gpu in gpu_query():
            processes.extend(gpu.processes)
        for process in processes:
            if process["command"].startswith("python") and "script/fps_test.py" in process["full_command"]:
                vram = process["gpu_memory_usage"]
                break
        print("Overrides: ", " ".join(hydra.core.hydra_config.HydraConfig.get().overrides.task))
        print(f"GPU Memory Usage: {vram} MiB, FPS: {fps:,d}")

if __name__ == "__main__":
    main()