import random
import sys
sys.path.append('..')
from pathlib import Path

import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import matplotlib.pyplot as plt

@hydra.main(config_path=str(Path(__file__).parent.parent.joinpath("cfg")), config_name="config_test", version_base="1.3")
def main(cfg: DictConfig):
    
    import torch
    import numpy as np

    from quaddif.env import build_env
    from quaddif.algo import build_agent
    from quaddif.utils.math import quaternion_to_euler

    device_idx = cfg.device
    device = f"cuda:{device_idx}" if torch.cuda.is_available() and device_idx != -1 else "cpu"
    print(f"Using device {device}.")
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
    cfg.network = ckpt_cfg.network
    
    env = build_env(cfg.env, device=device)
    
    agent = build_agent(cfg.algo, env, device)
    agent.load(ckpt_path)
    
    pbar = tqdm(range(cfg.n_steps))
    time = torch.arange(cfg.n_steps, device=device) * env.dt
    pos = torch.zeros(cfg.env.n_envs, cfg.n_steps, 3, device=device)
    vel = torch.zeros(cfg.env.n_envs, cfg.n_steps, 3, device=device)
    acc = torch.zeros(cfg.env.n_envs, cfg.n_steps, 3, device=device)
    rpy = torch.zeros(cfg.env.n_envs, cfg.n_steps, 3, device=device) 
    try:
        with torch.no_grad():
            obs = env.reset()
            pbar = tqdm(range(cfg.n_steps))
            for i in pbar:
                t1 = pbar._time()
                env.detach()
                # log env state into buffer
                pos[:, i] = env.p
                vel[:, i] = env.v
                acc[:, i] = env.a
                rpy[:, i] = quaternion_to_euler(env.q) * 180 / torch.pi
                
                action, policy_info = agent.act(obs, test=True)
                obs, loss, terminated, env_info = env.step(env.rescale_action(action))
                if cfg.algo.name != 'world' and hasattr(agent, "reset"):
                    agent.reset(env_info["reset"])

    except KeyboardInterrupt:
        print("Interrupted.")
    
    env_index = 0
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    df = pd.DataFrame({
        "time": time[:pbar.n].cpu().numpy(),
        "pos_x": pos[env_index, :pbar.n, 0].cpu().numpy(),
        "pos_y": pos[env_index, :pbar.n, 1].cpu().numpy(),
        "pos_z": pos[env_index, :pbar.n, 2].cpu().numpy(),
        "vel_x": vel[env_index, :pbar.n, 0].cpu().numpy(),
        "vel_y": vel[env_index, :pbar.n, 1].cpu().numpy(),
        "vel_z": vel[env_index, :pbar.n, 2].cpu().numpy(),
        "acc_x": acc[env_index, :pbar.n, 0].cpu().numpy(),
        "acc_y": acc[env_index, :pbar.n, 1].cpu().numpy(),
        "acc_z": acc[env_index, :pbar.n, 2].cpu().numpy(),
        "roll": rpy[env_index, :pbar.n, 0].cpu().numpy(),
        "pitch": rpy[env_index, :pbar.n, 1].cpu().numpy(),
        "yaw": rpy[env_index, :pbar.n, 2].cpu().numpy()
    })
    df.to_csv(Path(output_dir).joinpath(f"env_{env_index}_traj.csv"), index=False)
    
    for (data, name) in zip(
        [pos, vel, acc, rpy],
        ["pos", "vel", "acc", "euler angles"]
    ):
        fig = plt.figure(dpi=200)
        plt.suptitle(name)
        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.plot(time[:pbar.n].cpu().numpy(), data[env_index, :pbar.n, i].cpu().numpy())
            plt.xlabel("time(s)")
            if name == "euler angles":
                plt.ylabel(["roll", "pitch", "yaw"][i] + "(deg)")
            else:
                plt.ylabel("xyz"[i]+" axis")
            plt.grid()
        plt.tight_layout()
        plt.savefig(Path(output_dir).joinpath(f"env_{env_index}_{name}.png"))
    
if __name__ == "__main__":
    main()