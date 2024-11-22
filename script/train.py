import os
import random
import sys
sys.path.append('..')

import isaacgym
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
from line_profiler import LineProfiler
import cv2

from quaddif.env import PositionControl, ObstacleAvoidance
from quaddif.algo import SHAC, APG_stochastic, APG, PPO, SHAC
from quaddif.utils.env import RecordEpisodeStatistics
from quaddif.utils.device import idle_device
from quaddif.utils.logger import Logger
from quaddif.utils.assets import ObstacleManager

def on_step_cb(state, action, policy_info, env_info):
    # type: (torch.Tensor, torch.Tensor, dict, dict[str, torch.Tensor]) -> None
    if "camera" in env_info.keys():
        N, C = 64, 1
        H, W = env_info["camera"].shape[-2:]
        NH = NW = int(N**0.5)
        scale = 4
        disp_image = env_info["camera"][:N].reshape(NH, NW, C, H, W).permute(2, 0, 3, 1, 4).reshape(C, NH*H, NW*W).cpu().numpy().transpose(1, 2, 0)
        disp_image = cv2.normalize(disp_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_image = cv2.resize(cv2.cvtColor(disp_image, cv2.COLOR_GRAY2BGR), (int(NH*H*scale), int(NW*W*scale)), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('image', disp_image)
        cv2.waitKey(1)

profiler = LineProfiler()
profiler.add_function(SHAC.learn)
profiler.add_function(APG_stochastic.learn)
profiler.add_function(APG.learn)
profiler.add_function(PPO.learn)
profiler.add_function(SHAC.learn)
profiler.add_function(ObstacleAvoidance.step)
profiler.add_function(ObstacleAvoidance.state)
profiler.add_function(ObstacleAvoidance.loss_fn)
profiler.add_function(ObstacleAvoidance.reset_idx)
profiler.add_function(ObstacleAvoidance.render_camera)
profiler.add_function(ObstacleManager.randomize_asset_pose)

@hydra.main(config_path="../cfg", config_name="config")
@profiler
def main(cfg: DictConfig):
    device_idx = f"{idle_device()}" if cfg.device is None else f"{cfg.device}"
    print("Using device", device_idx)
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    
    if cfg.seed != -1:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = cfg.torch_deterministic
    
    ENV_CLASS = {
        "position_control": PositionControl,
        "obstacle_avoidance": ObstacleAvoidance
    }[cfg.env.name]
    env = RecordEpisodeStatistics(ENV_CLASS(cfg.env, cfg.dynamics, device=device))
    
    AGENT_CLASS = {
        "ppo": PPO,
        "shac": SHAC,
        "apg": APG,
        "apg_sto": APG_stochastic
    }[cfg.algo.name]
    agent = AGENT_CLASS.build(cfg, env, device)
    
    logger = Logger(cfg)
    # AGENT_CLASS.learn(cfg, agent, env, logger, on_step_cb=on_step_cb)
    AGENT_CLASS.learn(cfg, agent, env, logger)
    
    if env.renderer is not None:
        env.renderer.close()
    global logdir
    logdir = logger.logdir

if __name__ == "__main__":
    main()
    with open(os.path.join(logdir, "runtime_profile.txt"), "w", encoding="utf-8") as f:
        profiler.print_stats(stream=f, output_unit=1e-3)