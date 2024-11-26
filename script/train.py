from typing import *
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
from tqdm import tqdm
import cv2

from quaddif.env import PositionControl, ObstacleAvoidance
from quaddif.algo import SHAC, APG_stochastic, APG, PPO
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
profiler.add_function(SHAC.step)
profiler.add_function(APG_stochastic.step)
profiler.add_function(APG.step)
profiler.add_function(PPO.step)
profiler.add_function(SHAC.step)
profiler.add_function(ObstacleAvoidance.step)
profiler.add_function(ObstacleAvoidance.state)
profiler.add_function(ObstacleAvoidance.loss_fn)
profiler.add_function(ObstacleAvoidance.reset_idx)
profiler.add_function(ObstacleAvoidance.render_camera)
profiler.add_function(ObstacleManager.randomize_asset_pose)

@profiler
def learn(
    cfg: DictConfig,
    agent: Union[SHAC, APG, APG_stochastic, PPO],
    env: Union[PositionControl, ObstacleAvoidance],
    logger: Logger,
    on_step_cb: Optional[Callable] = None,
    on_update_cb: Optional[Callable] = None
):
    state = env.reset()
    pbar = tqdm(range(cfg.n_updates))
    for i in pbar:
        t1 = pbar._time()
        env.detach()
        policy_info, env_info, losses, grad_norms = agent.step(cfg, env, state, on_step_cb)
        l_episode = env_info["stats"]["l"].float().mean().item()
        success_rate = env_info['stats']['success_rate']
        pbar.set_postfix({
            "param_norm": f"{grad_norms['actor_grad_norm']:.3f}",
            "loss": f"{env_info['loss_components']['total_loss']:.3f}",
            "l_episode": f"{l_episode:.1f}",
            "success_rate": f"{success_rate:.2f}",
            "fps": f"{(cfg.l_rollout*cfg.env.n_envs)/(pbar._time()-t1):,.0f}"})
        log_info = {
            "env_loss": env_info["loss_components"],
            "agent_loss": losses,
            "agent_grad_norm": grad_norms,
            "metrics": {"l_episode": l_episode, "success_rate": success_rate}
        }
        if "value" in policy_info.keys():
            log_info["value"] = policy_info["value"].mean().item()
        logger.log_scalars(log_info, i+1)

        if on_update_cb is not None:
            on_update_cb(log_info=log_info)


@hydra.main(config_path="../cfg", config_name="config")
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
    # learn(cfg, agent, env, logger, on_step_cb=on_step_cb)
    learn(cfg, agent, env, logger)
    
    if env.renderer is not None:
        env.renderer.close()
    global logdir
    logdir = logger.logdir

if __name__ == "__main__":
    main()
    with open(os.path.join(logdir, "runtime_profile.txt"), "w", encoding="utf-8") as f:
        profiler.print_stats(stream=f, output_unit=1e-3)