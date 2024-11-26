from typing import *
import os
import random
import sys
sys.path.append('..')

import isaacgym
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import cv2

from quaddif.env import PositionControl, ObstacleAvoidance
from quaddif.algo import SHAC, APG_stochastic, APG, PPO
from quaddif.utils.env import RecordEpisodeStatistics
from quaddif.utils.device import idle_device

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

@torch.no_grad()
def test(
    cfg: DictConfig,
    agent: Union[SHAC, APG, APG_stochastic, PPO],
    env: Union[PositionControl, ObstacleAvoidance],
    on_step_cb: Optional[Callable] = None
):
    state = env.reset()
    pbar = tqdm(range(50000))
    for i in pbar:
        t1 = pbar._time()
        env.detach()
        action, policy_info = agent.act(state, test=True)
        state, loss, terminated, env_info = env.step(action)
        l_episode = env_info["stats"]["l"].float().mean().item()
        success_rate = env_info['stats']['success_rate']
        pbar.set_postfix({
            "loss": f"{env_info['loss_components']['total_loss']:.3f}",
            "l_episode": f"{l_episode:.1f}",
            "success_rate": f"{success_rate:.2f}",
            "fps": f"{cfg.env.n_envs/(pbar._time()-t1):,.0f}"})
        if on_step_cb is not None:
            on_step_cb(
                state=state,
                action=action,
                policy_info=policy_info,
                env_info=env_info)

@hydra.main(config_path="../cfg", config_name="config")
def main(cfg: DictConfig):
    device_idx = f"{idle_device()}" if cfg.device is None else f"{cfg.device}"
    print("Using device", device_idx)
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    
    assert cfg.checkpoint is not None
    cfg_path = os.path.join(cfg.checkpoint, ".hydra", "config.yaml")
    print(cfg.checkpoint, cfg_path)
    ckpt_cfg = OmegaConf.load(cfg_path)
    cfg.algo = ckpt_cfg.algo
    cfg.dynamics = ckpt_cfg.dynamics
    
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
    agent.load(os.path.join(cfg.checkpoint, "checkpoints"))
    
    # test(cfg, agent, env, on_step_cb=on_step_cb)
    test(cfg, agent, env)
    
    if env.renderer is not None:
        env.renderer.close()

if __name__ == "__main__":
    main()