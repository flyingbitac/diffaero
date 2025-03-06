from typing import *
import os
import random
import sys
sys.path.append('..')

import torch
import torchvision
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import cv2
import imageio

from quaddif.env import ENV_ALIAS
from quaddif.algo import AGENT_ALIAS
from quaddif.utils.logger import RecordEpisodeStatistics, Logger
from quaddif.utils.device import get_idle_device
from quaddif.algo.dreamerv3.world import WorldExporter

def display_image(state, action, policy_info, env_info):
    # type: (torch.Tensor, torch.Tensor, dict, dict[str, torch.Tensor]) -> None
    if "sensor" in env_info.keys():
        N, C = 64, 1
        H, W = env_info["sensor"].shape[-2:]
        NH = NW = int(N**0.5)
        scale = 4
        disp_image = env_info["sensor"][:N].reshape(NH, NW, C, H, W).permute(2, 0, 3, 1, 4).reshape(C, NH*H, NW*W).cpu().numpy().transpose(1, 2, 0)
        disp_image = cv2.normalize(disp_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_image = cv2.resize(cv2.cvtColor(disp_image, cv2.COLOR_GRAY2BGR), (int(NW*W*scale), int(NH*H*scale)), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('image', disp_image)
        cv2.waitKey(1)

def save_video_mp4(cfg: DictConfig, video_array: np.ndarray, logger: Logger, name: str):
    # save the video using imageio
    path = os.path.join(logger.logdir, "video")
    if not os.path.exists(path):
        os.makedirs(path)
    with imageio.get_writer(os.path.join(path, name), fps=1/cfg.env.dt) as video:
        for frame_index in range(video_array.shape[0]):
            frame = video_array[frame_index]
            video.append_data(frame)

def save_video_tensorboard(cfg: DictConfig, video_array: np.ndarray, logger: Logger, tag: str, step: int):
    logger.log_video(tag, video_array, step=step, fps=1/cfg.env.dt)

@torch.no_grad()
def test(
    cfg: DictConfig,
    agent,
    env,
    logger: Logger,
    on_step_cb: Optional[Callable] = None
):
    if cfg.record_video:
        H_video, W_video = env.renderer.video_H, env.renderer.video_W
        H_depth, W_depth = cfg.sensor.height, cfg.sensor.width
        H_scale, W_scale = H_video / H_depth, W_video / W_depth
        H_depth = H_video if H_scale >= W_scale else int(H_depth * W_scale)
        W_depth = W_video if W_scale >= H_scale else int(W_depth * H_scale)
        H, W = H_video, W_video + W_depth
        video_array = np.empty((cfg.n_envs, env.max_steps, H, W, 3), dtype=np.uint8)
    
    state = env.reset()
    pbar = tqdm(range(10000))
    n_resets = 1
    n_survive = 0
    n_success = 0
    for i in pbar:
        t1 = pbar._time()
        env.detach()
        action, policy_info = agent.act(state, test=True)
        state, loss, terminated, env_info = env.step(action)
        if cfg.algo.name != 'world':
            agent.reset(env_info["reset"])
        l_episode = (env_info["stats"]["l"] - 1) * env.dt
        n_resets += env_info["reset"].sum().item()
        n_survive += env_info["truncated"].sum().item()
        n_success += env_info["success"].sum().item()
        pbar.set_postfix({
            "l_episode": f"{l_episode:.1f}",
            "survive_rate": f"{n_survive / n_resets:.2f}",
            "success_rate": f"{n_success / n_resets:.2f}",
            "fps": f"{int(cfg.env.n_envs/(pbar._time()-t1)):,d}"})
        
        log_info = {"metrics": {"success_rate": n_success / n_resets}}
        logger.log_scalars(log_info, i+1)
        
        if cfg.record_video:
            rgb_image: np.ndarray = env.renderer.render_fpp()
            index = (np.arange(env.n_envs), env.progress.cpu().numpy()-1)
            depth_image = torchvision.transforms.Resize(
                (H_depth, W_depth), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(env_info["sensor"])
            depth_image = (depth_image * 255).to(torch.uint8).unsqueeze(-1).expand(-1, -1, -1, 3).cpu().numpy()
            image = np.concatenate([rgb_image, depth_image], axis=-2)
            video_array[index] = image
            
            if env_info["reset"].sum().item() > env_info["success"].sum().item(): # some episodes failed
                failed = torch.logical_and(env_info["reset"], ~env_info["success"])
                idx = failed.nonzero().flatten()[0]
                video_length = env_info["l"][idx] - 1
                if cfg.video_saveas == "mp4":
                    save_video_mp4(cfg, video_array[idx, :video_length], logger, f"failed_{i+1}.mp4")
                elif cfg.video_saveas == "tensorboard":
                    save_video_tensorboard(cfg, video_array[idx.unsqueeze(0), :video_length], logger, "video/fail", i+1)
            if env_info["success"].sum().item() > 0: # some episodes succeeded
                idx = env_info["success"].nonzero().flatten()[0]
                video_length = min(env_info["l"][idx].item(), int(env_info["arrive_time"][idx].item()/env.dt) + 100) - 1
                if cfg.video_saveas == "mp4":
                    save_video_mp4(cfg, video_array[idx, :video_length], logger, f"success_{i+1}.mp4")
                elif cfg.video_saveas == "tensorboard":
                    save_video_tensorboard(cfg, video_array[idx.unsqueeze(0), :video_length], logger, "video/success", i+1)
            
        if on_step_cb is not None:
            on_step_cb(
                state=state,
                action=action,
                policy_info=policy_info,
                env_info=env_info)

@hydra.main(config_path="../cfg", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    device_idx = f"{get_idle_device()}" if cfg.device is None else f"{cfg.device}"
    device = f"cuda:{device_idx}" if torch.cuda.is_available() and device_idx != "-1" else "cpu"
    print(f"Using device {device}.")
    device = torch.device(device)
    
    assert cfg.checkpoint is not None
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(cfg.checkpoint)), ".hydra", "config.yaml")
    ckpt_cfg = OmegaConf.load(cfg_path)
    cfg.algo = ckpt_cfg.algo
    cfg.dynamics = ckpt_cfg.dynamics
    if cfg.algo.name != 'world':
        cfg.network = ckpt_cfg.network
    else:
        cfg.algo.common.is_test = True
    
    if cfg.seed != -1:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = cfg.torch_deterministic
    
    env_class = ENV_ALIAS[cfg.env.name]
    env = RecordEpisodeStatistics(env_class(cfg.env, device=device))
    
    agent_class = AGENT_ALIAS[cfg.algo.name]
    agent = agent_class.build(cfg, env, device)
    agent.load(cfg.checkpoint)
    
    
    logger = Logger(cfg, run_name=cfg.runname)
    # test(cfg, agent, env, logger, on_step_cb=display_image)
    
    export_path = os.path.join(logger.logdir,"exportckpt")
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    exporter = WorldExporter(agent)
    exporter.export(export_path)
    
    test(cfg, agent, env, logger)
    
    if env.renderer is not None:
        env.renderer.close()

if __name__ == "__main__":
    main()