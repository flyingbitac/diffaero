from typing import *
import os

import torch
import torchvision
import numpy as np
from line_profiler import LineProfiler
from tqdm import tqdm
import cv2
import imageio
from omegaconf import DictConfig

from quaddif.utils.exporter import PolicyExporter
from quaddif.utils.logger import RecordEpisodeStatistics, Logger

def display_image(state, action, policy_info, env_info):
    # type: (torch.Tensor, torch.Tensor, dict, dict[str, torch.Tensor]) -> None
    if "sensor" in env_info.keys():
        N, C = min(64, state.size(0)), 1
        H, W = env_info["sensor"].shape[-2:]
        NH = NW = int(N**0.5)
        scale = 4
        disp_image = env_info["sensor"][:N].reshape(NH, NW, C, H, W).permute(2, 0, 3, 1, 4).reshape(C, NH*H, NW*W).cpu().numpy().transpose(1, 2, 0)
        disp_image = cv2.normalize(disp_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_image = cv2.resize(cv2.cvtColor(disp_image, cv2.COLOR_GRAY2BGR), (int(NW*W*scale), int(NH*H*scale)), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('image', disp_image)
        cv2.waitKey(1)

class TrainRunner:
    def __init__(self, cfg: DictConfig, logger: Logger, env: RecordEpisodeStatistics, agent):
        self.cfg = cfg
        self.logger = logger
        self.env = env
        self.agent = agent
        
        self.profiler = LineProfiler()
        if hasattr(self.env, "update_sensor_data"):
            self.profiler.add_function(self.env.update_sensor_data)
        if self.env.renderer is not None:
            self.profiler.add_function(self.env.renderer.render)
        self.profiler.add_function(self.env.env.step)
        self.profiler.add_function(self.env.get_observations)
        self.profiler.add_function(self.env.loss_fn)
        self.profiler.add_function(self.env.reset_idx)
        self.profiler.add_function(self.agent.step)
        self.run = self.profiler(self.run)
    
    def run(self):
        obs = self.env.reset()
        max_success_rate = 0
        pbar = tqdm(range(self.cfg.n_updates))
        on_step_cb = display_image if self.cfg.display_image else None
        for i in pbar:
            t1 = pbar._time()
            self.env.detach()
            obs, policy_info, env_info, losses, grad_norms = self.agent.step(self.cfg, self.env, obs, on_step_cb=on_step_cb)
            l_episode = (env_info["stats"]["l"] - 1) * self.env.dt
            success_rate = env_info["stats"]["success_rate"]
            survive_rate = env_info["stats"]["survive_rate"]
            arrive_time = env_info["stats"]["arrive_time"]
            if self.cfg.algo.name != 'world':
                pbar.set_postfix({
                    # "param_norm": f"{grad_norms['actor_grad_norm']:.3f}",
                    "loss": f"{env_info['loss_components']['total_loss']:.3f}",
                    "l_episode": f"{l_episode:.1f}",
                    "success_rate": f"{success_rate:.2f}",
                    "survive_rate": f"{survive_rate:.2f}",
                    "fps": f"{int(self.cfg.l_rollout*self.cfg.n_envs/(pbar._time()-t1)):,d}"})
            log_info = {
                "env_loss": env_info["loss_components"],
                "agent_loss": losses,
                "agent_grad_norm": grad_norms,
                "metrics": {
                    "l_episode": l_episode,
                    "success_rate": success_rate,
                    "survive_rate": survive_rate,
                    "arrive_time": arrive_time}
            }
            if "value" in policy_info.keys():
                log_info["value"] = policy_info["value"].mean().item()
            if "WorldModel/state_total_loss" in policy_info.keys():
                log_info.update(policy_info)
            if (i+1) % 10 == 0:
                self.logger.log_scalars(log_info, i+1)
            
            if success_rate >= max_success_rate:
                max_success_rate = success_rate
                self.agent.save(os.path.join(self.logger.logdir, "best"))
    
    def close(self):
        ckpt_path = os.path.join(self.logger.logdir, "checkpoints")
        self.agent.save(ckpt_path)
        print(f"The checkpoint is saved to {ckpt_path}.")
        print(f"Run `python script/test.py checkpoint={ckpt_path} use_training_cfg=True` to evaluate.")
        if self.cfg.export:
            PolicyExporter(self.agent.policy_net).export(path=ckpt_path, verbose=True, export_pnnx=False)
        if self.env.renderer is not None:
            self.env.renderer.close()

        with open(os.path.join(self.logger.logdir, "runtime_profile.txt"), "w", encoding="utf-8") as f:
            self.profiler.print_stats(stream=f, output_unit=1e-3)


class TestRunner:
    def __init__(self, cfg: DictConfig, logger: Logger, env: RecordEpisodeStatistics, agent):
        self.cfg = cfg
        self.logger = logger
        self.env = env
        self.agent = agent

    def save_video_mp4(self, video_array: np.ndarray, name: str):
        # save the video using imageio
        path = os.path.join(self.logger.logdir, "video")
        if not os.path.exists(path):
            os.makedirs(path)
        with imageio.get_writer(os.path.join(path, name), fps=1/self.dt) as video:
            for frame_index in range(video_array.shape[0]):
                frame = video_array[frame_index]
                video.append_data(frame)

    def save_video_tensorboard(self, video_array: np.ndarray, tag: str, step: int):
        self.logger.log_video(tag, video_array, step=step, fps=1/self.dt)
    
    @torch.no_grad()
    def run(self):
        if self.cfg.record_video:
            H_video, W_video = self.env.renderer.video_H, self.env.renderer.video_W
            H_depth, W_depth = self.cfg.sensor.height, self.cfg.sensor.width
            H_scale, W_scale = H_video / H_depth, W_video / W_depth
            H_depth = H_video if H_scale >= W_scale else int(H_depth * W_scale)
            W_depth = W_video if W_scale >= H_scale else int(W_depth * H_scale)
            H, W = H_video, W_video + W_depth
            video_array = np.empty((self.env.renderer.n_envs, self.env.max_steps, H, W, 3), dtype=np.uint8)
        
        obs = self.env.reset()
        pbar = tqdm(range(self.cfg.n_steps))
        n_resets = 1
        n_survive = 0
        n_success = 0
        for i in pbar:
            t1 = pbar._time()
            self.env.detach()
            action, policy_info = self.agent.act(obs, test=True)
            if self.cfg.algo.name != "yopo":
                action = self.env.rescale_action(action)
            obs, loss, terminated, env_info = self.env.step(action)
            if self.cfg.algo.name != 'world' and hasattr(self.agent, "reset"):
                self.agent.reset(env_info["reset"])
            l_episode = (env_info["stats"]["l"] - 1) * self.env.dt
            n_resets += env_info["reset"].sum().item()
            n_survive += env_info["truncated"].sum().item()
            n_success += env_info["success"].sum().item()
            arrive_time = env_info["stats"]["arrive_time"]
            pbar.set_postfix({
                "l_episode": f"{l_episode:.1f}",
                "success_rate": f"{n_success / n_resets:.2f}",
                "survive_rate": f"{n_survive / n_resets:.2f}",
                "fps": f"{int(self.cfg.env.n_envs/(pbar._time()-t1)):,d}"})
            
            log_info = {
                "env_loss": env_info["loss_components"],
                "metrics": {
                    "l_episode": l_episode,
                    "success_rate": n_success / n_resets,
                    "survive_rate": n_survive / n_resets,
                    "arrive_time": arrive_time}}
            self.logger.log_scalars(log_info, i+1)
            
            if self.cfg.record_video:
                n_envs = self.env.renderer.n_envs
                rgb_image: np.ndarray = self.env.renderer.render_fpp()
                index = (np.arange(n_envs), self.env.progress[:n_envs].cpu().numpy()-1)
                depth_image = torchvision.transforms.Resize(
                    (H_depth, W_depth), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(env_info["sensor"][:n_envs])
                depth_image = (depth_image * 255).to(torch.uint8).unsqueeze(-1).expand(-1, -1, -1, 3).cpu().numpy()
                image = np.concatenate([rgb_image, depth_image], axis=-2)
                video_array[index] = image
                
                if env_info["reset"][:n_envs].sum().item() > env_info["success"][:n_envs].sum().item(): # some episodes failed
                    failed = torch.logical_and(env_info["reset"], ~env_info["success"])[:n_envs]
                    idx = failed.nonzero().flatten()[0]
                    video_length = env_info["l"][idx] - 1
                    if self.cfg.video_saveas == "mp4":
                        self.save_video_mp4(video_array[idx, :video_length], f"failed_{i+1}.mp4")
                    elif self.cfg.video_saveas == "tensorboard":
                        self.save_video_tensorboard(video_array[idx.unsqueeze(0), :video_length], "video/fail", i+1)
                if env_info["success"][:n_envs].sum().item() > 0: # some episodes succeeded
                    idx = env_info["success"][:n_envs].nonzero().flatten()[0]
                    video_length = min(env_info["l"][idx].item(), int(env_info["arrive_time"][idx].item()/self.env.dt) + 100) - 1
                    if self.cfg.video_saveas == "mp4":
                        self.save_video_mp4(video_array[idx, :video_length], f"success_{i+1}.mp4")
                    elif self.cfg.video_saveas == "tensorboard":
                        self.save_video_tensorboard(video_array[idx.unsqueeze(0), :video_length], "video/success", i+1)
            
            if self.cfg.display_image:
                display_image(
                    state=obs,
                    action=action,
                    policy_info=policy_info,
                    env_info=env_info)
    
    def close(self):
        if self.cfg.export:
            ckpt_path = os.path.join(self.logger.logdir, "checkpoints")
            PolicyExporter(self.agent.policy_net).export(path=ckpt_path, verbose=True, export_pnnx=False)
        
        if self.env.renderer is not None:
            self.env.renderer.close()