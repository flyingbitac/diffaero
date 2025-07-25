from typing import Union, Optional
from collections import defaultdict
import os
from copy import deepcopy

import torch
import torchvision
import numpy as np
from line_profiler import LineProfiler
import hydra
from tqdm import tqdm
import cv2
import imageio
from omegaconf import DictConfig

from diffaero.env.base_env import BaseEnv, BaseEnvMultiAgent
from diffaero.utils.logger import Logger

def display_image(obs, action, policy_info, env_info):
    # type: (torch.Tensor, torch.Tensor, dict, dict[str, torch.Tensor]) -> None
    if "sensor" in env_info.keys():
        N, C = min(64, obs['perception'].size(0)), 1
        H, W = env_info["sensor"].shape[-2:]
        NH = NW = int(N**0.5)
        scale = 4
        disp_image = env_info["sensor"][:N].reshape(NH, NW, C, H, W).permute(2, 0, 3, 1, 4).reshape(C, NH*H, NW*W).cpu().numpy().transpose(1, 2, 0)
        disp_image = cv2.normalize(disp_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_image = cv2.resize(cv2.cvtColor(disp_image, cv2.COLOR_GRAY2BGR), (int(NW*W*scale), int(NH*H*scale)), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('image', disp_image)
        cv2.waitKey(1)

def timeit(fn):
    """
    Profile this function using `LineProfiler` during training.
    
    Usage:
    ```
    from diffaero.utils.runner import timeit
    @timeit
    def foo():
        return
    
    class Bar:
        @timeit
        def foo(self):
            return
    ```
    """
    def wrapper(*args, **kwargs):
        if fn not in TrainRunner.profiler.functions:
            # Add the function to the profiler
            TrainRunner.profiler.add_function(fn)
        # Call the original function
        return fn(*args, **kwargs)
    return wrapper


class RecordEpisodeStatistics:
    def __init__(self, env: Union[BaseEnv, BaseEnvMultiAgent], max_window_length: Optional[int] = None):
        self.env = env
        self.n_envs = getattr(env, "n_envs", 1)
        self.device = env.device
        self.max_window_length = self.n_envs if max_window_length is None else max_window_length
        # dictionary to be used to store statistical metrics
        # metric will be calculated in a sliding-window manner
        # with length of the window = n_envs
        self.stats = defaultdict(lambda: torch.zeros(self.max_window_length, dtype=torch.float, device=self.device))
        self.window_length = defaultdict(lambda: 0)
    
    def __getattr__(self, name: str):
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)
    
    def step(self, *args, **kwargs):
        state, (loss, reward), terminated, extra = self.env.step(*args, **kwargs)
        # dictionary to be used to store scalar metrics
        extra["stats"] = {} # Dict[str, float]
        # traverse through all metrics to be sliding-window-averaged
        for k, v in extra.get("stats_raw", {}).items(): # Dict[str, Tensor]
            assert v.ndim == 1
            # construct a queue to record new data and discard old ones
            l = v.size(0)
            if l > 0:
                self.stats[k] = torch.roll(self.stats[k], shifts=-l, dims=0)
                self.stats[k][-l:] = v
            # write the scalar metrics back to the extra info provided by the environment
            l_window = min(self.max_window_length, self.window_length[k] + l)
            if l_window > 0:
                extra["stats"][k] = self.stats[k][-l_window:].mean().item()
            self.window_length[k] = l_window
        return state, (loss, reward), terminated, extra


class TrainRunner:
    profiler = LineProfiler()
    def __init__(self, cfg: DictConfig, logger: Logger, env: Union[BaseEnv, BaseEnvMultiAgent], agent):
        self.cfg = cfg
        self.logger = logger
        self.env = RecordEpisodeStatistics(env)
        self.agent = agent
        self.run = self.profiler(self.run)
        self.max_success_rate = 0.
        
        if cfg.torch_profile:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if env.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.torch_profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=10, active=10, repeat=1, skip_first=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    dir_name=os.path.join(self.logger.logdir, "profiling_data"),
                    use_gzip=True
                ),
            )
        else: 
            self.torch_profiler = None
    
    def run(self):
        """Start training."""
        # make progress bars to display on different rows in multirun mode
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get() # type: ignore
        is_multirun = hydra_cfg.mode == hydra.types.RunMode.MULTIRUN # type: ignore
        job_id = hydra_cfg.job.num if is_multirun else 0
        desc = f"Job {job_id:2d}" if is_multirun else ""
        pbar = tqdm(range(self.cfg.n_updates), position=job_id%self.cfg.n_jobs, desc=desc)
        
        obs = self.env.reset()
        on_step_cb = display_image if self.cfg.display_image else None
        if self.torch_profiler is not None:
            self.torch_profiler.start()
        for i in pbar:
            if self.torch_profiler is not None:
                self.torch_profiler.step()
            t1 = pbar._time()
            self.env.detach()
            obs, policy_info, env_info, losses, grad_norms = self.agent.step(self.cfg, self.env, obs, on_step_cb=on_step_cb)
            l_episode = env_info["stats"].get("l_episode", 0.)
            success_rate = env_info["stats"].get("success_rate", 0.)
            survive_rate = env_info["stats"].get("survive_rate", 0.)
            pbar.set_postfix({
                "loss": f"{env_info['loss_components']['total_loss']:6.3f}",
                "l_episode": f"{l_episode:4.1f}",
                "success_rate": f"{success_rate:.2f}",
                "survive_rate": f"{survive_rate:.2f}",
                "fps": f"{int(self.cfg.l_rollout*self.cfg.n_envs/(pbar._time()-t1)):,d}"})
            if i % self.cfg.log_freq == 0:
                log_info = {
                    "env_loss": env_info["loss_components"],
                    "agent_loss": losses,
                    "agent_grad_norm": grad_norms,
                    "metrics": env_info["stats"]
                }
                if "value" in policy_info.keys():
                    log_info["value"] = policy_info["value"].mean().item()
                if "grid" in policy_info.keys():
                    self.logger.log_images("grid", deepcopy(policy_info["grid"]), i+1)
                    del(policy_info["grid"])
                if "WorldModel/state_total_loss" in policy_info.keys():
                    log_info.update({k: v for k, v in policy_info.items() if k.startswith("WorldModel")})
                self.logger.log_scalars(log_info, i+1)
            
            if i % 100 == 0 and any([k.startswith("grid") for k in policy_info.keys()]):
                grid_gt = self.env.visualize_grid(policy_info["grid_gt"])
                grid_pred = self.env.visualize_grid(policy_info["grid_pred"])
                grid = np.concatenate([grid_gt, grid_pred], axis=1).transpose(2, 0, 1)
                self.logger.log_image("grid", grid, i+1)
            
            if success_rate >= self.max_success_rate:
                self.max_success_rate = success_rate
                if i > 0 and i % self.cfg.save_freq == 0:
                    self.agent.save(os.path.join(self.logger.logdir, "best"))
    
    def close(self) -> float:
        """
        Save (and export) the trained policy, 
        close the environment renderer, 
        and write the profiled data to the disk.
        """
        if self.torch_profiler is not None and self.torch_profiler.step_num == self.cfg.n_updates:
            self.torch_profiler.stop()
            print(self.torch_profiler.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=20))
        ckpt_path = os.path.join(self.logger.logdir, "checkpoints")
        self.agent.save(ckpt_path)
        Logger.info(f"The checkpoint is saved to {ckpt_path}.")
        Logger.info(f"Run `python script/test.py checkpoint={ckpt_path} use_training_cfg=True` to evaluate.")
        if any(dict(self.cfg.export).values()):
            self.agent.export(
                path=ckpt_path,
                export_cfg=self.cfg.export,
                verbose=True
            )
        if self.env.renderer is not None:
            self.env.renderer.close()

        with open(os.path.join(self.logger.logdir, "runtime_profile.txt"), "w", encoding="utf-8") as f:
            self.profiler.print_stats(stream=f, output_unit=1e-3)
        
        self.logger.close()
        
        return self.max_success_rate


class TestRunner:
    def __init__(self, cfg: DictConfig, logger: Logger, env: Union[BaseEnv, BaseEnvMultiAgent], agent):
        self.cfg = cfg
        self.logger = logger
        self.env = RecordEpisodeStatistics(env, max_window_length=cfg.n_steps)
        self.agent = agent
        self.success_rate = 0.

    def save_video_mp4(self, video_array: np.ndarray, name: str):
        # save the video using imageio
        path = os.path.join(self.logger.logdir, "video")
        if not os.path.exists(path):
            os.makedirs(path)
        with imageio.get_writer(os.path.join(path, name), fps=1/self.env.dt) as video:
            for frame_index in range(video_array.shape[0]):
                frame = video_array[frame_index]
                video.append_data(frame)

    def save_video_tensorboard(self, video_array: np.ndarray, tag: str, step: int):
        self.logger.log_video(tag, video_array, step=step, fps=1/self.env.dt)
    
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
            l_episode = env_info["stats"].get("l_episode", 0.)
            success_rate = env_info["stats"].get("success_rate", 0.)
            survive_rate = env_info["stats"].get("survive_rate", 0.)
            pbar.set_postfix({
                "l_episode": f"{l_episode:.1f}",
                "success_rate": f"{success_rate:.2f}",
                "survive_rate": f"{survive_rate:.2f}",
                "fps": f"{int(self.cfg.env.n_envs/(pbar._time()-t1)):,d}"})
            log_info = {
                "env_loss": env_info.get("loss_components", {}), 
                "metrics": env_info["stats"]}
            self.logger.log_scalars(log_info, i+1)
            self.success_rate = n_success / n_resets
            
            if self.cfg.record_video:
                n_envs = self.env.renderer.n_envs
                rgb_image: np.ndarray = self.env.renderer.render_fpp(self.env.states_for_render())
                index = (np.arange(n_envs), self.env.progress[:n_envs].cpu().numpy()-1)
                depth_image = torchvision.transforms.Resize(
                    (H_depth, W_depth), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(env_info["sensor"][:n_envs])
                depth_image = (depth_image * 255).to(torch.uint8).unsqueeze(-1).expand(-1, -1, -1, 3).cpu().numpy()
                image = np.concatenate([rgb_image, depth_image], axis=-2)
                video_array[index] = image
                reset, success = env_info["reset"][:n_envs], env_info["success"][:n_envs]
                if reset.sum().item() > success.sum().item(): # some episodes failed
                    failed = torch.logical_and(reset, ~success)[:n_envs]
                    idx = failed.nonzero().flatten()[0]
                    video_length = env_info["l"][idx].item() - 1
                    if self.cfg.video_saveas == "mp4":
                        self.save_video_mp4(video_array[idx, :video_length], f"failed_{i+1}.mp4")
                    elif self.cfg.video_saveas == "tensorboard":
                        self.save_video_tensorboard(video_array[idx.unsqueeze(0), :video_length], "video/fail", i+1)
                if success.sum().item() > 0: # some episodes succeeded
                    idx = success.nonzero().flatten()[0]
                    video_length = min(env_info["l"][idx].item(), int(env_info["arrive_time"][idx].item()/self.env.dt) + 100) - 1
                    if self.cfg.video_saveas == "mp4":
                        self.save_video_mp4(video_array[idx, :video_length], f"success_{i+1}.mp4")
                    elif self.cfg.video_saveas == "tensorboard":
                        self.save_video_tensorboard(video_array[idx.unsqueeze(0), :video_length], "video/success", i+1)
            
            if self.cfg.display_image:
                display_image(
                    obs=obs,
                    action=action,
                    policy_info=policy_info,
                    env_info=env_info)
    
    def close(self):
        if any(dict(self.cfg.export).values()):
            ckpt_path = os.path.join(self.logger.logdir, "checkpoints")
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            self.agent.export(
                path=ckpt_path,
                export_cfg=self.cfg.export,
                verbose=True
            )
        if self.env.renderer is not None:
            self.env.renderer.close()
        
        self.logger.close()
        
        return self.success_rate