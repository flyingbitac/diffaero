import sys
sys.path.append('..')

import isaacgym
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from line_profiler import LineProfiler
import cv2

from quaddif.env import PointMassObstacleAvoidance
from quaddif.algo.APG import APG, APG_stocastic
from quaddif.utils.env import RecordEpisodeStatistics
from quaddif.utils.device import idle_device
from quaddif.utils.logger import Logger
from quaddif.utils.render import ObstacleAvoidanceRenderer

AGENT_CLASS = APG

profiler = LineProfiler()
profiler.add_function(PointMassObstacleAvoidance.step)
profiler.add_function(ObstacleAvoidanceRenderer.step)

@hydra.main(config_path="../cfg", config_name="config_oa")
@profiler
def main(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{idle_device()}" if cfg.device is None else f"{cfg.device}"
    print("Using device", os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = Logger(cfg)
    
    env: PointMassObstacleAvoidance = RecordEpisodeStatistics(PointMassObstacleAvoidance(cfg.env, device=device))
    state, image = env.reset()
    
    agent = AGENT_CLASS(
        cfg=cfg.algo,
        state_dim=env.state_dim + cfg.env.render.image_size[1] * cfg.env.render.image_size[2],
        hidden_dim=list(cfg.algo.hidden_dim),
        action_dim=env.action_dim,
        min_action=env.min_action,
        max_action=env.max_action,
        l_rollout=cfg.l_rollout,
        device=device)
    pbar = tqdm(range(cfg.n_updates))
    
    for i in pbar:
        # 超级重要，为了后续轨迹的loss梯度不反向传播到此前的状态，要先把梯度截断
        env.cut_grad()
        for t in range(cfg.l_rollout):
            action, info = agent.act(torch.concat([state, image.reshape(env.n_envs, -1)], dim=-1))
            (state, image), loss, terminated, extra = env.step(action)
            agent.record_loss(loss, info, extra)
            # N = 64
            # C, H, W = list(cfg.env.render.image_size)
            # NH = NW = int(N**0.5)
            # disp_image = image[:N].reshape(NH, NW, C, H, W).permute(2, 0, 3, 1, 4).reshape(C, NH*H, NW*W).cpu().numpy().transpose(1, 2, 0)
            # disp_image = cv2.normalize(disp_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # disp_image = cv2.resize(cv2.cvtColor(disp_image, cv2.COLOR_GRAY2BGR), (NH*H*4, NW*W*4), interpolation=cv2.INTER_NEAREST)
            # cv2.imshow('image', disp_image)
            # cv2.waitKey(1)
            
        actor_loss, grad_norm = agent.update_actor()
        
        # log data
        l_episode = extra["stats"]["l"].float().mean().item()
        success_rate = extra['stats']['success_rate']
        pbar.set_postfix({
            "param_norm": f"{grad_norm:.3f}",
            "loss": f"{loss.mean():.3f}",
            "l_episode": f"{l_episode:.3f}",
            "success_rate": f"{success_rate:.2f}"})
        logger.log_scalars({
            "env_loss": extra["loss_components"],
            "agent_loss": {"actor_loss": actor_loss, "actor_grad_norm": grad_norm},
            "metrics": {"l_episode": l_episode, "success_rate": success_rate}}, i)
    
    env.renderer.close()
    global logdir
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

if __name__ == "__main__":
    import os
    main()
    with open(os.path.join(logdir, "runtime_profile.txt"), "w", encoding="utf-8") as f:
        profiler.print_stats(stream=f, output_unit=1e-3)