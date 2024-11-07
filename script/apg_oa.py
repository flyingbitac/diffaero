import sys
sys.path.append('..')

import isaacgym
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from line_profiler import LineProfiler
import cv2

from quaddif.env import PointMassObstacleAvoidance
from quaddif.algo.APG import APG, APG_stocastic, learn
from quaddif.utils.env import RecordEpisodeStatistics
from quaddif.utils.device import idle_device
from quaddif.utils.logger import Logger, CallBack
from quaddif.utils.render import ObstacleAvoidanceRenderer

AGENT_CLASS = APG_stocastic

profiler = LineProfiler()
profiler.add_function(PointMassObstacleAvoidance.step)
profiler.add_function(ObstacleAvoidanceRenderer.step)

class CallBack(CallBack):
    def __init__(self, cfg):
        self.cfg = cfg
    def on_step(self, **kwargs):
        N = 64
        C, H, W = list(self.cfg.env.render.image_size)
        NH = NW = int(N**0.5)
        disp_image = kwargs["extra"]["camera"][:N].reshape(NH, NW, C, H, W).permute(2, 0, 3, 1, 4).reshape(C, NH*H, NW*W).cpu().numpy().transpose(1, 2, 0)
        disp_image = cv2.normalize(disp_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_image = cv2.resize(cv2.cvtColor(disp_image, cv2.COLOR_GRAY2BGR), (NH*H*4, NW*W*4), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('image', disp_image)
        cv2.waitKey(1)
    def on_update(self, **kwargs):
        pass

@hydra.main(config_path="../cfg", config_name="config_oa")
@profiler
def main(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{idle_device()}" if cfg.device is None else f"{cfg.device}"
    print("Using device", os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = Logger(cfg)
    
    env: PointMassObstacleAvoidance = RecordEpisodeStatistics(PointMassObstacleAvoidance(cfg.env, device=device))
    agent = AGENT_CLASS.build(cfg, env, device)
    
    # learn(cfg, agent, env, logger, CallBack(cfg))
    learn(cfg, agent, env, logger)
    
    env.renderer.close()
    global logdir
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

if __name__ == "__main__":
    import os
    main()
    with open(os.path.join(logdir, "runtime_profile.txt"), "w", encoding="utf-8") as f:
        profiler.print_stats(stream=f, output_unit=1e-3)