import os
import sys
sys.path.append('..')

import isaacgym
import torch
import hydra
from omegaconf import DictConfig
from line_profiler import LineProfiler

from quaddif.env import PointMassPositionControl, PointMassObstacleAvoidance
from quaddif.algo.APG import APG, APG_stocastic, learn
from quaddif.utils.env import RecordEpisodeStatistics
from quaddif.utils.device import idle_device
from quaddif.utils.logger import Logger

AGENT_CLASS = APG_stocastic

# profiler = LineProfiler()
# profiler.add_function(PointMassPositionControl.step)
# profiler.add_function(AGENT_CLASS.update_actor)

@hydra.main(config_path="../cfg", config_name="config")
# @profiler
def main(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{idle_device()}" if cfg.device is None else f"{cfg.device}"
    print("Using device", os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = Logger(cfg)
    
    ENV_CLASS = {
        "position_control": PointMassPositionControl,
        "obstacle_avoidance": PointMassObstacleAvoidance
    }[cfg.env.name]
    env = RecordEpisodeStatistics(ENV_CLASS(cfg.env, device=device))
    agent = AGENT_CLASS.build(cfg, env, device)
    
    learn(cfg, agent, env, logger)
    
    if not cfg.env.render.headless:
        env.renderer.close()
    global logdir
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

if __name__ == "__main__":
    main()
    # with open(os.path.join(logdir, "runtime_profile.txt"), "w", encoding="utf-8") as f:
    #     profiler.print_stats(stream=f, output_unit=1e-3)