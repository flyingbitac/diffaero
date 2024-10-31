import isaacgym
import torch
import hydra
import os
import sys
sys.path.append('..')
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from line_profiler import LineProfiler

from quaddif.env import *
from quaddif.algo.APG import APG, APG_stocastic
from quaddif.utils.env import RecordEpisodeStatistics
from quaddif.utils.device import idle_device
from quaddif.utils.logger import Logger

AGENT_CLASS = APG_stocastic

# profiler = LineProfiler()
# profiler.add_function(ENV_CLASS.step)

@hydra.main(config_path="../cfg", config_name="config_pc")
# @profiler
def main(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{idle_device()}" if cfg.device is None else f"{cfg.device}"
    print("Using device", os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = Logger(cfg)
    
    env: PointMassPositionControl = RecordEpisodeStatistics(PointMassPositionControl(cfg.env, device=device))
    state = env.reset()
    
    agent = AGENT_CLASS.build(cfg, env, device)
    pbar = tqdm(range(cfg.n_updates))
    
    for i in pbar:
        # 超级重要，为了后续轨迹的loss梯度不反向传播到此前的状态，要先把梯度截断
        env.cut_grad()
        for t in range(cfg.l_rollout):
            action, info = agent.act(state)
            state, loss, terminated, extra = env.step(action)
            agent.record_loss(loss, info, extra)
            
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
    if not cfg.env.render.headless:
        env.renderer.close()
    
    global logdir
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

if __name__ == "__main__":
    main()
    # with open(os.path.join(logdir, "runtime_profile.txt"), "w", encoding="utf-8") as f:
    #     profiler.print_stats(stream=f, output_unit=1e-3)