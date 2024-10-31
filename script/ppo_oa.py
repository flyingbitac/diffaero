import isaacgym
import torch
import hydra
import sys
sys.path.append('..')
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from line_profiler import LineProfiler

from quaddif.env import *
from quaddif.algo.PPO import PPO
from quaddif.utils.env import RecordEpisodeStatistics
from quaddif.utils.device import idle_device
from quaddif.utils.logger import Logger

ENV_CLASS = PointMassObstacleAvoidance
AGENT_CLASS = PPO

profiler = LineProfiler()
profiler.add_function(ENV_CLASS.step)
profiler.add_function(ObstacleAvoidanceRenderer.step)

@hydra.main(config_path="../cfg", config_name="config_oa")
@profiler
def main(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{idle_device()}" if cfg.device is None else f"{cfg.device}"
    print("Using device", os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = Logger(cfg)
    
    env: ENV_CLASS = RecordEpisodeStatistics(ENV_CLASS(cfg.env, device=device))
    drone_state, image = env.reset()
    terminated = torch.zeros(env.n_envs, dtype=torch.float, device=device)
    
    agent = AGENT_CLASS(
            cfg=cfg.algo,
            state_dim=env.state_dim + cfg.env.render.image_size[1] * cfg.env.render.image_size[2],
            hidden_dim=list(cfg.algo.hidden_dim),
            action_dim=env.action_dim,
            min_action=env.min_action,
            max_action=env.max_action,
            n_envs=env.n_envs,
            l_rollout=cfg.l_rollout,
            device=device)
    pbar = tqdm(range(cfg.n_updates))
    
    for i in pbar:
        # 超级重要，为了后续轨迹的loss梯度不反向传播到此前的状态，要先把梯度截断
        # env.cut_grad()
        with torch.no_grad():
            for t in range(cfg.l_rollout):
                state = torch.concat([drone_state, image.reshape(env.n_envs, -1)], dim=-1)
                action, info = agent.act(state)
                (next_drone_state, next_image), loss, next_terminated, extra = env.step(action)
                next_state = torch.concat([next_drone_state, next_image.reshape(env.n_envs, -1)], dim=-1)
                agent.add(state, 1-loss/10, terminated, info)
                terminated = next_terminated
                drone_state, image = next_drone_state, next_image
        
        advantages, target_values = agent.bootstrap(next_state, next_terminated)
        for _ in range(cfg.algo.n_epoch):
            losses, grad_norms = agent.train(advantages, target_values)
        
        # log data
        l_episode = extra["stats"]["l"].float().mean().item()
        success_rate = extra['stats']['success_rate']
        losses.update(grad_norms)
        pbar.set_postfix({
            "param_norm": f"{grad_norms['actor_grad_norm']:.3f}",
            "loss": f"{loss.mean():.3f}",
            "l_episode": f"{l_episode:.3f}",
            "success_rate": f"{success_rate:.2f}"})
        logger.log_scalars({
            "env_loss": extra["loss_components"],
            "agent_loss": losses,
            "metrics": {"l_episode": l_episode, "success_rate": success_rate}}, i)
    
    global logdir
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

if __name__ == "__main__":
    import os
    main()
    with open(os.path.join(logdir, "runtime_profile.txt"), "w", encoding="utf-8") as f:
        profiler.print_stats(stream=f, output_unit=1e-3)