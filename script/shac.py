import torch
import hydra
import sys
sys.path.append('..')
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from line_profiler import LineProfiler

from quaddif.env import PointMassPositionControl
from quaddif.algo.SHAC import SHAC
from quaddif.utils.env import RecordEpisodeStatistics
from quaddif.utils.device import idle_device
from quaddif.utils.logger import Logger

# profiler = LineProfiler()
# profiler.add_function(ENV_CLASS.step)

@hydra.main(config_path="../cfg", config_name="config")
# @profiler
def main(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{idle_device()}" if cfg.device is None else f"{cfg.device}"
    print("Using device", os.environ["CUDA_VISIBLE_DEVICES"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = Logger(cfg)
    
    env: PointMassPositionControl = RecordEpisodeStatistics(PointMassPositionControl(cfg.env, device=device))
    state = env.reset()
    terminated = env.terminated()
    
    agent = SHAC(
        cfg=cfg.algo,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        min_action=-15,
        max_action=15,
        n_envs=cfg.env.n_envs,
        l_rollout=cfg.l_rollout,
        device=device)
    pbar = tqdm(range(cfg.n_updates))
    
    for i in pbar:
        # 超级重要，为了后续轨迹的loss梯度不反向传播到此前的状态，要先把梯度截断
        env.cut_grad()
        # print(agent.actor_loss, state, agent.buffer.states, agent.buffer.values, agent.buffer.logprobs)
        for t in range(cfg.l_rollout):
            action, info = agent.act(state)
            next_state, loss, next_terminated, extra = env.step(action)
            agent.record_loss(loss, info, extra, last_step=(t==cfg.l_rollout-1))
            agent.buffer.add(
                state, 1-loss/10, terminated, info["value"].detach())
            state, terminated = next_state, next_terminated
        target_values = agent.bootstrap2(extra["state_before_reset"].detach(), next_terminated)
        actor_loss, actor_grad_norm = agent.update_actor()
        agent.clear_loss()
        critic_loss, critic_grad_norm = agent.update_critic(target_values)
        agent.buffer.clear()
        
        # log data
        l_episode = extra["stats"]["l"].float().mean().item()
        success_rate = extra['stats']['success_rate']
        pbar.set_postfix({
            "param_norm": f"({actor_grad_norm:.3f}, {critic_grad_norm:.3f})",
            "loss": f"({actor_loss:.3f}, {critic_loss:.3f})",
            "l_episode": f"{l_episode:.3f}",
            "success_rate": f"{success_rate:.2f}"})
        logger.log_scalars({
            "param_norm": {"actor": actor_grad_norm, "critic": critic_grad_norm},
            "loss": {"actor": actor_loss, "critic": critic_loss},
            "l_episode": l_episode,
            "success_rate": success_rate}, i)
    
    global logdir
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

if __name__ == "__main__":
    import os
    main()
    # with open(os.path.join(logdir, "runtime_profile.txt"), "w", encoding="utf-8") as f:
    #     profiler.print_stats(stream=f, output_unit=1e-3)