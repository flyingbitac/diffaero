import random
from time import sleep
import sys
sys.path.append('..')

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../cfg", config_name="config_train", version_base="1.3")
def main(cfg: DictConfig):
    
    import torch
    import numpy as np
    
    from quaddif.env import build_env
    from quaddif.algo import build_agent
    from quaddif.utils.device import get_idle_device
    from quaddif.utils.logger import Logger
    from quaddif.utils.runner import TrainRunner

    if cfg.device is None and cfg.n_jobs > 1:
        sleep(random.random() * 3)
    device_idx = get_idle_device() if cfg.device is None else cfg.device
    device = f"cuda:{device_idx}" if torch.cuda.is_available() and device_idx != "-1" else "cpu"
    print(f"Using device {device}.")
    device = torch.device(device)
    
    if cfg.seed != -1:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = cfg.torch_deterministic

    env = build_env(cfg.env, device=device)
    
    agent = build_agent(cfg.algo, env, device)
    
    runname = f"__{cfg.runname}" if len(cfg.runname) > 0 else ""
    logger = Logger(cfg, run_name=runname)
    
    # profiler = LineProfiler()
    # if hasattr(env, "update_sensor_data"):
    #     profiler.add_function(env_class.update_sensor_data)
    # if env.renderer is not None:
    #     profiler.add_function(env.renderer.render)
    # profiler.add_function(env_class.step)
    # profiler.add_function(env_class.state)
    # profiler.add_function(env_class.loss_fn)
    # profiler.add_function(agent_class.step)
    # @profiler
    def learn(
        on_step_cb: Optional[Callable] = None
    ):
        state = env.reset()
        max_success_rate = 0
        pbar = tqdm(range(cfg.n_updates), ncols=100)
        for i in pbar:
            t1 = pbar._time()
            env.detach()
            state, policy_info, env_info, losses, grad_norms = agent.step(cfg, env, state, on_step_cb)
            l_episode = (env_info["stats"]["l"] - 1) * env.dt
            success_rate = env_info["stats"]["success_rate"]
            survive_rate = env_info["stats"]["survive_rate"]
            arrive_time = env_info["stats"]["arrive_time"]
            if cfg.algo.name != 'world':
                pbar.set_postfix({
                    # "param_norm": f"{grad_norms['actor_grad_norm']:.3f}",
                    "loss": f"{env_info['loss_components']['total_loss']:.3f}",
                    "l_episode": f"{l_episode:.1f}",
                    "success_rate": f"{success_rate:.2f}",
                    "survive_rate": f"{survive_rate:.2f}",
                    "fps": f"{int(cfg.l_rollout*cfg.env.n_envs/(pbar._time()-t1)):,d}"})
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
                logger.log_scalars(log_info, i+1)
            
            if success_rate >= max_success_rate:
                max_success_rate = success_rate
                agent.save(os.path.join(logger.logdir, "best"))
    
    try:
        runner.run()
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        max_success_rate = runner.close()
    
    global logdir
    logdir = logger.logdir
    # with open(os.path.join(logdir, "runtime_profile.txt"), "w", encoding="utf-8") as f:
    #     profiler.print_stats(stream=f, output_unit=1e-3)

if __name__ == "__main__":
    main()