from pathlib import Path
import os
import sys
sys.path.append('..')

import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm


from diffaero.algo import build_agent
from diffaero.env import build_env  
from diffaero.algo.dreamerv3.world import World_Agent, train_agents
from diffaero.utils.logger import Logger

@hydra.main(config_path=str(Path(__file__).parent.parent.joinpath("cfg")), config_name="config_finetune")
def main(cfg:DictConfig):
    device = f"cuda:{cfg.device}" if torch.cuda.is_available() and cfg.device != -1 else "cpu"
    device = torch.device(device)
    cfg.algo.world_state_env.use_extern = True
    env = build_env(cfg.env, device)
    agent = build_agent(cfg.algo, env, device)
    agent.replaybuffer.load_external(cfg.extern_path)
    assert isinstance(agent, World_Agent), "Only support World Agent for finetune!!!"
    if cfg.extern_path != "":
        agent.replaybuffer.load_external(cfg.extern_path)
    logger = Logger(cfg, cfg.runname)
    best_rewards_sum = 0
    with tqdm(total=cfg.n_updates) as pbar:
        for update in range(cfg.n_updates):
            finetune_info = agent.finetune()
            for k, v in finetune_info.items():
                logger.log_scalar(k, v, update)
            pbar.update(1)
            pbar.set_postfix({'reward_sum':finetune_info["reward_sum"]})
            if finetune_info["reward_sum"] > best_rewards_sum:
                agent.save(os.path.join(logger.logdir, "best"))
                best_rewards_sum = finetune_info["reward_sum"]

if __name__ == '__main__':
    main()