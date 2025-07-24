from typing import Union

import torch
from omegaconf import DictConfig

from diffaero.algo.PPO import PPO, AsymmetricPPO, PPO_RPL
from diffaero.algo.APG import APG, APG_stochastic
from diffaero.algo.SHAC import SHAC, SHAC_Q, SHAC_PPO, SHAC_RPL, SHA2C
from diffaero.algo.MASHAC import MASHAC
from diffaero.algo.dreamerv3 import World_Agent
from diffaero.algo.GRID import GRID
from diffaero.algo.grid_wm import GRIDWM
from diffaero.algo.YOPO import YOPO

AGENT_ALIAS = {
    "ppo": PPO,
    "appo": AsymmetricPPO,
    "ppo_rpl": PPO_RPL,
    "shac": SHAC,
    "sha2c": SHA2C,
    "mashac": MASHAC,
    "shac_q": SHAC_Q,
    "shac_ppo": SHAC_PPO,
    "shac_rpl": SHAC_RPL,
    "apg": APG,
    "apg_sto": APG_stochastic,
    "world": World_Agent,
    "grid": GRID,
    "grid_wm": GRIDWM,
    "yopo": YOPO,
}

def build_agent(cfg: DictConfig, env, device: torch.device):
    return AGENT_ALIAS[cfg.name].build(cfg, env, device)