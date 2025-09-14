from typing import Union

import torch
from omegaconf import DictConfig

from diffaero.algo.PPO import PPO, AsymmetricPPO
from diffaero.algo.APG import APG, APG_stochastic
from diffaero.algo.SHAC import SHAC, SHA2C
from diffaero.algo.MASHAC import MASHAC
from diffaero.algo.dreamerv3 import World_Agent
AGENT_ALIAS = {
    "ppo": PPO,
    "appo": AsymmetricPPO,
    "shac": SHAC,
    "sha2c": SHA2C,
    "mashac": MASHAC,
    "apg": APG,
    "apg_sto": APG_stochastic,
    "world": World_Agent,
}

def build_agent(cfg: DictConfig, env, device: torch.device):
    return AGENT_ALIAS[cfg.name].build(cfg, env, device)