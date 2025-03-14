from typing import Union

import torch
from omegaconf import DictConfig

from quaddif.algo.PPO import *
from quaddif.algo.APG import *
from quaddif.algo.SHAC import *
from quaddif.algo.MASHAC import *
from quaddif.algo.dreamerv3 import *
from quaddif.algo.YOPO import *

AGENT_ALIAS = {
    "ppo": PPO,
    "ppo_rpl": PPO_RPL,
    "shac": SHAC,
    "mashac": MASHAC,
    "shac_q": SHAC_Q,
    "shac_rpl": SHAC_RPL,
    "apg": APG,
    "apg_sto": APG_stochastic,
    "world": World_Agent,
    "yopo": YOPO,
}

def build_agent(cfg: DictConfig, env, device: torch.device):
    return AGENT_ALIAS[cfg.name].build(cfg, env, device)