from typing import Union

import torch
from omegaconf import DictConfig

from diffaero.env.position_control import PositionControl, Sim2RealPositionControl
from diffaero.env.position_control_multi_agent import MultiAgentPositionControl
from diffaero.env.obstacle_avoidance import ObstacleAvoidance
from diffaero.env.racing import Racing

ENV_ALIAS = {
    "position_control": PositionControl,
    "sim2real_position_control": Sim2RealPositionControl,
    "multi_agent_position_control": MultiAgentPositionControl,
    "obstacle_avoidance": ObstacleAvoidance,
    "racing": Racing
}

def build_env(cfg, device):
    # type: (DictConfig, torch.device) -> Union[PositionControl, MultiAgentPositionControl, ObstacleAvoidance, Racing]
    env_class = ENV_ALIAS[cfg.name]
    return env_class(cfg, device)