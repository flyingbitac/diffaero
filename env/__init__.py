from typing import Union

import torch
from omegaconf import DictConfig

from diffaero.env.position_control import PositionControl, Sim2RealPositionControl, MultiAgentPositionControl
from diffaero.env.position_control import PositionControl, MultiAgentPositionControl
from diffaero.env.obstacle_avoidance import ObstacleAvoidance, ObstacleAvoidanceYOPO, ObstacleAvoidanceGrid
from diffaero.env.racing import Racing

ENV_ALIAS = {
    "position_control": PositionControl,
    "sim2real_position_control": Sim2RealPositionControl,
    "multi_agent_position_control": MultiAgentPositionControl,
    "obstacle_avoidance": ObstacleAvoidance,
    "obstacle_avoidance_yopo": ObstacleAvoidanceYOPO,
    "racing": Racing
}

def build_env(cfg, device):
    # type: (DictConfig, torch.device) -> Union[PositionControl, MultiAgentPositionControl, ObstacleAvoidance, ObstacleAvoidanceYOPO]
    env_class = ENV_ALIAS[cfg.name]
    if env_class == ObstacleAvoidance and cfg.enable_grid:
        env_class = ObstacleAvoidanceGrid
    return env_class(cfg, device)