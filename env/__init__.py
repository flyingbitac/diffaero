from typing import Union

import torch
from omegaconf import DictConfig

from quaddif.env.position_control import PositionControl, MultiAgentPositionControl
from quaddif.env.obstacle_avoidance import ObstacleAvoidance, ObstacleAvoidanceYOPO, ObstacleAvoidanceGrid

ENV_ALIAS = {
    "position_control": PositionControl,
    "multi_agent_position_control": MultiAgentPositionControl,
    "obstacle_avoidance": ObstacleAvoidance,
    "obstacle_avoidance_yopo": ObstacleAvoidanceYOPO,
    "obstacle_avoidance_grid": ObstacleAvoidanceGrid,
    "racing": Racing
}

def build_env(cfg, device):
    # type: (DictConfig, torch.device) -> Union[PositionControl, MultiAgentPositionControl, ObstacleAvoidance, ObstacleAvoidanceYOPO]
    return ENV_ALIAS[cfg.name](cfg, device)