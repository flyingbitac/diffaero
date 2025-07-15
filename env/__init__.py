from typing import Union

import torch
from omegaconf import DictConfig

from quaddif.env.position_control import PositionControl, Sim2RealPositionControl, MultiAgentPositionControl
from quaddif.env.position_control import PositionControl, MultiAgentPositionControl
from quaddif.env.obstacle_avoidance import ObstacleAvoidance, ObstacleAvoidanceYOPO, ObstacleAvoidanceGrid
from quaddif.env.racing import Racing

ENV_ALIAS = {
    "position_control": PositionControl,
    "sim2real_position_control": Sim2RealPositionControl,
    "multi_agent_position_control": MultiAgentPositionControl,
    "obstacle_avoidance": ObstacleAvoidance,
    "obstacle_avoidance_yopo": ObstacleAvoidanceYOPO,
    "obstacle_avoidance_grid": ObstacleAvoidanceGrid,
    "racing": Racing
}

def build_env(cfg, device):
    # type: (DictConfig, torch.device) -> Union[PositionControl, MultiAgentPositionControl, ObstacleAvoidance, ObstacleAvoidanceYOPO]
    return ENV_ALIAS[cfg.name](cfg, device)