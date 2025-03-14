from typing import Union

import torch
from omegaconf import DictConfig

from .pointmass import PointMassModel
from .quadrotor import QuadrotorModel

DYNAMICS_ALIAS = {
    "pointmass": PointMassModel,
    "quadrotor": QuadrotorModel
}

def build_dynamics(cfg, device):
    # type: (DictConfig, torch.device) -> Union[PointMassModel, QuadrotorModel]
    return DYNAMICS_ALIAS[cfg.name](cfg, device)