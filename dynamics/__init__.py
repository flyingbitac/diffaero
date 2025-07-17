from typing import Union

import torch
from omegaconf import DictConfig

from .pointmass import ContinuousPointMassModel, DiscretePointMassModel, PointMassModelBase
from .quadrotor import QuadrotorModel

DYNAMICS_ALIAS = {
    "countinuous_pointmass": ContinuousPointMassModel,
    "discrete_pointmass": DiscretePointMassModel,
    "quadrotor": QuadrotorModel
}

def build_dynamics(cfg, device):
    # type: (DictConfig, torch.device) -> Union[ContinuousPointMassModel, DiscretePointMassModel, QuadrotorModel]
    return DYNAMICS_ALIAS[cfg.name](cfg, device)