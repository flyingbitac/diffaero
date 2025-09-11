from typing import Union, Dict, Tuple, List, Optional

from omegaconf import DictConfig
import torch.nn as nn

from .networks import MLP, CNN, RNN, RCNN, build_network
from .agents import (
    AgentBase,
    DeterministicActor,
    StochasticActor,
    CriticQ,
    CriticV,
    ActorCriticBase,
    StochasticActorCriticQ,
    StochasticActorCriticV
)