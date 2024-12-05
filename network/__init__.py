from typing import Union

from omegaconf import DictConfig

from quaddif.network.mlp import *
from quaddif.network.cnn import *

NETWORKS = {
    "mlp": {
        "deterministic_actor": DeterministicActorMLP,
        "stochastic_actor": StochasticActorMLP,
        "critic": CriticMLP,
        "stochastic_actor_critic": StochasticActorCriticMLP,
        "rpl_actor_critic": RPLActorCriticMLP
    },
    "cnn": {
        "deterministic_actor": DeterministicActorCNN,
        "stochastic_actor": StochasticActorCNN,
        "critic": CriticCNN,
        "stochastic_actor_critic": StochasticActorCriticCNN,
        "rpl_actor_critic": RPLActorCriticCNN
    },
}

def DeterministicActor(algo_cfg):
    # type: (DictConfig) -> Union[DeterministicActorMLP, DeterministicActorCNN]
    return NETWORKS[algo_cfg.network.name]["deterministic_actor"]

def StochasticActor(algo_cfg):
    # type: (DictConfig) -> Union[StochasticActorMLP, StochasticActorCNN]
    return NETWORKS[algo_cfg.network.name]["stochastic_actor"]

def Critic(algo_cfg):
    # type: (DictConfig) -> Union[CriticMLP, CriticCNN]
    return NETWORKS[algo_cfg.network.name]["critic"]

def StochasticActorCritic(algo_cfg):
    # type: (DictConfig) -> Union[StochasticActorCriticMLP, StochasticActorCriticCNN]
    return NETWORKS[algo_cfg.network.name]["stochastic_actor_critic"]

def RPLActorCritic(algo_cfg):
    # type: (DictConfig) -> Union[RPLActorCriticMLP, RPLActorCriticCNN]
    return NETWORKS[algo_cfg.network.name]["rpl_actor_critic"]