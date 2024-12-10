from typing import Union

from omegaconf import DictConfig

from quaddif.network.mlp import *
from quaddif.network.cnn import *

NETWORKS = {
    "mlp": {
        "deterministic_actor": DeterministicActorMLP,
        "stochastic_actor": StochasticActorMLP,
        "critic_v": CriticVMLP,
        "critic_q": CriticQMLP,
        "stochastic_actor_critic_v": StochasticActorCriticVMLP,
        "stochastic_actor_critic_q": StochasticActorCriticQMLP,
        "rpl_actor_critic": RPLActorCriticMLP
    },
    "cnn": {
        "deterministic_actor": DeterministicActorCNN,
        "stochastic_actor": StochasticActorCNN,
        "critic_v": CriticVCNN,
        "critic_q": CriticQCNN,
        "stochastic_actor_critic_v": StochasticActorCriticVCNN,
        "stochastic_actor_critic_q": StochasticActorCriticQCNN,
        "rpl_actor_critic": RPLActorCriticCNN
    },
}

def DeterministicActor(
    algo_cfg: DictConfig,
    state_dim: Union[int, Tuple[int, Tuple[int, int]]],
    hidden_dim: Union[int, List[int]],
    action_dim: int
) -> Union[DeterministicActorMLP, DeterministicActorCNN]:
    return NETWORKS[algo_cfg.network.name]["deterministic_actor"](state_dim, hidden_dim, action_dim)

def StochasticActor(
    algo_cfg: DictConfig,
    state_dim: Union[int, Tuple[int, Tuple[int, int]]],
    hidden_dim: Union[int, List[int]],
    action_dim: int
) -> Union[StochasticActorMLP, StochasticActorCNN]:
    return NETWORKS[algo_cfg.network.name]["stochastic_actor"](state_dim, hidden_dim, action_dim)

def Critic_V(
    algo_cfg: DictConfig,
    state_dim: Union[int, Tuple[int, Tuple[int, int]]],
    hidden_dim: Union[int, List[int]]
) -> Union[CriticVMLP, CriticVCNN]:
    return NETWORKS[algo_cfg.network.name]["critic_v"](state_dim, hidden_dim)

def Critic_Q(
    algo_cfg: DictConfig,
    state_dim: Union[int, Tuple[int, Tuple[int, int]]],
    hidden_dim: Union[int, List[int]],
    action_dim: int
) -> Union[CriticQMLP, CriticQCNN]:
    return NETWORKS[algo_cfg.network.name]["critic_q"](state_dim, hidden_dim, action_dim)

def StochasticActorCritic_V(
    algo_cfg: DictConfig,
    state_dim: Union[int, Tuple[int, Tuple[int, int]]],
    hidden_dim: Union[int, List[int]],
    action_dim: int
) -> Union[StochasticActorCriticVMLP, StochasticActorCriticVCNN]:
    return NETWORKS[algo_cfg.network.name]["stochastic_actor_critic_v"](state_dim, hidden_dim, action_dim)

def StochasticActorCritic_Q(
    algo_cfg: DictConfig,
    state_dim: Union[int, Tuple[int, Tuple[int, int]]],
    hidden_dim: Union[int, List[int]],
    action_dim: int
) -> Union[StochasticActorCriticQMLP, StochasticActorCriticQCNN]:
    return NETWORKS[algo_cfg.network.name]["stochastic_actor_critic_q"](state_dim, hidden_dim, action_dim)

def RPLActorCritic(
    algo_cfg: DictConfig,
    anchor_ckpt: str,
    state_dim: Tuple[int, Tuple[int, int]],
    anchor_state_dim: int,
    hidden_dim: int,
    action_dim: int,
    rpl_action: bool = True
) -> Union[RPLActorCriticMLP, RPLActorCriticCNN]:
    return NETWORKS[algo_cfg.network.name]["rpl_actor_critic"](
        anchor_ckpt, state_dim, anchor_state_dim, hidden_dim, action_dim, rpl_action)