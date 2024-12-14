from typing import Union

from omegaconf import DictConfig

from quaddif.network.mlp import *
from quaddif.network.cnn import *
from quaddif.network.rnn import *

def DeterministicActor(
    algo_cfg: DictConfig,
    state_dim: Union[int, Tuple[int, Tuple[int, int]]],
    hidden_dim: Union[int, List[int]],
    action_dim: int
) -> Union[DeterministicActorMLP, DeterministicActorCNN, DeterministicActorRNN]:
    if algo_cfg.network.name == "mlp":
        return DeterministicActorMLP(state_dim, hidden_dim, action_dim)
    elif algo_cfg.network.name == "cnn":
        return DeterministicActorCNN(state_dim, hidden_dim, action_dim)
    elif algo_cfg.network.name == "rnn":
        return DeterministicActorRNN(state_dim, hidden_dim, action_dim, algo_cfg.network.rnn_hidden_dim, algo_cfg.network.rnn_n_layers)

def StochasticActor(
    algo_cfg: DictConfig,
    state_dim: Union[int, Tuple[int, Tuple[int, int]]],
    hidden_dim: Union[int, List[int]],
    action_dim: int
) -> Union[StochasticActorMLP, StochasticActorCNN, StochasticActorRNN]:
    if algo_cfg.network.name == "mlp":
        return StochasticActorMLP(state_dim, hidden_dim, action_dim)
    elif algo_cfg.network.name == "cnn":
        return StochasticActorCNN(state_dim, hidden_dim, action_dim)
    elif algo_cfg.network.name == "rnn":
        return StochasticActorRNN(state_dim, hidden_dim, action_dim, algo_cfg.network.rnn_hidden_dim, algo_cfg.network.rnn_n_layers)

def StochasticActorCritic_V(
    algo_cfg: DictConfig,
    state_dim: Union[int, Tuple[int, Tuple[int, int]]],
    hidden_dim: Union[int, List[int]],
    action_dim: int
) -> Union[StochasticActorCriticVMLP, StochasticActorCriticVCNN, StochasticActorCriticVRNN]:
    if algo_cfg.network.name == "mlp":
        return StochasticActorCriticVMLP(state_dim, hidden_dim, action_dim)
    elif algo_cfg.network.name == "cnn":
        return StochasticActorCriticVCNN(state_dim, hidden_dim, action_dim)
    elif algo_cfg.network.name == "rnn":
        return StochasticActorCriticVRNN(state_dim, hidden_dim, action_dim, algo_cfg.network.rnn_hidden_dim, algo_cfg.network.rnn_n_layers)

def StochasticActorCritic_Q(
    algo_cfg: DictConfig,
    state_dim: Union[int, Tuple[int, Tuple[int, int]]],
    hidden_dim: Union[int, List[int]],
    action_dim: int
) -> Union[StochasticActorCriticQMLP, StochasticActorCriticQCNN, StochasticActorCriticQRNN]:
    if algo_cfg.network.name == "mlp":
        return StochasticActorCriticQMLP(state_dim, hidden_dim, action_dim)
    elif algo_cfg.network.name == "cnn":
        return StochasticActorCriticQCNN(state_dim, hidden_dim, action_dim)
    elif algo_cfg.network.name == "rnn":
        return StochasticActorCriticQRNN(state_dim, hidden_dim, action_dim, algo_cfg.network.rnn_hidden_dim, algo_cfg.network.rnn_n_layers)

def RPLActorCritic(
    algo_cfg: DictConfig,
    anchor_ckpt: str,
    state_dim: Tuple[int, Tuple[int, int]],
    anchor_state_dim: int,
    hidden_dim: int,
    action_dim: int,
    rpl_action: bool = True
) -> Union[RPLActorCriticMLP, RPLActorCriticCNN]:
    if algo_cfg.network.name == "mlp":
        return RPLActorCriticMLP(anchor_ckpt, state_dim, anchor_state_dim, hidden_dim, action_dim, rpl_action)
    elif algo_cfg.network.name == "cnn":
        return RPLActorCriticCNN(anchor_ckpt, state_dim, anchor_state_dim, hidden_dim, action_dim, rpl_action)