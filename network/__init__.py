from typing import Union, Dict, Tuple, List, Optional

from omegaconf import DictConfig
import torch.nn as nn

from quaddif.network.networks import MLP, CNN, RNN, RCNN

BACKBONE_ALIAS: Dict[str, Union[type[MLP], type[CNN], type[RNN], type[RCNN]]] = {
    "mlp": MLP,
    "cnn": CNN,
    "rnn": RNN,
    "rcnn": RCNN
}

def build_network(
    cfg: DictConfig,
    input_dim: Union[int, Tuple[int, Tuple[int, int]]],
    output_dim: int,
    output_act: Optional[nn.Module] = None
) -> Union[MLP, CNN, RNN, RCNN]:
    return BACKBONE_ALIAS[cfg.name](cfg, input_dim, output_dim, output_act)