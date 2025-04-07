from typing import Tuple, Dict, Union, Optional, List
from copy import deepcopy
import os
from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn as nn

from quaddif.network.agents import StochasticActor, DeterministicActor
from quaddif.network.networks import MLP, CNN, RNN, RCNN
from quaddif.dynamics.pointmass import point_mass_quat

class PolicyExporter(nn.Module):
    def __init__(self, actor: Union[StochasticActor, DeterministicActor]):
        super().__init__()
        self.is_stochastic = isinstance(actor, StochasticActor)
        self.is_recurrent = actor.is_rnn_based
        actor_net = actor.actor_mean if self.is_stochastic else actor.actor
        if self.is_recurrent:
            actor_net.hidden_state = torch.empty(0)
            self.hidden_shape = (actor_net.rnn_n_layers, 1, actor_net.rnn_hidden_dim)
        self.actor = deepcopy(actor_net).cpu()
        if isinstance(self.actor, MLP):
            self.forward = self.forward_MLP
        elif isinstance(self.actor, CNN):
            self.forward = self.forward_CNN
        elif isinstance(self.actor, RNN):
            self.forward = self.forward_RNN
        elif isinstance(self.actor, RCNN):
            self.forward = self.forward_RCNN
    
    def post_process(self, raw_action, min_action, max_action, orientation):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        raw_action = raw_action.tanh() if self.is_stochastic else raw_action
        action = (raw_action * 0.5 + 0.5) * (max_action - min_action) + min_action
        quat_xyzw = point_mass_quat(action, orientation)
        acc_norm = action.norm(p=2, dim=-1)
        return action, quat_xyzw, acc_norm
    
    def forward_MLP(self, state, orientation, min_action, max_action):
        # type: (Union[Tensor, Tuple[Tensor, Tensor]], Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        raw_action = self.actor.forward_export(state)
        action, quat_xyzw, acc_norm = self.post_process(raw_action, min_action, max_action, orientation=orientation)
        return action, quat_xyzw, acc_norm
    
    def forward_CNN(self, state, perception, orientation, min_action, max_action):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        raw_action = self.actor.forward_export(state=state, perception=perception)
        action, quat_xyzw, acc_norm = self.post_process(raw_action, min_action, max_action, orientation=orientation)
        return action, quat_xyzw, acc_norm
    
    def forward_RNN(self, state, orientation, min_action, max_action, hidden):
        # type: (Union[Tensor, Tuple[Tensor, Tensor]], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        raw_action, hidden = self.actor.forward_export(state, hidden=hidden)
        action, quat_xyzw, acc_norm = self.post_process(raw_action, min_action, max_action, orientation=orientation)
        return action, quat_xyzw, acc_norm, hidden
    
    def forward_RCNN(self, state, perception, orientation, min_action, max_action, hidden):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        raw_action, hidden = self.actor.forward_export(state=state, perception=perception, hidden=hidden)
        action, quat_xyzw, acc_norm = self.post_process(raw_action, min_action, max_action, orientation=orientation)
        return action, quat_xyzw, acc_norm, hidden
    
    def export(
        self,
        path: str,
        export_jit,
        export_onnx,
        export_pnnx,
        verbose=False,
    ):
        if export_jit:
            self.export_jit(path, verbose)
        if export_onnx:
            self.export_onnx(path, export_pnnx=export_pnnx)
    
    def export_jit(self, path: str, verbose=False):
        traced_script_module = torch.jit.script(self)
        if verbose:
            print(traced_script_module.code)
        export_path = os.path.join(path, "exported_actor.pt2")
        traced_script_module.save(export_path)
        print(f"The checkpoint is compiled and exported to {export_path}.")
    
    def export_onnx(self, path: str, export_pnnx: bool = True):
        input_dim = self.actor.input_dim
        state_dim = input_dim[0] if isinstance(input_dim, tuple) else input_dim
        perception_dim = input_dim[1] if isinstance(input_dim, tuple) else None
        named_inputs = [
            ("state", torch.rand(1, state_dim)),
            ("orientation", torch.rand(1, 3)),
            ("min_action", torch.rand(1, 3)),
            ("max_action", torch.rand(1, 3))
        ]
        if perception_dim is not None:
            named_inputs.insert(1, ("perception", torch.rand(1, perception_dim[0], perception_dim[1])))
        output_names = [
            "action",
            "quat_xyzw_cmd",
            "acc_norm"]
        if self.is_recurrent:
            named_inputs.append(("hidden_in", torch.rand(self.hidden_shape)))
            output_names.append("hidden_out")
        export_path = os.path.join(path, "exported_actor.onnx")
        
        torch.onnx.export(self, list(zip(*named_inputs))[1], export_path, input_names=list(zip(*named_inputs))[0], output_names=output_names)
        print(f"The checkpoint is compiled and exported to {export_path}.")
        
        # if export_pnnx:
        #     import pnnx
        #     pnnx.convert(export_path, example_input)