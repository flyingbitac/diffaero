from typing import Tuple, Dict, Union, Optional, List
from copy import deepcopy
import os

import torch
from torch import Tensor
import torch.nn as nn

from quaddif.network.agents import StochasticActor, DeterministicActor
from quaddif.network.networks import MLP, CNN, RNN, RCNN
from quaddif.dynamics.pointmass import point_mass_quat
from quaddif.utils.math import axis_rotmat
from quaddif.utils.logger import Logger

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
        
        self.input_dim = self.actor.input_dim
        state_dim = self.input_dim[0] if isinstance(self.input_dim, tuple) else self.input_dim
        perception_dim = self.input_dim[1] if isinstance(self.input_dim, tuple) else None
        self.named_inputs = [
            ("state", torch.rand(1, state_dim)),
            ("orientation", torch.rand(1, 3)),
            ("min_action", torch.rand(1, 3)),
            ("max_action", torch.rand(1, 3))
        ]
        if perception_dim is not None:
            if isinstance(self.actor, (MLP, RNN)):
                self.named_inputs[0] = ("state", (torch.rand(1, state_dim), torch.rand(1, perception_dim[0], perception_dim[1])))
            elif isinstance(self.actor, (CNN, RCNN)):
                self.named_inputs.insert(1, ("perception", torch.rand(1, perception_dim[0], perception_dim[1])))
        self.output_names = [
            "action",
            "quat_xyzw_cmd",
            "acc_norm"
        ]
        if self.is_recurrent:
            self.named_inputs.append(("hidden_in", torch.rand(self.hidden_shape)))
            self.output_names.append("hidden_out")
    
    def post_process(self, raw_action, min_action, max_action, orientation):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        raw_action = raw_action.tanh() if self.is_stochastic else raw_action
        action = (raw_action * 0.5 + 0.5) * (max_action - min_action) + min_action
        x, y, _ = orientation.unbind(dim=-1)
        cos, sin = x / (x**2 + y**2).sqrt(), y / (x**2 + y**2).sqrt()
        zero, one = torch.zeros_like(cos), torch.ones_like(cos)
        Rz = torch.stack((cos, -sin, zero, sin, cos, zero, zero, zero, one), dim=-1).reshape(orientation.shape[:-1] + (3, 3))
        # Rz = axis_rotmat("Z", torch.atan2(orientation[:, 1], orientation[:, 0]))
        acc_cmd = torch.matmul(Rz, action.unsqueeze(-1)).squeeze(-1)
        quat_xyzw = point_mass_quat(acc_cmd, orientation)
        acc_norm = acc_cmd.norm(p=2, dim=-1)
        return acc_cmd, quat_xyzw, acc_norm
    
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
        verbose=False,
    ):
        if export_jit:
            self.export_jit(path, verbose)
        if export_onnx:
            self.export_onnx(path)
    
    @torch.no_grad()
    def export_jit(self, path: str, verbose=False):
        traced_script_module = torch.jit.script(self)
        if verbose:
            Logger.info("Code of scripted module: \n" + traced_script_module.code)
        export_path = os.path.join(path, "exported_actor.pt2")
        traced_script_module.save(export_path)
        Logger.info(f"The checkpoint is compiled and exported to {export_path}.")
    
    def export_onnx(self, path: str):
        export_path = os.path.join(path, "exported_actor.onnx")
        names, inputs = zip(*self.named_inputs)
        torch.onnx.export(
            model=self,
            args=inputs,
            f=export_path,
            input_names=names,
            output_names=self.output_names
        )
        Logger.info(f"The checkpoint is compiled and exported to {export_path}.")
