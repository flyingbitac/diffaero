from typing import Tuple, Dict, Union, Optional, List
from copy import deepcopy
import os

import torch
from torch import Tensor
import torch.nn as nn
from omegaconf import DictConfig
import onnxruntime as ort
import numpy as np

from diffaero.network.agents import StochasticActor, DeterministicActor
from diffaero.network.networks import MLP, CNN, RNN, RCNN
from diffaero.dynamics.pointmass import point_mass_quat
from diffaero.utils.logger import Logger

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
            ("state", torch.zeros(1, state_dim)),
            ("orientation", torch.zeros(1, 3)),
            ("Rz", torch.zeros(1, 3, 3)),
            ("min_action", torch.zeros(1, 3)),
            ("max_action", torch.zeros(1, 3)),
        ]
        if perception_dim is not None:
            if isinstance(self.actor, (MLP, RNN)):
                self.named_inputs[0] = ("state", (torch.zeros(1, state_dim), torch.zeros(1, perception_dim[0], perception_dim[1])))
            elif isinstance(self.actor, (CNN, RCNN)):
                self.named_inputs.insert(1, ("perception", torch.zeros(1, perception_dim[0], perception_dim[1])))
        self.output_names = [
            "action",
            "quat_xyzw_cmd",
            "acc_norm"
        ]
        if self.is_recurrent:
            self.named_inputs.append(("hidden_in", torch.zeros(self.hidden_shape)))
            self.output_names.append("hidden_out")
        
        self.obs_frame: str
        self.action_frame: str
    
    def post_process_local(self, raw_action, min_action, max_action, orientation, Rz, is_stochastic):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, bool) -> Tuple[Tensor, Tensor, Tensor]
        raw_action = raw_action.tanh() if is_stochastic else raw_action
        action = (raw_action * 0.5 + 0.5) * (max_action - min_action) + min_action
        acc_cmd = torch.matmul(Rz, action.unsqueeze(-1)).squeeze(-1)
        quat_xyzw = point_mass_quat(acc_cmd, orientation)
        acc_norm = acc_cmd.norm(p=2, dim=-1)
        return acc_cmd, quat_xyzw, acc_norm
    
    def post_process_world(self, raw_action, min_action, max_action, orientation, Rz, is_stochastic):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, bool) -> Tuple[Tensor, Tensor, Tensor]
        raw_action = raw_action.tanh() if is_stochastic else raw_action
        action = (raw_action * 0.5 + 0.5) * (max_action - min_action) + min_action
        quat_xyzw = point_mass_quat(action, orientation)
        acc_norm = action.norm(p=2, dim=-1)
        return action, quat_xyzw, acc_norm

    def post_process(self, raw_action, min_action, max_action, orientation, Rz):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        if self.action_frame == "local":
            return self.post_process_local(raw_action, min_action, max_action, orientation, Rz, self.is_stochastic)
        elif self.action_frame == "world":
            return self.post_process_world(raw_action, min_action, max_action, orientation, Rz, self.is_stochastic)
        else:
            raise ValueError(f"Unknown action frame: {self.action_frame}")
    
    def forward_MLP(self, state, orientation, Rz, min_action, max_action):
        # type: (Union[Tensor, Tuple[Tensor, Tensor]], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        raw_action = self.actor.forward_export(state)
        action, quat_xyzw, acc_norm = self.post_process(raw_action, min_action, max_action, orientation, Rz)
        return action, quat_xyzw, acc_norm
    
    def forward_CNN(self, state, perception, orientation, Rz, min_action, max_action):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        raw_action = self.actor.forward_export(state=state, perception=perception)
        action, quat_xyzw, acc_norm = self.post_process(raw_action, min_action, max_action, orientation, Rz)
        return action, quat_xyzw, acc_norm
    
    def forward_RNN(self, state, orientation, Rz, min_action, max_action, hidden_in):
        # type: (Union[Tensor, Tuple[Tensor, Tensor]], Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        raw_action, hidden_out = self.actor.forward_export(state, hidden=hidden_in)
        action, quat_xyzw, acc_norm = self.post_process(raw_action, min_action, max_action, orientation, Rz)
        return action, quat_xyzw, acc_norm, hidden_out
    
    def forward_RCNN(self, state, perception, orientation, Rz, min_action, max_action, hidden_in):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        raw_action, hidden_out = self.actor.forward_export(state=state, perception=perception, hidden=hidden_in)
        action, quat_xyzw, acc_norm = self.post_process(raw_action, min_action, max_action, orientation, Rz)
        return action, quat_xyzw, acc_norm, hidden_out
    
    def export(
        self,
        path: str,
        export_cfg: DictConfig,
        verbose=False,
    ):
        self.obs_frame = export_cfg.obs_frame
        self.action_frame = export_cfg.action_frame
        if export_cfg.jit:
            self.export_jit(path, verbose)
        if export_cfg.onnx:
            self.export_onnx(path)
    
    @torch.no_grad()
    def export_jit(self, path: str, verbose=False):
        traced_script_module = torch.jit.script(self)
        if verbose:
            Logger.info("Code of scripted module: \n" + traced_script_module.code)
        export_path = os.path.join(path, "exported_actor.pt2")
        traced_script_module.save(export_path)
        Logger.info(f"The checkpoint is compiled and exported to {export_path}.")
    
    @torch.no_grad()
    def export_onnx(self, path: str):
        export_path = os.path.join(path, "exported_actor.onnx")
        names, test_inputs = zip(*self.named_inputs)
        print(names)
        torch.onnx.export(
            model=self,
            args=test_inputs,
            f=export_path,
            input_names=names,
            output_names=self.output_names
        )
        Logger.info(f"The checkpoint is compiled and exported to {export_path}.")
        
        self.eval()
        ort_session = ort.InferenceSession(export_path)
        test_inputs = [torch.randn_like(input, device="cpu") for input in test_inputs]
        ort_inputs = {name: input.numpy() for name, input in zip(names, test_inputs)}
        ort_outs: Tuple[np.ndarray, ...] = ort_session.run(None, ort_inputs) # type: ignore
        torch_outs: Tuple[Tensor, ...] = self(*test_inputs)
        # compare ONNX Runtime and PyTorch results
        for i in range(len(ort_outs)):
            np.testing.assert_allclose(ort_outs[i], torch_outs[i].cpu().numpy(), rtol=1e-03, atol=1e-05, verbose=True)
        Logger.info(f"The onnx model at {export_path} is verified!")