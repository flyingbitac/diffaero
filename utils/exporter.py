from typing import Tuple, Dict, Union, Optional, List
from copy import deepcopy
import os
import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from quaddif.network.agents import StochasticActor, DeterministicActor
from quaddif.network.networks import MLP, CNN, RNN, RCNN
from quaddif.dynamics.pointmass import point_mass_quat
from quaddif.algo.dreamerv3.world import World_Agent
from quaddif.algo.dreamerv3.models.state_predictor import onehotsample

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
        verbose=False,
    ):
        if export_jit:
            self.export_jit(path, verbose)
        if export_onnx:
            self.export_onnx(path)
    
    @torch.no_grad()
    def export_jit(self, path: str, verbose=False):
        names, inputs = zip(*self.named_inputs)
        shapes = [tuple(input.shape) for input in inputs]
        traced_script_module = torch.jit.script(self, optimize=True, example_inputs=shapes)
        if verbose:
            print(traced_script_module.code)
        export_path = os.path.join(path, "exported_actor.pt2")
        traced_script_module.save(export_path)
        print(f"The checkpoint is compiled and exported to {export_path}.")
    
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
        print(f"The checkpoint is compiled and exported to {export_path}.")


class WorldExporter(nn.Module):
    def __init__(self, agent: World_Agent):
        super().__init__()
        self.use_symlog = agent.world_agent_cfg.common.use_symlog
        self.state_encoder = deepcopy(agent.state_model.state_encoder)
        if hasattr(agent.state_model, 'image_encoder'):
            self.image_encoder = deepcopy(agent.state_model.image_encoder)
            self.forward = self.forward_perc_prop
        else:
            self.forward = self.forward_prop
        self.inp_proj = deepcopy(agent.state_model.inp_proj)
        self.seq_model = deepcopy(agent.state_model.seq_model)
        self.act_state_proj = deepcopy(agent.state_model.act_state_proj)
        self.actor = deepcopy(agent.agent.actor_mean_std)
        
        self.register_buffer("hidden_state",torch.zeros(1,agent.state_model.cfg.hidden_dim))
        self.hidden_state = self.get_buffer("hidden_state")
        self.is_recurrent = True
        self.hidden_shape = agent.state_model.cfg.hidden_dim
    
    def sample_for_deploy(self,logits):
        probs = F.softmax(logits,dim=-1)
        return onehotsample(probs)
    
    def sample_with_post(self,feat):        
        post_logits = self.inp_proj(torch.cat([feat,self.hidden_state],dim=-1))
        b,d = post_logits.shape
        post_logits = post_logits.reshape(b,int(math.sqrt(d)),-1) # b l d -> b l c k
        post_sample = self.sample_for_deploy(post_logits)
        return post_sample

    def forward_perc_prop(self, state, perception, orientation, min_action, max_action, hidden):
        with torch.no_grad():
            if self.use_symlog:
                state = torch.sign(state) * torch.log(1 + torch.abs(state))
            state_feat = self.state_encoder(state)
            image_feat = self.image_encoder(perception.unsqueeze(0))
            feat = torch.cat([state_feat, image_feat], dim=-1)
            latent = self.sample_with_post(feat).flatten(1)
            mean_std = self.actor(torch.cat([latent,self.hidden_state],dim=-1))
            action, _ = torch.chunk(mean_std, 2, dim=-1)
            action = torch.tanh(action)
            self.sample_with_prior(latent, action)
            action, quat_xyzw, acc_norm = self.post_process(action, min_action, max_action, orientation)
        return action, quat_xyzw, acc_norm, hidden
            
    def forward_prop(self,state, orientation, min_action, max_action, hidden):
        with torch.no_grad():
            if self.use_symlog:
                state = torch.sign(state) * torch.log(1 + torch.abs(state))
            state_feat = self.state_encoder(state.unsqueeze(0))
            latent = self.sample_with_post(state_feat).flatten(1)
            mean_std = self.actor(torch.cat([latent,self.hidden_state],dim=-1))
            action, _ = torch.chunk(mean_std, 2, dim=-1)
            action = torch.tanh(action)
            self.sample_with_prior(latent,action)
            action, quat_xyzw, acc_norm = self.post_process(action, min_action, max_action, orientation)
        return action, quat_xyzw, acc_norm, hidden  
    
    def post_process(self, action, min_action, max_action, orientation):
        action = (action * 0.5 + 0.5) * (max_action - min_action) + min_action
        quat_xyzw = point_mass_quat(action, orientation)
        acc_norm = action.norm(p=2, dim=-1)
        return action, quat_xyzw, acc_norm
    
    def export_jit(self, path: str, verbose=False):
        traced_script_module = torch.jit.script(self)
        if verbose:
            print(traced_script_module.code)
        export_path = os.path.join(path, "exported_actor.pt2")
        traced_script_module.save(export_path)
        print(f"The checkpoint is compiled and exported to {export_path}.")
    
    def export(self, path: str, verbose=False, export_onnx=False, export_pnnx=False):
        self.export_jit(path, verbose)
