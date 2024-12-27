from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Tuple,Optional

import torch
from torch import Tensor

from quaddif.algo.dreamerv3.models.state_predictor import DepthStateModel,PercStateModel
from quaddif.algo.dreamerv3.models.blocks import symexp,symlog
from .replaybuffer import ReplayBuffer
from quaddif.utils.logger import Logger
# from models.rew_end_model import RewEndModel

ResetOutput = Tuple[torch.FloatTensor, Dict[str, Any]]
StepOutput = Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]
InitialCondition = Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]


@dataclass
class StateEnvConfig:
    horizon: int
    batch_size: int
    batch_length: int
    use_perception: bool = False

class StateEnv:
    def __init__(
        self,
        state_model: DepthStateModel,
        replaybuffer: ReplayBuffer,
        cfg: StateEnvConfig,
    ) -> None:
        self.state_model = state_model
        self.replaybuffer = replaybuffer
        self.cfg = cfg
        self.hidden = None

    @property
    def device(self) -> torch.device:
        return self.sampler.denoiser.device

    @torch.no_grad()
    def make_generator_init(self,):
        batch_size = self.cfg.batch_size
        batch_length = self.cfg.batch_length
        states, actions, _ , _, perceptions = self.replaybuffer.sample(batch_size, batch_length)
        hidden = None
            
        for i in range(batch_length):
            latent,_ = self.state_model.sample_with_post(states[:,i],perceptions[:,i],hidden)
            latent = self.state_model.flatten(latent)
            latent,_,hidden=self.state_model.sample_with_prior(latent,actions[:,i],hidden)

        latent = self.state_model.flatten(latent)
        self.latent = latent
        self.hidden = hidden
        self.drone_state,self.drone_perception = self.state_model.decode(latent,hidden)
        
        return latent,hidden,self.drone_state
        
    @torch.no_grad()
    def step(self,action:Tensor):
        assert action.ndim==2
        next_state,prior_sample,pred_reward,pred_end,hidden=self.state_model.predict_next(latent=self.latent,
                                                                                          act=action,
                                                                                          hidden=self.hidden)
        flattened_sample = prior_sample.view(*prior_sample.shape[:-2],-1)
        self.latent = flattened_sample
        self.drone_state = next_state
        self.hidden = hidden
        return next_state,flattened_sample,pred_reward,pred_end,hidden

class PercStateEnv:
    def __init__(
        self,
        perc_state_model: PercStateModel,
        replaybuffer: ReplayBuffer,
        cfg: StateEnvConfig,
    ) -> None:
        self.perc_state_model = perc_state_model
        self.replaybuffer = replaybuffer
        self.cfg = cfg
        self.hidden = None

    @property
    def device(self) -> torch.device:
        return self.sampler.denoiser.device

    @torch.no_grad()
    def make_generator_init(self,):
        batch_size = self.cfg.batch_size
        batch_length = self.cfg.batch_length
        states, actions, _ , _, perceptions = self.replaybuffer.sample(batch_size, batch_length)
        state_hidden = None
        perc_hidden = None
            
        for i in range(batch_length):
            state_latent,_,perc_latent,_ = self.perc_state_model.sample_with_post(states[:,i],perceptions[:,i],state_hidden,perc_hidden)
            state_latent = self.perc_state_model.flatten(state_latent)
            perc_latent = self.perc_state_model.flatten(perc_latent)
            state_latent,_,state_hidden,perc_latent,_,perc_hidden=self.perc_state_model.sample_with_prior(
                                        state_latent,perc_latent,actions[:,i],state_hidden,perc_hidden)

        state_latent = self.perc_state_model.flatten(state_latent)
        perc_latent = self.perc_state_model.flatten(perc_latent)
        self.state_latent = state_latent
        self.state_hidden = state_hidden
        self.perc_latent = perc_latent
        self.perc_hidden = perc_hidden
        
        self.drone_state = self.perc_state_model.decode(state_latent,perc_latent,state_hidden,perc_hidden)
        return state_latent,state_hidden,perc_latent,perc_hidden,self.drone_state
        
    @torch.no_grad()
    def step(self,action:Tensor):
        assert action.ndim==2
        state_prior_sample,perc_prior_sample,pred_reward,pred_end,state_hidden,perc_hidden= \
            self.perc_state_model.predict_next(
                state_latent=self.state_latent,
                perc_latent=self.perc_latent,
                act=action,
                state_hidden=self.state_hidden,
                perc_hidden=self.perc_hidden,
            )
            
        state_flattened_sample = state_prior_sample.view(*state_prior_sample.shape[:-2],-1)
        self.state_latent = state_flattened_sample
        perc_flattened_sample = perc_prior_sample.view(*perc_prior_sample.shape[:-2],-1)
        self.perc_latent = perc_flattened_sample
        self.state_hidden = state_hidden
        
        return state_hidden,perc_hidden,pred_reward,pred_end