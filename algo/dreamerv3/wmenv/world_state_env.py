from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Tuple,Optional

import torch
from torch import Tensor

from quaddif.algo.dreamerv3.models.state_predictor import DepthStateModel
from quaddif.algo.dreamerv3.models.blocks import symexp,symlog
from .replaybuffer import ReplayBuffer
from quaddif.utils.logger import Logger
# from models.rew_end_model import RewEndModel

ResetOutput = Tuple[torch.FloatTensor, Dict[str, Any]]
StepOutput = Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]
InitialCondition = Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]


@dataclass
class DepthStateEnvConfig:
    horizon: int
    batch_size: int
    batch_length: int
    use_perception: bool = False

class DepthStateEnv:
    def __init__(
        self,
        state_model: DepthStateModel,
        replaybuffer: ReplayBuffer,
        cfg: DepthStateEnvConfig,
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
            if perceptions != None:
                latent,_ = self.state_model.sample_with_post(states[:,i],perceptions[:,i],hidden)
            else:
                latent,_ = self.state_model.sample_with_post(states[:,i],None,hidden)
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
