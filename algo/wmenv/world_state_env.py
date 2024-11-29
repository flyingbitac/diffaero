from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Tuple,Optional

import torch
from torch import Tensor

from drone_gym.models.state_predictor import StateModel,StateModelCfg
from drone_gym.models.blocks import symexp,symlog
from .replaybuffer import ReplayBuffer

from drone_gym.utils.cfg import Logger
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
        state_model: StateModel,
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
        states, actions, _ , _, _, perceptions = self.replaybuffer.sample(batch_size, batch_length)
        hidden = None

        if self.cfg.use_perception:
            feats = perceptions
        else:
            feats = states
            
        for i in range(batch_length):
            latent,_ = self.state_model.sample_with_post(feats[:,i],hidden)
            latent = self.state_model.flatten(latent)
            latent,_,hidden=self.state_model.sample_with_prior(latent,actions[:,i],hidden)

        latent = self.state_model.flatten(latent)
        self.latent = latent
        self.hidden = hidden
        self.drone_state = self.state_model.decode(latent,hidden)
        
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
