from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Tuple,Optional

import torch
from torch import Tensor

from diffaero.algo.dreamerv3.models.state_predictor import DepthStateModel
from diffaero.algo.dreamerv3.models.blocks import symexp,symlog
from .replaybuffer import ReplayBuffer
from diffaero.utils.logger import Logger
from diffaero.utils.runner import timeit
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
    use_extern: bool = False

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
        self.use_extern = cfg.use_extern

    @torch.no_grad()
    @timeit
    def make_generator_init(self,):
        batch_size = self.cfg.batch_size
        batch_length = self.cfg.batch_length
        if self.use_extern:
            states, actions, perceptions = self.replaybuffer.sample_extern(batch_size, batch_length)
        else:
            states, actions, _ , _, perceptions, _, _ = self.replaybuffer.sample(batch_size, batch_length)
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
        return latent, hidden
        
    @torch.no_grad()
    @timeit
    def step(self,action:Tensor):
        assert action.ndim==2
        prior_sample,pred_reward,pred_end,hidden,grid = self.state_model.predict_next(latent=self.latent, act=action, hidden=self.hidden)
        flattened_sample = prior_sample.view(*prior_sample.shape[:-2],-1)
        self.latent = flattened_sample
        self.hidden = hidden
        return flattened_sample,pred_reward,pred_end,hidden,grid
    
    def decode(self, latents:Tensor, hiddens:Tensor):
        _, videos = self.state_model.decode(latents, hiddens)
        assert videos.ndim == 4, f"Expected videos to have 4 dimensions, got {videos.ndim}"
        return videos
