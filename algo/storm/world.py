import sys
import pathlib
from collections import deque
folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
sys.path.insert(2, str(folder.parent.parent.parent))

import torch
import torch.nn as nn

from diffaero.algo.dreamerv3.wmenv.replaybuffer import ReplayBuffer
from diffaero.algo.dreamerv3.models.agent import ActorCriticAgent, ActorCriticConfig
from diffaero.algo.dreamerv3.models.blocks import symlog
from sub_models.functions_losses import symexp
from sub_models.world_models import WorldModel, MSELoss

@torch.no_grad()
def world_model_imagine_data(replay_buffer: ReplayBuffer,
                             world_model: WorldModel, agent: ActorCriticAgent,
                             imagine_batch_size, imagine_demonstration_batch_size,
                             imagine_context_length, imagine_batch_length,
                             log_video, logger):
    '''
    Sample context from replay buffer, then imagine data with world model and agent
    '''
    world_model.eval()
    agent.eval()

    sample_obs, sample_action, sample_reward, sample_termination, sample_perception = replay_buffer.sample(
        imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length)
    sample_obs_dict = dict(perception=sample_perception, state=sample_obs)
    latent, action, reward_hat, termination_hat = world_model.imagine_data(
        agent, sample_obs_dict, sample_action,
        imagine_batch_size=imagine_batch_size+imagine_demonstration_batch_size,
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
        logger=logger
    )
    return latent, action, None, None, reward_hat, termination_hat

class STORM:
    def __init__(self, cfg, device):
        storm_cfg = cfg.algo
        device_idx = f"{cfg.device}"
        device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
        self.world_model = WorldModel(in_channels=storm_cfg.wm.inchannels, action_dim=3, 
                                      transformer_max_length=storm_cfg.wm.transformer_max_length,
                                      transformer_hidden_dim=storm_cfg.wm.transformer_hidden_dim, 
                                      transformer_num_heads=storm_cfg.wm.transformer_num_heads).to(device)
        agent_cfg = ActorCriticConfig(feat_dim=storm_cfg.wm.transformer_hidden_dim+self.world_model.stoch_flattened_dim,
                                      num_layers=storm_cfg.agent.num_layers, hidden_dim=storm_cfg.agent.hidden_dim,
                                      action_dim=3, gamma=storm_cfg.agent.gamma, lambd=storm_cfg.agent.lambd,
                                      entropy_coef=storm_cfg.agent.entropy_coef, device=device)
        self.agent = ActorCriticAgent(agent_cfg, None).to(device)
        self.device = device
        self.context_obs = deque(maxlen=16)
        self.context_action = deque(maxlen=16)
    
    @torch.no_grad()
    def act(self, obs:torch.Tensor, test=False):
        if type(obs)!=torch.Tensor:
            state,perception = obs['state'],obs['perception'].unsqueeze(1)
        else:
            state,perception = obs,None
        if self.world_agent_cfg.common.use_symlog:
            state = symlog(state)   
        latent = self.state_model.sample_with_post(state,perception,self.hidden,True)[0].flatten(1)
        action = self.agent.sample(torch.cat([latent,self.hidden],dim=-1),test)[0]
        self.hidden = self.state_model.sample_with_prior(latent,action,self.hidden,True)[2]
        return action,None
