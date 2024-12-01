import os
import sys
sys.path.append('..')
import random

import isaacgym
from quaddif.algo.models.state_predictor import StateModel,PercStateModel
from quaddif.algo.models.agent import ActorCriticAgent
from quaddif.algo.models.blocks import symlog
from quaddif.algo.wmenv.world_state_env import PercStateEnv
from quaddif.algo.wmenv.replaybuffer import ReplayBuffer
from quaddif.algo.wmenv.utils import configure_opt

import isaacgym
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
from line_profiler import LineProfiler
from tqdm import tqdm
import cv2

from quaddif.env import PositionControl, ObstacleAvoidance
from quaddif.algo import SHAC, APG_stochastic, APG, PPO
from quaddif.utils.env import RecordEpisodeStatistics
from quaddif.utils.device import idle_device
from quaddif.utils.logger import Logger

@torch.no_grad()
def collect_imagine_trj(env:PercStateEnv,agent:ActorCriticAgent,cfg:DictConfig):
    feats = []
    rewards = []
    actions = []
    ends = []
    org_samples = []
    imagine_length = cfg.imagine_length
    state_latent,state_hidden,perc_latent,perc_hidden,_ = env.make_generator_init()
    for i in range(imagine_length):
        feat = torch.cat([state_hidden,perc_hidden],dim=-1)
        feats.append(feat)
        action,org_sample = agent.sample(feat)
        state_hidden,perc_hidden,reward,end = env.step(action)
        rewards.append(reward)
        actions.append(action)
        org_samples.append(org_sample)
        ends.append(end)
    feat = torch.cat([state_hidden,perc_hidden],dim=-1)
    feats.append(feat)
    feats = torch.stack(feats,dim=1)
    actions = torch.stack(actions,dim=1)
    org_samples = torch.stack(org_samples,dim=1)
    rewards = torch.stack(rewards,dim=1)
    ends = torch.stack(ends,dim=1)

    return feats,actions,rewards,ends,org_samples

def train_agents(agent:ActorCriticAgent,state_env:PercStateEnv,cfg:DictConfig,logger:Logger):
    trainingcfg = getattr(cfg,"actor_critic").training
    feats,actions,rewards,ends,org_samples = collect_imagine_trj(state_env,agent,trainingcfg)
    agent.update(feats,org_samples,rewards,ends,logger)
    reward_sum = rewards.sum(dim=-1).mean()
    logger.log('ActorCritic/avg_return', reward_sum.item())
    
def train_worldmodel(world_model:PercStateModel,replaybuffer:ReplayBuffer,opt,training_hyper,logger:Logger):
    for _ in range(training_hyper.worldmodel_update_freq):
        sample_state, sample_action, sample_reward, sample_termination,sample_reward_components,sample_perc = \
                                        replaybuffer.sample(training_hyper.batch_size,training_hyper.batch_length)
        if not training_hyper.use_multirew:
            total_loss, state_rep_loss, state_dyn_loss, state_rec_loss, perc_rep_loss, perc_dyn_loss, \
            perc_rec_loss, rew_loss, end_loss = world_model.compute_loss(sample_state, sample_perc, sample_action, sample_reward, sample_termination)
        else:
            assert 0==1
    
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(world_model.parameters(),training_hyper.max_grad_norm)
    opt.step()
    opt.zero_grad()

    logger.log("WorldModel/total_loss",total_loss.item())
    logger.log("WorldModel/state_rep_loss",state_rep_loss.item())
    logger.log("WorldModel/state_dyn_loss",state_dyn_loss.item())
    logger.log("WorldModel/state_rec_loss",state_rec_loss.item())
    logger.log("WorldModel/perc_rep_loss",perc_rep_loss.item())
    logger.log("WorldModel/perc_dyn_loss",perc_dyn_loss.item())
    logger.log("WorldModel/perc_rec_loss",perc_rec_loss.item())
    logger.log("WorldModel/grad_norm",grad_norm.item())
    logger.log("WorldModel/rew_loss",rew_loss.item())
    logger.log("WorldModel/end_loss",end_loss.item())

###

@hydra.main(config_path="../cfg", config_name="config")
def main(cfg: DictConfig):
    device_idx = f"{idle_device()}" if cfg.device is None else f"{cfg.device}"
    print("Using device", device_idx)
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    env = RecordEpisodeStatistics(ObstacleAvoidance(cfg.env, cfg.dynamics, device=device))

    world_agent_cfg = getattr(cfg,"algo")
    world_agent_cfg.replaybuffer.device = f"cuda:{device_idx}"
    world_agent_cfg.replaybuffer.num_envs = cfg.n_envs
    world_agent_cfg.replaybuffer.state_dim = 13
    world_agent_cfg.actor_critic.model.device = f"cuda:{device_idx}"
    world_agent_cfg.common.device = f"cuda:{device_idx}"
    
    statemodelcfg = getattr(world_agent_cfg,"state_predictor").state_model
    statemodelcfg.state_dim = 13
    percmodelcfg = getattr(world_agent_cfg,"state_predictor").perception_model
    actorcriticcfg = getattr(world_agent_cfg,"actor_critic").model
    actorcriticcfg.feat_dim = percmodelcfg.hidden_dim + statemodelcfg.hidden_dim
    actorcriticcfg.hidden_dim = percmodelcfg.hidden_dim
    
    buffercfg = getattr(world_agent_cfg,"replaybuffer")
    buffercfg.use_perception = True
    worldcfg = getattr(world_agent_cfg,"world_state_env")
    training_hyper = getattr(world_agent_cfg,"state_predictor").training
    training_hyper.use_multirew = statemodelcfg.use_multirew
    logger = Logger(cfg)
    
    agent = ActorCriticAgent(actorcriticcfg,env).to(device)
    perc_state_model = PercStateModel(statemodelcfg,percmodelcfg).to(device)
    replaybuffer = ReplayBuffer(buffercfg)
    world_model_env = PercStateEnv(perc_state_model,replaybuffer,worldcfg)
    opt = configure_opt(perc_state_model,**getattr(world_agent_cfg,"state_predictor").optimizer)
    
    print(agent)
    print(perc_state_model)

    if cfg.seed != -1:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = cfg.torch_determinstic
    
    obs = env.reset()

    state_hidden = torch.zeros(cfg.n_envs,statemodelcfg.hidden_dim,device=device)
    perc_hidden = torch.zeros(cfg.n_envs,percmodelcfg.hidden_dim,device=device)
    os.mkdir(os.path.join(logger.logdir,"transitions"))
    os.makedirs(f"./ckpt/{world_agent_cfg.common.run_name}/percstatemodel",exist_ok=True)
    os.makedirs(f"./ckpt/{world_agent_cfg.common.run_name}/actorcritic",exist_ok=True)
    global_step = 0
    for i in tqdm(range(world_agent_cfg.common.total_timesteps//cfg.n_envs),ncols=200):
        with torch.no_grad():
            state,perception = obs["state"],obs["perception"].flatten(1)
            if world_agent_cfg.common.use_symlog:
                state,perception = symlog(state),symlog(perception)
            if replaybuffer.ready() or world_agent_cfg.common.use_checkpoint:
                state_latent,_,perc_latent,_ = perc_state_model.sample_with_post(
                    state,perception,state_hidden,perc_hidden)
                state_latent,perc_latent = state_latent.flatten(1),perc_latent.flatten(1)
                action,org_sample = agent.sample(torch.cat([state_hidden,perc_hidden],dim=-1))
                next_obs,reward,terminated,info = env.step(action)
                _,_,state_hidden,_,_,perc_hidden = perc_state_model.sample_with_prior(state_latent,
                                                                  perc_latent,action,state_hidden,perc_hidden)
                reward = 1.0 - reward * 0.1
                replaybuffer.append(state,action,reward,terminated,None,perception)
            else:
                action = torch.randn(cfg.n_envs,3,device=device)
                next_obs, reward, terminated, info = env.step(action)
                reward = 1. - reward * 0.1
                replaybuffer.append(state,action,reward,terminated,None,perception)
            
            global_step += cfg.n_envs
            l_episode = info["stats"]["l"].float().mean().item()
            success_rate = info['stats']['success_rate']
            log_info = {
                "env_loss": info["loss_components"],
                "metrics": {"l_episode": l_episode, "success_rate": success_rate}
            }
            logger.log_scalars(log_info, global_step)
            
            if terminated.any():
                for i in range(cfg.n_envs):
                    if terminated[i]:
                        state_hidden[i] = 0
                        perc_hidden[i] = 0
        
        if replaybuffer.ready():
            train_worldmodel(perc_state_model,replaybuffer,opt,training_hyper,logger)
            train_agents(agent,world_model_env,world_agent_cfg,logger)

        if i % 10000 == 0:
            torch.save(perc_state_model.state_dict(),f"./ckpt/{world_agent_cfg.common.run_name}/percstatemodel/percstatemodel_{i}.pth")
            torch.save(agent.state_dict(),f"./ckpt/{world_agent_cfg.common.run_name}/actorcritic/actorcritic_{i}.pth")
        
        obs = next_obs
        
        if env.renderer is not None:
            env.renderer.close()
    
if __name__ == "__main__":
    main()