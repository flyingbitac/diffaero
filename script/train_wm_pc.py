import os
import sys
sys.path.append('..')
import random

import isaacgym
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
from line_profiler import LineProfiler
from tqdm import tqdm
import cv2

from quaddif.algo.dreamerv3.models.state_predictor import StateModel
from quaddif.algo.dreamerv3.models.agent import ActorCriticAgent
from quaddif.algo.dreamerv3.models.blocks import symlog
from quaddif.algo.dreamerv3.wmenv.world_state_env import StateEnv
from quaddif.algo.dreamerv3.wmenv.replaybuffer import ReplayBuffer
from quaddif.algo.dreamerv3.wmenv.utils import configure_opt
from quaddif.env import PositionControl
from quaddif.utils.device import idle_device
from quaddif.utils.logger import Logger,RecordEpisodeStatistics

@torch.no_grad()
def collect_imagine_trj(env:StateEnv,agent:ActorCriticAgent,cfg:DictConfig):
    feats = []
    rewards = []
    ends = []
    actions = []
    states = []
    org_samples = []
    imagine_length = cfg.imagine_length
    latent,hidden,state = env.make_generator_init()
    for i in range(imagine_length):
        feat = torch.cat([latent,hidden],dim=-1)
        feats.append(feat)
        states.append(state)
        action,org_sample = agent.sample(feat)
        _,latent,reward,end,hidden = env.step(action)
        rewards.append(reward)
        actions.append(action)
        org_samples.append(org_sample)
        ends.append(end)
    feat = torch.cat([latent,hidden],dim=-1)
    feats.append(feat)
    states.append(state)
    feats = torch.stack(feats,dim=1)
    actions = torch.stack(actions,dim=1)
    org_samples = torch.stack(org_samples,dim=1)
    rewards = torch.stack(rewards,dim=1)
    ends = torch.stack(ends,dim=1)

    return feats,actions,rewards,ends,states,org_samples

def train_agents(agent:ActorCriticAgent,state_env:StateEnv,cfg:DictConfig,logger:Logger):
    trainingcfg = getattr(cfg,"actor_critic").training
    feats,actions,rewards,ends,states,org_samples = collect_imagine_trj(state_env,agent,trainingcfg)
    # rewards = rewards/100.
    agent.update(feats,org_samples,rewards,ends,logger)
    reward_sum = rewards.sum(dim=-1).mean()
    logger.log('ActorCritic/avg_return', reward_sum.item())
    
def train_worldmodel(world_model:StateModel,replaybuffer:ReplayBuffer,opt,training_hyper,logger:Logger):
    for _ in range(training_hyper.worldmodel_update_freq):
        sample_state, sample_action, sample_reward, sample_termination,sample_reward_components,_ = replaybuffer.sample(training_hyper.batch_size,training_hyper.batch_length)
        if not training_hyper.use_multirew:
            total_loss,rep_loss,dyn_loss,rec_loss,rew_loss,end_loss = world_model.compute_loss(sample_state, sample_action, sample_reward, sample_termination)
        else:
            total_loss,rep_loss,dyn_loss,rec_loss,rew_loss,end_loss = world_model.compute_loss(sample_state, sample_action, sample_reward_components, sample_termination)
    
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(world_model.parameters(),training_hyper.max_grad_norm)
    opt.step()
    opt.zero_grad()

    logger.log("WorldModel/state_total_loss",total_loss.item())
    logger.log("WorldModel/state_rep_loss",rep_loss.item())
    logger.log("WorldModel/state_dyn_loss",dyn_loss.item())
    logger.log("WorldModel/state_rec_loss",rec_loss.item())
    logger.log("WorldModel/grad_norm",grad_norm.item())
    logger.log("WorldModel/state_rew_loss",rew_loss.item())
    logger.log("WorldModel/state_end_loss",end_loss.item())

@hydra.main(config_path="../cfg", config_name="config")
def main(cfg: DictConfig):
    device_idx = f"{idle_device()}" if cfg.device is None else f"{cfg.device}"
    print("Using device", device_idx)
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")

    env = RecordEpisodeStatistics(PositionControl(cfg.env, cfg.dynamics, device=device))
    
    world_agent_cfg = getattr(cfg,"algo")
    world_agent_cfg.replaybuffer.device = f"cuda:{device_idx}"
    world_agent_cfg.replaybuffer.num_envs = cfg.n_envs
    world_agent_cfg.replaybuffer.state_dim = 13
    world_agent_cfg.actor_critic.model.device = f"cuda:{device_idx}"
    world_agent_cfg.common.device = f"cuda:{device_idx}"
    
    statemodelcfg = getattr(world_agent_cfg,"state_predictor").state_model
    statemodelcfg.state_dim = 13
    actorcriticcfg = getattr(world_agent_cfg,"actor_critic").model
    buffercfg = getattr(world_agent_cfg,"replaybuffer")
    worldcfg = getattr(world_agent_cfg,"world_state_env")
    training_hyper = getattr(world_agent_cfg,"state_predictor").training
    training_hyper.use_multirew = statemodelcfg.use_multirew
    logger = Logger(cfg)
    
    agent = ActorCriticAgent(actorcriticcfg,env).to(device)
    state_model = StateModel(statemodelcfg).to(device)
    replaybuffer = ReplayBuffer(buffercfg)
    world_model_env = StateEnv(state_model,replaybuffer,worldcfg)
    opt = configure_opt(state_model,**getattr(world_agent_cfg,'state_predictor').optimizer)
    
    print(agent)
    print(state_model)

    if cfg.seed != -1:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = cfg.torch_determinstic
    
    obs = env.reset()
    
    hidden = torch.zeros(cfg.n_envs, statemodelcfg.hidden_dim,device=device)
    os.mkdir(os.path.join(logger.logdir,"transitions"))
    os.makedirs(f"./ckpt/{world_agent_cfg.common.run_name}/statemodel",exist_ok=True)
    os.makedirs(f"./ckpt/{world_agent_cfg.common.run_name}/actorcritic",exist_ok=True)
    global_step = 0
    for i in tqdm(range(world_agent_cfg.common.total_timesteps//cfg.n_envs),ncols=200):
        with torch.no_grad():
            if world_agent_cfg.common.use_symlog:
                obs = symlog(obs)
            if replaybuffer.ready() or world_agent_cfg.common.use_checkpoint:
                latent = state_model.sample_with_post(obs,hidden)[0].flatten(1)
                action,org_sample = agent.sample(torch.cat([latent, hidden],dim=-1))
                next_obs, rewards, terminated, info = env.step(action)
                prior_sample,_,hidden = state_model.sample_with_prior(latent,action,hidden)
                rewards = 1 - rewards * 0.1
                replaybuffer.append(obs,action,rewards,terminated)
            else:
                action = torch.randn(cfg.n_envs,3,device=device)
                next_obs, rewards, terminated, info = env.step(action)
                rewards = 1 - rewards * 0.1
                replaybuffer.append(obs,action,rewards,terminated)
                
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
                        hidden[i] = 0
        
        if replaybuffer.ready():
            train_worldmodel(state_model,replaybuffer,opt,training_hyper,logger)
            train_agents(agent,world_model_env,world_agent_cfg,logger)
        
        if i % 10000 == 0:
            torch.save(state_model.state_dict(), f"./ckpt/{world_agent_cfg.common.run_name}/statemodel/statemodel_{i}.pth")
            torch.save(agent.state_dict(), f"./ckpt/{world_agent_cfg.common.run_name}/actorcritic/actorcritic_{i}.pth")
        obs = next_obs
    
        if env.renderer is not None:
            env.renderer.close()
    
if __name__ == "__main__":
    main()