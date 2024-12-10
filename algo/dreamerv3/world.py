from quaddif.algo.dreamerv3.models.state_predictor import StateModel
from quaddif.algo.dreamerv3.models.agent import ActorCriticAgent
from quaddif.algo.dreamerv3.models.blocks import symlog
from quaddif.algo.dreamerv3.wmenv.world_state_env import StateEnv
from quaddif.algo.dreamerv3.wmenv.replaybuffer import ReplayBuffer
from quaddif.algo.dreamerv3.wmenv.utils import configure_opt
import os

import torch

@torch.no_grad()
def collect_imagine_trj(env:StateEnv,agent:ActorCriticAgent,cfg):
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

def train_agents(agent:ActorCriticAgent,state_env:StateEnv,cfg):
    trainingcfg = getattr(cfg,"actor_critic").training
    feats,actions,rewards,ends,states,org_samples = collect_imagine_trj(state_env,agent,trainingcfg)
    # rewards = rewards/100.
    agent_info = agent.update(feats,org_samples,rewards,ends)
    reward_sum = rewards.sum(dim=-1).mean()
    agent_info['reward_sum'] = reward_sum.item()
    # logger.log('ActorCritic/avg_return', reward_sum.item())
    return agent_info
    
def train_worldmodel(world_model:StateModel,replaybuffer:ReplayBuffer,opt,training_hyper):
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
    
    world_info = {
        'WorldModel/state_total_loss':total_loss.item(),
        'WorldModel/state_rep_loss':rep_loss.item(),
        'WorldModel/state_dyn_loss':dyn_loss.item(),
        'WorldModel/state_rec_loss':rec_loss.item(),
        'WorldModel/grad_norm':grad_norm.item(),
        'WorldModel/state_rew_loss':rew_loss.item(),
        'WorldModel/state_end_loss':end_loss.item()
    }
    
    return world_info

class World_Agent:
    def __init__(self,cfg,env,device):
        device_idx = f"{cfg.device}"
        device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
        world_agent_cfg = getattr(cfg,"algo")
        world_agent_cfg.replaybuffer.device = f"cuda:{device_idx}"
        world_agent_cfg.replaybuffer.num_envs = cfg.n_envs
        world_agent_cfg.replaybuffer.state_dim = 13
        world_agent_cfg.actor_critic.model.device = f"cuda:{device_idx}"
        world_agent_cfg.common.device = f"cuda:{device_idx}"
        self.cfg = cfg
        self.world_agent_cfg = world_agent_cfg
        
        statemodelcfg = getattr(world_agent_cfg,"state_predictor").state_model
        statemodelcfg.state_dim = 157
        statemodelcfg.hidden_dim = 512
        statemodelcfg.latent_dim = 1024
        actorcriticcfg = getattr(world_agent_cfg,"actor_critic").model
        buffercfg = getattr(world_agent_cfg,"replaybuffer")
        buffercfg.state_dim = 157
        worldcfg = getattr(world_agent_cfg,"world_state_env")
        training_hyper = getattr(world_agent_cfg,"state_predictor").training
        training_hyper.use_multirew = statemodelcfg.use_multirew
        self.training_hyper = training_hyper
        
        self.agent = ActorCriticAgent(actorcriticcfg,env).to(device)
        self.state_model = StateModel(statemodelcfg).to(device)
        self.replaybuffer = ReplayBuffer(buffercfg)
        self.world_model_env = StateEnv(self.state_model,self.replaybuffer,worldcfg)
        self.opt = configure_opt(self.state_model,**getattr(world_agent_cfg,'state_predictor').optimizer)

        self.hidden = torch.zeros(cfg.n_envs, statemodelcfg.hidden_dim, device=device)
    
    @torch.no_grad()
    def act(self,obs,test=False):
        state,perception = obs['state'],obs['perception'].flatten(1)
        state = torch.cat([state,perception],dim=-1)
        if self.world_agent_cfg.common.use_symlog:
            state = symlog(state)   
        latent = self.state_model.sample_with_post(state,self.hidden)[0].flatten(1)
        action = self.agent.sample(torch.cat([latent,self.hidden],dim=-1))[0]
        self.hidden = self.state_model.sample_with_prior(latent,action,self.hidden)[2]
        return action,None

    def step(self,cfg,env,obs,on_step_cb=None):
        policy_info = {}
        with torch.no_grad():
            state,perception = obs['state'],obs['perception'].flatten(1)
            state = torch.cat([state,perception],dim=-1)
            if self.world_agent_cfg.common.use_symlog:
                state = symlog(state)
            if self.replaybuffer.ready() or self.world_agent_cfg.common.use_checkpoint:
                latent = self.state_model.sample_with_post(state,self.hidden)[0].flatten(1)
                action = self.agent.sample(torch.cat([latent,self.hidden],dim=-1))[0]
                self.hidden = self.state_model.sample_with_prior(latent,action,self.hidden)[2]
            else:
                action = torch.randn(self.cfg.n_envs,3,device=state.device)
            next_obs,rewards,terminated,env_info = env.step(action)
            rewards = 10.*(1-rewards*0.1)
            self.replaybuffer.append(state,action,rewards,terminated)
            
            if terminated.any():
                for i in range(cfg.n_envs):
                    if terminated[i]:
                        self.hidden[i] = 0
            
        if self.replaybuffer.ready():
            world_info = train_worldmodel(self.state_model,self.replaybuffer,self.opt,self.training_hyper)
            agent_info = train_agents(self.agent,self.world_model_env,self.world_agent_cfg)
            policy_info.update(world_info)
            policy_info.update(agent_info)
        
        obs = next_obs
            
        return obs,policy_info, env_info, 0.0, 0.0

    def save(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_model.state_dict(), f"{path}/statemodel.pth")
        torch.save(self.agent.state_dict(), f"{path}/agent.pth")
        
    def load(self,path):
        self.state_model.load_state_dict(torch.load(os.path.join(path, "statemodel.pth")))
        self.agent.load_state_dict(torch.load(os.path.join(path, "agent.pth")))
    
    @staticmethod
    def build(cfg,env,device):
        return World_Agent(cfg,env,device)