import os
from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from quaddif.algo.dreamerv3.models.state_predictor import DepthStateModel,onehotsample
from quaddif.algo.dreamerv3.models.agent import ActorCriticAgent
from quaddif.algo.dreamerv3.models.blocks import symlog
from quaddif.algo.dreamerv3.wmenv.world_state_env import DepthStateEnv
from quaddif.algo.dreamerv3.wmenv.replaybuffer import ReplayBuffer
from quaddif.algo.dreamerv3.wmenv.utils import configure_opt
from quaddif.dynamics.pointmass import point_mass_quat

@torch.no_grad()
def collect_imagine_trj(env:DepthStateEnv,agent:ActorCriticAgent,cfg):
    feats,rewards,ends,actions,states,org_samples = [],[],[],[],[],[]
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

def train_agents(agent:ActorCriticAgent,state_env:DepthStateEnv,cfg):
    trainingcfg = getattr(cfg,"actor_critic").training
    feats,_,rewards,ends,_,org_samples = collect_imagine_trj(state_env,agent,trainingcfg)
    agent_info = agent.update(feats,org_samples,rewards,ends)
    reward_sum = rewards.sum(dim=-1).mean()
    agent_info['reward_sum'] = reward_sum.item()
    return agent_info
    
def train_worldmodel(world_model:DepthStateModel,replaybuffer:ReplayBuffer,opt,training_hyper):
    for _ in range(training_hyper.worldmodel_update_freq):
        sample_state, sample_action, sample_reward, sample_termination,sample_perception = \
                                            replaybuffer.sample(training_hyper.batch_size,training_hyper.batch_length)
        total_loss,rep_loss,dyn_loss,rec_loss,rew_loss,end_loss = \
            world_model.compute_loss(sample_state, sample_perception, sample_action, sample_reward, sample_termination)
    
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
        world_agent_cfg.replaybuffer.state_dim = 10
        world_agent_cfg.actor_critic.model.device = f"cuda:{device_idx}"
        world_agent_cfg.common.device = f"cuda:{device_idx}"
        self.cfg = cfg
        self.world_agent_cfg = world_agent_cfg
        
        statemodelcfg = getattr(world_agent_cfg,"state_predictor").state_model
        statemodelcfg.state_dim = 10
        actorcriticcfg = getattr(world_agent_cfg,"actor_critic").model
        actorcriticcfg.feat_dim = statemodelcfg.hidden_dim + statemodelcfg.latent_dim
        actorcriticcfg.hidden_dim = statemodelcfg.hidden_dim
        buffercfg = getattr(world_agent_cfg,"replaybuffer")
        buffercfg.state_dim = 10
        worldcfg = getattr(world_agent_cfg,"world_state_env")
        training_hyper = getattr(world_agent_cfg,"state_predictor").training
        self.training_hyper = training_hyper
        
        if cfg.env.name=='position_control':
            statemodelcfg.only_state = True
            buffercfg.use_perception = False
        
        self.agent = ActorCriticAgent(actorcriticcfg,env).to(device)
        self.state_model = DepthStateModel(statemodelcfg).to(device)
        if not world_agent_cfg.common.is_test:
            self.replaybuffer = ReplayBuffer(buffercfg)
            self.world_model_env = DepthStateEnv(self.state_model,self.replaybuffer,worldcfg)
        self.opt = configure_opt(self.state_model,**getattr(world_agent_cfg,'state_predictor').optimizer)
        
        if world_agent_cfg.common.checkpoint_path != None:
            self.load(world_agent_cfg.common.checkpoint_path)

        self.hidden = torch.zeros(cfg.n_envs, statemodelcfg.hidden_dim, device=device)
    
    @torch.no_grad()
    def act(self,obs,test=False):
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

    def step(self,cfg,env,obs,on_step_cb=None):
        policy_info = {}
        with torch.no_grad():
            if type(obs)!=torch.Tensor:
                state,perception = obs['state'],obs['perception'].unsqueeze(1)
            else:
                state,perception = obs,None
            if self.world_agent_cfg.common.use_symlog:
                state = symlog(state)
            if self.replaybuffer.ready() or self.world_agent_cfg.common.use_checkpoint:
                latent = self.state_model.sample_with_post(state,perception,self.hidden)[0].flatten(1)
                action = self.agent.sample(torch.cat([latent,self.hidden],dim=-1))[0]
                self.hidden = self.state_model.sample_with_prior(latent,action,self.hidden)[2]
            else:
                action = torch.randn(self.cfg.n_envs,3,device=state.device)
            next_obs,rewards,terminated,env_info = env.step(action)
            rewards = 10.*(1-rewards*0.1)
            self.replaybuffer.append(state,action,rewards,terminated,perception)
            
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

class WorldExporter(nn.Module):
    def __init__(self,agent:World_Agent):
        super().__init__()
        self.state_encoder = deepcopy(agent.state_model.state_encoder)
        self.inp_proj = deepcopy(agent.state_model.inp_proj)
        self.seq_model = deepcopy(agent.state_model.seq_model)
        self.act_state_proj = deepcopy(agent.state_model.act_state_proj)
        self.actor = deepcopy(agent.agent.actor_mean)
        
        self.register_buffer("hidden_state",torch.zeros(1,agent.state_model.cfg.hidden_dim))
        self.hidden_state = self.get_buffer("hidden_state")
    
    def sample_for_deploy(self,logits):
        probs = F.softmax(logits,dim=-1)
        return onehotsample(probs)
    
    def sample_with_post(self,state):
        feat = self.state_encoder(state)
        
        post_logits = self.inp_proj(torch.cat([feat,self.hidden_state],dim=-1))
        b,d = post_logits.shape
        post_logits = post_logits.reshape(b,int(math.sqrt(d)),-1) # b l d -> b l c k
        
        post_sample = self.sample_for_deploy(post_logits)
        return post_sample
    
    def sample_with_prior(self,latent,act):
        assert latent.ndim==act.ndim==2
        state_act = self.act_state_proj(torch.cat([latent,act],dim=-1))
        self.hidden_state = self.seq_model(state_act,self.hidden_state)

    def forward(self,state):
        with torch.no_grad():
            state = torch.sign(state) * torch.log(1 + torch.abs(state))
            latent = self.sample_with_post(state).flatten(1)
            action = self.actor(torch.cat([latent,self.hidden_state],dim=-1))
            action = torch.tanh(action)
            self.sample_with_prior(latent,action)
        return action
    
    @torch.jit.export
    def post_process(self,acc,orientation):
        quat_xyzw = point_mass_quat(acc, orientation)
        acc_norm = acc.norm(p=2, dim=-1)
        return quat_xyzw, acc_norm
    
    @torch.jit.export
    def reset(self):
        self.hidden_state.zero_()
    
    @torch.jit.export
    def rescale(self,raw_action,min_action,max_action):
        return (raw_action*0.5+0.5)*(max_action-min_action)+min_action
    
    def export(self,path:str,verbose=False):
        traced_script_module = torch.jit.script(self)
        if verbose:
            print(traced_script_module.code)
            print(traced_script_module.post_process.code)
            print(traced_script_module.reset.code)
            print(traced_script_module.rescale.code)
        save_path = os.path.join(path,"exportckpt.pt")
        traced_script_module.save(save_path)