from typing import *
import os
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig

from diffaero.env import PositionControl, ObstacleAvoidance, Racing
from diffaero.algo.dreamerv3.models.state_predictor import DepthStateModel, onehotsample
from diffaero.algo.dreamerv3.models.agent import ActorCriticAgent
from diffaero.algo.dreamerv3.models.blocks import symlog
from diffaero.algo.dreamerv3.wmenv.world_state_env import DepthStateEnv
from diffaero.algo.dreamerv3.wmenv.replaybuffer import ReplayBuffer
from diffaero.algo.dreamerv3.wmenv.utils import configure_opt
from diffaero.utils.runner import timeit
from diffaero.dynamics.pointmass import point_mass_quat, PointMassModelBase

@torch.no_grad()
@timeit
def collect_imagine_trj(env: DepthStateEnv, agent: ActorCriticAgent, cfg: DictConfig):
    latents, hiddens, rewards, ends, actions, org_samples = [], [], [], [], [], []
    imagine_length = cfg.imagine_length
    latent, hidden = env.make_generator_init()

    for i in range(imagine_length):
        latents.append(latent)
        hiddens.append(hidden)
        action, org_sample = agent.sample(torch.cat([latent, hidden], dim=-1))
        latent, reward, end, hidden, _ = env.step(action)
        rewards.append(reward)
        actions.append(action)
        org_samples.append(org_sample)
        ends.append(end)

    latents.append(latent)
    hiddens.append(hidden)
    latents = torch.stack(latents, dim=1)
    hiddens = torch.stack(hiddens, dim=1)
    actions = torch.stack(actions, dim=1)
    org_samples = torch.stack(org_samples, dim=1)
    rewards = torch.stack(rewards, dim=1)
    ends = torch.stack(ends, dim=1)

    return latents, hiddens, actions, rewards, ends, org_samples

@torch.no_grad()
def generate_video(env: DepthStateEnv, agent: ActorCriticAgent, cfg: DictConfig, imagine_length:int=64):
    cfg.imagine_length = imagine_length
    latents, hiddens, _, _, _, _ = collect_imagine_trj(env, agent, cfg)   
    videos = env.decode(latents, hiddens)
    videos = videos[::videos.shape[0]//16]
    return videos.unsqueeze(2).repeat(1, 1, 3, 1, 1) # B L 3 H W

@timeit
def train_agents(agent: ActorCriticAgent, state_env: DepthStateEnv, cfg: DictConfig):
    trainingcfg = getattr(cfg, "actor_critic").training
    latents, hiddens,  _, rewards, ends, org_samples = collect_imagine_trj(state_env, agent, trainingcfg)
    agent_info = agent.update(torch.cat([latents, hiddens], dim=-1), org_samples, rewards, ends)
    reward_sum = rewards.sum(dim=-1).mean()
    agent_info["reward_sum"] = reward_sum.item()
    return agent_info

@timeit
def train_worldmodel(
    world_model: DepthStateModel,
    replaybuffer: ReplayBuffer,
    opt: torch.optim.Optimizer,
    training_hyper: DictConfig,
    scaler: torch.amp.GradScaler
):
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=training_hyper.use_amp):
        for _ in range(training_hyper.worldmodel_update_freq):
            sample_state, sample_action, sample_reward, sample_termination, sample_perception, sample_grid, sample_vis = \
                replaybuffer.sample(training_hyper.batch_size,training_hyper.batch_length)
            total_loss, rep_loss, dyn_loss, rec_loss, rew_loss, end_loss, grid_loss = \
                world_model.compute_loss(
                    sample_state,
                    sample_perception,
                    sample_action,
                    sample_reward,
                    sample_termination,
                    sample_grid,
                    sample_vis
                )
    
    if scaler is not None:
        scaler.scale(total_loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(world_model.parameters(), training_hyper.max_grad_norm)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    world_info = {
        'WorldModel/state_total_loss':total_loss.item(),
        'WorldModel/state_rep_loss':rep_loss.item(),
        'WorldModel/state_dyn_loss':dyn_loss.item(),
        'WorldModel/state_rec_loss':rec_loss.item(),
        'WorldModel/grad_norm':grad_norm.item(),
        'WorldModel/state_rew_loss':rew_loss.item(),
        'WorldModel/state_end_loss':end_loss.item(),
        'WorldModel/state_grid_loss':grid_loss.item(),
    }

    return world_info

class World_Agent:
    def __init__(self, cfg: DictConfig, env: Union[PositionControl, ObstacleAvoidance, ObstacleAvoidanceGrid], device: torch.device):
        self.cfg = cfg
        self.n_envs = env.n_envs
        device_idx = device.index
        if isinstance(env.dynamics, PointMassModelBase) and not isinstance(env, Racing):
            state_dim = 9
        else:
            state_dim = 13
        use_grid = True if isinstance(env, ObstacleAvoidanceGrid) else False
        world_agent_cfg = deepcopy(cfg)
        world_agent_cfg = cfg
        world_agent_cfg.replaybuffer.device = f"cuda:{device_idx}"
        world_agent_cfg.replaybuffer.num_envs = self.n_envs
        world_agent_cfg.replaybuffer.state_dim = state_dim
        world_agent_cfg.actor_critic.model.device = f"cuda:{device_idx}"
        world_agent_cfg.common.device = f"cuda:{device_idx}"
        self.world_agent_cfg = world_agent_cfg

        statemodelcfg = getattr(world_agent_cfg, "state_predictor").state_model
        statemodelcfg.state_dim = state_dim
        actorcriticcfg = getattr(world_agent_cfg, "actor_critic").model
        actorcriticcfg.feat_dim = statemodelcfg.hidden_dim + statemodelcfg.latent_dim
        actorcriticcfg.hidden_dim = statemodelcfg.hidden_dim
        
        buffercfg = getattr(world_agent_cfg, "replaybuffer")
        buffercfg.state_dim = state_dim
        worldcfg = getattr(world_agent_cfg, "world_state_env")
        training_hyper = getattr(world_agent_cfg, "state_predictor").training
        self.training_hyper = training_hyper
        
        if use_grid:
            grid_dim = env.n_grid_points
            buffercfg.use_grid = True
            buffercfg.grid_dim = grid_dim
            statemodelcfg.use_grid = True
            statemodelcfg.grid_dim = grid_dim

        if isinstance(env, PositionControl) or isinstance(env, Racing):
            statemodelcfg.only_state = True
            buffercfg.use_perception = False
            statemodelcfg.state_dim = state_dim
            world_agent_cfg.replaybuffer.state_dim = state_dim
        
        self.agent = ActorCriticAgent(actorcriticcfg,env).to(device)
        self.state_model = DepthStateModel(statemodelcfg).to(device)
        if not world_agent_cfg.common.is_test:
            self.replaybuffer = ReplayBuffer(buffercfg)
            self.world_model_env = DepthStateEnv(self.state_model, self.replaybuffer, worldcfg)
        self.opt = configure_opt(self.state_model, **getattr(world_agent_cfg, "state_predictor").optimizer)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.training_hyper.use_amp)

        if world_agent_cfg.common.checkpoint_path is not None:
            self.load(world_agent_cfg.common.checkpoint_path)

        self.num_steps = 0
        self.hidden = torch.zeros(cfg.n_envs, statemodelcfg.hidden_dim, device=device)

    @torch.no_grad()
    def act(self, obs, test=False):
        if type(obs) != torch.Tensor:
            state, perception = obs["state"], obs["perception"].unsqueeze(1)
        else:
            state, perception = obs, None
        if self.world_agent_cfg.common.use_symlog:
            state = symlog(state)
        latent = self.state_model.sample_with_post(state, perception, self.hidden, True)[0].flatten(1)
        action = self.agent.sample(torch.cat([latent, self.hidden], dim=-1), test)[0]
        prior_sample, _, self.hidden = self.state_model.sample_with_prior(latent, action, self.hidden, True)
        return action, None

    @timeit
    def step(self, cfg, env, logger, obs, on_step_cb):
        policy_info = {}

        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                state, perception = obs['state'], obs['perception'].unsqueeze(1)
                if 'grid' in obs.keys():
                    grid, vis = obs['grid'], obs['visible_map']
                else:
                    grid, vis = None, None
            else:
                state, perception = obs, None
            if self.world_agent_cfg.common.use_symlog:
                state = symlog(state)
            if self.replaybuffer.ready() or self.world_agent_cfg.common.checkpoint_path is not None:
                latent = self.state_model.sample_with_post(state, perception, self.hidden)[0].flatten(1)
                action = self.agent.sample(torch.cat([latent, self.hidden], dim=-1))[0]
                prior_sample, _, self.hidden = self.state_model.sample_with_prior(latent, action, self.hidden)
            else:
                action = torch.randn(self.n_envs,3,device=state.device)
            next_obs, (loss, rewards), terminated, env_info = env.step(env.rescale_action(action))
            rewards = rewards*10.
            self.replaybuffer.append(state, action, rewards, terminated, perception, grid, vis)
            
            if terminated.any():
                zeros = torch.zeros_like(self.hidden)
                self.hidden = torch.where(terminated.unsqueeze(-1), zeros, self.hidden)
                # for i in range(self.n_envs):
                #     if terminated[i]:
                #         self.hidden[i] = 0
        
        if self.replaybuffer.ready():
            world_info = train_worldmodel(self.state_model, self.replaybuffer, self.opt, self.training_hyper, self.scaler)
            agent_info = train_agents(self.agent, self.world_model_env, self.world_agent_cfg)
            policy_info.update(world_info)
            policy_info.update(agent_info)

        obs = next_obs
        self.num_steps+=1

        if self.num_steps%2500==0:
            logger_video = generate_video(self.world_model_env, self.agent, self.world_agent_cfg.actor_critic.training, 64)
            policy_info["video"] = logger_video

        return obs, policy_info, env_info, 0.0, 0.0

    def finetune(self):
        agent_info = train_agents(self.agent, self.world_model_env, self.world_agent_cfg)
        return agent_info

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_model.state_dict(), f"{path}/statemodel.pth")
        torch.save(self.agent.state_dict(), f"{path}/agent.pth")

    def load(self, path):
        self.state_model.load_state_dict(torch.load(os.path.join(path, "statemodel.pth")))
        self.agent.load_state_dict(torch.load(os.path.join(path, "agent.pth")))

    @staticmethod
    def build(cfg, env, device):
        return World_Agent(cfg, env, device)
    
    def export(
        self,
        path: str,
        export_cfg: DictConfig,
        verbose=False,
    ):
        WorldExporter(self).export(path, export_cfg, verbose)

class WorldExporter(nn.Module):
    def __init__(self, agent):
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
        
        self.named_inputs = [
            ("state", torch.zeros(1, 9)),
            ("orientation", torch.zeros(1, 3)),
            ("Rz", torch.zeros(1, 3, 3)),
            ("min_action", torch.zeros(1, 3)),
            ("max_action", torch.zeros(1, 3)),
        ]
        if hasattr(agent.state_model, 'image_encoder'):
            self.named_inputs.insert(1, ("perception", torch.rand(1, 9, 16)))
        self.output_names = [
            "action",
            "quat_xyzw_cmd",
            "acc_norm"
        ]
        
        # self.register_buffer("hidden_state",torch.zeros(1,agent.state_model.cfg.hidden_dim))
        # self.hidden_state = self.get_buffer("hidden_state")
        self.is_recurrent = True
        self.hidden_shape = (1, agent.state_model.cfg.hidden_dim)
        
        self.obs_frame: str
        self.action_frame: str
        
        if self.is_recurrent:
            self.named_inputs.append(("hidden_in", torch.rand(self.hidden_shape)))
            self.output_names.append("hidden_out")
    
    def sample_for_deploy(self,logits):
        probs = F.softmax(logits,dim=-1)
        return onehotsample(probs)
    
    def sample_with_post(self,feat,hidden):        
        post_logits = self.inp_proj(torch.cat([feat,hidden],dim=-1))
        b,d = post_logits.shape
        post_logits = post_logits.reshape(b,int(math.sqrt(d)),-1) # b l d -> b l c k
        post_sample = self.sample_for_deploy(post_logits)
        return post_sample

    def sample_with_prior(self, latent, act, hidden):
        assert latent.ndim == act.ndim == 2
        state_act = self.act_state_proj(torch.cat([latent, act], dim=-1))
        hidden = self.seq_model(state_act, hidden)
        return hidden

    def forward_perc_prop(self, state, perception, orientation, Rz, min_action, max_action, hidden):
        with torch.no_grad():
            if self.use_symlog:
                state = torch.sign(state) * torch.log(1 + torch.abs(state))
            state_feat = self.state_encoder(state)
            image_feat = self.image_encoder(perception.unsqueeze(0))
            feat = torch.cat([state_feat, image_feat], dim=-1)
            latent = self.sample_with_post(feat,hidden).flatten(1)
            mean_std = self.actor(torch.cat([latent,hidden],dim=-1))
            action, _ = torch.chunk(mean_std, 2, dim=-1)
            action = torch.tanh(action)
            hidden = self.sample_with_prior(latent, action, hidden)
            action, quat_xyzw, acc_norm = self.post_process(action, min_action, max_action, orientation, Rz)
        return action, quat_xyzw, acc_norm, hidden
            
    def forward_prop(self, state, orientation, Rz, min_action, max_action, hidden):
        with torch.no_grad():
            if self.use_symlog:
                state = torch.sign(state) * torch.log(1 + torch.abs(state))
            state_feat = self.state_encoder(state)
            latent = self.sample_with_post(state_feat,hidden).flatten(1)
            mean_std = self.actor(torch.cat([latent,hidden],dim=-1))
            action, _ = torch.chunk(mean_std, 2, dim=-1)
            action = torch.tanh(action)
            hidden = self.sample_with_prior(latent,action,hidden)
            action, quat_xyzw, acc_norm = self.post_process(action, min_action, max_action, orientation, Rz)
        return action, quat_xyzw, acc_norm, hidden  
    
    def post_process_local(self, raw_action, min_action, max_action, orientation, Rz):
        action = (raw_action * 0.5 + 0.5) * (max_action - min_action) + min_action
        acc_cmd = torch.matmul(Rz, action.unsqueeze(-1)).squeeze(-1)
        quat_xyzw = point_mass_quat(acc_cmd, orientation)
        acc_norm = acc_cmd.norm(p=2, dim=-1)
        return acc_cmd, quat_xyzw, acc_norm

    def post_process_world(self, raw_action, min_action, max_action, orientation, Rz):
        action = (raw_action * 0.5 + 0.5) * (max_action - min_action) + min_action
        quat_xyzw = point_mass_quat(action, orientation)
        acc_norm = action.norm(p=2, dim=-1)
        return action, quat_xyzw, acc_norm
    
    def post_process(self, raw_action, min_action, max_action, orientation, Rz):
        if self.action_frame == "local":
            return self.post_process_local(raw_action, min_action, max_action, orientation, Rz)
        elif self.action_frame == "world":
            return self.post_process_world(raw_action, min_action, max_action, orientation, Rz)
        else:
            raise ValueError(f"Unknown action frame: {self.action_frame}")
    
    def export_jit(self, path: str, verbose=False):
        traced_script_module = torch.jit.script(self)
        if verbose:
            print(traced_script_module.code)
        export_path = os.path.join(path, "exported_actor.pt2")
        traced_script_module.save(export_path)
        print(f"The checkpoint is compiled and exported to {export_path}.")
    
    def export_onnx(self, path:str):
        export_path = os.path.join(path, "exported_actor.onnx")
        names, inputs = zip(*self.named_inputs)
        for inp in inputs:
            print(inp.device)
        torch.onnx.export(
            model=self.to('cpu'),
            args=inputs,
            f=export_path,
            input_names=names,
            output_names=self.output_names
        )
        print(f"The checkpoint is compiled and exported to {export_path}.")
        
    def export(
        self,
        path: str,
        export_cfg: DictConfig,
        verbose=False,
    ):
        self.obs_frame = export_cfg.obs_frame
        self.action_frame = export_cfg.action_frame
        if export_cfg.jit:
            self.export_jit(path, verbose)
        if export_cfg.onnx:
            self.export_onnx(path)
    
    # def export(self, path: str, verbose=False, export_onnx=False, export_pnnx=False):
    #     if export_onnx:
    #         self.export_onnx(path)
    #     else:
    #         self.export_jit(path, verbose)
