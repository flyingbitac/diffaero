from typing import *
import os

import torch
from omegaconf import DictConfig

from quaddif.env import PositionControl, ObstacleAvoidance, ObstacleAvoidanceGrid
from quaddif.algo.dreamerv3.models.state_predictor import DepthStateModel
from quaddif.algo.dreamerv3.models.agent import ActorCriticAgent
from quaddif.algo.dreamerv3.models.blocks import symlog
from quaddif.algo.dreamerv3.wmenv.world_state_env import DepthStateEnv
from quaddif.algo.dreamerv3.wmenv.replaybuffer import ReplayBuffer
from quaddif.algo.dreamerv3.wmenv.utils import configure_opt
from quaddif.utils.runner import timeit

@torch.no_grad()
def collect_imagine_trj(env: DepthStateEnv, agent: ActorCriticAgent, cfg: DictConfig):
    feats, rewards, ends, actions, org_samples = [], [], [], [], []
    imagine_length = cfg.imagine_length
    latent, hidden = env.make_generator_init()

    for i in range(imagine_length):
        feat = torch.cat([latent, hidden], dim=-1)
        feats.append(feat)
        action, org_sample = agent.sample(feat)
        latent, reward, end, hidden = env.step(action)
        rewards.append(reward)
        actions.append(action)
        org_samples.append(org_sample)
        ends.append(end)

    feat = torch.cat([latent, hidden], dim=-1)
    feats.append(feat)
    feats = torch.stack(feats, dim=1)
    actions = torch.stack(actions, dim=1)
    org_samples = torch.stack(org_samples, dim=1)
    rewards = torch.stack(rewards, dim=1)
    ends = torch.stack(ends, dim=1)

    return feats, actions, rewards, ends, org_samples

@timeit
def train_agents(agent: ActorCriticAgent, state_env: DepthStateEnv, cfg: DictConfig):
    trainingcfg = getattr(cfg, "actor_critic").training
    feats, _, rewards, ends, org_samples = collect_imagine_trj(state_env, agent, trainingcfg)
    agent_info = agent.update(feats, org_samples, rewards, ends)
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
            sample_state, sample_action, sample_reward, sample_termination, sample_perception, sample_grid, sample_visible_map = \
                replaybuffer.sample(training_hyper.batch_size,training_hyper.batch_length)
            total_loss, rep_loss, dyn_loss, rec_loss, rew_loss, end_loss, grid_loss, grid_acc, grid_precision = \
                world_model.compute_loss(
                    sample_state,
                    sample_perception,
                    sample_action,
                    sample_reward,
                    sample_termination,
                    sample_grid,
                    sample_visible_map
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
        'WorldModel/state_grid_acc':grid_acc.item(),
        'WorldModel/state_grid_precision':grid_precision.item(),
    }

    return world_info

class World_Agent:
    def __init__(self, cfg: DictConfig, env: Union[PositionControl, ObstacleAvoidance], device: torch.device):
        self.cfg = cfg
        self.n_envs = env.n_envs
        device_idx = device.index
        world_agent_cfg = cfg
        world_agent_cfg.replaybuffer.device = f"cuda:{device_idx}"
        world_agent_cfg.replaybuffer.num_envs = self.n_envs
        world_agent_cfg.replaybuffer.state_dim = 10
        world_agent_cfg.actor_critic.model.device = f"cuda:{device_idx}"
        world_agent_cfg.common.device = f"cuda:{device_idx}"
        self.world_agent_cfg = world_agent_cfg

        statemodelcfg = getattr(world_agent_cfg, "state_predictor").state_model
        statemodelcfg.state_dim = 10
        actorcriticcfg = getattr(world_agent_cfg, "actor_critic").model
        actorcriticcfg.feat_dim = statemodelcfg.hidden_dim + statemodelcfg.latent_dim
        actorcriticcfg.hidden_dim = statemodelcfg.hidden_dim
        buffercfg = getattr(world_agent_cfg, "replaybuffer")
        buffercfg.state_dim = 10
        worldcfg = getattr(world_agent_cfg, "world_state_env")
        training_hyper = getattr(world_agent_cfg, "state_predictor").training
        self.training_hyper = training_hyper

        if isinstance(env, PositionControl):
            statemodelcfg.only_state = True
            buffercfg.use_perception = False
            statemodelcfg.state_dim = 10
            world_agent_cfg.replaybuffer.state_dim = 10
        if isinstance(env, ObstacleAvoidanceGrid):
            statemodelcfg.grid_dim = env.n_grid_points
            buffercfg.grid_dim = env.n_grid_points
            statemodelcfg.use_grid = True
            buffercfg.use_grid = True
        
        self.agent = ActorCriticAgent(actorcriticcfg,env).to(device)
        self.state_model = DepthStateModel(statemodelcfg).to(device)
        if not world_agent_cfg.common.is_test:
            self.replaybuffer = ReplayBuffer(buffercfg)
            self.world_model_env = DepthStateEnv(self.state_model, self.replaybuffer, worldcfg)
        self.opt = configure_opt(self.state_model, **getattr(world_agent_cfg, "state_predictor").optimizer)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.training_hyper.use_amp)

        if world_agent_cfg.common.checkpoint_path is not None:
            self.load(world_agent_cfg.common.checkpoint_path)

        self.hidden = torch.zeros(self.n_envs, statemodelcfg.hidden_dim, device=device)

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
        self.hidden = self.state_model.sample_with_prior(latent, action, self.hidden, True)[2]
        return action, None

    @timeit
    def step(self, cfg, env, obs, on_step_cb):
        policy_info = {}
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                state, perception = obs['state'], obs['perception'].unsqueeze(1)
                if 'grid' in obs:
                    grid = obs['grid']
                    visible_map = obs["visible_map"]
                else:
                    grid, visible_map = None, None
            else:
                state, perception, grid, visible_map = obs, None, None, None
            if self.world_agent_cfg.common.use_symlog:
                state = symlog(state)
            if self.replaybuffer.ready() or self.world_agent_cfg.common.checkpoint_path is not None:
                latent = self.state_model.sample_with_post(state, perception, self.hidden)[0].flatten(1)
                action = self.agent.sample(torch.cat([latent, self.hidden], dim=-1))[0]
                self.hidden = self.state_model.sample_with_prior(latent, action, self.hidden)[2]
            else:
                action = torch.randn(self.n_envs,3,device=state.device)
            next_obs,rewards,terminated,env_info = env.step(env.rescale_action(action))
            rewards = 10.*(1-rewards*0.1)
            self.replaybuffer.append(state, action, rewards, terminated, perception, grid, visible_map)
            
            if terminated.any():
                for i in range(self.n_envs):
                    if terminated[i]:
                        self.hidden[i] = 0

        if self.replaybuffer.ready():
            world_info = train_worldmodel(self.state_model, self.replaybuffer, self.opt, self.training_hyper, self.scaler)
            agent_info = train_agents(self.agent, self.world_model_env, self.world_agent_cfg)
            policy_info.update(world_info)
            policy_info.update(agent_info)

        obs = next_obs

        return obs, policy_info, env_info, 0.0, 0.0

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