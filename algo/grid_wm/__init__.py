from typing import Tuple, Dict, Union, Optional, List
import math
import os
from copy import deepcopy

from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import tensordict
from tensordict import TensorDict

from diffaero.env.obstacle_avoidance_grid import ObstacleAvoidanceGrid
from diffaero.dynamics.pointmass import point_mass_quat
from diffaero.algo.buffer import RolloutBufferGRID
from diffaero.network.agents import StochasticActor, StochasticAsymmetricActorCriticV
from diffaero.utils.runner import timeit
from diffaero.utils.logger import Logger
from .world.backbone import WorldModel, WorldModelTesttime


class GRIDWM:
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: Tuple[int, Tuple[int, int]],
        state_dim: int,
        action_dim: int,
        n_envs: int,
        grid_cfg: DictConfig,
        device: torch.device
    ):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_envs = n_envs
        
        self.l_rollout: int = cfg.l_rollout
        self.discount: float = cfg.gamma
        self.lmbda: float = cfg.lmbda
        self.batch_size: int = cfg.wm.train.batch_size
        self.n_epochs: int = cfg.wm.train.n_epochs
        self.grid_points: List[int] = grid_cfg.n_points
        self.n_grid_points = math.prod(self.grid_points)
        self.device = device
        
        self.odom_free: bool = cfg.odom_free
        self.state_dim = obs_dim[0]
        
        # world model
        self.wm = WorldModel(obs_dim, cfg.wm, grid_cfg).to(device)
        if cfg.wm.compile:
            self.wm = torch.compile(self.wm, mode="reduce-overhead")
        # replay buffer
        self.buffer=RolloutBufferGRID(
            l_rollout=self.l_rollout,
            buffer_size=int(cfg.wm.train.buffer_size),
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            grid_dim=self.n_grid_points,
            device=device
        )
        # agent
        self.actor_cfg: DictConfig = cfg.agent.actor
        self.critic_cfg: DictConfig = cfg.agent.critic
        actor_input_dim = self.wm.deter_dim + self.wm.latent_dim + (3 if self.odom_free else obs_dim[0])
        self.agent = StochasticAsymmetricActorCriticV(self.actor_cfg, self.critic_cfg, actor_input_dim, state_dim, action_dim).to(device)
        # optimizer
        self.actor_optim = torch.optim.Adam(self.agent.actor.parameters(), lr=self.actor_cfg.lr)
        self.critic_optim = torch.optim.Adam(self.agent.critic.parameters(), lr=self.critic_cfg.lr)

        self.entropy_weight: float = cfg.agent.entropy_weight
        self.actor_loss = torch.tensor(0., device=self.device)
        self.rollout_gamma = torch.ones(self.n_envs, device=self.device)
        self.cumulated_loss = torch.zeros(self.n_envs, device=self.device)
        self.entropy_loss = torch.tensor(0., device=self.device)
        self.deter: Tensor | None = None
    
    def make_state_input(self, obs: TensorDict) -> Tensor:
        state = obs["state"]
        target_vel, odom_info = state[..., :3], state[..., 3:]
        inputs = [target_vel]
        if not self.odom_free:
            inputs.append(odom_info)
        return torch.cat(inputs, dim=-1)
    
    @timeit
    def act(self, obs, test=False):
        # type: (TensorDict, bool) -> Tuple[Tensor, Dict[str, Tensor]]
        with torch.no_grad():
            if self.deter is None:
                self.deter = torch.zeros(obs['state'].shape[0], self.wm.deter_dim, device=obs['state'].device)
            latent = self.wm.encode(obs['perception'], obs['state'], self.deter)
            actor_input = torch.cat([latent, self.deter, self.make_state_input(obs)], dim=-1)
        action, sample, logprob, entropy = self.agent.get_action(actor_input, test=test)
        self.deter = self.wm.recurrent(latent, self.deter, action)
        return action, {"latent": latent, "sample": sample, "logprob": logprob, "entropy": entropy}
    
    def record_loss(self, loss, policy_info, env_info, last_step=False):
        # type: (Tensor, Dict[str, Tensor], Dict[str, Tensor], bool) -> Tensor
        reset = torch.ones_like(env_info["reset"]) if last_step else env_info["reset"]
        truncated = torch.ones_like(env_info["reset"]) if last_step else env_info["truncated"]
        # add cumulated loss if rollout ends or trajectory ends (terminated or truncated)
        self.cumulated_loss = self.cumulated_loss + self.rollout_gamma * loss
        cumulated_loss = self.cumulated_loss[reset].sum()
        # add terminal value if rollout ends or truncated
        next_value = self.agent.get_value(env_info["next_state_before_reset"])
        terminal_value = (self.rollout_gamma * self.discount * next_value)[truncated].sum()
        assert terminal_value.requires_grad and env_info["next_state_before_reset"].requires_grad
        # add up the discounted cumulated loss, the terminal value and the entropy loss
        self.actor_loss = self.actor_loss + cumulated_loss - terminal_value
        # self.actor_loss = self.actor_loss + terminal_value
        self.entropy_loss = self.entropy_loss - policy_info["entropy"].sum()
        # reset the discount factor, clear the cumulated loss if trajectory ends
        self.rollout_gamma = torch.where(reset, 1, self.rollout_gamma * self.discount)
        self.cumulated_loss = torch.where(reset, 0, self.cumulated_loss)
        return next_value.detach()
    
    @timeit
    def update_wm(self):
        if not self.buffer.size >= self.batch_size:
            return {}, {}, None, None
        for _ in range(self.n_epochs):
            observations, actions, terminated, rewards = self.buffer.sample4wm(self.batch_size)
            # find ground truth and visible grid
            ground_truth_grid = observations["grid"]
            visible_map = observations["visible_map"]
            total_loss, grad_norms, grid_pred = self.wm.update(
                obs=observations['perception'],
                state=observations['state'],
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                gt_grids=ground_truth_grid,
                visible_map=visible_map
            )
        
        if grid_pred is not None:
            visible_grid_gt_for_plot = ground_truth_grid & visible_map
            visible_grid_pred_for_plot = grid_pred & visible_map
            
            n_missd_predictions = torch.sum(visible_grid_gt_for_plot != visible_grid_pred_for_plot, dim=-1) # [batch_size, l_rollout]
            env_idx, time_idx = torch.where(n_missd_predictions == n_missd_predictions.max())
            env_idx, time_idx = env_idx[0], time_idx[0]
            selected_grid_gt = visible_grid_gt_for_plot[env_idx, time_idx].reshape(*self.grid_points)
            selected_grid_pred = visible_grid_pred_for_plot[env_idx, time_idx].reshape(*self.grid_points)
        else:
            selected_grid_gt, selected_grid_pred = None, None
        
        return total_loss, grad_norms, selected_grid_gt, selected_grid_pred
    
    @torch.no_grad()
    def bootstrap(
        self,
        next_values: Tensor,
        rewards: Tensor,
        dones: Tensor,
        terminated: Tensor
    ):
        # value of the next obs should be zero if the next obs is a terminal obs
        next_values = next_values * (1 - terminated)
        if self.lmbda == 0.:
            target_values = rewards + self.discount * next_values
        else:
            target_values = torch.zeros_like(next_values).to(self.device)
            Ai = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)
            Bi = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)
            lam = torch.ones(self.n_envs, dtype=torch.float32, device=self.device)
            next_values[-1] = 1.
            for i in reversed(range(self.l_rollout)):
                lam = lam * self.lmbda * (1. - dones[i]) + dones[i]
                Ai = (1. - dones[i]) * (
                    self.discount * (self.lmbda * Ai + next_values[i]) + \
                    (1. - lam) / (1. - self.lmbda) * rewards[i])
                Bi = self.discount * (next_values[i] * dones[i] + Bi * (1. - dones[i])) + \
                     rewards[i]
                target_values[i] = (1.0 - self.lmbda) * Ai + lam * Bi
        return target_values.view(-1)
    
    @timeit
    def update_critic(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        T, N = self.l_rollout, self.n_envs
        state, next_values, rewards, dones, terminated = self.buffer.sample4critic(self.batch_size)
        target_values = self.bootstrap(next_values, rewards, dones, terminated)
        batch_indices = torch.randperm(T*N, device=self.device)
        mb_size = T*N // self.critic_cfg.n_minibatch
        for start in range(0, T*N, mb_size):
            end = start + mb_size
            mb_indices = batch_indices[start:end]
            values = self.agent.get_value(state.flatten(0, 1)[mb_indices])
            critic_loss = F.mse_loss(values, target_values[mb_indices])
            self.critic_optim.zero_grad()
            critic_loss.backward()
            if self.critic_cfg.max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_norm=self.critic_cfg.max_grad_norm)
            else:
                grad_norm = torch.nn.utils.get_total_norm(self.agent.critic.parameters())
            self.critic_optim.step()
        return {"critic_loss": critic_loss.item()}, {"critic_grad_norm": grad_norm.item()}
    
    @timeit
    def update_actor(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        actor_loss = self.actor_loss / (self.n_envs * self.l_rollout)
        entropy_loss = self.entropy_loss / (self.n_envs * self.l_rollout)
        total_loss = actor_loss + self.entropy_weight * entropy_loss
        self.actor_optim.zero_grad()
        total_loss.backward()
        if self.actor_cfg.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), max_norm=self.actor_cfg.max_grad_norm)
        else:
            grad_norm = torch.nn.utils.get_total_norm(self.agent.actor.parameters())
        self.actor_optim.step()
        return {"actor_loss": actor_loss.mean().item(), "entropy_loss": entropy_loss.mean().item()}, {"actor_grad_norm": grad_norm}
    
    @timeit
    def step(self, cfg: DictConfig, env: ObstacleAvoidanceGrid, obs: TensorDict, on_step_cb=None):
        # env.prev_visible_map.fill_(False) # clear the memory that wm shouldn't have
        self.clear_loss()
        for t in range(self.l_rollout):
            action, policy_info = self.act(obs)
            state = env.get_state()
            with torch.no_grad():
                value = self.agent.get_value(state)
            policy_info["value"] = value
            next_obs, (loss, reward), terminated, env_info = env.step(
                env.rescale_action(action), next_state_before_reset=True)
            self.reset(env_info['reset'])
            next_value = self.record_loss(loss, policy_info, env_info, last_step=(t==cfg.l_rollout-1))
            self.buffer.add(
                obs=obs,
                state=state,
                action=action,
                reward=reward,
                value=value,
                next_done=env_info["reset"],
                next_terminated=terminated,
                next_value=next_value
            )
            obs = next_obs
            if on_step_cb is not None:
                on_step_cb(
                    obs=obs,
                    action=action,
                    policy_info=policy_info,
                    env_info=env_info)
        wm_losses, wm_grad_norm, selected_grid_gt, selected_grid_pred = self.update_wm()
        actor_losses, actor_grad_norms = self.update_actor()
        critic_losses, critic_grad_norms = self.update_critic()
        losses = {**wm_losses, **actor_losses, **critic_losses}
        grad_norms = {**wm_grad_norm, **actor_grad_norms, **critic_grad_norms}
        self.detach()
        if selected_grid_gt is not None and selected_grid_pred is not None:
            policy_info.update({
                "grid_gt": selected_grid_gt,
                "grid_pred": selected_grid_pred
            })
        return obs, policy_info, env_info, losses, grad_norms

    def clear_loss(self):
        self.rollout_gamma.fill_(1.)
        self.actor_loss.detach_().fill_(0.)
        self.cumulated_loss.detach_().fill_(0.)
        self.entropy_loss.detach_().fill_(0.)
    
    def save(self, path):
        self.wm.save(path)
        self.agent.save(path)
    
    def load(self, path):
        self.wm.load(path)
        self.agent.load(path)
    
    def detach(self):
        self.deter.detach_()
    
    def reset(self, env_idx: Tensor):
        self.deter[env_idx] = 0.
    
    @staticmethod
    def build(cfg: DictConfig, env: ObstacleAvoidanceGrid, device: torch.device):
        if hasattr(env.cfg, "grid"):
            return GRIDWM(
                cfg=cfg,
                obs_dim=env.obs_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                n_envs=env.n_envs,
                grid_cfg=env.cfg.grid,
                device=device)
        else:
            return GRIDWMTesttime(
                cfg=cfg,
                obs_dim=env.obs_dim,
                action_dim=env.action_dim,
                device=device
            )
    
    def export(
        self,
        path: str,
        export_cfg: DictConfig,
        verbose=False,
    ):
        return
        testtime = GRIDWMTesttime(self.cfg, self.obs_dim, self.action_dim, self.device)
        testtime.load(path)
        testtime.export(path, export_cfg, verbose)


class GRIDWMTesttime:
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: Tuple[int, Tuple[int, int]],
        action_dim: int,
        device: torch.device
    ):
        self.obs_dim = obs_dim
        self.wm = WorldModelTesttime(obs_dim, cfg.wm).to(device)
        
        self.odom_free = cfg.odom_free
        self.state_dim = obs_dim[0]
        actor_input_dim = self.wm.deter_dim + self.wm.latent_dim + (3 if self.odom_free else obs_dim[0])
        
        self.actor = StochasticActor(cfg.actor.network, actor_input_dim, action_dim).to(device)
        self.device = device
        self.deter: Tensor = None
    
    def make_state_input(self, obs: TensorDict) -> Tensor:
        state = obs["state"]
        target_vel, odom_info = state[..., :3], state[..., 3:]
        inputs = [target_vel]
        if not self.odom_free:
            inputs.append(odom_info)
        return torch.cat(inputs, dim=-1)
    
    @timeit
    def act(self, obs, test=False):
        # type: (TensorDict, bool) -> Tuple[Tensor, Dict[str, Tensor]]
        with torch.no_grad():
            if self.deter is None:
                self.deter = torch.zeros(obs['state'].shape[0], self.wm.deter_dim, device=obs['state'].device)
            Logger.info(obs['state'].shape)
            latent = self.wm.encode(obs['perception'], obs['state'], self.deter)
            actor_input = torch.cat([latent, self.deter, self.make_state_input(obs)], dim=-1)
        action, sample, logprob, entropy = self.actor(actor_input, test=test)
        self.deter = self.wm.recurrent(latent, self.deter, action)
        return action, {"latent": latent, "sample": sample, "logprob": logprob, "entropy": entropy}
    
    def save(self, path):
        self.wm.save(path)
        self.actor.save(path)
    
    def load(self, path):
        self.wm.load(path)
        self.actor.load(path)
    
    def detach(self):
        self.deter.detach_()
    
    def reset(self, env_idx: Tensor):
        self.deter[env_idx] = 0.
    
    def export(
        self,
        path: str,
        export_cfg: DictConfig,
        verbose=False,
    ):
        GRIDWMExporter(self).export(path, export_cfg, verbose)


class GRIDWMExporter(nn.Module):
    def __init__(self, agent: GRIDWMTesttime):
        super().__init__()
        self.wm_encoder = deepcopy(agent.wm).cpu()
        self.actor = deepcopy(agent.actor.actor_mean).cpu()
        self.odom_free = agent.odom_free
        state_dim, perception_dim = agent.state_dim, agent.obs_dim[1]
        Logger.warning(state_dim)
        self.named_inputs = [
            ("state", torch.zeros(1, state_dim)),
            ("perception", torch.zeros(1, perception_dim[0], perception_dim[1])),
            ("orientation", torch.zeros(1, 3)),
            ("Rz", torch.zeros(1, 3, 3)),
            ("min_action", torch.zeros(1, 3)),
            ("max_action", torch.zeros(1, 3)),
            ("hidden_in", torch.zeros(1, self.wm_encoder.deter_dim)),
        ]
        self.output_names = [
            "action",
            "quat_xyzw_cmd",
            "acc_norm",
            "hidden_out"
        ]
        self.obs_frame: str
        self.action_frame: str
    
    def make_state_input(self, state: Tensor) -> Tensor:
        target_vel, odom_info = state[..., :3], state[..., 3:]
        inputs = [target_vel]
        if not self.odom_free:
            inputs.append(odom_info)
        return torch.cat(inputs, dim=-1)
    
    def forward(self, state, perception, orientation, Rz, min_action, max_action, hidden):
        latent = self.wm_encoder.encode(perception, state, hidden)
        actor_input = torch.cat([latent, hidden, self.make_state_input(state)], dim=-1)
        raw_action = self.actor.forward_export(actor_input).tanh()
        hidden = self.wm_encoder.recurrent(latent, hidden, raw_action)
        action, quat_xyzw, acc_norm = self.post_process(raw_action, min_action, max_action, orientation, Rz)
        return action, quat_xyzw, acc_norm, hidden

    def post_process(self, raw_action, min_action, max_action, orientation, Rz):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        if self.action_frame == "local":
            return self.post_process_local(raw_action, min_action, max_action, orientation, Rz)
        elif self.action_frame == "world":
            return self.post_process_world(raw_action, min_action, max_action, orientation, Rz)
        else:
            raise ValueError(f"Unknown action frame: {self.action_frame}")
    
    def post_process_local(self, raw_action, min_action, max_action, orientation, Rz):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        action = (raw_action.tanh() * 0.5 + 0.5) * (max_action - min_action) + min_action
        acc_cmd = torch.matmul(Rz, action.unsqueeze(-1)).squeeze(-1)
        quat_xyzw = point_mass_quat(acc_cmd, orientation)
        acc_norm = acc_cmd.norm(p=2, dim=-1)
        return acc_cmd, quat_xyzw, acc_norm
    
    def post_process_world(self, raw_action, min_action, max_action, orientation, Rz):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        action = (raw_action.tanh() * 0.5 + 0.5) * (max_action - min_action) + min_action
        quat_xyzw = point_mass_quat(action, orientation)
        acc_norm = action.norm(p=2, dim=-1)
        return action, quat_xyzw, acc_norm
    
    def export(
        self,
        path: str,
        export_cfg: DictConfig,
        verbose=False,
    ):
        self.obs_frame = export_cfg.obs_frame
        self.action_frame = export_cfg.action_frame
        if export_cfg.jit:
            self.export_jit(path, verbose) # NOTE: failed because torch.jit dose not support einops
        if export_cfg.onnx:
            self.export_onnx(path)
    
    @torch.no_grad()
    def export_jit(self, path: str, verbose=False):
        names, inputs = zip(*self.named_inputs)
        shapes = [tuple(input.shape) for input in inputs]
        traced_script_module = torch.jit.script(self, optimize=True, example_inputs=shapes)
        if verbose:
            print(traced_script_module.code)
        export_path = os.path.join(path, "exported_actor.pt2")
        traced_script_module.save(export_path)
        print(f"The checkpoint is compiled and exported to {export_path}.")
    
    def export_onnx(self, path: str):
        export_path = os.path.join(path, "exported_actor.onnx")
        names, inputs = zip(*self.named_inputs)
        torch.onnx.export(
            model=self,
            args=inputs,
            f=export_path,
            input_names=names,
            output_names=self.output_names
        )
        print(f"The checkpoint is compiled and exported to {export_path}.")