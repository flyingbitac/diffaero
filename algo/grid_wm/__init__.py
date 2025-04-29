from typing import Tuple, Dict, Union, Optional, List
import math
import os
from copy import deepcopy

from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn
import tensordict
from tensordict import TensorDict

from quaddif.env.obstacle_avoidance import ObstacleAvoidanceGrid
from quaddif.dynamics.pointmass import point_mass_quat
from quaddif.algo.buffer import RolloutBufferGRID
from quaddif.network.agents import  StochasticActor
from quaddif.utils.runner import timeit
from .world.backbone import WorldModel, WorldModelTesttime


class GRIDWM:
    def __init__(
        self,
        cfg: DictConfig,
        obs_dim: Tuple[int, Tuple[int, int]],
        action_dim: int,
        grid_cfg: DictConfig,
        device: torch.device
    ):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.l_rollout: int = cfg.l_rollout
        self.batch_size: int = cfg.wm.train.batch_size
        self.n_epochs: int = cfg.wm.train.n_epochs
        self.grid_points: List[int] = grid_cfg.n_points
        self.n_grid_points = math.prod(self.grid_points)
        
        self.input_target_vel: bool = cfg.actor.input_target_vel
        self.input_quat: bool = cfg.actor.input_quat
        self.input_vel: bool = cfg.actor.input_vel
        
        # world model
        self.wm = WorldModel(obs_dim, cfg.wm, grid_cfg).to(device)
        if cfg.wm.compile:
            self.wm = torch.compile(self.wm, mode="reduce-overhead")
        # replay buffer
        self.buffer=RolloutBufferGRID(
            l_rollout=self.l_rollout,
            buffer_size=int(cfg.wm.train.buffer_size),
            obs_dim=obs_dim,
            action_dim=action_dim,
            grid_dim=self.n_grid_points,
            device=device
        )
        # actor
        actor_input_dim = (
            self.wm.deter_dim + self.wm.latent_dim + 
            self.input_target_vel * 3 + self.input_quat * 4 + self.input_vel * 3
        )
        self.actor = StochasticActor(cfg.actor.network, actor_input_dim, action_dim).to(device)
        # optimizer
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.lr)
        
        self.entropy_weight: float = cfg.actor.entropy_weight
        self.max_grad_norm: float = cfg.actor.max_grad_norm
        self.entropy_loss = torch.zeros(1, device=device)
        self.actor_loss = torch.zeros(1, device=device)
        self.device = device
        self.deter: Tensor = None
    
    def make_state_input(self, obs: TensorDict) -> Tensor:
        state = obs["state"]
        target_vel, quat, vel = state[..., 0:3], state[..., 3:7], state[..., 7:10]
        inputs = []
        if self.input_target_vel:
            inputs.append(target_vel)
        if self.input_quat:
            inputs.append(quat)
        if self.input_vel:
            inputs.append(vel)
        return torch.cat(inputs, dim=-1)
    
    @timeit
    def act(self, obs, test=False):
        # type: (TensorDict, bool) -> Tuple[Tensor, Dict[str, Tensor]]
        with torch.no_grad():
            if self.deter is None:
                self.deter = torch.zeros(obs['state'].shape[0], self.wm.deter_dim, device=obs['state'].device)
            latent = self.wm.encode(obs['perception'], obs['state'], self.deter)
            actor_input = torch.cat([latent, self.deter, self.make_state_input(obs)], dim=-1)
        action, sample, logprob, entropy = self.actor(actor_input, test=test)
        self.deter = self.wm.recurrent(latent, self.deter, action)
        return action, {"latent": latent, "sample": sample, "logprob": logprob, "entropy": entropy}
    
    def record_loss(self, loss, policy_info, env_info):
        # type: (Tensor, Dict[str, Tensor], Dict[str, Tensor]) -> None
        self.actor_loss += loss.mean()
        self.entropy_loss -= policy_info["entropy"].mean()
    
    @timeit
    def update_wm(self):
        if not self.buffer.size >= self.batch_size:
            return {}, {}
        for _ in range(self.n_epochs):
            observations, actions, dones, rewards = self.buffer.sample(self.batch_size)
            # find ground truth and visible grid
            ground_truth_grid = observations["grid"]
            visible_map = observations["visible_map"]
            total_loss, grad_norms, grid_pred = self.wm.update(
                obs=observations['perception'],
                state=observations['state'],
                actions=actions,
                rewards=rewards,
                terminals=dones,
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
    
    @timeit
    def update_actor(self):
        # type: () -> Tuple[Dict[str, float], Dict[str, float]]
        actor_loss = self.actor_loss / self.l_rollout
        entropy_loss = self.entropy_loss / self.l_rollout
        total_loss = actor_loss + self.entropy_weight * entropy_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.actor.parameters()]) ** 0.5
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        self.actor_loss = torch.zeros(1, device=self.device)
        self.entropy_loss = torch.zeros(1, device=self.device)
        return {"actor_loss": actor_loss.mean().item(), "entropy_loss": entropy_loss.mean().item()}, {"actor_grad_norm": grad_norm}
    
    @timeit
    def step(self, cfg: DictConfig, env: ObstacleAvoidanceGrid, obs: TensorDict, on_step_cb=None):
        # env.prev_visible_map.fill_(False) # clear the memory that wm shouldn't have
        rollout_obs, rollout_dones, rollout_actions, rollout_rewards = [], [], [], []
        for _ in range(self.l_rollout):
            action, policy_info = self.act(obs)
            next_obs, loss, terminated, env_info = env.step(env.rescale_action(action), need_obs_before_reset=False)
            self.reset(env_info['reset'])
            self.record_loss(loss, policy_info, env_info)
            rollout_obs.append(obs)
            rollout_actions.append(action)
            rollout_dones.append(terminated)
            rollout_rewards.append(10.*(1. - 0.1*loss).detach())
            obs = next_obs
            if on_step_cb is not None:
                on_step_cb(
                    obs=obs,
                    action=action,
                    policy_info=policy_info,
                    env_info=env_info)
        self.buffer.add(
            obs=tensordict.stack(rollout_obs, dim=1),
            action=torch.stack(rollout_actions, dim=1),
            done=torch.stack(rollout_dones, dim=1),
            reward=torch.stack(rollout_rewards, dim=1)
        )
        wm_losses, wm_grad_norm, selected_grid_gt, selected_grid_pred = self.update_wm()
        actor_losses, actor_grad_norms = self.update_actor()
        losses = {**wm_losses, **actor_losses}
        grad_norms = {**wm_grad_norm, **actor_grad_norms}
        self.detach()
        if selected_grid_gt is not None and selected_grid_pred is not None:
            policy_info.update({
                "grid_gt": selected_grid_gt,
                "grid_pred": selected_grid_pred
            })
        return obs, policy_info, env_info, losses, grad_norms
    
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
    
    @staticmethod
    def build(cfg: DictConfig, env: ObstacleAvoidanceGrid, device: torch.device):
        if hasattr(env.cfg, "grid"):
            return GRIDWM(
                cfg=cfg,
                obs_dim=env.obs_dim,
                action_dim=env.action_dim,
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
        export_jit,
        export_onnx,
        verbose=False,
    ):
        testtime = GRIDWMTesttime(self.cfg, self.obs_dim, self.action_dim, self.device)
        testtime.load(path)
        testtime.export(path, export_jit, export_onnx, verbose)


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
        
        self.input_target_vel: bool = cfg.actor.input_target_vel
        self.input_quat: bool = cfg.actor.input_quat
        self.input_vel: bool = cfg.actor.input_vel
        self.state_dim = self.input_target_vel * 3 + self.input_quat * 4 + self.input_vel * 3
        actor_input_dim = self.wm.deter_dim + self.wm.latent_dim + self.state_dim
        
        self.actor = StochasticActor(cfg.actor.network, actor_input_dim, action_dim).to(device)
        self.device = device
        self.deter: Tensor = None
    
    def make_state_input(self, obs: TensorDict) -> Tensor:
        state = obs["state"]
        target_vel, quat, vel = state[..., 0:3], state[..., 3:7], state[..., 7:10]
        inputs = []
        if self.input_target_vel:
            inputs.append(target_vel)
        if self.input_quat:
            inputs.append(quat)
        if self.input_vel:
            inputs.append(vel)
        return torch.cat(inputs, dim=-1)
    
    @timeit
    def act(self, obs, test=False):
        # type: (TensorDict, bool) -> Tuple[Tensor, Dict[str, Tensor]]
        with torch.no_grad():
            if self.deter is None:
                self.deter = torch.zeros(obs['state'].shape[0], self.wm.deter_dim, device=obs['state'].device)
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
        export_jit,
        export_onnx,
        verbose=False,
    ):
        GRIDWMExporter(self).export(path, export_jit, export_onnx, verbose)


class GRIDWMExporter(nn.Module):
    def __init__(self, agent: GRIDWMTesttime):
        super().__init__()
        self.wm_encoder = deepcopy(agent.wm).cpu()
        self.actor = deepcopy(agent.actor.actor_mean).cpu()
        state_dim, perception_dim = agent.state_dim, agent.obs_dim[1]
        self.named_inputs = [
            ("state", torch.rand(1, state_dim)),
            ("perception", torch.rand(1, perception_dim[0], perception_dim[1])),
            ("orientation", torch.rand(1, 3)),
            ("min_action", torch.rand(1, 3)),
            ("max_action", torch.rand(1, 3)),
            ("hidden_in", torch.rand(1, self.wm_encoder.deter_dim)),
        ]
        self.output_names = [
            "action",
            "quat_xyzw_cmd",
            "acc_norm",
            "hidden_out"
        ]
    
    def forward(self, state, perception, orientation, min_action, max_action, hidden):
        latent = self.wm_encoder.encode(perception, state, hidden)
        actor_input = torch.cat([latent, hidden, state], dim=-1)
        raw_action = self.actor.forward_export(actor_input).tanh()
        hidden = self.wm_encoder.recurrent(latent, hidden, raw_action)
        action, quat_xyzw, acc_norm = self.post_process(raw_action, min_action, max_action, orientation=orientation)
        return action, quat_xyzw, acc_norm, hidden

    def post_process(self, raw_action, min_action, max_action, orientation):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        action = (raw_action * 0.5 + 0.5) * (max_action - min_action) + min_action
        quat_xyzw = point_mass_quat(action, orientation)
        acc_norm = action.norm(p=2, dim=-1)
        return action, quat_xyzw, acc_norm
    
    def export(
        self,
        path: str,
        export_jit,
        export_onnx,
        verbose=False,
    ):
        if export_jit:
            self.export_jit(path, verbose) # NOTE: failed because torch.jit dose not support einops
        if export_onnx:
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