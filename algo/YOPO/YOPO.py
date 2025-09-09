from typing import Tuple, Optional, Dict, Callable
import os
import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from tensordict import TensorDict
from pytorch3d import transforms as T
from omegaconf import DictConfig
import taichi as ti

from .functions import (
    post_process,
    get_coef_matrix,
    get_coef_matrices,
    solve_coef,
    get_traj_point,
    get_traj_points
)
from .network import YOPONet
from diffaero.env.obstacle_avoidance_yopo import ObstacleAvoidanceYOPO
from diffaero.dynamics.pointmass import point_mass_quat
from diffaero.utils.math import mvp, rk4, quat_rotate
from diffaero.utils.render import torch2ti
from diffaero.utils.runner import timeit
from diffaero.utils.logger import Logger

class YOPO(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        img_h: int,
        img_w: int,
        device: torch.device
    ):
        super().__init__()
        self.tmax: Tensor = torch.tensor(cfg.tmax, device=device)
        self.inv_coef_mat_tmax: Tensor = torch.inverse(get_coef_matrix(self.tmax)) # [6, 6]
        self.n_points_per_sec: int = cfg.n_points_per_sec
        self.n_points: int = int(self.n_points_per_sec * cfg.tmax)
        self.t_vec = torch.linspace(0, cfg.tmax, self.n_points, device=device)
        self.coef_mats: Tensor = get_coef_matrices(self.t_vec) # [n_points, 6, 6]
        self.min_pitch: float = cfg.min_pitch * torch.pi / 180.
        self.max_pitch: float = cfg.max_pitch * torch.pi / 180.
        self.min_yaw: float = cfg.min_yaw * torch.pi / 180.
        self.max_yaw: float = cfg.max_yaw * torch.pi / 180.
        self.n_pitch: int = cfg.n_pitch # H
        self.n_yaw: int = cfg.n_yaw     # W
        self.dpitch_range = (self.max_pitch - self.min_pitch) / (2 * (self.n_pitch - 1))
        self.dyaw_range = (self.max_yaw - self.min_yaw) / (2 * (self.n_yaw - 1))
        self.drpy_min = torch.tensor([cfg.r_min, self.dpitch_range, self.dyaw_range], device=device)
        self.drpy_max = torch.tensor([cfg.r_max, self.dpitch_range, self.dyaw_range], device=device)
        self.dv_range: float = cfg.dv_range
        self.da_range: float = cfg.da_range
        self.G = torch.tensor([[0., 0., 9.81]], device=device)
        self.gamma: float = cfg.gamma
        self.expl_prob: float = cfg.expl_prob
        self.grad_norm: Optional[float] = cfg.grad_norm
        self.t_next = torch.tensor(cfg.t_next, device=device) if cfg.t_next is not None else self.t_vec[1]
        self.device = device
        self.img_h = img_h
        self.img_w = img_w
        
        pitches = torch.linspace(self.min_pitch, self.max_pitch, self.n_pitch, device=device)
        yaws = torch.linspace(self.max_yaw, self.min_yaw, self.n_yaw, device=device)

        pitches, yaws = torch.meshgrid(pitches, yaws, indexing="ij")
        rolls = torch.zeros_like(pitches)
        self.euler_angles = torch.stack([yaws, pitches, rolls], dim=-1).reshape(-1, 3) # [n_pitch*n_yaw, 3]
        self.rpy_base = torch.stack([torch.zeros_like(pitches), pitches, yaws], dim=-1).reshape(-1, 3) # [n_pitch*n_yaw, 3]

        # convert coordinates from primitive frame to body frame
        self.rotmat_p2b = T.euler_angles_to_matrix(self.euler_angles, convention="ZYX")
        # convert coordinates from body frame to primitive frame
        self.rotmat_b2p = self.rotmat_p2b.transpose(-2, -1)
        
        self.net = YOPONet(
            H_out=self.n_pitch,
            W_out=self.n_yaw,
            feature_dim=cfg.feature_dim,
            head_hidden_dim=cfg.head_hidden_dim,
            out_dim=10
        ).to(device)
        # self.net = torch.compile(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
    
    def body2primitive(self, vec_b: Tensor):
        return mvp(self.rotmat_b2p, vec_b)

    def primitive2body(self, vec_p: Tensor):
        return mvp(self.rotmat_p2b, vec_p)

    @timeit
    def inference(self, obs: TensorDict) -> Tuple[Tensor, Tensor]:
        depth_image = obs["perception"]
        target_vel_b, v_curr_b, a_curr_b = obs["state"][..., :3], obs["state"][..., 3:6], obs["state"][..., 6:9]
        p_curr_b = torch.zeros_like(v_curr_b)
        
        rotmat_b2p = self.rotmat_b2p.unsqueeze(0)
        target_vel_p = mvp(rotmat_b2p, target_vel_b.unsqueeze(1)) # [N, HW, 3]
        v_curr_p = mvp(rotmat_b2p, v_curr_b.unsqueeze(1)) # [N, HW, 3]
        a_curr_p = mvp(rotmat_b2p, a_curr_b.unsqueeze(1)) # [N, HW, 3]
        state_input = torch.cat([target_vel_p, v_curr_p, a_curr_p], dim=-1) # [N, HW, 9]

        net_output: Tensor = self.net(depth_image, state_input) # [N, HW, 10]
        p_end_b, v_end_b, a_end_b, score = post_process( # [N, HW, (3, 3, 3)], [N, HW]
            output=net_output,
            rpy_base=self.rpy_base,
            drpy_min=self.drpy_min,
            drpy_max=self.drpy_max,
            dv_range=self.dv_range,
            da_range=self.da_range,
            rotmat_p2b=self.rotmat_p2b
        )
        coef_xyz = solve_coef( # [N, HW, 6, 3]
            self.inv_coef_mat_tmax[None, None, ...], # [1, 1, 6, 6]
            p_curr_b.unsqueeze(1).expand_as(p_end_b), # [N, HW, 3]
            v_curr_b.unsqueeze(1).expand_as(v_end_b), # [N, HW, 3]
            a_curr_b.unsqueeze(1).expand_as(a_end_b), # [N, HW, 3]
            p_end_b, # [N, HW, 3]
            v_end_b, # [N, HW, 3]
            a_end_b  # [N, HW, 3]
        )
        return score, coef_xyz
    
    def act(self, obs: TensorDict, test: bool = False):
        self.eval()
        quat_xyzw = obs["state"][..., 9:13]
        if test:
            obs["state"][..., 6:9] = 0.
        N, HW = quat_xyzw.size(0), self.n_pitch * self.n_yaw
        score, coef_xyz = self.inference(obs)
        best_idx = score.argmax(dim=-1) # [N, ]
        if not test:
            random_idx = torch.randint(0, HW, (N, ), device=self.device)
            use_random = torch.rand(N, device=self.device) < self.expl_prob
            patch_index = torch.where(use_random, random_idx, best_idx)
        else:
            patch_index = best_idx
        patch_index = patch_index.reshape(N, 1, 1, 1).expand(-1, -1, 6, 3)
        coef_best = torch.gather(coef_xyz, 1, patch_index).squeeze(1) # [N, 6, 3]

        p_next_b, v_next_b, a_next_b = get_traj_point(self.t_next, coef_best) # [N, 3]
        a_next_w = quat_rotate(quat_xyzw, a_next_b) + self.G
        policy_info = {
            "traj_coef": coef_xyz,
            "best_coef": coef_best,
            "best_idx": best_idx
        }
        return a_next_w, policy_info

    def render_trajectories(self, env, policy_info, p_w, rotmat_b2w):
        # type: (ObstacleAvoidanceYOPO, Dict[str, Tensor], Tensor, Tensor) -> None
        if env.renderer is not None and env.renderer.enable_rendering and not env.renderer.headless:
            renderer_n_envs = env.renderer.n_envs
            if not hasattr(self, "lines_tensor"):
                self.lines_tensor = torch.empty(renderer_n_envs, self.n_pitch*self.n_yaw, self.n_points-1, 2, 3, device=self.device) # [N, HW, T-1, 2, 3]
                self.lines_field = ti.Vector.field(3, dtype=ti.f32, shape=(renderer_n_envs * (self.n_pitch*self.n_yaw) * (self.n_points-1) * 2))
                self.color_tensor = torch.ones(renderer_n_envs, self.n_pitch*self.n_yaw, self.n_points-1, 2, 3, device=self.device) # [N, HW, T-1, 2, 3]
                self.color_field = ti.Vector.field(3, dtype=ti.f32, shape=(renderer_n_envs * (self.n_pitch*self.n_yaw) * (self.n_points-1) * 2))
            
            p_traj_b, _, _ = get_traj_points(self.coef_mats, policy_info["traj_coef"]) # [N, HW, T, 3]
            p_traj_w = mvp(rotmat_b2w.unsqueeze(1).unsqueeze(1), p_traj_b) + p_w.unsqueeze(1).unsqueeze(1) # [N, HW, T, 3]
            p_traj_render = p_traj_w[:renderer_n_envs].to(torch.float32) + env.renderer.env_origin.unsqueeze(1).unsqueeze(1) # [N, HW, T, 3]
            self.lines_tensor[..., 0, :] = p_traj_render[..., :-1, :]
            self.lines_tensor[..., 1, :] = p_traj_render[..., 1:, :]
            self.lines_field.from_torch(torch2ti(self.lines_tensor.flatten(end_dim=-2)))
            
            best_idx = policy_info["best_idx"][:renderer_n_envs] # [N, ]
            env_idx = torch.arange(renderer_n_envs, device=self.device)
            self.color_tensor.fill_(1.)
            self.color_tensor[env_idx, best_idx] = torch.tensor([0., 1., 0.], device=self.device).view(1, 1, 1, 3)
            self.color_field.from_torch(self.color_tensor.flatten(end_dim=-2))
            
            env.renderer.gui_scene.lines(self.lines_field, per_vertex_color=self.color_field, width=3.)

    @timeit
    def step(self, cfg: DictConfig, env: ObstacleAvoidanceYOPO, logger: Logger, obs: TensorDict, on_step_cb=None):
        N, HW, T = env.n_envs, self.n_pitch * self.n_yaw, self.n_points - 1
        p_w, rotmat_b2w = env.p, env.dynamics.R
        
        self.train()
        for _ in range(cfg.algo.n_epochs):
            # traverse the trajectory and cumulate the loss
            score, coef_xyz = self.inference(obs) # [N, HW, 6, 3]
            
            p_traj_b, v_traj_b, a_traj_b = get_traj_points(self.coef_mats[1:], coef_xyz) # [N, HW, T, 3]
            p_traj_w = mvp(rotmat_b2w.unsqueeze(1), p_traj_b.reshape(N, HW*T, 3)) + p_w.unsqueeze(1)
            v_traj_w = mvp(rotmat_b2w.unsqueeze(1), v_traj_b.reshape(N, HW*T, 3))
            a_traj_w = mvp(rotmat_b2w.unsqueeze(1), a_traj_b.reshape(N, HW*T, 3)) + self.G.unsqueeze(1)
            _, traj_reward, _, dead, _ = env.reward_fn(p_traj_w, v_traj_w, a_traj_w)
            traj_reward, dead = traj_reward.reshape(N, HW, T), dead.reshape(N, HW, T).float().cumsum(dim=2)
            survive = (dead == 0.).float()
            traj_reward_avg = torch.sum(traj_reward * survive, dim=-1) / survive.sum(dim=-1).clamp(min=1.)
            score_loss = F.mse_loss(score, traj_reward_avg.detach())
            
            total_loss = -traj_reward_avg.mean() + 0.01 * score_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        if self.grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.grad_norm)
        else:
            grads = [p.grad for p in self.net.parameters() if p.grad is not None]
            grad_norm = torch.nn.utils.get_total_norm(grads)
        
        with torch.no_grad():
            action, policy_info = self.act(obs)
            self.render_trajectories(env, policy_info, p_w, rotmat_b2w)
            next_obs, (_, _), terminated, env_info = env.step(action)
            if on_step_cb is not None:
                on_step_cb(
                    obs=obs,
                    action=action,
                    policy_info=policy_info,
                    env_info=env_info)
        
        losses = {
            "traj_reward": traj_reward.mean().item(),
            "score_loss": score_loss.item(),
            "total_loss": total_loss.item()}
        grad_norms = {"actor_grad_norm": grad_norm}
        
        return next_obs, policy_info, env_info, losses, grad_norms

    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.net.state_dict(), os.path.join(path, "network.pth"))
    
    def load(self, path: str):
        self.net.load_state_dict(torch.load(os.path.join(path, "network.pth")))
    
    @staticmethod
    def build(cfg: DictConfig, env: ObstacleAvoidanceYOPO, device: torch.device):
        return YOPO(
            cfg=cfg,
            img_h=env.sensor.H,
            img_w=env.sensor.W,
            device=device
        )
    
    def forward(self, state: Tensor, perception: Tensor, orientation: Tensor):
        target_vel_b, v_curr_b, a_curr_b, quat_xyzw = state[:, :3], state[:, 3:6], state[:, 6:9], state[:, 9:13]
        
        rotmat_b2p = self.rotmat_b2p.unsqueeze(0)
        target_vel_p = mvp(rotmat_b2p, target_vel_b.unsqueeze(1)) # [1, HW, 3]
        p_curr_b = torch.zeros_like(v_curr_b)
        v_curr_p = mvp(rotmat_b2p, v_curr_b.unsqueeze(1)) # [1, HW, 3]
        a_curr_p = mvp(rotmat_b2p, a_curr_b.unsqueeze(1)) # [1, HW, 3]
        state_input = torch.cat([target_vel_p, v_curr_p, a_curr_p], dim=-1) # [1, HW, 9]

        net_output: Tensor = self.net(perception.unsqueeze(1), state_input) # [1, HW, 10]
        p_end_b, v_end_b, a_end_b, score = post_process( # [1, HW, (3, 3, 3)], [N, HW]
            output=net_output,
            rpy_base=self.rpy_base,
            drpy_min=self.drpy_min,
            drpy_max=self.drpy_max,
            dv_range=self.dv_range,
            da_range=self.da_range,
            rotmat_p2b=self.rotmat_p2b
        )
        coef_xyz = solve_coef( # [1, HW, 6, 3]
            self.inv_coef_mat_tmax[None, None, ...], # [1, 1, 6, 6]
            p_curr_b.unsqueeze(1).expand_as(p_end_b), # [1, HW, 3]
            v_curr_b.unsqueeze(1).expand_as(v_end_b), # [1, HW, 3]
            a_curr_b.unsqueeze(1).expand_as(a_end_b), # [1, HW, 3]
            p_end_b, # [1, HW, 3]
            v_end_b, # [1, HW, 3]
            a_end_b  # [1, HW, 3]
        )
        
        best_idx = score.argmax(dim=-1) # [1, ]
        best_idx = best_idx.reshape(-1, 1, 1, 1).expand(-1, -1, 6, 3)
        coef_best = torch.gather(coef_xyz, 1, best_idx).squeeze(1) # [1, 6, 3]

        p_next_b, v_next_b, a_next_b = get_traj_point(self.t_next, coef_best) # [1, 3]
        a_next_w = quat_rotate(quat_xyzw, a_next_b) + self.G
        
        quat_xyzw = point_mass_quat(a_next_w, orientation)
        acc_norm = a_next_w.norm(p=2, dim=-1)
        return a_next_w, quat_xyzw, acc_norm

    def export(
        self,
        path: str,
        export_cfg: DictConfig,
        verbose: bool = False,
    ):
        example_input = {
            "state": torch.randn(1, 13, device=self.device),
            "perception": torch.randn(1, self.img_h, self.img_w, device=self.device),
            "orientation": torch.randn(1, 3, device=self.device)
        }
        if export_cfg.onnx:
            export_path = os.path.join(path, "exported_actor.onnx")
            names, test_inputs = zip(*example_input.items())
            torch.onnx.export(
                model=self,
                args=test_inputs,
                f=export_path,
                input_names=names,
                output_names=["action", "quat_xyzw_cmd", "acc_norm"]
            )
            Logger.info(f"The checkpoint is compiled and exported to {export_path}.")