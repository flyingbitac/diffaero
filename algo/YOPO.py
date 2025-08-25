from typing import Tuple, Optional
import os
import sys
sys.path.append('..')

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d import transforms as T
from omegaconf import DictConfig
import taichi as ti

from diffaero.env.obstacle_avoidance_yopo import ObstacleAvoidanceYOPO
from diffaero.utils.math import mvp
from diffaero.utils.render import torch2ti
from diffaero.utils.runner import timeit
from diffaero.utils.logger import Logger

class YOPONet(nn.Module):
    def __init__(
        self,
        H_out: int,
        W_out: int,
        feature_dim: int,
        head_hidden_dim: int,
        out_dim: int = 10
    ):
        super().__init__()
        self.H_out = H_out
        self.W_out = W_out
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ELU(),
            # nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            # nn.ELU(),
            # nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            # nn.ELU(),
            nn.Conv2d(16, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((H_out, W_out))
        )
        self.head = nn.Sequential(
            nn.Conv2d(feature_dim+9, head_hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.Conv2d(head_hidden_dim, head_hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.Conv2d(head_hidden_dim, out_dim, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(
        self,
        rotmat_b2p: Tensor,
        depth_image: Tensor,
        obs_b: Tensor # []
    ):
        N, C, HW = obs_b.size(0), self.out_dim, self.H_out * self.W_out
        feat = self.net(depth_image)
        obs_p = torch.matmul(rotmat_b2p.unsqueeze(0), obs_b.unsqueeze(1))
        obs_p = obs_p.reshape(N, self.H_out, self.W_out, 9).permute(0, 3, 1, 2)
        feat = torch.cat([feat, obs_p], dim=1)
        return self.head(feat).reshape(N, C, HW).permute(0, 2, 1)

@torch.jit.script
def rpy2xyz(rpy: Tensor) -> Tensor:
    """
    Convert radius, pitch and yaw angles to xyz coordinates.
    """
    r, pitch, yaw = rpy.unbind(dim=-1)
    return torch.stack([
        r * torch.cos(pitch) * torch.cos(yaw),
        r * torch.cos(pitch) * torch.sin(yaw),
        r * -torch.sin(pitch)
    ], dim=-1)

@torch.jit.script
def post_process(
    output: Tensor,
    rpy_base: Tensor,
    dr_range: float,
    dpitch_range: float,
    dyaw_range: float,
    dv_range: float,
    da_range: float,
    rotmat_p2b: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    d_rpy, v_p, a_p, score = output.chunk(4, dim=-1) # #channels: [3, 3, 3, 1]
    rpy_range = torch.tensor([dr_range, dpitch_range, dyaw_range], device=rpy_base.device)
    rpy = rpy_base.unsqueeze(0) + torch.tanh(d_rpy) * rpy_range.expand_as(d_rpy)
    # print(rpy.view(-1, 3).min(dim=0).values, rpy.view(-1, 3).max(dim=0).values)
    p_b = rpy2xyz(rpy)
    v_p, a_p = dv_range * torch.tanh(v_p), da_range * torch.tanh(a_p)
    v_b = torch.matmul(rotmat_p2b, v_p.unsqueeze(-1)).squeeze(-1)
    a_b = torch.matmul(rotmat_p2b, a_p.unsqueeze(-1)).squeeze(-1)
    return p_b, v_b, a_b, score.squeeze(-1)

@torch.jit.script
def get_coef_matrix(t: Tensor) -> Tensor:
    device = t.device
    t = float(t.item())
    coef_mat = torch.tensor([
        [1., 0.,   0.,     0.,      0.,      0.],
        [0., 1.,   0.,     0.,      0.,      0.],
        [0., 0.,   2.,     0.,      0.,      0.],
        [1.,  t, t**2,   t**3,    t**4,    t**5],
        [0., 1.,  2*t, 3*t**2,  4*t**3,  5*t**4],
        [0., 0.,   2.,    6*t, 12*t**2, 20*t**3],
    ], device=device)
    return coef_mat

@torch.jit.script
def get_coef_matrices(t_vec: Tensor) -> Tensor:
    coef_mats = torch.stack([get_coef_matrix(t) for t in t_vec], dim=0)
    return coef_mats

@torch.jit.script
def solve_coef(
    tmax: Tensor,
    p0: Tensor,
    v0: Tensor,
    a0: Tensor,
    pt: Tensor,
    vt: Tensor,
    at: Tensor
):
    pvapva = torch.stack([p0, v0, a0, pt, vt, at], dim=-2) # [..., 6, 3]
    coef_mat_inv = torch.inverse(get_coef_matrix(tmax)) # [6, 6]
    coef_xyz = torch.matmul(coef_mat_inv, pvapva) # [..., 6, 3]
    return coef_xyz

@torch.jit.script
def get_traj_point(
    t: Tensor,
    coef_xyz: Tensor # [..., 6, 3]
):
    coef_mat = get_coef_matrix(t)[3:, :] # [3, 6]
    pva = torch.matmul(coef_mat, coef_xyz) # [..., 3(pva), 3(xyz)]
    p, v, a = pva.unbind(dim=-2) # [..., 3(xyz)]
    return p, v, a

@torch.jit.script
def get_traj_points(
    t_vec: Tensor, # [T]
    coef_xyz: Tensor # [..., 6, 3]
):
    coef_mat = get_coef_matrices(t_vec)[..., 3:, :] # [T, 3, 6]
    pva = torch.matmul(coef_mat.unsqueeze(0), coef_xyz.unsqueeze(1)) # [..., T, 3(pva), 3(xyz)]
    p, v, a = pva.unbind(dim=-2) # [..., T, 3(xyz)]
    return p, v, a

class YOPO:
    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device
    ):
        self.tmax: Tensor = torch.tensor(cfg.tmax, device=device)
        self.n_points_per_sec: int = cfg.n_points_per_sec
        self.n_points: int = int(self.n_points_per_sec * cfg.tmax)
        self.t_vec = torch.linspace(0, cfg.tmax, self.n_points, device=device)
        self.min_pitch: float = cfg.min_pitch * torch.pi / 180.
        self.max_pitch: float = cfg.max_pitch * torch.pi / 180.
        self.min_yaw: float = cfg.min_yaw * torch.pi / 180.
        self.max_yaw: float = cfg.max_yaw * torch.pi / 180.
        self.n_pitch: int = cfg.n_pitch
        self.n_yaw: int = cfg.n_yaw
        self.n_optim_steps: int = cfg.n_optim_steps
        self.r_base: float = cfg.r
        self.dr_range: float = self.r_base
        self.dpitch_range: float = cfg.dpitch_range * torch.pi / 180.
        self.dyaw_range: float = cfg.dyaw_range * torch.pi / 180.
        self.dv_range: float = cfg.dv_range
        self.da_range: float = cfg.da_range
        self.G = torch.tensor([[0., 0., 9.81]], device=device)
        self.gamma: float = cfg.gamma
        self.expl_prob: float = cfg.expl_prob
        self.grad_norm: Optional[float] = cfg.grad_norm
        self.optimized_inference: bool = cfg.optimized_inference
        self.real_dynamics_rollout: bool = cfg.real_dynamics_rollout
        self.update_best_traj_only: bool = cfg.update_best_traj_only
        self.t_next = torch.tensor(cfg.t_next, device=device) if cfg.t_next is not None else self.t_vec[1]
        self.device = device
        
        pitches = torch.linspace(self.min_pitch, self.max_pitch, self.n_pitch, device=device)
        yaws = torch.linspace(self.max_yaw, self.min_yaw, self.n_yaw, device=device)

        pitches, yaws = torch.meshgrid(pitches, yaws, indexing="ij")
        rolls = torch.zeros_like(pitches)
        self.euler_angles = torch.stack([yaws, pitches, rolls], dim=-1).reshape(-1, 3)
        self.rpy_base = torch.stack([torch.full_like(yaws, self.r_base), pitches, yaws], dim=-1).reshape(-1, 3)

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
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
    
    @timeit
    def inference(self, p_w, rotmat_b2w, v_w, a_w, target_vel_w, depth_image):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
        rotmat_w2b = rotmat_b2w.transpose(-2, -1)
        target_vel_b = mvp(rotmat_w2b, target_vel_w)
        p_curr_b = torch.zeros_like(p_w)
        v_curr_b = mvp(rotmat_w2b, v_w)
        a_curr_b = mvp(rotmat_w2b, a_w) - self.G
        
        obs = torch.stack([target_vel_b, v_curr_b, a_curr_b], dim=-1)
        net_output: Tensor = self.net(self.rotmat_b2p, depth_image, obs)
        p_end_b, v_end_b, a_end_b, score = post_process(
            output=net_output,
            rpy_base=self.rpy_base,
            dr_range=self.dr_range,
            dpitch_range=self.dpitch_range,
            dyaw_range=self.dyaw_range,
            dv_range=self.dv_range,
            da_range=self.da_range,
            rotmat_p2b=self.rotmat_p2b
        )
        coef_xyz = solve_coef( # [N, HW, 6, 3]
            self.tmax,
            p_curr_b.unsqueeze(1).expand_as(p_end_b),
            v_curr_b.unsqueeze(1).expand_as(v_end_b),
            a_curr_b.unsqueeze(1).expand_as(a_end_b),
            p_end_b,
            v_end_b,
            a_end_b
        )
        return score, coef_xyz
    
    def act(self, obs: Tuple[Tensor, ...], test: bool = False, env: Optional[ObstacleAvoidanceYOPO] = None):
        p_w, rotmat_b2w, v_w, a_w, target_vel_w, depth_image = obs
        rotmat_w2b = rotmat_b2w.transpose(-2, -1)
        N, HW = rotmat_b2w.size(0), self.n_pitch * self.n_yaw
        
        score, coef_xyz = self.inference(p_w, rotmat_b2w, v_w, a_w, target_vel_w, depth_image)

        best_idx = score.argmin(dim=-1)
        if not test:
            random_idx = torch.randint(0, HW, (N, ), device=self.device)
            use_random = torch.rand(N, device=self.device) < self.expl_prob
            grid_index = torch.where(use_random, random_idx, best_idx)
        else:
            grid_index = best_idx
        grid_index = grid_index.reshape(N, 1, 1, 1).expand(-1, -1, 6, 3)
        coef_best = torch.gather(coef_xyz, 1, grid_index).squeeze(1)
        
        if env is not None and self.optimized_inference:
            losses = []
            for _ in range(self.n_optim_steps):
                coef_best = coef_best.detach().requires_grad_(True)
                # pva_w = torch.cat([p_w, v_w, a_w], dim=-1).unsqueeze(-2).expand(-1, len(self.t_vec)-1, -1)
                
                p_t_b, v_t_b, a_t_b = get_traj_points(self.t_vec[1:], coef_best)
                p_t_w = mvp(rotmat_w2b, p_t_b) + p_w.unsqueeze(1)
                v_t_w = mvp(rotmat_w2b, v_t_b)
                a_t_w = mvp(rotmat_w2b, a_t_b) + self.G.unsqueeze(1)
                # pva_w = self.dynamics_step(env, pva_w, a_t_w)
                # loss, _, dead = env.loss_fn(*pva_w.chunk(3, dim=-1))
                loss, _, dead = env.loss_fn(p_t_w, v_t_w, a_t_w)
                loss.mean().backward()
                losses.append(loss.mean().item())
                coef_best = coef_best - 1 * coef_best.grad
                
            Logger.debug(f"Losses: {losses[0]-losses[-1]:.4f}")
        
        p_next_b, v_next_b, a_next_b = get_traj_point(self.t_next, coef_best) # [N, 3]
        a_next_w = torch.matmul(rotmat_b2w, a_next_b.unsqueeze(-1)).squeeze(-1)
        acc = a_next_w + self.G
        return acc, {"coef_best": coef_best}

    def dynamics_step(self, env: ObstacleAvoidanceYOPO, pva: Tensor, a_next: Tensor):
        pva_next = env.dynamics.solver(env.dynamics.dynamics, pva, a_next, dt=1./self.n_points_per_sec, M=4)
        return pva_next

    @timeit
    def step(self, cfg: DictConfig, env: ObstacleAvoidanceYOPO, logger: Logger, obs: Tuple[Tensor], on_step_cb=None):
        N, HW = env.n_envs, self.n_pitch * self.n_yaw
        
        p_w, quat_xyzw, v_w, a_w, target_vel_w, depth_image = obs
        rotmat_b2w = T.quaternion_to_matrix(quat_xyzw.roll(1, dims=-1))
        
        for _ in range(cfg.algo.n_epochs):
            # traverse the trajectory and cumulate the loss
            score, coef_xyz = self.inference(p_w, quat_xyzw, v_w, a_w, target_vel_w, depth_image)
            
            if self.update_best_traj_only:
                best_idx = score.argmin(dim=-1).reshape(N, 1, 1, 1).expand(-1, -1, 6, 3)
                coef_xyz = torch.gather(coef_xyz, 1, best_idx)
                cumulative_loss = torch.zeros(N, 1, device=self.device)
                survive = torch.ones(N, 1, device=self.device, dtype=torch.bool)
                survive_steps = torch.ones(N, 1, device=self.device)
                pva_w = torch.cat([p_w, v_w, a_w], dim=-1).unsqueeze(-2)
            else:
                cumulative_loss = torch.zeros(N, HW, device=self.device)
                survive = torch.ones(N, HW, device=self.device, dtype=torch.bool)
                survive_steps = torch.ones(N, HW, device=self.device)
                pva_w = torch.cat([p_w, v_w, a_w], dim=-1).unsqueeze(-2).expand(-1, HW, -1)
            
            for i, t in enumerate(self.t_vec[1:]):
                p_t_b, v_t_b, a_t_b = get_traj_point(t, coef_xyz) # [N, 3]
                p_t_w = torch.matmul(rotmat_b2w.unsqueeze(1), p_t_b.unsqueeze(-1)).squeeze(-1) + p_w.unsqueeze(1)
                v_t_w = torch.matmul(rotmat_b2w.unsqueeze(1), v_t_b.unsqueeze(-1)).squeeze(-1)
                a_t_w = torch.matmul(rotmat_b2w.unsqueeze(1), a_t_b.unsqueeze(-1)).squeeze(-1) + self.G.unsqueeze(0)
                if self.real_dynamics_rollout:
                    pva_w = self.dynamics_step(env, pva_w, a_t_w)
                    loss, _, dead = env.loss_fn(*pva_w.chunk(3, dim=-1))
                else:
                    loss, _, dead = env.loss_fn(p_t_w, v_t_w, a_t_w)
                survive = survive & ~dead
                survive_steps += survive.float()
                cumulative_loss = cumulative_loss + loss * survive.float() * self.gamma ** i
            cumulative_loss = cumulative_loss / survive_steps
            score_loss = F.mse_loss(score, cumulative_loss.detach())
            cumulative_loss = cumulative_loss.mean()
            total_loss = cumulative_loss + score_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        grad_norm = sum([p.grad.data.norm().item() ** 2 for p in self.net.parameters()]) ** 0.5
        if self.grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.grad_norm)
        
        # with torch.no_grad():
        action, policy_info = self.act(obs, env=env)
        # render the trajectory
        if env.renderer is not None and env.renderer.enable_rendering and not env.renderer.headless:
            n_envs = env.renderer.n_envs
            lines_tensor = torch.zeros(n_envs, self.n_points-1, 2, 3, device=self.device, dtype=torch.float32)
            lines_field = ti.Vector.field(3, dtype=ti.f32, shape=(n_envs * (self.n_points-1) * 2))
            for i, t in enumerate(self.t_vec):
                p_t_b, v_t_b, a_t_b = get_traj_point(t, policy_info["coef_best"]) # [N, 3]
                p_t_w = torch.matmul(rotmat_b2w, p_t_b.unsqueeze(-1)).squeeze(-1) + p_w
                p_t_w = p_t_w[:n_envs].to(torch.float32) + env.renderer.env_origin
                if i != self.n_points - 1:
                    lines_tensor[:, i, 0] = p_t_w
                if i != 0:
                    lines_tensor[:, i-1, 1] = p_t_w
            lines_field.from_torch(torch2ti(lines_tensor.flatten(end_dim=-2)))                
            env.renderer.gui_scene.lines(lines_field, color=(1., 1., 1.), width=3.)
        
        with torch.no_grad():
            next_obs, loss, terminated, env_info = env.step(action)
            if on_step_cb is not None:
                on_step_cb(
                    obs=obs,
                    action=action,
                    policy_info=policy_info,
                    env_info=env_info)
        
        losses = {
            "cumulative_loss": cumulative_loss.item(),
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
        return YOPO(cfg, device)