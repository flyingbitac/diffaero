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
from diffaero.utils.math import mvp, rk4
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
        depth_image: Tensor,
        obs_p: Tensor # []
    ):
        N, C, HW = obs_p.size(0), self.out_dim, self.H_out * self.W_out
        feat = self.net(depth_image)
        obs_p = obs_p.reshape(N, self.H_out, self.W_out, 9).permute(0, 3, 1, 2) # [N, 9, H_out, W_out]
        feat = torch.cat([feat, obs_p], dim=1) # [N, feature_dim + 9, H_out, W_out]
        return self.head(feat).reshape(N, C, HW).permute(0, 2, 1) # [N, H_out*W_out, out_dim]

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
    output: Tensor, # [N, HW, 10]
    rpy_base: Tensor, # [n_pitch*n_yaw, 3]
    drpy_range: Tensor, # [3,]
    dv_range: float,
    da_range: float,
    rotmat_p2b: Tensor, # [n_pitch*n_yaw, 3, 3]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    d_rpy, v_p, a_p, score = output.chunk(4, dim=-1) # [N, HW, n_channels], #channels: [3, 3, 3, 1]
    rpy = rpy_base.unsqueeze(0) + torch.tanh(d_rpy) * drpy_range.expand_as(d_rpy) # [N, HW, 3]
    # print(rpy.view(-1, 3).min(dim=0).values, rpy.view(-1, 3).max(dim=0).values)
    p_b = rpy2xyz(rpy) # [N, HW, 3]
    v_p, a_p = dv_range * torch.tanh(v_p), da_range * torch.tanh(a_p) # [N, HW, 3]
    # v_b = torch.matmul(rotmat_p2b, v_p.unsqueeze(-1)).squeeze(-1)
    # a_b = torch.matmul(rotmat_p2b, a_p.unsqueeze(-1)).squeeze(-1)
    v_b = mvp(rotmat_p2b.unsqueeze(0), v_p)
    a_b = mvp(rotmat_p2b.unsqueeze(0), a_p)
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
    inv_coef_mat: Tensor,
    p0: Tensor,
    v0: Tensor,
    a0: Tensor,
    pt: Tensor,
    vt: Tensor,
    at: Tensor
):
    pvapva = torch.stack([p0, v0, a0, pt, vt, at], dim=-2) # [N, HW, 6, 3]
    coef_xyz = torch.matmul(inv_coef_mat, pvapva) # [..., 6, 3]
    return coef_xyz

@torch.jit.script
def get_traj_point(
    t: Tensor,
    coef_xyz: Tensor # [N, HW, 6, 3]
):
    coef_mat = get_coef_matrix(t)[3:, :] # [3, 6]
    pva = torch.matmul(coef_mat, coef_xyz) # [N, HW, 3(pva), 3(xyz)]
    p, v, a = pva.unbind(dim=-2) # [N, HW, 3(xyz)]
    return p, v, a

@torch.jit.script
def get_traj_points(
    t_vec: Tensor, # [T]
    coef_xyz: Tensor # [N, HW, 6, 3]
):
    coef_mat = get_coef_matrices(t_vec)[None, None, :, 3:, :] # [1, 1, T, 3, 6]
    pva = torch.matmul(coef_mat, coef_xyz.unsqueeze(2)) # [N, HW, T, 3(pva), 3(xyz)]
    p, v, a = pva.unbind(dim=-2) # [..., T, 3(xyz)]
    return p, v, a

# @torch.jit.script
def discrete_point_mass_dynamics_world(
    X: Tensor,
    U: Tensor,
    dt: float,
    G_vec: Tensor,
    lmbda: Tensor,
):
    """Dynamics function for discrete point mass model in world frame."""
    p, v, a_thrust = X[..., :3], X[..., 3:6], X[..., 6:9]
    next_p = p + dt * (v + 0.5 * (a_thrust + G_vec) * dt)
    control_delay_factor = 1 - torch.exp(-lmbda*dt)
    a_thrust_cmd = U
    next_a = torch.lerp(a_thrust, a_thrust_cmd, control_delay_factor)
    next_v = v + dt * (0.5 * (a_thrust + next_a) + G_vec)
    
    next_state = torch.cat([next_p, next_v, next_a], dim=-1)
    return next_state

class YOPO:
    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device
    ):
        self.tmax: Tensor = torch.tensor(cfg.tmax, device=device)
        self.inv_coef_mat: Tensor = torch.inverse(get_coef_matrix(self.tmax)) # [6, 6]
        self.n_points_per_sec: int = cfg.n_points_per_sec
        self.n_points: int = int(self.n_points_per_sec * cfg.tmax)
        self.t_vec = torch.linspace(0, cfg.tmax, self.n_points, device=device)
        self.min_pitch: float = cfg.min_pitch * torch.pi / 180.
        self.max_pitch: float = cfg.max_pitch * torch.pi / 180.
        self.min_yaw: float = cfg.min_yaw * torch.pi / 180.
        self.max_yaw: float = cfg.max_yaw * torch.pi / 180.
        self.n_pitch: int = cfg.n_pitch # H
        self.n_yaw: int = cfg.n_yaw     # W
        self.r_base: float = cfg.r
        self.dr_range: float = self.r_base
        self.dpitch_range: float = cfg.dpitch_range * torch.pi / 180.
        self.dyaw_range: float = cfg.dyaw_range * torch.pi / 180.
        self.drpy_range = torch.tensor([self.dr_range, self.dpitch_range, self.dyaw_range], device=device)
        self.dv_range: float = cfg.dv_range
        self.da_range: float = cfg.da_range
        self.G = torch.tensor([[0., 0., 9.81]], device=device)
        self.gamma: float = cfg.gamma
        self.expl_prob: float = cfg.expl_prob
        self.grad_norm: Optional[float] = cfg.grad_norm
        self.t_next = torch.tensor(cfg.t_next, device=device) if cfg.t_next is not None else self.t_vec[1]
        self.device = device
        
        pitches = torch.linspace(self.min_pitch, self.max_pitch, self.n_pitch, device=device)
        yaws = torch.linspace(self.max_yaw, self.min_yaw, self.n_yaw, device=device)

        pitches, yaws = torch.meshgrid(pitches, yaws, indexing="ij")
        rolls = torch.zeros_like(pitches)
        self.euler_angles = torch.stack([yaws, pitches, rolls], dim=-1).reshape(-1, 3) # [n_pitch*n_yaw, 3]
        self.rpy_base = torch.stack([torch.full_like(yaws, self.r_base), pitches, yaws], dim=-1).reshape(-1, 3) # [n_pitch*n_yaw, 3]

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
        self.net = torch.compile(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
    
    def body2primitive(self, vec_b: Tensor):
        return mvp(self.rotmat_b2p, vec_b)

    def primitive2body(self, vec_p: Tensor):
        return mvp(self.rotmat_p2b, vec_p)

    @timeit
    def inference(self, p_w, rotmat_b2w, v_w, a_w, target_vel_w, depth_image):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
        rotmat_w2b = rotmat_b2w.transpose(-2, -1)
        target_vel_b = mvp(rotmat_w2b, target_vel_w)
        p_curr_b = torch.zeros_like(p_w)
        v_curr_b = mvp(rotmat_w2b, v_w)
        a_curr_b = mvp(rotmat_w2b, a_w)
        
        rotmat_b2p = self.rotmat_b2p.unsqueeze(0)
        target_vel_p = mvp(rotmat_b2p, target_vel_b.unsqueeze(1)) # [N, HW, 3]
        v_curr_p = mvp(rotmat_b2p, v_curr_b.unsqueeze(1)) # [N, HW, 3]
        a_curr_p = mvp(rotmat_b2p, a_curr_b.unsqueeze(1)) # [N, HW, 3]
        state_input = torch.cat([target_vel_p, v_curr_p, a_curr_p], dim=-1) # [N, HW, 9]

        net_output: Tensor = self.net(depth_image, state_input) # [N, HW, 10]
        p_end_b, v_end_b, a_end_b, score = post_process( # [N, HW, (3, 3, 3)], [N, HW]
            output=net_output,
            rpy_base=self.rpy_base,
            drpy_range=self.drpy_range,
            dv_range=self.dv_range,
            da_range=self.da_range,
            rotmat_p2b=self.rotmat_p2b
        )
        coef_xyz = solve_coef( # [N, HW, 6, 3]
            self.inv_coef_mat[None, None, ...], # [1, 1, 6, 6]
            p_curr_b.unsqueeze(1).expand_as(p_end_b), # [N, HW, 3]
            v_curr_b.unsqueeze(1).expand_as(v_end_b), # [N, HW, 3]
            a_curr_b.unsqueeze(1).expand_as(a_end_b), # [N, HW, 3]
            p_end_b, # [N, HW, 3]
            v_end_b, # [N, HW, 3]
            a_end_b  # [N, HW, 3]
        )
        return score, coef_xyz
    
    def act(self, obs: Tuple[Tensor, ...], test: bool = False, env: Optional[ObstacleAvoidanceYOPO] = None):
        p_w, rotmat_b2w, v_w, a_w, target_vel_w, depth_image = obs
        N, HW = rotmat_b2w.size(0), self.n_pitch * self.n_yaw
        
        score, coef_xyz = self.inference(p_w, rotmat_b2w, v_w, a_w, target_vel_w, depth_image)

        best_idx = score.argmin(dim=-1) # [N, ]
        if not test:
            random_idx = torch.randint(0, HW, (N, ), device=self.device)
            use_random = torch.rand(N, device=self.device) < self.expl_prob
            patch_index = torch.where(use_random, random_idx, best_idx)
        else:
            patch_index = best_idx
        patch_index = patch_index.reshape(N, 1, 1, 1).expand(-1, -1, 6, 3)
        coef_best = torch.gather(coef_xyz, 1, patch_index).squeeze(1) # [N, 6, 3]

        p_next_b, v_next_b, a_next_b = get_traj_point(self.t_next, coef_best) # [N, 3]
        a_next_w = mvp(rotmat_b2w, a_next_b) + self.G
        return a_next_w, {"coef_best": coef_best}

    @timeit
    def step(self, cfg: DictConfig, env: ObstacleAvoidanceYOPO, logger: Logger, obs: Tuple[Tensor, ...], on_step_cb=None):
        N, HW = env.n_envs, self.n_pitch * self.n_yaw
        
        p_w, rotmat_b2w, v_w, a_w, target_vel_w, depth_image = obs
        
        for _ in range(cfg.algo.n_epochs):
            # traverse the trajectory and cumulate the loss
            score, coef_xyz = self.inference(p_w, rotmat_b2w, v_w, a_w, target_vel_w, depth_image) # [N, HW, 6, 3]
            
            T = self.n_points - 1
            traj_loss = torch.zeros(N, HW, T, device=self.device)
            survive = torch.ones(N, HW, T, device=self.device, dtype=torch.bool)

            p_traj_b, v_traj_b, a_traj_b = get_traj_points(self.t_vec[1:], coef_xyz) # [N, HW, T, 3]
            p_traj_w = mvp(rotmat_b2w.unsqueeze(1), p_traj_b.reshape(N, HW*T, 3)) + p_w.unsqueeze(1)
            v_traj_w = mvp(rotmat_b2w.unsqueeze(1), v_traj_b.reshape(N, HW*T, 3))
            a_traj_w = mvp(rotmat_b2w.unsqueeze(1), a_traj_b.reshape(N, HW*T, 3)) + self.G.unsqueeze(1)
            traj_loss, _, dead = env.loss_fn(p_traj_w, v_traj_w, a_traj_w)
            traj_loss, dead = traj_loss.reshape(N, HW, T), dead.reshape(N, HW, T).float().cumsum(dim=2)
            survive = (dead == 0.).float()
            traj_loss = torch.sum(traj_loss * survive, dim=-1) / survive.sum(dim=-1).clamp(min=1.)
            score_loss = F.mse_loss(score, traj_loss.detach())
            
            total_loss = traj_loss.mean() + 0.01 * score_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        if self.grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.grad_norm)
        else:
            grads = [p.grad for p in self.net.parameters() if p.grad is not None]
            grad_norm = torch.nn.utils.get_total_norm(grads)
        
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
            next_obs, (loss, _), terminated, env_info = env.step(action)
            if on_step_cb is not None:
                on_step_cb(
                    obs=obs,
                    action=action,
                    policy_info=policy_info,
                    env_info=env_info)
        
        losses = {
            "traj_loss": traj_loss.mean().item(),
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

    def export(
        self,
        path: str,
        export_cfg: DictConfig,
        verbose: bool = False,
    ):
        pass