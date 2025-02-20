from typing import *
import sys
sys.path.append('..')

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d import transforms as T
from tensordict import TensorDict
from omegaconf import DictConfig
from tqdm import tqdm
import taichi as ti

from quaddif.env import ObstacleAvoidance
from quaddif.utils.render import torch2ti

class YOPONet(nn.Module):
    def __init__(
        self,
        H_out: int,
        W_out: int,
        feature_dim: int,
    ):
        super().__init__()
        self.H_out = H_out
        self.W_out = W_out
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
            nn.Conv2d(feature_dim+9, 256, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.Conv2d(256, 10, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(
        self,
        rotmat_b2p: Tensor,
        depth_image: Tensor,
        state_b: Tensor # []
    ):
        N, C, HW = state_b.size(0), 10, self.H_out * self.W_out
        feat = self.net(depth_image)
        state_p = torch.matmul(rotmat_b2p.unsqueeze(0), state_b.unsqueeze(1))
        state_p = state_p.reshape(N, self.H_out, self.W_out, 9).permute(0, 3, 1, 2)
        feat = torch.cat([feat, state_p], dim=1)
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
    r_range: float,
    pitch_range: float,
    yaw_range: float,
    v_range: float,
    a_range: float,
    rotmat_p2b: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    d_rpy, v_p, a_p, score = output.chunk(4, dim=-1) # #channels: [3, 3, 3, 1]
    rpy_range = torch.tensor([r_range, pitch_range, yaw_range], device=rpy_base.device)
    rpy = rpy_base.unsqueeze(0) + torch.tanh(d_rpy) * rpy_range.expand_as(d_rpy)
    # print(rpy.view(-1, 3).min(dim=0).values, rpy.view(-1, 3).max(dim=0).values)
    p_b = rpy2xyz(rpy)
    v_p, a_p = v_range * torch.tanh(v_p), a_range * torch.tanh(a_p)
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
    p, v, a = pva.unbind(dim=-2)
    return p, v, a

class YOPO:
    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device
    ):
        self.tmax: float = cfg.tmax
        self.n_points: int = int(cfg.n_points_per_sec * self.tmax)
        self.t_vec = torch.linspace(0, self.tmax, self.n_points, device=device)
        self.min_pitch: float = cfg.min_pitch * torch.pi / 180.
        self.max_pitch: float = cfg.max_pitch * torch.pi / 180.
        self.min_yaw: float = cfg.min_yaw * torch.pi / 180.
        self.max_yaw: float = cfg.max_yaw * torch.pi / 180.
        self.n_pitch: int = cfg.n_pitch
        self.n_yaw: int = cfg.n_yaw
        self.feature_dim: int = cfg.feature_dim
        self.r_base: float = cfg.r
        self.r_range: float = cfg.r_range
        self.pitch_range: float = cfg.pitch_range * torch.pi / 180.
        self.yaw_range: float = cfg.yaw_range * torch.pi / 180.
        self.v_range: float = cfg.v_range
        self.a_range: float = cfg.a_range
        self.G = torch.tensor([[0., 0., 9.81]], device=device)
        self.n_epochs: int = cfg.n_epochs
        self.expl_prob: float = cfg.expl_prob
        self.device = device
        
        pitches = torch.linspace(self.min_pitch, self.max_pitch, self.n_pitch, device=device)
        yaws = torch.linspace(self.max_yaw, self.min_yaw, self.n_yaw, device=device)

        pitches, yaws = torch.meshgrid(pitches, yaws, indexing="ij")
        rolls = torch.zeros_like(pitches)
        self.euler_angles = torch.stack([yaws, pitches, rolls], dim=-1).reshape(-1, 3)
        print(self.euler_angles)
        self.rpy_base = torch.stack([torch.full_like(yaws, self.r_base), pitches, yaws], dim=-1).reshape(-1, 3)

        # convert coordinates from primitive frame to body frame
        self.rotmat_p2b = T.euler_angles_to_matrix(self.euler_angles, convention="ZYX")
        print(self.rotmat_p2b)
        # convert coordinates from body frame to primitive frame
        self.rotmat_b2p = self.rotmat_p2b.transpose(-2, -1)
        
        self.net = YOPONet(
            H_out=self.n_pitch,
            W_out=self.n_yaw,
            feature_dim=self.feature_dim).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
    
    def act(
        self,
        env: ObstacleAvoidance,
    ):
        rotmat_b2w = T.quaternion_to_matrix(env.q.roll(1, dims=-1))
        rotmat_w2b = rotmat_b2w.transpose(-2, -1)
        target_vel_b = torch.matmul(rotmat_w2b, env.target_vel.unsqueeze(-1)).squeeze(-1)
        depth_image = env.sensor_tensor.unsqueeze(1)
        p_curr_b = torch.zeros_like(env.p)
        v_curr_b = torch.matmul(rotmat_w2b, env.v.unsqueeze(-1)).squeeze(-1)
        a_curr_b = torch.matmul(rotmat_w2b, env.a.unsqueeze(-1)).squeeze(-1) - self.G
        
        N, HW = p_curr_b.size(0), self.n_pitch * self.n_yaw
        state = torch.stack([target_vel_b, v_curr_b, a_curr_b], dim=-1)
        net_output: Tensor = self.net(self.rotmat_b2p, depth_image, state)
        tqdm.write(str(target_vel_b[0]))
        p_end_b, v_end_b, a_end_b, score = post_process(
            output=net_output,
            rpy_base=self.rpy_base,
            r_range=self.r_range,
            pitch_range=self.pitch_range,
            yaw_range=self.yaw_range,
            v_range=self.v_range,
            a_range=self.a_range,
            rotmat_p2b=self.rotmat_p2b
        )
        # tqdm.write(str(p_end_b[0]))
        best_idx = score.argmin(dim=-1).reshape(N, 1, 1).expand(-1, -1, 3) # [N, 1, 3]
        if self.expl_prob > 0.:
            random_idx = torch.randint(0, HW, (N, 1, 3), device=self.device)
            use_random = torch.rand(N, 1, 3, device=self.device) < self.expl_prob
            grid_index = torch.where(use_random, random_idx, best_idx)
        else:
            grid_index = best_idx
        p_best_b = torch.gather(p_end_b.reshape(N, HW, 3), 1, grid_index).squeeze(1)
        v_best_b = torch.gather(v_end_b.reshape(N, HW, 3), 1, grid_index).squeeze(1)
        a_best_b = torch.gather(a_end_b.reshape(N, HW, 3), 1, grid_index).squeeze(1)
        tqdm.write(str(v_best_b[0])+"\n")
        coef_xyz = solve_coef( # [N, 6, 3]
            torch.tensor(self.tmax, device=self.device),
            p_curr_b,
            v_curr_b,
            a_curr_b,
            p_best_b,
            v_best_b,
            a_best_b
        )
        p_next_b, v_next_b, a_next_b = get_traj_point(self.t_vec[1], coef_xyz) # [N, 3]
        a_next_w = torch.matmul(rotmat_b2w, a_next_b.unsqueeze(-1)).squeeze(-1)
        return a_next_w + self.G, coef_xyz

    def step(
        self,
        env: ObstacleAvoidance,
    ):
        N, HW = env.n_envs, self.n_pitch * self.n_yaw
        
        rotmat_b2w = T.quaternion_to_matrix(env.q.roll(1, dims=-1))
        rotmat_w2b = rotmat_b2w.transpose(-2, -1)
        target_vel_b = torch.matmul(rotmat_w2b, env.target_vel.unsqueeze(-1)).squeeze(-1)
        depth_image = env.sensor_tensor.unsqueeze(1)
        p_curr_b = torch.zeros_like(env.p)
        v_curr_b = torch.matmul(rotmat_w2b, env.v.unsqueeze(-1)).squeeze(-1)
        a_curr_b = torch.matmul(rotmat_w2b, env.a.unsqueeze(-1)).squeeze(-1) - self.G
        state = torch.stack([target_vel_b, v_curr_b, a_curr_b], dim=-1)
        
        for _ in range(self.n_epochs):
            cumulative_loss = torch.zeros(N, HW, device=self.device)
            survive = torch.ones(N, HW, device=self.device, dtype=torch.bool)
            survive_time = torch.ones(N, HW, device=self.device)
            
            net_output: Tensor = self.net(self.rotmat_b2p, depth_image, state)
            p_end_b, v_end_b, a_end_b, score = post_process(
                output=net_output,
                rpy_base=self.rpy_base,
                r_range=self.r_range,
                pitch_range=self.pitch_range,
                yaw_range=self.yaw_range,
                v_range=self.v_range,
                a_range=self.a_range,
                rotmat_p2b=self.rotmat_p2b
            )
            coef_xyz = solve_coef( # [N, 6, 3]
                torch.tensor(self.tmax, device=self.device),
                p_curr_b.unsqueeze(1).expand_as(p_end_b),
                v_curr_b.unsqueeze(1).expand_as(v_end_b),
                a_curr_b.unsqueeze(1).expand_as(a_end_b),
                p_end_b,
                v_end_b,
                a_end_b
            )
            for t in self.t_vec[1:]:
                p_t_b, v_t_b, a_t_b = get_traj_point(t, coef_xyz) # [N, 3]
                p_t_w = torch.matmul(rotmat_b2w.unsqueeze(1), p_t_b.unsqueeze(-1)).squeeze(-1) + env.p.unsqueeze(1)
                v_t_w = torch.matmul(rotmat_b2w.unsqueeze(1), v_t_b.unsqueeze(-1)).squeeze(-1)
                a_t_w = torch.matmul(rotmat_b2w.unsqueeze(1), a_t_b.unsqueeze(-1)).squeeze(-1) + self.G.unsqueeze(0)
                loss, _, dead = loss_fn(env, p_t_w, v_t_w, a_t_w)
                survive = survive & ~dead
                survive_time += survive.float()
                cumulative_loss = cumulative_loss + loss * survive.float()
            cumulative_loss = cumulative_loss / survive_time
            score_loss = F.mse_loss(score, cumulative_loss.detach())
            total_loss = cumulative_loss.mean() + score_loss * 0.001
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        with torch.no_grad():
            a_next_w, coef_xyz = self.act(env)
            if env.renderer is not None and env.renderer.enable_rendering and not env.renderer.headless:
                n_envs = env.renderer.n_envs
                # lines_tensor = env.renderer.env_origin.reshape(n_envs, 1, 1, 3).expand(-1, self.n_points-1, 2, -1)
                lines_tensor = torch.zeros(n_envs, self.n_points-1, 2, 3, device=self.device, dtype=torch.float32)
                lines_field = ti.Vector.field(3, dtype=ti.f32, shape=(n_envs * (self.n_points-1) * 2))
                for i, t in enumerate(self.t_vec):
                    p_t_b, v_t_b, a_t_b = get_traj_point(t, coef_xyz) # [N, 3]
                    # p_t_w = torch.matmul(rotmat_b2w, p_t_b.unsqueeze(-1)).squeeze(-1)
                    p_t_w = torch.matmul(rotmat_b2w, p_t_b.unsqueeze(-1)).squeeze(-1) + env.p
                    # print(p_t_w.shape)
                    if i != self.n_points - 1:
                        lines_tensor[:, i, 0] = p_t_w[:n_envs].to(torch.float32) + env.renderer.env_origin
                    if i != 0:
                        lines_tensor[:, i-1, 1] = p_t_w[:n_envs].to(torch.float32) + env.renderer.env_origin
                    # if i == 1:
                    #     break
                lines_field.from_torch(torch2ti(lines_tensor.flatten(end_dim=-2)))                
                env.renderer.gui_scene.lines(lines_field, color=(1., 1., 1.), width=3.)
            
            env_info = env_step(env, a_next_w)
        
        return cumulative_loss.mean().item(), score_loss.item(), env_info

def env_step(env, a_w):
    # type: (ObstacleAvoidance, Tensor) -> Tuple[TensorDict, Tensor, Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
    env.model.step(a_w)
    terminated, truncated = env.terminated(), env.truncated()
    env.progress += 1
    if env.renderer is not None:
        env.renderer.step(**env.state_for_render())
        env.renderer.render()
        truncated = torch.full_like(truncated, env.renderer.gui_states["reset_all"]) | truncated
    reset = terminated | truncated
    reset_indices = reset.nonzero().view(-1)
    arrived = (env.p - env.target_pos).norm(dim=-1) < 0.5
    env.arrive_time.copy_(torch.where(arrived & (env.arrive_time == 0), env.progress.float() * env.dt, env.arrive_time))
    success = arrived & truncated
    env.update_sensor_data()
    if reset_indices.numel() > 0:
        env.reset_idx(reset_indices)
    with torch.no_grad():
        _, loss_components, _ = loss_fn(env, env.p.unsqueeze(1), env.v.unsqueeze(1), env.a.unsqueeze(1))
    extra = {
        "truncated": truncated,
        "l": env.progress.clone(),
        "reset": reset,
        "reset_indicies": reset_indices,
        "success": success,
        "arrive_time": env.arrive_time.clone(),
        "next_state_before_reset": env.state(with_grad=True),
        "loss_components": loss_components,
        "sensor": env.sensor_tensor.clone(),
    }
    return extra

def loss_fn(env, _p, _v, _a):
    # type: (ObstacleAvoidance, Tensor, Tensor, Tensor) -> Tuple[Tensor, Dict[str, float], Tensor]
    p, v, a = _p.detach(), _v.detach(), _a.detach()
    target_relpos = env.target_pos.unsqueeze(1) - p
    target_dist = target_relpos.norm(dim=-1)
    target_vel = target_relpos / torch.max(target_dist / env.max_vel, torch.ones_like(target_dist)).unsqueeze(-1)
    virtual_radius = 0.2
    # calculating the closest point on each sphere to the quadrotor
    sphere_relpos = env.obstacle_manager.p_spheres.unsqueeze(2) - p.unsqueeze(1) # [n_envs, n_spheres, 3]
    dist2surface_sphere = (sphere_relpos.norm(dim=-1) - env.obstacle_manager.r_spheres.unsqueeze(2)).clamp(min=0) # [n_envs, n_spheres]
    # calculating the closest point on each cube to the quadrotor
    nearest_point = p.unsqueeze(1).clamp(
        min=env.obstacle_manager.box_min.unsqueeze(2),
        max=env.obstacle_manager.box_max.unsqueeze(2)) # [n_envs, n_cubes, 3]
    cube_relpos = nearest_point - p.unsqueeze(1) # [n_envs, n_cubes, 3]
    dist2surface_cube = cube_relpos.norm(dim=-1).clamp(min=0) # [n_envs, n_cubes]
    # concatenate the relative direction and distance to the surface of both type of obstacles
    obstacle_reldirection = F.normalize(torch.cat([sphere_relpos, cube_relpos], dim=1), dim=-1) # [n_envs, n_obstacles, 3]
    dist2surface = torch.cat([dist2surface_sphere, dist2surface_cube], dim=1) # [n_envs, n_obstacles]
    dist2surface = (dist2surface - env.r_drone - virtual_radius).clamp(min=0)
    # calculate the obstacle avoidance loss
    approaching_vel = torch.sum(obstacle_reldirection * _v.unsqueeze(1), dim=-1) # [n_envs, n_obstacles]
    approaching = approaching_vel > 0
    avoiding_vel = torch.norm(_v.unsqueeze(1) - approaching_vel.detach().unsqueeze(-1) * obstacle_reldirection, dim=-1) # [n_envs, n_obstacles]
    approaching_penalty, most_dangerous = (torch.where(approaching, approaching_vel, 0.) * dist2surface.neg().exp()).max(dim=1) # [n_envs]
    avoiding_reward = torch.where(approaching, avoiding_vel, 0.) * dist2surface.neg().exp() # [n_envs, n_obstacles]
    avoiding_reward = avoiding_reward.gather(dim=1, index=most_dangerous.unsqueeze(1)).squeeze(1) # [n_envs]
    oa_loss = approaching_penalty - 0.5 * avoiding_reward
    
    pos_loss = 1 - target_relpos.norm(dim=-1).neg().exp()
    
    vel_diff = torch.norm(_v - target_vel, dim=-1)
    vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
    
    total_loss = vel_loss + 4 * oa_loss + 5 * pos_loss
    loss_components = {
        "vel_loss": vel_loss.mean().item(),
        "pos_loss": pos_loss.mean().item(),
        "oa_loss": oa_loss.mean().item(),
        "total_loss": total_loss.mean().item()
    }
    
    collision = torch.any(dist2surface < env.r_drone, dim=1) # [n_envs]
    collision = collision | (p[..., 2] - env.r_drone < env.z_ground_plane)
    out_of_bound = torch.any(p < -1.5*env.L, dim=-1) | torch.any(p > 1.5*env.L, dim=-1)
    
    dead = collision
    
    return total_loss, loss_components, dead
