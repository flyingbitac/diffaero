from typing import Tuple, Optional
import sys
sys.path.append('..')

import torch
from torch import Tensor

from diffaero.utils.math import mvp, tanh_unsquash
from diffaero.utils.logger import Logger

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
    drpy_min: Tensor, # [3, ]
    drpy_max: Tensor, # [3, ]
    dv_range: float,
    da_range: float,
    rotmat_p2b: Tensor, # [n_pitch*n_yaw, 3, 3]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    d_rpy, v_p, a_p, score = output.chunk(4, dim=-1) # [N, HW, n_channels], #channels: [3, 3, 3, 1]
    rpy = rpy_base.unsqueeze(0) + tanh_unsquash(d_rpy, drpy_min, drpy_max) # [N, HW, 3]
    p_b = rpy2xyz(rpy) # [N, HW, 3]
    v_p, a_p = dv_range * torch.tanh(v_p), da_range * torch.tanh(a_p) # [N, HW, 3]
    v_b = mvp(rotmat_p2b.unsqueeze(0), v_p)
    a_b = mvp(rotmat_p2b.unsqueeze(0), a_p)
    return p_b, v_b, a_b, score.squeeze(-1)

@torch.jit.script
def get_coef_matrix(t: Tensor) -> Tensor:
    device = t.device
    # I hate this too
    t = float(t.item()) # type: ignore
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
    coef_mats: Tensor, # [T, 6, 6]
    coef_xyz: Tensor # [N, HW, 6, 3]
):
    coef_mat = coef_mats[None, None, :, 3:, :] # [1, 1, T, 3, 6]
    pva = torch.matmul(coef_mat, coef_xyz.unsqueeze(2)) # [N, HW, T, 3(pva), 3(xyz)]
    p, v, a = pva.unbind(dim=-2) # [..., T, 3(xyz)]
    return p, v, a

@torch.jit.script
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
