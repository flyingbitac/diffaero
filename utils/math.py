# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Callable, Union, Optional, Tuple, List

import torch
from torch import Tensor
from pytorch3d import transforms as T

# Runge-Kutta 4th Order Method
def rk4(f, X0, U, dt, M=1):
    # type: (Callable[[Tensor, Tensor], Tensor], Tensor, Tensor, float, int) -> Tensor
    DT = dt / M
    X1 = X0
    for _ in range(M):
        k1 = DT * f(X1, U)
        k2 = DT * f(X1 + 0.5 * k1, U)
        k3 = DT * f(X1 + 0.5 * k2, U)
        k4 = DT * f(X1 + k3, U)
        X1 = X1 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return X1

# Euler Integration
def EulerIntegral(f, X0, U, dt, M=1):
    # type: (Callable[[Tensor, Tensor], Tensor], Tensor, Tensor, float, int) -> Tensor
    DT = dt / M
    X1 = X0
    for _ in range(M):
        X1 = X1 + DT * f(X1, U)
    return X1

@torch.jit.script
def euler_to_quaternion(roll, pitch, yaw):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)

def random_quat_from_eular_zyx(
    yaw_range:   Tuple[float, float] = (-torch.pi, torch.pi),
    pitch_range: Tuple[float, float] = (-torch.pi, torch.pi),
    roll_range:  Tuple[float, float] = (-torch.pi, torch.pi),
    size: Union[int, Tuple[int, int]] = 1,
    device = None
) -> Tuple[float, float, float, float]:
    """
    Return a quaternion with eular angles uniformly sampled from given range.
    
    Args:
        yaw_range:   range of yaw angle in radians.
        pitch_range: range of pitch angle in radians.
        roll_range:  range of roll angle in radians.
    
    Returns:
        Real and imagine part of the quaternion.
    """
    yaw = torch.rand(size, device=device) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
    pitch = torch.rand(size, device=device) * (pitch_range[1] - pitch_range[0]) + pitch_range[0]
    roll = torch.rand(size, device=device) * (roll_range[1] - roll_range[0]) + roll_range[0]
    quat_xyzw = euler_to_quaternion(roll, pitch, yaw)
    return quat_xyzw

@torch.jit.script
def quat_rotate(quat_xyzw: Tensor, v: Tensor) -> Tensor:
    q_w = quat_xyzw[..., -1]
    q_vec = quat_xyzw[..., :3]
    a = v * (q_w ** 2 - q_vec.pow(2).sum(dim=-1)).unsqueeze(-1)
    b = 2. * q_w.unsqueeze(-1) * torch.cross(q_vec, v, dim=-1)
    c = 2. * q_vec * (q_vec * v).sum(dim=-1, keepdim=True)
    return a + b + c

@torch.jit.script
def quat_rotate_inverse(quat_xyzw: Tensor, v: Tensor) -> Tensor:
    q_w = quat_xyzw[..., -1]
    q_vec = quat_xyzw[..., :3]
    a = v * (q_w ** 2 - q_vec.pow(2).sum(dim=-1)).unsqueeze(-1)
    b = 2. * q_w.unsqueeze(-1) * torch.cross(q_vec, v, dim=-1)
    c = 2. * q_vec * (q_vec * v).sum(dim=-1, keepdim=True)
    return a - b + c

@torch.jit.script
def quat_axis(quat_xyzw: Tensor, axis: int = 0) -> Tensor:
    basis_vec = torch.zeros(quat_xyzw.shape[0], 3, device=quat_xyzw.device)
    basis_vec[..., axis] = 1
    return quat_rotate(quat_xyzw, basis_vec)

@torch.jit.script
def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).reshape(shape)

    return quat

def quat_standardize(quat_xyzw: torch.Tensor) -> torch.Tensor:
    return torch.where(quat_xyzw[..., -1:] < 0, -quat_xyzw, quat_xyzw)

@torch.jit.script
def quat_inv(quat_xyzw: Tensor) -> Tensor:
    return torch.cat([-quat_xyzw[..., :3], quat_xyzw[..., 3:4]], dim=-1)

@torch.jit.script
def axis_rotmat(axis: str, angle: Tensor) -> Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    else: # axis == "Z"
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def rand_range(
    min: float,
    max: float,
    size: Union[int, Tuple[int, ...]],
    device: torch.device
):
    if isinstance(size, int):
        size = (size,)
    return torch.rand(*size, device=device) * (max - min) + min

@torch.jit.script
def quaternion_to_euler(quat_xyzw: Tensor) -> Tensor:
    # return T.matrix_to_euler_angles(T.quaternion_to_matrix(quat_wxyz), "ZYX")[..., [2, 1, 0]]
    x, y, z, w = quat_xyzw.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
    pitch = torch.asin(2.0 * (w * y - x * z))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    return torch.stack([roll, pitch, yaw], dim=-1)

@torch.jit.script
def quaternion_invert(quat_wxyz: Tensor) -> Tensor:
    neg = torch.ones_like(quat_wxyz)
    neg[..., 1:] = -1
    return quat_wxyz * neg

@torch.jit.script
def quaternion_apply(quat_wxyz: Tensor, point: Tensor) -> Tensor:
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = torch.zeros_like(point[..., 0:1])
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = T.quaternion_raw_multiply(
        T.quaternion_raw_multiply(quat_wxyz, point_as_quaternion),
        quaternion_invert(quat_wxyz),
    )
    return out[..., 1:]

@torch.jit.script
def mvp(mat: Tensor, vec: Tensor) -> Tensor:
    """
    Matrix-vector product.
    
    Args:
        mat: A tensor of shape (..., n, m).
        vec: A tensor of shape (..., m).
    
    Returns:
        A tensor of shape (..., n).
    """
    if mat.dim() < 2 or vec.dim() < 1:
        raise ValueError("Input dimensions are not compatible for matrix-vector multiplication.")
    return torch.matmul(mat, vec.unsqueeze(-1)).squeeze(-1)