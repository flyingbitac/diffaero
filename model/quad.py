from typing import Union, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from quaddif.utils.math import *

class QuadrotorModel(nn.Module):
    def __init__(self, cfg: DictConfig, dt: float, n_substeps: int, device: torch.device):
        super().__init__()
        self.device = device
        self.dt = dt
        self.n_substeps = n_substeps
        
        wrap = lambda x: torch.tensor(x, device=device, dtype=torch.float32)
        
        self._m = wrap(cfg.m)         # total mass
        self._arm_l = wrap(cfg.arm_l)    # arm length
        self._c_tau = wrap(cfg.c_tau)  # torque constant
        
        c, d = self._c_tau, self._arm_l / (2**0.5)
        self._tau_thrust_matrix = wrap([
            [ d, -d, -d,  d],
            [-d,  d, -d,  d],
            [ c,  c, -c, -c],
            [ 1,  1,  1,  1]])
        
        self._G = wrap(cfg.g)
        self._G_vec = wrap([0.0, 0.0, self._G])
        self._J = torch.diag(torch.tensor(list(cfg.J), device=device))     # inertia
        self._J_inv = torch.linalg.inv(self._J)
        self._D = torch.diag(torch.tensor(list(cfg.D), device=device))
        
        self._v_xy_max = wrap(float('inf'))
        self._v_z_max = wrap(float('inf'))
        self._omega_xy_max = wrap(cfg.max_w_xy)
        self._omega_z_max = wrap(cfg.max_w_z)
        self._T_max = wrap(cfg.max_T)
        self._T_min = wrap(cfg.min_T)
        
        self._X_lb = wrap([-float('inf'), -float('inf'), -float('inf'),
                           -self._v_xy_max, -self._v_xy_max, -self._v_z_max,
                           -1, -1, -1, -1,
                           -self._omega_xy_max, -self._omega_xy_max, -self._omega_z_max])

        self._X_ub = wrap([float('inf'), float('inf'), float('inf'),
                           self._v_xy_max, self._v_xy_max, self._v_z_max,
                           1, 1, 1, 1,
                           self._omega_xy_max, self._omega_xy_max, self._omega_z_max])

        self._U_lb = wrap([self._T_min, self._T_min, self._T_min, self._T_min])
        self._U_ub = wrap([self._T_max, self._T_max, self._T_max, self._T_max])

    def dynamics(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        # Unpacking state and input variables
        p, v, q, w = X[:, :3], X[:, 3:6], X[:, 6:10], X[:, 10:13]

        # Calculate torques and thrust
        tau_thrust = torch.matmul(self._tau_thrust_matrix, U.T).T
        tau, thrust = tau_thrust[:, :3], tau_thrust[:, 3]
        
        M = tau - torch.cross(w, torch.matmul(self._J, w.T).T, dim=-1)
        w_dot = torch.matmul(self._J_inv, M.T).T

        # Drag force
        fdrag = quat_rotate(q, (self._D @ quat_rotate(quat_inv(q), v).T).T)
        
        # thrust acceleration
        thrust_acc = quat_axis(q, 2) * (-thrust / self._m).unsqueeze(-1)
        
        # overall acceleration
        acc = thrust_acc - self._G_vec - fdrag / self._m
        
        # quaternion derivative
        q_dot = 0.5 * quat_mul(q, torch.cat((w, torch.zeros((q.size(0), 1), device=self.device)), dim=-1))
        
        # State derivatives
        X_dot = torch.concat([v, acc, q_dot.detach(), w_dot.detach()], dim=-1)
        
        return X_dot

    def step_rk4(self, X, U):
        X1 = rk4(self.dynamics, X, U, dt=self.dt, M=self.n_substeps)
        q_l = torch.norm(X1[:, 6:10], dim=1, keepdim=True).detach()
        X1[:, 6:10] = X1[:, 6:10] / q_l
        return X1

    def step_euler(self, X, U):
        X1 = EulerIntegral(self.dynamics, X, U, dt=self.dt, M=self.n_substeps)
        q_l = torch.norm(X1[:, 6:10], dim=1, keepdim=True).detach()
        X1[:, 6:10] = X1[:, 6:10] / q_l
        return X1

    def forward(self, X0, U):
        return self.step_euler(X0, U)

class PointMassModel(nn.Module):
    def __init__(self, cfg: DictConfig, dt: float, n_substeps: int, device: torch.device):
        super().__init__()
        self.device = device
        self.dt = dt
        self.n_substeps = n_substeps
        
        wrap = lambda x: torch.tensor(x, device=device, dtype=torch.float32)
        
        self._G = wrap(cfg.g)
        self._G_vec = wrap([0.0, 0.0, self._G])
        self._D = torch.diag(torch.tensor(list(cfg.D), device=device))
        self.lmbda = cfg.lmbda # soft control latency
        
    def dynamics(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        # Unpacking state and input variables
        p, v, a = X[:, :3], X[:, 3:6], X[:, 6:9]
        
        a_dot = self.lmbda * (U - a) / self.dt

        # State derivatives
        X_dot = torch.concat([v, a - self._G_vec, a_dot], dim=-1)
        
        return X_dot

    def step_rk4(self, X, U):
        return rk4(self.dynamics, X, U, dt=self.dt, M=self.n_substeps)

    def step_euler(self, X, U):
        return EulerIntegral(self.dynamics, X, U, dt=self.dt, M=self.n_substeps)

    def forward(self, X0, U):
        return self.step_rk4(X0, U)