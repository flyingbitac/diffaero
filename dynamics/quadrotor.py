import torch
from torch import Tensor
from omegaconf import DictConfig

from quaddif.dynamics.controller import RateController
from quaddif.utils.math import *

class QuadrotorModel:
    def __init__(
        self,
        cfg: DictConfig,
        n_envs: int,
        dt: float,
        n_substeps: int,
        device: torch.device
    ):
        self.device = device
        self.state_dim = 13
        self.action_dim = 4
        self._state = torch.zeros(n_envs, self.state_dim, device=device)
        self._acc = torch.zeros(n_envs, 3, device=device)
        self.dt = dt
        self.n_substeps = n_substeps
        
        if cfg.solver_type == "euler":
            self.solver = EulerIntegral
        elif cfg.solver_type == "rk4":
            self.solver = rk4
        
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
        self._J = torch.diag(wrap(list(cfg.J))) # inertia
        self._J_inv = torch.linalg.inv(self._J)
        self._D = torch.diag(wrap(list(cfg.D))) # drag coefficients
        
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
        
        self.controller = RateController(self._m, self._J, self._G, cfg.controller, self.device)
        self.min_action = self.controller.min_action
        self.max_action = self.controller.max_action
    
    def detach(self):
        self._state = self._state.detach()
        self._acc = self._acc.detach()

    def dynamics(self, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        # Unpacking state and input variables
        p, q, v, w = X[:, :3], X[:, 3:7], X[:, 7:10], X[:, 10:13]
        # Calculate torques and thrust
        # T1, T2, T3, T4 = U[:, 0], U[:, 1], U[:, 2], U[:, 3]
        # taux   = (T1 + T4 - T2 - T3) * self._arm_l / torch.sqrt(torch.tensor(2.0))
        # tauy   = (T1 + T3 - T2 - T4) * self._arm_l / torch.sqrt(torch.tensor(2.0))
        # tauz   = (T3 + T4 - T1 - T2) * self._c_tau
        # thrust = (T1 + T2 + T3 + T4)
        # torque = torch.stack((taux, tauy, tauz), dim=1)
        thrust, torque = self.controller(q, w, U)
        
        M = torque - torch.cross(w, torch.matmul(self._J, w.T).T, dim=-1)
        w_dot = torch.matmul(self._J_inv, M.T).T

        # Drag force
        fdrag = quat_rotate(q, (self._D @ quat_rotate(quat_inv(q), v).T).T)
        
        # thrust acceleration
        thrust_acc = quat_axis(q, 2) * (thrust / self._m).unsqueeze(-1)
        
        # overall acceleration
        acc = thrust_acc - self._G_vec - fdrag / self._m
        self._acc = acc
        
        # quaternion derivative
        q_dot = 0.5 * quat_mul(q, torch.cat((w, torch.zeros((q.size(0), 1), device=self.device)), dim=-1))
        
        # State derivatives
        X_dot = torch.concat([v, q_dot, acc, w_dot], dim=-1)
        
        return X_dot

    def step(self, U: Tensor) -> None:
        new_state = self.solver(self.dynamics, self._state, U, dt=self.dt, M=self.n_substeps)
        q_l = torch.norm(new_state[:, 3:7], dim=1, keepdim=True).detach()
        new_state[:, 3:7] = new_state[:, 3:7] / q_l
        self._state = new_state
    
    @property
    def p(self) -> Tensor: return self._state[:, 0:3].detach()
    @property
    def q(self) -> Tensor: return self._state[:, 3:7].detach()
    @property
    def v(self) -> Tensor: return self._state[:, 7:10].detach()
    @property
    def w(self) -> Tensor: return self._state[:, 10:13].detach()
    @property
    def a(self) -> Tensor: return self._acc.detach()
    @property
    def _p(self) -> Tensor: return self._state[:, 0:3]
    @property
    def _q(self) -> Tensor: return self._state[:, 3:7]
    @property
    def _v(self) -> Tensor: return self._state[:, 7:10]
    @property
    def _w(self) -> Tensor: return self._state[:, 10:13]
    @property
    def _a(self) -> Tensor: return self._acc
