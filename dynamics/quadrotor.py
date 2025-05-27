import torch
from torch import Tensor
from omegaconf import DictConfig

from quaddif.dynamics.controller import RateController
from quaddif.utils.math import *
from quaddif.utils.randomizer import build_randomizer

class QuadrotorModel:
    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device
    ):
        self.type = "quadrotor"
        self.device = device
        self.state_dim = 13
        self.action_dim = 4
        self.n_envs: int = cfg.n_envs
        self.dt: float = cfg.dt
        self.n_substeps: int = cfg.n_substeps
        self._state = torch.zeros(self.n_envs, self.state_dim, device=device)
        self._acc = torch.zeros(self.n_envs, 3, device=device)
        
        if cfg.solver_type == "euler":
            self.solver = EulerIntegral
        elif cfg.solver_type == "rk4":
            self.solver = rk4
        
        wrap = lambda x: torch.tensor(x, device=device, dtype=torch.float32)
        
        self._m = build_randomizer(cfg.m, self.n_envs, device) # total mass
        self._arm_l = build_randomizer(cfg.arm_l, self.n_envs, device) # arm length
        self._c_tau = build_randomizer(cfg.c_tau, self.n_envs, device) # torque constant
        
        self._G = wrap(cfg.g)
        self._G_vec = wrap([0.0, 0.0, self._G])
        
        # inertia
        self.J_xy = build_randomizer(cfg.J.xy, self.n_envs, device)
        self.J_z = build_randomizer(cfg.J.z, self.n_envs, device)
        # drag coefficients
        self.D_xy = build_randomizer(cfg.D.xy, self.n_envs, device)
        self.D_z = build_randomizer(cfg.D.z, self.n_envs, device)
        
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
        
        self.controller = RateController(self._m.value, self._J, self._G, cfg.controller, self.device)
        self.min_action = self.controller.min_action
        self.max_action = self.controller.max_action
    
    @property
    def _tau_thrust_matrix(self) -> Tensor:
        c, d = self._c_tau.value, self._arm_l.value / (2**0.5)
        ones = torch.ones(self.n_envs, 4, device=c.device, dtype=c.dtype)
        _tau_thrust_matrix = torch.stack([
            torch.stack([ d, -d, -d,  d], dim=-1),
            torch.stack([-d,  d, -d,  d], dim=-1),
            torch.stack([ c,  c, -c, -c], dim=-1),
            ones], dim=-2)
        print(_tau_thrust_matrix.shape, _tau_thrust_matrix[0])
        return _tau_thrust_matrix

    @property
    def _J(self) -> Tensor:
        J = torch.zeros(self.n_envs, 3, 3, device=self.device)
        J[:, 0, 0] = self.J_xy.value
        J[:, 1, 1] = self.J_xy.value
        J[:, 2, 2] = self.J_z.value
        return J
    
    @property
    def _J_inv(self) -> Tensor:
        J_inv = torch.zeros(self.n_envs, 3, 3, device=self.device)
        J_inv[:, 0, 0] = 1. / self.J_xy.value
        J_inv[:, 1, 1] = 1. / self.J_xy.value
        J_inv[:, 2, 2] = 1. / self.J_z.value
        return J_inv
    
    @property
    def _D(self) -> Tensor:
        D = torch.zeros(self.n_envs, 3, 3, device=self.device)
        D[:, 0, 0] = self.D_xy.value
        D[:, 1, 1] = self.D_xy.value
        D[:, 2, 2] = self.D_z.value
        return D
    
    def detach(self):
        self._state.detach_()
        self._acc.detach_()

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
        
        M = torque - torch.cross(w, torch.matmul(self._J, w.unsqueeze(-1)).squeeze(-1), dim=-1)
        w_dot = torch.matmul(self._J_inv, M.unsqueeze(-1)).squeeze(-1)

        # Drag force
        fdrag = quat_rotate(q, (self._D @ quat_rotate(quat_inv(q), v).unsqueeze(-1)).squeeze(-1))
        
        # thrust acceleration
        thrust_acc = quat_axis(q, 2) * (thrust / self._m.value).unsqueeze(-1)
        
        # overall acceleration
        acc = thrust_acc - self._G_vec - fdrag / self._m.value.unsqueeze(-1)
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
    
    def reset_idx(self, env_idx: Tensor) -> None:
        mask = torch.zeros_like(self._acc, dtype=torch.bool)
        mask[env_idx] = True
        self._acc = torch.where(mask, 0., self._acc)
    
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
