from typing import Tuple, Dict, Union

from omegaconf import DictConfig
import torch
from torch import Tensor

from quaddif.dynamics import build_dynamics
from quaddif.dynamics.pointmass import point_mass_quat

class BaseEnv:
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.dynamics = build_dynamics(cfg.dynamics, device)
        self.dynamic_type: str = self.dynamics.type
        self.n_agents: int = cfg.n_agents
        self.dt: float = cfg.dt
        self.L: float = cfg.length
        self.n_envs: int = cfg.n_envs
        if not isinstance(self, BaseEnvMultiAgent):
            assert self.n_agents == 1
            self.target_pos = torch.zeros(self.n_envs, 3, device=device)
        if self.n_agents > 1:
            assert isinstance(self, BaseEnvMultiAgent)
        self.progress = torch.zeros(self.n_envs, device=device, dtype=torch.long)
        self.arrive_time = torch.zeros(self.n_envs, device=device, dtype=torch.float)
        self.max_steps: int = int(cfg.max_time / cfg.dt)
        self.cfg = cfg
        self.device = device
        self.max_vel = torch.zeros(self.n_envs, device=device)
        self.min_target_vel: float = cfg.min_target_vel
        self.max_target_vel: float = cfg.max_target_vel
        self.use_old_obs_proc = cfg.use_old_obs_proc
    
    def get_observations(self, with_grad=False):
        raise NotImplementedError
    
    def detach(self):
        self.dynamics.detach()
    
    @property
    def p(self): return self.dynamics.p
    @property
    def v(self): return self.dynamics.v
    @property
    def a(self): return self.dynamics.a
    @property
    def w(self): return self.dynamics.w
    @property
    def q(self) -> Tensor:
        if self.dynamic_type == "pointmass" and self.dynamics.align_yaw_with_target_direction:
            return point_mass_quat(self.a, orientation=self.target_vel)
        else:
            return self.dynamics.q
    @property
    def _p(self): return self.dynamics._p
    @property
    def _v(self): return self.dynamics._v
    @property
    def _a(self): return self.dynamics._a
    @property
    def _w(self): return self.dynamics._w
    @property
    def _q(self) -> Tensor:
        if self.dynamic_type == "pointmass" and self.dynamics.align_yaw_with_target_direction:
            return point_mass_quat(self._a, orientation=self.target_vel)
        else:
            return self.dynamics._q
    
    @property
    def target_vel(self):
        target_relpos = self.target_pos - self.p
        target_dist = target_relpos.norm(dim=-1) # [n_envs]
        return target_relpos / torch.max(target_dist / self.max_vel, torch.ones_like(target_dist)).unsqueeze(-1)

    def step(self, action, need_obs_before_reset=True):
        # type: (Tensor, bool) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        raise NotImplementedError
    
    def state_for_render(self):
        # type: () -> Tensor
        raise NotImplementedError
    
    def loss_fn(self, target_vel, action):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Dict[str, float]]
        raise NotImplementedError

    def reset_idx(self, env_idx: Tensor):
        raise NotImplementedError
    
    def reset(self):
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
        return self.get_observations()
    
    def terminated(self) -> Tensor:
        raise NotImplementedError
    
    def truncated(self) -> Tensor:
        return self.progress >= self.max_steps
    
    def rescale_action(self, action: Tensor) -> Tensor:
        return self.dynamics.min_action + (self.dynamics.max_action - self.dynamics.min_action) * (action + 1) / 2

class BaseEnvMultiAgent(BaseEnv):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        assert self.n_agents > 1
        self.target_pos_base = torch.zeros(self.n_envs, self.n_agents, 3, device=device)
        self.target_pos_rel  = torch.zeros(self.n_envs, self.n_agents, 3, device=device)

    def step(self, action, need_global_state_before_reset=True):
        # type: (Tensor, bool) -> Tuple[Tuple[Tensor, Tensor], Tensor, Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        raise NotImplementedError

    @property
    def target_pos(self):
        return self.target_pos_base + self.target_pos_rel

    @property
    def target_vel(self): # TODO
        # 这里要改成每个环境中的num_agents个飞机分别以距离自身最近的target_pos为目标计算相对的target_relpos:
        target_relpos = self.target_pos - self.p
        # target_relpos = self.multidrone_targetpos
        target_dist = target_relpos.norm(dim=-1) # [n_envs, n_agents]
        return target_relpos / torch.max(target_dist / self.max_vel.unsqueeze(-1), torch.ones_like(target_dist)).unsqueeze(-1)

    @property
    def allocated_target_pos(self):
        """Allocate a target for each agent."""
        # 计算每架飞机相对于每个目标点的距离并找出最近的目标点的索引
        distance = torch.norm(self.p[:, :, None] - self.target_pos[:, None, :], dim=-1) # [n_envs, n_agents, n_agents]
        closest_target_indices = torch.min(distance, dim=-1).indices # [n_envs, n_agents]
        # 使用gather方法获取最接近的目标位置
        closest_targets = self.target_pos.gather(dim=1, index=closest_target_indices.unsqueeze(-1).expand_as(self.target_pos)) # [n_envs, n_agents, 3]
        # 计算每个无人机到其最接近目标点的相对位置向量
        return closest_targets # [n_envs, n_agents, 3]

    @property
    def internal_min_distance(self) -> Tensor:
        # 计算每个环境中无人机之间的距离
        distances = torch.norm(self._p[:, :, None, :] - self._p[:, None, :, :], dim=-1) # [n_envs, n_agents, n_agents]
        # 去除对角线元素
        diag = torch.diag(torch.ones(self.n_agents, device=self.device)).unsqueeze(0).expand(self.n_envs, -1, -1)
        distances = torch.where(diag.bool(), float('inf'), distances)
        # 找到每个agent的最小距离
        min_distances = distances.min(dim=-1).values
        return min_distances
    
    def get_observations(self, with_grad=False):
        raise NotImplementedError
    
    def get_global_state(self, with_grad=False):
        raise NotImplementedError
    
    def get_obs_and_state(self, with_grad=False):
        if self.use_old_obs_proc:
            if self.dynamic_type == "pointmass":
                # 全局状态 为所有agent自身状态(p+q+v)+目标位置
                global_state = torch.cat([self._p, self._q, self._v], dim=-1)
                global_state = global_state.reshape(global_state.shape[0], global_state.shape[1]*global_state.shape[2]).unsqueeze(1)
                global_state = torch.cat([global_state, self.target_pos.reshape(self.target_pos.shape[0], self.target_pos.shape[1]*self.target_pos.shape[2]).unsqueeze(1), self.box_size.unsqueeze(1)], dim=-1)

                # global state维度不对，太短，对不上设置的global_state_dim
                
                # agent观测 每个agent的观测为 自身状态(目标方向单位向量+自身姿态四元数+自身速度向量)+所有其他agent相对状态(p+v)+目标位置
                obs = []
                for i in range(self.n_agents):
                    obs_i = torch.zeros((self.n_envs, self.obs_dim), device=self.device)
                    # 获取i-th agent的p q v
                    p_i = self.p[:, i, :]
                    q_i = self.q[:, i, :]
                    v_i = self.v[:, i, :]
                    # 获取i-th agent对自己状态的观测
                    obs_i[:, :3*self.n_agents+4+3] = torch.cat([self.target_vel.reshape(self.target_vel.shape[0], self.target_vel.shape[1]*self.target_vel.shape[2]), q_i, v_i], dim=-1)
                    # 获取i-th agent对其他agent的观测
                    for j in range(self.n_agents):
                        # 逐个计算i-th agent对其他agent的观测值，包括相对位置和速度，不计算对自己的观测，所以i=j时跳过
                        if j == i:
                            continue
                        p_j = self.p[:, j, :]
                        v_j = self.v[:, j, :]
                        p_ji = p_j - p_i
                        v_ji = v_j - v_i
                        if j < i:
                            obs_i[:, (3*self.n_agents+4+3) + 6*j:(3*self.n_agents+4+3) + 6*(j+1)] = torch.cat([p_ji, v_ji], dim=-1)
                        else:
                            obs_i[:, (3*self.n_agents+4+3) + 6*(j-1):((3*self.n_agents+4+3)) + 6*j] = torch.cat([p_ji, v_ji], dim=-1)
                    # 获取i-th agent对目标位置的观测
                    related_pos = self.target_pos - p_i[:, None, :]
                    obs_i[:, -3*self.n_agents:] = related_pos.reshape(related_pos.shape[0], related_pos.shape[1]*related_pos.shape[2])
                    obs.append(obs_i)
                obs = torch.stack(obs, dim=1) # (num_envs, num_agents, obs_dim)
            else:
                raise NotImplementedError
                state = torch.cat([self.target_vel, self._q, self._v], dim=-1)
            if with_grad:
                return obs, global_state.squeeze(1)
            else:
                return obs.detach(), global_state.squeeze(1).detach()
        else:
            return self.get_observations(with_grad), self.get_global_state(with_grad)
    
    def reset(self):
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
        return self.get_obs_and_state()