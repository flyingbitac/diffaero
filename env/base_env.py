from typing import Tuple, Dict, Union, Optional

from omegaconf import DictConfig
import torch
from torch import Tensor
from tensordict import TensorDict

from quaddif.dynamics import build_dynamics
from quaddif.dynamics.pointmass import point_mass_quat, PointMassModelBase
from quaddif.utils.randomizer import RandomizerManager, build_randomizer
from quaddif.utils.render import PositionControlRenderer, ObstacleAvoidanceRenderer

class BaseEnv:
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.randomizer = RandomizerManager(cfg.randomizer)
        self.dynamics = build_dynamics(cfg.dynamics, device)
        self.dynamic_type: str = self.dynamics.type
        self.action_dim = self.dynamics.action_dim
        self.n_agents: int = cfg.n_agents
        self.dt: float = cfg.dt
        self.n_envs: int = cfg.n_envs
        self.L = build_randomizer(cfg.length, [self.n_envs], device=device)
        if not isinstance(self, BaseEnvMultiAgent):
            assert self.n_agents == 1
            self.target_pos = torch.zeros(self.n_envs, 3, device=device)
            self.init_pos = torch.zeros(self.n_envs, 3, device=device)
            self.last_action = torch.zeros(self.n_envs, self.action_dim, device=device)
        if self.n_agents > 1:
            self.init_pos = torch.zeros(self.n_envs, self.n_agents, 3, device=device)
            self.last_action = torch.zeros(self.n_envs, self.n_agents, self.action_dim, device=device)
            assert isinstance(self, BaseEnvMultiAgent)
        self.progress = torch.zeros(self.n_envs, device=device, dtype=torch.long)
        self.arrive_time = torch.zeros(self.n_envs, device=device, dtype=torch.float)
        self.max_steps: int = int(cfg.max_time / cfg.dt)
        self.wait_before_truncate: float = cfg.wait_before_truncate
        self.cfg = cfg
        self.loss_weights: DictConfig = cfg.loss_weights
        self.reward_weights: DictConfig = cfg.reward_weights
        self.device = device
        self.max_vel = torch.zeros(self.n_envs, device=device)
        self.min_target_vel: float = cfg.min_target_vel
        self.max_target_vel: float = cfg.max_target_vel
        self.renderer: Optional[Union[PositionControlRenderer, ObstacleAvoidanceRenderer]]
    
    def get_observations(self, with_grad=False):
        raise NotImplementedError
    
    def get_state(self, with_grad=False):
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
        if isinstance(self.dynamics, PointMassModelBase) and self.dynamics.align_yaw_with_target_direction:
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
        if isinstance(self.dynamics, PointMassModelBase) and self.dynamics.align_yaw_with_target_direction:
            return point_mass_quat(self._a, orientation=self.target_vel)
        else:
            return self.dynamics._q
    
    @property
    def target_vel(self):
        target_relpos = self.target_pos - self.p
        target_dist = target_relpos.norm(dim=-1) # [n_envs]
        return target_relpos / torch.max(target_dist / self.max_vel, torch.ones_like(target_dist)).unsqueeze(-1)

    def _step(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Common step logic for single agent environments."""
        # simulation step
        self.dynamics.step(action)
        # termination and truncation logic
        terminated, truncated = self.terminated(), self.truncated()
        self.progress += 1
        if self.renderer is not None:
            self.renderer.render(self.states_for_render())
            # truncate if `reset_all` is commanded by the user from GUI
            truncated = torch.full_like(truncated, self.renderer.gui_states["reset_all"]) | truncated
        # arrival flag denoting if the agent has reached the target position
        arrived = (self.p - self.target_pos).norm(dim=-1) < 0.5
        curr_time = self.progress.float() * self.dt
        # time that the agents approached the target positions for the first time
        self.arrive_time.copy_(torch.where(arrived & (self.arrive_time == 0), curr_time, self.arrive_time))
        # truncate if the agents have been at the target positions for a while
        truncated |= arrived & (curr_time > (self.arrive_time + self.wait_before_truncate))
        # average velocity of the agents
        avg_vel = (self.init_pos - self.target_pos).norm(dim=-1) / self.arrive_time
        # success flag denoting whether the agent has reached the target position at the end of the episode
        success = arrived & truncated
        # update last action
        self.last_action.copy_(action.detach())
        return terminated, truncated, success, avg_vel
    
    def states_for_render(self):
        # type: () -> Dict[str, Tensor]
        raise NotImplementedError
    
    def loss_and_reward(self, action):
        # type: (Tensor) -> Tuple[Tensor, Tensor, Dict[str, float]]
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

    @property
    def target_pos(self):
        return self.target_pos_base + self.target_pos_rel

    def step(self, action, need_global_state_before_reset=True) -> Tuple[
        Union[Tuple[Tensor, Tensor], Tuple[TensorDict, Tensor]],
        Tensor,
        Tensor,
        Dict[str, Union[Dict[str, Tensor], Dict[str, float], Tensor]]
    ]:
        raise NotImplementedError

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
        return self.get_observations(with_grad), self.get_global_state(with_grad)
    
    def reset(self):
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
        return self.get_obs_and_state()