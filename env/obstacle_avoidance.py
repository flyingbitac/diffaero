from typing import Tuple, Dict, Union

from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch3d import transforms as T

from quaddif.env.base_env import BaseEnv
from quaddif.model.quad import QuadrotorModel, PointMassModel
from quaddif.utils.render import ObstacleAvoidanceRenderer
from quaddif.utils.math import unitization, axis_rotmat, rand_range


class PointMassObstacleAvoidance(BaseEnv):
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.model = PointMassModel(cfg.quad, cfg.dt, cfg.n_substeps, device)
        self.vision = cfg.vision
        self.renderer = ObstacleAvoidanceRenderer(cfg.render, device.index)
        self.asset_poses = torch.zeros(
            cfg.n_envs, self.renderer.asset_manager.assets_per_env, 7, device=device)
        self.n_obstacles = self.renderer.n_obstacles
        self.r_obstacles = self.renderer.r_obstacles
        if self.vision:
            # flattened depth image as additional observation
            self.state_dim = 13 + cfg.render.image_size[1] * cfg.render.image_size[2]
        else:
            # relative position of obstacles as additional observation
            self.state_dim = 13 + self.n_obstacles * 3
        self.action_dim = 3
        super(PointMassObstacleAvoidance, self).__init__(cfg, device)
    
    def state(self, with_grad=False):
        state = [self.target_vel(), self._v, self._a, self.q]
        if self.vision:
            state.append(self.renderer.camera_tensor.flatten(1))
        else:
            obst_relpos = self.asset_poses[:, :self.n_obstacles, :3] - self._p.unsqueeze(1)
            sorted_idx = obst_relpos.norm(dim=-1).argsort(dim=-1).unsqueeze(-1).expand(-1, -1, 3)
            state.append(obst_relpos.gather(dim=1, index=sorted_idx).flatten(1))
        state = torch.cat(state, dim=-1)
        return state if with_grad else state.detach()
    
    def step(self, action):
        # type: (Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        action = self.rescale_action(action)
        self._state = self.model(self._state, action)
        self._vel_ema.lerp_(self._v, self.vel_ema_factor)
        self.progress += 1
        terminated, truncated = self.terminated(), self.truncated()
        reset = terminated | truncated
        reset_indices = reset.nonzero().squeeze(-1)
        success = truncated & ((self.p - self.target_pos).norm(dim=-1) < 0.5)
        target_vel = self.target_vel()
        loss, loss_components = self.loss_fn(target_vel, action)
        extra = {
            "truncated": truncated,
            "l": self.progress.clone(),
            "reset": reset,
            "reset_indicies": reset_indices,
            "success": success,
            "next_state_before_reset": self.state(with_grad=True),
            "loss_components": loss_components
        }
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        self.renderer.step(*self.state_for_render())
        if self.vision:
            self.renderer.render_camera()
            extra["camera"] = self.renderer.camera_tensor.clone()
        elif not self.renderer.headless:
            self.renderer.render()
        return self.state(), loss, terminated, extra
    
    def state_for_render(self):
        w = torch.zeros_like(self.v)
        drone_state = torch.concat([self.p, self.q, self.v, w], dim=-1)
        assets_state = torch.cat([
            self.asset_poses,
            torch.zeros(self.n_envs, self.asset_poses.size(1), 6, device=self.device)
        ], dim=-1)
        return torch.concat([drone_state.unsqueeze(1), assets_state], dim=1), self.target_pos
    
    def loss_fn(self, target_vel, action):
        # type: (torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]
        pos_loss = -(-(self._p-self.target_pos).norm(dim=-1)).exp()
        
        vel_diff = (self._vel_ema - target_vel).norm(dim=-1)
        vel_loss = F.smooth_l1_loss(vel_diff, torch.zeros_like(vel_diff), reduction="none")
        
        obstacle_pos = self.asset_poses[:, :self.n_obstacles, :3] # [n_envs, n_obstacles, 3]
        obstacle_relpos = obstacle_pos - self.p.unsqueeze(1)
        # obstacle_relpos = obstacle_pos - self._p.unsqueeze(1)
        dist2surface = (obstacle_relpos.norm(dim=-1) - self.r_obstacles).clamp(min=0.1)
        cos_sim = torch.cosine_similarity(obstacle_relpos, self._v.unsqueeze(1), dim=-1)
        oa_loss = (cos_sim.clamp(min=0) / dist2surface.exp()).sum(dim=-1)
        
        stable_loss = F.mse_loss(self._a, action, reduction="none").sum(dim=-1)
        
        total_loss = vel_loss + 3 * oa_loss + 0.003 * stable_loss + 5 * pos_loss
        # total_loss = vel_loss + 0.003 * stable_loss
        loss_components = {
            "vel_loss": vel_loss.mean().item(),
            "pos_loss": pos_loss.mean().item(),
            "stable_loss": stable_loss.mean().item(),
            "oa_loss": oa_loss.mean().item(),
            "total_loss": total_loss.mean().item()
        }
        
        return total_loss, loss_components

    def reset_idx(self, env_idx):
        n_resets = len(env_idx)
        state_mask = torch.zeros_like(self._state)
        state_mask[env_idx] = 1
        p_new = rand_range(-self.L+1, self.L-1, size=(self.n_envs, 3), device=self.device)
        v_new = torch.zeros_like(self.v)
        a_new = torch.zeros_like(self.a)
        new_state = torch.cat([p_new, v_new, a_new], dim=-1)
        self._state = torch.where(state_mask.bool(), new_state, self._state)
        
        # target position
        min_init_dist = 1.5 * self.L
        N = 10
        x = y = z = torch.linspace(-self.L+1, self.L-1, N, device=self.device)
        random_idx = torch.randperm(N**3, device=self.device)
        xyz = torch.stack(torch.meshgrid(x, y, z), dim=-1).reshape(-1, 3)[random_idx]
        validility: torch.BoolTensor = (xyz[None, ...] - self.p[env_idx, None, :]).norm(dim=-1) > min_init_dist
        sub_idx = validility.nonzero()
        env_sub_idx = torch.tensor([(sub_idx[:, 0] == i).sum() for i in range(n_resets)]).roll(1, dims=0)
        env_sub_idx[0] = 0
        env_sub_idx = torch.cumsum(env_sub_idx, dim=0)
        self.target_pos[env_idx] = xyz[sub_idx[env_sub_idx, 1]]
        assert torch.all((self.p[env_idx] - xyz[sub_idx[env_sub_idx, 1]]).norm() > min_init_dist).item()
        
        # obstacle position
        obstacle_pos_quat, mask = self.renderer.asset_manager.randomize_asset_pose(
            env_idx=env_idx,
            drone_init_pos=self.p[env_idx],
            target_pos=self.target_pos[env_idx],
            safety_range=self.r_obstacles.max().item()+0.5
        )
        self.asset_poses[env_idx] = obstacle_pos_quat
            
        self.progress[env_idx] = 0
    
    def reset(self):
        super().reset()
        self.renderer.step(*self.state_for_render())
        return self.state()
    
    def terminated(self) -> torch.Tensor:
        out_of_bound = torch.any(self.p < -1.5*self.L, dim=-1) | \
                       torch.any(self.p >  1.5*self.L, dim=-1)
        collision = self.renderer.check_collisions()
        return out_of_bound | collision