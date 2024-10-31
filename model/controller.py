import torch
import pytorch3d.transforms as p3d_transforms
from omegaconf import DictConfig

class BaseController:
    """Convert the action from RL agent to force and torques to be applied on the drone."""
    def __init__(
        self,
        mass: torch.Tensor,
        inertia: torch.Tensor,
        gravity: torch.Tensor,
        cfg: DictConfig,
        device: torch.device
    ):
        self.cfg = cfg
        self.device = device
        self.mass = mass
        self.inertia = inertia
        self.gravity = gravity
        self.thrust_ratio = cfg.thrust_ratio
        self.torque_ratio = cfg.torque_ratio
        
        # lower bound of controller output (actual normed force & torque)
        self.min_thrust = torch.tensor(cfg.min_normed_thrust, device=device)
        self.min_torque = torch.tensor(cfg.min_normed_torque, device=device)
        
        # upper bound of controller output (actual normed force & torque)
        self.max_thrust = torch.tensor(cfg.max_normed_thrust, device=device)
        self.max_torque = torch.tensor(cfg.max_normed_torque, device=device)

    def __call__(self, state, action):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        raise NotImplementedError

    def postprocess(self, normed_thrust, normed_torque):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        normed_torque = normed_torque * self.torque_ratio
        normed_thrust = normed_thrust * self.thrust_ratio
        # compensate gravity
        if self.cfg.compensate_gravity:
            normed_thrust += 1.
        thrust = normed_thrust * self.gravity * self.mass
        torque = normed_torque * self.inertia
        return thrust, torque


class RateController(BaseController):
    """
    Body Rate Controller.
    
    Take desired thrust, roll rate, picth rate, and yaw rate as input
    and output actual force and torque to be applied on the robot.
    """
    def __init__(
        self,
        mass: torch.Tensor,
        inertia: torch.Tensor,
        gravity: torch.Tensor,
        cfg: DictConfig,
        device: torch.device
    ):
        super().__init__(mass, inertia, gravity, cfg, device)
        self.K_angvel = torch.tensor(cfg.K_angvel, device=device)
        
        # lower bound of controller input (action)
        self.min_action = torch.tensor([
            cfg.min_normed_thrust,
            cfg.min_roll_rate,
            cfg.min_pitch_rate,
            cfg.min_yaw_rate
        ], device=device)
        
        # upper bound of controller input (action)
        self.max_action = torch.tensor([
            cfg.max_normed_thrust,
            cfg.max_roll_rate,
            cfg.max_pitch_rate,
            cfg.max_yaw_rate
        ], device=device)
    
    def __call__(self, state, action):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        
        # quaternion with real component first
        R_b2i = p3d_transforms.quaternion_to_matrix(state.roll(1, dims=-1))
        # for numeric stability, very important
        R_b2i.clamp_(min=-1.0+1e-6, max=1.0-1e-6)
        # Convert current rotation matrix to euler angles
        R_i2b = torch.transpose(R_b2i, -1, -2)
        
        desired_angvel_b = action[:, 1:]
        actual_angvel_b = torch.bmm(R_i2b, state[:, 10:13].unsqueeze(2)).squeeze(2)
        angvel_err = desired_angvel_b - actual_angvel_b
        
        # Ω × JΩ
        cross = torch.cross(actual_angvel_b, self.inertia * actual_angvel_b, dim=1)
        cross.div_(torch.max(cross.norm(dim=-1, keepdim=True) / 100,
                             torch.tensor(1., device=cross.device)).detach())
        angacc = self.torque_ratio * self.K_angvel * angvel_err
        torque = self.inertia * angacc + cross
        thrust = action[:, 0] * self.thrust_ratio * self.gravity * self.mass
        return thrust, torque

@torch.jit.script
def compute_vee_map(skew_matrix: torch.Tensor) -> torch.Tensor:
    """Return vee map of skew matrix."""
    vee_map = torch.stack(
        [-skew_matrix[:, 1, 2], skew_matrix[:, 0, 2], -skew_matrix[:, 0, 1]], dim=1)
    return vee_map