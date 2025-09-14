from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Literal, Callable
from dataclasses import MISSING
from collections.abc import Sequence
from dataclasses import dataclass

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.assets.articulation import Articulation
from isaaclab.utils.noise import NoiseCfg
from isaaclab.utils import DelayBuffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.envs.mdp.actions import JointAction, JointActionCfg

import math

from isaaclab.markers import VisualizationMarkers
from isaaclab.utils import DelayBuffer

MOTOR_DIRECTION_CW = -1
MOTOR_DIRECTION_CCW = 1


@dataclass
class RotorConfig:

    baselinkName: str = MISSING
    """ Name of the base link of the drone. This is used to find the joints and links"""
    jointNames: list[str] = MISSING
    """ Names of the joints that the motors are attached to. """
    linkNames: list[str] = MISSING
    """ Names of the links that the motors are attached to. """
    turningDirections: list[int] = MISSING
    """ Directions of the motors, either MOTOR_DIRECTION_CW or MOTOR_DIRECTION_CCW. """
    timeConstantUp: float = 0.00125
    """ Time constant for the motor to reach its maximum speed when increasing the speed. """
    timeConstantDown: float = 0.0025
    """ Time constant for the motor to reach its maximum speed when decreasing the speed. """
    maxRotVelocity: float = 3800.0
    """ Maximum rotation velocity of the motors in RPM. """
    minRotVelocity: float = 100.0
    """ Minimum rotation velocity of the motors in RPM. """
    motorConstant: float = 4.33948e-07
    """ Force constant of the motors in N/(RPM). """
    momentConstant: float = 0.00932
    """ Moment constant of the motors in Nm/(RPM). """
    rotorDragCoefficient: float = 0.0000175
    """ Drag coefficient of the rotors. This is used to calculate the drag force on the
    rotors based on the relative airspeed. """
    rollingMomentCoefficient: float = 0.0
    """ Rolling moment coefficient of the rotors. This is used to calculate the rolling moment
    on the drone based on the relative airspeed. """
    rotorVelocitySlowdownSim: float = 100.0
    """ Factor to slow down the rotor velocity in simulation. This is used to avoid numerical
    instabilities when the rotor velocity is too high. """
    maxRelativeAirspeed: float = 25.0
    """ Maximum relative airspeed of the rotors. This is used to calculate the drag force
    on the rotors based on the relative airspeed. """
    bodyDragCoefficient: float = 0
    """ Drag coefficient of the body. This is used to calculate the drag force on the body"""


class RotorModel:
    def __init__(self, robot_assert: Articulation, rotor_config: RotorConfig):
        self._robot_assert = robot_assert
        self._rotor_config = rotor_config

        # find the joints and links
        self.joint_ids, _ = self._robot_assert.find_joints(
            self._rotor_config.jointNames, preserve_order=True
        )
        self.link_ids, _ = self._robot_assert.find_bodies(
            self._rotor_config.linkNames, preserve_order=True
        )
        self.baselink_ids, _ = self._robot_assert.find_bodies(
            [self._rotor_config.baselinkName], preserve_order=True
        )
        if len(self.joint_ids) != len(self.link_ids):
            raise ValueError(
                "Number of joints and links must be the same. "
                f"Got {len(self.joint_ids)} joints and {len(self.link_ids)} links."
            )
        if len(self.joint_ids) != len(self._rotor_config.turningDirections):
            raise ValueError(
                "Number of joints and turning directions must be the same. "
                f"Got {len(self.joint_ids)} joints and "
                f"{len(self._rotor_config.turningDirections)} turning directions."
            )

        # check if max thrust > mass
        total_thrust = (
            self._rotor_config.motorConstant
            * self._rotor_config.maxRotVelocity**2
            * self.num_rotors
        )
        total_gravity = self._robot_assert.data.default_mass[0].sum() * 9.81
        if total_thrust < total_gravity:
            raise ValueError(
                f"Maximum thrust of the rotors is too low. Max thrust: {total_thrust}, "
                f"mass of the robot: {total_gravity}."
            )
        else:
            print(
                f"Maximum thrust of the rotors is sufficient. "
                f"Max thrust: {total_thrust}, "
                f"mass of the robot: {total_gravity}."
            )

        # init buffers
        self.param_turningDirections = torch.tensor(
            self._rotor_config.turningDirections,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.param_timeConstantUp = torch.full(
            (self.num_robot, self.num_rotors),
            self._rotor_config.timeConstantUp,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.param_timeConstantDown = torch.full(
            (self.num_robot, self.num_rotors),
            self._rotor_config.timeConstantDown,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.param_maxRotVelocity = torch.full(
            (self.num_robot, self.num_rotors),
            self._rotor_config.maxRotVelocity,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.param_minRotVelocity = torch.full(
            (self.num_robot, self.num_rotors),
            self._rotor_config.minRotVelocity,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.param_motorConstant = torch.full(
            (self.num_robot, self.num_rotors),
            self._rotor_config.motorConstant,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.param_momentConstant = torch.full(
            (self.num_robot, self.num_rotors),
            self._rotor_config.momentConstant,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.param_rotorDragCoefficient = torch.full(
            (self.num_robot, self.num_rotors),
            self._rotor_config.rotorDragCoefficient,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.param_bodyDragCoefficient = torch.full(
            (self.num_robot, 1),
            self._rotor_config.bodyDragCoefficient,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.param_rollingMomentCoefficient = torch.full(
            (self.num_robot, self.num_rotors),
            self._rotor_config.rollingMomentCoefficient,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.param_maxRelativeAirspeed = torch.full(
            (self.num_robot, self.num_rotors),
            self._rotor_config.maxRelativeAirspeed,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        # init buffers for rotor velocities
        self.rotor_velocities = torch.zeros(
            (self.num_robot, self.num_rotors),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.link_zero_buf = torch.zeros(
            (self.num_robot, self.num_rotors, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.link_force_buf = torch.zeros(
            (self.num_robot, self.num_rotors, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.link_torque_buf = torch.zeros(
            (self.num_robot, self.num_rotors, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.baselink_force_buf = torch.zeros(
            (self.num_robot, 1, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.baselink_torque_buf = torch.zeros(
            (self.num_robot, 1, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.baselink_zero_buf = torch.zeros(
            (self.num_robot, 1, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.rotor_command = torch.zeros(
            (self.num_robot, self.num_rotors),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        self.zero_rotor_constants = torch.zeros(
            (self.num_robot, self.num_rotors),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

    def compute(
        self, rotor_command: torch.Tensor, dt: float, write_to_sim: bool = True
    ):
        """
        Compute the rotor velocities and forces based on the rotor commands.
        The rotor commands are expected to be in the range [0, 1].
        """
        self.link_force_buf.zero_()
        self.link_torque_buf.zero_()

        self.rotor_command = rotor_command.clamp(0.0, 1.0)
        self.target_rotor_velocities = (
            self.rotor_command * (self.param_maxRotVelocity - self.param_minRotVelocity)
            + self.param_minRotVelocity
        )

        # compute the forces and torques on the links
        # rotor is not reversed, so the force is always positive
        force = torch.square(self.rotor_velocities) * self.param_motorConstant

        rotor_lin_vel_b = math_utils.quat_rotate_inverse(
            self._robot_assert.data.body_quat_w[:, self.link_ids, :],
            self._robot_assert.data.body_lin_vel_w[:, self.link_ids, :],
        )
        rotor_lin_vel_z = rotor_lin_vel_b[:, :, 2]
        force_scale = torch.clamp(
            1.0 - (rotor_lin_vel_z / self.param_maxRelativeAirspeed), 0, 1
        )
        # TODO:: check shape
        self.link_force_buf[:, :, 2] += force * force_scale
        velocity_perpendicular_to_rotor_axis = rotor_lin_vel_b[:, :, [0, 1]]
        # air_drag = -std::abs(real_motor_velocity) * rotor_drag_coefficient_ * velocity_perpendicular_to_rotor_axis;
        air_drag_xy = (
            -self.rotor_velocities * self.param_rotorDragCoefficient
        ).unsqueeze(-1) * velocity_perpendicular_to_rotor_axis
        self.link_force_buf[:, :, :2] += air_drag_xy

        drag_torque = -force * self.param_momentConstant * self.param_turningDirections
        self.link_torque_buf[:, :, 2] += drag_torque
        #   rolling_moment = -std::abs(real_motor_velocity) * turning_direction_ * rolling_moment_coefficient_ * velocity_perpendicular_to_rotor_axis;
        rolling_moment_xy = (
            -self.rotor_velocities
            * self.param_turningDirections
            * self.param_rollingMomentCoefficient
        ).unsqueeze(-1) * velocity_perpendicular_to_rotor_axis
        self.link_torque_buf[:, :, :2] += rolling_moment_xy

        # update the rotor velocities based on the target velocities
        tau = torch.where(
            self.target_rotor_velocities > self.rotor_velocities,
            self.param_timeConstantUp,
            self.param_timeConstantDown,
        )
        alpha = torch.exp(-dt / tau)
        self.rotor_velocities = (
            alpha * self.rotor_velocities + (1.0 - alpha) * self.target_rotor_velocities
        ).clamp(self.zero_rotor_constants, self.param_maxRotVelocity)
        self.baselink_torque_buf = self.link_torque_buf.sum(dim=1, keepdim=True)
        self.baselink_force_buf = -(
            self.param_bodyDragCoefficient * self._robot_assert.data.root_lin_vel_b
        ).unsqueeze(1)
        if write_to_sim:
            self._robot_assert.set_external_force_and_torque(
                forces=self.link_force_buf,
                torques=self.link_zero_buf,
                body_ids=self.link_ids,
            )

            self._robot_assert.set_external_force_and_torque(
                self.baselink_force_buf,
                self.baselink_torque_buf,
                body_ids=self.baselink_ids,  # Assuming the first body is the root body
            )

            # write the rotor velocities to the simulation for visualization
            self._robot_assert.write_joint_velocity_to_sim(
                velocity=self.rotor_velocities
                / self._rotor_config.rotorVelocitySlowdownSim
                * self.param_turningDirections,
                joint_ids=self.joint_ids,
            )

    def reset(self, env_ids: Sequence[int] | None = None):
        self.rotor_velocities[env_ids] = 0.0
        self.link_force_buf[env_ids] = 0.0
        self.link_torque_buf[env_ids] = 0.0
        self._robot_assert.write_joint_velocity_to_sim(
            torch.zeros(
                len(env_ids), self.num_rotors, dtype=torch.float32, device=self.device
            ),
            env_ids=env_ids,
        )

    @property
    def num_rotors(self) -> int:
        """Number of rotors in the model."""
        return len(self.joint_ids)

    @property
    def num_robot(self) -> int:
        return self._robot_assert.num_instances

    @property
    def device(self):
        """Device of the model."""
        return self._robot_assert.device


@dataclass
class ControlAllocationCfg:
    """
    Configuration for the control allocation.
    This is used to allocate the rotor commands to the motors.
    """

    mixer_px4: tuple[tuple[float, ...], ...] = (
        (-0.70711, 0.70711, 1.00000, 0, 0, -1.00000),
        (0.70711, -0.70711, 1.00000, 0, 0, -1.00000),
        (0.70711, 0.70711, -1.00000, 0, 0, -1.00000),
        (-0.70711, -0.70711, -1.00000, 0, 0, -1.00000),
    )
    """Mixer matrix for the PX4 control allocation. Get from px4 console with `control_allocation status' with modified version of PX4 firmware."""


class ControlAllocationSimple:
    def __init__(self, cfg: ControlAllocationCfg, env: ManagerBasedRLEnv):
        """
        Control allocation for the rotor commands.
        This is used to allocate the rotor commands to the motors.
        The configuration is based on the PX4 mixer.
        """
        self._env = env
        self._cfg = cfg
        self._mixer = torch.tensor(
            self._cfg.mixer_px4,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        # convert from px4 coordinates to isaaclab coordinates
        self._mixer[:, 1] *= -1.0  # flip y-axis
        self._mixer[:, 4] *= -1.0  # flip y-axis for roll
        self._mixer[:, 2] *= -1.0  # flip z-axis
        self._mixer[:, 5] *= -1.0  # flip z-axis for thrust
        self._mixer = self._mixer.unsqueeze(0)

    def torque_4d_to_torque_6d(self, torque_4d: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                torque_4d[:, :3],  # roll, pitch, yaw
                torch.zeros_like(torque_4d[:, :2]),  # x, y
                torque_4d[:, [3]],  #  z
            ],
            dim=-1,
        )

    def compute(self, torque_command: torch.Tensor) -> torch.Tensor:
        """
        Compute the rotor commands based on the torque command.
        torque command is expected to be tensor of shape (num_envs, 6)
        where the last dimension is (roll, pitch, yaw, x, y, z).
        The output is a tensor of shape (num_envs, num_rotors).
        """
        if torque_command.shape[-1] != 6:
            raise ValueError(
                f"Torque command must have shape (num_envs, 6), got {torque_command.shape}"
            )

        # compute the rotor commands
        rotor_commands = torch.matmul(
            self._mixer, torque_command.unsqueeze(-1)
        ).squeeze(-1)

        # clamp the rotor commands to [0, 1]
        return rotor_commands.clamp(0.0, 1.0)

    @property
    def device(self):
        return self._env.device

    @property
    def num_rotors(self) -> int:
        """Number of rotors in the model."""
        return self._mixer.shape[0]

    @property
    def num_inputs(self) -> int:
        """Number of inputs to the control allocation."""
        return self._mixer.shape[1]


class RateController:
    def __init__(
        self, cfg: RateControllerParams, asset: Articulation, env: ManagerBasedRLEnv
    ):
        self.cfg = cfg
        self._env = env
        self._asset = asset

        self.prev_rate_error = torch.zeros(
            self.num_envs,
            self.torque_dim,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self._rate_int = torch.zeros(
            self.num_envs,
            self.torque_dim,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self._lim_int = torch.tensor(
            self.cfg.limit_int,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).repeat(self.num_envs, 1)
        self._gain_k = torch.tensor(
            self.cfg.gain_k,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).repeat(self.num_envs, 1)
        self._gain_p = torch.tensor(
            self.cfg.gain_p,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).repeat(self.num_envs, 1)
        self._gain_i = torch.tensor(
            self.cfg.gain_i,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).repeat(self.num_envs, 1)
        self._gain_d = torch.tensor(
            self.cfg.gain_d,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).repeat(self.num_envs, 1)

        yaw_lpf_timeconstant = 1.0 / (6.28318530718 * self.cfg.yaw_torque_cutoff_hz)
        denominator = yaw_lpf_timeconstant + self._env.physics_dt

        self._yaw_lpf_alpha = self._env.physics_dt / denominator

        self.last_torque = torch.zeros(
            self.num_envs,
            self.torque_dim,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )

    def update_integral(self, rate_error, dt):
        i_factor = rate_error / self.deg2rad(400.0)
        i_factor = torch.clamp(1.0 - i_factor * i_factor, 0.0, 1.0)
        rate_i = self._rate_int + i_factor * self._gain_i * rate_error * dt
        if torch.any(torch.isnan(rate_i)) or torch.any(torch.isinf(rate_i)):
            return
        self._rate_int = torch.clip(rate_i, -self._lim_int, self._lim_int)

    def compute(self, target_rate: torch.Tensor, current_rate: torch.Tensor, dt: float):
        rate_error = target_rate - current_rate
        derivative_error = (rate_error - self.prev_rate_error) / dt
        self.prev_rate_error = rate_error
        torque = self._gain_k * (
            self._gain_p * rate_error + self._rate_int - self._gain_d * derivative_error
        )
        self.update_integral(rate_error, dt)

        # apply low-pass filter to yaw torque
        torque[:, 2] = self.last_torque[:, 2] + self._yaw_lpf_alpha * (
            torque[:, 2] - self.last_torque[:, 2]
        )
        self.last_torque = torque.clone()
        return torque.clamp(-1.0, 1.0)

    def reset(self, env_ids: Sequence[int] | None = None):
        self.prev_rate_error[env_ids] = 0.0
        self._rate_int[env_ids] = 0.0
        self.last_torque[env_ids] = 0.0

    def deg2rad(self, deg):
        return deg / 180.0 * 3.14159265358979323846

    @property
    def torque_dim(self):
        return 3

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device


@dataclass
class RateControllerParams:
    limit_int: tuple[float, float, float] = (0.3, 0.3, 0.3)
    """ The limits of the integral term."""

    gain_k: tuple[float, float, float] = (1.0, 1.0, 1.0)

    gain_p: tuple[float, float, float] = (0.28, 0.4, 0.6)
    """ The proportional gains."""

    gain_i: tuple[float, float, float] = (0.2, 0.2, 0.06)
    """ The integral gains."""

    gain_d: tuple[float, float, float] = (0.0006, 0.0008, 0.0)
    """ The derivative gains."""

    yaw_torque_cutoff_hz: float = 2
    """ The cutoff frequency for the yaw torque in Hz. This is used to filter the yaw torque"""


class RotorActionTerm(ActionTerm):
    _cfg: RotorActionTermCfg
    _rotor_config: RotorConfig

    def __init__(self, cfg: RotorActionTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._rotor_config = cfg.rotor_config
        self._motor_model = RotorModel(
            robot_assert=self._asset,
            rotor_config=self._rotor_config,
        )

        self._raw_actions = torch.zeros(
            self.num_envs,
            self.action_dim,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )

        self._processed_actions = torch.zeros_like(
            self._raw_actions,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )

    @property
    def num_rotors(self):
        return self._motor_model.num_rotors

    @property
    def action_dim(self) -> int:
        """Dimension of Action"""
        return self._motor_model.num_rotors

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions

        self._processed_actions = self._raw_actions.clamp(0.0, 1.0)

    def apply_actions(self):
        self._motor_model.compute(
            rotor_command=self._processed_actions,
            dt=self._env.physics_dt,
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        self._motor_model.reset(env_ids=env_ids)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


@dataclass
class RotorActionTermCfg(ActionTermCfg):
    """
    Configuration for the rotor action term.
    This term applies the rotor commands to the robot articulation.
    """

    class_type: type[ActionTerm] = RotorActionTerm
    """Type of the action term."""

    rotor_config: RotorConfig = MISSING
    """Configuration for the rotor model."""

    asset_name: str = MISSING
    """Name of the asset to which the action term is applied."""


class RotorAction6DTerm(ActionTerm):
    cfg: RotorAction6DTermCfg
    _env: ManagerBasedEnv

    def __init__(self, cfg: RotorAction6DTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._rotor_config = cfg.rotor_config
        self._motor_model = RotorModel(
            robot_assert=self._asset,
            rotor_config=self._rotor_config,
        )
        self._control_allocation = ControlAllocationSimple(
            cfg=cfg.control_allocation_cfg, env=env
        )

        self._raw_actions = torch.zeros(
            self.num_envs,
            6,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )

        self._processed_actions = torch.zeros_like(
            self._raw_actions,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.rotor_commands = torch.zeros(
            (self.num_envs, self._motor_model.num_rotors),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )

        self._command_scale = torch.tensor(
            self.cfg.command_scale,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).reshape(1, -1)
        self._command_clip_min = torch.tensor(
            self.cfg.command_clip_min,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).reshape(1, -1)
        self._command_clip_max = torch.tensor(
            self.cfg.command_clip_max,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).reshape(1, -1)

    @property
    def num_rotors(self):
        return self._motor_model.num_rotors

    @property
    def action_dim(self) -> int:
        """Dimension of Action"""
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions

        # scale the actions
        self._processed_actions = (self._raw_actions * self._command_scale).clamp(
            self._command_clip_min, self._command_clip_max
        )

    def apply_actions(self):
        self.rotor_commands = self._control_allocation.compute(
            torque_command=self._processed_actions,
        )

        self._motor_model.compute(
            rotor_command=self.rotor_commands,
            dt=self._env.physics_dt,
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        self._motor_model.reset(env_ids=env_ids)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self.rotor_commands[env_ids] = 0.0


@dataclass
class RotorAction6DTermCfg(ActionTermCfg):
    """
    Configuration for the 6D rotor action term.
    This term applies the rotor commands to the robot articulation in 6D space.
    """

    class_type: type[ActionTerm] = RotorAction6DTerm
    """Type of the action term."""

    rotor_config: RotorConfig = MISSING
    """Configuration for the rotor model."""

    asset_name: str = MISSING
    """Name of the asset to which the action term is applied."""

    control_allocation_cfg: ControlAllocationCfg = MISSING
    """Configuration for the control allocation."""

    command_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    """Scale for the command values. This is used to scale the actions before applying them."""
    command_clip_min: tuple[float, ...] = (-5.0, -5.0, -1.0, 0.0, 0.0, 0.0)
    """Clip values for the command values. This is used to clip the actions before applying them."""
    command_clip_max: tuple[float, ...] = (5.0, 5.0, 1.0, 0.0, 0.0, 1.0)
    """Clip values for the command values. This is used to clip the actions before applying them."""


class RateAction(ActionTerm):
    cfg: RateActionCfg
    _asset: Articulation

    def __init__(self, cfg: RateActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._rotor_config = cfg.rotor_config
        self._motor_model = RotorModel(
            robot_assert=self._asset, rotor_config=self._rotor_config
        )
        self._control_allocation = ControlAllocationSimple(
            cfg=cfg.control_allocation_cfg, env=self._env
        )
        self._rate_controller = RateController(
            cfg=cfg.rate_controller_cfg, asset=self._asset, env=self._env
        )

        self._raw_actions = torch.zeros(
            self.num_envs,
            self.action_dim,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )

        self._processed_actions = torch.zeros_like(
            self._raw_actions,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self._delayed_processed_actions = torch.zeros_like(
            self._processed_actions,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.rotor_commands = torch.zeros(
            (self.num_envs, self._motor_model.num_rotors),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )

        self._command_scale = torch.tensor(
            self.cfg.command_scale,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).reshape(1, -1)
        self._command_clip_min = torch.tensor(
            self.cfg.command_clip_min,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).reshape(1, -1)
        self._command_clip_max = torch.tensor(
            self.cfg.command_clip_max,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).reshape(1, -1)

        self._processed_action_delay_buffer = DelayBuffer(
            history_length=self.cfg.max_delay_phys_step,
            batch_size=self.num_envs,
            device=self.device,
        )

        self.ALL_ENV_IDS = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=torch.int32,
            requires_grad=False,
        )

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def num_rotors(self):
        return self._motor_model.num_rotors

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions

        # scale the actions
        self._processed_actions = (self._raw_actions * self._command_scale).clamp(
            self._command_clip_min, self._command_clip_max
        )

    @property
    def action_target_rate(self) -> torch.Tensor:
        """
        Returns the target rate for the action term.
        The target rate is computed by scaling the processed actions.
        """
        return self._processed_actions[:, :3]

    @property
    def delayed_action_target_rate(self) -> torch.Tensor:
        """
        Returns the target rate for the action term.
        The target rate is computed by scaling the processed actions.
        """
        return self._delayed_processed_actions[:, :3]

    @property
    def action_target_thrust(self) -> torch.Tensor:
        """
        Returns the target thrust for the action term.
        The target thrust is computed by scaling the processed actions.
        """
        return self._processed_actions[:, 3]

    @property
    def delayed_action_target_thrust(self) -> torch.Tensor:
        """
        Returns the target thrust for the action term.
        The target thrust is computed by scaling the processed actions.
        """
        return self._delayed_processed_actions[:, 3]

    def torque_thrust_6d(
        self, torque: torch.Tensor, thrust: torch.Tensor
    ) -> torch.Tensor:
        # merge the torque(3) and thrust(1) into a 6D tensor
        command = torch.zeros(
            (self.num_envs, 6),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        command[:, :3] = torque
        command[:, 5] = thrust
        return command

    def rate_noise_action(self, actitons: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the actions.
        This is used to add noise to the actions before applying them.
        """
        # create a noise tensor with the same shape as the actions in range [-1, 1]
        noise = torch.rand_like(actitons) * 2.0 - 1.0
        # scale the noise by the action scale
        noise *= self.cfg.rate_action_scale_noise
        # add the noise to the actions
        scale = 1.0 + noise
        return actitons * scale

    def thrust_noise_action(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the thrust actions.
        This is used to add noise to the thrust actions before applying them.
        """
        # create a noise tensor with the same shape as the actions in range [-1, 1]
        noise = torch.rand_like(actions) * 2.0 - 1.0
        # scale the noise by the action scale
        noise *= self.cfg.thrust_action_scale_noise
        # add the noise to the actions
        scale = 1.0 + noise
        return actions * scale

    def current_rate_noise(self, current_rate: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the current rate.
        This is used to add noise to the current rate before applying them.
        """
        # create a noise tensor with the same shape as the current rate in range [-1, 1]
        noise = torch.rand_like(current_rate) * 2.0 - 1.0
        # scale the noise by the action scale
        noise[:, 0] *= self.cfg.sense_rate_noise_scale[0]
        noise[:, 1] *= self.cfg.sense_rate_noise_scale[1]
        noise[:, 2] *= self.cfg.sense_rate_noise_scale[2]
        return current_rate + noise

    def apply_actions(self):
        self._delayed_processed_actions = self._processed_action_delay_buffer.compute(
            self._processed_actions
        )
        self.torque_command = self._rate_controller.compute(
            self.rate_noise_action(self.delayed_action_target_rate),
            self.current_rate_noise(self._asset.data.root_ang_vel_b),
            self._env.physics_dt,
        )
        self.torque_6d_command = self.torque_thrust_6d(
            torque=self.torque_command,
            thrust=self.thrust_noise_action(self.delayed_action_target_thrust),
        )

        self.rotor_commands = self._control_allocation.compute(
            torque_command=self.torque_6d_command,
        )

        self._motor_model.compute(
            rotor_command=self.rotor_commands,
            dt=self._env.physics_dt,
            write_to_sim=True,
        )
        # self._motor_model.compute(
        #     rotor_command=self.rotor_commands,
        #     dt=self._env.physics_dt,
        #     write_to_sim=False,
        # )
        # self._asset.set_external_force_and_torque(
        #     self._old_motor_model.rotor_thrust,
        #     self._old_motor_model.rotor_zero_moment,
        #     body_ids=self._motor_model.link_ids,
        # )
        # self._body_torque_buffer = self._old_motor_model.rotor_moment.sum(
        #     dim=1, keepdim=True
        # )
        # self._body_force_buffer = -self._asset.data.root_lin_vel_b.unsqueeze(1) * 0.4
        # self._asset.set_external_force_and_torque(
        #     self._body_force_buffer,
        #     self._body_torque_buffer,
        #     body_ids=[0],  # Assuming the first body is the root body
        # )

        # self._old_motor_model.calculate_rotor_dynamic(
        #     body_velocity=self._asset.data.root_ang_vel_b, cmds=self.rotor_commands
        # )
        # self._asset.set_external_force_and_torque(
        #     self._old_motor_model.rotor_thrust,
        #     self._old_motor_model.rotor_zero_moment,
        #     body_ids=self._motor_model.link_ids,
        # )
        # self._body_torque_buffer = self._old_motor_model.rotor_moment.sum(
        #     dim=1, keepdim=True
        # )
        # self._body_force_buffer = (
        #     -self._asset.data.root_lin_vel_b.unsqueeze(1)
        #     * self._motor_model.param_bodyDragCoefficient
        # )
        # assert torch.allclose(
        #     self._body_force_buffer, self._motor_model.baselink_force_buf
        # ), "Body force buffer does not match the motor model's baselink force buffer."
        # assert torch.allclose(
        #     self._body_torque_buffer, self._motor_model.baselink_torque_buf
        # ), "Body torque buffer does not match the motor model's baselink torque buffer."

        # assert torch.allclose(
        #     self._motor_model.link_force_buf,
        #     self._old_motor_model.rotor_thrust,
        #     atol=0.1,
        #     rtol=0.05,
        # ), "Link force buffer does not match the old motor model's rotor thrust."
        # self._asset.set_external_force_and_torque(
        #     self._body_force_buffer,
        #     self._body_torque_buffer,
        #     body_ids=[0],  # Assuming the first body is the root body
        # )

        # assert two model process same torque

    def reset(self, env_ids: Sequence[int] | None = None):
        self._motor_model.reset(env_ids=env_ids)
        self._rate_controller.reset(env_ids=env_ids)
        # self._control_allocation.re
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self.rotor_commands[env_ids] = 0.0

        # set a new random delay for environments in env_ids
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)
        time_lags = torch.randint(
            low=self.cfg.min_delay_phys_step,
            high=self.cfg.max_delay_phys_step + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self.device,
        )
        # set delays
        self._processed_action_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self._processed_action_delay_buffer.reset(env_ids)


@dataclass
class RateActionCfg(ActionTermCfg):
    """
    Configuration for the rate action term.
    This term applies the rate commands to the robot articulation.
    """

    class_type: type[ActionTerm] = RateAction
    """Type of the action term."""

    asset_name: str = MISSING
    """Name of the asset to which the action term is applied."""

    rotor_config: RotorConfig = MISSING
    """Configuration for the rotor model."""

    control_allocation_cfg: ControlAllocationCfg = MISSING
    """Configuration for the control allocation."""

    rate_controller_cfg: RateControllerParams = MISSING
    """Configuration for the rate controller."""

    command_scale: tuple[float, ...] = MISSING
    """Scale factor for action"""

    command_clip_min: tuple[float, ...] = MISSING
    """Minimum values for the action commands."""
    command_clip_max: tuple[float, ...] = MISSING
    """Maximum values for the action commands."""

    thrust_action_scale_noise: float = 0
    """Configuration for the noise applied to the thrust action."""
    rate_action_scale_noise: float = 0
    """Configuration for the noise applied to the rate action."""

    min_delay_phys_step: int = 0
    max_delay_phys_step: int = 0

    sense_rate_noise_scale: tuple[float, float, float] = (0.0, 0.0, 0.0)


# param type
PARAM_TYPE = Literal[
    "motorConstant",
    "momentConstant",
    "rollingMomentCoefficient",
    "rotorDragCoefficient",
]
from .command import TrajectoryRolloutCommand


class DeltaActionRateActionTerm(ActionTerm):
    cfg: DeltaActionCfg
    _asset: Articulation
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: RateActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._rotor_config = cfg.rotor_config
        self._motor_model = RotorModel(
            robot_assert=self._asset, rotor_config=self._rotor_config
        )
        self._control_allocation = ControlAllocationSimple(
            cfg=cfg.control_allocation_cfg, env=self._env
        )
        self._rate_controller = RateController(
            cfg=cfg.rate_controller_cfg, asset=self._asset, env=self._env
        )

        self._raw_actions = torch.zeros(
            self.num_envs,
            self.action_dim,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )

        self._processed_actions = torch.zeros_like(
            self._raw_actions,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self._delayed_processed_actions = torch.zeros_like(
            self._processed_actions,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.rotor_commands = torch.zeros(
            (self.num_envs, self._motor_model.num_rotors),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )

        self._command_scale = torch.tensor(
            self.cfg.command_scale,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).reshape(1, -1)
        self._command_clip_min = torch.tensor(
            self.cfg.command_clip_min,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).reshape(1, -1)
        self._command_clip_max = torch.tensor(
            self.cfg.command_clip_max,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).reshape(1, -1)

        self._processed_action_delay_buffer = DelayBuffer(
            history_length=self.cfg.max_delay_phys_step,
            batch_size=self.num_envs,
            device=self.device,
        )

        self._command: TrajectoryRolloutCommand = self._env.command_manager.get_term(
            self.cfg.ref_command_name
        )

        self.ALL_ENV_IDS = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=torch.int32,
            requires_grad=False,
        )

    @property
    def action_dim(self) -> int:
        return 4 + 4

    @property
    def num_rotors(self):
        return self._motor_model.num_rotors

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):

        # split action along dim1 4:4
        delta_action, rotor_drag_coef = torch.split(actions, [4, 4], dim=1)

        rotor_drag_coef.clamp_(min=-0.5, max=0.5)
        delta_action.clamp_(min=-2, max=2)
        # store the raw actions
        self._raw_actions[:] = actions

        # scale the actions
        ref_action = self._command.get_current_state("action_rate")

        self._processed_actions = ref_action + delta_action

        # change rotor drag coeff
        self._motor_model.param_rotorDragCoefficient = (
            self._rotor_config.rotorDragCoefficient * (1.0 + rotor_drag_coef)
        )

    @property
    def action_target_rate(self) -> torch.Tensor:
        """
        Returns the target rate for the action term.
        The target rate is computed by scaling the processed actions.
        """
        return self._processed_actions[:, :3]

    @property
    def delayed_action_target_rate(self) -> torch.Tensor:
        """
        Returns the target rate for the action term.
        The target rate is computed by scaling the processed actions.
        """
        return self._delayed_processed_actions[:, :3]

    @property
    def action_target_thrust(self) -> torch.Tensor:
        """
        Returns the target thrust for the action term.
        The target thrust is computed by scaling the processed actions.
        """
        return self._processed_actions[:, 3]

    @property
    def delayed_action_target_thrust(self) -> torch.Tensor:
        """
        Returns the target thrust for the action term.
        The target thrust is computed by scaling the processed actions.
        """
        return self._delayed_processed_actions[:, 3]

    def torque_thrust_6d(
        self, torque: torch.Tensor, thrust: torch.Tensor
    ) -> torch.Tensor:
        # merge the torque(3) and thrust(1) into a 6D tensor
        command = torch.zeros(
            (self.num_envs, 6),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        command[:, :3] = torque
        command[:, 5] = thrust
        return command

    def rate_noise_action(self, actitons: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the actions.
        This is used to add noise to the actions before applying them.
        """
        # create a noise tensor with the same shape as the actions in range [-1, 1]
        noise = torch.rand_like(actitons) * 2.0 - 1.0
        # scale the noise by the action scale
        noise *= self.cfg.rate_action_scale_noise
        # add the noise to the actions
        scale = 1.0 + noise
        return actitons * scale

    def thrust_noise_action(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the thrust actions.
        This is used to add noise to the thrust actions before applying them.
        """
        # create a noise tensor with the same shape as the actions in range [-1, 1]
        noise = torch.rand_like(actions) * 2.0 - 1.0
        # scale the noise by the action scale
        noise *= self.cfg.thrust_action_scale_noise
        # add the noise to the actions
        scale = 1.0 + noise
        return actions * scale

    def current_rate_noise(self, current_rate: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the current rate.
        This is used to add noise to the current rate before applying them.
        """
        # create a noise tensor with the same shape as the current rate in range [-1, 1]
        noise = torch.rand_like(current_rate) * 2.0 - 1.0
        # scale the noise by the action scale
        noise[:, 0] *= self.cfg.sense_rate_noise_scale[0]
        noise[:, 1] *= self.cfg.sense_rate_noise_scale[1]
        noise[:, 2] *= self.cfg.sense_rate_noise_scale[2]
        return current_rate + noise

    def apply_actions(self):
        self._delayed_processed_actions = self._processed_action_delay_buffer.compute(
            self._processed_actions
        )
        self.torque_command = self._rate_controller.compute(
            self.rate_noise_action(self.delayed_action_target_rate),
            self.current_rate_noise(self._asset.data.root_ang_vel_b),
            self._env.physics_dt,
        )
        self.torque_6d_command = self.torque_thrust_6d(
            torque=self.torque_command,
            thrust=self.thrust_noise_action(self.delayed_action_target_thrust),
        )

        self.rotor_commands = self._control_allocation.compute(
            torque_command=self.torque_6d_command,
        )

        self._motor_model.compute(
            rotor_command=self.rotor_commands,
            dt=self._env.physics_dt,
            write_to_sim=True,
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        self._motor_model.reset(env_ids=env_ids)
        self._rate_controller.reset(env_ids=env_ids)
        # self._control_allocation.re
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self.rotor_commands[env_ids] = 0.0

        # set a new random delay for environments in env_ids
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)
        time_lags = torch.randint(
            low=self.cfg.min_delay_phys_step,
            high=self.cfg.max_delay_phys_step + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self.device,
        )
        # set delays
        self._processed_action_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self._processed_action_delay_buffer.reset(env_ids)


@dataclass
class DeltaActionCfg(RateActionCfg):
    class_type = DeltaActionRateActionTerm

    ref_command_name: str = MISSING