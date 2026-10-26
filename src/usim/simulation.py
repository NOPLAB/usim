"""Dependency-free mobile robot contracts shared by simulator ports and ROS I/O."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Callable, Protocol


class ConfigurationError(ValueError):
    """Identify an invalid simulation parameter at its public boundary."""

    parameter: str

    def __init__(self, parameter: str) -> None:
        self.parameter = parameter
        super().__init__(f'invalid simulation parameter: {parameter}')


@dataclass(frozen=True, slots=True)
class Ros2Config:
    """Standard ROS topic and service configuration."""

    node_name: str = 'usim_ros2_bridge'
    cmd_vel_topic: str = '/cmd_vel'
    odom_topic: str = '/odom'
    rgb_topic: str = '/camera/color/image_raw'
    depth_topic: str = '/camera/depth/image_raw'
    motor_service: str = '/motor_power'
    scan_topic: str = '/scan_raw'
    publish_lidar_tf: bool = True
    points_topic: str = '/lidar/points'

    def __post_init__(self) -> None:
        for name, value in (
            ('node_name', self.node_name),
            ('cmd_vel_topic', self.cmd_vel_topic),
            ('odom_topic', self.odom_topic),
            ('rgb_topic', self.rgb_topic),
            ('depth_topic', self.depth_topic),
            ('motor_service', self.motor_service),
            ('scan_topic', self.scan_topic),
            ('points_topic', self.points_topic),
        ):
            if not value.strip():
                raise ConfigurationError(name)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Native world inputs and a wall-time run limit after engine initialization."""

    world: Path
    robot_urdf: Path
    robot_name: str = 'mobile_robot'
    base_link: str = 'base_link'
    left_joint: str = 'left_wheel_joint'
    right_joint: str = 'right_wheel_joint'
    wheel_radius: float = 0.08
    wheel_separation: float = 0.32
    camera_offset: tuple[float, float, float] = (0.12, 0.0, 0.28)
    camera_width: int = 640
    camera_height: int = 480
    camera_hz: float = 10.0
    camera_enabled: bool = True
    headless: bool = False
    max_seconds: float = 0.0
    ros: Ros2Config | None = field(default_factory=Ros2Config)
    left_joints: tuple[str, ...] | None = None
    right_joints: tuple[str, ...] | None = None
    initial_position: tuple[float, float, float] | None = None
    initial_yaw: float = 0.0
    real_time: bool = False
    lidar_enabled: bool = False
    lidar_offset: tuple[float, float, float] = (0.0, 0.0, 0.3)
    lidar_hz: float = 10.0
    lidar_samples: int = 360
    lidar_angle_min: float = -math.pi
    lidar_angle_max: float = math.pi
    lidar_min_range: float = 0.05
    lidar_max_range: float = 30.0
    lidar_vertical_angles: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ('wheel_radius', self.wheel_radius),
            ('wheel_separation', self.wheel_separation),
            ('camera_hz', self.camera_hz),
            ('lidar_hz', self.lidar_hz),
            ('lidar_max_range', self.lidar_max_range),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ConfigurationError(name)
        if not math.isfinite(self.max_seconds) or self.max_seconds < 0:
            raise ConfigurationError('max_seconds')
        if not math.isfinite(self.initial_yaw):
            raise ConfigurationError('initial_yaw')
        for name in ('initial_position', 'lidar_offset'):
            value = getattr(self, name)
            if value is None and name == 'initial_position':
                continue
            if (
                value is None
                or len(value) != 3
                or any(not math.isfinite(component) for component in value)
            ):
                raise ConfigurationError(name)
            object.__setattr__(self, name, tuple(value))
        if not isinstance(self.lidar_samples, int) or self.lidar_samples < 2:
            raise ConfigurationError('lidar_samples')
        if self.lidar_vertical_angles is not None:
            if (
                not isinstance(self.lidar_vertical_angles, (tuple, list))
                or not self.lidar_vertical_angles
                or any(
                    not math.isfinite(angle) or abs(angle) > math.pi / 2
                    for angle in self.lidar_vertical_angles
                )
            ):
                raise ConfigurationError('lidar_vertical_angles')
            object.__setattr__(self, 'lidar_vertical_angles', tuple(self.lidar_vertical_angles))
        if (
            not math.isfinite(self.lidar_angle_min)
            or not math.isfinite(self.lidar_angle_max)
            or self.lidar_angle_min >= self.lidar_angle_max
        ):
            raise ConfigurationError('lidar angles')
        if (
            not math.isfinite(self.lidar_min_range)
            or self.lidar_min_range < 0
            or self.lidar_min_range >= self.lidar_max_range
        ):
            raise ConfigurationError('lidar_min_range')
        if self.camera_width < 1 or self.camera_height < 1:
            raise ConfigurationError('camera dimensions')
        if len(self.camera_offset) != 3 or any(
            not math.isfinite(value) for value in self.camera_offset
        ):
            raise ConfigurationError('camera_offset')
        if not all((self.robot_name, self.base_link, self.left_joint, self.right_joint)):
            raise ConfigurationError('robot and frame names')
        if self.left_joint == self.right_joint:
            raise ConfigurationError('wheel joints must differ')
        for name, group in (
            ('left_joints', self.left_joints),
            ('right_joints', self.right_joints),
        ):
            if group is not None:
                if not isinstance(group, (tuple, list)) or not group:
                    raise ConfigurationError(name)
                if any(not isinstance(joint, str) or not joint.strip() for joint in group):
                    raise ConfigurationError(name)
                if len(set(group)) != len(group):
                    raise ConfigurationError(name)
                object.__setattr__(self, name, tuple(group))
        if set(self.left_wheel_joints) & set(self.right_wheel_joints):
            raise ConfigurationError('wheel groups must be disjoint')
        if len(self.left_wheel_joints) != len(self.right_wheel_joints):
            raise ConfigurationError('wheel groups must have equal pair counts')

    @property
    def left_wheel_joints(self) -> tuple[str, ...]:
        """Selected left-side drives; an omitted group retains the singular API."""
        return self.left_joints if self.left_joints is not None else (self.left_joint,)

    @property
    def right_wheel_joints(self) -> tuple[str, ...]:
        """Selected right-side drives; an omitted group retains the singular API."""
        return self.right_joints if self.right_joints is not None else (self.right_joint,)


@dataclass(frozen=True, slots=True)
class Velocity:
    """Body-forward velocity in metres/second and yaw velocity in radians/second."""

    linear: float = 0.0
    angular: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.linear) or not math.isfinite(self.angular):
            raise ConfigurationError('velocity must be finite')


@dataclass(frozen=True, slots=True)
class RobotState:
    """World-source pose and body velocity at a simulation timestamp."""

    sim_time: float
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    velocity: Velocity


class Simulator(Protocol):
    """Own an engine's blocking lifecycle, not fictitious cross-engine lockstep."""

    def run(self, configuration: SimulationConfig, *, stop: Event | None = None) -> None:
        """Run until cancellation, deadline or failure and clean up owned resources."""
        ...


def wheel_velocities(
    linear: float, angular: float, radius: float, separation: float
) -> tuple[float, float]:
    """Convert body motion and physical geometry into wheel radians/second."""
    if not all(math.isfinite(value) for value in (linear, angular, radius, separation)):
        raise ConfigurationError('wheel motion must be finite')
    if radius <= 0 or separation <= 0:
        raise ConfigurationError('wheel geometry must be positive')
    return (
        (linear - angular * separation / 2) / radius,
        (linear + angular * separation / 2) / radius,
    )


class CommandGate:
    """Mutable motor gate with deterministic, injectable wall-clock watchdog timing."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock: Callable[[], float] = clock
        self._enabled: bool = False
        self._velocity: Velocity = Velocity()
        self._received_at: float | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_motor(self, enabled: bool) -> None:
        """Every motor request clears stored commands; enabling needs a fresh command."""
        self._enabled = enabled
        self._velocity = Velocity()
        self._received_at = None

    def command(self, velocity: Velocity) -> None:
        """Accept finite parsed motion only while enabled and clamp actuator demand."""
        if self._enabled:
            self._velocity = Velocity(
                max(-0.4, min(0.4, velocity.linear)), max(-1.0, min(1.0, velocity.angular))
            )
            self._received_at = self._clock()

    def sample(self) -> Velocity:
        """Return zero after a motor toggle or more than 0.5 seconds without a command."""
        if not self._enabled or self._received_at is None:
            return Velocity()
        if self._clock() - self._received_at > 0.5:
            return Velocity()
        return self._velocity
