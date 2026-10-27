"""Separate model, placement, controller, agent and execution specifications."""

from dataclasses import dataclass, field
from math import isfinite

from usim.model import ImportRequirements, ModelError, ModelSource


@dataclass(frozen=True, slots=True)
class Pose:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if len(self.position) != 3 or len(self.orientation_xyzw) != 4:
            raise ModelError('Pose requires three position and four quaternion coordinates')
        if not all(isfinite(value) for value in self.position + self.orientation_xyzw):
            raise ModelError('Pose coordinates must be finite')
        norm = sum(value * value for value in self.orientation_xyzw)
        if abs(norm - 1.0) > 1e-6:
            raise ModelError('Pose quaternion must have unit length')


@dataclass(frozen=True, slots=True)
class EntitySpec:
    name: str
    source: ModelSource
    pose: Pose = field(default_factory=Pose)
    requirements: ImportRequirements = field(default_factory=ImportRequirements)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ModelError('Entity name must not be empty')


@dataclass(frozen=True, slots=True)
class ControlSpec:
    joint_names: tuple[str, ...] = ()
    position_joints: tuple[str, ...] = ()
    velocity_joints: tuple[str, ...] = ()
    effort_joints: tuple[str, ...] = ()
    initial_qpos: tuple[float, ...] | None = None
    native_mode: str = 'pd_joint_pos'

    def __post_init__(self) -> None:
        groups = self.position_joints + self.velocity_joints + self.effort_joints
        if len(set(groups)) != len(groups):
            raise ModelError('Control groups must be disjoint')
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ModelError('Joint order must contain unique names')
        if self.joint_names and not set(groups) <= set(self.joint_names):
            raise ModelError('Controlled joints must belong to the joint order')
        if self.initial_qpos is not None:
            if not all(isfinite(value) for value in self.initial_qpos):
                raise ModelError('Initial positions must be finite')
            if self.joint_names and len(self.initial_qpos) != len(self.joint_names):
                raise ModelError('Initial positions must match the joint order')


@dataclass(frozen=True, slots=True)
class SensorSpec:
    name: str
    kind: str
    parent: str | None = None
    pose: Pose = field(default_factory=Pose)
    width: int = 640
    height: int = 480
    rate: float = 10.0

    def __post_init__(self) -> None:
        if not self.name.strip() or self.kind not in {'rgb', 'depth', 'rgbd', 'lidar'}:
            raise ModelError('A sensor needs a name and a supported sensor kind')
        if self.width < 1 or self.height < 1 or not isfinite(self.rate) or self.rate <= 0:
            raise ModelError('Sensor dimensions and rate must be positive')


@dataclass(frozen=True, slots=True)
class RegisteredAgentRef:
    backend: str
    uid: str

    def __post_init__(self) -> None:
        if not self.backend.strip() or not self.uid.strip():
            raise ModelError('Registered agent backend and UID must not be empty')


@dataclass(frozen=True, slots=True)
class AgentSpec:
    name: str
    model: ModelSource | RegisteredAgentRef
    control: ControlSpec = field(default_factory=ControlSpec)
    sensors: tuple[SensorSpec, ...] = ()
    pose: Pose = field(default_factory=Pose)
    requirements: ImportRequirements = field(default_factory=ImportRequirements)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ModelError('Agent name must not be empty')
        names = tuple(sensor.name for sensor in self.sensors)
        if len(names) != len(set(names)):
            raise ModelError('Agent sensor names must be unique')


@dataclass(frozen=True, slots=True)
class SceneSpec:
    entities: tuple[EntitySpec, ...] = ()
    agents: tuple[AgentSpec, ...] = ()
    world: ModelSource | None = None

    def __post_init__(self) -> None:
        names = tuple(entity.name for entity in self.entities + self.agents)
        if any(not name.strip() for name in names) or len(set(names)) != len(names):
            raise ModelError('Scene entity and agent names must be nonempty and unique')


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    mode: str = 'episode'
    backend: str = 'cpu'
    n_envs: int = 1
    sim_rate: float = 30.0
    max_episode_steps: int = 200
    render_mode: str = 'none'
    obs_mode: str = 'state'
    native_task: str = 'JointControl'

    def __post_init__(self) -> None:
        if self.mode not in {'episode', 'continuous'}:
            raise ModelError(f'Unknown execution mode: {self.mode}')
        if self.n_envs < 1 or self.max_episode_steps < 1:
            raise ModelError('Environment count and episode length must be positive')
        if not isfinite(self.sim_rate) or self.sim_rate <= 0:
            raise ModelError('Simulation rate must be finite and positive')
