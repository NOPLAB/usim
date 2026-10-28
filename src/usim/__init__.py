"""Portable robot simulation with optional mobile and manipulation runtimes."""

from usim.robot import MobileRobot, RobotGeometryError, render_robot
from usim.simulation import Ros2Config, RobotState, SimulationConfig, Simulator, Velocity
from usim.interface import EpisodeSimulator
from usim.types import JointCommand, Observation, SimulatorConfig, StepResult
from usim.factory import (
    capabilities,
    create_runner,
    create_simulator,
    list_runners,
    list_simulators,
    register_simulator,
)
from usim.capability import (
    BackendDescription,
    CapabilityRequest,
    Phase,
    SupportReport,
    SupportStatus,
    UnsupportedOperation,
)
from usim.facets import ControlTargets, JointLayout, SensorLayout, StateSchema
from usim.model import (
    FormatLoader,
    ImportContext,
    ImportRequirements,
    ImportReport,
    ImportedModel,
    ModelConverter,
    ModelError,
    ModelFormat,
    ModelSource,
)
from usim.model_io import ImportPlan, ModelIO, OwnedModel
from usim.runtime import create_backend, list_backends
from usim.scene import (
    EditArticulation,
    EntityHandle,
    RemoveEntity,
    ReplaceModel,
    SetJointState,
    SetParameter,
    SetPose,
    SpawnEntity,
)
from usim.specs import (
    AgentSpec,
    ControlSpec,
    EntitySpec,
    ExecutionSpec,
    Pose,
    RegisteredAgentRef,
    SceneSpec,
    SensorSpec,
)


__all__ = [
    'MobileRobot',
    'RobotGeometryError',
    'render_robot',
    'Ros2Config',
    'RobotState',
    'SimulationConfig',
    'Simulator',
    'Velocity',
    'Observation',
    'SimulatorConfig',
    'StepResult',
    'create_simulator',
    'list_simulators',
    'register_simulator',
    'EpisodeSimulator',
    'JointCommand',
    'create_runner',
    'list_runners',
    'capabilities',
    'AgentSpec',
    'BackendDescription',
    'CapabilityRequest',
    'ControlSpec',
    'ControlTargets',
    'EditArticulation',
    'EntityHandle',
    'EntitySpec',
    'ExecutionSpec',
    'FormatLoader',
    'ImportContext',
    'ImportPlan',
    'ImportRequirements',
    'ImportReport',
    'ImportedModel',
    'JointLayout',
    'ModelError',
    'ModelFormat',
    'ModelConverter',
    'ModelIO',
    'ModelSource',
    'OwnedModel',
    'Phase',
    'Pose',
    'RegisteredAgentRef',
    'RemoveEntity',
    'ReplaceModel',
    'SceneSpec',
    'SensorLayout',
    'SensorSpec',
    'SetJointState',
    'SetParameter',
    'SetPose',
    'SpawnEntity',
    'StateSchema',
    'SupportReport',
    'SupportStatus',
    'UnsupportedOperation',
    'create_backend',
    'list_backends',
]
