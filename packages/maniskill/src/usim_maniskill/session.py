"""Native session ownership, safe-point changes, and topology reconstruction."""

from contextlib import ExitStack
from math import isfinite
from typing import TYPE_CHECKING

from usim.capability import (
    CapabilityRequest,
    Phase,
    SupportReport,
    SupportStatus,
    UnsupportedOperation,
    assert_never,
)
from usim.groups import AgentGroup, ExecutionGroup
from usim.model import ImportRequirements, ModelError, ModelSource
from usim.scene import (
    EditArticulation,
    RemoveEntity,
    ReplaceModel,
    SceneBindings,
    SceneChange,
    SceneEditor,
    SetJointState,
    SetParameter,
    SetPose,
    SpawnEntity,
)
from usim.specs import ExecutionSpec, RegisteredAgentRef, SceneSpec
from usim_maniskill.backend import Backend, TOPOLOGY, changed_spec, unsupported, validate_scene
from usim_maniskill.models import inspect, validate_agent
from usim_maniskill.native import create_environment, native_pose

if TYPE_CHECKING:
    from usim_maniskill.models import Asset


def _sources(spec: SceneSpec) -> tuple[tuple[str, ModelSource, ImportRequirements], ...]:
    entries = [(entity.name, entity.source, entity.requirements) for entity in spec.entities]
    for agent in spec.agents:
        match agent.model:
            case ModelSource() as source:
                entries.append((agent.name, source, agent.requirements))
            case RegisteredAgentRef():
                pass
    return tuple(entries)


class Session:
    """Own one mutable native world and all import artifacts until explicit close."""

    def __init__(self, backend: Backend, spec: SceneSpec, execution: ExecutionSpec):
        from usim_maniskill.execution import Episode

        self.backend = backend
        self.spec = spec
        self.options = execution
        self.models = backend.models
        self._resources = ExitStack()
        self._env = None
        self._phase = Phase.BUILDING
        self._entities = {}
        self._assets: dict[str, Asset] = {}
        self._targets = {}
        self.scene = SceneEditor(
            tuple(entity.name for entity in spec.entities + spec.agents),
            SceneBindings(
                self._assess_change, self._apply, lambda: self.phase, self._fail, execution.n_envs
            ),
        )
        self.agents = AgentGroup(spec.agents)
        self.execution = ExecutionGroup(episode=Episode(self))
        succeeded = False
        try:
            self._build()
            succeeded = True
        finally:
            if not succeeded:
                self._fail()

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def state(self):
        from usim_maniskill.collections import States

        return States(self)

    def state_for(self, name: str):
        return self.state.for_entity(name)

    @property
    def control(self):
        from usim_maniskill.collections import Controls

        return Controls(self)

    def assess(self, request: CapabilityRequest) -> SupportReport:
        from usim_maniskill.collections import assess_session

        return assess_session(self, request)

    def _require(self):
        if self._env is None or self.phase in {Phase.CLOSED, Phase.FAILED}:
            raise ModelError(f'ManiSkill session is {self.phase.value}')
        return self._env

    def _build(self) -> None:
        from usim_maniskill.facets import numpy_copy

        self._assets = {}
        for name, source, requirements in _sources(self.spec):
            plan = self.models.plan(source, requirements)
            owned = self._resources.enter_context(self.models.load(plan))
            self._assets[name] = owned.value
        for agent in self.spec.agents:
            if agent.name not in self._assets:
                continue
            validate_agent(agent, self._assets[agent.name])
        self._env = create_environment(self.spec, self.options, self._assets)
        self._resources.callback(self._env.close)
        native = self._env.unwrapped
        self._entities = dict(getattr(native, 'usim_entities', {}))
        if self.spec.agents:
            spec = self.spec.agents[0]
            robot = native.agent.robot
            self._entities[spec.name] = robot
            names = tuple(joint.name for joint in robot.get_active_joints())
            if spec.control.joint_names and spec.control.joint_names != names:
                raise ModelError(f'joint_names must match native state order: {names}')
            if spec.control.initial_qpos is not None and len(spec.control.initial_qpos) != len(
                names
            ):
                raise ModelError('Initial positions must match native joint count')
            if spec.requirements.base != 'authored':
                fixed = spec.requirements.base == 'fixed'
                if not (numpy_copy(robot.fixed_root_link) == fixed).all():
                    raise ModelError('Requested base mode differs from the native agent')
        self.agents = AgentGroup(self.spec.agents)
        self._phase = Phase.READY
        self.execution.reset()

    def _sync(self) -> None:
        native = self._require().unwrapped
        if native.gpu_sim_enabled:
            native.scene._gpu_apply_all()
            native.scene.px.gpu_update_articulation_kinematics()
            native.scene._gpu_fetch_all()

    def _assess_change(self, change: SceneChange, request: CapabilityRequest) -> SupportReport:
        report = self.assess(request)
        if report.status is SupportStatus.UNSUPPORTED:
            return report
        if request.operation in TOPOLOGY:
            replacement = changed_spec(self.spec, change)
            try:
                validate_scene(replacement, self.options)
                for _, source, requirements in _sources(replacement):
                    self.models.plan(source, requirements)
                for agent in replacement.agents:
                    if isinstance(agent.model, ModelSource):
                        validate_agent(agent, inspect(agent.model, agent.requirements))
            except (ModelError, ValueError, UnsupportedOperation) as error:
                return unsupported(str(error))
            return report
        match change:
            case SetPose(entity=name) | SetJointState(entity=name) | SetParameter(entity=name):
                native = self._entities[name]
            case SpawnEntity() | RemoveEntity() | ReplaceModel() | EditArticulation():
                return unsupported('Topology changes require the matching operation request')
        match change:
            case SetParameter(parameter=parameter, value=value):
                if self._require().unwrapped.gpu_sim_enabled:
                    return unsupported('Joint friction edits are supported only on CPU')
                names = (
                    tuple(joint.name for joint in native.get_active_joints())
                    if hasattr(native, 'get_active_joints')
                    else ()
                )
                if parameter not in {f'joint.{name}.friction' for name in names}:
                    return unsupported('Only joint.<name>.friction is an editable parameter')
                if not isfinite(value) or value < 0:
                    return unsupported('Joint friction must be finite and nonnegative')
                return SupportReport(SupportStatus.SUPPORTED)
            case SetJointState(positions=positions, velocities=velocities):
                if not hasattr(native, 'get_active_joints'):
                    return unsupported('Joint state edits require an articulation')
                names = {joint.name for joint in native.get_active_joints()}
                for entries in (positions, velocities):
                    if len({name for name, _ in entries}) != len(entries):
                        return unsupported('A joint state edit cannot repeat a joint')
                    if any(name not in names or not isfinite(value) for name, value in entries):
                        return unsupported(
                            'Joint state edits require known joints and finite values'
                        )
            case SetPose():
                return report
            case SpawnEntity() | RemoveEntity() | ReplaceModel() | EditArticulation():
                raise ModelError('A topology change requires its corresponding capability request')
            case unreachable:
                assert_never(unreachable)
        return report

    def _apply(self, change: SceneChange) -> None:
        from usim_maniskill.facets import numpy_copy

        match change:
            case SpawnEntity() | RemoveEntity() | ReplaceModel() | EditArticulation():
                self.spec = changed_spec(self.spec, change)
                self._resources.close()
                self._resources = ExitStack()
                self._env = None
                self._phase = Phase.BUILDING
                self._build()
            case SetPose(entity=name, pose=pose):
                self._entities[name].set_pose(native_pose(pose))
                self._sync()
            case SetJointState(entity=name, positions=positions, velocities=velocities):
                entity = self._entities[name]
                names = [joint.name for joint in entity.get_active_joints()]
                for entries, get, put in (
                    (positions, entity.get_qpos, entity.set_qpos),
                    (velocities, entity.get_qvel, entity.set_qvel),
                ):
                    if entries:
                        values = numpy_copy(get())
                        for joint, value in entries:
                            values[:, names.index(joint)] = value
                        put(values)
                self._sync()
                if self.spec.agents and name == self.spec.agents[0].name:
                    self._require().unwrapped.agent.controller.reset()
                    self._targets.clear()
            case SetParameter(entity=name, parameter=parameter, value=value):
                for joint in self._entities[name].get_active_joints():
                    if parameter == f'joint.{joint.name}.friction':
                        joint.set_friction(value)
            case unreachable:
                assert_never(unreachable)

    def _fail(self) -> None:
        try:
            self.close()
        finally:
            self._phase = Phase.FAILED

    def close(self) -> None:
        self._env = None
        self._phase = Phase.CLOSED
        self._targets.clear()
        self._entities.clear()
        self._assets.clear()
        self._resources.close()

    def __enter__(self):
        self._require()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
