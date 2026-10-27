"""Live scene ownership and explicit safe-point edits for continuous Gazebo."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from usim.capability import (
    CapabilityRequest,
    Phase,
    SupportReport,
    UnsupportedOperation,
)
from usim.model import ImportContext, ImportRequirements, ModelError, ModelSource
from usim.model_io import OwnedModel
from usim.scene import (
    SceneBindings,
    SceneChange,
    SceneEditor,
    SetJointState,
)
from usim.specs import EntitySpec, ExecutionSpec, RegisteredAgentRef, SceneSpec
from usim_gazebo.transport import Json, RequestPort
from usim_gazebo.sensors import SensorChannel, attach
from usim_gazebo.models import GazeboModel

if TYPE_CHECKING:
    from usim_gazebo.contracts import Provider as Backend


from .editing import apply_change, assess_change, assess_live, pose_fields, unsupported


class Session:
    """Mutable session state changes only after a native service acknowledgement."""

    def __init__(
        self,
        backend: Backend,
        spec: SceneSpec,
        execution: ExecutionSpec,
        factory: Callable[[Path], RequestPort],
    ) -> None:
        from usim_gazebo.facets import Agents, Controls, Execution, States

        self.backend = backend
        self.phase = Phase.BUILDING
        self.models = backend.models
        self._resources = ExitStack()
        self._entities: dict[str, EntitySpec] = {}
        self._models: dict[str, OwnedModel[GazeboModel]] = {}
        self._sensor_channels: dict[str, tuple[SensorChannel, ...]] = {}
        self._spawn_xml: dict[str, str] = {}
        self._agent_specs = {agent.name: agent for agent in spec.agents}
        self._services: frozenset[str] = frozenset()
        self._port: RequestPort | None = None
        self.execution = Execution(self)
        self.state = States(self)
        self.control = Controls(self)
        self.agents = Agents(self)
        succeeded = False
        try:
            if execution.mode != 'continuous' or execution.n_envs != 1:
                unsupported('Gazebo requires continuous execution and one world').require()
            if execution.backend != 'cpu' or execution.render_mode not in {'none', 'rgb_array'}:
                unsupported('Grouped Gazebo supports CPU headless rendering').require()
            if execution.obs_mode not in {'state', 'rgb', 'depth', 'rgbd', 'sensor_data'}:
                unsupported('Unknown Gazebo observation mode').require()
            self.directory = Path(
                self._resources.enter_context(TemporaryDirectory(prefix='usim-gazebo-grouped-'))
            )
            entities = list(spec.entities)
            for agent in spec.agents:
                if isinstance(agent.model, RegisteredAgentRef):
                    raise UnsupportedOperation(
                        unsupported(
                            'Load an explicit robot model; no Gazebo agent registry is installed'
                        )
                    )
                if agent.control.position_joints:
                    unsupported(
                        'Joint-position controllers are not provided by Gazebo properties services'
                    ).require()
                entities.append(
                    EntitySpec(agent.name, agent.model, agent.pose, agent.requirements)
                )
            # Validate every asset before any native process starts.
            for entity in entities:
                self._entities[entity.name] = entity
                self._models[entity.name] = self._load(entity.source, entity.requirements)
            for agent in spec.agents:
                xml, channels = attach(self._models[agent.name].value, agent.sensors)
                self._spawn_xml[agent.name] = xml
                self._sensor_channels[agent.name] = channels
                names = {joint.name for joint in self._models[agent.name].value.joints}
                control = agent.control
                requested = set(
                    control.joint_names + control.velocity_joints + control.effort_joints
                )
                if not requested <= names:
                    raise ModelError(f'Unknown controlled joints: {sorted(requested - names)}')
            world = (
                '<sdf version="1.6"><world name="default"><physics name="ode" type="ode">'
                f'<max_step_size>{1 / execution.sim_rate}</max_step_size>'
                f'<real_time_update_rate>{execution.sim_rate}</real_time_update_rate>'
                '</physics></world></sdf>'
            )
            if spec.world is not None:
                world = self._load(spec.world, ImportRequirements(role='scene')).value.xml
            channels = tuple(
                channel for group in self._sensor_channels.values() for channel in group
            )
            requested_kinds = {
                'rgb': {'rgb'},
                'depth': {'depth'},
                'rgbd': {'rgb', 'depth'},
            }.get(execution.obs_mode, set())
            attached_kinds = {channel.layout.kind for channel in channels}
            if not requested_kinds <= attached_kinds:
                unsupported('Observation mode requires matching SensorSpec attachments').require()
            if execution.render_mode == 'rgb_array' and 'rgb' not in attached_kinds:
                unsupported('RGB rendering requires an RGB or RGBD SensorSpec').require()
            self._port = factory(self.directory)
            self._resources.callback(self._port.close)
            self._port.ended.add_done_callback(self._transport_ended)
            result = self._port.request(
                'initialize',
                {
                    'world': world,
                    'rendering': any(item.message_type == 'Image' for item in channels),
                },
            )
            services = result.get('services')
            if not isinstance(services, list) or not all(
                isinstance(item, str) for item in services
            ):
                raise ModelError('Worker did not report its service capabilities')
            self._services = frozenset(item for item in services if isinstance(item, str))
            if channels:
                self._port.request(
                    'subscribe_sensors',
                    {
                        'channels': [
                            {'topic': channel.topic, 'message_type': channel.message_type}
                            for channel in channels
                        ]
                    },
                )
            for entity in entities:
                self._spawn(entity)
            self.scene = SceneEditor(
                tuple(self._entities),
                SceneBindings(
                    self._assess_change,
                    self._apply,
                    lambda: self.phase,
                    self._fail,
                ),
            )
            self.phase = Phase.READY
            for agent in spec.agents:
                if agent.control.initial_qpos is not None:
                    order = agent.control.joint_names or tuple(
                        joint.name for joint in self._models[agent.name].value.joints
                    )
                    if len(order) != len(agent.control.initial_qpos):
                        raise ModelError('Initial joint positions do not match the articulation')
                    change = SetJointState(
                        agent.name,
                        tuple(zip(order, agent.control.initial_qpos)),
                    )
                    self.scene.apply(self.scene.plan(change))
                if agent.control.velocity_joints:
                    self.assess(CapabilityRequest('control.velocity', self.phase)).require()
                if agent.control.effort_joints:
                    self.assess(CapabilityRequest('control.effort', self.phase)).require()
            succeeded = True
        finally:
            if not succeeded:
                self.phase = Phase.FAILED
                self._resources.close()

    def _load(self, source: ModelSource, requirements: ImportRequirements):
        plan = self.models.plan(source, requirements)
        return self._resources.enter_context(
            self.models.load(plan, context=ImportContext(self.directory, requirements))
        )

    def _call(self, service: str, fields: dict[str, Json]) -> dict[str, Json]:
        if self.phase in {Phase.FAILED, Phase.CLOSED} or self._port is None:
            raise ModelError('Gazebo session is not live')
        succeeded = False
        try:
            result = self._port.request('service', {'service': service, 'fields': fields})
            succeeded = True
            return result
        finally:
            if not succeeded:
                self._fail()

    def _spawn(self, spec: EntitySpec) -> None:
        self._call(
            '/spawn_entity',
            {
                'name': spec.name,
                'xml': self._spawn_xml.get(spec.name, self._models[spec.name].value.xml),
                'initial_pose': pose_fields(spec.pose),
                'reference_frame': 'world',
            },
        )

    def assess(self, request: CapabilityRequest) -> SupportReport:
        return assess_live(self, request)

    def _assess_change(self, change: SceneChange, request: CapabilityRequest) -> SupportReport:
        return assess_change(self, change, request)

    def _apply(self, change: SceneChange) -> None:
        apply_change(self, change)

    def _fail(self) -> None:
        if self.phase in {Phase.CLOSED, Phase.FAILED}:
            return
        self.phase = Phase.FAILED
        self.execution._stopped.set()
        try:
            if self._port is not None:
                self._port.close()
        finally:
            self._resources.close()

    def _transport_ended(self, _future: Future[None]) -> None:
        self._fail()

    def close(self) -> None:
        if self.phase is not Phase.CLOSED:
            self.phase = Phase.CLOSED
            self.execution._stopped.set()
            try:
                if self._port is not None:
                    self._port.close()
            finally:
                self._resources.close()

    def __enter__(self) -> Session:
        if self.phase in {Phase.FAILED, Phase.CLOSED}:
            raise ModelError('Gazebo session is not live')
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
