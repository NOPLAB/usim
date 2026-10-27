"""Per-entity state and gated controls over acknowledged Gazebo services."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
import threading
from typing import TYPE_CHECKING

from usim.capability import CapabilityRequest, Phase
from usim.facets import ControlTargets, JointLayout, StateSchema
from usim.groups import AgentGroup, ControlBindings, ExecutionGroup, JointControlGroup, TargetMap
from usim.model import ModelError
from usim.specs import ControlSpec, Pose
from usim.types import Observation
from usim_gazebo.transport import Json
from usim_gazebo.sensors import decode

if TYPE_CHECKING:
    from usim_gazebo.contracts import SessionView as Session


def scalar(value: float | Sequence[float]) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if len(value) != 1:
        raise ModelError('Gazebo accepts a scalar or a one-world target batch')
    return float(value[0])


def number(value: Json) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not isfinite(value):
        raise ModelError('Gazebo response requires a finite numeric field')
    return float(value)


def fields(value: Json) -> dict[str, Json]:
    if not isinstance(value, dict):
        raise ModelError('Gazebo response requires a structured message field')
    return value


class EntityState:
    """Revision-bound, unbatched SI state; separate joint calls are not atomic snapshots."""

    def __init__(self, session: Session, name: str) -> None:
        self._session, self._handle = session, session.scene.handle(name)
        joints = session._models[name].value.joints
        agent = session._agent_specs.get(name)
        if agent is not None and agent.control.joint_names:
            by_name = {joint.name: joint for joint in joints}
            joints = tuple(by_name[item] for item in agent.control.joint_names)
        self.schema = StateSchema(
            name,
            tuple(
                JointLayout(
                    joint.name,
                    joint.kind,
                    (index, index + 1),
                    (index, index + 1),
                    'm' if joint.kind == 'prismatic' else 'rad',
                )
                for index, joint in enumerate(joints)
            ),
            tuple(channel.layout for channel in session._sensor_channels.get(name, ())),
            batched=False,
        )

    def _read(self):
        import numpy as np

        session = self._session
        name = session.scene.resolve(self._handle)
        session.assess(CapabilityRequest('state.read', session.phase)).require()
        positions, velocities = [], []
        for joint in self.schema.joints:
            result = session._call(
                '/get_joint_properties', {'joint_name': name + '::' + joint.name}
            )
            position, rate = result['position'], result['rate']
            if (
                not isinstance(position, list)
                or not isinstance(rate, list)
                or len(position) != 1
                or len(rate) != 1
            ):
                raise ModelError('Native joint dimensions differ from the parsed articulation')
            positions.append(number(position[0]))
            velocities.append(number(rate[0]))
        return np.asarray(positions, dtype=float), np.asarray(velocities, dtype=float)

    def get_qpos(self):
        return self._read()[0]

    def get_qvel(self):
        return self._read()[1]

    def get_pose(self) -> Pose:
        session = self._session
        name = session.scene.resolve(self._handle)
        session.assess(CapabilityRequest('state.pose', session.phase)).require()
        response = session._call('/get_entity_state', {'name': name, 'reference_frame': 'world'})
        pose = fields(fields(response['state'])['pose'])
        position, orientation = fields(pose['position']), fields(pose['orientation'])
        return Pose(
            (number(position['x']), number(position['y']), number(position['z'])),
            (
                number(orientation['x']),
                number(orientation['y']),
                number(orientation['z']),
                number(orientation['w']),
            ),
        )

    def get_observation(self) -> Observation:
        positions, velocities = self._read()
        session = self._session
        name = session.scene.resolve(self._handle)
        frames, metadata = {}, {}
        rgb = depth = None
        for channel in session._sensor_channels.get(name, ()):
            if session._port is None:
                raise ModelError('Gazebo worker is closed')
            sample = session._port.request('read_sensor', {'topic': channel.topic})
            frame = decode(channel, sample)
            frames[channel.layout.name] = frame
            metadata[channel.layout.name] = {
                key: value for key, value in sample.items() if key not in {'data', 'ranges'}
            }
            if channel.layout.kind == 'rgb' and rgb is None:
                rgb = frame
            if channel.layout.kind == 'depth' and depth is None:
                depth = frame
        return Observation(
            qpos=positions,
            qvel=velocities,
            rgb_image=rgb,
            depth_image=depth,
            extra={'pose': self.get_pose(), 'sensors': frames, 'sensor_metadata': metadata},
        )


class States:
    def __init__(self, session: Session) -> None:
        self._session = session

    def __getitem__(self, name: str) -> EntityState:
        return EntityState(self._session, name)

    def for_entity(self, name: str) -> EntityState:
        return self[name]

    def _default(self) -> EntityState:
        names = tuple(self._session._agent_specs) or tuple(self._session._entities)
        if len(names) != 1:
            raise ModelError('Select state[entity] explicitly for a multi-entity scene')
        return self[names[0]]

    @property
    def schema(self) -> StateSchema:
        return self._default().schema

    def get_qpos(self):
        return self._default().get_qpos()

    def get_qvel(self):
        return self._default().get_qvel()

    def get_observation(self) -> Observation:
        return self._default().get_observation()


class EntityControl(JointControlGroup):
    """A motor starts disabled; disabling clears native motor and effort commands."""

    def __init__(self, session: Session, name: str) -> None:
        self._session, self._handle = session, session.scene.handle(name)
        self._joints = {joint.name: joint for joint in session._models[name].value.joints}
        self.enabled = False
        agent = session._agent_specs.get(name)
        spec = agent.control if agent is not None else ControlSpec(joint_names=tuple(self._joints))
        super().__init__(
            spec,
            ControlBindings(
                self._validate,
                velocities=self._velocities,
                efforts=self._efforts,
            ),
        )
        self._active_velocity: set[str] = set()
        self._active_effort: set[str] = set()

    def _validate(self, targets: ControlTargets) -> None:
        session = self._session
        session.scene.resolve(self._handle)
        if not self.enabled:
            raise ModelError('Motor is disabled; explicitly enable it before applying controls')
        for values, mode in ((targets.velocities, 'velocity'), (targets.efforts, 'effort')):
            if values:
                session.assess(CapabilityRequest('control.' + mode, session.phase)).require()
            for name, value in values.items():
                if name not in self._joints:
                    raise ModelError(f'Unknown controlled joint: {name}')
                scalar(value)
                if self._joints[name].effort <= 0:
                    raise ModelError(f'Joint {name} requires an authored positive effort limit')
                if mode == 'effort' and abs(scalar(value)) > self._joints[name].effort:
                    raise ModelError(f'Joint {name} effort exceeds its authored limit')

    def _velocities(self, values: TargetMap) -> None:
        name = self._session.scene.resolve(self._handle)
        succeeded = False
        try:
            for joint, value in values.items():
                self._session._call(
                    '/set_joint_properties',
                    {
                        'joint_name': name + '::' + joint,
                        'ode_joint_config': {
                            'vel': [scalar(value)],
                            'fmax': [self._joints[joint].effort],
                        },
                    },
                )
                self._active_velocity.add(joint)
            succeeded = True
        finally:
            if not succeeded:
                self._session._fail()

    def _efforts(self, values: TargetMap) -> None:
        name = self._session.scene.resolve(self._handle)
        succeeded = False
        try:
            for joint, value in values.items():
                scoped = name + '::' + joint
                self._session._call('/clear_joint_forces', {'joint_name': scoped})
                self._session._call(
                    '/apply_joint_effort',
                    {
                        'joint_name': scoped,
                        'effort': scalar(value),
                        'duration': {'sec': -1, 'nanosec': 0},
                    },
                )
                self._active_effort.add(joint)
            succeeded = True
        finally:
            if not succeeded:
                self._session._fail()

    def set_motor(self, enabled: bool) -> None:
        name = self._session.scene.resolve(self._handle)
        self.enabled = False
        succeeded = False
        try:
            for joint in self._active_velocity:
                self._session._call(
                    '/set_joint_properties',
                    {
                        'joint_name': name + '::' + joint,
                        'ode_joint_config': {'vel': [0.0], 'fmax': [0.0]},
                    },
                )
            for joint in self._active_effort:
                self._session._call('/clear_joint_forces', {'joint_name': name + '::' + joint})
            self._active_velocity.clear()
            self._active_effort.clear()
            self.enabled = enabled
            succeeded = True
        finally:
            if not succeeded:
                self._session._fail()


class Controls:
    def __init__(self, session: Session) -> None:
        self._session = session
        self._controls: dict[str, EntityControl] = {}

    def __getitem__(self, name: str) -> EntityControl:
        if name not in self._controls:
            self._controls[name] = EntityControl(self._session, name)
        return self._controls[name]

    def for_entity(self, name: str) -> EntityControl:
        return self[name]

    def discard(self, name: str) -> None:
        if name in self._controls:
            self._controls.pop(name).set_motor(False)

    def _default(self) -> EntityControl:
        names = tuple(self._session._agent_specs) or tuple(self._session._entities)
        if len(names) != 1:
            raise ModelError('Select control[entity] explicitly for a multi-entity scene')
        return self[names[0]]

    def apply(self, targets: ControlTargets) -> None:
        self._default().apply(targets)

    def set_motor(self, enabled: bool) -> None:
        self._default().set_motor(enabled)

    def set_positions(self, targets: TargetMap) -> None:
        self._default().set_positions(targets)

    def set_velocities(self, targets: TargetMap) -> None:
        self._default().set_velocities(targets)

    def set_efforts(self, targets: TargetMap) -> None:
        self._default().set_efforts(targets)

    def disable(self) -> None:
        for control in self._controls.values():
            control.set_motor(False)

    def invalidate(self) -> None:
        self.disable()
        self._controls.clear()


class Agents:
    def __init__(self, session: Session) -> None:
        self._session = session

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._session._agent_specs)

    def describe(self, name: str):
        return AgentGroup(tuple(self._session._agent_specs.values())).describe(name)


class Execution(ExecutionGroup):
    """Continuous execution blocks on an event, never on fabricated episode steps."""

    def __init__(self, session: Session) -> None:
        super().__init__(continuous=self)
        self._session = session
        self._stopped = threading.Event()

    def resume(self) -> None:
        session = self._session
        session.assess(CapabilityRequest('execution.resume', session.phase)).require()
        session._call('/unpause_physics', {})
        session.phase = Phase.RUNNING
        self._stopped.clear()

    def pause(self) -> None:
        session = self._session
        session.assess(CapabilityRequest('execution.pause', session.phase)).require()
        session.control.disable()
        session._call('/pause_physics', {})
        session.phase = Phase.PAUSED

    def run(self) -> None:
        self.resume()
        self._stopped.wait()
        if self._session.phase is Phase.FAILED:
            raise ModelError('Gazebo worker terminated during continuous execution')

    def stop(self) -> None:
        if self._session.phase not in {Phase.FAILED, Phase.CLOSED}:
            self.pause()
        self._stopped.set()
