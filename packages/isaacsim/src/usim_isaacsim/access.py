"""Named, revision-scoped state and control views with no physics stepping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from usim.facets import ControlTargets, JointLayout, StateSchema
from usim.model import ModelError
from usim.scene import EntityHandle
from usim.specs import AgentSpec, Pose
from usim.types import Observation

if TYPE_CHECKING:
    from usim_isaacsim.contracts import SessionView as Session
    from usim_isaacsim.scene_native import NativeEntity


@dataclass(frozen=True, slots=True)
class State:
    session: Session
    handle: EntityHandle

    def _entry(self) -> NativeEntity:
        self.session.require_ready()
        return self.session._native.entities[self.session.scene.resolve(self.handle)]

    @property
    def schema(self) -> StateSchema:
        entry = self._entry()
        kinds = dict(entry.imported.value.joints)
        return StateSchema(
            entry.spec.name,
            tuple(
                JointLayout(
                    name,
                    kinds[name],
                    (index, index + 1),
                    (index, index + 1),
                    'm' if kinds[name] == 'prismatic' else 'rad',
                )
                for index, name in enumerate(entry.joint_names)
            ),
            self.session._sensors.layouts(entry.spec.name),
            batched=False,
        )

    def get_qpos(self) -> NDArray[np.generic]:
        entry = self._entry()
        if not entry.joint_names:
            return np.empty(0)
        return np.asarray(
            entry.native.get_joint_positions(joint_indices=np.asarray(entry.joint_indices))
        ).copy()

    def get_qvel(self) -> NDArray[np.generic]:
        entry = self._entry()
        if not entry.joint_names:
            return np.empty(0)
        return np.asarray(
            entry.native.get_joint_velocities(joint_indices=np.asarray(entry.joint_indices))
        ).copy()

    def get_observation(self) -> Observation:
        entry = self._entry()
        sensors = self.session._sensors.observe(entry.spec.name)
        return Observation(
            sensors.rgb,
            sensors.depth,
            self.get_qpos(),
            self.get_qvel(),
            {'sensors': sensors.frames, 'sensor_timestamps': sensors.timestamps},
        )

    def get_pose(self) -> Pose:
        position, orientation = self._entry().native.get_world_pose()
        return Pose(
            (float(position[0]), float(position[1]), float(position[2])),
            (
                float(orientation[1]),
                float(orientation[2]),
                float(orientation[3]),
                float(orientation[0]),
            ),
        )


@dataclass(frozen=True, slots=True)
class Control:
    session: Session
    handle: EntityHandle

    def apply(self, targets: ControlTargets) -> None:
        entry = State(self.session, self.handle)._entry()
        from isaacsim.core.utils.types import ArticulationAction

        names = entry.joint_names
        if not names:
            raise ModelError('Named controls require an articulation')
        selected = set(names)
        velocities = set(entry.control.velocity_joints)
        efforts = set(entry.control.effort_joints)
        positions = set(entry.control.position_joints) or (selected - velocities - efforts)
        actions = []
        for values, allowed, field in (
            (targets.positions, positions, 'joint_positions'),
            (targets.velocities, velocities, 'joint_velocities'),
            (targets.efforts, efforts, 'joint_efforts'),
        ):
            if not set(values) <= allowed:
                raise ModelError(f'{field} contains unknown or differently controlled joints')
            if not values:
                continue
            raw = [np.asarray(value, dtype=float) for value in values.values()]
            if any(value.size != 1 for value in raw):
                raise ModelError('Isaac single-environment control requires one target per joint')
            vector = np.asarray([value.item() for value in raw])
            indices = np.asarray([entry.joint_indices[names.index(name)] for name in values])
            if field == 'joint_positions':
                properties = entry.native.dof_properties
                if np.any(
                    (vector < properties['lower'][indices])
                    | (vector > properties['upper'][indices])
                ):
                    raise ModelError('Position targets exceed articulation limits')
            actions.append(ArticulationAction(**{field: vector}, joint_indices=indices))
        # Prepare every mode before the first mutation, including finite/batch checks.
        for action in actions:
            entry.native.apply_action(action)

    def set_positions(self, targets: Mapping[str, float | Sequence[float]]) -> None:
        self.apply(ControlTargets(positions=targets))

    def set_velocities(self, targets: Mapping[str, float | Sequence[float]]) -> None:
        self.apply(ControlTargets(velocities=targets))

    def set_efforts(self, targets: Mapping[str, float | Sequence[float]]) -> None:
        self.apply(ControlTargets(efforts=targets))


@dataclass(frozen=True, slots=True)
class States:
    session: Session

    def for_entity(self, name: str) -> State:
        self.session.require_ready()
        return State(self.session, self.session.scene.handle(name))

    @property
    def schema(self) -> StateSchema:
        return self.for_entity(self.session.primary_entity()).schema

    def get_qpos(self) -> NDArray[np.generic]:
        return self.for_entity(self.session.primary_entity()).get_qpos()

    def get_qvel(self) -> NDArray[np.generic]:
        return self.for_entity(self.session.primary_entity()).get_qvel()

    def get_observation(self) -> Observation:
        self.session.require_ready()
        if (
            not any(entry.joint_names for entry in self.session._native.entities.values())
            and len(self.session._specs) != 1
        ):
            return Observation(qpos=np.empty(0), qvel=np.empty(0))
        return self.for_entity(self.session.primary_entity()).get_observation()


@dataclass(frozen=True, slots=True)
class Controls:
    session: Session

    def for_entity(self, name: str) -> Control:
        self.session.require_ready()
        return Control(self.session, self.session.scene.handle(name))

    def apply(self, targets: ControlTargets) -> None:
        self.for_entity(self.session.primary_entity()).apply(targets)

    def set_positions(self, targets: Mapping[str, float | Sequence[float]]) -> None:
        self.apply(ControlTargets(positions=targets))

    def set_velocities(self, targets: Mapping[str, float | Sequence[float]]) -> None:
        self.apply(ControlTargets(velocities=targets))

    def set_efforts(self, targets: Mapping[str, float | Sequence[float]]) -> None:
        self.apply(ControlTargets(efforts=targets))


@dataclass(frozen=True, slots=True)
class Agents:
    session: Session

    @property
    def names(self) -> tuple[str, ...]:
        self.session.require_ready()
        return tuple(self.session._agents)

    def describe(self, name: str) -> AgentSpec:
        self.session.require_ready()
        if name not in self.session._agents:
            raise ModelError(f'Unknown agent: {name}')
        return self.session._agents[name]
