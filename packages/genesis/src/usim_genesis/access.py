"""Revision-scoped state and control facets over real Genesis entities."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from usim.facets import ControlTargets, StateSchema
from usim.groups import ControlBindings, JointControlGroup, StateBindings, StateGroup
from usim.model import ModelError
from usim.scene import EntityHandle
from usim.specs import ControlSpec
from usim.types import Observation

from .native import BoundEntity, numpy, resolve_joint_name

if TYPE_CHECKING:
    from .contracts import SessionView as Session


class EntityState:
    def __init__(self, session: 'Session', handle: EntityHandle):
        self._session = session
        self._handle = handle

    @property
    def _bound(self) -> BoundEntity:
        self._session.require_usable()
        return self._session.entities[self._session.scene.resolve(self._handle)]

    @property
    def schema(self) -> StateSchema:
        return self._bound.schema

    def get_qpos(self) -> NDArray[np.generic]:
        bound = self._bound
        values = numpy(bound.native.get_qpos()).reshape(bound.schema.n_envs, -1)
        for joint in bound.schema.joints:
            start, end = joint.qpos_range
            if end - start in {4, 7}:
                offset = end - 4
                values[:, offset:end] = values[:, [offset + 1, offset + 2, offset + 3, offset]]
        return values

    def get_qvel(self) -> NDArray[np.generic]:
        bound = self._bound
        return numpy(bound.native.get_dofs_velocity()).reshape(bound.schema.n_envs, -1)

    def get_observation(self) -> Observation:
        bound = self._bound
        quaternion = numpy(bound.native.get_quat()).reshape(bound.schema.n_envs, 4)
        observation = Observation(
            qpos=self.get_qpos(),
            qvel=self.get_qvel(),
            extra={
                'entity': bound.spec.name,
                'position': numpy(bound.native.get_pos()).reshape(bound.schema.n_envs, 3),
                'orientation_xyzw': quaternion[:, [1, 2, 3, 0]],
            },
        )
        frames = {
            camera.spec.name: camera.read(
                (int(self._session.steps[0]), self._session.options.sim_rate),
                bound.schema.n_envs,
            )
            for camera in self._session._built.cameras.get(bound.spec.name, ())
        }
        observation.extra['sensors'] = frames
        for frame in frames.values():
            if observation.rgb_image is None:
                observation.rgb_image = frame.rgb
            if observation.depth_image is None:
                observation.depth_image = frame.depth
        lidar_frames = {
            lidar.spec.name: lidar.read(
                (int(self._session.steps[0]), self._session.options.sim_rate),
                bound.schema.n_envs,
            )
            for lidar in self._session._built.lidars.get(bound.spec.name, ())
        }
        observation.extra['sensors'] = {**frames, **lidar_frames}
        return observation


class StateAccess:
    def __init__(self, session: 'Session'):
        self._session = session

    def for_entity(self, name: str) -> StateGroup:
        state = EntityState(self._session, self._session.scene.handle(name))
        return StateGroup(
            state.schema, StateBindings(state.get_observation, state.get_qpos, state.get_qvel)
        )

    @property
    def schema(self) -> StateSchema:
        return self.for_entity(self._session.primary()).schema

    def get_qpos(self) -> NDArray[np.generic]:
        return self.for_entity(self._session.primary()).get_qpos()

    def get_qvel(self) -> NDArray[np.generic]:
        return self.for_entity(self._session.primary()).get_qvel()

    def get_observation(self) -> Observation:
        return self.for_entity(self._session.primary()).get_observation()


class EntityControl(JointControlGroup):
    def __init__(self, session: 'Session', handle: EntityHandle):
        self._session = session
        self._handle = handle
        super().__init__(
            self.spec,
            ControlBindings(
                self._validate,
                lambda values: self._write(ControlTargets(positions=values)),
                lambda values: self._write(ControlTargets(velocities=values)),
                lambda values: self._write(ControlTargets(efforts=values)),
            ),
        )

    @property
    def spec(self) -> ControlSpec:
        name = self._session.scene.resolve(self._handle)
        return self._session.controllers[name]

    @property
    def schema(self) -> StateSchema:
        return EntityState(self._session, self._handle).schema

    def prepare(
        self, targets: ControlTargets
    ) -> tuple[tuple[str, NDArray[np.generic], NDArray[np.generic]], ...]:
        """Resolve and validate every channel before writing any native command."""
        self._session.require_usable()
        name = self._session.scene.resolve(self._handle)
        bound = self._session.entities[name]
        entity = bound.native
        groups = []
        for mode, allowed, values in (
            ('position', self.spec.position_joints, targets.positions),
            ('velocity', self.spec.velocity_joints, targets.velocities),
            ('force', self.spec.effort_joints, targets.efforts),
        ):
            indices, columns = [], []
            for joint_name, value in values.items():
                if joint_name not in allowed:
                    raise ModelError(f'{joint_name!r} is not configured for {mode} control')
                column = np.asarray(value, dtype=float)
                if column.ndim > 1 or (
                    column.ndim == 1 and column.shape != (self._session.options.n_envs,)
                ):
                    raise ModelError('Control target batch must match n_envs')
                indices.append(
                    entity.get_joint(resolve_joint_name(bound, joint_name)).dofs_idx_local[0]
                )
                columns.append(np.broadcast_to(column, (self._session.options.n_envs,)))
            if indices:
                groups.append((mode, np.asarray(indices), np.column_stack(columns)))
        return tuple(groups)

    def _validate(self, targets: ControlTargets) -> None:
        self.prepare(targets)

    def _write(self, targets: ControlTargets) -> None:
        groups = self.prepare(targets)
        entity = self._session.entities[self._session.scene.resolve(self._handle)].native
        with self._session.mutation():
            for mode, indices, values in groups:
                getattr(entity, f'control_dofs_{mode}')(values, indices)


class ControlAccess:
    def __init__(self, session: 'Session'):
        self._session = session

    def for_agent(self, name: str) -> EntityControl:
        if name not in self._session.controllers:
            raise ModelError(f'Entity has no declared controller: {name}')
        return EntityControl(self._session, self._session.scene.handle(name))

    def for_entity(self, name: str) -> EntityControl:
        return self.for_agent(name)

    @property
    def schema(self) -> StateSchema:
        return self.for_agent(self._session.primary()).schema

    @property
    def spec(self) -> ControlSpec:
        return self.for_agent(self._session.primary()).spec

    def apply(self, targets: ControlTargets) -> None:
        self.for_agent(self._session.primary()).apply(targets)

    def set_positions(self, targets: Mapping[str, float | Sequence[float]]) -> None:
        self.apply(ControlTargets(positions=targets))

    def set_velocities(self, targets: Mapping[str, float | Sequence[float]]) -> None:
        self.apply(ControlTargets(velocities=targets))

    def set_efforts(self, targets: Mapping[str, float | Sequence[float]]) -> None:
        self.apply(ControlTargets(efforts=targets))
