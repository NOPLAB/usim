"""Preflighted native edits and explicit new-episode topology changes."""

from contextlib import ExitStack
from dataclasses import replace
from math import isfinite
from typing import TYPE_CHECKING
from xml.etree.ElementTree import ParseError

import numpy as np

from usim.capability import CapabilityRequest, SupportReport, SupportStatus, assert_never
from usim.model import ModelError
from usim.scene import (
    EditArticulation,
    RemoveEntity,
    ReplaceModel,
    SceneChange,
    SetJointState,
    SetParameter,
    SetPose,
    SpawnEntity,
)
from usim.specs import SceneSpec

from .construction import prepare
from .native import resolve_joint_name

if TYPE_CHECKING:
    from .contracts import SessionView as Session


def changed_scene(spec: SceneSpec, change: SceneChange) -> SceneSpec:
    match change:
        case SpawnEntity(spec=entity):
            return replace(spec, entities=spec.entities + (entity,))
        case RemoveEntity(entity=name):
            return replace(
                spec,
                entities=tuple(entity for entity in spec.entities if entity.name != name),
                agents=tuple(agent for agent in spec.agents if agent.name != name),
                world=None if name == '__world__' else spec.world,
            )
        case (
            ReplaceModel(entity=name, source=source) | EditArticulation(entity=name, source=source)
        ):
            return replace(
                spec,
                entities=tuple(
                    replace(entity, source=source) if entity.name == name else entity
                    for entity in spec.entities
                ),
                agents=tuple(
                    replace(agent, model=source) if agent.name == name else agent
                    for agent in spec.agents
                ),
                world=source if name == '__world__' else spec.world,
            )
        case SetPose() | SetJointState() | SetParameter():
            raise ModelError('A native state edit does not replace the scene specification')
        case unreachable:
            assert_never(unreachable)


def joint_values(session: 'Session', change: SetJointState):
    bound = session.entities[change.entity]
    entity = bound.native
    result = []
    for pairs in (change.positions, change.velocities):
        names = [name for name, _ in pairs]
        if len(set(names)) != len(names):
            raise ModelError('Joint state edits require unique names in each channel')
        indices = []
        for name, value in pairs:
            joint = entity.get_joint(resolve_joint_name(bound, name))
            if joint is None or len(joint.dofs_idx_local) != 1 or joint.n_qs != 1:
                raise ModelError(f'Joint state edit requires a scalar joint: {name}')
            if not isfinite(value):
                raise ModelError('Joint state edits require finite values')
            indices.append(joint.dofs_idx_local[0])
        result.append(
            (
                np.asarray(indices, dtype=int),
                np.tile([value for _, value in pairs], (session.options.n_envs, 1)),
            )
        )
    return tuple(result)


class Edits:
    def __init__(self, session: 'Session'):
        self.session = session

    def assess(self, change: SceneChange, request: CapabilityRequest) -> SupportReport:
        support = self.session.backend.assess(request)
        if not support.supported:
            return support
        try:
            match change:
                case SetPose():
                    pass
                case SetJointState():
                    joint_values(self.session, change)
                case SetParameter(parameter=parameter, value=value, entity=name):
                    if not isfinite(value):
                        raise ModelError('Parameters must be finite')
                    if parameter not in {'friction', 'mass', 'kp', 'kv'}:
                        raise ModelError(f'Unsupported Genesis parameter: {parameter}')
                    if value < 0 or (parameter == 'mass' and value == 0):
                        raise ModelError('Parameter is outside its physical range')
                    if parameter == 'friction' and not 0.01 <= value <= 5:
                        raise ModelError('Genesis friction must be in [0.01, 5]')
                    entity = self.session.entities[name].native
                    if parameter == 'mass' and not any(not link.is_fixed for link in entity.links):
                        raise ModelError('A completely fixed entity has no movable mass')
                    if parameter in {'kp', 'kv'} and not entity.n_dofs:
                        raise ModelError('Entity has no controllable DOFs')
                case SpawnEntity() | RemoveEntity() | ReplaceModel() | EditArticulation():
                    with ExitStack() as resources:
                        prepare(
                            changed_scene(self.session.spec, change),
                            self.session.models,
                            resources,
                        )
                case unreachable:
                    assert_never(unreachable)
        except (ModelError, OSError, ParseError, ImportError) as error:
            return SupportReport(SupportStatus.UNSUPPORTED, (str(error),))
        return support

    def apply(self, change: SceneChange) -> None:
        match change:
            case SetPose(entity=name, pose=pose):
                entity = self.session.entities[name].native
                ids = np.arange(self.session.options.n_envs)
                x, y, z, w = pose.orientation_xyzw
                entity.set_pos(np.tile(pose.position, (len(ids), 1)), envs_idx=ids)
                entity.set_quat(np.tile((w, x, y, z), (len(ids), 1)), envs_idx=ids)
            case SetJointState(entity=name):
                groups = joint_values(self.session, change)
                entity = self.session.entities[name].native
                ids = np.arange(self.session.options.n_envs)
                for method, (indices, values) in zip(
                    ('set_dofs_position', 'set_dofs_velocity'), groups
                ):
                    if len(indices):
                        getattr(entity, method)(values, indices, envs_idx=ids)
            case SetParameter(entity=name, parameter=parameter, value=value):
                entity = self.session.entities[name].native
                match parameter:
                    case 'friction':
                        entity.set_friction(value)
                    case 'mass':
                        entity.set_mass(value)
                    case 'kp' | 'kv':
                        getattr(entity, f'set_dofs_{parameter}')(
                            np.full(entity.n_dofs, value), np.arange(entity.n_dofs)
                        )
                    case _:
                        raise ModelError(f'Unsupported Genesis parameter: {parameter}')
            case SpawnEntity() | RemoveEntity() | ReplaceModel() | EditArticulation():
                self.session.rebuild(changed_scene(self.session.spec, change))
            case unreachable:
                assert_never(unreachable)
        self.session.invalidate_sensors()
