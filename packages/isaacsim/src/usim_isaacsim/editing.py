"""Scene-edit assessment and application at explicit Isaac safe points."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

from usim.capability import CapabilityRequest, SupportReport, SupportStatus
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

if TYPE_CHECKING:
    from usim_isaacsim.contracts import SessionView as Session


class Edits:
    def __init__(self, session: Session) -> None:
        self.session = session

    def assess(self, change: SceneChange, request: CapabilityRequest) -> SupportReport:
        report = self.session.backend.assess(request)
        if not report.supported:
            return report
        entities = self.session._native.entities
        try:
            match change:
                case SetPose(entity=name):
                    if entities[name].spec.requirements.role == 'scene':
                        raise ModelError('A scene may contain dynamic children; replace its asset')
                case SetJointState(entity=name, positions=positions, velocities=velocities):
                    entry = entities[name]
                    if not entry.joint_names:
                        raise ModelError('Joint-state changes require an articulation')
                    for values in (positions, velocities):
                        names = [joint for joint, _ in values]
                        if len(names) != len(set(names)) or not set(names) <= set(
                            entry.joint_names
                        ):
                            raise ModelError(
                                'Joint-state edit contains duplicate or unknown joints'
                            )
                        if not all(isfinite(value) for _, value in values):
                            raise ModelError('Joint-state values must be finite')
                    properties = entry.native.dof_properties
                    for joint, value in positions:
                        index = entry.joint_indices[entry.joint_names.index(joint)]
                        if not properties['lower'][index] <= value <= properties['upper'][index]:
                            raise ModelError('Joint-state position exceeds native limits')
                case SetParameter(entity=name, parameter=parameter, value=value):
                    if (
                        parameter != 'mass'
                        or entities[name].spec.requirements.role != 'rigid_body'
                        or not isfinite(value)
                        or value <= 0
                    ):
                        raise ModelError('Only finite positive rigid-body mass is editable')
                case SpawnEntity(spec=spec):
                    if spec.name == '_usim_world':
                        raise ModelError('_usim_world is reserved for the scene world')
                    if spec.requirements.role == 'articulation' and any(
                        entry.spec.requirements.role == 'articulation'
                        for entry in entities.values()
                    ):
                        raise ModelError(
                            'Only one managed articulation per shared world is supported'
                        )
                    self.session.models.plan(spec.source, spec.requirements)
                case (
                    ReplaceModel(entity=name, source=source)
                    | EditArticulation(entity=name, source=source)
                ):
                    self.session.models.plan(source, entities[name].spec.requirements)
                case RemoveEntity():
                    pass
        except ModelError as error:
            return SupportReport(SupportStatus.UNSUPPORTED, (str(error),))
        return report

    def apply(self, change: SceneChange) -> None:
        native = self.session._native
        self.session._sensors.invalidate()
        match change:
            case SetPose(entity=name, pose=pose):
                native.pose(name, pose)
                self.session._specs[name] = replace(self.session._specs[name], pose=pose)
                if name in self.session._agents:
                    self.session._agents[name] = replace(self.session._agents[name], pose=pose)
            case SetJointState(entity=name, positions=positions, velocities=velocities):
                entry = native.entities[name]
                for values, setter in (
                    (positions, entry.native.set_joint_positions),
                    (velocities, entry.native.set_joint_velocities),
                ):
                    if values:
                        indices = np.asarray(
                            [
                                entry.joint_indices[entry.joint_names.index(joint)]
                                for joint, _ in values
                            ]
                        )
                        setter(np.asarray([value for _, value in values]), joint_indices=indices)
            case SetParameter(entity=name, value=value):
                native.entities[name].native.set_mass(value)
            case SpawnEntity(spec=spec):
                self.session._specs[spec.name] = spec
                self.session.rebuild()
            case RemoveEntity(entity=name):
                del self.session._specs[name]
                self.session._agents.pop(name, None)
                self.session.rebuild()
            case (
                ReplaceModel(entity=name, source=source)
                | EditArticulation(entity=name, source=source)
            ):
                self.session._specs[name] = replace(self.session._specs[name], source=source)
                if name in self.session._agents:
                    self.session._agents[name] = replace(self.session._agents[name], model=source)
                self.session.rebuild()
