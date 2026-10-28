"""State and physical control collections for one owned MuJoCo session."""

from math import isfinite

from usim.facets import ControlTargets
from usim.model import ModelError
from usim.types import Observation


class State:
    def __init__(self, session):
        self.session = session

    @property
    def schema(self):
        self.session._require()
        return self.session.schema

    def for_entity(self, name):
        self.session._entity(name)
        return self

    def get_qpos(self):
        self.session._require()
        return self.session.data.qpos.copy()

    def get_qvel(self):
        self.session._require()
        return self.session.data.qvel.copy()

    def get_observation(self):
        return Observation(
            qpos=self.get_qpos(),
            qvel=self.get_qvel(),
            extra={'time': float(self.session.data.time)},
        )


class Control:
    def __init__(self, session):
        self.session = session

    def for_entity(self, name):
        self.session._entity(name)
        return self

    def apply(self, targets: ControlTargets):
        s = self.session
        s._require()
        writes = (
            ('position', targets.positions, s.control_spec.position_joints),
            ('velocity', targets.velocities, s.control_spec.velocity_joints),
            ('effort', targets.efforts, s.control_spec.effort_joints),
        )
        partitioned = any(
            (
                s.control_spec.position_joints,
                s.control_spec.velocity_joints,
                s.control_spec.effort_joints,
            )
        )
        pending = {}
        for mode, values, selected in writes:
            for name, value in values.items():
                available = selected if partitioned else s.control_names
                if name not in available or name not in s.scalar:
                    raise ModelError(f'Unknown or incompatible {mode} joint: {name}')
                if not isinstance(value, (float, int)):
                    if len(value) != 1:
                        raise ModelError('A single environment requires one target value')
                    value = value[0]
                if not isfinite(value):
                    raise ModelError('Joint targets must be finite')
                pending[name] = (mode, float(value))
        s.targets.update(pending)

    def set_positions(self, targets):
        self.apply(ControlTargets(positions=targets))

    def set_velocities(self, targets):
        self.apply(ControlTargets(velocities=targets))

    def set_efforts(self, targets):
        self.apply(ControlTargets(efforts=targets))
