"""Revision-bound entity collections and live capability assessment."""

from typing import TYPE_CHECKING

from usim.capability import (
    CapabilityContext,
    CapabilityRequest,
    SupportReport,
    SupportStatus,
    assess_context,
)
from usim.facets import ControlTargets, StateSchema
from usim.groups import StateGroup, TargetMap
from usim.model import ModelError
from usim.specs import RegisteredAgentRef
from usim.types import Observation
from usim_maniskill.backend import unsupported
from usim_maniskill.facets import Control, state_group

if TYPE_CHECKING:
    from usim_maniskill.session import Session


class States:
    """Expose each entity, with the robot (or first object) as the default view."""

    def __init__(self, session: 'Session'):
        session._require()
        self._session = session
        self._handles = {name: session.scene.handle(name) for name in session._entities}
        self._default = (
            session.spec.agents[0].name
            if session.spec.agents
            else next(iter(self._handles))
            if len(self._handles) == 1
            else None
        )

    def for_entity(self, name: str) -> StateGroup:
        if name not in self._handles:
            raise ModelError(f'Unknown state entity: {name}')
        return state_group(self._session, self._handles[name])

    def _primary(self) -> StateGroup:
        if self._default is None:
            raise ModelError('Select state.for_entity(name) for an empty or multi-entity scene')
        return self.for_entity(self._default)

    @property
    def schema(self) -> StateSchema:
        return self._primary().schema

    def get_observation(self) -> Observation:
        return self._primary().get_observation()

    def get_qpos(self):
        return self._primary().get_qpos()

    def get_qvel(self):
        return self._primary().get_qvel()


class Controls:
    """Expose the agent's controls without assigning controllers to passive objects."""

    def __init__(self, session: 'Session'):
        session._require()
        self._session = session
        self._handles = {
            agent.name: session.scene.handle(agent.name) for agent in session.spec.agents
        }

    def for_entity(self, name: str) -> Control:
        self._session._require()
        if name not in self._handles:
            self._session.scene.handle(name)
            unsupported(f'No controller is bound to scene entity {name}').require()
        self._session.scene.resolve(self._handles[name])
        return Control(self._session, self._handles[name])

    def _primary(self) -> Control:
        if not self._handles:
            unsupported('The scene has no controllable agent').require()
        return self.for_entity(next(iter(self._handles)))

    def apply(self, targets: ControlTargets) -> None:
        self._primary().apply(targets)

    def set_positions(self, targets: TargetMap) -> None:
        self._primary().set_positions(targets)

    def set_velocities(self, targets: TargetMap) -> None:
        self._primary().set_velocities(targets)

    def set_efforts(self, targets: TargetMap) -> None:
        self._primary().set_efforts(targets)

    def set_native_action(self, action) -> None:
        self._primary().set_native_action(action)


def assess_session(session: 'Session', request: CapabilityRequest) -> SupportReport:
    report = assess_context(
        session.backend,
        request,
        CapabilityContext(session.phase, session.options.n_envs, session.options.mode),
    )
    if report.status is SupportStatus.UNSUPPORTED:
        return report
    if request.operation.startswith('control.'):
        if not session.spec.agents:
            return unsupported('The scene has no controllable agent')
        spec = session.spec.agents[0]
        if request.operation != 'control.native_action':
            if isinstance(spec.model, RegisteredAgentRef):
                return unsupported('Registered agents expose native actions, not named targets')
            control = spec.control
            groups = control.position_joints + control.velocity_joints + control.effort_joints
            supported = {
                'control.position': bool(control.position_joints) or not groups,
                'control.velocity': bool(control.velocity_joints),
                'control.effort': bool(control.effort_joints),
            }
            if not supported.get(request.operation, False):
                return unsupported('No joints are bound to that control mode')
    if request.operation == 'state.read' and not session._entities:
        return unsupported('The scene contains no state-bearing entity')
    if request.operation.startswith('sensors.'):
        kind = request.operation.removeprefix('sensors.')
        available = (
            {sensor.kind for sensor in session.state.schema.sensors}
            if session.spec.agents
            else set()
        )
        required = {'rgb', 'depth'} if kind == 'rgbd' else {kind}
        if not required <= available:
            return unsupported('This session has no sensor producing the requested texture')
    return report
