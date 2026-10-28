"""Grouped CPU Bullet provider; native imports occur only when opening a session."""

from usim.capability import (
    BackendDescription,
    CapabilityRequest,
    Phase,
    SupportReport,
    SupportStatus,
)
from usim.model import ModelError, ModelSource
from usim.specs import ExecutionSpec, SceneSpec
from .models import model_io, unsupported

OPERATIONS = frozenset(
    {
        'models.load',
        'state.read',
        'control.position',
        'control.velocity',
        'control.effort',
        'agents.describe',
        'execution.episode',
        'scene.set_pose',
        'scene.set_joint_state',
    }
)


class Backend:
    def __init__(self):
        self.models = model_io()

    def describe(self) -> BackendDescription:
        return BackendDescription(
            'bullet',
            OPERATIONS,
            frozenset(format.value for format in self.models.formats),
            frozenset({'episode'}),
            registration_required=False,
        )

    def assess(self, request: CapabilityRequest) -> SupportReport:
        if request.phase in {Phase.CLOSED, Phase.FAILED}:
            return unsupported('Bullet session has no live resources')
        if request.n_envs != 1:
            return unsupported('Bullet supports one environment per session')
        if request.operation not in OPERATIONS:
            return unsupported(f'Bullet does not implement {request.operation}')
        if request.phase is Phase.RUNNING and request.operation != 'state.read':
            return unsupported('Bullet writes require an episode safe point')
        return SupportReport(SupportStatus.SUPPORTED, scope='all_envs')

    def open(self, scene: SceneSpec, execution: ExecutionSpec = ExecutionSpec()):
        if execution.mode != 'episode' or execution.backend not in {'cpu', 'auto'}:
            unsupported('Bullet supplies CPU episode execution only').require()
        if execution.n_envs != 1:
            unsupported('Bullet supports one environment per session').require()
        if execution.render_mode not in {'none', 'human'}:
            unsupported(
                'Bullet supports headless or GUI execution, not camera observations'
            ).require()
        if execution.obs_mode not in {'state', 'none', 'state_dict'}:
            unsupported('Bullet supplies joint and base state observations only').require()
        if execution.native_task != 'JointControl' or scene.world is not None:
            unsupported(
                'Bullet accepts explicit model entities, not registered tasks or worlds'
            ).require()
        for agent in scene.agents:
            if not isinstance(agent.model, ModelSource) or agent.sensors:
                unsupported('Bullet requires direct models without requested sensors').require()
            if agent.requirements.role != 'articulation':
                raise ModelError('Bullet agents must be articulations')
            if agent.control.native_mode != 'pd_joint_pos':
                unsupported('Bullet binds explicit physical control groups only').require()
            if (
                agent.control.joint_stiffness is not None
                or agent.control.joint_damping is not None
            ):
                unsupported('Bullet native gains are not physical stiffness or damping').require()
        from .session import Session

        return Session(self, scene, execution)
