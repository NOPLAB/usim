"""Metadata-only discovery and explicit native session construction."""

from usim.capability import (
    BackendDescription,
    CapabilityRequest,
    Phase,
    SupportReport,
    SupportStatus,
    UnsupportedOperation,
)
from usim.model import ImportRequirements, ModelError, ModelSource
from usim.specs import ExecutionSpec, Pose, SceneSpec
from .models import model_io
from .support import unsupported

OPERATIONS = frozenset(
    {
        'models.load',
        'state.read',
        'control.position',
        'control.velocity',
        'control.effort',
        'agents.describe',
        'execution.episode',
        'scene.set_joint_state',
    }
)


class Backend:
    name = 'mujoco'
    gpu = False

    def __init__(self):
        self.models = model_io()

    def describe(self):
        return BackendDescription(
            self.name,
            OPERATIONS,
            frozenset(format.value for format in self.models.formats),
            frozenset({'episode'}),
        )

    def assess(self, request: CapabilityRequest):
        if request.phase in {Phase.CLOSED, Phase.FAILED}:
            return unsupported('The session has no live resources')
        if request.n_envs != 1:
            return unsupported('This MuJoCo integration supports one environment per session')
        if request.operation not in OPERATIONS:
            return unsupported(f'{self.name} does not implement {request.operation}')
        if request.phase is Phase.RUNNING:
            return unsupported('Operations require an episode safe point')
        return SupportReport(SupportStatus.SUPPORTED)

    def open(self, scene: SceneSpec, execution: ExecutionSpec = ExecutionSpec()):
        from .session import Session

        if execution.mode != 'episode' or execution.n_envs != 1:
            unsupported('MuJoCo supports single-environment episode execution').require()
        allowed = {'gpu', 'cuda', 'auto'} if self.gpu else {'cpu', 'auto'}
        if execution.backend not in allowed:
            raise ModelError(f'Invalid {self.name} compute backend: {execution.backend}')
        if execution.render_mode != 'none' or execution.obs_mode not in {'state', 'none'}:
            unsupported('This integration exposes state observations, without rendering').require()
        if execution.native_task != 'JointControl':
            unsupported('MuJoCo loads caller-owned models, not registered tasks').require()
        entries = scene.entities + scene.agents
        if len(entries) + int(scene.world is not None) != 1:
            unsupported('Supply one complete native world, entity, or agent model').require()
        if scene.agents:
            agent = scene.agents[0]
            source = agent.model
            if not isinstance(source, ModelSource):
                raise ModelError('MuJoCo does not supply registered agents')
            if agent.pose != Pose():
                unsupported('Model placement is authored in the native source').require()
            if agent.sensors:
                unsupported('Grouped sensors are not exposed by this integration').require()
            if agent.control.native_mode != 'pd_joint_pos':
                unsupported('Grouped controls use explicit physical joint targets').require()
            requirements = agent.requirements
            name = agent.name
        elif scene.entities:
            entity = scene.entities[0]
            if entity.pose != Pose():
                unsupported('Model placement is authored in the native source').require()
            source = entity.source
            requirements = entity.requirements
            name = entity.name
        else:
            source = scene.world
            if source is None:
                raise UnsupportedOperation(unsupported('A native model source is required'))
            requirements = ImportRequirements(role='scene')
            name = 'world'
        return Session(self, scene, execution, name, self.models.plan(source, requirements))


class WarpBackend(Backend):
    name = 'mujoco-warp'
    gpu = True
