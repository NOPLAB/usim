"""Grouped ManiSkill provider. Importing this module needs no native dependencies."""

from dataclasses import replace

from usim.capability import (
    BackendDescription,
    CapabilityRequest,
    Phase,
    SupportReport,
    SupportStatus,
    assert_never,
)
from usim.model import ModelError, ModelSource
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
from usim.specs import ExecutionSpec, RegisteredAgentRef, SceneSpec
from usim_maniskill.models import model_io


TOPOLOGY = frozenset(
    {
        'scene.spawn',
        'scene.remove',
        'scene.replace',
        'scene.edit_articulation',
    }
)
OPERATIONS = TOPOLOGY | {
    'models.load',
    'state.read',
    'control.position',
    'control.velocity',
    'control.effort',
    'control.native_action',
    'agents.describe',
    'execution.episode',
    'scene.set_pose',
    'scene.set_joint_state',
    'scene.set_parameter',
    'sensors.rgb',
    'sensors.depth',
    'sensors.rgbd',
}


def unsupported(reason: str) -> SupportReport:
    return SupportReport(SupportStatus.UNSUPPORTED, (reason,))


def validate_scene(scene: SceneSpec, execution: ExecutionSpec) -> None:
    """Validate declarative requests before creating or changing a native world."""
    if execution.mode != 'episode':
        unsupported('ManiSkill supplies native episode execution, not continuous run').require()
    if execution.backend not in {'cpu', 'physx_cpu', 'gpu', 'physx_cuda', 'auto'}:
        raise ModelError(f'Unknown ManiSkill compute backend: {execution.backend}')
    if not float(execution.sim_rate).is_integer():
        raise ModelError('ManiSkill requires an integer control frequency')
    if execution.n_envs > 1 and execution.backend in {'cpu', 'physx_cpu'}:
        unsupported('Multiple environments require a GPU or auto compute backend').require()
    if scene.world is not None:
        unsupported('Supply independent model entities, not an opaque native world').require()
    if len(scene.agents) > 1:
        unsupported('The grouped ManiSkill session supports at most one agent').require()
    if execution.native_task != 'JointControl':
        if scene.entities or len(scene.agents) != 1 or scene.agents[0].sensors:
            unsupported('Native registered tasks own their scene objects and sensors').require()
        agent = next(iter(scene.agents), None)
        if agent is None or not isinstance(agent.model, RegisteredAgentRef):
            unsupported('Direct models use the generic JointControl task').require()
    for agent in scene.agents:
        match agent.model:
            case RegisteredAgentRef(backend=backend):
                if backend != 'maniskill':
                    unsupported('A registered agent must belong to ManiSkill').require()
                if (
                    agent.control.position_joints
                    or agent.control.velocity_joints
                    or agent.control.effort_joints
                ):
                    unsupported(
                        'Registered agents retain native controller semantics; use native actions'
                    ).require()
            case ModelSource():
                if agent.requirements.role != 'articulation':
                    raise ModelError('A direct agent must be an articulation')
                if agent.control.native_mode != 'pd_joint_pos':
                    unsupported('Direct agents bind explicit physical control groups').require()
            case unreachable:
                assert_never(unreachable)
        for sensor in agent.sensors:
            if sensor.kind not in {'rgb', 'depth', 'rgbd'}:
                unsupported(f'No native grouped {sensor.kind} sensor').require()
            if sensor.rate != execution.sim_rate:
                unsupported('Camera rate must equal the native control frequency').require()
            textures = execution.obs_mode.split('+')
            required = {'rgb', 'depth'} if sensor.kind == 'rgbd' else {sensor.kind}
            if execution.obs_mode == 'rgbd':
                textures = ['rgb', 'depth']
            if not required <= set(textures):
                unsupported(
                    'The native observation mode must include requested sensor textures'
                ).require()


class Backend:
    """Compose asset import and a lazy native runtime without global agent configuration."""

    def __init__(self):
        self.models = model_io()

    def describe(self) -> BackendDescription:
        return BackendDescription(
            'maniskill',
            OPERATIONS,
            frozenset(format.value for format in self.models.formats),
            frozenset({'episode'}),
            registration_required=False,
        )

    def assess(self, request: CapabilityRequest) -> SupportReport:
        if request.phase in {Phase.CLOSED, Phase.FAILED}:
            return unsupported('The scene has no live native resources')
        if request.operation not in OPERATIONS:
            return unsupported(f'ManiSkill does not implement {request.operation}')
        if request.operation.startswith('control.') and request.phase is Phase.RUNNING:
            return unsupported('Control writes require an episode safe point')
        if request.operation in TOPOLOGY:
            if request.phase is Phase.RUNNING:
                return unsupported('Reconfiguration requires an episode safe point')
            if request.phase is not Phase.BUILDING and request.require_same_episode:
                return unsupported(
                    'Topology changes reconstruct the scene and start a new episode'
                )
            return SupportReport(
                SupportStatus.SUPPORTED,
                effects=('reconstruct_scene', 'reset_episode', 'invalidate_handles'),
                scope='all_envs',
            )
        if request.operation in {'scene.set_pose', 'scene.set_joint_state', 'scene.set_parameter'}:
            if request.phase is Phase.RUNNING:
                return unsupported('State and parameter edits require an episode safe point')
        if request.operation == 'scene.set_parameter':
            return SupportReport(
                SupportStatus.CONDITIONAL,
                ('CPU articulation joint friction is editable; a concrete joint is required',),
            )
        return SupportReport(SupportStatus.SUPPORTED, scope='all_envs')

    def open(self, scene: SceneSpec, execution: ExecutionSpec = ExecutionSpec()):
        validate_scene(scene, execution)
        from usim_maniskill.session import Session

        return Session(self, scene, execution)


def changed_spec(spec: SceneSpec, change: SceneChange) -> SceneSpec:
    match change:
        case SpawnEntity(spec=entity):
            return replace(spec, entities=spec.entities + (entity,))
        case RemoveEntity(entity=name):
            return replace(
                spec,
                entities=tuple(entity for entity in spec.entities if entity.name != name),
                agents=tuple(agent for agent in spec.agents if agent.name != name),
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
            )
        case SetPose() | SetJointState() | SetParameter():
            raise ModelError('A state edit is not a topology replacement')
        case unreachable:
            assert_never(unreachable)
