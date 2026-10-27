"""Discoverable grouped Isaac backend; engine imports begin only on open."""

from typing import TYPE_CHECKING

from usim.capability import (
    BackendDescription,
    CapabilityRequest,
    Phase,
    SupportReport,
    SupportStatus,
)
from usim.model import ModelError, ModelSource
from usim.specs import AgentSpec, ExecutionSpec, RegisteredAgentRef, SceneSpec
from usim_isaacsim.native_runtime import ImportRuntime

if TYPE_CHECKING:
    from usim_isaacsim.session import Session


class Backend:
    def __init__(self) -> None:
        # Planning XML is dependency-free; USD inspection and loading deliberately
        # require imports() or open(), so no offline identity loader claims success.
        self._imports = ImportRuntime()
        self.models = self._imports.models

    def imports(self) -> ImportRuntime:
        """Own a standalone native model-import context; close models before leaving it."""
        return ImportRuntime()

    def describe(self) -> BackendDescription:
        return BackendDescription(
            'isaacsim',
            frozenset(
                {
                    'models.load',
                    'state.read',
                    'control.position',
                    'control.velocity',
                    'control.effort',
                    'agents.direct',
                    'sensors.rgb',
                    'sensors.depth',
                    'sensors.rgbd',
                    'sensors.lidar',
                    'execution.episode',
                    'execution.continuous',
                    'scene.set_pose',
                    'scene.set_joint_state',
                    'scene.set_parameter',
                    'scene.spawn',
                    'scene.remove',
                    'scene.replace',
                    'scene.edit_articulation',
                }
            ),
            frozenset(format.value for format in self.models.formats),
            frozenset({'episode', 'continuous'}),
        )

    def assess(self, request: CapabilityRequest) -> SupportReport:
        if request.n_envs != 1:
            return SupportReport(SupportStatus.UNSUPPORTED, ('Isaac requires n_envs=1',))
        if request.operation not in self.describe().operations:
            return SupportReport(SupportStatus.UNSUPPORTED, ('Operation is not implemented',))
        if request.phase in {Phase.CLOSED, Phase.FAILED}:
            return SupportReport(SupportStatus.UNSUPPORTED, ('Session is not usable',))
        if request.phase == Phase.RUNNING:
            return SupportReport(
                SupportStatus.CONDITIONAL,
                ('Stop execution before accessing native scene views',),
                required_phase=Phase.PAUSED,
            )
        if request.operation in {
            'scene.spawn',
            'scene.remove',
            'scene.replace',
            'scene.edit_articulation',
        }:
            if request.require_same_episode:
                return SupportReport(
                    SupportStatus.UNSUPPORTED,
                    ('Topology changes require explicit permission to rebuild the episode',),
                    effects=('reset_episode', 'rebind_all', 'invalidate_handles'),
                )
            return SupportReport(
                SupportStatus.SUPPORTED,
                effects=('reset_episode', 'rebind_all', 'invalidate_handles'),
            )
        return SupportReport(SupportStatus.SUPPORTED)

    def resolve_agent(self, agent: AgentSpec) -> AgentSpec:
        if any(
            sensor.kind == 'lidar' and (sensor.height != 1 or sensor.width < 2)
            for sensor in agent.sensors
        ):
            raise ModelError('Isaac planar lidar requires height=1 and at least two rays')
        if agent.requirements.role != 'articulation':
            raise ModelError('Isaac agents require an articulation model role')
        if agent.control.native_mode != 'pd_joint_pos':
            raise ModelError('Grouped Isaac uses named position, velocity and effort drive groups')
        match agent.model:
            case ModelSource():
                return agent
            case RegisteredAgentRef():
                raise ModelError('Isaac requires an externally supplied ModelSource')

    def open(self, scene: SceneSpec, execution: ExecutionSpec = ExecutionSpec()) -> 'Session':
        from usim_isaacsim.session import Session

        if execution.n_envs != 1:
            raise ModelError('Isaac grouped execution requires n_envs=1')
        if execution.backend != 'cpu':
            raise ModelError('Grouped Isaac currently exposes CPU articulation state')
        if execution.obs_mode not in {'state', 'rgb', 'rgbd'} or execution.render_mode not in {
            'none',
            'human',
            'rgb_array',
        }:
            raise ModelError(
                'Grouped Isaac supports state/RGB-D and none/human/rgb_array rendering'
            )
        if execution.obs_mode in {'rgb', 'rgbd'}:
            kinds = {sensor.kind for agent in scene.agents for sensor in agent.sensors}
            if not kinds & {'rgb', 'rgbd'} or (
                execution.obs_mode == 'rgbd' and not kinds & {'depth', 'rgbd'}
            ):
                raise ModelError('Declare matching camera SensorSpecs for RGB-D observations')
        if execution.native_task != 'JointControl':
            raise ModelError('Grouped Isaac supports only JointControl')
        return Session(self, scene, execution)
