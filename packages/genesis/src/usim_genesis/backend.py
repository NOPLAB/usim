"""Grouped Genesis backend; native libraries are imported only when a session opens."""

from typing import TYPE_CHECKING

from usim.capability import (
    BackendDescription,
    CapabilityRequest,
    Phase,
    SupportReport,
    SupportStatus,
)
from usim.specs import ExecutionSpec, SceneSpec

from .models import model_io

if TYPE_CHECKING:
    from .session import Session


STRUCTURAL = frozenset(
    {
        'scene.spawn',
        'scene.remove',
        'scene.replace',
        'scene.edit_articulation',
    }
)
OPERATIONS = STRUCTURAL | frozenset(
    {
        'models.load',
        'state.read',
        'control.position',
        'control.velocity',
        'control.effort',
        'agents.direct',
        'execution.episode',
        'scene.set_pose',
        'scene.set_joint_state',
        'scene.set_parameter',
        'sensors.camera',
        'sensors.rgb',
        'sensors.depth',
        'sensors.rgbd',
        'sensors.lidar',
    }
)


class Backend:
    def __init__(self) -> None:
        self.models = model_io()

    def describe(self) -> BackendDescription:
        return BackendDescription(
            'genesis',
            OPERATIONS,
            frozenset(format.value for format in self.models.formats),
            frozenset({'episode'}),
        )

    def assess(self, request: CapabilityRequest) -> SupportReport:
        if request.phase in {Phase.CLOSED, Phase.FAILED}:
            return SupportReport(SupportStatus.UNSUPPORTED, ('Session is not usable',))
        if request.n_envs < 1:
            return SupportReport(
                SupportStatus.UNSUPPORTED, ('Environment count must be positive',)
            )
        if request.operation not in OPERATIONS:
            return SupportReport(
                SupportStatus.UNSUPPORTED,
                (f'Genesis does not expose {request.operation}',),
            )
        if request.phase is Phase.RUNNING:
            return SupportReport(
                SupportStatus.CONDITIONAL,
                ('Pause at an episode safe point first',),
                required_phase=Phase.PAUSED,
            )
        if request.operation in STRUCTURAL and request.phase is not Phase.BUILDING:
            if request.require_same_episode:
                return SupportReport(
                    SupportStatus.UNSUPPORTED,
                    ('Genesis topology changes require a rebuilt scene and a new episode',),
                )
            return SupportReport(
                SupportStatus.SUPPORTED,
                effects=(
                    'scene_rebuild',
                    'new_episode',
                    'invalidate_handles',
                    'clear_commands',
                    'reset_state',
                    'reset_parameters',
                ),
            )
        return SupportReport(SupportStatus.SUPPORTED)

    def open(self, scene: SceneSpec, execution: ExecutionSpec = ExecutionSpec()) -> 'Session':
        from .session import Session

        return Session(self, scene, execution)
