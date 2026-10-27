"""Grouped Gazebo Classic provider, with no native imports during discovery."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from usim.capability import (
    BackendDescription,
    CapabilityRequest,
    Phase,
    SupportReport,
    SupportStatus,
)
from usim.model import ModelFormat
from usim.model_io import ModelIO
from usim.specs import ExecutionSpec, SceneSpec
from usim_gazebo.models import FormatProvider, GazeboModel
from usim_gazebo.transport import ContainerPort, RequestPort


OPERATIONS = frozenset(
    {
        'models.load',
        'state.read',
        'state.pose',
        'control.velocity',
        'control.effort',
        'agents.describe',
        'sensors.rgb',
        'sensors.depth',
        'sensors.rgbd',
        'sensors.lidar',
        'execution.continuous',
        'execution.pause',
        'execution.resume',
        'scene.set_pose',
        'scene.set_joint_state',
        'scene.set_parameter',
        'scene.spawn',
        'scene.remove',
        'scene.replace',
        'scene.edit_articulation',
    }
)


class Backend:
    """Compose native formats and launch an isolated continuous world on open."""

    def __init__(
        self,
        *,
        image: str = 'usim-gazebo:local',
        engine: str = 'docker',
        timeout: float = 30.0,
        transport_factory: Callable[[Path], RequestPort] | None = None,
    ) -> None:
        self.models: ModelIO[GazeboModel] = ModelIO[GazeboModel].compose(
            (
                FormatProvider(ModelFormat.URDF),
                FormatProvider(ModelFormat.SDF),
            )
        )
        self._factory = transport_factory or (
            lambda directory: ContainerPort(directory, image=image, engine=engine, timeout=timeout)
        )

    def describe(self) -> BackendDescription:
        return BackendDescription(
            'gazebo',
            OPERATIONS,
            frozenset(format.value for format in self.models.formats),
            frozenset({'continuous'}),
        )

    def assess(self, request: CapabilityRequest) -> SupportReport:
        if request.n_envs != 1:
            return SupportReport(SupportStatus.UNSUPPORTED, ('Gazebo owns one continuous world',))
        if request.phase in {Phase.CLOSED, Phase.FAILED}:
            return SupportReport(SupportStatus.UNSUPPORTED, ('The session is not live',))
        if request.operation not in OPERATIONS:
            return SupportReport(
                SupportStatus.UNSUPPORTED,
                ('No native implementation for ' + request.operation,),
            )
        if request.operation in {'scene.replace', 'scene.edit_articulation'}:
            if request.require_same_episode:
                return SupportReport(
                    SupportStatus.UNSUPPORTED,
                    ('Replacing topology deletes and respawns the entity; native state is lost',),
                    effects=('entity_state_reset', 'handles_invalidated'),
                )
        return SupportReport(
            SupportStatus.CONDITIONAL,
            ('Open a session to verify ROS services and request-specific model support',),
        )

    def open(
        self,
        scene: SceneSpec,
        execution: ExecutionSpec = ExecutionSpec(mode='continuous'),
    ):
        from usim_gazebo.session import Session

        return Session(self, scene, execution, self._factory)
