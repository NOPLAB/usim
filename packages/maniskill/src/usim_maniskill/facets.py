"""State readers and control writers bound to a revision of one native scene."""

from dataclasses import replace
from typing import TYPE_CHECKING

from usim.capability import Phase
from usim.facets import ControlTargets, JointLayout, SensorLayout, StateSchema
from usim.groups import ControlBindings, JointControlGroup, StateBindings, StateGroup, TargetMap
from usim.model import ModelError
from usim.scene import EntityHandle
from usim.specs import RegisteredAgentRef
from usim.types import Observation

if TYPE_CHECKING:
    from usim_maniskill.session import Session


def numpy_copy(value):
    """Copy native data to CPU without importing torch or numpy at module import."""
    import numpy as np

    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def joint_names(entity) -> tuple[str, ...]:
    if hasattr(entity, 'get_active_joints'):
        return tuple(joint.name for joint in entity.get_active_joints())
    return ()


def observation(session: 'Session', name: str, handle: EntityHandle | None = None) -> Observation:
    native = session._require().unwrapped
    if handle is not None:
        session.scene.resolve(handle)
    entity = session._entities.get(name)
    import numpy as np

    qpos = (
        numpy_copy(entity.get_qpos())
        if entity is not None and hasattr(entity, 'get_qpos')
        else np.empty((session.options.n_envs, 0))
    )
    qvel = (
        numpy_copy(entity.get_qvel())
        if entity is not None and hasattr(entity, 'get_qvel')
        else np.empty((session.options.n_envs, 0))
    )
    raw = native.get_obs()
    rgb = depth = None
    extra = {'raw_obs': raw}
    if isinstance(raw, dict) and 'sensor_data' in raw:
        sensors = {
            sensor: {kind: numpy_copy(value) for kind, value in textures.items()}
            for sensor, textures in raw['sensor_data'].items()
        }
        for textures in sensors.values():
            if 'depth' in textures:
                # ManiSkill's standard camera transforms produce integer millimetres.
                textures['depth'] = textures['depth'].astype(np.float32) / 1000.0
        extra['sensor_data'] = sensors
        for textures in sensors.values():
            if rgb is None and 'rgb' in textures:
                rgb = textures['rgb']
            if depth is None and 'depth' in textures:
                depth = textures['depth']
    if entity is not None:
        pose = numpy_copy(entity.pose.raw_pose)
        extra['pose'] = pose[..., [0, 1, 2, 4, 5, 6, 3]]
    return Observation(rgb_image=rgb, depth_image=depth, qpos=qpos, qvel=qvel, extra=extra)


def state_group(session: 'Session', handle: EntityHandle) -> StateGroup:
    session._require()
    name = session.scene.resolve(handle)
    entity = session._entities[name]
    joints = []
    for index, joint in enumerate(
        entity.get_active_joints() if hasattr(entity, 'get_active_joints') else ()
    ):
        kind = str(joint.type)
        unit = 'm' if kind == 'prismatic' else 'rad'
        joints.append(
            JointLayout(
                joint.name,
                kind,
                (index, index + 1),
                (index, index + 1),
                unit,
            )
        )
    if joints and numpy_copy(entity.get_qpos()).shape[-1] != len(joints):
        raise ModelError('This native articulation requires a non-scalar joint state layout')
    sensors = []
    requested = {}
    for agent in session.spec.agents:
        if agent.name == name:
            for sensor in agent.sensors:
                requested[sensor.name] = (
                    sensor.width,
                    sensor.height,
                    sensor.parent or 'world',
                    ('rgb', 'depth') if sensor.kind == 'rgbd' else (sensor.kind,),
                )
            native = session._require().unwrapped
            textures = session.options.obs_mode.split('+')
            if session.options.obs_mode == 'rgbd':
                textures = ['rgb', 'depth']
            for config in getattr(native, '_sensor_configs', {}).values():
                if config.uid not in requested:
                    mount = config.mount
                    frame = getattr(mount, 'name', mount) if mount is not None else 'world'
                    requested[config.uid] = (
                        config.width,
                        config.height,
                        frame,
                        tuple(kind for kind in ('rgb', 'depth') if kind in textures),
                    )
    for uid, (width, height, frame, kinds) in requested.items():
        for kind in kinds:
            sensors.append(
                SensorLayout(
                    f'{uid}.{kind}',
                    kind,
                    (height, width, 3 if kind == 'rgb' else 1),
                    'uint8' if kind == 'rgb' else 'float32',
                    'color' if kind == 'rgb' else 'm',
                    frame,
                )
            )
    schema = StateSchema(name, tuple(joints), tuple(sensors), session.options.n_envs)

    def read() -> Observation:
        return observation(session, name, handle)

    def joints_only(positions: bool):
        import numpy as np

        session._require()
        session.scene.resolve(handle)
        native_entity = session._entities[name]
        if not joint_names(native_entity):
            return np.empty((session.options.n_envs, 0))
        return numpy_copy(native_entity.get_qpos() if positions else native_entity.get_qvel())

    return StateGroup(
        schema, StateBindings(read, lambda: joints_only(True), lambda: joints_only(False))
    )


class Control(JointControlGroup):
    """Named direct-model targets, or explicitly native registered-agent actions."""

    def __init__(self, session: 'Session', handle: EntityHandle):
        self._session = session
        self._handle = handle
        spec = session.spec.agents[0]
        names = joint_names(session._entities[spec.name])
        control = spec.control
        registered = isinstance(spec.model, RegisteredAgentRef)
        if not registered:
            groups = control.position_joints + control.velocity_joints + control.effort_joints
            control = replace(
                control,
                joint_names=names,
                position_joints=control.position_joints if groups else names,
            )
        self._control = control
        super().__init__(
            control,
            ControlBindings(
                self._validate,
                None if registered else lambda values: self._write('positions', values),
                None if registered else lambda values: self._write('velocities', values),
                None if registered else lambda values: self._write('efforts', values),
            ),
        )

    def _validate(self, targets: ControlTargets) -> None:
        self._session._require()
        self._session.scene.resolve(self._handle)
        if self._session.phase is Phase.RUNNING:
            raise ModelError('Control writes require an episode safe point')
        for values in (targets.positions, targets.velocities, targets.efforts):
            for value in values.values():
                if (
                    not isinstance(value, (int, float))
                    and len(value) != self._session.options.n_envs
                ):
                    raise ModelError('A control target batch must match n_envs')

    def _write(self, mode: str, values: TargetMap) -> None:
        import numpy as np

        session = self._session
        native = session._require().unwrapped
        names = {
            'positions': self._control.position_joints,
            'velocities': self._control.velocity_joints,
            'efforts': self._control.joint_names,
        }[mode]
        current = session._targets.get(mode)
        if current is None:
            current = np.zeros((session.options.n_envs, len(names)), dtype=np.float32)
            if mode == 'positions':
                qpos = numpy_copy(native.agent.robot.get_qpos())
                order = joint_names(native.agent.robot)
                current[:] = qpos[:, [order.index(name) for name in names]]
        else:
            current = current.copy()
        for name, value in values.items():
            current[:, names.index(name)] = value
        succeeded = False
        try:
            if mode == 'efforts':
                native.agent.robot.set_qf(current)
            else:
                from mani_skill.utils.common import to_tensor

                native.agent.controller.controllers[mode].set_action(
                    to_tensor(current, device=native.device)
                )
            session._sync()
            session._targets[mode] = current
            succeeded = True
        finally:
            if not succeeded:
                session._fail()

    def set_native_action(self, action) -> None:
        """Pass native scaling and ordering through unchanged; never advance physics."""
        import numpy as np

        session = self._session
        session._require()
        session.scene.resolve(self._handle)
        if session.phase is Phase.RUNNING:
            raise ModelError('Control writes require an episode safe point')
        from mani_skill.utils.common import to_tensor

        native = session._require().unwrapped
        action = to_tensor(action, device=native.device)
        if session.options.n_envs == 1 and action.ndim == 1:
            action = action[None, :]
        expected = (session.options.n_envs, *native.agent.single_action_space.shape)
        if tuple(action.shape) != expected:
            raise ModelError(f'Native action shape must be {expected}')
        if not np.isfinite(numpy_copy(action)).all():
            raise ModelError('Native actions must be finite')
        succeeded = False
        try:
            native.agent.set_action(action)
            native.usim_action = action
            session._sync()
            succeeded = True
        finally:
            if not succeeded:
                session._fail()
