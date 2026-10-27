"""Native Gazebo request assessment and acknowledged scene mutation."""

from dataclasses import replace
from math import isfinite

from usim.capability import CapabilityRequest, Phase, SupportReport, SupportStatus
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
from usim.specs import Pose

from .contracts import SessionView
from .transport import Json


REQUIRED_SERVICES = {
    'scene.set_pose': ('/set_entity_state',),
    'scene.set_joint_state': ('/set_model_configuration',),
    'scene.set_parameter': ('/get_link_properties', '/set_link_properties'),
    'scene.spawn': ('/spawn_entity',),
    'scene.remove': ('/delete_entity',),
    'scene.replace': ('/delete_entity', '/spawn_entity', '/get_entity_state'),
    'scene.edit_articulation': ('/delete_entity', '/spawn_entity', '/get_entity_state'),
    'state.read': ('/get_joint_properties',),
    'state.pose': ('/get_entity_state',),
    'control.velocity': ('/set_joint_properties',),
    'control.effort': ('/apply_joint_effort', '/clear_joint_forces'),
    'execution.pause': ('/pause_physics',),
    'execution.resume': ('/unpause_physics',),
    'execution.continuous': ('/unpause_physics', '/pause_physics'),
}


def pose_fields(pose: Pose) -> dict[str, Json]:
    return {
        'position': dict(zip(('x', 'y', 'z'), pose.position)),
        'orientation': dict(zip(('x', 'y', 'z', 'w'), pose.orientation_xyzw)),
    }


def unsupported(reason: str) -> SupportReport:
    return SupportReport(SupportStatus.UNSUPPORTED, (reason,))


def assess_live(session: SessionView, request: CapabilityRequest) -> SupportReport:
    request = replace(request, phase=session.phase, n_envs=1)
    generic = session.backend.assess(request)
    if generic.status is SupportStatus.UNSUPPORTED:
        return generic
    if request.operation.startswith('sensors.'):
        kind = request.operation.removeprefix('sensors.')
        requested = {'rgb', 'depth'} if kind == 'rgbd' else {kind}
        attached = {
            channel.layout.kind
            for channels in session._sensor_channels.values()
            for channel in channels
        }
        if not requested <= attached:
            return unsupported('The requested sensor kind is not attached to this scene')
    missing = set(REQUIRED_SERVICES.get(request.operation, ())) - session._services
    if missing:
        return unsupported('Unavailable Gazebo services: ' + ', '.join(sorted(missing)))
    if request.operation.startswith('scene.') and request.phase not in {
        Phase.READY,
        Phase.PAUSED,
        Phase.BUILDING,
    }:
        return SupportReport(
            SupportStatus.CONDITIONAL,
            ('Pause the continuous world before editing',),
            required_phase=Phase.PAUSED,
        )
    effects = ()
    if request.operation in {'scene.replace', 'scene.edit_articulation'}:
        effects = (
            'entity_state_reset',
            'controller_reset',
            'sensor_bindings_removed',
            'handles_invalidated',
        )
    return SupportReport(SupportStatus.SUPPORTED, effects=effects)


def assess_change(
    session: SessionView, change: SceneChange, request: CapabilityRequest
) -> SupportReport:
    support = session.assess(request)
    if not support.supported:
        return support
    match change:
        case SetJointState(entity=name, positions=positions, velocities=velocities):
            if velocities:
                return unsupported('SetModelConfiguration writes positions, not velocities')
            names = {joint.name for joint in session._models[name].value.joints}
            supplied = [joint for joint, _ in positions]
            if not positions or len(set(supplied)) != len(supplied):
                return unsupported('Provide unique named joint positions')
            if not set(supplied) <= names or not all(isfinite(value) for _, value in positions):
                return unsupported('Joint-state positions must be finite and name existing joints')
        case SetParameter(entity=name, parameter=parameter, value=value):
            link, separator, field = parameter.partition('.')
            if (
                not separator
                or field != 'mass'
                or link not in session._models[name].value.links
                or not isfinite(value)
                or value <= 0
            ):
                return unsupported('Supported parameter is <link>.mass with a positive mass')
        case SpawnEntity(spec=spec):
            return session.models.assess(spec.source, spec.requirements)
        case (
            ReplaceModel(entity=name, source=source) | EditArticulation(entity=name, source=source)
        ):
            model_support = session.models.assess(source, session._entities[name].requirements)
            return support if model_support.supported else model_support
        case SetPose() | RemoveEntity():
            pass
    return support


def apply_change(session: SessionView, change: SceneChange) -> None:
    if isinstance(change, (SpawnEntity, RemoveEntity, ReplaceModel, EditArticulation)):
        # SceneEditor invalidates all topology-scoped handles after the operation.
        # Stop native commands while the old handles still resolve.
        session.control.invalidate()
    match change:
        case SetPose(entity=name, pose=pose):
            session._call(
                '/set_entity_state',
                {
                    'state': {
                        'name': name,
                        'pose': pose_fields(pose),
                        'reference_frame': 'world',
                    },
                },
            )
            session._entities[name] = replace(session._entities[name], pose=pose)
        case SetJointState(entity=name, positions=positions):
            session._call(
                '/set_model_configuration',
                {
                    'model_name': name,
                    'joint_names': [joint for joint, _ in positions],
                    'joint_positions': [value for _, value in positions],
                },
            )
        case SetParameter(entity=name, parameter=parameter, value=value):
            link = name + '::' + parameter.partition('.')[0]
            previous = session._call('/get_link_properties', {'link_name': link})
            fields = {
                key: previous[key]
                for key in (
                    'com',
                    'gravity_mode',
                    'ixx',
                    'iyy',
                    'izz',
                    'ixy',
                    'ixz',
                    'iyz',
                )
            }
            fields.update(link_name=link, mass=value)
            session._call('/set_link_properties', fields)
        case SpawnEntity(spec=spec):
            model = session._load(spec.source, spec.requirements)
            session._models[spec.name] = model
            session._spawn(spec)
            session._entities[spec.name] = spec
        case RemoveEntity(entity=name):
            session.control.discard(name)
            session._call('/delete_entity', {'name': name})
            session._models.pop(name).close()
            session._entities.pop(name)
            session._agent_specs.pop(name, None)
            session._sensor_channels.pop(name, None)
            session._spawn_xml.pop(name, None)
        case (
            ReplaceModel(entity=name, source=source) | EditArticulation(entity=name, source=source)
        ):
            old = session._entities[name]
            replacement = session._load(source, old.requirements)
            pose = session.state[name].get_pose()
            session.control.discard(name)
            session._call('/delete_entity', {'name': name})
            session._models.pop(name).close()
            session._models[name] = replacement
            session._sensor_channels.pop(name, None)
            session._spawn_xml.pop(name, None)
            session._entities[name] = replace(old, source=source, pose=pose)
            session._spawn(session._entities[name])
            # A new topology cannot inherit an old controller or agent binding.
            session._agent_specs.pop(name, None)
