"""Parsed model assets and conservative ManiSkill 3.0.1 import evidence.

Parsing does not create a physics world. The generic task realizes these assets
through ManiSkill's native builders, then verifies the reported joint mapping.
"""

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from usim.capability import assert_never
from usim.model import (
    ImportContext,
    ImportedModel,
    ImportReport,
    ImportRequirements,
    ModelError,
    ModelFormat,
    ModelSource,
)
from usim.model_inspection import require_fidelity
from usim.model_io import ModelIO
from usim.specs import AgentSpec


@dataclass(frozen=True, slots=True)
class Asset:
    source: ModelSource
    requirements: ImportRequirements
    joints: tuple[tuple[str, str], ...]
    links: tuple[str, ...]
    fixed_base: bool
    report: ImportReport


def validate_agent(spec: AgentSpec, asset: Asset) -> None:
    names = {name for name, _ in asset.joints}
    selected = (
        spec.control.position_joints + spec.control.velocity_joints + spec.control.effort_joints
    )
    if not names or not set(selected) <= names:
        raise ModelError('Direct agents need active joints and known control selections')
    if spec.control.joint_names and set(spec.control.joint_names) != names:
        raise ModelError('joint_names must describe all native active joints')
    if spec.control.initial_qpos is not None and len(spec.control.initial_qpos) != len(names):
        raise ModelError('Initial positions must match native joint count')
    control = spec.control
    if control.joint_stiffness is not None and control.velocity_joints:
        raise ModelError('Velocity controllers do not support injected joint stiffness')
    if any(
        value is not None
        for value in (control.joint_stiffness, control.joint_damping, control.joint_force_limits)
    ):
        driven = set(control.position_joints + control.velocity_joints) if selected else names
        if driven != names:
            raise ModelError('Injected controller gains require every joint to be PD controlled')
    if any(sensor.parent and sensor.parent not in asset.links for sensor in spec.sensors):
        raise ModelError('A sensor parent must name an imported robot link')


def _document(source: ModelSource) -> ET.Element:
    """Mirror the pinned MJCF importer's top-level include support for inspection."""
    root = ET.parse(source.path).getroot()
    for include in root.findall('include'):
        path = source.path.parent / include.attrib['file']
        included = ET.parse(path).getroot()
        if included.findall('.//include'):
            raise ModelError('The pinned native loader does not support recursive MJCF includes')
        root.remove(include)
        root.extend(included)
    if root.findall('.//include'):
        raise ModelError('The pinned native loader supports only top-level MJCF includes')
    return root


def inspect(source: ModelSource, requirements: ImportRequirements) -> Asset:
    """Reject unsupported structural features before allocating a native scene."""
    if source.selector is not None:
        raise ModelError('ManiSkill imports one complete asset; selectors are unsupported')
    if requirements.role not in {'articulation', 'rigid_body'}:
        raise ModelError('ManiSkill entity imports support articulation and rigid_body roles')
    if len(source.resources) > 1:
        raise ModelError('ManiSkill accepts at most one package resource directory')
    if source.resources and not source.resources[0].is_dir():
        raise ModelError('The ManiSkill package resource root must be a directory')
    root = _document(source)
    lost: set[str] = set()
    unknown: set[str] = set()
    changed: set[str] = set()
    joints = []
    links = []
    authored_fixed = True
    match source.format:
        case ModelFormat.URDF:
            if root.tag != 'robot' or not root.findall('link'):
                raise ModelError('URDF requires a robot containing links')
            links = [link.get('name', '') for link in root.findall('link')]
            for joint in root.findall('joint'):
                kind = joint.get('type', '')
                if kind not in {'fixed', 'revolute', 'continuous', 'prismatic'}:
                    raise ModelError(f'Unsupported URDF joint type: {kind}')
                for tag in ('parent', 'child'):
                    endpoint = joint.find(tag)
                    if endpoint is None:
                        raise ModelError('URDF joints require parent and child links')
                    if endpoint.get('link') not in links:
                        raise ModelError('URDF joint refers to an unknown link')
                if kind != 'fixed':
                    joints.append((joint.get('name', ''), kind))
            if root.findall('.//mimic'):
                unknown.add('mimic')
            if root.findall('transmission'):
                lost.add('actuators')
            if root.findall('.//gazebo') or root.findall('.//sensor'):
                lost.add('sensors')
        case ModelFormat.MJCF:
            if root.tag != 'mujoco':
                raise ModelError('MJCF requires a mujoco root')
            world = root.find('worldbody')
            if world is None:
                raise ModelError('Each MJCF entity must contain exactly one top-level body')
            bodies = world.findall('body')
            if len(bodies) != 1 or world.findall('geom'):
                raise ModelError('Each MJCF entity must contain exactly one top-level body')
            body = bodies[0]
            authored_fixed = body.find('freejoint') is None
            links = [node.get('name', '') for node in world.iter('body')]
            for default in root.findall('default'):
                for joint in default.iter('joint'):
                    if joint.get('type', 'hinge') not in {'hinge', 'slide'}:
                        raise ModelError('Only hinge and slide MJCF joint defaults are supported')
            for node in world.iter():
                if node.get('childclass', '').startswith('motor'):
                    raise ModelError('The pinned native loader ignores motor-class bodies')
            for joint in world.iter('joint'):
                kind = joint.get('type', 'hinge')
                if kind not in {'hinge', 'slide'}:
                    raise ModelError(f'Unsupported MJCF joint type: {kind}')
                joints.append((joint.get('name', ''), kind))
            if any(node.find('freejoint') is not None for node in body.findall('.//body')):
                raise ModelError('Only a root MJCF freejoint is supported')
            # The pinned loader ignores explicit inertial properties and geom mass/density.
            # Never promise MuJoCo dynamics fidelity from a syntactically successful import.
            lost.add('inertia')
            if root.find('actuator') is not None:
                lost.add('actuators')
            if root.find('sensor') is not None or world.findall('.//camera'):
                lost.add('sensors')
            if root.find('option') is not None:
                lost.add('solver')
            if root.find('equality') is not None or root.find('tendon') is not None:
                raise ModelError('MJCF equality constraints and tendons are unsupported')
            if root.find('contact') is not None:
                lost.add('collision')
            for geom in world.iter('geom'):
                if geom.get('type', 'sphere') not in {
                    'box',
                    'sphere',
                    'capsule',
                    'cylinder',
                    'mesh',
                }:
                    raise ModelError(f'Unsupported MJCF geometry: {geom.get("type")}')
                if geom.get('contype', '1') not in {'0', '1'}:
                    lost.add('collision')
                if geom.get('conaffinity', '1') != '1':
                    lost.add('collision')
                if geom.get('group', '0') not in {'0', '2'}:
                    lost.add('visual')
        case ModelFormat.USD | ModelFormat.SDF:
            raise ModelError(f'ManiSkill has no native {source.format.value} loader')
        case unreachable:
            assert_never(unreachable)
    if (
        len(links) != len(set(links))
        or len(joints) != len({name for name, _ in joints})
        or any(not name for name in links)
        or any(not name for name, _ in joints)
    ):
        raise ModelError('Model link and joint names must be nonempty and unique')
    if requirements.role == 'rigid_body' and joints:
        raise ModelError('A rigid_body import cannot contain active joints')
    for node in root.iter():
        for key in ('filename', 'file'):
            reference = node.get(key)
            if reference:
                path = Path(reference)
                if reference.startswith('package://'):
                    if not source.resources:
                        raise ModelError('package:// meshes require an explicit resource root')
                    path = source.resources[0] / reference.removeprefix('package://')
                elif not path.is_absolute():
                    compiler = root.find('compiler')
                    directory = ''
                    if compiler is not None and node.tag in {'mesh', 'texture'}:
                        directory = compiler.get(
                            'meshdir' if node.tag == 'mesh' else 'texturedir', ''
                        )
                    path = source.path.parent / directory / path
                if not path.is_file():
                    raise ModelError(f'Missing model resource: {path}')
    fixed = authored_fixed if requirements.base == 'authored' else requirements.base == 'fixed'
    if fixed != authored_fixed:
        changed.add('base')
    preserved = {'joint_structure', 'collision', 'inertia'} - lost
    unknown.update(requirements.preserve - preserved - lost)
    report = ImportReport(
        route=(f'load:{source.format.value}',),
        preserved=frozenset(preserved),
        changed=frozenset(changed),
        lost=frozenset(lost),
        unknown=frozenset(unknown),
        joint_mapping=tuple((name, name) for name, _ in joints),
        link_mapping=tuple((name, name) for name in links),
        notes=('Parsed entity asset; native construction and joint verification occur at open.',),
    )
    return Asset(source, requirements, tuple(joints), tuple(links), fixed, report)


@dataclass(frozen=True, slots=True)
class AssetLoader:
    """Load a validated asset, not an unexamined path or a fabricated native handle."""

    format: ModelFormat

    def assess(self, source: ModelSource, requirements: ImportRequirements) -> ImportReport:
        if source.format is not self.format:
            raise ModelError('Wrong model format for this native loader')
        return inspect(source, requirements).report

    def load(self, source: ModelSource, context: ImportContext) -> ImportedModel[Asset]:
        asset = inspect(source, context.requirements)
        require_fidelity(asset.report, context.requirements)
        return ImportedModel(source, asset, asset.report)


def model_io() -> ModelIO[Asset]:
    return ModelIO.compose((AssetLoader(ModelFormat.URDF), AssetLoader(ModelFormat.MJCF)))
