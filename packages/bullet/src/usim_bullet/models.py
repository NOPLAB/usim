"""Conservative import evidence for Bullet's native model loaders."""

from dataclasses import dataclass
from xml.etree import ElementTree as ET

from usim.capability import SupportReport, SupportStatus
from usim.model import (
    ImportContext,
    ImportedModel,
    ImportReport,
    ImportRequirements,
    ModelError,
    ModelFormat,
    ModelSource,
)
from usim.model_io import ModelIO


def unsupported(reason: str) -> SupportReport:
    return SupportReport(SupportStatus.UNSUPPORTED, (reason,))


@dataclass(frozen=True, slots=True)
class Asset:
    source: ModelSource
    requirements: ImportRequirements
    joints: tuple[str, ...]
    report: ImportReport


def inspect(source: ModelSource, requirements: ImportRequirements) -> Asset:
    if source.selector is not None or source.resources:
        raise ModelError('Bullet requires a whole model with model-relative resources')
    if requirements.role not in {'articulation', 'rigid_body'}:
        raise ModelError('Bullet supports articulation and rigid_body entities')
    root = ET.parse(source.path).getroot()
    lost = set()
    unknown = set()
    if source.format is ModelFormat.URDF:
        if root.tag != 'robot' or not root.findall('link'):
            raise ModelError('URDF requires a robot with links')
        joints = []
        for joint in root.findall('joint'):
            kind = joint.get('type')
            if kind not in {'fixed', 'revolute', 'continuous', 'prismatic'}:
                raise ModelError(f'Unsupported Bullet joint type: {kind}')
            if kind != 'fixed':
                joints.append(joint.get('name', ''))
        for tag in ('mimic', 'transmission', 'gazebo', 'sensor'):
            if root.findall(f'.//{tag}'):
                raise ModelError(f'Bullet does not preserve URDF {tag}')
        preserved = {'joint_structure', 'inertia', 'collision'}
    elif source.format is ModelFormat.MJCF:
        if root.tag != 'mujoco':
            raise ModelError('MJCF requires a mujoco root')
        world = root.find('worldbody')
        if world is None or len(world.findall('body')) != 1 or world.findall('geom'):
            raise ModelError('Each Bullet MJCF entity requires exactly one top-level body')
        if requirements.base != 'authored':
            raise ModelError('Bullet MJCF base mobility must be authored in the asset')
        for tag in ('include', 'default', 'equality', 'tendon', 'contact', 'freejoint'):
            if root.findall(f'.//{tag}'):
                raise ModelError(f'This Bullet importer does not support MJCF {tag}')
        joints = []
        for joint in world.iter('joint'):
            kind = joint.get('type', 'hinge')
            if kind not in {'hinge', 'slide'}:
                raise ModelError(f'Unsupported Bullet MJCF joint type: {kind}')
            joints.append(joint.get('name', ''))
        for tag, feature in (
            ('actuator', 'actuators'),
            ('sensor', 'sensors'),
            ('option', 'solver'),
        ):
            if root.find(tag) is not None:
                lost.add(feature)
        # Bullet's basic MJCF importer is not MuJoCo's dynamics compiler.
        preserved = {'joint_structure'}
        unknown.update({'inertia', 'collision'})
    else:
        raise ModelError(f'Bullet has no {source.format.value} loader in this provider')
    if any(not name for name in joints) or len(joints) != len(set(joints)):
        raise ModelError('Active joints must have nonempty unique names')
    if requirements.role == 'rigid_body' and joints:
        raise ModelError('A rigid_body cannot contain active joints')
    unknown.update(requirements.preserve - preserved - lost)
    report = ImportReport(
        route=(f'load:{source.format.value}',),
        preserved=frozenset(preserved),
        lost=frozenset(lost),
        unknown=frozenset(unknown),
        changed=frozenset({'base'})
        if source.format is ModelFormat.URDF and requirements.base == 'fixed'
        else frozenset(),
        joint_mapping=tuple((name, name) for name in joints),
        notes=('Native joint names and scalar layout are verified when opening the scene.',),
    )
    return Asset(source, requirements, tuple(joints), report)


@dataclass(frozen=True, slots=True)
class Loader:
    format: ModelFormat

    def assess(self, source: ModelSource, requirements: ImportRequirements) -> ImportReport:
        return inspect(source, requirements).report

    def load(self, source: ModelSource, context: ImportContext) -> ImportedModel[Asset]:
        asset = inspect(source, context.requirements)
        return ImportedModel(source, asset, asset.report)


def model_io() -> ModelIO[Asset]:
    return ModelIO.compose((Loader(ModelFormat.URDF), Loader(ModelFormat.MJCF)))
