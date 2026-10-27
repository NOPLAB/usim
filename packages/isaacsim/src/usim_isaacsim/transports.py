"""Explicit, content-assessed Isaac importer transports; no SDK import at discovery."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from xml.etree import ElementTree

from usim.model import (
    ConversionAssessment,
    ImportContext,
    ImportReport,
    ImportRequirements,
    ModelError,
    ModelFormat,
    ModelSource,
)


def _authored_inertia(link: ElementTree.Element) -> bool:
    inertia, mass = link.find('inertial/inertia'), link.find('inertial/mass')
    if inertia is None or mass is None:
        return False
    try:
        xx, yy, zz, xy, xz, yz = (
            float(inertia.get(name, '')) for name in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')
        )
        weight = float(mass.get('value', ''))
    except ValueError:
        return False
    return (
        all(isfinite(value) for value in (xx, yy, zz, xy, xz, yz, weight))
        and weight > 0
        and xx > 0
        and xx * yy - xy * xy > 0
        and xx * yy * zz + 2 * xy * xz * yz - xx * yz * yz - yy * xz * xz - zz * xy * xy > 0
    )


def _primitive_collision(geometry: ElementTree.Element) -> bool:
    if len(geometry) != 1 or geometry[0].tag not in {'box', 'sphere'}:
        return False
    shape = geometry[0]
    values = shape.get('size', '').split() if shape.tag == 'box' else [shape.get('radius', '')]
    try:
        return len(values) == (3 if shape.tag == 'box' else 1) and all(
            isfinite(float(value)) and float(value) > 0 for value in values
        )
    except ValueError:
        return False


def _xml_report(source: ModelSource, requirements: ImportRequirements) -> ImportReport:
    root = ElementTree.parse(source.path).getroot()
    expected = {ModelFormat.MJCF: 'mujoco', ModelFormat.URDF: 'robot'}
    if root.tag != expected[source.format]:
        raise ModelError(f'Expected {expected[source.format]} XML, found {root.tag}')
    if source.selector is not None:
        raise ModelError('Isaac XML transports import a whole model; selectors require USD')
    if requirements.role not in {'articulation', 'scene'}:
        raise ModelError('XML transports retain physics; use authored USD for object roles')
    if root.findall('.//include'):
        raise ModelError('Expand XML includes before content assessment')
    preserved = {'joint_structure'}
    changed = set()
    lost = set()
    unknown = {'inertia', 'collision', 'materials'}
    joints = root.findall('.//joint')
    match source.format:
        case ModelFormat.URDF:
            if any(
                joint.get('type') not in {'fixed', 'revolute', 'continuous', 'prismatic'}
                for joint in joints
            ):
                raise ModelError('Isaac transport supports fixed, revolute and prismatic URDF')
            links = root.findall('link')
            if not links:
                raise ModelError('URDF has no links')
            if all(_authored_inertia(link) for link in links):
                preserved.add('inertia')
                unknown.remove('inertia')
            collision = root.findall('.//collision/geometry')
            if collision and all(_primitive_collision(geometry) for geometry in collision):
                preserved.add('collision')
                unknown.remove('collision')
            if root.findall('.//collision/geometry/mesh') or root.findall(
                './/collision/geometry/cylinder'
            ):
                changed.add('collision')
                unknown.discard('collision')
            if root.findall('.//mimic'):
                unknown.add('mimic')
                unknown.add('joint_structure')
                preserved.discard('joint_structure')
            if root.findall('transmission'):
                lost.add('transmissions')
            if root.findall('gazebo'):
                lost.add('gazebo_extensions')
        case ModelFormat.MJCF:
            bodies = root.findall('.//worldbody/body')
            if not bodies:
                raise ModelError('MJCF has no world bodies')
            if any(
                joint.get('type', 'hinge') not in {'hinge', 'slide', 'free'} for joint in joints
            ):
                raise ModelError('Isaac transport requires scalar joints or a free base')
            if root.findall('.//default'):
                unknown.add('joint_structure')
                preserved.discard('joint_structure')
            for tag, feature in (
                ('actuator', 'actuators'),
                ('sensor', 'sensors'),
                ('tendon', 'tendons'),
                ('equality', 'constraints'),
                ('contact', 'contact_filters'),
            ):
                if root.find(tag) is not None:
                    lost.add(feature)
            # MuJoCo inertia inference, solver/contact and actuator semantics do not
            # establish PhysX fidelity merely because the importer accepts the XML.
            if any(geom.get('type') == 'mesh' for geom in root.findall('.//geom')):
                changed.add('collision')
                unknown.discard('collision')
        case _:
            raise ModelError('Only URDF and MJCF are Isaac XML transports')
    if requirements.base != 'authored':
        changed.add('base')
    return ImportReport(
        preserved=frozenset(preserved),
        changed=frozenset(changed),
        lost=frozenset(lost),
        unknown=frozenset(unknown | (requirements.preserve - preserved - changed - lost)),
        notes=('Physics conversion is not a claim of cross-engine dynamic equivalence.',),
    )


def import_xml(source: ModelSource, context: ImportContext) -> Path:
    """Execute an importer after the caller has started SimulationApp."""
    from isaacsim.core.utils.extensions import enable_extension

    base = {'authored': None, 'fixed': True, 'floating': False}[context.requirements.base]
    match source.format:
        case ModelFormat.MJCF:
            enable_extension('isaacsim.asset.importer.mjcf')
            from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

            result = MJCFImporter(
                MJCFImporterConfig(
                    mjcf_path=str(source.path),
                    usd_path=str(context.directory),
                    fix_base=base,
                    import_scene=context.requirements.role == 'scene',
                    joint_target_type='position',
                    joint_drive_type='force',
                )
            ).import_mjcf()
        case ModelFormat.URDF:
            enable_extension('isaacsim.asset.importer.urdf')
            from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

            result = URDFImporter(
                URDFImporterConfig(
                    urdf_path=str(source.path),
                    usd_path=str(context.directory),
                    merge_fixed_joints=False,
                    fix_base=base,
                    collision_from_visuals=False,
                    joint_target_type='position',
                    joint_drive_type='force',
                )
            ).import_urdf()
        case _:
            raise ModelError('Expected XML import source')
    path = Path(result)
    if not path.is_file():
        raise ModelError(f'Isaac import produced no USD: {path}')
    return path


@dataclass(frozen=True, slots=True)
class XmlToUsd:
    """One directed importer, requiring an explicitly owned live Kit runtime."""

    source_format: ModelFormat
    require_runtime: Callable[[], None]
    target_format: ModelFormat = ModelFormat.USD
    cost: float = 1.0

    @property
    def name(self) -> str:
        return f'isaacsim:{self.source_format.value}-to-usd'

    def assess(
        self, source: ModelSource, requirements: ImportRequirements
    ) -> ConversionAssessment:
        if source.format != self.source_format:
            return ConversionAssessment(False, reasons=('Source format does not match transport',))
        return ConversionAssessment(True, _xml_report(source, requirements))

    def convert(self, source: ModelSource, context: ImportContext) -> ModelSource:
        self.require_runtime()
        assessment = self.assess(source, context.requirements)
        if not assessment.supported:
            raise ModelError('; '.join(assessment.reasons))
        prepared = source
        if source.format == ModelFormat.URDF:
            root = ElementTree.parse(source.path).getroot()
            for element in root.findall('.//mesh') + root.findall('.//texture'):
                filename = element.get('filename')
                if not filename:
                    continue
                if filename.startswith('package://'):
                    package, _, relative = filename.removeprefix('package://').partition('/')
                    matches = [
                        (base if base.name == package else base / package) / relative
                        for base in source.resources
                        if (
                            (base if base.name == package else base / package) / relative
                        ).is_file()
                    ]
                    if len(matches) != 1:
                        raise ModelError(f'Package resource must resolve uniquely: {filename}')
                    resolved = matches[0]
                else:
                    resolved = source.path.parent / filename.removeprefix('file://')
                if not resolved.is_file():
                    raise ModelError(f'Missing URDF resource: {resolved}')
                element.set('filename', str(resolved.resolve()))
            expanded = context.directory / 'source.urdf'
            ElementTree.ElementTree(root).write(expanded, encoding='utf-8', xml_declaration=True)
            prepared = ModelSource(expanded, ModelFormat.URDF, resources=source.resources)
        return ModelSource(
            import_xml(prepared, context), ModelFormat.USD, resources=source.resources
        )
