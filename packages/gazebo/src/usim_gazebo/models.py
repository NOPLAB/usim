"""Parsed, resource-owning URDF/SDF assets for the Gazebo Classic factory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname
import xml.etree.ElementTree as ET

from usim.model import (
    ImportedModel,
    ImportContext,
    ImportReport,
    ImportRequirements,
    ModelError,
    ModelFormat,
    ModelSource,
)
from usim.model_inspection import require_fidelity
from usim_gazebo.assets import _package_path


@dataclass(frozen=True, slots=True)
class GazeboJoint:
    name: str
    kind: str
    effort: float


@dataclass(frozen=True, slots=True)
class GazeboModel:
    xml: str
    joints: tuple[GazeboJoint, ...]
    links: tuple[str, ...]
    world: bool = False


def _tree(source: ModelSource, requirements: ImportRequirements) -> ET.Element:
    root = ET.parse(source.path).getroot()
    match source.format:
        case ModelFormat.URDF:
            if root.tag != 'robot' or source.selector is not None:
                raise ModelError('URDF requires a robot root without a selector')
            if requirements.role == 'scene':
                raise ModelError('A URDF robot is not an SDF world')
        case ModelFormat.SDF:
            if root.tag != 'sdf':
                raise ModelError('SDF requires an sdf root')
            tag = 'world' if requirements.role == 'scene' else 'model'
            candidates = root.findall(tag)
            if source.selector is not None:
                candidates = [item for item in candidates if item.get('name') == source.selector]
            if len(candidates) != 1:
                raise ModelError(f'Select exactly one top-level SDF {tag}')
            selected = candidates[0]
            root = ET.Element('sdf', version=root.get('version', '1.6'))
            root.append(selected)
        case ModelFormat.MJCF | ModelFormat.USD:
            raise ModelError('Gazebo has no installed MJCF/USD converter')
    if root.find('.//include') is not None:
        raise ModelError(
            'Expand SDF includes before loading; nested resource resolution is unknown'
        )
    if requirements.base != 'authored':
        raise ModelError(
            'Gazebo imports authored base constraints; author fixed/floating in the asset'
        )
    return root


def _resources(root: ET.Element, source: ModelSource) -> list[tuple[ET.Element, str | None, Path]]:
    references = []
    for element in root.iter():
        if element.tag in {'mesh', 'texture'} and element.get('filename'):
            references.append((element, 'filename', element.attrib['filename']))
        if element.tag == 'uri' and element.text:
            references.append((element, None, element.text.strip()))
    result = []
    for element, attribute, value in references:
        value = unquote(value)
        match urlsplit(value).scheme:
            case 'package':
                relative = value.removeprefix('package://')
                candidates = [
                    candidate
                    for resource in source.resources
                    for candidate in (resource / relative, resource / relative.partition('/')[2])
                    if candidate.exists()
                ]
                path = candidates[0] if candidates else _package_path(value, source.path)[0]
            case 'model':
                relative = value.removeprefix('model://')
                candidates = [resource / relative for resource in source.resources]
                path = next((item for item in candidates if item.exists()), None)
                if path is None:
                    raise ModelError(f'Unresolved model resource {value}; provide resource roots')
            case 'file':
                path = Path(url2pathname(value.removeprefix('file://')))
            case '':
                path = source.path.parent / value
            case _:
                # A Windows absolute path is not a URI.
                path = Path(value)
                if not path.is_absolute():
                    raise ModelError(f'Unsupported resource URI: {value}')
        path = path.resolve()
        if not path.exists() or (attribute is not None and not path.is_file()):
            raise ModelError(f'Missing Gazebo resource: {path}')
        result.append((element, attribute, path))
    return result


def _structure(root: ET.Element) -> tuple[tuple[GazeboJoint, ...], tuple[str, ...]]:
    model = root if root.tag == 'robot' else root.find('model')
    if model is None:
        return (), ()
    links = tuple(link.get('name', '') for link in model.findall('link'))
    if not links or any(not name for name in links) or len(set(links)) != len(links):
        raise ModelError('Model requires unique named links')
    joints = []
    all_names = set()
    for joint in model.findall('joint'):
        name, kind = joint.get('name', ''), joint.get('type', '')
        if not name or name in all_names:
            raise ModelError('Model requires unique named joints')
        all_names.add(name)
        for endpoint in ('parent', 'child'):
            node = joint.find(endpoint)
            link = (
                (node.get('link') if root.tag == 'robot' else node.text)
                if node is not None
                else ''
            )
            if link not in links and not (endpoint == 'parent' and link == 'world'):
                raise ModelError(f'Joint {name} has an unknown {endpoint} link: {link}')
        if kind == 'fixed':
            continue
        if kind not in {'revolute', 'continuous', 'prismatic'}:
            raise ModelError(f'Joint {name}: unsupported state layout for {kind}')
        limit = joint.find('limit') if root.tag == 'robot' else joint.find('axis/limit')
        effort = 0.0
        if limit is not None:
            effort = float(
                limit.get('effort', '0') if root.tag == 'robot' else limit.findtext('effort', '0')
            )
        joints.append(GazeboJoint(name, kind, effort))
    return tuple(joints), links


@dataclass(frozen=True, slots=True)
class FormatProvider:
    """Validate native XML; never treat mere filename acceptance as an import."""

    format: ModelFormat

    def assess(self, source: ModelSource, requirements: ImportRequirements) -> ImportReport:
        if source.format != self.format:
            raise ModelError('Format provider does not match the source')
        root = _tree(source, requirements)
        joints, links = _structure(root)
        if requirements.role == 'rigid_body' and joints:
            raise ModelError('An articulated model cannot be imported as one rigid body')
        if requirements.role == 'visual':
            raise ModelError('Visual-only imports require removing authored physics explicitly')
        _resources(root, source)
        preserved = {'joint_structure', 'inertia', 'collision', 'visual', 'materials', 'resources'}
        unknown = set()
        lost = set()
        if root.find('.//plugin') is not None:
            lost.add('plugins')
        for feature, tag in (('mimic', 'mimic'), ('transmissions', 'transmission')):
            if root.find(f'.//{tag}') is not None:
                lost.add(feature)
        # URDF fixed-joint lumping changes the link/joint graph unless explicitly disabled.
        # Loading injects preserveFixedJoint to retain that graph.
        if root.find('.//sensor') is not None:
            unknown.add('sensors')
        unknown.update(requirements.preserve - preserved)
        report = ImportReport(
            preserved=frozenset(preserved),
            lost=frozenset(lost),
            unknown=frozenset(unknown),
            joint_mapping=tuple((joint.name, joint.name) for joint in joints),
            link_mapping=tuple((name, name) for name in links),
            notes=(
                'XML staging, not a claim of native physics validation; plugins are removed '
                'so authored controllers cannot bypass the motor gate.',
            ),
        )
        require_fidelity(report, requirements)
        return report

    def load(self, source: ModelSource, context: ImportContext) -> ImportedModel[GazeboModel]:
        report = self.assess(source, context.requirements)
        root = _tree(source, context.requirements)
        joints, links = _structure(root)
        folders: dict[Path, Path] = {}
        for element, attribute, path in _resources(root, source):
            folder = path.parent
            if folder not in folders:
                destination = context.directory / f'resource-{len(folders)}'
                # Adjacent material/texture files stay together. Do not recursively copy
                # this import's own staging tree when a model lives beside its assets.
                excluded = context.directory.resolve()
                shutil.copytree(
                    folder,
                    destination,
                    ignore=lambda directory, names: [
                        name
                        for name in names
                        if (Path(directory) / name).resolve() == excluded
                        or (Path(directory) / name).resolve() in excluded.parents
                    ],
                )
                folders[folder] = destination
            uri = (folders[folder] / path.name).as_uri()
            if attribute is None:
                element.text = uri
            else:
                element.set(attribute, uri)
        for parent in root.iter():
            for plugin in list(parent.findall('plugin')):
                parent.remove(plugin)
        if root.tag == 'robot':
            for joint in root.findall('joint'):
                if joint.get('type') == 'fixed':
                    extension = ET.SubElement(root, 'gazebo', reference=joint.attrib['name'])
                    ET.SubElement(extension, 'preserveFixedJoint').text = 'true'
        return ImportedModel(
            source,
            GazeboModel(
                ET.tostring(root, encoding='unicode'),
                joints,
                links,
                context.requirements.role == 'scene',
            ),
            report,
        )
