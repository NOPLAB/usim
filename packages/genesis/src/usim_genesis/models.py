"""Parsed model assets consumed by the Genesis scene builder."""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from xml.etree import ElementTree

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
from usim.model_io import ModelIO


@dataclass(frozen=True, slots=True)
class ParsedAsset:
    """An inspected source, with the joint declarations used during build preflight."""

    source: ModelSource
    joints: tuple[tuple[str, str], ...]
    root: ElementTree.Element | None = None


def parse(source: ModelSource) -> ParsedAsset:
    """Read real model contents without importing the physics runtime."""
    match source.format:
        case ModelFormat.URDF | ModelFormat.MJCF:
            if source.selector is not None:
                raise ModelError('Genesis XML import does not support subtree selectors')
            root = ElementTree.parse(source.path).getroot()
            expected = {ModelFormat.URDF: 'robot', ModelFormat.MJCF: 'mujoco'}[source.format]
            if root.tag != expected:
                raise ModelError(f'Expected {expected} XML root, got {root.tag}')
            # Inline native XML must keep references relative to the original asset,
            # not to the application's working directory.
            if source.format is ModelFormat.URDF:
                for element in root.iter():
                    filename = element.get('filename')
                    if filename is not None:
                        if filename.startswith('package://'):
                            relative = Path(filename.removeprefix('package://'))
                            candidates = [
                                base / relative for base in source.resources if base.is_dir()
                            ] + [
                                base / Path(*relative.parts[1:])
                                for base in source.resources
                                if base.name == relative.parts[0]
                            ]
                            resolved = next((path for path in candidates if path.is_file()), None)
                            if resolved is None:
                                raise ModelError(f'Unresolved model resource: {filename}')
                        else:
                            resolved = source.path.parent / filename.removeprefix('file://')
                        if not resolved.is_file():
                            raise ModelError(f'Missing model resource: {resolved}')
                        element.set('filename', str(resolved.resolve()))
            else:
                compiler = root.find('compiler')
                if compiler is None:
                    compiler = ElementTree.SubElement(root, 'compiler')
                for attribute in ('meshdir', 'texturedir'):
                    directory = compiler.get(attribute, compiler.get('assetdir', '.'))
                    compiler.set(attribute, str((source.path.parent / directory).resolve()))
                for include in root.iter('include'):
                    filename = include.get('file')
                    if filename is None or not (source.path.parent / filename).is_file():
                        raise ModelError(f'Missing MJCF include: {filename}')
                    include.set('file', str((source.path.parent / filename).resolve()))
            joints = tuple(
                (joint.attrib['name'], joint.get('type', 'hinge'))
                for joint in root.iter('joint')
                if 'name' in joint.attrib
            )
            return ParsedAsset(source, joints, root)
        case ModelFormat.USD:
            # OpenUSD is an optional import, not a dependency of discovery or XML loading.
            usd = import_module('pxr.Usd')
            stage = usd.Stage.Open(str(source.path))
            if not stage:
                raise ModelError(f'Cannot open USD stage: {source.path}')
            if source.selector is not None:
                prim = stage.GetPrimAtPath(source.selector)
                if not prim:
                    raise ModelError(f'Unknown USD prim: {source.selector}')
                prims = usd.PrimRange(prim)
            else:
                prims = stage.Traverse()
            joints = tuple(
                (str(prim.GetPath()), str(prim.GetTypeName()))
                for prim in prims
                if str(prim.GetTypeName()).endswith('Joint')
            )
            return ParsedAsset(source, joints)
        case ModelFormat.SDF:
            raise ModelError('Genesis has no native SDF loader')
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class GenesisLoader:
    """Parse an asset and report only the fidelity the selected native route provides."""

    format: ModelFormat

    def assess(self, source: ModelSource, requirements: ImportRequirements) -> ImportReport:
        if source.format is not self.format:
            raise ModelError('Genesis loader format does not match source format')
        asset = parse(source)
        if requirements.role == 'scene':
            raise ModelError(
                'Load individual entities; Genesis stage-wide scene import is unbound'
            )
        if source.format is ModelFormat.MJCF and requirements.base != 'authored':
            raise ModelError('Genesis MJCF base mode must be authored in the model')
        if requirements.role == 'rigid_body' and any(
            kind not in {'fixed', 'free', 'PhysicsFixedJoint'} for _, kind in asset.joints
        ):
            raise ModelError('Articulated source cannot be imported as one rigid body')
        preserved = frozenset({'joint_structure', 'inertia', 'collision'})
        unknown = frozenset({'materials', 'actuators', 'sensors'})
        notes = ('Collision simplification and fixed-link merging are disabled.',)
        lost: frozenset[str] = frozenset()
        if (
            source.format is ModelFormat.MJCF
            and asset.root is not None
            and asset.root.find('option') is not None
        ):
            lost = frozenset({'scene_options'})
            notes += ('Genesis ignores MJCF global simulation options; ExecutionSpec owns them.',)
        if source.format is ModelFormat.USD:
            # USD has more joint types than Genesis; do not promise unknown constraints.
            supported = {
                'PhysicsFixedJoint',
                'PhysicsRevoluteJoint',
                'PhysicsPrismaticJoint',
                'PhysicsSphericalJoint',
                'PhysicsJoint',
            }
            if any(kind not in supported for _, kind in asset.joints):
                preserved -= {'joint_structure'}
                unknown |= {'joint_structure'}
        return ImportReport(
            preserved=preserved,
            changed=frozenset({'base'}) if requirements.base != 'authored' else frozenset(),
            lost=lost,
            unknown=unknown | (requirements.preserve - preserved - lost),
            joint_mapping=tuple((name, name) for name, _ in asset.joints),
            notes=notes,
        )

    def load(self, source: ModelSource, context: ImportContext) -> ImportedModel[ParsedAsset]:
        report = self.assess(source, context.requirements)
        return ImportedModel(source, parse(source), report)


def model_io() -> ModelIO[ParsedAsset]:
    return ModelIO.compose(
        tuple(
            GenesisLoader(format)
            for format in (ModelFormat.URDF, ModelFormat.MJCF, ModelFormat.USD)
        )
    )
