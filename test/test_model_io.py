"""Deterministic composition tests using XML-backed providers, without engines."""

from dataclasses import dataclass, field, replace
from pathlib import Path
import subprocess
import sys
from typing import Final
from xml.etree import ElementTree as ET

import pytest

from usim.capability import SupportStatus
from usim.model import (
    ConversionAssessment,
    ImportContext,
    ImportedModel,
    ImportReport,
    ImportRequirements,
    ModelError,
    ModelFormat,
    ModelSource,
)
from usim.model_io import ModelIO

FEATURES: Final = frozenset({'joint_structure', 'inertia', 'collision'})
TAGS: Final = {ModelFormat.URDF: 'robot', ModelFormat.MJCF: 'mujoco', ModelFormat.SDF: 'sdf'}


def inspect_xml(source: ModelSource) -> ImportReport:
    root = ET.parse(source.path).getroot()
    if root.tag != TAGS[source.format]:
        raise ModelError('XML root does not match the declared format')
    if source.selector and root.find(f".//*[@name='{source.selector}']") is None:
        raise ModelError('Selected model element is absent')
    paths = {'joint_structure': './/joint', 'inertia': './/inertia', 'collision': './/collision'}
    found = frozenset(name for name, path in paths.items() if root.find(path) is not None)
    return ImportReport(preserved=found, unknown=FEATURES - found)


@dataclass(frozen=True, slots=True)
class XmlLoader:
    format: ModelFormat = ModelFormat.URDF
    final: ImportReport | None = None
    loads: list[Path] = field(default_factory=list)

    def assess(self, source: ModelSource, requirements: ImportRequirements) -> ImportReport:
        if source.format != self.format:
            raise ModelError('Loader only accepts its native format')
        return inspect_xml(source)

    def load(self, source: ModelSource, context: ImportContext) -> ImportedModel[tuple[str, ...]]:
        self.loads.append(context.directory)
        root = ET.parse(source.path).getroot()
        joints = tuple(joint.attrib['name'] for joint in root.iter('joint'))
        return ImportedModel(source, joints, self.final or inspect_xml(source))


@dataclass(frozen=True, slots=True)
class XmlTransport:
    name: str = 'to-urdf'
    source_format: ModelFormat = ModelFormat.MJCF
    target_format: ModelFormat = ModelFormat.URDF
    cost: float = 1.0
    lost: frozenset[str] = frozenset()
    unknown: frozenset[str] = frozenset()
    remove: str | None = None
    calls: list[ModelSource] = field(default_factory=list)

    def assess(
        self, source: ModelSource, requirements: ImportRequirements
    ) -> ConversionAssessment:
        report = inspect_xml(source)
        return ConversionAssessment(
            not report.unknown,
            replace(report, lost=self.lost, unknown=report.unknown | self.unknown),
            ('Transport requires joints, inertia and collision in the XML input',),
        )

    def convert(self, source: ModelSource, context: ImportContext) -> ModelSource:
        self.calls.append(source)
        tree = ET.parse(source.path)
        root = tree.getroot()
        root.tag = TAGS[self.target_format]
        for parent in root.iter():
            for child in list(parent):
                if child.tag == self.remove:
                    parent.remove(child)
        path = context.directory / f'{self.name}.xml'
        tree.write(path)
        return replace(source, path=path, format=self.target_format)


@pytest.fixture
def source(tmp_path: Path) -> ModelSource:
    path = tmp_path / 'arm.xml'
    path.write_text(
        '<mujoco><link name="arm"><inertial><inertia ixx="1" iyy="1" izz="1"/>'
        '</inertial><collision><geometry><sphere radius=".1"/></geometry></collision>'
        '</link><joint name="hinge" type="revolute"/></mujoco>',
        encoding='utf-8',
    )
    return ModelSource.from_path(path)


def test_native_loader_reads_real_xml_without_conversion(source: ModelSource):
    # Given an explicit loader for the source format.
    loader = XmlLoader(ModelFormat.MJCF)
    io = ModelIO.compose([loader])
    # When loading the assessed source.
    with io.load(io.plan(source)) as model:
        # Then native values and verified evidence come from the XML.
        assert model.value == ('hinge',)
        assert model.report.preserved == FEATURES
        assert model.source == source


def test_multi_hop_selects_cheapest_directed_route_and_controls_cycles(source: ModelSource):
    # Given a costly direct route, a cheaper two-hop route and a zero-cost cycle.
    first = XmlTransport('sdf', target_format=ModelFormat.SDF)
    second = XmlTransport('urdf', source_format=ModelFormat.SDF)
    cycle = XmlTransport('cycle', ModelFormat.SDF, ModelFormat.MJCF, 0)
    io = ModelIO.compose([XmlLoader()], [XmlTransport(cost=9), first, second, cycle])
    # When planning and executing without initializing any native engine.
    plan = io.plan(source)
    with io.load(plan) as model:
        # Then each real intermediate is passed to the next provider.
        assert plan.route == ('sdf', 'urdf', 'load:urdf')
        assert plan.cost == 2
        assert plan.support.status is SupportStatus.CONDITIONAL
        assert not plan.report.preserved
        assert model.value == ('hinge',)
        assert second.calls[0].format is ModelFormat.SDF
        assert not cycle.calls


def test_artifacts_live_until_close_and_external_context_survives(
    source: ModelSource, tmp_path: Path
):
    # Given caller-owned scratch space and one transport.
    io = ModelIO.compose([XmlLoader()], [XmlTransport()])
    model = io.load(io.plan(source), context=ImportContext(tmp_path))
    artifact = model.source.path
    assert ET.parse(artifact).getroot().tag == 'robot'
    # When closing the ownership handle twice.
    model.close()
    model.close()
    # Then generated assets are released without removing caller assets.
    assert model.closed and not artifact.exists()
    assert source.path.exists()
    with pytest.raises(ModelError):
        model.__enter__()


@pytest.mark.parametrize('unknown,lost', [(FEATURES, frozenset()), (frozenset(), FEATURES)])
def test_required_features_cannot_be_unknown_or_lost(
    source: ModelSource, unknown: frozenset[str], lost: frozenset[str]
):
    # Given contradictory preservation evidence from a transport.
    io = ModelIO.compose([XmlLoader()], [XmlTransport(unknown=unknown, lost=lost)])
    # When assessing the specific source.
    report = io.assess(source)
    # Then the route is rejected before native loading.
    assert report.status is SupportStatus.UNSUPPORTED and report.reasons


def test_nonrequired_loss_still_requires_explicit_permission(source: ModelSource):
    # Given a transport reporting a known visual loss.
    io = ModelIO.compose([XmlLoader()], [XmlTransport(lost=frozenset({'visuals'}))])
    assert io.assess(source).status is SupportStatus.UNSUPPORTED
    requirements = ImportRequirements(allow_loss=frozenset({'visuals'}))
    # When the caller explicitly allows that loss.
    with io.load(io.plan(source, requirements)) as model:
        # Then loss is reported rather than silently called preservation.
        assert model.report.lost == frozenset({'visuals'})
        assert 'visuals' not in model.report.preserved


@pytest.mark.parametrize('stage', ['intermediate', 'native', 'result'])
def test_changed_content_is_rejected_without_fallback(
    source: ModelSource, tmp_path: Path, stage: str
):
    # Given a chosen route that loses required evidence during execution.
    remove = 'collision' if stage != 'result' else None
    first = XmlTransport(remove=remove)
    final = ImportReport(preserved=FEATURES, unknown=frozenset({'collision'}))
    loader = XmlLoader(final=final if stage == 'result' else None)
    edges = [first, XmlTransport('fallback', cost=20)]
    if stage == 'intermediate':
        first = replace(first, target_format=ModelFormat.SDF)
        edges = [first, XmlTransport('second', source_format=ModelFormat.SDF), edges[1]]
    io = ModelIO.compose([loader], edges)
    plan = io.plan(source)
    # When the selected route reaches the failing boundary.
    with pytest.raises(ModelError):
        io.load(plan, context=ImportContext(tmp_path))
    # Then no alternative route is attempted and temporary artifacts are cleaned.
    assert not edges[-1].calls
    assert not list(tmp_path.glob('usim-import-*'))
    assert bool(loader.loads) == (stage == 'result')


@pytest.mark.parametrize(
    'change', ['bytes', 'resource', 'selector', 'requirements', 'composition', 'provider']
)
def test_plans_reject_changed_inputs(source: ModelSource, tmp_path: Path, change: str):
    # Given a plan tied to bytes, resources, selection, requirements and providers.
    resource = tmp_path / 'mesh.stl'
    resource.write_bytes(b'original mesh')
    selected = replace(source, selector='arm', resources=(resource,))
    io = ModelIO.compose([XmlLoader()], [XmlTransport()])
    plan = io.plan(selected)
    if change == 'bytes':
        source.path.write_text('<mujoco/>', encoding='utf-8')
    if change == 'resource':
        resource.write_bytes(b'changed! mesh')
    if change == 'selector':
        plan = replace(plan, source=replace(selected, selector='other'))
    if change == 'requirements':
        plan = replace(plan, requirements=ImportRequirements(role='visual'))
    if change == 'composition':
        io = ModelIO.compose([XmlLoader()], [XmlTransport()])
    if change == 'provider':
        io = replace(io, _transports=(XmlTransport(cost=2),))
    # When loading a stale plan.
    with pytest.raises(ModelError):
        io.load(plan)


def test_selection_and_resource_roots_reach_every_provider(source: ModelSource, tmp_path: Path):
    # Given explicit selection and external resource roots.
    selected = replace(source, selector='arm', resources=(tmp_path,))
    edge = XmlTransport()
    io = ModelIO.compose([XmlLoader()], [edge])
    # When converting and loading the selected source.
    with io.load(io.plan(selected)) as model:
        # Then transport and final source retain caller-supplied metadata.
        assert edge.calls[0] == selected
        assert model.source.selector == 'arm'
        assert model.source.resources == (tmp_path,)


@pytest.mark.parametrize('cost', [-1, float('nan'), float('inf')])
def test_invalid_transport_cost_is_rejected(cost: float):
    # Given invalid route weights, when composing, then reject before planning.
    with pytest.raises(ModelError):
        ModelIO.compose([XmlLoader()], [XmlTransport(cost=cost)])


def test_metadata_never_promises_reverse_or_unsupplied_native_routes(source: ModelSource):
    # Given only a directed MJCF-to-URDF transport but an MJCF native loader.
    io = ModelIO.compose([XmlLoader(ModelFormat.MJCF)], [XmlTransport()])
    # When querying metadata and an unreachable URDF source.
    report = io.assess(replace(source, format=ModelFormat.URDF))
    # Then conversion cannot be reversed or terminated at an unsupplied loader.
    assert io.native_formats == io.formats == frozenset({ModelFormat.MJCF})
    assert report.status is SupportStatus.UNSUPPORTED and report.reasons


@pytest.mark.parametrize('selector', [None, 'missing'])
def test_native_assessment_rejects_content_without_loading(
    source: ModelSource, selector: str | None
):
    # Given missing required structure or a missing source selector.
    loader = XmlLoader(ModelFormat.MJCF)
    if selector is None:
        source.path.write_text('<mujoco/>', encoding='utf-8')
    io = ModelIO.compose([loader])
    # When assessing the real input, then fail without loading a native model.
    report = io.assess(replace(source, selector=selector))
    assert report.status is SupportStatus.UNSUPPORTED and report.reasons
    assert not loader.loads


def test_planning_can_choose_a_proven_route_before_execution(source: ModelSource):
    # Given an unproven cheap converter and an assessed alternative.
    rejected = XmlTransport('unproven', unknown=FEATURES)
    accepted = XmlTransport('proven', cost=2)
    io = ModelIO.compose([XmlLoader()], [rejected, accepted])
    # When planning, then choose by content without executing either converter.
    plan = io.plan(source)
    assert plan.route == ('proven', 'load:urdf')
    assert not rejected.calls and not accepted.calls


def test_directory_resource_changes_invalidate_plan(source: ModelSource, tmp_path: Path):
    # Given a resource tree included in the original request.
    root = tmp_path / 'resources'
    root.mkdir()
    io = ModelIO.compose([XmlLoader(ModelFormat.MJCF)])
    plan = io.plan(replace(source, resources=(root,)))
    (root / 'new.mesh').write_bytes(b'new dependency')
    # When loading after the tree changes, then reject the stale request.
    with pytest.raises(ModelError):
        io.load(plan)


def test_imports_do_not_require_native_or_global_conversion_dependencies():
    # Given an interpreter that refuses every optional simulator/parser dependency.
    code = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'numpy', 'torch', 'genesis', 'mani_skill', 'sapien', 'isaacsim', 'pxr',
            'mujoco', 'trimesh', 'yourdfpy'
        }:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from usim.model_io import ModelIO
assert not ModelIO.compose([]).formats
"""
    # When importing and composing, then core model I/O remains dependency-free.
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
