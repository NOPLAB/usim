"""Shared authored presentation retains measured geometry and sparse-region fallback."""

from dataclasses import replace
import json

import numpy as np
import pytest

from test.env_pipeline_fixtures import plane_records, write_pcd
from usim_env_pipeline import MapConfig, Source, generate
from usim_env_pipeline.models import PipelineInputError


@pytest.fixture
def runtime():
    pytest.importorskip('open3d')
    pytest.importorskip('scipy')
    return pytest.importorskip('pxr')


def test_authored_surfaces_retain_exact_positions_faces_and_collision(tmp_path, runtime):
    from pxr import Usd, UsdGeom, UsdPhysics
    from usim_env_pipeline.scene import decorate

    # Given a real measured map containing dense surfaces and a point-only core.
    records = plane_records(invalid_core=True)
    source = tmp_path / 'input.pcd'
    write_pcd(source, records)
    output = tmp_path / 'map'
    generate(Source(source), output, MapConfig(workers=1, normal_policy='supplied'))
    # When the shared authored presentation is generated.
    report = decorate(output)
    scene = Usd.Stage.Open(str(output / report['world']))
    raw = Usd.Stage.Open(str(output / 'world.usda'))
    # Then original collision surfaces are exact and sparse points still exist.
    assert report['source_records'] == len(records)
    assert report['point_only_regions'] == 1
    assert report['positions_unchanged']
    assert report['collision_faces_unchanged']
    for prim in raw.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            actual = UsdGeom.Mesh.Get(scene, prim.GetPath())
            expected = UsdGeom.Mesh(prim)
            np.testing.assert_array_equal(
                actual.GetPointsAttr().Get(), expected.GetPointsAttr().Get()
            )
            np.testing.assert_array_equal(
                actual.GetFaceVertexIndicesAttr().Get(), expected.GetFaceVertexIndicesAttr().Get()
            )
            assert UsdPhysics.CollisionAPI(actual.GetPrim()).GetCollisionEnabledAttr().Get()
    assert scene.GetPrimAtPath('/World/PointRegions/t_10002_10000/Points')
    from usim_env_pipeline.verify import verify

    independent = verify(output)
    assert independent['records'] == len(records)
    assert independent['all_source_vertices_exact']


def test_resume_reuses_completed_regions_and_rebuilds_only_unfinished(tmp_path, runtime):
    # Given a real generated map with one missing regional completion marker.
    records = plane_records(invalid_core=True)
    source = tmp_path / 'input.pcd'
    write_pcd(source, records)
    output = tmp_path / 'map'
    config = MapConfig(workers=1, normal_policy='supplied')
    first = generate(Source(source), output, config)
    existing = output / 'points/t_10000_10000.usdc'
    before_bytes = existing.read_bytes()
    before_time = existing.stat().st_mtime_ns
    (output / 'reports/t_10001_10000.json').unlink()
    manifest_file = output / 'manifest.json'
    manifest = json.loads(manifest_file.read_text())
    manifest['status'] = 'generating'
    manifest_file.write_text(json.dumps(manifest))
    # When explicitly resuming the identical source with another worker count.
    resumed = generate(Source(source), output, replace(config, resume=True, workers=2))
    # Then metrics match, completed payloads were untouched, and the missing region completed.
    assert resumed == first
    assert existing.read_bytes() == before_bytes
    assert existing.stat().st_mtime_ns == before_time
    assert (output / 'reports/t_10001_10000.json').is_file()
    assert json.loads(manifest_file.read_text())['status'] == 'complete'


def test_resume_rejects_different_source_or_geometry_settings(tmp_path, runtime):
    # Given a complete artifact tied to exact source bytes and reconstruction settings.
    source = tmp_path / 'input.pcd'
    records = plane_records()
    write_pcd(source, records)
    output = tmp_path / 'map'
    config = MapConfig(workers=1, normal_policy='supplied')
    generate(Source(source), output, config)
    # When changing geometric support or original rows while requesting resume.
    with pytest.raises(PipelineInputError, match='settings differ'):
        generate(Source(source), output, replace(config, resume=True, halo=2.0))
    changed = records.copy()
    changed['z'][0] = 1
    write_pcd(source, changed)
    # Then neither stale source archive nor generated geometry is silently reused.
    with pytest.raises(PipelineInputError, match='bytes/provenance differ'):
        generate(Source(source), output, replace(config, resume=True))


def test_shared_posts_resolve_payloads_in_nested_output_directory(tmp_path, runtime):
    from pxr import Usd, UsdGeom, UsdPhysics
    from usim_env_pipeline.features import Post
    from usim_env_pipeline.structures import author_posts

    # Given a world and proxy cells in different directories.
    world = Usd.Stage.CreateNew(str(tmp_path / 'world.usda'))
    world.SetDefaultPrim(UsdGeom.Xform.Define(world, '/World').GetPrim())
    folder = tmp_path / 'authored/structures'
    folder.mkdir(parents=True)
    post = Post(4.0, 5.0, 4.5, 0.12, 350)
    # When the same generic post writer is used by scene and VTC placement.
    entries = author_posts(world, [(post, 0.0)], folder)
    world.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(tmp_path / 'world.usda'))
    # Then the bounded proxy is genuinely resolved and collidable.
    cylinder = stage.GetPrimAtPath(entries[0]['path'] + '/Post_0')
    assert cylinder and cylinder.IsA(UsdGeom.Cylinder)
    assert UsdPhysics.CollisionAPI(cylinder)
    assert entries[0]['min'][2] == pytest.approx(0.0)
    assert entries[0]['max'][2] == pytest.approx(4.5)


@pytest.mark.parametrize('infer_vegetation', [False, True])
def test_unlabelled_canopy_presentation_preserves_points_and_supported_radius(
    tmp_path, runtime, infer_vegetation
):
    from pxr import Usd, UsdGeom

    from test.test_vtc_vegetation import crown, ground
    from usim_env import GenerateRequest, get_environment
    from usim_env_pipeline.pcd import DTYPE
    from usim_env_pipeline.scene import decorate

    # Given measured ground and a crown, without semantic/species labels.
    positions = np.concatenate((ground(), crown(0)))
    records = np.zeros(len(positions), dtype=DTYPE)
    for axis, name in enumerate(('x', 'y', 'z')):
        records[name] = positions[:, axis]
    records['normal_z'][: len(ground())] = 1
    source, output = tmp_path / 'input.pcd', tmp_path / 'map'
    original = write_pcd(source, records)
    # When explicitly selecting measured-only or inferred vegetation presentation.
    if infer_vegetation:
        generate(Source(source), output, MapConfig(workers=1, normal_policy='supplied'))
        report = decorate(output)
    else:
        get_environment('tsudanuma').generate(
            GenerateRequest(source, output, MapConfig(workers=1))
        )
        report = json.loads((output / 'authored-scene.json').read_text())
    stage = Usd.Stage.Open(str(output / report['world']))

    # Then measured geometry/source survive, and any proxy fits its inferred radius.
    assert source.read_bytes() == original
    assert report['source_records'] == len(records)
    assert report['positions_unchanged']
    assert report['authored_trees'] == int(infer_vegetation)
    from usim_env_pipeline.verify import verify

    independent = verify(output)
    assert independent['ground_triangles'] == report['ground_triangles']
    if not infer_vegetation:
        assert not stage.GetPrimAtPath('/World/Vegetation')
        assert report['ground_triangles'] > 0
    else:
        tree = next(prim for prim in stage.Traverse() if prim.GetName().startswith('Tree_'))
        info = json.loads((output / 'authored/trees.json').read_text())[0]
        extent = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bounds = extent.ComputeWorldBound(tree).ComputeAlignedRange()
        radius = max(
            abs(limit[axis] - info[name])
            for limit in (bounds.GetMin(), bounds.GetMax())
            for axis, name in enumerate(('x', 'y'))
        )
        assert radius == pytest.approx(info['radius'], abs=1e-5)
