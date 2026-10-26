"""Source-preserving reconstruction regressions, with optional CPU geometry checks."""

import argparse
import hashlib
import json

import numpy as np
import pytest

from test.env_pipeline_fixtures import plane_records, write_pcd
from usim_env_pipeline import MapConfig, Source, generate
from usim_env_pipeline.cloud import colors
from usim_env_pipeline.models import PipelineInputError
from usim_env_pipeline.pcd import xyz


@pytest.fixture
def geometry():
    pytest.importorskip('open3d')
    pytest.importorskip('scipy')
    return pytest.importorskip('pxr')


def assert_archival_and_usd(output, records):
    from pxr import Usd, UsdGeom, UsdPhysics

    seen, cross_boundary = [], False
    reports = [json.loads(p.read_text()) for p in sorted((output / 'reports').glob('*.json'))]
    manifest = json.loads((output / 'manifest.json').read_text())
    for report in reports:
        name = report['name']
        with np.load(output / 'records' / f'{name}.npz') as archive:
            ids, regional = archive['ids'], archive['records']
            assert ids.dtype == np.int64
            assert regional.dtype == records.dtype
            assert regional.tobytes() == records[ids].tobytes()
            seen.extend(ids.tolist())
            np.testing.assert_array_equal(
                np.floor(xyz(regional)[:, :2] / manifest['tile_size']),
                np.broadcast_to(report['key'], (len(ids), 2)),
            )
            stage = Usd.Stage.Open(str(output / 'points' / f'{name}.usdc'))
            points = UsdGeom.Points(stage.GetPrimAtPath('/Tile/Points'))
            np.testing.assert_array_equal(points.GetIdsAttr().Get(), ids)
            np.testing.assert_array_equal(points.GetPointsAttr().Get(), xyz(regional))
            assert not points.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
        with np.load(output / 'meshes' / f'{name}.npz') as mesh_archive:
            vertex_ids, faces = mesh_archive['ids'], mesh_archive['faces']
            assert len(faces) == report['triangles']
            if not len(faces):
                assert not (output / 'meshes' / f'{name}.usdc').exists()
                continue
            stage = Usd.Stage.Open(str(output / 'meshes' / f'{name}.usdc'))
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath('/Tile/Surface'))
            prim = mesh.GetPrim()
            np.testing.assert_array_equal(prim.GetAttribute('source:recordIds').Get(), vertex_ids)
            np.testing.assert_array_equal(mesh.GetPointsAttr().Get(), xyz(records[vertex_ids]))
            np.testing.assert_array_equal(mesh.GetFaceVertexCountsAttr().Get(), 3)
            np.testing.assert_array_equal(
                np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), faces
            )
            assert prim.HasAPI(UsdPhysics.CollisionAPI)
            assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'none'
            triangles = xyz(records[vertex_ids])[faces]
            low = np.asarray(report['key']) * manifest['tile_size']
            centroid = triangles.astype(np.float64).mean(axis=1)
            assert (
                (centroid[:, :2] >= low) & (centroid[:, :2] < low + manifest['tile_size'])
            ).all()
            sides = np.floor(triangles[:, :, 0] / manifest['tile_size'])
            cross_boundary |= bool((sides.min(axis=1) != sides.max(axis=1)).any())
    assert sorted(seen) == list(range(len(records)))
    mesh_index = json.loads((output / 'index.json').read_text())
    point_index = json.loads((output / 'point-index.json').read_text())
    world_index = json.loads((output / 'world-index.json').read_text())
    assert len(mesh_index) == sum(bool(r['triangles']) for r in reports)
    assert len(point_index) == len(reports)
    assert world_index == mesh_index + point_index
    for entry in world_index:
        assert (output / entry['payload']).is_file()
    for layer in ('streamed.usda', 'points.usda', 'world.usda', 'archival-points.usda'):
        assert Usd.Stage.Open(str(output / layer))
    return reports, cross_boundary


def test_supplied_generation_archives_every_record_and_keeps_reference_unfused(tmp_path, geometry):
    records = plane_records(invalid_core=True)
    source_path, reference = tmp_path / 'survey.pcd', tmp_path / 'reference.pcd'
    source_bytes = write_pcd(source_path, records, padding=True)
    reference_records = plane_records()[:3].copy()
    reference_records['x'] += 1000
    reference_bytes = write_pcd(reference, reference_records)
    output = tmp_path / 'map'
    output.mkdir()
    source = Source(
        source_path,
        license='CC0-1.0',
        attribution='Test survey',
        expected_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )

    summary = generate(
        source,
        output,
        MapConfig(
            workers=1, normal_policy='supplied', references=(Source(reference, key='reference'),)
        ),
    )

    assert summary['records'] == len(records)
    assert summary['triangles'] > 0
    assert summary['normal_invalid'] == 3
    assert summary['estimated_normal_records'] == 0
    assert summary == json.loads((output / 'summary.json').read_text())
    reports, crossing = assert_archival_and_usd(output, records)
    assert crossing
    point_only = next(r for r in reports if r['key'] == [2, 0])
    assert point_only['records'] == 3
    assert point_only['triangles'] == 0
    assert point_only['name'] in summary['unmeshed_tiles']
    manifest = json.loads((output / 'manifest.json').read_text())
    assert set(manifest['sources']) == {'primary', 'reference'}
    assert manifest['license'] == 'CC0-1.0'
    assert manifest['frame'] == 'source XYZ'
    assert (output / manifest['sources']['primary']['path']).read_bytes() == source_bytes
    assert (output / manifest['sources']['reference']['path']).read_bytes() == reference_bytes
    assert manifest['attribution'] == 'Test survey'
    assert manifest['source_catalog'] == ''
    assert manifest['sources']['primary']['official_url'] == ''
    assert manifest['sources']['reference']['license'] == 'unspecified'


def test_xyz_only_ascii_generation_estimates_normals_without_changing_records(tmp_path, geometry):
    supplied = plane_records()
    records = np.empty(len(supplied), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    for field in ('x', 'y', 'z'):
        records[field] = supplied[field]
    path, output = tmp_path / 'xyz.pcd', tmp_path / 'map'
    original = write_pcd(path, records, encoding='ascii')

    summary = generate(Source(path), output, MapConfig(workers=1))

    assert summary['triangles'] > 0
    assert summary['estimated_normal_records'] == len(records)
    assert_archival_and_usd(output, records)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert (output / manifest['sources']['primary']['path']).read_bytes() == original


def test_auto_policy_estimates_only_missing_normals_in_a_mixed_cloud(tmp_path, geometry):
    records = plane_records()
    records['normal_z'][len(records) // 2] = 0
    path, output = tmp_path / 'mixed.pcd', tmp_path / 'map'
    write_pcd(path, records)

    summary = generate(Source(path), output, MapConfig(workers=1))

    assert summary['triangles'] > 0
    assert summary['estimated_normal_records'] == 1
    assert summary['normal_valid'] == len(records) - 1
    assert summary['normal_invalid'] == 1
    assert_archival_and_usd(output, records)


def test_ply_source_rgb_and_source_archive_survive_generation(tmp_path, geometry):
    from pxr import Usd, UsdGeom

    positions = xyz(plane_records())
    rgb = np.tile(
        np.array([[255, 0, 64], [0, 128, 255]], dtype=np.uint8), (len(positions) // 2 + 1, 1)
    )[: len(positions)]
    path = tmp_path / 'colored.ply'
    header = (
        'ply\nformat ascii 1.0\n'
        f'element vertex {len(positions)}\nproperty float x\nproperty float y\n'
        'property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n'
        'end_header\n'
    )
    rows = [' '.join(map(str, [*position, *color])) for position, color in zip(positions, rgb)]
    path.write_text(header + '\n'.join(rows) + '\n', encoding='ascii')
    original, output = path.read_bytes(), tmp_path / 'map'

    summary = generate(Source(path), output, MapConfig(workers=1))

    assert summary['triangles'] > 0
    manifest = json.loads((output / 'manifest.json').read_text())
    assert (output / manifest['sources']['primary']['path']).read_bytes() == original
    for archive_path in (output / 'records').glob('*.npz'):
        with np.load(archive_path) as archive:
            ids = archive['ids']
            np.testing.assert_allclose(colors(archive['records']), rgb[ids] / 255, atol=1e-7)
            stage = Usd.Stage.Open(str(output / 'points' / f'{archive_path.stem}.usdc'))
            points = UsdGeom.Points(stage.GetPrimAtPath('/Tile/Points'))
            np.testing.assert_allclose(
                points.GetDisplayColorPrimvar().Get(), rgb[ids] / 255, atol=1e-7
            )


@pytest.mark.parametrize('damage', ['hash', 'occupied', 'empty', 'nonfinite'])
def test_generation_rejects_invalid_inputs_without_overwriting_output(tmp_path, damage):
    records = plane_records()
    if damage == 'empty':
        records = records[:0]
    elif damage == 'nonfinite':
        records['x'][0] = np.nan
    path, output = tmp_path / 'survey.pcd', tmp_path / 'map'
    original = write_pcd(path, records)
    source = Source(path, expected_sha256='0' * 64) if damage == 'hash' else Source(path)
    if damage == 'occupied':
        output.mkdir()
        (output / 'keep.txt').write_text('untouched', encoding='ascii')

    with pytest.raises(PipelineInputError):
        generate(source, output, MapConfig(workers=1))

    assert path.read_bytes() == original
    if damage == 'occupied':
        assert (output / 'keep.txt').read_text() == 'untouched'
        assert sorted(p.name for p in output.iterdir()) == ['keep.txt']


def test_vtc_delegate_output_passes_independent_verifier(tmp_path, geometry):
    from usim_env_vtc.reconstruct import run
    from usim_env_vtc.verify_map import verify

    records = plane_records(invalid_core=True)
    path, reference, output = tmp_path / 'survey.pcd', tmp_path / 'older.pcd', tmp_path / 'map'
    original = write_pcd(path, records, padding=True)
    reference_bytes = write_pcd(reference, plane_records()[:3])

    run(
        argparse.Namespace(
            source=path,
            reference_2018=reference,
            output=output,
            tile_size=30.0,
            halo=1.5,
            workers=1,
        )
    )
    result = verify(output)

    assert result['records'] == len(records)
    assert result['triangles'] > 0
    reports, crossing = assert_archival_and_usd(output, records)
    assert crossing
    assert any(r['records'] == 3 and r['triangles'] == 0 for r in reports)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert set(manifest['sources']) == {'2019', '2018'}
    assert manifest['license'] == 'CC-BY-NC-SA-4.0'
    assert (output / manifest['sources']['2019']['path']).read_bytes() == original
    assert (output / manifest['sources']['2018']['path']).read_bytes() == reference_bytes


@pytest.mark.parametrize('encoding', ['ascii', 'binary'])
def test_strict_vtc_source_rejects_non_vtc_records(tmp_path, encoding):
    from usim_env_vtc.reconstruct import run

    records = plane_records()
    if encoding == 'binary':
        records = np.zeros(3, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    path = tmp_path / 'not-vtc.pcd'
    write_pcd(path, records, encoding=encoding)

    with pytest.raises(PipelineInputError):
        run(
            argparse.Namespace(
                source=path,
                reference_2018=None,
                output=tmp_path / 'map',
                tile_size=30.0,
                halo=1.5,
                workers=1,
            )
        )
