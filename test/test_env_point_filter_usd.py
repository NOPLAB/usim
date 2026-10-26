"""Real USD comparison selections retain original coordinates, colors and IDs."""

import json

import numpy as np
import pytest

from usim_env_pipeline.cloud import colors
from usim_env_pipeline.pcd import xyz
from usim_env_pipeline.point_filter_usd import publish_comparison


@pytest.mark.parametrize('selection', ['mixed', 'all', 'none'])
@pytest.mark.parametrize('color', ['intensity', 'rgb'])
@pytest.mark.parametrize('staged', [False, True])
def test_comparison_preserves_original_points_and_exact_rejection_overlay(
    tmp_path, selection, color, staged
):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdPhysics

    fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('normal_z', '<f4')]
    fields += (
        [('intensity', '<f4')] if color == 'intensity' else [('r', 'u1'), ('g', 'u1'), ('b', 'u1')]
    )
    records = np.zeros(7, dtype=fields)
    records['x'] = [61, -0.5, 29.9, 30.0, -31, 60, 1]
    records['y'] = [1, 31, 0, 0, -1, 32, 2]
    records['z'] = np.arange(7) * np.float32(0.125)
    records['normal_z'] = np.arange(7) + 1
    if color == 'intensity':
        records['intensity'] = [0, 255, 32, 128, 200, 16, 64]
    else:
        records['r'], records['g'], records['b'] = 20, np.arange(7) * 30, 240
    keep = np.array([True, False, True, False, True, False, True])
    if selection != 'mixed':
        keep[:] = selection == 'all'
    sor_keep = keep.copy()
    if staged:
        sor_keep[1] = True
    geometry_rejected = sor_keep & ~keep
    source = {
        'license': 'CC0-1.0',
        'attribution': 'Original survey',
        'official_url': 'https://example.org/survey',
        'sha256': 'a' * 64,
        'fields': [field[0] for field in fields],
    }
    report = {
        'schema': 1,
        'status': 'complete',
        'source': source,
        'frame': 'source XYZ',
        'color': color,
        'config': {'neighbors': 20, 'std_ratio': 3.0},
        'records': len(records),
        'retained_records': int(keep.sum()),
        'removed_records': int((~keep).sum()),
        'threshold_m': 3.0,
        'global_mean_m': 1.0,
        'global_std_m': 2.0 / 3.0,
    }
    np.save(tmp_path / 'records.npy', records)
    extra = {'sor_keep': sor_keep, 'geometry_rejected': geometry_rejected} if staged else {}
    np.savez(
        tmp_path / 'decisions.npz',
        ids=np.arange(len(records), dtype=np.int64),
        keep=keep,
        mean_distances=np.arange(len(records), dtype=np.float64),
        **extra,
    )
    (tmp_path / 'report.json').write_text(json.dumps(report), encoding='utf-8')
    original = (tmp_path / 'records.npy').read_bytes()

    stage = Usd.Stage.Open(str(publish_comparison(tmp_path)))

    assert stage
    assert {p.GetName() for p in stage.GetDefaultPrim().GetChildren()} == {
        'Before',
        'After',
        'Removed',
    }
    for mode in ('Before', 'After', 'Removed'):
        group = stage.GetPrimAtPath(f'/World/{mode}')
        assert UsdGeom.Imageable(group).ComputeVisibility() == (
            UsdGeom.Tokens.inherited if mode == 'After' else UsdGeom.Tokens.invisible
        )
        seen = []
        for prim in Usd.PrimRange(group):
            assert not prim.IsA(UsdGeom.Mesh)
            assert not prim.HasAPI(UsdPhysics.CollisionAPI)
            if not prim.IsA(UsdGeom.Points):
                continue
            points = UsdGeom.Points(prim)
            ids = np.asarray(points.GetIdsAttr().Get(), dtype=np.int64)
            assert len(ids)
            seen.extend(ids.tolist())
            actual_xyz = np.asarray(points.GetPointsAttr().Get())
            np.testing.assert_array_equal(actual_xyz, xyz(records[ids]))
            assert np.ptp(actual_xyz[:, 0]) <= 30
            assert np.ptp(actual_xyz[:, 1]) <= 30
            np.testing.assert_array_equal(
                points.GetWidthsAttr().Get(), np.array([0.055], dtype=np.float32)
            )
            assert points.GetWidthsInterpolation() == UsdGeom.Tokens.constant
            expected_colors = colors(records[ids])
            if mode == 'Removed':
                expected_colors[:] = 0.45
                expected_colors[~sor_keep[ids]] = (1, 0, 0)
                expected_colors[geometry_rejected[ids]] = (1, 1, 0)
            np.testing.assert_array_equal(points.GetDisplayColorPrimvar().Get(), expected_colors)
            assert not points.GetNormalsAttr().HasAuthoredValueOpinion()
        np.testing.assert_array_equal(
            sorted(seen), np.flatnonzero(keep) if mode == 'After' else np.arange(len(records))
        )
        standalone = Usd.Stage.Open(str(tmp_path / f'{mode.lower()}.usda'))
        assert (
            UsdGeom.Imageable(standalone.GetPrimAtPath(f'/World/{mode}')).ComputeVisibility()
            == UsdGeom.Tokens.inherited
        )
        metadata = standalone.GetRootLayer().customLayerData
        assert metadata['license'] == source['license']
        assert metadata['attribution'] == source['attribution']
        assert metadata['frame'] == report['frame']
        assert json.loads(metadata['sourceMetadata']) == source
    assert (tmp_path / 'records.npy').read_bytes() == original
