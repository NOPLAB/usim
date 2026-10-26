"""Real USD modes share original locations and independently color inclusive thresholds."""

import json
from pathlib import Path

import numpy as np
import pytest

from usim_env_pipeline.voxel_threshold_usd import publish_threshold_comparison


def test_five_modes_preserve_shared_points_and_exact_colors(tmp_path: Path):
    # Given original records, one SOR reject, and frozen per-cell populations.
    _ = pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom

    records = np.zeros(7, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    records['x'] = [0, 0.1, 1, 31, -31, 65, 66]
    sor = np.array([False, True, True, True, True, True, True])
    counts = np.array([0, 5, 6, 10, 21, 40, 41], dtype=np.int64)
    colors = [(1, 0, 0), (1, 0.5, 0), (1, 1, 0), (0, 1, 0), (0, 0.4, 1)]
    modes = [
        {
            'name': 'SOR' if limit == 0 else f'Voxel{limit}',
            'threshold': limit,
            'color': color,
            'retained_records': int((sor & (counts > limit)).sum()),
        }
        for limit, color in zip((0, 5, 10, 20, 40), colors)
    ]
    report = {
        'source': {'license': 'CC0', 'attribution': 'Survey', 'official_url': ''},
        'frame': 'source XYZ; meters; Z-up',
        'modes': modes,
    }
    np.save(tmp_path / 'records.npy', records)
    np.savez(
        tmp_path / 'decisions.npz',
        ids=np.arange(7, dtype=np.int64),
        sor_keep=sor,
        counts_per_record=counts,
    )
    (tmp_path / 'report.json').write_text(json.dumps(report))
    # When publishing real USD modes with a single shared original geometry set.
    stage = Usd.Stage.Open(str(publish_threshold_comparison(tmp_path)))
    # Then every mode shows all original IDs and only its own inclusive threshold colors.
    assert stage
    for mode in modes:
        seen = []
        group = stage.GetPrimAtPath(f'/World/{mode["name"]}')
        for prim in Usd.PrimRange(group):
            if not prim.IsA(UsdGeom.Points):
                continue
            points = UsdGeom.Points(prim)
            ids = np.asarray(points.GetIdsAttr().Get(), dtype=np.int64)
            seen.extend(ids.tolist())
            expected = np.full((len(ids), 3), 0.45, dtype=np.float32)
            expected[~sor[ids]] = (1, 0, 0)
            expected[sor[ids] & (counts[ids] <= mode['threshold'])] = mode['color']
            np.testing.assert_array_equal(points.GetDisplayColorPrimvar().Get(), expected)
            actual = np.asarray(points.GetPointsAttr().Get())
            np.testing.assert_array_equal(
                actual, np.column_stack([records[name][ids] for name in ('x', 'y', 'z')])
            )
        np.testing.assert_array_equal(sorted(seen), np.arange(7))
    assert len(list((tmp_path / 'comparison-points/base').glob('*.usdc'))) == 4
