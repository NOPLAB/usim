"""MIT tooling: independent coarse survey registration from structural XY evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
import open3d as o3d

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'usim_env_vtc'
from .pcd import normals, open_records, xyz


def raster(points: NDArray[np.float64]) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Measure vertical structural support; no source vertices are edited."""
    cells = np.floor(points[:, :2] / 2).astype(np.int64)
    low, high = cells.min(axis=0), cells.max(axis=0)
    shape = tuple((high - low + 1).tolist())
    minimum = np.full(shape, np.inf, dtype=np.float32)
    maximum = np.full(shape, -np.inf, dtype=np.float32)
    indices = tuple((cells - low).T)
    np.minimum.at(minimum, indices, points[:, 2])
    np.maximum.at(maximum, indices, points[:, 2])
    mask = (np.isfinite(minimum) & ((maximum - minimum) > 1)).astype(np.float32)
    return mask, low.astype(np.float64) * 2


def main() -> None:
    """Search XY orientation/translation globally, then test a rigid 3D refinement."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    primary, _ = open_records(args.primary)
    reference, _ = open_records(args.reference)
    a, b = xyz(reference[::31]).astype(np.float64), xyz(primary[::31]).astype(np.float64)
    target, origin_target = raster(b)
    best_score = -1.0
    best_angle, best_sign = 0.0, 1
    best = np.eye(4)
    for step in (5, 0.25):
        angles = (
            np.arange(0, 360, step)
            if step == 5
            else np.arange(best_angle - 5, best_angle + 5.01, step)
        )
        signs = (1, -1) if step == 5 else (best_sign,)
        for sign in signs:
            for angle in angles:
                radians = np.deg2rad(angle)
                rotation = np.array(
                    [
                        [np.cos(radians), -np.sin(radians), 0],
                        [np.sin(radians), np.cos(radians), 0],
                        [0, 0, 1],
                    ]
                ) @ np.diag([1, sign, 1])
                source, origin_source = raster(a @ rotation.T)
                correlation = fftconvolve(target, source[::-1, ::-1], mode='full')
                location = np.unravel_index(np.argmax(correlation), correlation.shape)
                score = float(correlation[location] / np.sqrt(source.sum() * target.sum()))
                if score > best_score:
                    best_score = score
                    best_angle, best_sign = float(angle), sign
                    best[:3, :3] = rotation
                    best[:2, 3] = (
                        origin_target
                        - origin_source
                        + (np.array(location) - np.array(source.shape) + 1) * 2
                    )
                    print(
                        f'XY_CANDIDATE angle={angle} handedness={sign} '
                        f'score={score:.6f} shift={best[:2, 3].tolist()}',
                        flush=True,
                    )
    na, va = normals(reference[::31])
    nb, vb = normals(primary[::31])
    flat_a, flat_b = va & (np.abs(na[:, 2]) > 0.8), vb & (np.abs(nb[:, 2]) > 0.8)
    source_xy = a[flat_a] @ best[:3, :3].T + best[:3, 3]
    tree_xy = cKDTree(b[flat_b, :2])
    distances, nearest = tree_xy.query(source_xy[:, :2], workers=8)
    pairs = distances < 1
    best[2, 3] = np.median(b[flat_b, 2][nearest[pairs]] - source_xy[pairs, 2])
    ca = o3d.geometry.PointCloud()
    ca.points = o3d.utility.Vector3dVector(a)
    ca.normals = o3d.utility.Vector3dVector(na)
    cb = o3d.geometry.PointCloud()
    cb.points = o3d.utility.Vector3dVector(b)
    cb.normals = o3d.utility.Vector3dVector(nb)
    ca, cb = ca.voxel_down_sample(0.4), cb.voxel_down_sample(0.4)
    for distance in (3, 1, 0.4):
        fit = o3d.pipelines.registration.registration_icp(
            ca,
            cb,
            distance,
            best,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.HuberLoss(k=0.2)
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )
        best = np.array(fit.transformation)
    tree = cKDTree(xyz(primary).astype(np.float64))
    heldout = xyz(reference[17::101]).astype(np.float64)
    heldout = heldout @ best[:3, :3].T + best[:3, 3]
    residual = tree.query(heldout, workers=8)[0]
    result = {
        'reference_to_primary': best.tolist(),
        'xy_structural_score': best_score,
        'method': 'global structural-height XY FFT, then robust point-to-plane ICP',
        'analysis_sampling_only': True,
        'source_records_modified': False,
        'heldout_distance_quantiles_m': np.quantile(residual, [0.1, 0.5, 0.9, 0.99]).tolist(),
        'within_0_3m_fraction': float(np.mean(residual < 0.3)),
        'source_counts': [len(reference), len(primary)],
        'license': 'CC-BY-NC-SA-4.0',
        'accepted_for_fusion': bool(np.median(residual) < 0.15 and np.mean(residual < 0.3) >= 0.7),
        'temporal_identity_established': False,
    }
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('REGISTRATION_VALIDATED ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
