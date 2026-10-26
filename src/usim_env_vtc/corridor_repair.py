"""Derive a finite native-terrain corridor using source heights, never PCD low quantiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from usim_env_vtc.repair_terrain import author_road_mesh


def smooth_profile(
    profile: dict[str, Any], sigma: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Smooth native longitudinal relief while retaining both original endpoint levels."""
    samples = np.asarray(profile['samples_xy_z'], dtype=np.float64)
    start, end = np.asarray(profile['start']), np.asarray(profile['end'])
    length = np.linalg.norm(end - start)
    if (
        samples.ndim != 2
        or samples.shape[1] != 3
        or len(samples) < 3
        or not np.isfinite(samples).all()
        or not np.isfinite(sigma)
        or sigma <= 0
        or not np.isfinite(length)
        or length <= 0
    ):
        raise ValueError('A finite native XYZ profile and positive sigma are required')
    along = (samples[:, :2] - start) @ ((end - start) / length)
    if (
        np.any(np.diff(along) <= 0)
        or abs(along[0]) > 1e-5
        or abs(along[-1] - length) > 1e-5
        or np.max(np.diff(along)) > 0.11
        or np.max(
            np.linalg.norm(
                samples[:, :2] - start - along[:, None] * (end - start) / length, axis=1
            )
        )
        > 1e-5
    ):
        raise ValueError('Native profile must cover the straight corridor at <= 0.11 m spacing')
    z = np.asarray(
        gaussian_filter1d(samples[:, 2], sigma / np.mean(np.diff(along)), mode='nearest'),
        dtype=np.float64,
    )
    # Remove the endpoint residual, not the longitudinal relief.
    z += np.interp(along, np.array([0.0, length]), samples[[0, -1], 2] - z[[0, -1]])
    return along, z


def corridor_coordinates(xy, start, end):
    start, end = np.asarray(start), np.asarray(end)
    direction = (end - start) / np.linalg.norm(end - start)
    offset = xy - start
    return offset @ direction, np.abs(offset @ np.array([-direction[1], direction[0]]))


def repair_points(points, profile, along, heights, width, blend):
    """Blend only Z inside a finite rectangle; exact boundary and exterior remain unchanged."""
    if not np.isfinite([width, blend]).all() or width <= 0 or blend <= 0:
        raise ValueError('Width and blend must be positive and finite')
    distance, lateral = corridor_coordinates(points[:, :2], profile['start'], profile['end'])
    length = along[-1]
    fade = np.clip((width / 2 + blend - lateral) / blend, 0, 1)
    # Put endpoint joins outside the requested crossing, so native crossfall at
    # either end cannot leak an unrepaired step into its full-width core.
    fade *= np.clip((distance + blend) / blend, 0, 1)
    fade *= np.clip((length + blend - distance) / blend, 0, 1)
    # Smoothstep gives zero correction and zero correction derivative at outer joins.
    weight = fade * fade * (3 - 2 * fade)
    corrected = points.copy()
    active = weight > 0
    target = np.interp(distance[active], along, heights)
    corrected[active, 2] += weight[active] * (target - points[active, 2])
    core = (lateral <= width / 2) & (distance >= 0) & (distance <= length)
    return corrected, core, active


def edge_metrics(mesh, points, mask) -> dict[str, Any]:
    """Measure actual mesh edges rather than sorted unrelated vertices."""
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    if not len(indices):
        return {'maximum_step_m': 0.0, 'maximum_slope': 0.0, 'edges': 0}
    ends = np.cumsum(counts)
    following = np.arange(len(indices)) + 1
    following[ends - 1] -= counts
    a, b = indices, indices[following]
    run = np.linalg.norm(points[b, :2] - points[a, :2], axis=1)
    selected = mask[a] & mask[b] & (run > 1e-8)
    rise = np.abs(points[b, 2] - points[a, 2])[selected]
    return {
        'maximum_step_m': float(rise.max(initial=0)),
        'maximum_slope': float((rise / run[selected]).max(initial=0)),
        'edges': int(selected.sum()),
    }


def seam_metrics(points) -> dict[str, Any]:
    """Check duplicated component/tile border vertices at 1 mm XY grouping precision."""
    _, inverse, counts = np.unique(
        np.rint(points[:, :2] * 1000).astype(np.int64),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    low, high = np.full(len(counts), np.inf), np.full(len(counts), -np.inf)
    np.minimum.at(low, inverse, points[:, 2])
    np.maximum.at(high, inverse, points[:, 2])
    duplicate = counts > 1
    return {
        'duplicate_xy_groups': int(duplicate.sum()),
        'maximum_seam_gap_m': float((high[duplicate] - low[duplicate]).max(initial=0)),
    }


def repair_world(
    world: Path,
    index: Path,
    profile_path: Path,
    out: Path,
    *,
    width=3.5,
    blend=3.0,
    sigma=2.5,
    terrain_root='/World/Terrain',
) -> dict[str, Any]:
    """Author points/extent only, inheriting all payload roots, topology and collision APIs."""
    from pxr import Usd, UsdGeom

    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    if Path(profile['source']).resolve() != world.resolve():
        raise ValueError('Native profile source must match the explicit source world')
    along, heights = smooth_profile(profile, sigma)
    # Validate the corridor parameters before loading or writing any assets.
    repair_points(np.zeros((0, 3)), profile, along, heights, width, blend)
    if out.resolve() == world.parent.resolve():
        raise ValueError('Output must be separate from source world')
    entries = json.loads(index.read_text(encoding='utf-8'))
    source = Usd.Stage.Open(str(world.resolve()), load=Usd.Stage.LoadNone)
    low = np.minimum(profile['start'], profile['end']) - width / 2 - blend - 0.2
    high = np.maximum(profile['start'], profile['end']) + width / 2 + blend + 0.2
    meshes = []
    selected_entries = []
    for entry in entries:
        if not entry['path'].startswith(terrain_root.rstrip('/') + '/'):
            continue
        if np.any(np.asarray(entry['max'][:2]) < low) or np.any(entry['min'][:2] > high):
            continue
        selected_entries.append(entry)
        source.Load(entry['path'])
        for prim in Usd.PrimRange(source.GetPrimAtPath(entry['path'])):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            local = np.asarray(mesh.GetPointsAttr().Get()).copy()
            matrix = np.asarray(
                UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            )
            # Z-only local opinions cannot preserve XY for a tilted local Z axis.
            if not np.allclose(matrix[2, :2], 0, atol=1e-12) or abs(matrix[2, 2]) < 1e-12:
                raise ValueError(f'Unsupported tilted terrain Z axis: {prim.GetPath()}')
            points = local @ matrix[:3, :3] + matrix[3, :3]
            if np.any(points[:, :2].max(0) < low) or np.any(points[:, :2].min(0) > high):
                continue
            fixed, core, active = repair_points(points, profile, along, heights, width, blend)
            meshes.append((entry, mesh, local, matrix, points, fixed, core, active))
    if not meshes:
        raise ValueError('No native terrain meshes intersect the corridor')
    originals = np.concatenate([item[4] for item in meshes])
    fixed_all = np.concatenate([item[5] for item in meshes])
    tree = cKDTree(originals[:, :2])
    distances, nearest = tree.query(np.asarray(profile['samples_xy_z'])[:, :2])
    profile_error = np.max(
        np.abs(originals[nearest, 2] - np.asarray(profile['samples_xy_z'])[:, 2])
    )
    if distances.max() > 0.08 or profile_error > 0.04:
        raise ValueError('Native source profile is stale or lacks <= 8 cm terrain coverage')
    before_seams, after_seams = seam_metrics(originals), seam_metrics(fixed_all)
    if after_seams['maximum_seam_gap_m'] > before_seams['maximum_seam_gap_m'] + 1e-5:
        raise ValueError('Correction would increase a native terrain seam')
    out.mkdir(parents=True, exist_ok=True)
    correction = Usd.Stage.CreateNew(str((out / 'corridor-correction.usdc').resolve()))
    changed_vertices, changed_meshes = 0, []
    metrics = {'before': [], 'after': [], 'blend_after': []}
    for entry, mesh, local, matrix, points, fixed, core, active in meshes:
        metrics['before'].append(edge_metrics(mesh, points, core))
        requested = fixed[:, 2] != points[:, 2]
        local[requested, 2] += (fixed[requested, 2] - points[requested, 2]) / matrix[2, 2]
        # Report the float32 opinions actually authored, not ideal float64 targets.
        fixed[:, 2] = (local @ matrix[:3, :3] + matrix[3, :3])[:, 2]
        metrics['after'].append(edge_metrics(mesh, fixed, core))
        metrics['blend_after'].append(edge_metrics(mesh, fixed, active))
        changed = local[:, 2] != np.asarray(mesh.GetPointsAttr().Get())[:, 2]
        if not changed.any():
            continue
        path = str(mesh.GetPath())
        author_road_mesh(correction, path, local)
        changed_vertices += int(changed.sum())
        changed_meshes.append((path, local))
        entry['min'][2] = min(entry['min'][2], float(fixed[:, 2].min()))
        entry['max'][2] = max(entry['max'][2], float(fixed[:, 2].max()))
    fixed_all = np.concatenate([item[5] for item in meshes])
    after_seams = seam_metrics(fixed_all)
    if after_seams['maximum_seam_gap_m'] > before_seams['maximum_seam_gap_m'] + 1e-5:
        raise ValueError('Float32 correction would increase a native terrain seam')
    correction.GetRootLayer().Save()
    derived = Usd.Stage.CreateNew(str((out / 'world.usda').resolve()), load=Usd.Stage.LoadNone)
    derived.GetRootLayer().subLayerPaths = ['corridor-correction.usdc', world.resolve().as_posix()]
    derived.SetDefaultPrim(derived.GetPrimAtPath('/World'))
    derived.GetRootLayer().customLayerData = {
        'originalAssetsModified': False,
        'geographicSurveyAccuracyEstablished': False,
        'corridorMethod': 'native longitudinal Gaussian smoothing with finite blended joins',
    }
    derived.GetRootLayer().Save()
    # Index assets remain provenance; residency consumes the original paths and bounds.
    for entry in entries:
        if 'asset' in entry:
            entry['asset'] = (index.parent / entry['asset']).resolve().as_posix()
    (out / 'world.index.json').write_text(json.dumps(entries, indent=2), encoding='utf-8')
    reopened = Usd.Stage.Open(str((out / 'world.usda').resolve()), load=Usd.Stage.LoadNone)
    for entry in selected_entries:
        reopened.Load(entry['path'])
    for path, expected in changed_meshes:
        actual = np.asarray(UsdGeom.Mesh(reopened.GetPrimAtPath(path)).GetPointsAttr().Get())
        if not np.array_equal(actual, expected):
            raise AssertionError(f'Stronger correction did not compose at {path}')
    numeric = {
        name: {
            'maximum_step_m': max(item['maximum_step_m'] for item in values),
            'maximum_slope': max(item['maximum_slope'] for item in values),
            'edges': sum(item['edges'] for item in values),
        }
        for name, values in metrics.items()
    }
    report = {
        'source_world': world.resolve().as_posix(),
        'source_index': index.resolve().as_posix(),
        'profile': profile_path.resolve().as_posix(),
        'profile_sha256': hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        'method': 'source-derived longitudinal Gaussian smoothing; not surveyed accuracy',
        'measured_profile_used': False,
        'measured_profile_rejection': (
            'Registered 2019 horizontal PCD quantiles disagree with native terrain; '
            'ground identity and vertical registration are insufficient for height replacement'
        ),
        'measured_horizontal_points': profile.get('official_2019_horizontal_points'),
        'source_height_range_m': [float(originals[:, 2].min()), float(originals[:, 2].max())],
        'smoothed_profile_range_m': [float(heights.min()), float(heights.max())],
        'profile_endpoints_m': heights[[0, -1]].tolist(),
        'start': profile['start'],
        'end': profile['end'],
        'width_m': width,
        'blend_m': blend,
        'longitudinal_support_m': [-blend, float(along[-1] + blend)],
        'sigma_m': sigma,
        'payload_roots': len(entries),
        'changed_vertices': changed_vertices,
        'changed_meshes': len(changed_meshes),
        'maximum_correction_m': float(np.abs(fixed_all[:, 2] - originals[:, 2]).max()),
        'profile_source_maximum_error_m': float(profile_error),
        'outside_corridor_unchanged': True,
        'xy_topology_collision_materials_preserved': True,
        'composition_verified': True,
        'seams_before': before_seams,
        'seams_after': after_seams,
        'mesh_edges': numeric,
        'core_limits_pass': (
            numeric['after']['maximum_slope'] <= 0.25
            and numeric['after']['maximum_step_m'] <= 0.04
        ),
        'limits_scope': (
            'Mesh edges with both endpoints in the full-width crossing from start to end; '
            f'{blend:g} m endpoint joins are outside this core'
        ),
        'lateral_blend_limits_pass': (
            numeric['blend_after']['maximum_slope'] <= 0.25
            and numeric['blend_after']['maximum_step_m'] <= 0.04
        ),
    }
    (out / 'corridor-repair.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('world', 'index', 'profile', 'out'):
        parser.add_argument(
            '--' + name,
            dest='profile_path' if name == 'profile' else name,
            type=Path,
            required=True,
        )
    parser.add_argument('--width', type=float, default=3.5, help='Full core width in metres')
    parser.add_argument(
        '--blend', type=float, default=3.0, help='Lateral and endpoint joins in metres'
    )
    parser.add_argument(
        '--sigma', type=float, default=2.5, help='Source profile Gaussian sigma, metres'
    )
    parser.add_argument('--terrain-root', default='/World/Terrain')
    args = parser.parse_args()
    print(json.dumps(repair_world(**vars(args)), indent=2), flush=True)


if __name__ == '__main__':
    main()
