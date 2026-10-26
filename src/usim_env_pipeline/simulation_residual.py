"""Retain complete original faces wherever the simulation replacements cannot cover them."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .simulation_coverage import occupied_bounds, terrain_covers, voxels_cover
from .simulation_ground import GroundConfig, Terrain
from .simulation_obstacles import ObstacleConfig, Occupancy
from .simulation_output import SimulationTile
from .simulation_planes import PlanarProxy
from .usd import Geometry, LayerMetadata, author_geometry


def publish_residual(
    output: Path,
    terrain: Terrain,
    obstacles: Occupancy,
    ground_config: GroundConfig,
    obstacle_config: ObstacleConfig,
    metadata: LayerMetadata,
    noise_ids: NDArray[np.int64],
    proxies: list[PlanarProxy] | None = None,
    ground_ids: NDArray[np.int64] | None = None,
) -> tuple[list[SimulationTile], int, int, int]:
    """Return payloads and original/replaced face counts using whole-footprint tests."""
    from pxr import Usd, UsdGeom

    folder = output / 'simulation-models'
    (folder / 'residual').mkdir(parents=True, exist_ok=True)
    (folder / 'face-accounting').mkdir(parents=True, exist_ok=True)
    entries: list[SimulationTile] = []
    models = json.loads((output / 'building-models/models.json').read_text(encoding='utf-8'))
    proxy_bounds = [
        (proxy, proxy.positions.min(0), proxy.positions.max(0)) for proxy in proxies or []
    ]
    total, replaced, discarded = 0, 0, 0
    source_entries = json.loads((output / 'authored-index.json').read_text(encoding='utf-8'))
    for entry in source_entries:
        if not entry['path'].startswith('/World/Regions/'):
            continue
        source = Usd.Stage.Open(str(output / entry['payload']))
        mesh = UsdGeom.Mesh.Get(source, '/Tile/Surface')
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
        ids = np.asarray(mesh.GetPrim().GetAttribute('source:recordIds').Get(), dtype=np.int64)
        triangles = points[faces].astype(np.float64)
        tile_lower, tile_upper = points.min(0), points.max(0)
        reason = np.zeros(len(faces), dtype=np.uint8)
        reason[
            terrain_covers(triangles, terrain, ground_config, obstacle_config.ground_tolerance)
        ] = 1
        # Ground authority is intentionally smoothed, not retained as noisy BPA or voxels.
        if ground_ids is not None:
            reason[np.isin(ids, ground_ids)[faces].all(1)] = 1
        reason[voxels_cover(triangles, obstacles.keys, obstacle_config.cell)] = 3
        for proxy, proxy_lower, proxy_upper in proxy_bounds:
            if (tile_upper < proxy_lower - 1e-5).any() or (tile_lower > proxy_upper + 1e-5).any():
                continue
            possible = ((triangles >= proxy_lower - 1e-5) & (triangles <= proxy_upper + 1e-5)).all(
                (1, 2)
            )
            selected = np.flatnonzero(possible)
            if not len(selected):
                continue
            local = np.stack(
                (triangles[selected] @ proxy.u, triangles[selected] @ proxy.v), axis=2
            )
            lower = np.floor(local.min(1) / obstacle_config.cell).astype(np.int64)
            upper = np.floor(np.nextafter(local.max(1), -np.inf) / obstacle_config.cell).astype(
                np.int64
            )
            covered = occupied_bounds(lower, np.maximum(lower, upper), proxy.keys)
            covered &= (
                np.abs(triangles[selected] @ proxy.normal + proxy.offset) <= 0.08 + 1e-5
            ).all(1)
            reason[selected[covered]] = 4
        for model in models:
            center, size, axes = map(np.asarray, (model['center'], model['size'], model['axes']))
            extent = np.abs(axes) @ (size / 2 + obstacle_config.building_margin)
            if (tile_upper < center - extent).any() or (tile_lower > center + extent).any():
                continue
            possible = ((triangles >= center - extent) & (triangles <= center + extent)).all(
                (1, 2)
            )
            selected = np.flatnonzero(possible)
            local = np.abs((triangles[selected] - center) @ axes) - size / 2
            inside = (
                np.linalg.norm(np.maximum(local, 0), axis=2) <= obstacle_config.building_margin
            ).all(1)
            reason[selected[inside]] = 2
        # A discarded noisy return must not reappear through an old BPA triangle.
        reason[np.isin(ids, noise_ids)[faces].any(1)] = 5
        name = entry['path'].rsplit('/', 1)[1]
        np.savez_compressed(folder / 'face-accounting' / f'{name}.npz', reason=reason)
        total += len(faces)
        replaced += int(((reason > 0) & (reason < 5)).sum())
        discarded += int((reason == 5).sum())
        retained = np.flatnonzero(reason == 0)
        if not len(retained):
            continue
        used, inverse = np.unique(faces[retained], return_inverse=True)
        path = folder / 'residual' / f'{name}.usdc'
        author_geometry(
            path,
            Geometry(
                points[used],
                ids[used],
                np.tile(np.array([[0.45, 0.43, 0.38]], dtype=np.float32), (len(used), 1)),
                inverse.reshape(-1, 3).astype(np.int32),
            ),
            metadata,
        )
        np.savez_compressed(
            path.with_suffix('.npz'),
            source_payload=np.asarray(entry['payload']),
            source_face_indices=retained,
            source_vertex_indices=used,
        )
        entries.append(
            {
                'path': f'/World/SimulationUnresolved/{name}',
                'payload': f'./simulation-models/residual/{name}.usdc',
                'kind': 'unresolved',
                'faces': len(retained),
            }
        )
    return entries, total, replaced, discarded
