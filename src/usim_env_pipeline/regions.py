"""Independent halo-backed regional reconstruction jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .cloud import colors
from .meshing import Neighborhood, reconstruct
from .models import MapConfig, RegionReport
from .pcd import normals, xyz
from .storage import tile_name, write_json
from .usd import Geometry, LayerMetadata, author_geometry


@dataclass(frozen=True, slots=True)
class RegionJob:
    output: Path
    key: tuple[int, int]
    occupied: frozenset[tuple[int, int]]
    config: MapConfig
    metadata: LayerMetadata


def region_job(job: RegionJob) -> RegionReport:
    output, key, config = job.output, job.key, job.config
    name = tile_name(key)
    low = np.array(key) * config.tile_size
    high = low + config.tile_size
    with np.load(output / 'records' / f'{name}.npz') as own:
        own_records, own_ids = own['records'], own['ids']
    own_xyz = xyz(own_records)
    _, own_valid = normals(own_records)
    author_geometry(
        output / 'points' / f'{name}.usdc',
        Geometry(own_xyz, own_ids, colors(own_records)),
        job.metadata,
    )
    records: list[NDArray[np.void]] = []
    ids: list[NDArray[np.int64]] = []
    for x in range(key[0] - 1, key[0] + 2):
        for y in range(key[1] - 1, key[1] + 2):
            if (x, y) not in job.occupied:
                continue
            with np.load(output / 'records' / f'{tile_name((x, y))}.npz') as archive:
                positions = xyz(archive['records'])
                inside = (
                    (positions[:, :2] >= low - config.halo)
                    & (positions[:, :2] < high + config.halo)
                ).all(axis=1)
                records.append(archive['records'][inside])
                ids.append(archive['ids'][inside])
    combined_records, combined_ids = np.concatenate(records), np.concatenate(ids)
    order = np.argsort(combined_ids)
    neighborhood = Neighborhood(combined_records[order], combined_ids[order], key)
    surface = reconstruct(neighborhood, config)
    np.savez_compressed(
        output / 'meshes' / f'{name}.npz',
        ids=surface.ids,
        faces=surface.faces,
        normals=surface.normals,
    )
    all_edges = np.sort(surface.faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, multiplicities = np.unique(all_edges, axis=0, return_counts=True)
    if len(surface.faces):
        author_geometry(
            output / 'meshes' / f'{name}.usdc',
            Geometry(
                surface.positions,
                surface.ids,
                colors(neighborhood.records[surface.record_indices]),
                surface.faces,
            ),
            job.metadata,
        )
    core_count = int(np.isin(surface.ids, own_ids).sum())
    report: RegionReport = {
        'name': name,
        'key': list(key),
        'records': len(own_ids),
        'normal_valid': int(own_valid.sum()),
        'normal_invalid': int((~own_valid).sum()),
        'halo_records': len(neighborhood.ids),
        'candidate_owned': surface.candidate_owned,
        'triangles': len(surface.faces),
        'rejected_support': surface.rejected_support,
        'mesh_vertices': len(surface.ids),
        'core_meshed_records': core_count,
        'unmeshed_records': len(own_ids) - core_count,
        'horizontal_triangles': surface.horizontal_triangles,
        'other_triangles': len(surface.faces) - surface.horizontal_triangles,
        'open_boundary_edges': int((multiplicities == 1).sum()),
        'nonmanifold_edges': int((multiplicities > 2).sum()),
        'estimated_normal_records': int(np.isin(surface.estimated_ids, own_ids).sum()),
        'min': own_xyz.min(axis=0).astype(float).tolist(),
        'max': own_xyz.max(axis=0).astype(float).tolist(),
    }
    if len(surface.faces):
        report['mesh_min'] = surface.positions.min(axis=0).astype(float).tolist()
        report['mesh_max'] = surface.positions.max(axis=0).astype(float).tolist()
    write_json(output / 'reports' / f'{name}.json', report)
    return report
