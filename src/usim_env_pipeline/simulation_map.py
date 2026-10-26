"""Create an independent supported terrain/building/obstacle simulation presentation."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError
from .pcd import open_records, sha256, xyz
from .simulation_cleanup import density_cleanup
from .simulation_ground import (
    GroundConfig,
    reconstruct_terrain,
    terrain_heights,
)
from .simulation_obstacles import ObstacleConfig, occupied_obstacles
from .simulation_output import covered_by_buildings, publish
from .simulation_residual import publish_residual
from .storage import write_json
from .usd import LayerMetadata


class SimulationReport(TypedDict):
    world: str
    source_sha256: str
    source_records: int
    ground_records: int
    building_records: int
    obstacle_records: int
    unresolved_records: int
    ground_cells: int
    ground_triangles: int
    ground_area_m2: float
    ground_inferred_nodes: int
    ground_smoothing_strength: float
    obstacle_voxels: int
    obstacle_triangles: int
    original_mesh_active: bool
    source_triangles: int
    replaced_source_triangles: int
    retained_source_triangles: int
    sparse_obstacle_voxels: int
    planar_records: int
    planar_models: int
    noise_records: int
    removed_noise_voxels: int
    removed_noise_triangles: int


def ground_support(output: Path) -> NDArray[np.int64]:
    """Use measured ground ownership, never infer a floor from a detached rooftop."""
    from pxr import Usd, UsdGeom

    ids = []
    entries = json.loads((output / 'authored-index.json').read_text(encoding='utf-8'))
    for entry in entries:
        if not entry['path'].startswith('/World/Ground/'):
            continue
        stage = Usd.Stage.Open(str(output / entry['payload']))
        mesh = UsdGeom.Mesh.Get(stage, '/Tile/Surface')
        ids.append(
            np.asarray(mesh.GetPrim().GetAttribute('source:recordIds').Get(), dtype=np.int64)
        )
    if not ids:
        raise PipelineInputError('A measured ground layer is required before simulation modeling')
    return np.unique(np.concatenate(ids))


def reconstruct(
    output: Path,
    ground_config: GroundConfig = GroundConfig(),
    obstacle_config: ObstacleConfig = ObstacleConfig(),
) -> SimulationReport:
    """Preserve originals while replacing noisy source approximations in a separate view."""
    from scipy.spatial import cKDTree

    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    source = manifest['sources'][manifest['primary_source']]
    if sha256(output / source['path']) != source['sha256']:
        raise PipelineInputError('Simulation source fingerprint changed')
    if not (output / 'buildings.usda').is_file():
        raise PipelineInputError('Generate the independent building models first')
    original_paths = [
        output / source['path'],
        output / 'authored.usda',
        output / 'buildings.usda',
    ]
    entries = json.loads((output / 'authored-index.json').read_text(encoding='utf-8'))
    original_paths.extend(output / entry['payload'] for entry in entries if 'payload' in entry)
    original_paths.extend((output / 'authored').rglob('*.usdc'))
    for directory in ('meshes', 'building-models'):
        original_paths.extend((output / directory).glob('*.usdc'))
        original_paths.extend((output / directory).glob('*.brep'))
        original_paths.extend((output / directory).glob('*.npz'))
    fingerprints = {path.relative_to(output).as_posix(): sha256(path) for path in original_paths}
    records, _ = open_records(output / source['path'])
    points = xyz(records)
    folder = output / 'simulation-models'
    folder.mkdir(parents=True, exist_ok=True)
    ids = ground_support(output)
    terrain = reconstruct_terrain(points, ids, ground_config)
    if not len(terrain.faces):
        raise PipelineInputError(
            'No supported simulation ground satisfies the fitting constraints'
        )
    print(f'SIMULATION_TERRAIN_FITTED cells={len(terrain.cells)}', flush=True)
    labels = np.zeros(len(points), dtype=np.uint8)
    measured_floor = cKDTree(points[ids, :2])
    for start in range(0, len(points), 250000):
        chunk = points[start : start + 250000]
        heights = terrain_heights(chunk[:, :2], terrain, ground_config)
        on_ground = np.abs(chunk[:, 2] - heights) <= obstacle_config.ground_tolerance
        distance, neighbor = measured_floor.query(chunk[:, :2], workers=4)
        on_ground |= (distance <= ground_config.nearest_support) & (
            np.abs(chunk[:, 2] - points[ids[neighbor], 2]) <= obstacle_config.ground_tolerance
        )
        on_ground &= np.isfinite(heights)
        labels[start : start + len(chunk)][on_ground] = 1
    labels[ids] = 1
    labels[
        covered_by_buildings(points, output / 'building-models', obstacle_config.building_margin)
    ] = 2
    candidates = occupied_obstacles(points, np.flatnonzero(labels == 0), obstacle_config)
    cleanup = density_cleanup(candidates, obstacle_config)
    labels[candidates.ids[~cleanup.keep[candidates.groups]]] = 5
    np.savez_compressed(
        folder / 'cleanup.npz',
        keys=candidates.keys,
        support_counts=cleanup.support_counts,
        keep=cleanup.keep,
    )
    removed_voxels = int((~cleanup.keep).sum())
    print(
        f'SIMULATION_OBSTACLES_CLEANED candidates={len(candidates.keys)} '
        f'removed_voxels={removed_voxels} removed_records={int((labels == 5).sum())}',
        flush=True,
    )
    del candidates, cleanup
    from .simulation_shape_approximation import reconstruct_shapes

    proxies = reconstruct_shapes(points, labels == 0, ids, output, obstacle_config.cell)
    for proxy in proxies:
        labels[proxy.support_ids] = 4
    obstacles = occupied_obstacles(points, np.flatnonzero(labels == 0), obstacle_config)
    labels[obstacles.ids] = 3
    np.savez_compressed(
        folder / 'terrain.npz',
        positions=terrain.positions,
        faces=terrain.faces,
        keys=terrain.keys,
        cells=terrain.cells,
        corner_indices=terrain.corner_indices,
        support_ids=terrain.support_ids,
        measured_ground_ids=ids,
        smoothing_strength=terrain.smoothing_strength,
    )
    np.savez_compressed(
        folder / 'obstacles.npz',
        keys=obstacles.keys,
        counts=obstacles.counts,
        ids=obstacles.ids,
        groups=obstacles.groups,
    )
    np.savez_compressed(folder / 'source-partition.npz', labels=labels)
    write_json(
        folder / 'config.json',
        {'ground': asdict(ground_config), 'obstacles': asdict(obstacle_config)},
    )
    write_json(folder / 'source-hashes.json', fingerprints)
    metadata = LayerMetadata(
        source['license'],
        source['attribution'],
        manifest['source_catalog'],
        manifest['frame'],
        'continuous smoothed ground, closed buildings and measured occupied obstacle volumes',
    )
    from .simulation_plane_output import publish_planes

    plane_tiles = publish_planes(output, proxies, metadata)
    retained, original_faces, replaced_faces, noise_faces = publish_residual(
        output,
        terrain,
        obstacles,
        ground_config,
        obstacle_config,
        metadata,
        np.flatnonzero(labels == 5),
        proxies,
        np.union1d(ids, np.flatnonzero(labels == 1)),
    )
    tiles = publish(output, terrain, obstacles, obstacle_config.cell, metadata, smooth=True)
    from pxr import Usd, UsdGeom

    world = Usd.Stage.Open(str(output / 'simulation.usda'))
    UsdGeom.Xform.Define(world, '/World/SimulationUnresolved')
    for tile in retained + plane_tiles:
        UsdGeom.Xform.Define(world, tile['path']).GetPrim().GetPayloads().AddPayload(
            tile['payload'], '/Tile'
        )
    world.GetRootLayer().Save()
    write_json(folder / 'index.json', tiles + retained + plane_tiles)
    counts = np.bincount(labels, minlength=6)
    report: SimulationReport = {
        'world': 'simulation.usda',
        'source_sha256': source['sha256'],
        'source_records': len(points),
        'ground_records': int(counts[1]),
        'building_records': int(counts[2]),
        'obstacle_records': int(counts[3]),
        'unresolved_records': int(counts[0]),
        'ground_cells': len(terrain.cells),
        'ground_triangles': len(terrain.faces),
        'ground_area_m2': len(terrain.cells) * ground_config.cell**2,
        'ground_inferred_nodes': int((~(terrain.support_ids >= 0).any(1)).sum()),
        'ground_smoothing_strength': terrain.smoothing_strength,
        'obstacle_voxels': len(obstacles.keys),
        'obstacle_triangles': sum(tile['faces'] for tile in tiles if tile['kind'] == 'obstacles'),
        'original_mesh_active': False,
        'source_triangles': original_faces,
        'replaced_source_triangles': replaced_faces,
        'retained_source_triangles': original_faces - replaced_faces - noise_faces,
        'sparse_obstacle_voxels': int(
            (obstacles.counts < obstacle_config.confidence_records).sum()
        ),
        'planar_records': int(counts[4]),
        'planar_models': len(proxies),
        'noise_records': int(counts[5]),
        'removed_noise_voxels': removed_voxels,
        'removed_noise_triangles': noise_faces,
    }
    write_json(output / 'simulation-report.json', report)
    print('SIMULATION_MAP_GENERATED ' + json.dumps(report), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    defaults = ObstacleConfig()
    parser.add_argument('--voxel-cell', type=float, default=defaults.cell)
    parser.add_argument('--cleanup-radius', type=float, default=defaults.cleanup_radius)
    parser.add_argument('--cleanup-min-cells', type=int, default=defaults.cleanup_min_cells)
    args = parser.parse_args(argv)
    try:
        reconstruct(
            args.map,
            obstacle_config=ObstacleConfig(
                cell=args.voxel_cell,
                cleanup_radius=args.cleanup_radius,
                cleanup_min_cells=args.cleanup_min_cells,
            ),
        )
    except PipelineInputError as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
