"""Compose supported simulation models without duplicate archival mesh collisions."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from .simulation_ground import Terrain
from .simulation_obstacles import Occupancy, voxel_surface
from .storage import tile_name, write_json
from .usd import Geometry, LayerMetadata, author_geometry, stage_base


class SimulationTile(TypedDict):
    path: str
    payload: str
    kind: str
    faces: int


def publish(
    output: Path,
    terrain: Terrain,
    obstacles: Occupancy,
    obstacle_cell: float,
    metadata: LayerMetadata,
    smooth: bool = False,
) -> list[SimulationTile]:
    """Publish one shared ground mesh and tiled occupied obstacle surfaces."""
    from pxr import Sdf, Usd, UsdGeom, Vt

    from .simulation_surface_smoothing import smooth_surface

    folder = output / 'simulation-models'
    world = stage_base(output / 'simulation.usda', '/World', metadata)
    world.GetRootLayer().subLayerPaths = ['./buildings.usda']
    # Inactive archives lose both rendering and collisions in this presentation only.
    for name in (
        'Regions',
        'PointRegions',
        'Ground',
        'BuildingResidual',
        'InferredStructures',
        'Vegetation',
    ):
        world.OverridePrim('/World/' + name).SetActive(False)
    entries: list[SimulationTile] = []
    obstacle_positions, obstacle_faces, obstacle_owners = voxel_surface(
        obstacles.keys, obstacle_cell
    )
    surfaces = (
        (
            'ground',
            'SimulationGround',
            terrain.positions,
            terrain.faces,
            terrain.positions[terrain.faces].mean(1)[:, :2],
            [0.30, 0.32, 0.33],
            'terrain.npz',
        ),
        (
            'obstacles',
            'SimulationObstacles',
            obstacle_positions,
            obstacle_faces,
            (obstacles.keys[obstacle_owners, :2] + 0.5) * obstacle_cell,
            [0.43, 0.46, 0.40],
            'obstacles.npz',
        ),
    )
    for kind, root, positions, faces, centroids, color, provenance in surfaces:
        UsdGeom.Xform.Define(world, '/World/' + root)
        if not len(faces):
            continue
        (folder / kind).mkdir(parents=True, exist_ok=True)
        owners = (
            np.zeros((len(faces), 2), np.int64)
            if kind == 'ground'
            else np.floor(centroids / 30.0).astype(np.int64)
        )
        keys, groups = np.unique(owners, axis=0, return_inverse=True)
        for number, key in enumerate(keys):
            name = tile_name((int(key[0]), int(key[1])))
            used, inverse = np.unique(faces[groups == number], return_inverse=True)
            local_faces = inverse.reshape(-1, 3).astype(np.int32)
            path = folder / kind / f'{name}.usdc'
            local_positions = positions[used]
            smoothed = smooth and kind == 'obstacles'
            if smoothed:
                local_positions = smooth_surface(local_positions, local_faces, obstacle_cell)
            author_geometry(
                path,
                Geometry(
                    local_positions,
                    np.empty(0, dtype=np.int64),
                    np.tile(np.asarray([color], dtype=np.float32), (len(used), 1)),
                    local_faces,
                ),
                metadata,
            )
            stage = Usd.Stage.Open(str(path))
            prim = stage.GetPrimAtPath('/Tile/Surface')
            if kind == 'ground' or smoothed:
                triangles = local_positions[local_faces]
                face_normals = np.cross(
                    triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
                )
                normals = np.zeros_like(local_positions)
                for corner in range(3):
                    np.add.at(normals, local_faces[:, corner], face_normals)
                normals /= np.linalg.norm(normals, axis=1)[:, None]
                mesh = UsdGeom.Mesh(prim)
                mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
                mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            prim.CreateAttribute(
                'simulation:smoothingLimit', Sdf.ValueTypeNames.Double, custom=True
            ).Set(0.2 * obstacle_cell if smoothed else 0.0)
            prim.RemoveProperty('source:recordIds')
            prim.CreateAttribute('simulation:kind', Sdf.ValueTypeNames.String, custom=True).Set(
                kind
            )
            prim.CreateAttribute('source:provenance', Sdf.ValueTypeNames.Asset, custom=True).Set(
                Sdf.AssetPath('../' + provenance)
            )
            stage.GetRootLayer().Save()
            np.savez_compressed(
                path.with_suffix('.npz'),
                vertex_indices=used,
                face_indices=np.flatnonzero(groups == number),
            )
            prim_path = f'/World/{root}/{name}'
            payload = f'./simulation-models/{kind}/{name}.usdc'
            UsdGeom.Xform.Define(world, prim_path).GetPrim().GetPayloads().AddPayload(
                payload, '/Tile'
            )
            entries.append(
                {'path': prim_path, 'payload': payload, 'kind': kind, 'faces': len(local_faces)}
            )
    world.GetRootLayer().Save()
    write_json(folder / 'index.json', entries)
    return entries


def covered_by_buildings(
    points: NDArray[np.float32], models_folder: Path, margin: float
) -> NDArray[np.bool_]:
    """Claim only source points inside bounded oriented building volumes plus a small margin."""
    import json

    covered = np.zeros(len(points), dtype=bool)
    for row in json.loads((models_folder / 'models.json').read_text(encoding='utf-8')):
        with np.load(models_folder / (row['name'] + '.npz')) as data:
            center, size, axes = data['center'], data['size'], data['axes']
        extent = np.abs(axes) @ (size / 2 + margin)
        candidates = np.flatnonzero(
            ((points >= center - extent) & (points <= center + extent)).all(1)
        )
        signed = np.abs((points[candidates] - center) @ axes) - size / 2
        within = np.linalg.norm(np.maximum(signed, 0), axis=1) <= margin
        covered[candidates[within]] = True
    return covered
