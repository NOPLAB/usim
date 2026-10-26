"""Audit closed residual slab footprints, source ownership and collision geometry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError
from .simulation_metrics import signed_volume

if TYPE_CHECKING:
    from pxr import Usd


def verify_planes(
    output: Path, stage: Usd.Stage, points: NDArray[np.float32], cell: float
) -> tuple[NDArray[np.int64], int]:
    from pxr import UsdGeom, UsdPhysics
    from scipy.spatial import cKDTree

    from .plane_types import Plane
    from .simulation_planar_completion import CompletionContext

    folder = output / 'simulation-models/planes'
    context = CompletionContext(points, cKDTree(points))
    ids_list = []
    total_faces = 0
    for row in json.loads((folder / 'index.json').read_text()):
        name = row['name']
        with np.load(folder / (name + '.npz')) as proxy:
            ids, keys = proxy['support_ids'], proxy['keys']
            normal, u, v, offset = proxy['normal'], proxy['u'], proxy['v'], float(proxy['offset'])
            positions, faces = proxy['positions'], proxy['faces']
            source_plane = int(proxy['source_plane'])
            source_path = Path(str(proxy['source_path']))
            observed_keys = proxy['observed_keys']
            completed = bool(proxy['completed'])
        if (
            source_path.parent.as_posix()
            not in ('optimized/planes', 'simulation-models/primitive-planes')
            or source_path.name != f'plane_{source_plane:04}.npz'
        ):
            raise PipelineInputError('Residual slab has an unrelated plane authority')
        with np.load(output / source_path) as original:
            if not np.isin(ids, original['ids']).all():
                raise PipelineInputError('Residual slab claimed points outside its source plane')
        basis = np.column_stack((u, v, normal))
        if not (
            np.allclose(basis.T @ basis, np.eye(3), atol=1e-8)
            and np.isclose(np.linalg.det(basis), 1)
            and (np.abs(points[ids] @ normal + offset) <= 0.08 + 1e-5).all()
        ):
            raise PipelineInputError('Residual slab coordinate frame or normal support changed')
        actual_keys = np.unique(
            np.floor(np.column_stack((points[ids] @ u, points[ids] @ v)) / cell).astype(np.int64),
            axis=0,
        )
        if not np.array_equal(actual_keys, observed_keys):
            raise PipelineInputError('Residual slab observation provenance changed')
        expected_keys = actual_keys
        if completed:
            expected_keys = context.footprint(Plane(normal, offset, ids), cell)
            if expected_keys is None:
                raise PipelineInputError('Residual slab completion has insufficient support')
            distance = cKDTree(actual_keys * cell).query(keys * cell)[0]
            if (distance > 0.2 + cell + 1e-6).any():
                raise PipelineInputError('Residual slab inferred beyond bounded planar support')
        if not np.array_equal(expected_keys, keys):
            raise PipelineInputError('Residual slab footprint or protected openings changed')
        local_z = positions @ normal + offset
        if not np.allclose(np.abs(local_z), 0.08, atol=5e-5):
            raise PipelineInputError('Residual slab is not a bounded 0.16 m solid')
        volume = signed_volume(positions, faces)
        if not np.isclose(volume, len(keys) * cell**2 * 0.16, rtol=1e-3, atol=1e-6):
            raise PipelineInputError('Residual slab volume disagrees with occupied footprint')
        edges = np.sort(
            np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
        )
        _, counts = np.unique(edges, axis=0, return_counts=True)
        if (counts % 2).any():
            raise PipelineInputError('Residual slab contains an open boundary')
        mesh = UsdGeom.Mesh.Get(stage, f'/World/SimulationPlanes/{name}/Surface')
        prim = mesh.GetPrim()
        if not (
            np.array_equal(mesh.GetPointsAttr().Get(), positions)
            and np.array_equal(
                np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), faces
            )
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'none'
            and not prim.HasAttribute('source:recordIds')
        ):
            raise PipelineInputError(
                'Residual slab rendering/collision differs from its authority'
            )
        ids_list.append(ids)
        total_faces += len(faces)
    ids = np.concatenate(ids_list) if ids_list else np.empty(0, dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise PipelineInputError('Residual planes claimed the same source record twice')
    return np.sort(ids), total_faces
