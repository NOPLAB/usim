"""Audit disjoint original-face ownership and exact retained measured collisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError

if TYPE_CHECKING:
    from pxr import Usd


def verify_faces(
    output: Path,
    stage: Usd.Stage,
    noise_ids: NDArray[np.int64],
    ground_ids: NDArray[np.int64] | None = None,
) -> tuple[int, int, int, int]:
    from pxr import Usd, UsdGeom, UsdPhysics

    folder = output / 'simulation-models'
    original_faces, replaced_faces, remaining_faces, noise_faces = 0, 0, 0, 0
    expected_paths = set()
    rows = json.loads((folder / 'index.json').read_text())
    retained_rows = {row['path']: row for row in rows if row['kind'] == 'unresolved'}
    for row in json.loads((output / 'authored-index.json').read_text()):
        if not row['path'].startswith('/World/Regions/'):
            continue
        name = row['path'].rsplit('/', 1)[1]
        original = Usd.Stage.Open(str(output / row['payload']))
        original_mesh = UsdGeom.Mesh.Get(original, '/Tile/Surface')
        with np.load(folder / 'face-accounting' / f'{name}.npz') as accounting:
            reason = accounting['reason']
        if len(reason) != len(original_mesh.GetFaceVertexCountsAttr().Get()) or (reason > 5).any():
            raise PipelineInputError('Original face partition is incomplete')
        faces = np.asarray(original_mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        ids = np.asarray(original_mesh.GetPrim().GetAttribute('source:recordIds').Get())
        if not np.array_equal(reason == 5, np.isin(ids, noise_ids)[faces].any(1)):
            raise PipelineInputError('Noise source triangles were retained or wrongly removed')
        if ground_ids is not None:
            floor_faces = np.isin(ids, ground_ids)[faces].all(1)
            if ((reason[floor_faces] == 0) | (reason[floor_faces] == 5)).any():
                raise PipelineInputError('Measured ground reappeared as residual geometry')
        retained = np.flatnonzero(reason == 0)
        original_faces += len(reason)
        replaced_faces += int(((reason > 0) & (reason < 5)).sum())
        noise_faces += int((reason == 5).sum())
        remaining_faces += len(retained)
        if not len(retained):
            continue
        path = f'/World/SimulationUnresolved/{name}'
        expected_paths.add(path)
        if path not in retained_rows:
            raise PipelineInputError(f'Unresolved source faces have no payload: {name}')
        with np.load(folder / 'residual' / f'{name}.npz') as indices:
            if str(indices['source_payload']) != row['payload']:
                raise PipelineInputError('Unresolved source payload ownership changed')
            owned, used = indices['source_face_indices'], indices['source_vertex_indices']
        original_points = np.asarray(original_mesh.GetPointsAttr().Get())
        mesh = UsdGeom.Mesh.Get(stage, path + '/Surface')
        prim = mesh.GetPrim()
        rendered = np.asarray(mesh.GetPointsAttr().Get())
        rendered_faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        if not (
            np.array_equal(owned, retained)
            and np.array_equal(rendered, original_points[used])
            and np.array_equal(used[rendered_faces], faces[owned])
            and np.array_equal(prim.GetAttribute('source:recordIds').Get(), ids[used])
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'none'
            and len(rendered_faces) == retained_rows[path]['faces']
        ):
            raise PipelineInputError(f'Unresolved measured geometry or collisions changed: {name}')
    if expected_paths != retained_rows.keys():
        raise PipelineInputError(
            'Unresolved face payloads include duplicate or unrelated geometry'
        )
    return original_faces, replaced_faces, remaining_faces, noise_faces
