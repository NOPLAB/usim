"""Independently verify closed building solids, source support and unchanged ground."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .building_types import BuildingConfig
from .cad_exchange import read_brep, read_step, tessellate
from .cad_kernel import faces
from .models import PipelineInputError
from .pcd import open_records, sha256, xyz
from .storage import write_json


def verify(output: Path) -> None:
    from OCP.BRepCheck import BRepCheck_NoError, BRepCheck_Shell
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from pxr import Usd, UsdGeom, UsdPhysics

    report = json.loads((output / 'building-report.json').read_text(encoding='utf-8'))
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    source = manifest['sources'][manifest['primary_source']]
    folder = output / 'building-models'
    config = BuildingConfig(**json.loads((folder / 'config.json').read_text(encoding='utf-8')))
    rows = json.loads((folder / 'models.json').read_text(encoding='utf-8'))
    facade_rows = json.loads((folder / 'facades.json').read_text(encoding='utf-8'))
    if sha256(output / source['path']) != source['sha256'] or (
        source['sha256'] != report['source_sha256'] or report['ground_modified']
    ):
        raise PipelineInputError('Building source fingerprint or ground policy changed')
    records, _ = open_records(output / source['path'])
    points = xyz(records)
    if len(points) != report['source_records']:
        raise PipelineInputError('Building source record count changed')
    facade_ids = []
    for row in facade_rows:
        with np.load(folder / 'facades' / f'{row["name"]}.npz') as data:
            ids, normal = data['support_ids'], data['normal']
            allowed = []
            for number in data['source_planes']:
                with np.load(output / 'optimized/planes' / f'plane_{number:04}.npz') as original:
                    if abs(original['equation'][2]) > np.sin(np.deg2rad(config.vertical_angle)):
                        raise PipelineInputError('Ground or roof was classified as building wall')
                    allowed.append(original['ids'])
            tangent = np.array([normal[1], -normal[0], 0.0])
            local = np.column_stack((points[ids] @ tangent, points[ids, 2]))
            if not (
                np.isin(ids, np.unique(np.concatenate(allowed))).all()
                and len(ids) == row['support_records']
                and np.abs(points[ids] @ normal + data['offset']).max()
                <= config.support_distance + 1e-5
                and (local >= data['lower'] - 1e-5).all()
                and (local <= data['upper'] + 1e-5).all()
            ):
                raise PipelineInputError(f'Building facade support mismatch: {row["name"]}')
            facade_ids.append(ids)
    stage = Usd.Stage.Open(str(output / report['world']))
    original_stage = Usd.Stage.Open(str(output / 'authored.usda'))
    if not stage.GetPrimAtPath('/World/Ground').IsActive():
        raise PipelineInputError('Building view removed the separate ground layer')
    ground = json.loads((folder / 'ground-source.json').read_text(encoding='utf-8'))
    for payload, fingerprint in ground.items():
        if sha256(output / payload) != fingerprint:
            raise PipelineInputError(f'Original ground bytes changed: {payload}')
    assignments, volume = [], 0.0
    errors = []
    for row in rows:
        shape = read_brep(folder / f'{row["name"]}.brep')
        shell = TopExp_Explorer(shape, TopAbs_SHELL)
        if (
            shape.ShapeType() != TopAbs_SOLID
            or len(faces(shape)) != 6
            or not shell.More()
            or BRepCheck_Shell(TopoDS.Shell_s(shell.Current())).Closed() != BRepCheck_NoError
        ):
            raise PipelineInputError(f'Building is not a closed cuboid solid: {row["name"]}')
        with np.load(folder / f'{row["name"]}.npz') as data:
            indices, ids = data['facade_indices'], data['support_ids']
            expected = np.unique(np.concatenate([facade_ids[index] for index in indices]))
            if not (
                np.array_equal(ids, expected)
                and len(ids) == row['support_records']
                and np.allclose(data['axes'].T @ data['axes'], np.eye(3), atol=1e-10)
                and np.isclose(np.linalg.det(data['axes']), 1)
                and (data['size'] > 0).all()
            ):
                raise PipelineInputError(f'Building cuboid/provenance mismatch: {row["name"]}')
            assignments.extend(indices.tolist())
            local_support = (points[ids] - data['center']) @ data['axes']
            signed = np.abs(local_support) - data['size'] / 2
            errors.append(
                np.where(
                    (signed > 0).any(axis=1),
                    np.linalg.norm(np.maximum(signed, 0), axis=1),
                    -signed.max(axis=1),
                )
            )
            vertices, triangles = tessellate(shape)
            local = (vertices - data['center']) @ data['axes']
            if not np.allclose(np.abs(local), data['size'] / 2, atol=1e-8):
                raise PipelineInputError('Building BRep does not match its oriented box')
            props = GProp_GProps()
            BRepGProp.VolumeProperties_s(shape, props)
            if not np.isclose(props.Mass(), row['volume_m3'], rtol=1e-9):
                raise PipelineInputError('Building solid volume mismatch')
            volume += props.Mass()
        mesh = UsdGeom.Mesh.Get(stage, f'/World/BuildingModels/{row["name"]}/Surface')
        if not (
            np.array_equal(mesh.GetPointsAttr().Get(), vertices.astype(np.float32))
            and np.array_equal(
                np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), triangles
            )
            and UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get()
            and UsdPhysics.MeshCollisionAPI(mesh.GetPrim()).GetApproximationAttr().Get()
            == 'convexHull'
            and not mesh.GetPrim().GetAttribute('source:recordIds')
            and Path(
                mesh.GetPrim().GetAttribute('building:authority').Get().resolvedPath
            ).resolve()
            == (folder / f'{row["name"]}.brep').resolve()
        ):
            raise PipelineInputError('Building render/collision is not its closed kernel geometry')
    if sorted(assignments) != list(range(len(facade_rows))):
        raise PipelineInputError('Facade support is missing or assigned to multiple models')
    for authority in (folder / 'model.brep', folder / 'model.step'):
        shape = read_step(authority) if authority.suffix == '.step' else read_brep(authority)
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        count = 0
        while explorer.More():
            count += 1
            explorer.Next()
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        if count != len(rows) or not np.isclose(props.Mass(), volume, rtol=1e-8):
            raise PipelineInputError('Building STEP/BRep lost solid topology or meter units')
    entries = json.loads((output / 'authored-index.json').read_text(encoding='utf-8'))
    residual_count, replaced_count = 0, 0
    for entry in entries:
        if not entry['path'].startswith(('/World/Regions/', '/World/Ground/')):
            continue
        original = UsdGeom.Mesh.Get(original_stage, entry['path'] + '/Surface')
        original_faces = np.asarray(original.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        if entry['path'].startswith('/World/Ground/'):
            target = UsdGeom.Mesh.Get(stage, entry['path'] + '/Surface')
            if not (
                target.GetPrim().IsActive()
                and np.array_equal(target.GetPointsAttr().Get(), original.GetPointsAttr().Get())
                and np.array_equal(
                    target.GetFaceVertexIndicesAttr().Get(),
                    original.GetFaceVertexIndicesAttr().Get(),
                )
            ):
                raise PipelineInputError('Ground collision geometry changed in the building view')
            continue
        name = entry['path'].rsplit('/', 1)[1]
        if stage.GetPrimAtPath(entry['path']).IsActive():
            residual_count += len(original_faces)
            continue
        target = UsdGeom.Mesh.Get(stage, f'/World/BuildingResidual/{name}/Surface')
        if not target:
            replaced_count += len(original_faces)
            continue
        with np.load(folder / 'residual' / f'{name}.npz') as data:
            indices = data['source_face_indices']
        if not (
            np.array_equal(target.GetPointsAttr().Get(), original.GetPointsAttr().Get())
            and np.array_equal(
                target.GetFaceVertexIndicesAttr().Get(), original_faces[indices].ravel()
            )
            and np.array_equal(
                target.GetPrim().GetAttribute('source:recordIds').Get(),
                original.GetPrim().GetAttribute('source:recordIds').Get(),
            )
        ):
            raise PipelineInputError('Unrelated original region geometry changed')
        # Horizontal source triangles must never be removed by a building fit.
        vertices = np.asarray(original.GetPointsAttr().Get())[original_faces]
        normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        horizontal = np.abs(normals[:, 2]) > np.linalg.norm(normals, axis=1) * np.sin(
            np.deg2rad(config.vertical_angle)
        )
        if not np.isin(np.flatnonzero(horizontal), indices).all():
            raise PipelineInputError('Building replacement removed a non-wall source triangle')
        residual_count += len(indices)
        replaced_count += len(original_faces) - len(indices)
    if not (
        residual_count == report['residual_triangles']
        and replaced_count == report['replaced_triangles']
        and len(rows) == report['building_cuboids'] + report['facade_cuboids']
        and len(facade_rows) == report['facades']
        and len(np.unique(np.concatenate(facade_ids))) == report['supported_records']
    ):
        raise PipelineInputError('Building replacement report mismatch')
    distances = np.concatenate(errors)
    validation = {
        'models': len(rows),
        'facades': len(facade_rows),
        'source_sha256': report['source_sha256'],
        'ground_unchanged': True,
        'support_surface_error_max_m': float(distances.max()),
        'support_surface_error_p95_m': float(np.quantile(distances, 0.95)),
    }
    write_json(folder / 'verification.json', validation)
    print(
        f'BUILDING_VERIFICATION_PASSED models={len(rows)} facades={len(facade_rows)} '
        f'replaced_triangles={replaced_count} ground_unchanged=True '
        f'support_surface_error_max_m={distances.max():.6f} '
        f'support_surface_error_p95_m={np.quantile(distances, 0.95):.6f}',
        flush=True,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    verify(parser.parse_args().map)


if __name__ == '__main__':
    main()
