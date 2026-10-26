"""Audit angular contact preservation, topology, and composed USD triangle collisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .models import PipelineInputError
from .pcd import sha256
from .simulation_angular import (
    coordinate_rows,
    flat_normals,
    protected,
    retained,
    shared_points,
    sources,
)
from . import simulation_polyhedron as polyhedra
from .simulation_curve_surface import CurveSurface
from .storage import write_json


def verify(output: Path, *, name: str = 'angular') -> None:
    """Load at most one payload per stage and compare authority and composed geometry."""
    from pxr import Usd, UsdGeom, UsdPhysics

    from .simulation_fit_verify import verify_ground

    output = output.resolve()
    folder = output / f'{name}-models'
    report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    for path, expected in report['source_hashes'].items():
        if sha256(Path(path)) != expected:
            raise PipelineInputError(f'Angular source authority changed: {path}')
    assets = sources(output)
    shared_keys = shared_points(assets)
    rows = json.loads((folder / 'index.json').read_text(encoding='utf-8'))
    changes = {row['path']: row for row in rows}
    fitted = json.loads((output / 'geometry-fit/index.json').read_text(encoding='utf-8'))
    kinds = {row['path']: row['kind'] for row in fitted}
    stage = Usd.Stage.Open(str(output / f'simulation-{name}.usda'), Usd.Stage.LoadNone)
    original = Usd.Stage.Open(str(output / 'simulation-grounded.usda'), Usd.Stage.LoadNone)
    if stage.GetPrimAtPath('/World/FittedPlanes').IsActive():
        raise PipelineInputError('Excluded FittedPlanes remain active for rendering or collision')
    if (
        report['inactive_roots'] != ['/World/FittedPlanes', '/World/PointRegions']
        or report['grounding_reverified']
        or not report['point_cloud_available']
    ):
        raise PipelineInputError('Angular report misstates its presentation or grounding scope')
    point_root = stage.GetPrimAtPath('/World/PointRegions')
    if not point_root or point_root.IsActive():
        raise PipelineInputError('Original point cloud must be present and initially inactive')
    point_rows = json.loads((output / 'point-index.json').read_text(encoding='utf-8'))
    if not point_rows:
        raise PipelineInputError('Original point archive has no payload entries')
    # Session activation audits real payloads without editing the authored angular layer.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        stage.OverridePrim('/World/PointRegions').SetActive(True)
    for entry in point_rows:
        prim = stage.GetPrimAtPath(entry['path'])
        if (
            not entry['path'].startswith('/World/PointRegions/')
            or not prim
            or not prim.HasAuthoredPayloads()
        ):
            raise PipelineInputError('Original point payload namespace is unavailable')
        stage.Load(entry['path'])
        cloud = UsdGeom.Points.Get(stage, entry['path'] + '/Points')
        if not cloud or not len(cloud.GetPointsAttr().Get()):
            raise PipelineInputError('Original point payload has no actual USD points')
        authority = Usd.Stage.Open(str(output / entry['payload']))
        raw = UsdGeom.Points.Get(authority, '/Tile/Points')
        if not (
            np.array_equal(cloud.GetPointsAttr().Get(), raw.GetPointsAttr().Get())
            and np.array_equal(cloud.GetIdsAttr().Get(), raw.GetIdsAttr().Get())
        ):
            raise PipelineInputError('Point comparison differs from original payload records')
        if UsdGeom.Imageable(cloud).ComputeVisibility() != 'inherited':
            raise PipelineInputError('Point comparison remains invisible when activated')
        stage.Unload(entry['path'])
    stage.GetSessionLayer().Clear()
    excluded = np.array(
        [
            asset.path == '/World/FittedPlanes' or asset.path.startswith('/World/FittedPlanes/')
            for asset in assets
        ]
    )
    with np.load(output / 'grounded-models/connections.npz') as graph:
        endpoints = graph['chunk_assets'][graph['witnesses']]
        excluded_witnesses = int(excluded[endpoints].any(axis=1).sum())
    if excluded_witnesses != report['excluded_plane_contact_witnesses']:
        raise PipelineInputError('Excluded plane contact witness accounting differs')
    stage.Load('/World/SimulationGround')
    verify_ground(output, stage)
    original.Load('/World/SimulationGround')
    floors = [
        next(
            UsdGeom.Mesh(prim)
            for prim in world.Traverse()
            if prim.IsA(UsdGeom.Mesh) and prim.GetPath().HasPrefix('/World/SimulationGround')
        )
        for world in (stage, original)
    ]
    if not (
        np.array_equal(
            UsdGeom.XformCache().GetLocalToWorldTransform(floors[0].GetPrim()),
            assets[0].matrix,
        )
        and np.array_equal(floors[0].GetNormalsAttr().Get(), floors[1].GetNormalsAttr().Get())
        and np.array_equal(
            floors[0].GetDisplayColorPrimvar().ComputeFlattened(),
            floors[1].GetDisplayColorPrimvar().ComputeFlattened(),
        )
    ):
        raise PipelineInputError('Angular publication changed floor transform or appearance')
    stage.Unload('/World/SimulationGround')
    original.Unload('/World/SimulationGround')
    expected = {
        row['path']
        for row in fitted
        if row['kind'] == 'curve'
        and original.GetPrimAtPath(row['path']).IsActive()
        and row['path'] != '/World/FittedPlanes'
        and not row['path'].startswith('/World/FittedPlanes/')
    }
    if set(changes) != expected:
        raise PipelineInputError('Angular overrides are not exactly the active curve rows')
    total_before, total_after, frozen_count = 0, 0, 0
    for index, asset in enumerate(assets[1:], 1):
        if asset.path == '/World/FittedPlanes' or asset.path.startswith('/World/FittedPlanes/'):
            prim = stage.GetPrimAtPath(asset.path)
            if prim and prim.IsActive():
                raise PipelineInputError('An excluded plane payload remains active')
            continue
        if not original.GetPrimAtPath(asset.path).IsActive():
            if stage.GetPrimAtPath(asset.path).IsActive():
                raise PipelineInputError('Angular publication reactivated a floating component')
            continue
        original.Load(asset.path)
        stage.Load(asset.path)
        old = UsdGeom.Mesh.Get(original, asset.path + '/Surface')
        mesh = UsdGeom.Mesh.Get(stage, asset.path + '/Surface')
        actual = CurveSurface(
            np.asarray(mesh.GetPointsAttr().Get(), np.float32),
            np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int32).reshape(-1, 3),
        )
        matrix = np.array(
            UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()), np.float64
        )
        if not np.array_equal(matrix, asset.matrix):
            raise PipelineInputError('Angular publication changed a world transform')
        if not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3):
            raise PipelineInputError('Angular output is not triangular')
        prim = mesh.GetPrim()
        if not (
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'none'
            and mesh.GetSubdivisionSchemeAttr().Get() == 'none'
        ):
            raise PipelineInputError('Angular render and collision geometry differ')
        if asset.path not in changes:
            if not (
                np.array_equal(actual.positions, old.GetPointsAttr().Get())
                and np.array_equal(
                    actual.faces, np.asarray(old.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
                )
                and np.array_equal(mesh.GetNormalsAttr().Get(), old.GetNormalsAttr().Get())
                and np.array_equal(
                    mesh.GetDisplayColorPrimvar().ComputeFlattened(),
                    old.GetDisplayColorPrimvar().ComputeFlattened(),
                )
            ):
                raise PipelineInputError('Angular publication changed an inherited slab')
        else:
            row = changes[asset.path]
            if kinds[asset.path] != 'curve' or row['asset'] != index:
                raise PipelineInputError('Angular payload identity differs from its source')
            with np.load((output / row['payload']).with_suffix('.npz')) as cache:
                candidate = CurveSurface(cache['positions'], cache['faces'])
                saved_ids, color = cache['protected_source_faces'], cache['color']
            if not (
                np.array_equal(actual.positions, candidate.positions)
                and np.array_equal(actual.faces, candidate.faces)
            ):
                raise PipelineInputError('Composed USD differs from angular triangle cache')
            reference, face_ids = retained(asset)
            frozen = protected(output, assets, index, reference, face_ids, shared_keys)
            if not np.array_equal(saved_ids, face_ids[frozen]):
                raise PipelineInputError('Angular protection provenance is incomplete')
            triangles = coordinate_rows(actual.positions[actual.faces].reshape(-1, 9))
            contacts = coordinate_rows(reference.positions[reference.faces[frozen]].reshape(-1, 9))
            if not np.isin(contacts, triangles).all():
                raise PipelineInputError('Angular publication changed a contact triangle')
            if polyhedra.boundary_coordinates(actual) != polyhedra.boundary_coordinates(reference):
                raise PipelineInputError('Angular publication changed seam boundary edges')
            if polyhedra.component_signatures(actual) != polyhedra.component_signatures(reference):
                raise PipelineInputError('Angular component topology or nonmanifold edges changed')
            if not polyhedra.nonmanifold_edges(actual) <= polyhedra.nonmanifold_edges(reference):
                raise PipelineInputError('Angular publication introduced new nonmanifold edges')
            if not (
                mesh.GetNormalsInterpolation() == 'uniform'
                and np.array_equal(mesh.GetNormalsAttr().Get(), flat_normals(actual))
                and mesh.GetDisplayColorPrimvar().GetInterpolation() == 'constant'
                and np.array_equal(mesh.GetDisplayColorPrimvar().Get(), color)
                and np.array_equal(
                    color, np.asarray(old.GetDisplayColorPrimvar().ComputeFlattened())[0:1]
                )
            ):
                raise PipelineInputError('Angular flat normals or existing color differ')
            if len(actual.faces) > len(reference.faces):
                raise PipelineInputError('Angular publication increased curve triangle count')
            if row['source_faces'] != len(reference.faces) or row['faces'] != len(actual.faces):
                raise PipelineInputError('Angular triangle accounting differs')
            total_before += len(reference.faces)
            total_after += len(actual.faces)
            frozen_count += int(frozen.sum())
        original.Unload(asset.path)
        stage.Unload(asset.path)
    if total_after >= total_before:
        raise PipelineInputError('Angular curves have no real triangle reduction')
    result = {
        'source_files_unchanged': True,
        'floor_and_retained_slabs_unchanged': True,
        'inactive_roots': ['/World/FittedPlanes', '/World/PointRegions'],
        'point_cloud_inactive_and_available': True,
        'grounding_reverified': False,
        'excluded_plane_contact_witnesses': report['excluded_plane_contact_witnesses'],
        'source_curve_faces': total_before,
        'angular_curve_faces': total_after,
        'protected_triangles': frozen_count,
        'contact_triangles_unchanged': True,
        'seam_boundary_edges_unchanged': True,
        'component_topology_preserved': True,
        'render_collision_identical': True,
        'flat_face_normals': True,
        'limitations': [
            'FittedPlanes are excluded from rendering and collision. Grounded keep masks and '
            'contact witnesses predate that exclusion; retained bodies are not certified grounded '
            'until connectivity is recomputed against the active angular geometry.',
            'Topology audit checks component Euler characteristics, boundaries and nonmanifold '
            'edge counts; it does not prove absence of self-intersections or cavity shape changes.',
            'Contacts retain saved witness chunks and exact shared vertices; unsaved near '
            'contacts and continuous clearance throughout each component are not certified.',
        ],
    }
    write_json(folder / 'verification.json', result)
    print('ANGULAR_VERIFIED ' + json.dumps(result, ensure_ascii=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--name', default='angular')
    args = parser.parse_args()
    verify(args.map, name=args.name)


if __name__ == '__main__':
    main()
