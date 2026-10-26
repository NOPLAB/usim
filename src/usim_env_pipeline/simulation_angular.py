"""Publish constrained angular curves over the unchanged grounded simulation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError
from .pcd import sha256
from .simulation_curve_surface import CurveSurface
from .simulation_polyhedron import polyhedron
from .storage import write_json


@dataclass(frozen=True, slots=True)
class Source:
    path: str
    archive: Path
    mask: Path | None
    order: Path
    matrix: NDArray[np.float64]


def sources(output: Path) -> list[Source]:
    rows = json.loads((output / 'grounded-models/index.json').read_text(encoding='utf-8'))
    masks = {row['path']: output / row['mask'] for row in rows}
    assets = json.loads((output / 'grounded-models/sources.json').read_text(encoding='utf-8'))
    return [
        Source(
            row['path'],
            output / row['archive'],
            masks.get(row['path']),
            output / row['order'],
            np.asarray(row['matrix'], np.float64),
        )
        for row in assets
    ]


def retained(source: Source) -> tuple[CurveSurface, NDArray[np.int64]]:
    """Keep original vertex IDs and original face order while applying grounded masks."""
    with np.load(source.archive) as data:
        points, faces = data['positions'], data['faces']
    mask = np.ones(len(faces), bool) if source.mask is None else np.load(source.mask)
    return CurveSurface(points, faces[mask]), np.flatnonzero(mask)


def coordinate_rows(
    points: NDArray[np.float32] | NDArray[np.float64],
) -> NDArray[np.void]:
    values = np.array(points, copy=True, order='C')
    values[values == 0] = 0
    return values.view(np.dtype((np.void, values.dtype.itemsize * values.shape[1]))).ravel()


def protected(
    output: Path,
    assets: list[Source],
    index: int,
    surface: CurveSurface,
    face_ids: NDArray[np.int64],
    shared_keys: NDArray[np.void],
) -> NDArray[np.bool_]:
    """Freeze witness chunks and triangles incident to exact shared world vertices."""
    source = assets[index]
    original = np.zeros(int(face_ids[-1]) + 1, bool)
    with np.load(output / 'grounded-models/connections.npz') as graph:
        chunks = np.unique(graph['witnesses'])
        chunks = chunks[graph['chunk_assets'][chunks] == index]
        order = np.load(source.order, mmap_mode='r')
        for chunk in chunks:
            ids = order[graph['chunk_starts'][chunk] : graph['chunk_stops'][chunk]]
            original[ids[ids < len(original)]] = True
    frozen = original[face_ids]
    used = np.unique(surface.faces)
    world = surface.positions[used].astype(np.float64)
    world = world @ source.matrix[:3, :3] + source.matrix[3, :3]
    keys = coordinate_rows(world)
    shared = np.zeros(len(surface.positions), bool)
    shared[used] = np.isin(keys, shared_keys)
    return frozen | shared[surface.faces].any(axis=1)


def shared_points(assets: list[Source]) -> NDArray[np.void]:
    """Index exact cross-payload contacts once, rather than rereading every source per curve."""
    keys = []
    for source in assets:
        mesh, _ = retained(source)
        points = mesh.positions[np.unique(mesh.faces)].astype(np.float64)
        points = points @ source.matrix[:3, :3] + source.matrix[3, :3]
        keys.append(np.unique(coordinate_rows(points)))
    points, counts = np.unique(np.concatenate(keys), return_counts=True)
    return points[counts > 1]


def flat_normals(surface: CurveSurface) -> NDArray[np.float32]:
    triangles = surface.positions[surface.faces].astype(np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0):
        raise PipelineInputError('Angular output contains a zero-area triangle')
    return (normals / lengths[:, None]).astype(np.float32)


def publish(output: Path, cell: float = 0.3, *, name: str = 'angular') -> None:
    """Replace active curve payloads only, preserving all inherited transforms."""
    from pxr import Sdf, Usd, UsdGeom, Vt

    from .simulation_fit_output import publish_surface
    from .usd import LayerMetadata, stage_base

    if not np.isfinite(cell) or cell <= 0:
        raise PipelineInputError('Angular cell must be finite and positive')
    output = output.resolve()
    folder = output / f'{name}-models'
    folder.mkdir(parents=True, exist_ok=True)
    assets = sources(output)
    shared_keys = shared_points(assets)
    rows = json.loads((output / 'geometry-fit/index.json').read_text(encoding='utf-8'))
    authority = json.loads((output / 'grounded-models/report.json').read_text(encoding='utf-8'))[
        'source_hashes'
    ]
    tracked = {Path(path) for path in authority}
    tracked.update(path for path in (output / 'grounded-models').rglob('*') if path.is_file())
    tracked.add(output / 'simulation-grounded.usda')
    point_rows = json.loads((output / 'point-index.json').read_text(encoding='utf-8'))
    tracked.update((output / 'points.usda', output / 'point-index.json'))
    tracked.update(output / row['payload'] for row in point_rows)
    fingerprints = {str(path): sha256(path) for path in tracked}
    if any(fingerprints[path] != value for path, value in authority.items()):
        raise PipelineInputError('Grounded source authority changed before angular publication')
    original = Usd.Stage.Open(str(output / 'simulation-grounded.usda'), Usd.Stage.LoadNone)
    data = original.GetRootLayer().customLayerData
    metadata = LayerMetadata(
        data['license'], data['attribution'], data['sourceCatalog'], data['frame'], data['color']
    )
    stage = stage_base(output / f'simulation-{name}.usda', '/World', metadata)
    stage.SetLoadRules(Usd.StageLoadRules.LoadNone())
    stage.GetRootLayer().subLayerPaths = ['./simulation-grounded.usda']
    points = original.GetPrimAtPath('/World/PointRegions')
    if not points or not points.IsDefined():
        stage.GetRootLayer().subLayerPaths.append('./points.usda')
    stage.GetRootLayer().customLayerData = data
    stage.OverridePrim('/World/FittedPlanes').SetActive(False)
    point_root = stage.OverridePrim('/World/PointRegions')
    point_root.SetActive(False)
    UsdGeom.Imageable(point_root).CreateVisibilityAttr(UsdGeom.Tokens.inherited)
    published = []
    for row in rows:
        if (
            row['kind'] != 'curve'
            or row['path'] == '/World/FittedPlanes'
            or row['path'].startswith('/World/FittedPlanes/')
            or not original.GetPrimAtPath(row['path']).IsActive()
        ):
            continue
        index = next(i for i, asset in enumerate(assets) if asset.path == row['path'])
        surface, face_ids = retained(assets[index])
        frozen = protected(output, assets, index, surface, face_ids, shared_keys)
        result = polyhedron(surface.positions, surface.faces, frozen, cell)
        original.Load(row['path'])
        old = UsdGeom.Mesh.Get(original, row['path'] + '/Surface')
        color = np.asarray(old.GetDisplayColorPrimvar().ComputeFlattened(), np.float32)[0:1]
        path = folder / 'meshes' / f'{index:04d}.usdc'
        publish_surface(path, result.surface, metadata)
        normals = flat_normals(result.surface)
        part = Usd.Stage.Open(str(path))
        payload_mesh = UsdGeom.Mesh.Get(part, '/Tile/Surface')
        payload_mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
        payload_mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
        payload_colors = payload_mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
        payload_colors.Set(Vt.Vec3fArray.FromNumpy(color))
        payload_colors.BlockIndices()
        part.GetRootLayer().Save()
        payload = './' + path.relative_to(output).as_posix()
        stage.OverridePrim(row['path']).GetPayloads().SetPayloads([Sdf.Payload(payload, '/Tile')])
        # Root opinions must beat old grounded root-level vertex appearance opinions.
        mesh = UsdGeom.Mesh.Define(stage, row['path'] + '/Surface')
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
        colors = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
        colors.Set(Vt.Vec3fArray.FromNumpy(color))
        colors.BlockIndices()
        mesh.GetPrim().CreateAttribute(
            'source:provenance', Sdf.ValueTypeNames.Asset, custom=True
        ).Set(Sdf.AssetPath('./' + path.with_suffix('.npz').relative_to(output).as_posix()))
        np.savez_compressed(
            path.with_suffix('.npz'),
            positions=result.surface.positions,
            faces=result.surface.faces,
            protected_source_faces=face_ids[frozen],
            color=color,
        )
        original.Unload(row['path'])
        published.append(
            {
                'path': row['path'],
                'asset': index,
                'payload': payload,
                'source_faces': len(surface.faces),
                'faces': len(result.surface.faces),
                'protected_faces': int(frozen.sum()),
                'reduced_components': result.reduced_components,
                'unchanged_components': result.unchanged_components,
            }
        )
        print(f'ANGULAR_PAYLOAD path={row["path"]} faces={published[-1]["faces"]}', flush=True)
    stage.GetRootLayer().Save()
    write_json(folder / 'index.json', published)
    full = [{'path': row['path'], 'kind': 'curve', 'payload': row['payload']} for row in published]
    grounded = json.loads((output / 'grounded-models/index.json').read_text(encoding='utf-8'))
    full.extend(
        {'path': row['path'], 'kind': 'facade', 'payload': row['payload']}
        for row in grounded
        if row['path'].startswith('/World/FittedFacades/')
        and row['payload'] is not None
        and original.GetPrimAtPath(row['path']).IsActive()
    )
    for row in full:
        row['archive'] = Path(row['payload']).with_suffix('.npz').as_posix()
    write_json(folder / 'full-index.json', full)
    excluded = np.array(
        [
            asset.path == '/World/FittedPlanes' or asset.path.startswith('/World/FittedPlanes/')
            for asset in assets
        ]
    )
    with np.load(output / 'grounded-models/connections.npz') as graph:
        endpoints = graph['chunk_assets'][graph['witnesses']]
        excluded_witnesses = int(excluded[endpoints].any(axis=1).sum())
    write_json(
        folder / 'report.json',
        {
            'source_hashes': fingerprints,
            'cell': cell,
            'inactive_roots': ['/World/FittedPlanes', '/World/PointRegions'],
            'point_cloud_available': True,
            'excluded_plane_contact_witnesses': excluded_witnesses,
            'grounding_reverified': False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--cell', type=float, default=0.3)
    parser.add_argument('--name', default='angular', help='Separate presentation artifact name')
    args = parser.parse_args()
    publish(args.map, args.cell, name=args.name)


if __name__ == '__main__':
    main()
