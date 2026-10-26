"""Publish a separately auditable planar presentation over immutable measured layers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
from typing import TypedDict

import numpy as np

from .models import PipelineInputError
from .pcd import open_records, sha256, xyz
from .plane_join import join_planes
from .planes import Patch, PlaneConfig, covered_faces, extract_planes, patch_mesh
from .storage import write_json
from .usd import Geometry, LayerMetadata, author_geometry, stage_base


class PlanarReport(TypedDict):
    source_sha256: str
    source_records: int
    planes: int
    supported_records: int
    input_triangles: int
    residual_triangles: int
    planar_triangles: int
    output_triangles: int
    planar_area: float
    interpolated_area: float
    enclosed_interpolated_area: float
    gaussian_bridged_area: float
    join_candidates: int
    joined_pairs: int
    max_support_displacement: float
    world: str


def author_patch(path: Path, patch: Patch, metadata: LayerMetadata) -> None:
    """Generated corners have plane support, not falsely assigned original vertex IDs."""
    from pxr import Sdf, Usd, UsdGeom, Vt

    color = [0.35, 0.37, 0.38] if abs(patch.plane.normal[2]) > 0.9 else [0.63, 0.62, 0.58]
    author_geometry(
        path,
        Geometry(
            patch.positions,
            np.empty(0, dtype=np.int64),
            np.tile(np.array([color], dtype=np.float32), (len(patch.positions), 1)),
            patch.faces,
        ),
        metadata,
    )
    stage = Usd.Stage.Open(str(path))
    mesh = UsdGeom.Mesh.Get(stage, '/Tile/Surface')
    mesh.GetPrim().RemoveProperty('source:recordIds')
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(patch.plane.normal[None].astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.constant)
    mesh.GetPrim().CreateAttribute('plane:equation', Sdf.ValueTypeNames.Double4, custom=True).Set(
        tuple(patch.plane.normal) + (patch.plane.offset,)
    )
    mesh.GetPrim().CreateAttribute(
        'plane:supportRecords', Sdf.ValueTypeNames.Int64, custom=True
    ).Set(len(patch.plane.ids))
    stage.GetRootLayer().Save()


def optimize(output: Path, config: PlaneConfig = PlaneConfig()) -> PlanarReport:
    """Fit on CPU, preserve recesses and residual geometry, and write optimized.usda."""
    from pxr import Usd, UsdGeom

    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    source = manifest['sources'][manifest['primary_source']]
    if sha256(output / source['path']) != source['sha256']:
        raise PipelineInputError('Planar source archive hash mismatch')
    records, _ = open_records(output / source['path'])
    points = xyz(records)
    folder = output / 'optimized'
    folder.mkdir(exist_ok=True)
    (folder / 'planes').mkdir(exist_ok=True)
    (folder / 'residual').mkdir(exist_ok=True)
    fits = []
    for path in sorted((output / 'records').glob('*.npz')):
        with np.load(path) as archive:
            ids = archive['ids']
        ids = ids[np.isfinite(points[ids]).all(1)]
        found = extract_planes(points, ids, config)
        fits.extend(found)
        print(f'PLANAR_REGION_FITTED name={path.stem} planes={len(found)}', flush=True)
    joined = join_planes(points, fits, config)
    fits = joined.planes
    np.savez_compressed(
        folder / 'joins.npz',
        equations=joined.equations,
        edges=joined.edges,
        accepted=joined.accepted,
    )
    print(
        f'GAUSSIAN_JOINS_PASSED candidates={len(joined.edges)} '
        f'accepted={int(joined.accepted.sum())} workers={config.workers}',
        flush=True,
    )
    metadata = LayerMetadata(
        source['license'],
        source['attribution'],
        manifest['source_catalog'],
        manifest['frame'],
        'measured fitted planes; bounded local interpolation; neutral original materials',
    )
    world = stage_base(output / 'optimized.usda', '/World', metadata)
    world.GetRootLayer().subLayerPaths = ['./authored.usda']
    for root in ('Regions', 'Ground', 'PointRegions'):
        world.OverridePrim(f'/World/{root}').SetActive(False)
    patches: list[Patch] = []
    owner = np.full(len(points), -1, dtype=np.int32)
    maximum, area, added, enclosed_area, bridge_area = 0.0, 0.0, 0.0, 0.0, 0.0
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        computed = list(pool.map(lambda fit: patch_mesh(points, fit, config), fits))
    for patch in computed:
        if patch is None:
            continue
        fit = patch.plane
        number = len(patches)
        name = f'plane_{number:04}'
        patches.append(patch)
        owner[fit.ids] = number
        path = folder / 'planes' / f'{name}.usdc'
        author_patch(path, patch, metadata)
        np.savez_compressed(
            path.with_suffix('.npz'),
            ids=fit.ids,
            equation=np.append(fit.normal, fit.offset),
            u=patch.u,
            v=patch.v,
            lower=patch.lower,
            mask=patch.mask,
            observed=patch.observed,
            protected=patch.protected,
            enclosed=patch.enclosed,
            bridged=patch.bridged,
            links=fit.links,
            link_models=fit.link_models,
            positions=patch.positions,
            faces=patch.faces,
        )
        UsdGeom.Xform.Define(
            world, f'/World/PlanarSurfaces/{name}'
        ).GetPrim().GetPayloads().AddPayload(f'./optimized/planes/{name}.usdc', '/Tile')
        maximum = max(maximum, float(np.abs(points[fit.ids] @ fit.normal + fit.offset).max()))
        area += float(patch.mask.sum() * config.cell**2)
        added += float((patch.mask & ~patch.observed).sum() * config.cell**2)
        enclosed_area += float(patch.enclosed.sum() * config.cell**2)
        bridge_area += float(patch.bridged.sum() * config.cell**2)
        print(f'PLANAR_PATCH_WRITTEN name={name} faces={len(patch.faces)}', flush=True)
    input_faces, residual_faces = 0, 0
    entries = json.loads((output / 'authored-index.json').read_text(encoding='utf-8'))
    for entry in entries:
        if not entry['path'].startswith(('/World/Regions/', '/World/Ground/')):
            continue
        stage = Usd.Stage.Open(str(output / entry['payload']))
        mesh = UsdGeom.Mesh.Get(stage, '/Tile/Surface')
        ids = np.asarray(mesh.GetPrim().GetAttribute('source:recordIds').Get(), dtype=np.int64)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
        positions = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        colors = np.asarray(mesh.GetDisplayColorPrimvar().Get(), dtype=np.float32)
        keep = np.ones(len(faces), dtype=bool)
        labels = owner[ids][faces]
        for number in np.unique(labels):
            if number < 0:
                continue
            selected = np.flatnonzero((labels == number).any(1) & keep)
            removed = covered_faces(positions, faces[selected], patches[number], config)
            keep[selected[removed]] = False
        input_faces += len(faces)
        residual_faces += int(keep.sum())
        if not keep.any():
            continue
        used, inverse = np.unique(faces[keep], return_inverse=True)
        name = entry['path'].replace('/World/', '').replace('/', '_')
        author_geometry(
            folder / 'residual' / f'{name}.usdc',
            Geometry(
                positions[used], ids[used], colors[used], inverse.reshape(-1, 3).astype(np.int32)
            ),
            metadata,
        )
        UsdGeom.Xform.Define(
            world, f'/World/ResidualSurfaces/{name}'
        ).GetPrim().GetPayloads().AddPayload(f'./optimized/residual/{name}.usdc', '/Tile')
    planar_faces = sum(len(patch.faces) for patch in patches)
    report: PlanarReport = {
        'source_sha256': source['sha256'],
        'source_records': len(points),
        'planes': len(patches),
        'supported_records': int((owner >= 0).sum()),
        'input_triangles': input_faces,
        'residual_triangles': residual_faces,
        'planar_triangles': planar_faces,
        'output_triangles': residual_faces + planar_faces,
        'planar_area': area,
        'interpolated_area': added,
        'enclosed_interpolated_area': enclosed_area,
        'gaussian_bridged_area': bridge_area,
        'join_candidates': len(joined.edges),
        'joined_pairs': int(joined.accepted.sum()),
        'max_support_displacement': maximum,
        'world': 'optimized.usda',
    }
    world.GetRootLayer().Save()
    write_json(output / 'planar-report.json', report)
    write_json(folder / 'config.json', asdict(config))
    print('PLANAR_OPTIMIZATION_PASSED ' + json.dumps(report), flush=True)
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise PipelineInputError('Planar workers must be positive')
    optimize(args.map, PlaneConfig(workers=args.workers))


if __name__ == '__main__':
    main()
