"""Audit fitted-plane provenance, interpolation boundaries and composed collision geometry."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .models import PipelineInputError
from .pcd import open_records, sha256, xyz
from .plane_join import gaussian_cost, link_costs
from .plane_repair import link_seed
from .planes import PlaneConfig


def verify(output: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.ndimage import binary_closing, binary_fill_holes, label
    from scipy.spatial import cKDTree

    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    source = manifest['sources'][manifest['primary_source']]
    report = json.loads((output / 'planar-report.json').read_text(encoding='utf-8'))
    config = PlaneConfig(
        **json.loads((output / 'optimized/config.json').read_text(encoding='utf-8'))
    )
    if sha256(output / source['path']) != report['source_sha256'] or (
        report['source_sha256'] != source['sha256']
    ):
        raise PipelineInputError('Original planar source hash changed')
    records, _ = open_records(output / source['path'])
    points = xyz(records)
    with np.load(output / 'optimized/joins.npz') as joins:
        equations, edges, accepted = joins['equations'], joins['edges'], joins['accepted']
    if not (
        len(edges) == report['join_candidates']
        and int(accepted.sum()) == report['joined_pairs']
        and np.allclose(gaussian_cost(edges[:, 2:4], edges[:, 4], config), edges[:, 5])
        and (edges[accepted, 5] <= config.join_cost).all()
    ):
        raise PipelineInputError('Gaussian join cost/report mismatch')
    accepted_pairs = {tuple(row[:2].astype(int)) for row in edges[accepted]}
    stage = Usd.Stage.Open(str(output / 'optimized.usda'))
    if len(points) != report['source_records']:
        raise PipelineInputError('Planar source count mismatch')
    for root in ('Regions', 'Ground', 'PointRegions'):
        if stage.GetPrimAtPath(f'/World/{root}').IsActive():
            raise PipelineInputError('Original overlapping presentation is active')
    planar_faces, residual_faces, area, added = 0, 0, 0.0, 0.0
    enclosed_area, bridge_area = 0.0, 0.0
    for number in range(report['planes']):
        name = f'plane_{number:04}'
        mesh = UsdGeom.Mesh.Get(stage, f'/World/PlanarSurfaces/{name}/Surface')
        with np.load(output / 'optimized/planes' / f'{name}.npz') as archive:
            equation, ids = archive['equation'], archive['ids']
            mask, observed, protected = archive['mask'], archive['observed'], archive['protected']
            enclosed, bridged = archive['enclosed'], archive['bridged']
            links, models = archive['links'], archive['link_models']
            vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            if not (
                np.array_equal(vertices, archive['positions'])
                and np.array_equal(faces, archive['faces'])
                and np.abs(points[ids] @ equation[:3] + equation[3]).max()
                <= config.tolerance + 1e-5
                and np.abs(vertices @ equation[:3] + equation[3]).max() < 1e-4
                and not (mask & protected & ~observed).any()
                and (mask | observed).sum() == mask.sum()
            ):
                raise PipelineInputError(f'Invalid plane support or protected boundary: {name}')
            basis = np.column_stack((archive['u'], archive['v']))
            source_cells = np.floor(points[ids] @ basis / config.cell).astype(np.int64)
            actual_observed = np.zeros_like(observed)
            actual_observed[tuple((source_cells - archive['lower']).T)] = True
            if not np.array_equal(observed, actual_observed):
                raise PipelineInputError(f'Observed footprint differs from source: {name}')
            if len(links) and not (
                np.isin(links, ids).all()
                and all(tuple(pair) in accepted_pairs for pair in models)
                and (
                    link_costs(points, links, equations[models], config) <= config.join_cost
                ).all()
            ):
                raise PipelineInputError(
                    f'Gaussian bridge lacks compatible source anchors: {name}'
                )
            bridge_seed = link_seed(
                points[links] @ basis / config.cell, archive['lower'], observed
            )
            radius = int(np.ceil(config.bridge_radius / config.cell))
            x, y = np.ogrid[-radius : radius + 1, -radius : radius + 1]
            disk = x * x + y * y <= radius * radius
            seed = observed | bridge_seed
            closed = binary_closing(seed, structure=disk) | seed
            local = binary_closing(observed, structure=disk) | observed
            holes, _ = label(binary_fill_holes(closed) & ~closed)
            eligible = np.bincount(holes.ravel()) * config.cell**2 <= config.hole_area
            eligible[0] = False
            if not (
                np.array_equal(enclosed, eligible[holes] & mask)
                and np.array_equal(bridged, (closed & ~local) & mask & ~observed)
            ):
                raise PipelineInputError(f'Uncertified interior/bridge completion: {name}')
            repaired_cells = np.argwhere(mask & ~observed & ~enclosed)
            if len(repaired_cells):
                distance = cKDTree(np.argwhere(seed) * config.cell).query(
                    repaired_cells * config.cell, workers=1
                )[0]
                if distance.max() > config.bridge_radius + config.cell + 1e-5:
                    raise PipelineInputError(f'Interpolation exceeds local support: {name}')
            triangles = vertices[faces]
            cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            if not (cross @ equation[:3] > 0).all():
                raise PipelineInputError(f'Plane winding is inconsistent: {name}')
            expected = float(mask.sum() * config.cell**2)
            if not np.isclose(np.linalg.norm(cross, axis=1).sum() / 2, expected, rtol=5e-4):
                raise PipelineInputError(f'Plane area differs from supported footprint: {name}')
            planar_faces += len(faces)
            area += expected
            added += float((mask & ~observed).sum() * config.cell**2)
            enclosed_area += float(enclosed.sum() * config.cell**2)
            bridge_area += float(bridged.sum() * config.cell**2)
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/ResidualSurfaces')):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        ids = np.asarray(prim.GetAttribute('source:recordIds').Get(), dtype=np.int64)
        if not np.array_equal(mesh.GetPointsAttr().Get(), points[ids]):
            raise PipelineInputError(f'Residual source vertices changed: {prim.GetPath()}')
        residual_faces += len(mesh.GetFaceVertexCountsAttr().Get())
    for root in ('PlanarSurfaces', 'ResidualSurfaces'):
        for prim in Usd.PrimRange(stage.GetPrimAtPath(f'/World/{root}')):
            if prim.IsA(UsdGeom.Mesh) and not (
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'none'
            ):
                raise PipelineInputError(
                    f'Optimized triangle collision is missing: {prim.GetPath()}'
                )
    if not (
        planar_faces == report['planar_triangles']
        and residual_faces == report['residual_triangles']
        and planar_faces + residual_faces == report['output_triangles']
        and np.isclose(area, report['planar_area'])
        and np.isclose(added, report['interpolated_area'])
        and np.isclose(enclosed_area, report['enclosed_interpolated_area'])
        and np.isclose(bridge_area, report['gaussian_bridged_area'])
    ):
        raise PipelineInputError('Optimized composition/report mismatch')
    print(
        'PLANAR_VERIFIED ' + json.dumps({'report': report, 'config': asdict(config)}), flush=True
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    args = parser.parse_args()
    verify(args.map)


if __name__ == '__main__':
    main()
