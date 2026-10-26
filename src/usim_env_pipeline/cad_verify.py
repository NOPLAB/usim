"""Independently audit CAD authority, hole removal, exchange units and USD derivation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .cad_exchange import read_brep, read_step, tessellate
from .cad_kernel import face_loops, face_normal, faces, make_planar_face, shape_stats, surface_area
from .cad_types import CadConfig
from .models import PipelineInputError
from .pcd import open_records, sha256, xyz


def verify(output: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    source = manifest['sources'][manifest['primary_source']]
    report = json.loads((output / 'cad-report.json').read_text(encoding='utf-8'))
    config = CadConfig(**json.loads((output / 'cad/config.json').read_text(encoding='utf-8')))
    rows = json.loads((output / 'cad/boards.json').read_text(encoding='utf-8'))
    records, _ = open_records(output / source['path'])
    points = xyz(records)
    if (
        sha256(output / source['path']) != source['sha256']
        or source['sha256'] != report['source_sha256']
        or len(records) != report['source_records']
    ):
        raise PipelineInputError('Original CAD source hash/count changed')
    model = read_brep(output / report['authority'])
    exchange = read_step(output / 'cad/model.step')
    native_stats, step_stats = shape_stats(model), shape_stats(exchange)
    if (
        native_stats['face_count'] != report['output_faces']
        or native_stats['hole_count'] != report['output_holes']
        or step_stats['face_count'] != native_stats['face_count']
        or step_stats['hole_count'] != native_stats['hole_count']
        or not np.isclose(step_stats['area_m2'], native_stats['area_m2'], rtol=1e-8)
    ):
        raise PipelineInputError('STEP units or CAD authority topology mismatch')
    stage = Usd.Stage.Open(str(output / report['world']))
    for root in ('Regions', 'Ground', 'PointRegions', 'ResidualSurfaces', 'PlanarSurfaces'):
        prim = stage.GetPrimAtPath(f'/World/{root}')
        if prim and prim.IsActive():
            raise PipelineInputError('Non-CAD mesh presentation is active')
    count, area, face_count, hole_count = 0, 0.0, 0, 0
    source_planes = []
    for row in rows:
        name = row['name']
        path = output / 'cad/boards' / name
        shape = read_brep(path.with_suffix('.brep'))
        stats = shape_stats(shape)
        if not (
            np.isclose(stats['area_m2'], row['area'], rtol=1e-9)
            and stats['face_count'] == row['faces']
            and stats['hole_count'] == row['holes']
        ):
            raise PipelineInputError(f'CAD board statistics mismatch: {name}')
        for face in faces(shape):
            for hole in face_loops(face)[1:]:
                if surface_area(make_planar_face(hole, normal=face_normal(face))) < (
                    config.hole_area - 1e-8
                ):
                    raise PipelineInputError(f'Small CAD internal wire remains: {name}')
        with np.load(path.with_suffix('.npz')) as provenance:
            plane_numbers = provenance['source_planes']
            ids = provenance['support_ids']
            equation = provenance['equation']
            expected = []
            for number in plane_numbers:
                with np.load(output / 'optimized/planes' / f'plane_{number:04}.npz') as original:
                    expected.append(original['ids'])
            if not (
                np.array_equal(np.unique(np.concatenate(expected)), ids)
                and plane_numbers.tolist() == row['source_planes']
                and len(ids) == row['support_records']
                and np.abs(points[ids] @ equation[:3] + equation[3]).max()
                <= 0.08 + config.parallel_distance + 1e-4
            ):
                raise PipelineInputError(f'CAD board source support mismatch: {name}')
            source_planes.extend(plane_numbers.tolist())
            corners = np.concatenate([loop for face in faces(shape) for loop in face_loops(face)])
            if np.abs(corners @ equation[:3] + equation[3]).max() > 1e-4:
                raise PipelineInputError(f'CAD board is not on its source-supported plane: {name}')
        mesh = UsdGeom.Mesh.Get(stage, f'/World/CADSurfaces/{name}/Surface')
        if not mesh or not mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI):
            raise PipelineInputError(f'CAD tessellation lacks collisions: {name}')
        vertices, triangles = tessellate(shape, config.deflection)
        actual_vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        actual_triangles = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        authority = mesh.GetPrim().GetAttribute('cad:authority').Get()
        if not (
            np.array_equal(actual_vertices, vertices.astype(np.float32))
            and np.array_equal(actual_triangles, triangles)
            and len(triangles) == row['triangles']
            and Path(authority.resolvedPath) == path.with_suffix('.brep').resolve()
            and not mesh.GetPrim().GetAttribute('source:recordIds')
        ):
            raise PipelineInputError(f'USD is not the CAD kernel tessellation: {name}')
        count += len(triangles)
        area += stats['area_m2']
        face_count += stats['face_count']
        hole_count += stats['hole_count']
    if not (
        len(rows) == report['output_boards']
        and len(source_planes) == len(set(source_planes))
        and count == report['display_triangles']
        and np.isclose(area, report['surface_area'], rtol=1e-9)
        and face_count == report['output_faces']
        and hole_count == report['output_holes']
    ):
        raise PipelineInputError('CAD report/source ownership mismatch')
    print(
        f'CAD_VERIFICATION_PASSED boards={len(rows)} faces={face_count} '
        f'holes={hole_count} triangles={count}',
        flush=True,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    verify(parser.parse_args().map)


if __name__ == '__main__':
    main()
