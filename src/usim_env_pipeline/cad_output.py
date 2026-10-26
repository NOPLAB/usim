"""Publish a CAD-only simulation view tessellated from canonical B-Rep boards."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .cad_exchange import tessellate, write_brep, write_step
from .cad_kernel import compound, shape_stats
from .cad_types import CadBoard, CadBoardReport, CadConfig
from .storage import write_json
from .usd import Geometry, LayerMetadata, author_geometry, stage_base


def publish(
    output: Path, boards: list[CadBoard], metadata: LayerMetadata, config: CadConfig
) -> list[CadBoardReport]:
    """B-Rep/STEP are authoritative; USD contains only their kernel tessellation."""
    from pxr import Sdf, Usd, UsdGeom, Vt

    folder = output / 'cad'
    (folder / 'boards').mkdir(parents=True, exist_ok=True)
    model = compound([board.shape for board in boards])
    write_brep(model, folder / 'model.brep')
    write_step(model, folder / 'model.step')
    world = stage_base(output / 'cad.usda', '/World', metadata)
    world.GetRootLayer().subLayerPaths = ['./authored.usda']
    for root in ('Regions', 'Ground', 'PointRegions'):
        world.OverridePrim(f'/World/{root}').SetActive(False)
    rows: list[CadBoardReport] = []
    for number, board in enumerate(boards):
        name = f'board_{number:04}'
        path = folder / 'boards' / f'{name}.usdc'
        positions, triangles = tessellate(board.shape, config.deflection)
        write_brep(board.shape, path.with_suffix('.brep'))
        color = [0.40, 0.43, 0.44] if abs(board.equation[2]) > 0.9 else [0.72, 0.72, 0.68]
        author_geometry(
            path,
            Geometry(
                positions.astype(np.float32),
                np.empty(0, dtype=np.int64),
                np.tile(np.array([color], dtype=np.float32), (len(positions), 1)),
                triangles,
            ),
            metadata,
        )
        stage = Usd.Stage.Open(str(path))
        mesh = UsdGeom.Mesh.Get(stage, '/Tile/Surface')
        prim = mesh.GetPrim()
        prim.RemoveProperty('source:recordIds')
        mesh.CreateNormalsAttr(
            Vt.Vec3fArray.FromNumpy(board.equation[None, :3].astype(np.float32))
        )
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.constant)
        prim.CreateAttribute('cad:authority', Sdf.ValueTypeNames.Asset, custom=True).Set(
            Sdf.AssetPath(f'./{name}.brep')
        )
        prim.CreateAttribute('cad:kernel', Sdf.ValueTypeNames.String, custom=True).Set(
            'OpenCASCADE 7.9.3'
        )
        prim.CreateAttribute('cad:sourcePlanes', Sdf.ValueTypeNames.IntArray, custom=True).Set(
            list(board.source_planes)
        )
        stage.GetRootLayer().Save()
        np.savez_compressed(
            path.with_suffix('.npz'),
            source_planes=np.asarray(board.source_planes, dtype=np.int32),
            support_ids=board.support_ids,
            equation=board.equation,
        )
        UsdGeom.Xform.Define(
            world, f'/World/CADSurfaces/{name}'
        ).GetPrim().GetPayloads().AddPayload(f'./cad/boards/{name}.usdc', '/Tile')
        stats = shape_stats(board.shape)
        rows.append(
            {
                'name': name,
                'source_planes': list(board.source_planes),
                'support_records': len(board.support_ids),
                'area': stats['area_m2'],
                'faces': stats['face_count'],
                'holes': stats['hole_count'],
                'triangles': len(triangles),
            }
        )
        print(f'CAD_BOARD_WRITTEN name={name} triangles={len(triangles)}', flush=True)
    world.GetRootLayer().Save()
    write_json(folder / 'boards.json', rows)
    return rows
