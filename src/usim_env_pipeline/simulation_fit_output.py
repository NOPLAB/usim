"""Render/collision-identical fitted surfaces with source provenance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom, Vt

from .simulation_curve_surface import CurveSurface
from .usd import Geometry, LayerMetadata, author_geometry


def publish_surface(path: Path, surface: CurveSurface, metadata: LayerMetadata) -> None:
    """Author a fitted mesh and area-weighted continuous surface normals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    author_geometry(
        path,
        Geometry(
            surface.positions,
            np.empty(0, np.int64),
            np.tile(np.array([[0.43, 0.46, 0.40]], np.float32), (len(surface.positions), 1)),
            surface.faces,
        ),
        metadata,
    )
    stage = Usd.Stage.Open(str(path))
    mesh = UsdGeom.Mesh.Get(stage, '/Tile/Surface')
    triangles = surface.positions[surface.faces].astype(np.float64)
    vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.zeros(surface.positions.shape, np.float64)
    for corner in range(3):
        np.add.at(normals, surface.faces[:, corner], vectors)
    lengths = np.linalg.norm(normals, axis=1)
    normals[lengths > 0] /= lengths[lengths > 0, None]
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    prim = mesh.GetPrim()
    prim.RemoveProperty('source:recordIds')
    prim.CreateAttribute('source:provenance', Sdf.ValueTypeNames.Asset, custom=True).Set(
        Sdf.AssetPath('./' + path.with_suffix('.npz').name)
    )
    stage.GetRootLayer().Save()
