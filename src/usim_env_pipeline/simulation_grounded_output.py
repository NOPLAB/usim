"""Author non-destructive whole-component subsets with unchanged appearance."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypedDict, assert_never

import numpy as np
from numpy.typing import NDArray

from .storage import write_json
from .usd import LayerMetadata


@dataclass(frozen=True, slots=True)
class CachedMesh:
    path: str
    payload: str
    archive: Path
    matrix: NDArray[np.float64]
    labels: Path
    order: Path
    face_count: int


class GroundedTile(TypedDict):
    path: str
    source_payload: str
    payload: str | None
    mask: str
    labels: str
    source_faces: int
    kept_faces: int


class AttributeInterpolation(str, Enum):
    VERTEX = 'vertex'
    FACE = 'uniform'
    CONSTANT = 'constant'


def subset_attribute(
    values: NDArray[np.float32],
    interpolation: AttributeInterpolation,
    vertices: NDArray[np.int64],
    faces: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Preserve the original USD attribute domain when filtering geometry."""
    match interpolation:
        case AttributeInterpolation.VERTEX:
            return values[vertices]
        case AttributeInterpolation.FACE:
            return values[faces]
        case AttributeInterpolation.CONSTANT:
            return values
        case unreachable:
            assert_never(unreachable)


def publish(
    output: Path,
    sources: list[CachedMesh],
    components: NDArray[np.bool_],
    merged: NDArray[np.int32],
    metadata: LayerMetadata,
    *,
    angular: bool = False,
    angular_name: str = 'angular',
) -> list[GroundedTile]:
    """Override filtered payloads only; inherit the untouched floor and original transforms."""
    from pxr import Sdf, Usd, UsdGeom, Vt

    from .simulation_curve_surface import CurveSurface
    from .simulation_fit_output import publish_surface
    from .usd import stage_base

    folder = output / (f'{angular_name}-grounded-models' if angular else 'grounded-models')
    source_file = f'simulation-{angular_name}.usda' if angular else 'simulation-fitted.usda'
    result_file = (
        f'simulation-{angular_name}-grounded.usda' if angular else 'simulation-grounded.usda'
    )
    world = stage_base(output / result_file, '/World', metadata)
    world.SetLoadRules(Usd.StageLoadRules.LoadNone())
    world.GetRootLayer().subLayerPaths = ['./' + source_file]
    source = Usd.Stage.Open(str(output / source_file), Usd.Stage.LoadNone)
    world.GetRootLayer().customLayerData = source.GetRootLayer().customLayerData
    published: list[GroundedTile] = []
    for number, item in enumerate(sources[1:], 1):
        labels = np.load(item.labels, mmap_mode='r')
        keep = components[merged[labels]]
        mask_file = folder / f'{number:04d}-keep.npy'
        np.save(mask_file, keep)
        payload: str | None = item.payload
        if not keep.any():
            world.OverridePrim(item.path).SetActive(False)
            payload = None
        elif not keep.all():
            with np.load(item.archive) as data:
                positions, faces = data['positions'], data['faces'][keep]
            used, inverse = np.unique(faces, return_inverse=True)
            surface = CurveSurface(positions[used], inverse.reshape(-1, 3).astype(np.int32))
            path = folder / 'meshes' / f'{number:04d}.usdc'
            publish_surface(path, surface, metadata)
            part = Usd.Stage.Open(str(path))
            part.GetRootLayer().subLayerPaths = [
                os.path.relpath(output / item.payload, path.parent).replace(os.sep, '/')
            ]
            part.GetRootLayer().Save()
            np.savez_compressed(
                path.with_suffix('.npz'),
                positions=surface.positions,
                faces=surface.faces,
                source_vertices=used,
                source_faces=np.flatnonzero(keep),
            )
            payload = './' + path.relative_to(output).as_posix()
            world.OverridePrim(item.path).GetPayloads().SetPayloads(
                [Sdf.Payload(payload, '/Tile')]
            )
            appearance = UsdGeom.Mesh.Define(world, item.path + '/Surface')
            with np.load(item.labels.with_suffix('.appearance.npz')) as attributes:
                normals_type = str(attributes['normals_interpolation']) if angular else 'vertex'
                colors_type = str(attributes['colors_interpolation']) if angular else 'vertex'
                normals = subset_attribute(
                    attributes['normals'], AttributeInterpolation(normals_type), used, keep
                )
                color_values = subset_attribute(
                    attributes['colors'], AttributeInterpolation(colors_type), used, keep
                )
                appearance.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
                appearance.SetNormalsInterpolation(normals_type)
                colors = appearance.CreateDisplayColorPrimvar(colors_type)
                colors.Set(Vt.Vec3fArray.FromNumpy(color_values))
                colors.BlockIndices()
        published.append(
            {
                'path': item.path,
                'source_payload': item.payload,
                'payload': payload,
                'mask': mask_file.relative_to(output).as_posix(),
                'labels': item.labels.relative_to(output).as_posix(),
                'source_faces': item.face_count,
                'kept_faces': int(keep.sum()),
            }
        )
    world.GetRootLayer().Save()
    write_json(folder / 'index.json', published)
    return published
