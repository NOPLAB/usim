"""Original facade recipes for buildings represented only by source cuboid envelopes."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

from .architecture_geometry import _Recipe

if TYPE_CHECKING:
    from pxr import Usd


def building_dimensions(prim: Usd.Prim) -> tuple[float, float, float] | None:
    """Identify large cuboid envelopes, excluding doors, thin walls and small props."""
    from pxr import UsdGeom

    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    size = (
        float(matrix.GetRow3(0).GetLength()),
        float(matrix.GetRow3(1).GetLength()),
        float(matrix.GetRow3(2).GetLength()),
    )
    return size if min(size) >= 4 else None


def author_building(
    stage: Usd.Stage, path: str, physical_size: tuple[float, float, float]
) -> None:
    """Fit independent window bays and floor bands inside a unit cuboid, preserving its collider."""
    from pxr import Sdf, UsdGeom, UsdShade

    recipe = _Recipe(stage, path, (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    recipe.box('CollisionEnvelope', (0, 0, 0), (1, 1, 1))
    UsdGeom.Imageable(stage.GetPrimAtPath(path + '/CollisionEnvelope')).CreateVisibilityAttr(
        UsdGeom.Tokens.invisible
    )
    floors = max(1, ceil(physical_size[2] / 3.5))
    for side in range(4):
        axis = side // 2
        along = 1 - axis
        thickness = 0.06 / physical_size[axis]
        fixed = 0 if side % 2 == 0 else 1 - thickness
        columns = max(2, ceil(physical_size[along] / 3.6))
        pier = min(0.12 / physical_size[along], 0.15 / columns)
        for level in range(floors):
            bottom, top = level / floors, (level + 1) / floors
            for column in range(columns):
                a, b = column / columns, (column + 1) / columns
                prefix = f'Wall_{side}_{level}_{column}'
                for label, x0, x1, z0, z1, material in (
                    ('Pier', a, a + pier, bottom, top, 'stone'),
                    ('Sill', a + pier, b, bottom, bottom + 0.24 / floors, 'stone'),
                    ('Header', a + pier, b, top - 0.15 / floors, top, 'stone'),
                    ('Window', a + pier, b, bottom + 0.24 / floors, top - 0.15 / floors, 'glass'),
                ):
                    low, high = [0.0, 0.0, z0], [1.0, 1.0, z1]
                    low[axis], high[axis] = fixed, fixed + thickness
                    low[along], high[along] = x0, x1
                    recipe.box(prefix + '_' + label, low, high, material, False)
    slab = min(0.18 / physical_size[2], 0.1)
    recipe.box('Roof', (0, 0, 1 - slab), (1, 1, 1), 'floor', False)
    recipe.box('Plinth', (0, 0, 0), (1, 1, slab), 'floor', False)
    glass = UsdShade.Shader(stage.GetPrimAtPath(path + '/Looks/glass/Surface'))
    glass.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(1)
    glass.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.24)
    glass.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.25)
    recipe.root.SetCustomDataByKey('genericFacadeNotSurveyed', True)
