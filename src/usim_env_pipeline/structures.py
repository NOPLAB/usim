"""Place measured narrow-column proxies into collidable regional payloads."""

from __future__ import annotations

from pathlib import Path
import os
from typing import TYPE_CHECKING

import numpy as np

from .features import Post
from .models import Bounds

if TYPE_CHECKING:
    from pxr import Usd


def author_posts(
    world: Usd.Stage,
    posts: list[tuple[Post, float]],
    out: Path,
) -> list[Bounds]:
    """Author conservative cylindrical proxies as bounded, collidable cell payloads."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    cells: dict[tuple[int, int], list[tuple[Post, float]]] = {}
    for post, z in posts:
        key = (int(np.floor(post.x / 30)), int(np.floor(post.y / 30)))
        cells.setdefault(key, []).append((post, z))
    entries = []
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for key, members in cells.items():
        name = f'posts_{key[0]}_{key[1]}'.replace('-', 'm')
        cell = Usd.Stage.CreateNew(str((out / (name + '.usdc')).resolve()))
        cell.SetDefaultPrim(UsdGeom.Xform.Define(cell, '/Cell').GetPrim())
        UsdGeom.SetStageUpAxis(cell, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(cell, 1)
        for index, (post, z) in enumerate(members):
            cylinder = UsdGeom.Cylinder.Define(cell, f'/Cell/Post_{index}')
            cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
            cylinder.CreateHeightAttr(post.height)
            cylinder.CreateRadiusAttr(post.radius)
            cylinder.CreateExtentAttr(
                [
                    (-post.radius, -post.radius, -post.height / 2),
                    (post.radius, post.radius, post.height / 2),
                ]
            )
            cylinder.CreateDisplayColorAttr([Gf.Vec3f(0.38, 0.4, 0.42)])
            cylinder.AddTranslateOp().Set((post.x, post.y, z + post.height / 2))
            UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
        cell.GetRootLayer().Save()
        path = '/World/InferredStructures/' + name
        prim = world.DefinePrim(path, 'Xform')
        payload = os.path.relpath(
            out / (name + '.usdc'), Path(world.GetRootLayer().realPath).parent
        ).replace('\\', '/')
        prim.GetPayloads().AddPayload(payload, '/Cell')
        cache.Clear()
        bounds = cache.ComputeWorldBound(cell.GetDefaultPrim()).ComputeAlignedRange()
        entries.append(
            {
                'path': path,
                'min': list(bounds.GetMin()),
                'max': list(bounds.GetMax()),
            }
        )
    return entries
