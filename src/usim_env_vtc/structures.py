"""Conservative, unidentified vertical-structure proxies supported by 2019 LiDAR."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from .native_view import Payload

if TYPE_CHECKING:
    from pxr import Usd


@dataclass(frozen=True, slots=True)
class Post:
    x: float
    y: float
    height: float
    radius: float
    supported_points: int


class Canopy(Protocol):
    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...

    @property
    def radius(self) -> float: ...


def infer_posts(
    points: NDArray[np.float64],
    trees: Sequence[Canopy],
    height: NDArray[np.float64],
) -> list[Post]:
    """Reject canopy support and broad planes; retain narrow multilevel columns."""
    from scipy.ndimage import label
    from scipy.spatial import cKDTree

    keep = (height > 1) & (height < 12)
    if trees:
        centers = np.asarray([(tree.x, tree.y) for tree in trees])
        distance, index = cKDTree(centers).query(points[:, :2], workers=4)
        radii = np.asarray([tree.radius for tree in trees])
        keep &= distance > radii[index] + 1
    cloud, h = points[keep], height[keep]
    if not len(cloud):
        return []
    origin = np.floor(cloud[:, :2].min(0)) - 1
    cells = np.floor((cloud[:, :2] - origin) / 0.3).astype(np.int64)
    shape = tuple((cells.max(0) + 2).tolist())
    occupied = np.zeros(shape, dtype=bool)
    occupied[tuple(cells.T)] = True
    groups, _ = label(occupied)
    owners = groups[tuple(cells.T)]
    order = np.argsort(owners)
    _, starts = np.unique(owners[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    posts = []
    for start, end in zip(starts, ends):
        selected = order[start:end]
        part, levels = cloud[selected], h[selected]
        if len(part) < 60:
            continue
        width = np.quantile(part[:, :2], 0.95, axis=0) - np.quantile(part[:, :2], 0.05, axis=0)
        span = float(np.quantile(levels, 0.98) - np.quantile(levels, 0.02))
        if width.max() < 0.8 and span > 1.5 and len(np.unique((levels * 2).astype(int))) >= 5:
            center = np.median(part[:, :2], axis=0)
            posts.append(
                Post(
                    float(center[0]),
                    float(center[1]),
                    float(np.quantile(levels, 0.98)),
                    max(0.05, float(width.max()) / 2),
                    len(part),
                )
            )
    return posts


def author_posts(
    world: Usd.Stage,
    posts: list[tuple[Post, float]],
    out: Path,
) -> list[Payload]:
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
        prim.GetPayloads().AddPayload(name + '.usdc', '/Cell')
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
