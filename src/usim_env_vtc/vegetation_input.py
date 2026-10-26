"""Verified VTC cloud registration and existing-building exclusion."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from numpy.typing import NDArray

from usim_env_pipeline.models import PipelineInputError

from .pcd import open_records, sha256, xyz
from .sources import PUBLIC_HASHES

if TYPE_CHECKING:
    from pxr import Usd


@dataclass(frozen=True, slots=True)
class VegetationConfig:
    source: Path
    registration: Path
    world: Path
    index: Path
    terrain: Path
    prototype: Path
    out: Path
    with_structures: bool = False
    bounds: tuple[float, float, float, float] = (-340, 30, -20, 290)


class Registration(TypedDict):
    source19_to_vtc: list[list[float]]
    source_sha256: str


def read_points(config: VegetationConfig) -> tuple[NDArray[np.float64], str]:
    registration: Registration = json.loads(config.registration.read_text(encoding='utf-8'))
    if sha256(config.source) != PUBLIC_HASHES['map_tc19_o085_f-04_t05.pcd']:
        raise PipelineInputError('Tree extraction requires the original verified 2019 PCD')
    matrix = np.asarray(registration['source19_to_vtc'], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise PipelineInputError('Require validated 2019-to-VTC homogeneous registration')
    records, _ = open_records(config.source)
    chunks = []
    low_x, low_y, high_x, high_y = config.bounds
    for start in range(0, len(records), 250000):
        source = xyz(records[start : start + 250000]).astype(np.float64)
        points = source @ matrix[:3, :3].T + matrix[:3, 3]
        selected = (
            (points[:, 0] > low_x)
            & (points[:, 0] < high_x)
            & (points[:, 1] > low_y)
            & (points[:, 1] < high_y)
        )
        chunks.append(points[selected])
    return np.concatenate(chunks), registration['source_sha256']


def exclude_buildings(
    base: Usd.Stage,
    points: NDArray[np.float64],
    bounds: tuple[float, float, float, float],
) -> NDArray[np.float64]:
    """Exclude existing nonvegetation footprints without rewriting the input points."""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    low_x, low_y, high_x, high_y = bounds
    filtered = points
    for prim in base.GetPrimAtPath('/World/Structures/VTC').GetChildren():
        if any(name in prim.GetName() for name in ('Tree', 'Pine', 'Bush')):
            continue
        extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = np.asarray(extent.GetMin()), np.asarray(extent.GetMax())
        if (
            np.isfinite(lo).all()
            and (hi - lo > 2).all()
            and lo[0] < high_x
            and hi[0] > low_x
            and lo[1] < high_y
            and hi[1] > low_y
        ):
            inside = ((filtered[:, :2] >= lo[:2] - 0.5) & (filtered[:, :2] <= hi[:2] + 0.5)).all(1)
            filtered = filtered[~inside]
    return filtered
