"""Closed residual slabs backed only by measured source-plane support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .plane_types import plane_basis
from .plane_types import Plane
from .simulation_planar_completion import CompletionContext
from .simulation_slab_mesh import slab_surface


@dataclass(frozen=True, slots=True)
class PlanarProxy:
    source_plane: int
    support_ids: NDArray[np.int64]
    normal: NDArray[np.float64]
    offset: float
    u: NDArray[np.float64]
    v: NDArray[np.float64]
    keys: NDArray[np.int64]
    positions: NDArray[np.float32]
    faces: NDArray[np.int32]
    source_path: Path
    observed_keys: NDArray[np.int64]
    completed: bool


def reconstruct_proxies(
    points: NDArray[np.float32],
    available: NDArray[np.bool_],
    ground_ids: NDArray[np.int64],
    plane_folder: Path,
    cell: float = 0.4,
    thickness: float = 0.16,
    completion: bool = False,
    minimum_clearance: float = 1.0,
    source_paths: list[Path] | None = None,
) -> list[PlanarProxy]:
    """Extrude occupied UV cells, never the archive's inferred mask or positions.

    Sorted archives take ownership on a private availability copy. Even one
    remaining measured record can support a cell of its original fitted plane.
    Horizontal planes need nearby measured ground below at least half their
    samples; the nearest XY ground return supplies that reference.
    """
    remaining = available.copy()
    ground = points[np.unique(ground_ids)].astype(np.float64)
    ground_tree = cKDTree(ground[:, :2]) if len(ground) else None
    context = CompletionContext(points, cKDTree(points)) if completion else None
    proxies = []
    paths = plane_folder.glob('plane_*.npz') if source_paths is None else source_paths
    for path in sorted(paths):
        with np.load(path) as archive:
            equation = archive['equation'].astype(np.float64)
            ids = np.unique(archive['ids']).astype(np.int64)
        normal = equation[:3]
        scale = np.linalg.norm(normal)
        normal = normal / scale
        offset = float(equation[3] / scale)
        ids = ids[remaining[ids]]
        support = points[ids].astype(np.float64)
        within = np.abs(support @ normal + offset) <= thickness / 2 + 1e-6
        ids, support = ids[within], support[within]
        if not len(ids):
            continue
        if abs(normal[2]) > 0.7:
            if ground_tree is None:
                continue
            distances, neighbors = ground_tree.query(support[:, :2])
            elevated = (distances <= 10.0) & (
                support[:, 2] - ground[neighbors, 2] > minimum_clearance
            )
            if np.count_nonzero(elevated) * 2 < len(ids):
                continue
        u, v = plane_basis(normal)
        keys = np.unique(
            np.floor(np.column_stack((support @ u, support @ v)) / cell).astype(np.int64),
            axis=0,
        )
        observed_keys = keys
        completed = False
        if context is not None:
            filled = context.footprint(Plane(normal, offset, ids), cell)
            if filled is not None:
                keys, completed = filled, True
        local, faces = slab_surface(keys, cell, thickness)
        positions = (
            local[:, :1] * u + local[:, 1:2] * v + (local[:, 2:3] - offset) * normal
        ).astype(np.float32)
        proxies.append(
            PlanarProxy(
                int(path.stem.removeprefix('plane_')),
                ids,
                normal,
                offset,
                u,
                v,
                keys,
                positions,
                faces,
                path,
                observed_keys,
                completed,
            )
        )
        remaining[ids] = False
    return proxies
