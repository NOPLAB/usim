"""Remove contained near-parallel aliases and fuse overlapping analytic CAD boards."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from .cad_kernel import (
    common_area,
    face_loops,
    faces,
    fuse_unify,
    project_to_plane,
    remove_small_holes,
    surface_area,
)
from .cad_types import CadBoard, CadConfig, CadJoinReport


def consolidate(
    boards: list[CadBoard],
    config: CadConfig,
    source_points: NDArray[np.float32] | NDArray[np.float64],
) -> tuple[list[CadBoard], list[CadJoinReport]]:
    """Only a spatially coincident board can disappear into a larger board."""
    ranked = sorted(boards, key=lambda board: -surface_area(board.shape))
    kept: list[CadBoard] = []
    decisions: list[CadJoinReport] = []
    cosine = np.cos(np.deg2rad(config.parallel_angle))
    for small in ranked:
        points = np.concatenate([loop for face in faces(small.shape) for loop in face_loops(face)])
        support = source_points[small.support_ids]
        support = support - np.outer(
            support @ small.equation[:3] + small.equation[3], small.equation[:3]
        )
        lower, upper = points.min(0), points.max(0)
        small_area = surface_area(small.shape)
        merged = False
        for index, big in enumerate(kept):
            if abs(small.equation[:3] @ big.equation[:3]) < cosine:
                continue
            distance = float(
                max(
                    np.abs(points @ big.equation[:3] + big.equation[3]).max(),
                    np.abs(support @ big.equation[:3] + big.equation[3]).max(),
                )
            )
            if distance > config.parallel_distance:
                continue
            big_points = np.concatenate(
                [loop for face in faces(big.shape) for loop in face_loops(face)]
            )
            if (upper < big_points.min(0) - config.parallel_distance).any() or (
                lower > big_points.max(0) + config.parallel_distance
            ).any():
                continue
            origin = -big.equation[:3] * big.equation[3]
            projected = project_to_plane(small.shape, origin, big.equation[:3])
            area = surface_area(projected)
            common = common_area(big.shape, projected)
            if common < min(area, small_area) * 0.02:
                continue
            contained = common / area >= config.containment
            shape = (
                big.shape
                if contained
                else remove_small_holes(fuse_unify([big.shape, projected]), config.hole_area)
            )
            kept[index] = replace(
                big,
                shape=shape,
                source_planes=tuple(sorted((*big.source_planes, *small.source_planes))),
                support_ids=np.union1d(big.support_ids, small.support_ids),
            )
            decisions.append(
                {
                    'smaller': small.source_planes[0],
                    'larger': big.source_planes[0],
                    'kind': 'contained' if contained else 'fused',
                    'max_projection_m': distance,
                    'overlap_m2': common,
                    'projected_m2': area,
                }
            )
            merged = True
            break
        if not merged:
            kept.append(small)
    return kept, decisions
