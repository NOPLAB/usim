"""Observed-plane intersections and independently supported straight boundaries."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations, product
from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from .edge_line_support import (
    SupportedLine,
    face_strips,
    plane_intersection,
    supported_intervals,
)
from .edge_plane_boundaries import observed_boundary_lines
from .edge_line_types import EdgeLineConfig as EdgeLineConfig, MeasuredEdge as MeasuredEdge
from .models import PipelineInputError
from .plane_types import Plane, PlaneConfig, plane_basis
from .planes import extract_planes, merge_planes


def _clip_core(
    line: SupportedLine, low: NDArray[np.float64], high: NDArray[np.float64]
) -> SupportedLine | None:
    """Clip an already observed segment for ownership, never create a box edge."""
    first, delta = line.endpoints[0], line.endpoints[1] - line.endpoints[0]
    begin, end = 0.0, 1.0
    for axis in range(3):
        if abs(delta[axis]) < 1e-12:
            if not low[axis] <= first[axis] < high[axis]:
                return None
            continue
        a, b = (low[axis] - first[axis]) / delta[axis], (high[axis] - first[axis]) / delta[axis]
        begin, end = max(begin, min(a, b)), min(end, max(a, b))
    if begin >= end:
        return None
    endpoints = first + np.array([begin, end])[:, None] * delta
    return SupportedLine(
        endpoints, line.witnesses, line.coverage, line.max_unsupported_gap, line.bin_origin
    )


def _ownership_parts(line: SupportedLine, config: EdgeLineConfig) -> Iterator[SupportedLine]:
    """Split an observed interval at ownership planes without a length gate."""
    grid_origin = np.asarray(config.tile_origin)
    first, delta = line.endpoints[0], np.diff(line.endpoints, axis=0)[0]
    cuts = {0.0, 1.0}
    for axis in range(3):
        if abs(delta[axis]) < 1e-12:
            continue
        start, stop = np.sort(line.endpoints[:, axis])
        begin = int(np.floor((start - grid_origin[axis]) / config.tile_size)) + 1
        end = int(np.ceil((stop - grid_origin[axis]) / config.tile_size))
        for boundary in range(begin, end):
            position = grid_origin[axis] + boundary * config.tile_size
            t = float((position - first[axis]) / delta[axis])
            if 0 < t < 1:
                cuts.add(t)
    ordered = sorted(cuts)
    for begin, end in zip(ordered[:-1], ordered[1:], strict=True):
        yield SupportedLine(
            first + np.array([begin, end])[:, None] * delta,
            line.witnesses,
            line.coverage,
            line.max_unsupported_gap,
            line.bin_origin,
        )


def region_lines(
    xyz: NDArray[np.float64],
    boundary_seeds: NDArray[np.bool_],
    config: EdgeLineConfig = EdgeLineConfig(),
    *,
    core_bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
) -> list[MeasuredEdge]:
    """Fit observed planes; declared cores require complete radius-plus-voxel context."""
    settings = PlaneConfig(
        voxel=config.voxel,
        tolerance=config.plane_tolerance,
        min_support=config.min_plane_points,
        max_planes=config.max_planes,
        iterations=config.iterations,
        seed=config.seed,
    )
    members = np.arange(len(xyz), dtype=np.int64)
    discovery = members
    if core_bounds is not None:
        low, high = core_bounds
        discovery = members[((xyz >= low - config.halo) & (xyz < high + config.halo)).all(axis=1)]
    points = xyz.astype(np.float32)
    detected = merge_planes(points, extract_planes(points, discovery, settings), settings)
    # Physical-scale surface evidence comes from the complete local cloud,
    # not from the already selected thin plane inliers.
    _, first = np.unique(np.floor(xyz[discovery] / config.voxel), axis=0, return_index=True)
    base = xyz[discovery[first]]
    outer = np.ones(len(xyz), dtype=np.bool_)
    outer[discovery] = False
    supplied = np.concatenate((base, xyz[outer]))
    _, first = np.unique(np.floor(supplied / config.voxel), axis=0, return_index=True)
    representatives = supplied[first]
    tree = KDTree(representatives, leafsize=16)
    planes = []
    for plane in detected:
        ids = discovery[
            np.abs(xyz[discovery] @ plane.normal + plane.offset) <= config.plane_tolerance
        ]
        u, v = plane_basis(plane.normal)
        uv = np.column_stack((xyz[ids] @ u, xyz[ids] @ v))
        observed = np.unique(np.floor(uv / config.voxel).astype(np.int64), axis=0)
        sampled_face = (
            np.abs(representatives @ plane.normal + plane.offset) <= config.plane_tolerance
        )
        if core_bounds is not None:
            low, high = core_bounds
            sampled_face &= ((representatives >= low) & (representatives < high)).all(axis=1)
        sample_rows = np.flatnonzero(sampled_face)
        if not len(sample_rows):
            continue
        probes = sample_rows[
            np.linspace(
                0, len(sample_rows) - 1, min(config.normal_probes, len(sample_rows)), dtype=int
            )
        ]
        neighborhoods = tree.query_ball_point(representatives[probes], config.surface_radius)
        coherent = []
        for neighbors in neighborhoods:
            local = representatives[neighbors]
            delta = local - local.mean(0)
            eigen, vectors = np.linalg.eigh(delta.T @ delta)
            coherent.append(
                len(local) >= 8
                and eigen[1] > eigen[2] * 0.02
                and eigen[0] <= 0.05 * eigen.sum()
                and abs(vectors[:, 0] @ plane.normal) >= np.cos(np.deg2rad(20))
            )
        if (
            len(ids) >= config.min_plane_points
            and len(observed) * config.voxel**2 >= config.min_plane_area
            and sampled_face.any()
            and np.mean(coherent) >= config.min_normal_coherence
        ):
            planes.append(Plane(plane.normal, plane.offset, ids))
    results = []
    for first, second in combinations(planes, 2):
        equations = np.stack(
            (np.r_[first.normal, first.offset], np.r_[second.normal, second.offset])
        )
        intersection = plane_intersection(equations[0], equations[1], config.min_dihedral)
        if intersection is None:
            continue
        origin, direction = intersection
        groups = face_strips(
            xyz,
            members,
            first,
            second,
            origin,
            direction,
            config.support,
            plane_tolerance=config.plane_tolerance,
            normal_advantage=config.normal_advantage,
        )
        for line in supported_intervals(xyz, groups, origin, direction, config.support):
            results.append(MeasuredEdge(line, 2, equations, (first.ids, second.ids)))
    for plane in planes:
        seeds = plane.ids[boundary_seeds[plane.ids]]
        equations = np.stack((np.r_[plane.normal, plane.offset], np.zeros(4)))
        for line in observed_boundary_lines(xyz, plane, seeds, config.support):
            results.append(MeasuredEdge(line, 1, equations, (plane.ids,)))
    return results


def extract_edge_lines(
    xyz: NDArray[np.float64],
    boundary_seeds: NDArray[np.bool_],
    config: EdgeLineConfig = EdgeLineConfig(),
) -> Iterator[MeasuredEdge]:
    """Visit all frozen observations in ordered cubic cores with complete local halos."""
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or xyz.dtype != np.float64
        or not len(xyz)
        or not np.isfinite(xyz).all()
        or boundary_seeds.shape != (len(xyz),)
        or boundary_seeds.dtype != np.bool_
    ):
        raise PipelineInputError(
            'Line extraction requires finite XYZ and matching boundary proposals'
        )
    # Keep face discovery fixed when moving the ownership grid. Otherwise an
    # ownership experiment changes RANSAC inputs and normal-coherence decisions.
    cells, inverse = np.unique(
        np.floor(xyz / config.tile_size).astype(np.int64),
        axis=0,
        return_inverse=True,
    )
    order = np.argsort(inverse, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(inverse[order])) + 1, len(order)]
    lookup = {tuple(cell): (starts[i], starts[i + 1]) for i, cell in enumerate(cells)}
    progress = -1
    candidates = []
    # Apply the requested length to reconciled intervals, not halo-limited pieces.
    local_config = replace(config, support=replace(config.support, min_length=config.support.step))
    margin = max(config.halo, config.surface_radius + config.voxel)
    extent = int(np.ceil(margin / config.tile_size))
    for number, cell in enumerate(cells):
        a, b = lookup[tuple(cell)]
        if b - a < config.min_plane_points:
            continue
        parts = []
        for offset in product(range(-extent, extent + 1), repeat=3):
            entry = lookup.get(tuple(cell + offset))
            if entry is not None:
                parts.append(order[entry[0] : entry[1]])
        ids = np.sort(np.concatenate(parts))
        low = cell * config.tile_size
        high = low + config.tile_size
        inside = ((xyz[ids] >= low - margin) & (xyz[ids] < high + margin)).all(1)
        ids = ids[inside]
        local = xyz[ids]
        # Every boundary query must fit inside the actual halo, not its crop perimeter.
        complete = ((local >= low - margin + 0.35) & (local < high + margin - 0.35)).all(1)
        mapped_faces: dict[int, NDArray[np.int64]] = {}
        for edge in region_lines(
            local, boundary_seeds[ids] & complete, local_config, core_bounds=(low, high)
        ):
            clipped = _clip_core(edge.line, low, high)
            if clipped is None:
                continue
            witnesses = tuple(ids[rows] for rows in clipped.witnesses)
            parents = []
            for face in edge.face_members:
                if id(face) not in mapped_faces:
                    mapped_faces[id(face)] = ids[face]
                parents.append(mapped_faces[id(face)])
            for owned in _ownership_parts(clipped, config):
                candidates.append(
                    MeasuredEdge(
                        SupportedLine(
                            owned.endpoints,
                            witnesses,
                            owned.coverage,
                            owned.max_unsupported_gap,
                            owned.bin_origin,
                        ),
                        edge.kind,
                        edge.equations,
                        tuple(parents),
                    )
                )
        bucket = (number + 1) * 10 // len(cells)
        if bucket != progress:
            print(f'EDGE_LINES_STAGE planes {number + 1}/{len(cells)}', flush=True)
            progress = bucket
    from .edge_reconciliation import reconcile_edge_lines

    yield from reconcile_edge_lines(xyz, candidates, config)
