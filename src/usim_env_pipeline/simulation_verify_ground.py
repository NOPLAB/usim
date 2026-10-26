"""Audit continuous ground topology and the measured-anchor smoothing equation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError
from .simulation_ground import GroundConfig, Terrain


@dataclass(frozen=True, slots=True)
class GroundAudit:
    area_m2: float
    slope_max: float
    center_support_max_m: float
    inferred_nodes: int
    boundary_edges: int
    anchor_error_max_m: float


def verify_ground(
    terrain: Terrain,
    points: NDArray[np.float32],
    measured_ids: NDArray[np.int64],
    config: GroundConfig,
) -> GroundAudit:
    """Require one connected disk, shared lattice corners and smooth supported heights."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    from .simulation_ground_fit import fit_nodes

    support = terrain.support_ids
    valid = support >= 0
    anchored = valid.any(1)
    if (
        not np.isin(support[valid], measured_ids).all()
        or (valid.sum(1)[anchored] < config.min_neighbors).any()
        or len(np.unique(terrain.keys, axis=0)) != len(terrain.keys)
        or not np.isfinite(terrain.positions).all()
        or not np.isfinite(terrain.smoothing_strength)
        or terrain.smoothing_strength < 1
    ):
        raise PipelineInputError('Ground node support or shared vertex uniqueness failed')
    if not np.allclose(terrain.positions[:, :2], terrain.keys * config.cell, atol=1e-5):
        raise PipelineInputError('Ground positions do not match the shared lattice')
    expected_keys = terrain.cells[:, None] + np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    if not np.array_equal(terrain.keys[terrain.corner_indices], expected_keys):
        raise PipelineInputError('Ground cell corners do not share lattice nodes')
    expected_faces = np.stack(
        (terrain.corner_indices[:, [0, 1, 2]], terrain.corner_indices[:, [0, 2, 3]]), axis=1
    ).reshape(-1, 3)
    if not np.array_equal(terrain.faces, expected_faces):
        raise PipelineInputError('Ground faces do not follow the continuous cell coverage')
    edges = np.sort(
        np.concatenate(
            [terrain.faces[:, [0, 1]], terrain.faces[:, [1, 2]], terrain.faces[:, [2, 0]]]
        ),
        axis=1,
    )
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(terrain.positions), len(terrain.positions)),
    )
    components = connected_components(graph, directed=False, return_labels=False)
    boundary = edges[counts == 1]
    boundary_degrees = np.bincount(boundary.ravel(), minlength=len(terrain.positions))
    euler = len(terrain.positions) - len(edges) + len(terrain.faces)
    if (
        components != 1
        or euler != 1
        or (counts > 2).any()
        or not len(boundary)
        or (boundary_degrees[boundary_degrees > 0] != 2).any()
    ):
        raise PipelineInputError('Ground must be one connected mesh without interior holes')
    triangles = terrain.positions[terrain.faces].astype(np.float64)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    slopes = np.linalg.norm(cross[:, :2], axis=1) / cross[:, 2]
    if (cross[:, 2] <= 0).any() or (slopes > config.max_slope + 1e-4).any():
        raise PipelineInputError('Simulation terrain has inverted faces or excessive slopes')
    xy = terrain.keys.astype(np.float64) * config.cell
    heights, provenance = fit_nodes(
        points[measured_ids].astype(np.float64), measured_ids, xy, config
    )
    expected_anchors = np.isfinite(heights)
    provenance[~expected_anchors] = -1
    if not np.array_equal(support, provenance):
        raise PipelineInputError('Ground anchor provenance does not match measured support')
    basis = np.column_stack((xy - xy.mean(0), np.ones(len(xy))))
    reference = basis @ np.linalg.lstsq(basis[anchored], heights[anchored], rcond=None)[0]
    residual = terrain.positions[:, 2].astype(np.float64) - reference
    # Independently evaluate the Euler-Lagrange equation, without using the solver.
    sides = edges[(terrain.keys[edges[:, 0]] != terrain.keys[edges[:, 1]]).sum(1) == 1]
    divergence = np.zeros(len(xy))
    delta = residual[sides[:, 0]] - residual[sides[:, 1]]
    np.add.at(divergence, sides[:, 0], delta)
    np.add.at(divergence, sides[:, 1], -delta)
    divergence[anchored] += (
        terrain.positions[anchored, 2] - heights[anchored]
    ) / terrain.smoothing_strength
    if np.abs(divergence).max() > 2e-5:
        raise PipelineInputError('Ground does not satisfy the smooth gap-filling equation')
    anchor_error = np.abs(terrain.positions[anchored, 2] - heights[anchored]).max()
    nearest = cKDTree(points[measured_ids, :2]).query(triangles.mean(1)[:, :2])[0]
    return GroundAudit(
        float(cross[:, 2].sum() / 2),
        float(slopes.max()),
        float(nearest.max()),
        int((~anchored).sum()),
        len(boundary),
        float(anchor_error),
    )
