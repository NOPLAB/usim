"""Screened harmonic smoothing on the shared terrain lattice."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError
from .simulation_ground_config import GroundConfig


def smooth_heights(
    xy: NDArray[np.float64],
    corners: NDArray[np.int64],
    heights: NDArray[np.float64],
    config: GroundConfig,
) -> tuple[NDArray[np.float64], float]:
    """Smooth measured anchors and harmonically fill gaps without introducing holes.

    Minimize squared anchor error plus neighboring residual-height differences.
    Subtracting the fitted reference plane preserves a measured slope, including
    at the exterior. Missing nodes receive no data weight or fabricated source IDs.
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import cg

    anchored = np.isfinite(heights)
    basis = np.column_stack((xy - xy.mean(0), np.ones(len(xy))))
    coefficients = np.linalg.lstsq(basis[anchored], heights[anchored], rcond=None)[0]
    if np.linalg.norm(coefficients[:2]) >= config.max_slope:
        raise PipelineInputError('Measured floor cannot form a continuous slope-bounded terrain')
    reference = basis @ coefficients
    edges = np.concatenate(
        [corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 3]], corners[:, [3, 0]]]
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    first, second = edges.T
    degree = np.bincount(edges.ravel(), minlength=len(xy)).astype(np.float64)
    adjacency = coo_matrix(
        (
            -np.ones(2 * len(edges)),
            (np.r_[first, second], np.r_[second, first]),
        ),
        shape=(len(xy), len(xy)),
    ).tocsr()
    laplacian = adjacency + diags(degree)
    rhs = np.where(anchored, heights - reference, 0)
    strength = 1.0
    residual = np.zeros(len(xy))
    # Select the first scale whose complete surface satisfies the slope contract.
    # Increasing regularization approaches the admissible measured reference plane.
    while True:
        diagonal = strength * degree + anchored
        residual, status = cg(
            strength * laplacian + diags(anchored.astype(np.float64)),
            rhs,
            x0=residual,
            M=diags(1 / diagonal),
            rtol=1e-9,
            atol=1e-10,
            maxiter=4000,
        )
        if status != 0:
            raise PipelineInputError(f'Continuous terrain smoothing did not converge: {status}')
        result = reference + residual
        z = result[corners]
        gradients = (
            np.stack(
                (
                    np.column_stack((z[:, 1] - z[:, 0], z[:, 2] - z[:, 1])),
                    np.column_stack((z[:, 2] - z[:, 3], z[:, 3] - z[:, 0])),
                ),
                axis=1,
            )
            / config.cell
        )
        if np.linalg.norm(gradients, axis=2).max() <= config.max_slope:
            return result, strength
        strength *= 4
