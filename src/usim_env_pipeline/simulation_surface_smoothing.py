"""Deterministic, seam-preserving CPU smoothing of voxel boundary triangles."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .models import PipelineInputError


def _invalid_faces(
    original: NDArray[np.float64],
    candidate: NDArray[np.float32],
    faces: NDArray[np.int32],
) -> NDArray[np.bool_]:
    """Check oriented area in bounded chunks, including float32 output rounding."""
    invalid = np.empty(len(faces), dtype=np.bool_)
    for start in range(0, len(faces), 65536):
        stop = start + 65536
        before = original[faces[start:stop]]
        after = candidate[faces[start:stop]].astype(np.float64)
        reference = np.cross(before[:, 1] - before[:, 0], before[:, 2] - before[:, 0])
        current = np.cross(after[:, 1] - after[:, 0], after[:, 2] - after[:, 0])
        dot = np.einsum('ij,ij->i', reference, current)
        area_squared = np.einsum('ij,ij->i', reference, reference)
        invalid[start:stop] = ~np.isfinite(dot) | (dot <= area_squared * 1e-12)
    return invalid


def smooth_surface(
    positions: NDArray[np.float32], faces: NDArray[np.int32], cell: float
) -> NDArray[np.float32]:
    """Return new positions after four full Taubin pairs (eight alternating passes).

    Uniform edge-neighbor averaging uses lambda=0.5 followed by mu=-0.53.
    This is Taubin-style smoothing, not Open3D's distance-weighted implementation.
    Edges used other than twice pin both endpoints, preserving independent tile
    seams and nonmanifold contacts. Vertex order and triangle topology never change.
    Final displacement is capped at 0.2 * cell after the alternating passes.
    Clipping a positive pass first would bias its negative compensation outward.

    Final oriented triangle areas must remain positive relative to the originals.
    Offending vertices are deterministically halved up to eight times, then
    restored locally until all faces are valid. Already degenerate input triangles
    are rejected because a noncollapsing result cannot preserve their orientation.
    Working storage is O(vertices + faces); adjacency is CSR, not a dense matrix.
    Face checks use chunks of at most 65536 triangles. No inputs are mutated.
    """
    if not math.isfinite(cell) or cell <= 0:
        raise PipelineInputError('cell must be finite and positive')
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise PipelineInputError('positions must have shape (N, 3)')
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise PipelineInputError('faces must have shape (M, 3)')
    if not np.isfinite(positions).all():
        raise PipelineInputError('positions must be finite')
    if faces.size and (faces.min() < 0 or faces.max() >= len(positions)):
        raise PipelineInputError('face indices must reference positions')
    result = positions.copy()
    if not len(faces):
        return result
    original = positions.astype(np.float64)
    if _invalid_faces(original, result, faces).any():
        raise PipelineInputError('input triangles must have nonzero finite oriented area')

    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    fixed = np.zeros(len(positions), dtype=np.bool_)
    fixed[edges[counts != 2].ravel()] = True
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    columns = np.concatenate((edges[:, 1], edges[:, 0]))
    degree = np.bincount(rows, minlength=len(positions))
    fixed |= degree == 0
    adjacency = csr_matrix(
        (1.0 / degree[rows], (rows, columns)), shape=(len(positions), len(positions))
    )
    del edges, counts, rows, columns, degree

    current = original.copy()
    limit = 0.2 * cell
    # Leave a small rounding margin; the final float32 displacement is checked too.
    cap = limit * (1.0 - 1e-6)
    for _ in range(4):
        for coefficient in (0.5, -0.53):
            current += coefficient * (adjacency @ current - current)
            current[fixed] = original[fixed]
    displacement = current - original
    lengths = np.linalg.norm(displacement, axis=1)
    displacement *= np.minimum(1.0, cap / np.maximum(lengths, cap))[:, None]
    current = original + displacement

    result = current.astype(np.float32)
    rounded_lengths = np.linalg.norm(result.astype(np.float64) - original, axis=1)
    result[rounded_lengths > limit] = positions[rounded_lengths > limit]
    for _ in range(8):
        invalid = _invalid_faces(original, result, faces)
        if not invalid.any():
            return result
        involved = np.unique(faces[invalid])
        current[involved] = original[involved] + 0.5 * (
            result[involved].astype(np.float64) - original[involved]
        )
        result[involved] = current[involved].astype(np.float32)

    # Each unsuccessful round restores at least one moved vertex permanently.
    # Neighboring faces are rechecked because a partial rollback can affect them.
    while True:
        invalid = _invalid_faces(original, result, faces)
        if not invalid.any():
            return result
        involved = np.unique(faces[invalid])
        result[involved] = positions[involved]
