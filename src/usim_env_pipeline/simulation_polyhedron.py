"""Constrained low-poly surfaces with immutable seams and contact triangles."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .simulation_curve_surface import CurveSurface
from .simulation_polyhedron_decimation import decimate
from .simulation_polyhedron_decimation import nonmanifold_edges as nonmanifold_edges
from .simulation_ground_connectivity import mesh_components
from .models import PipelineInputError
from .simulation_metrics import signed_volume


@dataclass(frozen=True, slots=True)
class Polyhedron:
    surface: CurveSurface
    reduced_components: int
    unchanged_components: int


def boundary_edges(faces: NDArray[np.int32]) -> NDArray[np.int32]:
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    return edges[counts == 1]


def topology(surface: CurveSurface) -> tuple[int, int, int]:
    edges = np.sort(
        np.concatenate(
            (
                surface.faces[:, [0, 1]],
                surface.faces[:, [1, 2]],
                surface.faces[:, [2, 0]],
            )
        ),
        axis=1,
    )
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    return (
        mesh_components(surface.positions, surface.faces).count,
        len(surface.positions) - len(edges) + len(surface.faces),
        int((counts > 2).sum()),
    )


def boundary_coordinates(surface: CurveSurface) -> set[tuple[float, ...]]:
    """Compare geometric edges independently of a mesh's vertex ordering."""
    result = set()
    for first, second in surface.positions[boundary_edges(surface.faces)]:
        low, high = sorted((tuple(map(float, first)), tuple(map(float, second))))
        result.add(low + high)
    return result


def component_signatures(surface: CurveSurface) -> list[tuple[int, int, int]]:
    """Check component-wise Euler characteristic and nonmanifold counts."""
    parts = mesh_components(surface.positions, surface.faces)
    signatures = []
    for component in range(parts.count):
        faces = surface.faces[parts.face_parts == component]
        used, inverse = np.unique(faces, return_inverse=True)
        signatures.append(
            topology(
                CurveSurface(surface.positions[used], inverse.reshape(-1, 3).astype(np.int32))
            )
        )
    return sorted(signatures)


def polyhedron(
    positions: NDArray[np.float32],
    faces: NDArray[np.int32],
    protected_faces: NDArray[np.bool_],
    cell: float = 0.3,
) -> Polyhedron:
    """Decimate movable patches with fixed boundaries; retain invalid candidate components."""
    if not math.isfinite(cell) or cell <= 0:
        raise PipelineInputError('Polyhedron cell width must be finite and positive')
    parts = mesh_components(positions, faces)
    face_order = np.argsort(parts.face_parts, kind='stable')
    offsets = np.r_[0, np.cumsum(np.bincount(parts.face_parts, minlength=parts.count))]
    outputs = []
    reduced, unchanged = 0, 0
    for component in range(parts.count):
        selected = face_order[offsets[component] : offsets[component + 1]]
        used, inverse = np.unique(faces[selected], return_inverse=True)
        reference = CurveSurface(positions[used], inverse.reshape(-1, 3).astype(np.int32))
        boundaries = boundary_edges(reference.faces)
        candidate = decimate(reference, protected_faces[selected], cell)
        accepted = len(candidate.faces) > 0 and topology(candidate) == topology(reference)
        if accepted:
            accepted = boundary_coordinates(candidate) == boundary_coordinates(reference)
        if accepted and not len(boundaries):
            original_volume = signed_volume(reference.positions, reference.faces)
            candidate_volume = signed_volume(candidate.positions, candidate.faces)
            accepted = original_volume * candidate_volume > 0
        if accepted:
            patch = positions[faces[selected][protected_faces[selected]]].reshape(-1, 9)
            triangles = np.ascontiguousarray(candidate.positions[candidate.faces].reshape(-1, 9))
            keys = triangles.view(np.dtype((np.void, 36))).ravel()
            required = np.ascontiguousarray(patch).view(np.dtype((np.void, 36))).ravel()
            accepted = np.isin(required, keys).all()
        surface = candidate if accepted else reference
        if len(surface.faces) < len(reference.faces):
            reduced += 1
        else:
            unchanged += 1
        outputs.append(surface)
    vertices, triangles, offset = [], [], 0
    for surface in outputs:
        vertices.append(surface.positions)
        triangles.append(surface.faces + offset)
        offset += len(surface.positions)
    return Polyhedron(
        CurveSurface(np.concatenate(vertices), np.concatenate(triangles)),
        reduced,
        unchanged,
    )
