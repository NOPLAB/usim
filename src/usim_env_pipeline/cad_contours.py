"""Trace observed planar occupancy boundaries as oriented CAD polygon wires."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .cad_types import CadArchive
from .models import PipelineInputError


def signed_area(loop: NDArray[np.float64]) -> float:
    return float(
        np.sum(loop[:, 0] * np.roll(loop[:, 1], -1) - loop[:, 1] * np.roll(loop[:, 0], -1)) / 2
    )


def boundary_loops(mask: NDArray[np.bool_]) -> list[NDArray[np.float64]]:
    """Keep occupied cells to the left; diagonal contacts remain separate wires."""
    padded = np.pad(mask, 1)
    neighbors = [
        padded[1:-1, :-2],
        padded[2:, 1:-1],
        padded[1:-1, 2:],
        padded[:-2, 1:-1],
    ]
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for side, neighbor in enumerate(neighbors):
        cells = np.argwhere(mask & ~neighbor)
        starts = cells + corners[side]
        ends = cells + corners[(side + 1) % 4]
        for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
            adjacency.setdefault(tuple(start), []).append(tuple(end))
    loops = []
    while adjacency:
        start = next(iter(adjacency))
        current, previous = start, None
        vertices = []
        while True:
            vertices.append(current)
            options = adjacency[current]
            if previous is None or len(options) == 1:
                following = options[0]
            else:
                incoming = np.subtract(current, previous)
                # Maximum left turn separates cells that only touch at one vertex.
                following = max(
                    options,
                    key=lambda point: (
                        incoming[0] * (point[1] - current[1])
                        - incoming[1] * (point[0] - current[0])
                    ),
                )
            options.remove(following)
            if not options:
                del adjacency[current]
            previous, current = current, following
            if current == start:
                break
        pending = [vertices]
        while pending:
            ring = pending.pop()
            seen: dict[tuple[int, int], int] = {}
            split = False
            for index, point in enumerate(ring):
                if point in seen:
                    first = seen[point]
                    pending.extend((ring[first:index], ring[:first] + ring[index:]))
                    split = True
                    break
                seen[point] = index
            if split:
                continue
            loop = np.asarray(ring, dtype=np.float64)
            directions = np.roll(loop, -1, axis=0) - loop
            previous_directions = np.roll(directions, 1, axis=0)
            turns = (
                directions[:, 0] * previous_directions[:, 1]
                - directions[:, 1] * previous_directions[:, 0]
            )
            loop = loop[turns != 0]
            if len(loop) < 3 or signed_area(loop) == 0:
                raise PipelineInputError('Degenerate CAD occupancy boundary')
            loops.append(loop)
    return loops


def simplify_loop(loop: NDArray[np.float64], tolerance: float) -> NDArray[np.float64]:
    """Bound the contour change; preserve the winding needed by the CAD face maker."""
    if tolerance <= 0:
        return loop
    anchor = int(np.argmax(np.linalg.norm(loop - loop[0], axis=1)))
    kept = {0, anchor}
    extended = np.vstack((loop, loop[0]))
    pending = [(0, anchor), (anchor, len(loop))]
    while pending:
        first, last = pending.pop()
        if last - first < 2:
            continue
        segment = extended[last] - extended[first]
        points = extended[first + 1 : last] - extended[first]
        factors = np.clip(points @ segment / float(segment @ segment), 0, 1)
        distances = np.linalg.norm(points - factors[:, None] * segment, axis=1)
        furthest = int(np.argmax(distances))
        if distances[furthest] > tolerance:
            middle = first + 1 + furthest
            kept.add(middle)
            pending.extend(((first, middle), (middle, last)))
    result = loop[sorted(kept)]
    # Tiny loops cannot become a line; CAD hole removal decides their fate.
    return result if len(result) >= 3 else loop


def world_loops(archive: CadArchive, cell: float, tolerance: float) -> list[NDArray[np.float64]]:
    """Reconstruct polygon coordinates from the source-supported plane, not triangles."""
    normal, offset = archive['equation'][:3], archive['equation'][3]
    basis = np.vstack((archive['u'], archive['v']))
    result = []
    for loop in boundary_loops(archive['mask']):
        uv = (loop + archive['lower']) * cell
        if signed_area(loop) > 0:
            uv = simplify_loop(uv, tolerance)
        result.append(uv @ basis - normal * offset)
    return result
