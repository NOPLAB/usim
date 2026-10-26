"""Fit supported surfaces before retaining irregular geometry as bounded smooth volume."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .simulation_planes import PlanarProxy, reconstruct_proxies
from .simulation_primitive_detection import detect_primitives
from .storage import write_json


def reconstruct_shapes(
    points: NDArray[np.float32],
    available: NDArray[np.bool_],
    ground_ids: NDArray[np.int64],
    output: Path,
    cell: float,
) -> list[PlanarProxy]:
    """Prefer bounded fitted faces while preserving disjoint source ownership."""
    proxies = reconstruct_proxies(
        points,
        available,
        ground_ids,
        output / 'optimized/planes',
        cell=cell,
        completion=True,
        minimum_clearance=0.25,
    )
    remaining = available.copy()
    for proxy in proxies:
        remaining[proxy.support_ids] = False
    folder = output / 'simulation-models/primitive-planes'
    count = detect_primitives(points, remaining, folder)
    proxies.extend(
        reconstruct_proxies(
            points,
            remaining,
            ground_ids,
            folder,
            cell=cell,
            completion=True,
            minimum_clearance=0.25,
            source_paths=[folder / f'plane_{number:04}.npz' for number in range(count)],
        )
    )
    write_json(
        output / 'simulation-models/approximation.json',
        {
            'completion_radius_m': 0.2,
            'maximum_enclosed_hole_m2': 0.16,
            'minimum_planar_area_m2': 0.5,
            'minimum_horizontal_clearance_m': 0.25,
            'detected_planes': sum(proxy.source_path.parent == folder for proxy in proxies),
            'inferred_planar_cells': sum(
                len(proxy.keys) - len(proxy.observed_keys) for proxy in proxies
            ),
        },
    )
    return proxies
