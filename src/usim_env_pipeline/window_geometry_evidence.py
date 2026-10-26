"""Source-only observation evidence for already fitted finite front strips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from .edge_line_support import LineSupport, SupportedLine, supported_intervals
from .window_geometry_fit import observed_surface_rows
from .window_member_sections import section_profile

SUPPORT: Final = LineSupport(
    width=0.10,
    step=0.10,
    max_gap=0.25,
    min_length=0.5,
    min_points=12,
    min_bin_points=1,
    min_coverage=0.7,
)


@dataclass(frozen=True, slots=True)
class PatchObservationEvidence:
    rows: NDArray[np.int64]
    gains: NDArray[np.float64]
    component_mask: NDArray[np.bool_]
    parts: tuple[SupportedLine, ...]
    point_p95_m: tuple[float, ...]
    rejected_p95_m: tuple[float, ...]


def cell_observation_weights(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Give every fixed world-space 5 cm source cell a total budget of one.

    All original observations determine the denominator, not only those eligible
    for a particular model. This is a capped composite score, not independent
    likelihood evidence or arbitrary thinning invariance.
    """
    _, cells, counts = np.unique(
        np.floor(xyz / 0.05).astype(np.int64),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    return np.asarray(1.0 / counts[cells], dtype=np.float64)


def section_station_indices(positions: NDArray[np.float64], length: float) -> NDArray[np.int64]:
    """Map in-span observations, including the terminal point, to 15 cm stations."""
    bins = np.floor(positions / 0.15).astype(np.int64)
    inside = (positions >= 0) & (positions <= length)
    mapped = np.full(len(positions), -1, dtype=np.int64)
    mapped[inside] = np.minimum(bins[inside], int(np.ceil(length / 0.15)) - 1)
    return mapped


def legacy_section_acceptance_mask(
    xyz: NDArray[np.float64],
    endpoints: NDArray[np.float64],
    normal: NDArray[np.float64],
    rows: NDArray[np.int64],
) -> NDArray[np.bool_]:
    """Reproduce the rejected hard-profile experiment, not physical contradiction.

    False combines missing profile information, centre/extent conditions and
    sampled-population PCA disagreement. It is not a source-point member label.
    The primary residual selector does not use this experimental veto.
    No additional independent observation reward is introduced.
    """
    _, accepted, _ = section_profile(xyz, endpoints, normal)
    direction = endpoints[1] - endpoints[0]
    length = float(np.linalg.norm(direction))
    direction /= length
    bins = section_station_indices(
        np.asarray((xyz[rows] - endpoints[0]) @ direction, dtype=np.float64), length
    )
    inside = (bins >= 0) & (bins < len(accepted))
    mask = np.zeros(len(rows), dtype=np.bool_)
    mask[inside] = accepted[bins[inside]]
    return mask


def patch_observation_evidence(
    xyz: NDArray[np.float64],
    endpoints: NDArray[np.float64],
    normal: NDArray[np.float64],
    half_width_m: float,
    component_rows: NDArray[np.int64],
    tree: KDTree,
) -> PatchObservationEvidence:
    """Validate intervals first, then score the same finite surface against source.

    Components propose a shape; they do not reserve their original observations.
    Common rows include observed junctions, but neither another depth sheet nor
    unexportable longitudinal spans. Geometry and source coordinates stay fixed.
    """
    direction = endpoints[1] - endpoints[0]
    direction /= np.linalg.norm(direction)
    rows = observed_surface_rows(xyz, endpoints, normal, half_width_m)
    parts = supported_intervals(xyz, (rows,), endpoints[0], direction, SUPPORT)
    valid: list[SupportedLine] = []
    accepted_p95: list[float] = []
    rejected_p95: list[float] = []
    inside = np.zeros(len(rows), dtype=np.bool_)
    positions = (xyz[rows] - endpoints[0]) @ direction
    for part in parts:
        span = part.endpoints[1] - part.endpoints[0]
        samples = (
            part.endpoints[0]
            + np.linspace(0, 1, int(np.ceil(np.linalg.norm(span) / 0.025)) + 1)[:, None] * span
        )
        p95 = float(np.quantile(tree.query(samples)[0], 0.95))
        if p95 > 0.10:
            rejected_p95.append(p95)
            continue
        valid.append(part)
        accepted_p95.append(p95)
        limits = (part.endpoints - endpoints[0]) @ direction
        inside |= (positions >= float(limits.min())) & (positions <= float(limits.max()))
    eligible = rows[inside]
    radial = np.asarray(np.cross(normal, direction), dtype=np.float64)
    radial /= np.linalg.norm(radial)
    delta = xyz[eligible] - endpoints[0]
    excess = np.maximum(np.abs(delta @ radial) - half_width_m, 0.0)
    gains = np.asarray(
        np.clip(1.0 - ((delta @ normal) / 0.06) ** 2 - (excess / 0.025) ** 2, 0.0, 1.0),
        dtype=np.float64,
    )
    return PatchObservationEvidence(
        eligible,
        gains,
        np.isin(eligible, component_rows),
        tuple(valid),
        tuple(accepted_p95),
        tuple(rejected_p95),
    )
