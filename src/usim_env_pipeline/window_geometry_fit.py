"""Finite observed strip fitting; derived surfaces never replace source points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from typing_extensions import assert_never

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .window_member_sections import member_section_centres

CentreRule: TypeAlias = Literal['observed_median', 'transverse_midpoint']


@dataclass(frozen=True, slots=True)
class MemberPatch:
    endpoints: NDArray[np.float64]
    original_rows: NDArray[np.int64]
    centres: NDArray[np.float64]
    normal: NDArray[np.float64]
    half_width_m: float
    centre_rms_m: float
    plane_rms_m: float
    direction_change_deg: float

    @property
    def corners(self) -> NDArray[np.float64]:
        """Return a measured front strip, not an inferred closed solid."""
        direction = self.endpoints[1] - self.endpoints[0]
        direction /= np.linalg.norm(direction)
        radial = np.asarray(np.cross(self.normal, direction), dtype=np.float64)
        radial /= np.linalg.norm(radial)
        offsets = self.half_width_m * np.array([-1, 1, 1, -1])[:, None] * radial
        return self.endpoints[[0, 0, 1, 1]] + offsets


def observed_surface_rows(
    xyz: NDArray[np.float64],
    endpoints: NDArray[np.float64],
    normal: NDArray[np.float64],
    half_width_m: float,
) -> NDArray[np.int64]:
    """Measure an already fitted finite surface, including observed junction points.

    Section-component omissions are not missing observations. Re-measure the
    selected surface against original points without filling a gap or a backside.
    """
    direction = endpoints[1] - endpoints[0]
    length = float(np.linalg.norm(direction))
    direction /= length
    radial = np.asarray(np.cross(normal, direction), dtype=np.float64)
    radial /= np.linalg.norm(radial)
    delta = xyz - endpoints[0]
    t = delta @ direction
    return np.flatnonzero(
        (t >= 0)
        & (t <= length)
        & (np.abs(delta @ radial) <= half_width_m + 0.025)
        & (np.abs(delta @ normal) <= 0.06)
    ).astype(np.int64)


def fit_member_patch(
    xyz: NDArray[np.float64],
    endpoints: NDArray[np.float64],
    normal: NDArray[np.float64],
    *,
    centre_rule: CentreRule = 'observed_median',
) -> MemberPatch | None:
    """Fit complete bounded sections with equal station weights and robust residuals.

    The seed determines the local search, not a Manhattan orientation. Independent
    depth components are retained by the existing original-point section routine.
    Its component union is recorded as a hypothesis, not verified member identity.
    Three distinct stations constrain the affine fit; finite length and original
    point support still decide whether the hypothesis can be exported.
    The centre rule is an explicit geometric hypothesis, not a region-specific
    fallback. Partial or occluded sections need not identify a physical midpoint.
    """
    origin = endpoints[0]
    direction = endpoints[1] - origin
    length = float(np.linalg.norm(direction))
    direction /= length
    radial = np.asarray(np.cross(normal, direction), dtype=np.float64)
    radial /= np.linalg.norm(radial)
    sections = member_section_centres(xyz, origin, direction, normal, 0.0, length)
    match centre_rule:
        case 'observed_median':
            centres = np.unique(sections.centres, axis=0)
        case 'transverse_midpoint':
            centres = np.unique(sections.extent_centres, axis=0)
        case _:
            assert_never(centre_rule)
    if len(centres) < 3:
        return None
    depth_axis = normal - direction * float(normal @ direction)
    depth_axis /= np.linalg.norm(depth_axis)
    frame = np.column_stack((radial, depth_axis))
    t = (centres - origin) @ direction
    observed = (centres - origin) @ frame

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        return (parameters[:2] + t[:, None] * parameters[2:] - observed).ravel()

    result = least_squares(residual, np.zeros(4), loss='soft_l1', f_scale=0.03, max_nfev=100)
    if not result.success:
        return None
    parameters = np.asarray(result.x, dtype=np.float64)
    fitted_direction = direction + frame @ parameters[2:]
    fitted_direction /= np.linalg.norm(fitted_direction)
    angle = float(np.rad2deg(np.arccos(np.clip(float(fitted_direction @ direction), -1.0, 1.0))))
    if angle > 15.0:
        return None
    low = max(0.0, float(t.min()) - 0.05)
    high = min(length, float(t.max()) + 0.05)
    if high - low < 0.5:
        return None
    fitted_origin = origin + frame @ parameters[:2]
    vector = direction + frame @ parameters[2:]
    fitted_ends = fitted_origin + np.array([low, high])[:, None] * vector
    points = xyz[sections.original_rows]
    projected = (points - fitted_ends[0]) @ fitted_direction
    span = float(np.linalg.norm(fitted_ends[1] - fitted_ends[0]))
    rows = sections.original_rows[(projected >= 0) & (projected <= span)]
    if len(rows) < 12:
        return None
    points = xyz[rows]
    axes = np.asarray(
        np.linalg.svd(points - points.mean(axis=0), full_matrices=False)[2], dtype=np.float64
    )
    fitted_normal = axes[-1] - fitted_direction * float(axes[-1] @ fitted_direction)
    fitted_normal /= np.linalg.norm(fitted_normal)
    if float(fitted_normal @ normal) < 0:
        fitted_normal = -fitted_normal
    fitted_radial = np.cross(fitted_normal, fitted_direction)
    fitted_radial /= np.linalg.norm(fitted_radial)
    transverse = (points - fitted_origin) @ fitted_radial
    width = float(np.diff(np.quantile(transverse, [0.05, 0.95]))[0])
    centre_residual = np.linalg.norm(np.cross(centres - fitted_origin, fitted_direction), axis=1)
    plane_residual = (points - fitted_ends[0]) @ fitted_normal
    return MemberPatch(
        fitted_ends,
        np.asarray(rows, dtype=np.int64),
        centres,
        fitted_normal,
        width / 2,
        float(np.sqrt(np.mean(centre_residual**2))),
        float(np.sqrt(np.mean(plane_residual**2))),
        angle,
    )
