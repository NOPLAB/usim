"""Finite patch selection with one explanation budget per observed source cell."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, eye, hstack


@dataclass(frozen=True, slots=True)
class PatchSelection:
    indices: NDArray[np.int64]
    objective: float | None
    lower_bound: float | None
    gap: float | None
    optimal: bool
    feasible: bool
    status: int
    message: str


def select_patches(
    incidence: NDArray[np.bool_],
    costs: NDArray[np.float64],
    *,
    conflicts: NDArray[np.int64] | None = None,
    seconds: float = 120.0,
) -> PatchSelection:
    """Minimize truncated occupied-cell loss plus finite model description costs.

    Each cell's reward is bounded by one even when many models explain it. The
    binary variables choose patches; continuous coverage variables need no
    integrality because their optimum is already zero or one for fixed patches.
    An absent solver incumbent is not silently treated as a valid empty model.
    """
    cells, models = incidence.shape
    if not models:
        return PatchSelection(np.empty(0, dtype=np.int64), 0.0, 0.0, 0.0, True, True, 0, 'empty')
    constraint = hstack((-csc_matrix(incidence, dtype=np.float64), eye(cells)), format='csc')
    constraints = [LinearConstraint(constraint, -np.inf, 0.0)]
    if conflicts is not None and len(conflicts):
        pair_rows = np.repeat(np.arange(len(conflicts)), 2)
        exclusion = csc_matrix(
            (np.ones(2 * len(conflicts)), (pair_rows, conflicts.ravel())),
            shape=(len(conflicts), models + cells),
        )
        constraints.append(LinearConstraint(exclusion, -np.inf, 1.0))
    result = milp(
        c=np.r_[costs, -np.ones(cells)],
        integrality=np.r_[np.ones(models), np.zeros(cells)],
        bounds=Bounds(0.0, 1.0),
        constraints=constraints,
        options={'time_limit': seconds, 'mip_rel_gap': 0.0},
    )
    feasible = result.x is not None
    indices = (
        np.flatnonzero(np.asarray(result.x[:models]) > 0.5).astype(np.int64)
        if feasible
        else np.empty(0, dtype=np.int64)
    )
    return PatchSelection(
        indices,
        float(result.fun) if feasible else None,
        float(result.mip_dual_bound) if feasible else None,
        float(result.mip_gap) if feasible else None,
        result.status == 0,
        feasible,
        int(result.status),
        str(result.message),
    )
