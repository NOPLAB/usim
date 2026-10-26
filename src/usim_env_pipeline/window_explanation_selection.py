"""Sparse facility selection with one best explanation per source observation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

from .window_geometry_selection import PatchSelection


@dataclass(frozen=True, slots=True)
class ExplanationEvidence:
    """Trusted sparse observation/model edges and candidate-independent weights.

    Edge arrays have equal length, gains are in (0, 1], and weights are positive.
    Observation indices address weights; model indices address supplied costs.
    Repeated edges are allowed but cannot increase an observation's budget.
    """

    observations: NDArray[np.int64]
    models: NDArray[np.int64]
    gains: NDArray[np.float64]
    weights: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ExplanationSelection:
    selection: PatchSelection
    owners: NDArray[np.int64]
    gains: NDArray[np.float64]


def select_explanations(
    evidence: ExplanationEvidence,
    costs: NDArray[np.float64],
    *,
    conflicts: NDArray[np.int64] | None = None,
    seconds: float = 120.0,
) -> ExplanationSelection:
    """Choose fixed models by residual benefit, without adding overlapping gains.

    Binary model variables open explanations. Continuous edge assignments have
    an observation simplex budget and may only use open models. For fixed model
    choices, the best explanation is an extreme-point assignment; ties are
    reconstructed by lowest model index. This is an accounting decision, not
    physical ownership of shared junction witnesses.
    """
    observations, models, gains = evidence.observations, evidence.models, evidence.gains
    count, edges, points = len(costs), len(gains), len(evidence.weights)
    owners = np.full(points, -1, dtype=np.int64)
    winning = np.zeros(points, dtype=np.float64)
    if not count or not edges:
        empty = PatchSelection(np.empty(0, dtype=np.int64), 0.0, 0.0, 0.0, True, True, 0, 'empty')
        return ExplanationSelection(empty, owners, winning)
    pair_count = 0 if conflicts is None else len(conflicts)
    edge_indices = np.arange(edges, dtype=np.int64)
    rows = [observations, points + edge_indices, points + edge_indices]
    columns = [count + edge_indices, count + edge_indices, models]
    values = [np.ones(edges), np.ones(edges), -np.ones(edges)]
    if conflicts is not None and pair_count:
        rows.append(points + edges + np.repeat(np.arange(pair_count), 2))
        columns.append(conflicts.ravel())
        values.append(np.ones(2 * pair_count))
    constraint = csc_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(points + edges + pair_count, count + edges),
    )
    result = milp(
        c=np.r_[costs, -evidence.weights[observations] * gains],
        integrality=np.r_[np.ones(count), np.zeros(edges)],
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(
            constraint, -np.inf, np.r_[np.ones(points), np.zeros(edges), np.ones(pair_count)]
        ),
        options={'time_limit': seconds, 'mip_rel_gap': 0.0},
    )
    feasible = result.x is not None
    selected = (
        np.flatnonzero(np.asarray(result.x[:count]) > 0.5).astype(np.int64)
        if feasible
        else np.empty(0, dtype=np.int64)
    )
    if len(selected):
        active = np.flatnonzero(np.isin(models, selected))
        order = np.lexsort((models[active], -gains[active], observations[active]))
        ranked = active[order]
        first = ranked[np.r_[True, np.diff(observations[ranked]) != 0]]
        owners[observations[first]] = models[first]
        winning[observations[first]] = gains[first]
    selection = PatchSelection(
        selected,
        float(result.fun) if feasible else None,
        float(result.mip_dual_bound) if feasible else None,
        float(result.mip_gap) if feasible else None,
        result.status == 0,
        feasible,
        int(result.status),
        str(result.message),
    )
    return ExplanationSelection(selection, owners, winning)
