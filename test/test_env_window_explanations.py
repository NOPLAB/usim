"""Residual explanation credit belongs to original observations, not aliases."""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from usim_env_pipeline.window_explanation_selection import (
    ExplanationEvidence,
    select_explanations,
)
from usim_env_pipeline.window_geometry_selection import select_patches


def evidence(gains: list[list[float]], weights: list[float] | None = None) -> ExplanationEvidence:
    dense = np.asarray(gains, dtype=np.float64)
    observations, models = np.nonzero(dense)
    return ExplanationEvidence(
        observations.astype(np.int64),
        models.astype(np.int64),
        dense[observations, models],
        np.ones(len(dense)) if weights is None else np.asarray(weights, dtype=np.float64),
    )


def test_binary_gains_match_legacy_optimum() -> None:
    links = evidence([[1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1]])
    costs = np.array([0.2, 0.3, 0.4])
    legacy = select_patches(
        np.array([[1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1]], dtype=np.bool_), costs
    )

    result = select_explanations(links, costs)

    assert result.selection.optimal
    assert result.selection.objective == pytest.approx(legacy.objective)


def test_overlapping_weak_aliases_cannot_add_to_full_credit() -> None:
    links = evidence([[0.6, 0.6]])

    result = select_explanations(links, np.array([0.1, 0.1]))

    assert len(result.selection.indices) == 1
    np.testing.assert_allclose(result.gains, [0.6])
    assert result.selection.objective == pytest.approx(-0.5)


def test_quality_beats_cheaper_weak_competitor() -> None:
    links = evidence([[0.95, 0.5], [0.95, 0.5]])

    result = select_explanations(links, np.array([0.4, 0.05]))

    np.testing.assert_array_equal(result.selection.indices, [0])
    assert result.selection.objective == pytest.approx(-1.5)


def test_alias_cannot_steal_extra_credit() -> None:
    links = evidence([[0.9, 0.6, 0], [0.9, 0, 0], [0, 0.6, 0.9], [0, 0, 0.9]])

    result = select_explanations(links, np.full(3, 0.1))

    np.testing.assert_array_equal(result.selection.indices, [0, 2])
    np.testing.assert_array_equal(result.owners, [0, 0, 2, 2])
    assert result.selection.objective == pytest.approx(-3.4)


def test_crossing_arms_survive_with_shared_observation_counted_once() -> None:
    links = evidence([[0.8, 0.8], [0.9, 0], [0, 0.9], [0, 0]])
    costs = np.array([0.2, 0.2])

    result = select_explanations(links, costs)

    np.testing.assert_array_equal(result.selection.indices, [0, 1])
    np.testing.assert_array_equal(result.owners, [0, 0, 1, -1])
    np.testing.assert_allclose(result.gains, [0.8, 0.9, 0.9, 0])
    assert result.selection.objective == pytest.approx(-2.2)
    assert costs[result.selection.indices].sum() - links.weights @ result.gains == pytest.approx(
        result.selection.objective
    )


@pytest.mark.parametrize('model_count', [0, 2])
def test_empty_edges_are_valid_without_solver(
    model_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    links = ExplanationEvidence(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
        np.ones(3),
    )

    def unexpected_solver_call(*args: float, **kwargs: float) -> None:
        pytest.fail('Empty evidence must not invoke the solver')

    monkeypatch.setattr(
        'usim_env_pipeline.window_explanation_selection.milp', unexpected_solver_call
    )

    result = select_explanations(links, np.full(model_count, 0.1))

    assert result.selection.optimal and result.selection.feasible
    assert result.selection.objective == 0
    assert result.selection.lower_bound == 0
    assert result.selection.gap == 0
    assert result.selection.indices.size == 0
    np.testing.assert_array_equal(result.owners, [-1, -1, -1])
    np.testing.assert_array_equal(result.gains, [0, 0, 0])


def test_input_arrays_are_not_mutated() -> None:
    links = evidence([[0.7, 0.9], [0.8, 0]], [2, 3])
    inputs = (links.observations, links.models, links.gains, links.weights)
    originals = tuple(array.copy() for array in inputs)

    select_explanations(links, np.array([0.1, 0.2]))

    for array, original in zip(inputs, originals, strict=True):
        np.testing.assert_array_equal(array, original)


def test_observation_edge_and_model_permutations_preserve_unique_selection() -> None:
    dense = np.array([[0.9, 0.5, 0], [0.8, 0, 0], [0, 0.4, 0.95]])
    costs = np.array([0.2, 0.15, 0.3])
    model_order = np.array([2, 0, 1])
    observation_order = np.array([2, 0, 1])
    links = evidence(dense[observation_order][:, model_order].tolist(), [4, 2, 3])
    links = ExplanationEvidence(
        links.observations[::-1], links.models[::-1], links.gains[::-1], links.weights
    )

    result = select_explanations(links, costs[model_order])

    np.testing.assert_array_equal(np.sort(model_order[result.selection.indices]), [0, 2])
    np.testing.assert_array_equal(model_order[result.owners], [2, 0, 0])
    assert result.selection.objective == pytest.approx(-7.5)


def test_replicated_observations_with_split_weights_preserve_objective() -> None:
    links = evidence([[0.9, 0.5], [0.8, 0.4]], [3, 2])
    replicated = ExplanationEvidence(
        np.r_[links.observations, links.observations + 2],
        np.tile(links.models, 2),
        np.tile(links.gains, 2),
        np.tile(links.weights / 2, 2),
    )
    original = select_explanations(links, np.array([0.2, 0.1]))

    result = select_explanations(replicated, np.array([0.2, 0.1]))

    np.testing.assert_array_equal(result.selection.indices, original.selection.indices)
    assert result.selection.objective == pytest.approx(original.selection.objective)


def test_source_conflict_preserves_independent_model() -> None:
    links = evidence([[0.9, 0.8, 0], [0.9, 0, 0], [0, 0.8, 0], [0, 0, 0.95]])

    result = select_explanations(
        links, np.full(3, 0.2), conflicts=np.array([[0, 1]], dtype=np.int64)
    )

    np.testing.assert_array_equal(result.selection.indices, [0, 2])
    np.testing.assert_array_equal(result.owners, [0, 0, -1, 2])


def test_best_gain_then_lowest_model_index_owns_shared_observation() -> None:
    links = evidence([[0.6, 0.8], [0.7, 0.7], [1, 0], [0, 1]])

    result = select_explanations(links, np.array([0.1, 0.1]))

    np.testing.assert_array_equal(result.owners, [1, 0, 0, 1])
    np.testing.assert_allclose(result.gains, [0.8, 0.7, 1, 1])


def test_duplicate_edges_do_not_create_additional_credit() -> None:
    links = ExplanationEvidence(
        np.array([0, 0, 0], dtype=np.int64),
        np.array([0, 0, 1], dtype=np.int64),
        np.array([0.6, 0.6, 0.9]),
        np.ones(1),
    )

    result = select_explanations(links, np.array([0.1, 0.2]))

    np.testing.assert_array_equal(result.selection.indices, [1])
    assert result.selection.objective == pytest.approx(-0.7)


def test_no_incumbent_retains_unsolved_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    links = evidence([[0.9]])

    def no_incumbent(*args: float, **kwargs: float) -> OptimizeResult:
        return OptimizeResult(x=None, status=1, message='limit', fun=None)

    monkeypatch.setattr('usim_env_pipeline.window_explanation_selection.milp', no_incumbent)

    result = select_explanations(links, np.array([0.1]))

    assert not result.selection.feasible
    assert not result.selection.optimal
    assert result.selection.objective is None
    assert result.selection.lower_bound is None
    assert result.selection.gap is None
    assert result.selection.status == 1
    np.testing.assert_array_equal(result.owners, [-1])
