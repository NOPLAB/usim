"""Ownership fragments reconcile only through real finite original face support."""

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
import pytest

from usim_env_pipeline.edge_line_geometry import EdgeLineConfig, MeasuredEdge
from usim_env_pipeline.edge_line_support import LineSupport, SupportedLine, supported_intervals
from usim_env_pipeline.edge_reconciliation import reconcile_edge_lines


def corner(
    length: float = 1.8, *, offset: float = 0.0
) -> tuple[NDArray[np.float64], MeasuredEdge, EdgeLineConfig]:
    t, width = np.meshgrid(np.arange(round(length * 40) + 1) / 40, [0.04, 0.09, 0.14])
    floor = np.column_stack((t.ravel(), width.ravel() + offset, np.zeros(t.size)))
    wall = np.column_stack((t.ravel(), np.full(t.size, offset), width.ravel()))
    xyz = np.vstack((floor, wall))
    groups = (np.arange(t.size, dtype=np.int64), np.arange(t.size, 2 * t.size, dtype=np.int64))
    equations = np.array([[0.0, 0, 1, 0], [0, 1, 0, -offset]])
    config = EdgeLineConfig(support=LineSupport(min_length=min(length - 0.01, 1.5)))
    line = supported_intervals(
        xyz, groups, np.array([0.0, offset, 0]), np.array([1.0, 0, 0]), config.support
    )[0]
    return xyz, MeasuredEdge(line, 2, equations), config


def clip(edge: MeasuredEdge, low: float, high: float) -> MeasuredEdge:
    endpoints = edge.line.endpoints.copy()
    endpoints[:, 0] = [low, high]
    return replace(edge, line=replace(edge.line, endpoints=endpoints))


def assert_supported(
    xyz: NDArray[np.float64], edges: list[MeasuredEdge], config: EdgeLineConfig
) -> None:
    for edge in edges:
        a, b = edge.line.endpoints
        direction = (b - a) / np.linalg.norm(b - a)
        assert np.linalg.norm(b - a) >= config.support.min_length - 1e-10
        for face, ids in enumerate(edge.line.witnesses):
            assert ids.dtype == np.int64
            assert len(ids) == len(np.unique(ids))
            t = (xyz[ids] - a) @ direction
            assert t.min() >= -1e-10
            assert t.max() <= np.linalg.norm(b - a) + 1e-10
            distance = np.linalg.norm(xyz[ids] - a - t[:, None] * direction, axis=1)
            assert distance.max() <= config.support.width + 1e-10
            residual = np.abs(xyz[ids] @ edge.equations[face, :3] + edge.equations[face, 3])
            assert residual.max() <= config.plane_tolerance
        fresh = supported_intervals(xyz, edge.line.witnesses, a, direction, config.support)
        assert len(fresh) == 1
        np.testing.assert_allclose(fresh[0].endpoints, edge.line.endpoints, atol=1e-10)
        assert fresh[0].coverage == pytest.approx(edge.line.coverage)
        assert fresh[0].max_unsupported_gap == pytest.approx(edge.line.max_unsupported_gap)


@pytest.mark.parametrize('length', [0.8, 1.8])
def test_short_supported_ownership_halves_are_restored(length: float):
    xyz, edge, config = corner(length)
    halves = [clip(edge, 0, length / 2), clip(edge, length / 2, length)]
    assert all(
        np.linalg.norm(e.line.endpoints[1] - e.line.endpoints[0]) < config.support.min_length
        for e in halves
    )
    result = reconcile_edge_lines(xyz, halves, config)
    assert len(result) == 1
    np.testing.assert_allclose(result[0].line.endpoints, edge.line.endpoints)
    assert_supported(xyz, result, config)


def test_overlapping_duplicates_merge_with_reversed_plane_order_and_direction():
    xyz, edge, config = corner(2.0)
    other = clip(edge, 0.75, 2)
    other = replace(
        other,
        equations=-other.equations[::-1],
        line=replace(
            other.line,
            endpoints=other.line.endpoints[::-1],
            witnesses=other.line.witnesses[::-1],
        ),
    )
    result = reconcile_edge_lines(xyz, [other, clip(edge, 0, 1.25), other], config)
    assert len(result) == 1
    np.testing.assert_allclose(result[0].line.endpoints[:, 0], [0, 2])
    assert_supported(xyz, result, config)


@pytest.mark.parametrize('gap', [0.15, 0.4])
def test_disjoint_owned_intervals_never_bridge_even_with_halo_support(gap: float):
    xyz, edge, config = corner(3.0)
    config = replace(config, support=replace(config.support, min_length=0.75))
    result = reconcile_edge_lines(xyz, [clip(edge, 0, 1), clip(edge, 1 + gap, 3)], config)
    assert len(result) == 2
    np.testing.assert_allclose(result[0].line.endpoints[:, 0], [0, 1])
    np.testing.assert_allclose(result[1].line.endpoints[:, 0], [1 + gap, 3])
    assert_supported(xyz, result, config)


def test_adjacent_distinct_parallel_corners_do_not_merge_within_plane_tolerance():
    xyz, edge, config = corner()
    nearby, other, _ = corner(offset=0.02)
    other = replace(
        other,
        line=replace(
            other.line,
            witnesses=(edge.line.witnesses[0], other.line.witnesses[1] + len(xyz)),
        ),
    )
    xyz = np.vstack((xyz, nearby))
    result = reconcile_edge_lines(xyz, [other, edge], config)
    assert len(result) == 2
    np.testing.assert_allclose(sorted(e.line.endpoints[0, 1] for e in result), [0, 0.02])
    assert_supported(xyz, result, config)


def test_noisy_shared_witness_predictions_deduplicate_without_averaging_coordinates():
    xyz, edge, config = corner()
    equations = edge.equations.copy()
    equations[1, 3] = -0.006
    endpoints = edge.line.endpoints + [0, 0.006, 0]
    noisy = replace(edge, equations=equations, line=replace(edge.line, endpoints=endpoints))
    before = xyz.copy()
    result = reconcile_edge_lines(xyz, [noisy, edge], config)
    assert len(result) == 1
    assert any(np.array_equal(result[0].equations, e.equations) for e in [edge, noisy])
    np.testing.assert_array_equal(xyz, before)
    assert_supported(xyz, result, config)


def test_union_refit_uses_original_faces_not_either_biased_prediction():
    xyz, edge, config = corner()
    predictions = []
    for offset, low, high in ((-0.006, 0.0, 1.2), (0.006, 0.6, 1.8)):
        equations = edge.equations.copy()
        equations[1, 3] = -offset
        clipped = clip(edge, low, high)
        predictions.append(
            replace(
                clipped,
                equations=equations,
                line=replace(clipped.line, endpoints=clipped.line.endpoints + [0, offset, 0]),
            )
        )
    result = reconcile_edge_lines(xyz, predictions, config)
    assert len(result) == 1
    np.testing.assert_allclose(result[0].line.endpoints, edge.line.endpoints, atol=1e-10)
    assert_supported(xyz, result, config)


def test_ownership_clips_do_not_replace_the_same_certified_parent_model():
    xyz, edge, config = corner()
    xyz[edge.line.witnesses[1], 1] += 0.02
    result = reconcile_edge_lines(xyz, [clip(edge, 0, 0.9), clip(edge, 0.9, 1.8)], config)
    assert len(result) == 1
    np.testing.assert_array_equal(result[0].equations, edge.equations)
    np.testing.assert_allclose(result[0].line.endpoints, edge.line.endpoints, atol=1e-10)
    assert_supported(xyz, result, config)


def test_owned_span_trims_halo_witnesses_without_mutating_any_source_array():
    xyz, edge, config = corner(3.0)
    clipped = clip(edge, 0.5, 2.5)
    arrays = [xyz, clipped.line.endpoints, clipped.equations, *clipped.line.witnesses]
    before = [value.copy() for value in arrays]
    for value in arrays:
        value.setflags(write=False)
    result = reconcile_edge_lines(xyz, [clipped], config)
    assert len(result) == 1
    np.testing.assert_allclose(result[0].line.endpoints[:, 0], [0.5, 2.5])
    assert_supported(xyz, result, config)
    for value, old in zip(arrays, before):
        np.testing.assert_array_equal(value, old)


def test_rotated_translated_fragments_and_input_order_produce_same_real_line():
    xyz, edge, config = corner()
    rotation = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0], [0.48, 0.64, 0.6]])
    shift = np.array([143.0, -89, 21])
    halves = [clip(edge, 0, 0.9), clip(edge, 0.9, 1.8)]
    moved = []
    for half in halves:
        equations = half.equations.copy()
        equations[:, :3] = equations[:, :3] @ rotation
        equations[:, 3] -= equations[:, :3] @ shift
        moved.append(
            replace(
                half,
                equations=equations,
                line=replace(half.line, endpoints=half.line.endpoints @ rotation + shift),
            )
        )
    actual = reconcile_edge_lines(xyz @ rotation + shift, moved, config)
    repeated = reconcile_edge_lines(xyz @ rotation + shift, moved[::-1], config)
    assert len(actual) == len(repeated) == 1
    expected = edge.line.endpoints @ rotation + shift
    assert np.allclose(actual[0].line.endpoints, expected) or np.allclose(
        actual[0].line.endpoints, expected[::-1]
    )
    np.testing.assert_array_equal(actual[0].line.endpoints, repeated[0].line.endpoints)
    assert_supported(xyz @ rotation + shift, actual, config)


def test_actual_residual_width_and_independent_face_strips_are_revalidated():
    xyz, edge, config = corner()
    for mutation in ['residual', 'width', 'shared']:
        changed = xyz.copy()
        if mutation == 'residual':
            changed[edge.line.witnesses[1], 1] += 0.08
        elif mutation == 'width':
            changed[edge.line.witnesses[1], 2] += 0.5
        else:
            changed[:, 1:] = 0
        assert reconcile_edge_lines(changed, [edge], config) == []


def test_actual_large_support_gap_splits_a_stale_continuous_candidate():
    xyz, edge, config = corner(3.0)
    groups = tuple(ids[(xyz[ids, 0] <= 1) | (xyz[ids, 0] >= 1.5)] for ids in edge.line.witnesses)
    stale = replace(edge, line=replace(edge.line, witnesses=groups))
    config = replace(config, support=replace(config.support, min_length=0.75))
    result = reconcile_edge_lines(xyz, [stale], config)
    assert len(result) == 2
    assert result[0].line.endpoints[:, 0].max() <= 1
    assert result[1].line.endpoints[:, 0].min() >= 1.5
    assert_supported(xyz, result, config)


def test_one_face_shorter_than_other_has_only_final_interval_witnesses():
    xyz, edge, config = corner(3.0)
    groups = (
        edge.line.witnesses[0],
        edge.line.witnesses[1][
            (xyz[edge.line.witnesses[1], 0] >= 0.5) & (xyz[edge.line.witnesses[1], 0] <= 2.5)
        ],
    )
    result = reconcile_edge_lines(
        xyz, [replace(edge, line=replace(edge.line, witnesses=groups))], config
    )
    assert len(result) == 1
    np.testing.assert_allclose(result[0].line.endpoints[:, 0], [0.5, 2.5])
    assert_supported(xyz, result, config)


def test_independent_face_sampling_does_not_erode_the_entire_supported_line():
    xyz, edge, config = corner()
    xyz[edge.line.witnesses[1], 0] += 0.007
    original = supported_intervals(
        xyz, edge.line.witnesses, np.zeros(3), np.array([1.0, 0, 0]), config.support
    )[0]
    result = reconcile_edge_lines(xyz, [replace(edge, line=original)], config)
    assert len(result) == 1
    np.testing.assert_allclose(result[0].line.endpoints[:, 0], [0.025, 1.782])
    fresh = supported_intervals(
        xyz, result[0].line.witnesses, np.zeros(3), np.array([1.0, 0, 0]), config.support
    )
    assert len(fresh) == 1
    np.testing.assert_allclose(fresh[0].endpoints, result[0].line.endpoints)
    assert fresh[0].coverage == pytest.approx(result[0].line.coverage)
    assert fresh[0].max_unsupported_gap == pytest.approx(result[0].line.max_unsupported_gap)


@pytest.mark.parametrize('gate', ['length', 'occupancy', 'coverage'])
def test_stale_scores_cannot_bypass_final_support_gates(gate: str):
    xyz, edge, config = corner(3.0)
    if gate == 'length':
        edge = clip(edge, 0, 0.5)
    elif gate == 'occupancy':
        config = replace(config, support=replace(config.support, min_bin_points=25))
    else:
        groups = tuple(
            ids[(np.floor((xyz[ids, 0] + 1e-9) / 0.1).astype(int) % 2) == 0]
            for ids in edge.line.witnesses
        )
        edge = replace(edge, line=replace(edge.line, witnesses=groups))
    assert reconcile_edge_lines(xyz, [edge], config) == []


def boundary(
    xyz: NDArray[np.float64], edge: MeasuredEdge, offset: float = 0.0
) -> tuple[NDArray[np.float64], MeasuredEdge]:
    t = np.arange(121) / 40
    points = np.column_stack((t, np.full(len(t), offset), np.zeros(len(t))))
    ids = np.arange(len(xyz), len(xyz) + len(points), dtype=np.int64)
    line = SupportedLine(np.array([[0.0, offset, 0], [3, offset, 0]]), (ids,), 1.0, 0.025)
    equations = np.stack((edge.equations[0], np.zeros(4)))
    return np.vstack((xyz, points)), MeasuredEdge(line, 1, equations)


def test_valid_crease_replaces_only_coincident_boundary_interval():
    xyz, edge, config = corner(3.0)
    xyz, border = boundary(xyz, edge)
    crease = clip(edge, 1, 2)
    config = replace(config, support=replace(config.support, min_length=0.75))
    result = reconcile_edge_lines(xyz, [border, crease, border], config)
    assert [e.kind for e in result].count(2) == 1
    ends = sorted(e.line.endpoints[:, 0].tolist() for e in result if e.kind == 1)
    np.testing.assert_allclose(ends, [[0, 1], [2, 3]])
    assert_supported(xyz, result, config)


def test_nearby_distinct_boundary_is_not_erased_by_a_crease():
    xyz, edge, config = corner(3.0)
    xyz, border = boundary(xyz, edge, offset=-0.02)
    result = reconcile_edge_lines(xyz, [edge, border], config)
    assert sorted(e.kind for e in result) == [1, 2]
    assert_supported(xyz, result, config)


def test_rejected_crease_cannot_erase_a_valid_boundary():
    xyz, edge, config = corner(3.0)
    xyz, border = boundary(xyz, edge)
    edge = replace(
        edge,
        line=replace(edge.line, witnesses=(edge.line.witnesses[0], np.array([], dtype=np.int64))),
    )
    result = reconcile_edge_lines(xyz, [edge, border], config)
    assert len(result) == 1 and result[0].kind == 1
    np.testing.assert_allclose(result[0].line.endpoints[:, 0], [0, 3])


def test_empty_input_has_no_edges():
    assert reconcile_edge_lines(np.empty((0, 3)), [], EdgeLineConfig()) == []
