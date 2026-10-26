"""Exact radius partitions and conservative island candidates against independent oracles."""

from pathlib import Path

import numpy as np
import pytest

from test.test_env_live_lof import save_records
from usim_env_pipeline.models import PipelineInputError
from usim_env_pipeline.pcd import sha256
from usim_env_pipeline.point_islands import (
    classify_islands,
    compute_island_graph,
    distance_components,
)


def dense_partitions(xyz, radii):
    squared = np.sum((xyz[:, None, :] - xyz[None, :, :]) ** 2, axis=2)
    columns = []
    for radius in radii:
        labels = np.full(len(xyz), -1, dtype=np.int32)
        label = 0
        for row in range(len(xyz)):
            if labels[row] >= 0:
                continue
            pending = [row]
            labels[row] = label
            while pending:
                current = pending.pop()
                for neighbor in np.flatnonzero(squared[current] <= radius * radius):
                    if labels[neighbor] < 0:
                        labels[neighbor] = label
                        pending.append(int(neighbor))
            label += 1
        columns.append(labels)
    return np.column_stack(columns)


@pytest.mark.parametrize('limit', [100, 200, 1_000_000])
def test_exact_radius_partitions_without_batch_or_grid_dependence(limit: int):
    xyz = np.random.default_rng(98).uniform(-0.5, 0.5, (97, 3))
    radii = (0.125, 0.25, 0.5)
    actual = distance_components(xyz, radii, workers=1, edge_batch_limit=limit)
    np.testing.assert_array_equal(actual, dense_partitions(xyz, radii))
    shifted = distance_components(xyz + [143.25, -10.5, 2.0], radii, workers=1)
    np.testing.assert_array_equal(shifted, actual)


def test_closed_boundary_and_real_gap_not_cell_adjacency():
    xyz = np.array([[0, 0, 0], [0.125, 0, 0], [0.251, 0, 0], [0.5, 0.5, 0.5]])
    radii = (0.125, 0.25, 0.5)
    actual = distance_components(xyz, radii, workers=1)
    np.testing.assert_array_equal(actual, dense_partitions(xyz, radii))
    assert actual[0, 0] == actual[1, 0] != actual[2, 0]


def test_duplicates_do_not_overflow_sparse_boolean_edges():
    xyz = np.repeat(np.array([[0, 0, 0], [0.0625, 0, 0]], dtype=np.float64), 300, axis=0)
    labels = distance_components(xyz, (0.08, 0.1, 0.2), workers=1)
    np.testing.assert_array_equal(labels, 0)


def test_identical_clique_needs_no_large_radius_neighborhood():
    xyz = np.full((100, 3), -5.0)
    labels = distance_components(xyz, workers=1, edge_batch_limit=1)
    np.testing.assert_array_equal(labels, 0)


def make_graph(tmp_path: Path):
    axis = np.linspace(-0.3, 0.3, 13)
    x, y = np.meshgrid(axis, axis)
    ground = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    blob = np.array(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
        dtype=np.float64,
    ) * 0.025 + [0, 0, 2]
    px, py = np.meshgrid(np.linspace(-0.03, 0.03, 4), np.linspace(-0.03, 0.03, 4))
    planar = np.column_stack((px.ravel(), py.ravel(), np.full(px.size, 3.0)))
    linear = np.column_stack((np.linspace(0, 0.1, 8), np.zeros(8), np.full(8, 4.0)))
    xyz = np.vstack((ground, blob, planar, linear))
    source = save_records(tmp_path / 'records.npy', xyz)
    ids = np.arange(len(xyz), dtype=np.int64)
    np.save(tmp_path / 'ids.npy', ids)
    graph = compute_island_graph(source, tmp_path / 'ids.npy', tmp_path / 'cache', workers=1)
    return graph, (len(ground), len(ground) + len(blob), len(ground) + len(blob) + len(planar))


def test_candidates_protect_planar_linear_and_not_largest_only(tmp_path: Path):
    graph, (blob, planar, linear) = make_graph(tmp_path)
    before = sha256(tmp_path / 'records.npy')
    selected = classify_islands(graph, anchor_source_ids=(0,), workers=1)
    np.testing.assert_array_equal(selected.classes[:blob], 0)
    np.testing.assert_array_equal(selected.classes[blob:planar], 1)
    np.testing.assert_array_equal(selected.classes[planar:], 2)
    assert not selected.classes.flags.writeable
    assert sha256(tmp_path / 'records.npy') == before
    # Connecting a real member as a new anchor changes support, not just its color.
    anchored = classify_islands(graph, anchor_source_ids=(0, blob), workers=1)
    np.testing.assert_array_equal(anchored.classes[blob:planar], 0)
    assert anchored.supported[graph.labels[blob, 1]]
    aggressive = classify_islands(
        graph, anchor_source_ids=(0,), protect_planar=False, protect_linear=False, workers=1
    )
    np.testing.assert_array_equal(aggressive.classes[blob:], 1)
    # A large disconnected object is review even when another component is larger.
    no_candidates = classify_islands(graph, anchor_source_ids=(blob,), max_diameter=0.2, workers=1)
    np.testing.assert_array_equal(no_candidates.classes[:blob], 2)
    assert linear > planar


def test_fine_gap_coarse_support_is_review_not_removal(tmp_path: Path):
    xyz = np.array([[0, 0, 0], [0.05, 0, 0], [0.2, 0, 0]], dtype=np.float64)
    source = save_records(tmp_path / 'records.npy', xyz)
    np.save(tmp_path / 'ids.npy', np.arange(3, dtype=np.int64))
    graph = compute_island_graph(source, tmp_path / 'ids.npy', tmp_path / 'cache', workers=1)
    fine = classify_islands(graph, radius=0.1, anchor_source_ids=(0,), workers=1)
    np.testing.assert_array_equal(fine.classes, [0, 0, 2])
    assert fine.coarse_supported[graph.labels[2, 0]]
    coarse = classify_islands(graph, radius=0.2, anchor_source_ids=(0,), workers=1)
    np.testing.assert_array_equal(coarse.classes, 0)


def test_exact_body_distance_not_bounds_or_centroid(tmp_path: Path):
    xyz = np.array([[0, 0, 0], [1, 1, 0], [0.5, 0.5, 3]], dtype=np.float64)
    source = save_records(tmp_path / 'records.npy', xyz)
    np.save(tmp_path / 'ids.npy', np.arange(3, dtype=np.int64))
    graph = compute_island_graph(
        source, tmp_path / 'ids.npy', tmp_path / 'cache', radii=(1.5, 2.0), workers=1
    )
    selected = classify_islands(graph, radius=1.5, anchor_source_ids=(0,), workers=1)
    assert selected.distance_to_body[graph.labels[2, 0]] == pytest.approx(np.sqrt(9.5))


def test_cache_and_original_selection_isolation(tmp_path: Path):
    graph, _ = make_graph(tmp_path)
    again = compute_island_graph(
        tmp_path / 'records.npy', tmp_path / 'ids.npy', tmp_path / 'cache', workers=1
    )
    assert again.cached and not graph.cached
    np.testing.assert_array_equal(again.labels, graph.labels)
    assert not again.labels.flags.writeable and not again.source_ids.flags.writeable
    np.save(tmp_path / 'ids.npy', graph.source_ids[::2])
    changed = compute_island_graph(
        tmp_path / 'records.npy', tmp_path / 'ids.npy', tmp_path / 'cache', workers=1
    )
    assert not changed.cached and changed.selected_ids_sha256 != graph.selected_ids_sha256
    assert len(changed.source_ids) == len(graph.source_ids[::2])


@pytest.mark.parametrize('radii', [(), (0,), (-1,), (0.2, 0.1), (np.nan,), (True,)])
def test_invalid_radii(radii):
    with pytest.raises(PipelineInputError):
        distance_components(np.zeros((3, 3)), radii)


@pytest.mark.parametrize('anchors', [(), (-1,), (100000,), (0, 0)])
def test_invalid_anchors_preserve_graph(tmp_path: Path, anchors):
    graph, _ = make_graph(tmp_path)
    with pytest.raises(PipelineInputError):
        classify_islands(graph, anchor_source_ids=anchors)


def test_explicit_query_limit_is_not_neighbor_truncation():
    xyz = np.array([[0, 0, 0], [0.0625, 0, 0]], dtype=np.float64)
    with pytest.raises(PipelineInputError, match='edge batch limit'):
        distance_components(xyz, edge_batch_limit=1)
