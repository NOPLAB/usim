"""Bounded-memory exact distance components and conservative floating-island candidates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256 as hash_bytes
from itertools import chain
import json
from math import isfinite, sqrt
from pathlib import Path
from time import perf_counter
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .models import PipelineInputError
from .pcd import sha256
from .storage import write_json

ALGORITHM_VERSION: Final = 'exact-clique-streamed-islands-v1'


@dataclass(frozen=True, slots=True)
class IslandComponents:
    counts: NDArray[np.int64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    shape_ratios: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class IslandGraph:
    source_ids: NDArray[np.int64]
    xyz: NDArray[np.float64]
    radii: tuple[float, ...]
    labels: NDArray[np.int32]
    components: tuple[IslandComponents, ...]
    source_sha256: str
    selected_ids_sha256: str
    elapsed_seconds: float
    cached: bool
    cache_dir: Path


@dataclass(frozen=True, slots=True)
class IslandSelection:
    classes: NDArray[np.uint8]
    component_classes: NDArray[np.uint8]
    distance_to_body: NDArray[np.float64]
    supported: NDArray[np.bool_]
    coarse_supported: NDArray[np.bool_]
    anchor_source_ids: tuple[int, ...]
    elapsed_seconds: float


def _roots(parent: NDArray[np.int64], nodes: NDArray[np.int64]) -> NDArray[np.int64]:
    """Compress the mutable component forest for queried nodes."""
    roots = parent[nodes]
    while True:
        next_roots = parent[roots]
        if np.array_equal(roots, next_roots):
            break
        roots = next_roots
    parent[nodes] = roots
    return roots


def _join(parent: NDArray[np.int64], first: NDArray[np.int64], second: NDArray[np.int64]) -> None:
    """Merge a batch in native sparse-graph code, not a Python loop per edge."""
    first, second = _roots(parent, first), _roots(parent, second)
    different = first != second
    if not different.any():
        return
    first, second = first[different], second[different]
    vertices, inverse = np.unique(np.r_[first, second], return_inverse=True)
    length = len(first)
    graph = coo_matrix(
        (np.ones(length, dtype=np.bool_), (inverse[:length], inverse[length:])),
        shape=(len(vertices), len(vertices)),
    )
    count, labels = connected_components(graph, directed=False)
    representatives = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(representatives, labels, vertices)
    parent[vertices] = representatives[labels]


def distance_components(
    xyz: NDArray[np.float64],
    radii: tuple[float, ...] = (0.1, 0.2, 0.4),
    *,
    workers: int = 8,
    edge_batch_limit: int = 1_000_000,
) -> NDArray[np.int32]:
    """Closed Euclidean radius components; cells only compress guaranteed cliques.

    Every distinct-cell point pair within the largest radius is considered.
    An edge already connected at the smallest radius can be skipped at all
    larger radii by graph nesting. A batch is reduced over its current roots;
    no full radius graph or fixed-k truncated neighborhood is ever stored.
    """
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or xyz.dtype != np.float64
        or not len(xyz)
        or len(xyz) >= np.iinfo(np.int32).max
        or not np.isfinite(xyz).all()
    ):
        raise PipelineInputError('XYZ must be a nonempty finite Nx3 float64 array')
    if (
        not isinstance(radii, tuple)
        or not radii
        or any(type(r) not in (int, float) or not isfinite(r) or r <= 0 for r in radii)
        or any(a >= b for a, b in zip(radii[:-1], radii[1:]))
    ):
        raise PipelineInputError('Radii must be increasing distinct positive finite values')
    if (
        type(workers) is not int
        or workers < 1
        or type(edge_batch_limit) is not int
        or edge_batch_limit < 1
    ):
        raise PipelineInputError('Workers and edge batch limit must be positive integers')
    count = len(xyz)
    cell = radii[0] * 0.99 / sqrt(3)
    scaled = (xyz - xyz.min(axis=0)) / cell
    if not np.isfinite(scaled).all() or scaled.max() >= 2**52:
        raise PipelineInputError('Coordinate extent exceeds exact spatial-index precision')
    keys = np.floor(scaled).astype(np.int64)
    del scaled
    _, cells = np.unique(keys, axis=0, return_inverse=True)
    del keys
    cell_count = int(cells.max()) + 1
    low = np.full((cell_count, 3), np.inf)
    high = np.full((cell_count, 3), -np.inf)
    for axis in range(3):
        np.minimum.at(low[:, axis], cells, xyz[:, axis])
        np.maximum.at(high[:, axis], cells, xyz[:, axis])
    if np.any(np.sum((high - low) ** 2, axis=1) > radii[0] ** 2):
        raise PipelineInputError('Floating-point spatial indexing violated the cell-clique bound')
    del low, high
    parents = [np.arange(cell_count, dtype=np.int64) for _ in radii]
    print(f'ISLAND_STAGE cliques rows={count} cells={cell_count}', flush=True)
    if cell_count > 1:
        tree = cKDTree(xyz)
        query_radius = float(np.nextafter(radii[-1], np.inf))
        maximum_neighbors = 0
        total_neighbors = 0
        cumulative_neighbors = np.empty(count + 1, dtype=np.int64)
        cumulative_neighbors[0] = 0
        for start in range(0, count, 32768):
            sizes = tree.query_ball_point(
                xyz[start : start + 32768],
                query_radius,
                return_length=True,
                workers=workers,
            )
            maximum_neighbors = max(maximum_neighbors, int(sizes.max()))
            total_neighbors += int(sizes.sum(dtype=np.int64))
            cumulative_neighbors[start + 1 : start + len(sizes) + 1] = cumulative_neighbors[
                start
            ] + np.cumsum(sizes)
        if maximum_neighbors > edge_batch_limit:
            raise PipelineInputError(
                'One radius query exceeds the edge batch limit; increase the explicit limit'
            )
        print(
            f'ISLAND_STAGE preflight directed_neighbors={total_neighbors} '
            f'maximum_neighbors={maximum_neighbors} edge_batch_limit={edge_batch_limit}',
            flush=True,
        )
        previous = -1
        start = 0
        while start < count:
            end = min(
                start + 8192,
                count,
                int(
                    np.searchsorted(
                        cumulative_neighbors,
                        cumulative_neighbors[start] + edge_batch_limit,
                        side='right',
                    )
                )
                - 1,
            )
            neighbors = tree.query_ball_point(
                xyz[start:end], query_radius, workers=workers, return_sorted=False
            )
            sizes = np.fromiter((len(row) for row in neighbors), dtype=np.int64)
            first = np.repeat(np.arange(start, end, dtype=np.int64), sizes)
            second = np.fromiter(chain.from_iterable(neighbors), dtype=np.int64)
            del neighbors
            useful = (first < second) & (cells[first] != cells[second])
            first, second = first[useful], second[useful]
            first_cells, second_cells = cells[first], cells[second]
            distinct = _roots(parents[0], first_cells) != _roots(parents[0], second_cells)
            first, second = first[distinct], second[distinct]
            first_cells, second_cells = first_cells[distinct], second_cells[distinct]
            delta = xyz[first] - xyz[second]
            squared = (delta[:, 0] ** 2 + delta[:, 1] ** 2) + delta[:, 2] ** 2
            for parent, radius in zip(parents, radii):
                accepted = squared <= radius * radius
                _join(parent, first_cells[accepted], second_cells[accepted])
            bucket = end * 20 // count
            if bucket != previous:
                print(f'ISLAND_STAGE edges {end}/{count}', flush=True)
                previous = bucket
            start = end
    labels = np.empty((count, len(radii)), dtype=np.int32)
    all_cells = np.arange(cell_count, dtype=np.int64)
    for column, parent in enumerate(parents):
        roots = _roots(parent, all_cells)[cells]
        _, inverse = np.unique(roots, return_inverse=True)
        # Canonical order follows the first original-order observation, not grid origin.
        first_rows = np.full(int(inverse.max()) + 1, count, dtype=np.int64)
        np.minimum.at(first_rows, inverse, np.arange(count))
        order = np.argsort(first_rows)
        remap = np.empty(len(order), dtype=np.int32)
        remap[order] = np.arange(len(order), dtype=np.int32)
        labels[:, column] = remap[inverse]
    return labels


def _component_stats(xyz: NDArray[np.float64], labels: NDArray[np.int32]) -> IslandComponents:
    count = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=count).astype(np.int64)
    low = np.full((count, 3), np.inf)
    high = np.full((count, 3), -np.inf)
    for axis in range(3):
        np.minimum.at(low[:, axis], labels, xyz[:, axis])
        np.maximum.at(high[:, axis], labels, xyz[:, axis])
    centered = xyz - low[labels]
    mean = np.column_stack(
        [np.bincount(labels, weights=centered[:, axis]) / sizes for axis in range(3)]
    )
    covariance = np.empty((count, 3, 3), dtype=np.float64)
    for first in range(3):
        for second in range(first, 3):
            value = (
                np.bincount(labels, weights=centered[:, first] * centered[:, second]) / sizes
                - mean[:, first] * mean[:, second]
            )
            covariance[:, first, second] = value
            covariance[:, second, first] = value
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
    ratios = np.ones((count, 2), dtype=np.float64)
    np.divide(
        eigenvalues[:, 0], eigenvalues[:, 1], out=ratios[:, 0], where=eigenvalues[:, 1] > 1e-24
    )
    np.divide(
        eigenvalues[:, 1], eigenvalues[:, 2], out=ratios[:, 1], where=eigenvalues[:, 2] > 1e-24
    )
    for values in (sizes, low, high, ratios):
        values.setflags(write=False)
    return IslandComponents(sizes, low, high, ratios)


def compute_island_graph(
    records_path: Path,
    source_ids_path: Path,
    cache_dir: Path,
    *,
    radii: tuple[float, ...] = (0.1, 0.2, 0.4),
    workers: int = 8,
) -> IslandGraph:
    """Cache exact components of a frozen original-ID selection without editing its source."""
    started = perf_counter()
    source_hash, ids_hash = sha256(records_path), sha256(source_ids_path)
    try:
        ids: NDArray[np.int64] = np.load(source_ids_path, mmap_mode='r', allow_pickle=False)
        records: NDArray[np.void] = np.load(records_path, mmap_mode='r', allow_pickle=False)
    except (OSError, ValueError) as error:
        raise PipelineInputError('Cannot load original records or selected source IDs') from error
    if (
        records.ndim != 1
        or records.dtype.fields is None
        or not {'x', 'y', 'z'}.issubset(records.dtype.fields)
        or ids.ndim != 1
        or ids.dtype != np.int64
        or not len(ids)
        or ids[0] < 0
        or ids[-1] >= len(records)
        or np.any(ids[1:] <= ids[:-1])
    ):
        raise PipelineInputError('Selection must contain increasing unique int64 original IDs')
    # Snapshot IDs without retaining a Windows file lock on the input selection.
    ids = ids.copy()
    ids.setflags(write=False)
    if (
        not isinstance(radii, tuple)
        or not radii
        or any(type(r) not in (int, float) or not isfinite(r) or r <= 0 for r in radii)
        or any(a >= b for a, b in zip(radii[:-1], radii[1:]))
        or type(workers) is not int
        or workers < 1
    ):
        raise PipelineInputError('Radii and workers are invalid')
    for name in ('x', 'y', 'z'):
        field = records.dtype[name]
        if field.shape or field.kind not in 'biuf':
            raise PipelineInputError('Original XYZ fields must be scalar numeric values')
    metadata = {
        'algorithm': ALGORITHM_VERSION,
        'source_sha256': source_hash,
        'selected_ids_sha256': ids_hash,
        'radii': list(radii),
        'edge_predicate': 'float64 squared_distance <= radius_squared',
        'selected_records': len(ids),
    }
    identity = hash_bytes(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    owned = cache_dir / f'{ALGORITHM_VERSION}-{identity}'
    xyz: NDArray[np.float64] = np.column_stack(
        [records[name][ids] for name in ('x', 'y', 'z')]
    ).astype(np.float64)
    del records
    if not np.isfinite(xyz).all():
        raise PipelineInputError('All selected XYZ must be finite')
    completion = owned / 'complete.json'
    cached = completion.exists()
    if cached:
        if completion.read_text(encoding='utf-8') != json.dumps(metadata, indent=2) + '\n':
            raise PipelineInputError('Completed island graph metadata disagrees with input')
    else:
        owned.mkdir(parents=True, exist_ok=True)
        labels = distance_components(xyz, radii, workers=workers)
        np.save(owned / 'labels.npy', labels)
        for column in range(len(radii)):
            stats = _component_stats(xyz, labels[:, column])
            for name in ('counts', 'lower', 'upper', 'shape_ratios'):
                np.save(owned / f'{column}-{name}.npy', getattr(stats, name))
        if sha256(records_path) != source_hash or sha256(source_ids_path) != ids_hash:
            raise PipelineInputError('Input changed during island component computation')
        write_json(completion, metadata)
    labels = np.load(owned / 'labels.npy', mmap_mode='r', allow_pickle=False)
    if labels.shape != (len(ids), len(radii)) or labels.dtype != np.int32:
        raise PipelineInputError('Completed component labels have invalid shape or dtype')
    components: list[IslandComponents] = []
    for column in range(len(radii)):
        sizes = np.load(owned / f'{column}-counts.npy', mmap_mode='r', allow_pickle=False)
        low = np.load(owned / f'{column}-lower.npy', mmap_mode='r', allow_pickle=False)
        high = np.load(owned / f'{column}-upper.npy', mmap_mode='r', allow_pickle=False)
        ratios = np.load(owned / f'{column}-shape_ratios.npy', mmap_mode='r', allow_pickle=False)
        if (
            sizes.ndim != 1
            or sizes.dtype != np.int64
            or np.any(sizes <= 0)
            or sizes.sum() != len(ids)
            or low.shape != (len(sizes), 3)
            or high.shape != low.shape
            or ratios.shape != (len(sizes), 2)
            or any(value.dtype != np.float64 for value in (low, high, ratios))
            or any(not np.isfinite(value).all() for value in (low, high, ratios))
            or np.any(low > high)
            or labels[:, column].min() < 0
            or labels[:, column].max() >= len(sizes)
            or not np.array_equal(np.bincount(labels[:, column]), sizes)
        ):
            raise PipelineInputError('Completed component statistics disagree with labels')
        components.append(IslandComponents(sizes, low, high, ratios))
    if sha256(records_path) != source_hash or sha256(source_ids_path) != ids_hash:
        raise PipelineInputError('Input changed while loading island graph')
    xyz.setflags(write=False)
    return IslandGraph(
        ids,
        xyz,
        radii,
        labels,
        tuple(components),
        source_hash,
        ids_hash,
        perf_counter() - started,
        cached,
        owned,
    )


def classify_islands(
    graph: IslandGraph,
    *,
    radius: float = 0.2,
    anchor_source_ids: tuple[int, ...] = (59854, 94908),
    max_diameter: float = 1.0,
    min_body_gap: float = 0.4,
    protect_planar: bool = True,
    protect_linear: bool = True,
    workers: int = 8,
) -> IslandSelection:
    """Flag small persistently unsupported components; planar/linear evidence stays review.

    The largest-radius component containing each selected component must also
    be small, unsupported and separated from the anchor-connected body. This
    prevents fragments of a larger disconnected surface being labeled removable.
    No source is deleted; the classifications are candidates, not truth labels.
    """
    started = perf_counter()
    if (
        radius not in graph.radii
        or type(max_diameter) not in (int, float)
        or not isfinite(max_diameter)
        or max_diameter <= 0
        or type(min_body_gap) not in (int, float)
        or not isfinite(min_body_gap)
        or min_body_gap < 0
        or type(workers) is not int
        or workers < 1
        or type(protect_planar) is not bool
        or type(protect_linear) is not bool
        or not isinstance(anchor_source_ids, tuple)
        or not anchor_source_ids
        or any(type(value) is not int for value in anchor_source_ids)
        or len(set(anchor_source_ids)) != len(anchor_source_ids)
    ):
        raise PipelineInputError(
            'Radius, anchor IDs or candidate classification parameters invalid'
        )
    anchor_rows = np.searchsorted(graph.source_ids, anchor_source_ids)
    if np.any(anchor_rows >= len(graph.source_ids)) or not np.array_equal(
        graph.source_ids[anchor_rows], anchor_source_ids
    ):
        raise PipelineInputError('Every anchor must be an actual retained original source ID')
    column = graph.radii.index(radius)
    labels = graph.labels[:, column]
    coarse = graph.labels[:, -1]
    count, coarse_count = len(graph.components[column].counts), len(graph.components[-1].counts)
    selected_supported = np.zeros(count, dtype=np.bool_)
    selected_supported[labels[anchor_rows]] = True
    coarse_supported = np.zeros(coarse_count, dtype=np.bool_)
    coarse_supported[coarse[anchor_rows]] = True
    parent_low = np.full(count, coarse_count, dtype=np.int32)
    parent_high = np.full(count, -1, dtype=np.int32)
    np.minimum.at(parent_low, labels, coarse)
    np.maximum.at(parent_high, labels, coarse)
    if not np.array_equal(parent_low, parent_high):
        raise PipelineInputError('Radius component hierarchy is not nested')
    body = coarse_supported[coarse]
    body_hash = hash_bytes(body.tobytes()).hexdigest()
    distances_path = graph.cache_dir / f'body-{body_hash}.npy'
    if not distances_path.exists():
        # Exact nearest distances are stored per source row, allowing component
        # inspection at every radius without bounding-box or centroid estimates.
        tree = cKDTree(graph.xyz[body])
        distances = np.zeros(len(body), dtype=np.float64)
        for start in range(0, len(body), 8192):
            stop = min(start + 8192, len(body))
            local = np.flatnonzero(~body[start:stop]) + start
            if len(local):
                queried, _ = tree.query(graph.xyz[local], k=1, eps=0, workers=workers)
                distances[local] = queried
        pending = distances_path.with_suffix('.pending.npy')
        np.save(pending, distances)
        pending.replace(distances_path)
    try:
        distances = np.load(distances_path, mmap_mode='r', allow_pickle=False)
    except (OSError, ValueError) as error:
        raise PipelineInputError('Cannot load completed anchor-body distances') from error
    if (
        distances.shape != (len(body),)
        or distances.dtype != np.float64
        or not np.isfinite(distances).all()
        or np.any(distances < 0)
        or np.any(distances[body] != 0)
    ):
        raise PipelineInputError('Completed anchor-body distances are invalid')
    gaps = np.full(count, np.inf)
    np.minimum.at(gaps, labels, distances)
    coarse_gaps = np.full(coarse_count, np.inf)
    np.minimum.at(coarse_gaps, coarse, distances)
    coarse_stats = graph.components[-1]
    diameter = np.linalg.norm(coarse_stats.upper - coarse_stats.lower, axis=1)
    planar = (
        (coarse_stats.counts >= 12)
        & (coarse_stats.shape_ratios[:, 0] <= 0.04)
        & (coarse_stats.shape_ratios[:, 1] > 0.04)
    )
    linear = (coarse_stats.counts >= 6) & (coarse_stats.shape_ratios[:, 1] <= 0.04)
    removable = (
        ~coarse_supported
        & (diameter <= max_diameter)
        & (coarse_gaps >= min_body_gap)
        & ~(planar & protect_planar)
        & ~(linear & protect_linear)
    )
    statuses = np.full(count, 2, dtype=np.uint8)
    statuses[removable[parent_low]] = 1
    statuses[selected_supported] = 0
    classes = statuses[labels]
    inherited_support = coarse_supported[parent_low]
    for values in (classes, statuses, gaps, selected_supported, inherited_support):
        values.setflags(write=False)
    return IslandSelection(
        classes,
        statuses,
        gaps,
        selected_supported,
        inherited_support,
        anchor_source_ids,
        perf_counter() - started,
    )


@dataclass(init=False)
class _Arguments(argparse.Namespace):
    records: Path
    source_ids: Path
    cache: Path
    anchors: list[int]
    radius: float
    max_diameter: float
    min_body_gap: float
    workers: int
    output: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument('--records', type=Path, required=True)
    _ = parser.add_argument('--source-ids', type=Path, required=True)
    _ = parser.add_argument('--cache', type=Path, required=True)
    _ = parser.add_argument('--anchors', type=int, nargs='+', default=[59854, 94908])
    _ = parser.add_argument('--radius', type=float, default=0.2)
    _ = parser.add_argument('--max-diameter', type=float, default=1.0)
    _ = parser.add_argument('--min-body-gap', type=float, default=0.4)
    _ = parser.add_argument('--workers', type=int, default=8)
    _ = parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(namespace=_Arguments())
    graph = compute_island_graph(args.records, args.source_ids, args.cache, workers=args.workers)
    selection = classify_islands(
        graph,
        radius=args.radius,
        anchor_source_ids=tuple(args.anchors),
        max_diameter=args.max_diameter,
        min_body_gap=args.min_body_gap,
        workers=args.workers,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / 'classes.npy', selection.classes)
    report = {
        'source_sha256': graph.source_sha256,
        'selected_ids_sha256': graph.selected_ids_sha256,
        'selected_records': len(graph.source_ids),
        'radius': args.radius,
        'radii': list(graph.radii),
        'components': [len(value.counts) for value in graph.components],
        'anchors': list(selection.anchor_source_ids),
        'max_diameter': args.max_diameter,
        'min_body_gap': args.min_body_gap,
        'retained': int(np.count_nonzero(selection.classes == 0)),
        'candidate': int(np.count_nonzero(selection.classes == 1)),
        'review': int(np.count_nonzero(selection.classes == 2)),
        'graph_seconds': graph.elapsed_seconds,
        'classification_seconds': selection.elapsed_seconds,
        'cached': graph.cached,
    }
    write_json(args.output / 'result.json', report)
    print('POINT_ISLANDS_COMPLETE ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
