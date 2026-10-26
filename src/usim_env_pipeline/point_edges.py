"""Streamed geometric edge-point features over immutable original observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256 as hash_bytes
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .models import PipelineInputError
from .pcd import sha256
from .storage import write_json

ALGORITHM_VERSION: Final = 'local-pca-angular-edge-points-v1'


@dataclass(frozen=True, slots=True)
class EdgeNeighborhood:
    neighbors: int = 64
    radius: float = 0.3
    min_neighbors: int = 8
    max_surface_variation: float = 0.08

    def __post_init__(self) -> None:
        if (
            type(self.neighbors) is not int
            or not 8 <= self.neighbors <= 1024
            or type(self.min_neighbors) is not int
            or not 3 <= self.min_neighbors <= self.neighbors
            or type(self.radius) not in (int, float)
            or not isfinite(self.radius)
            or self.radius <= 0
            or type(self.max_surface_variation) not in (int, float)
            or not isfinite(self.max_surface_variation)
            or not 0 < self.max_surface_variation < 1 / 3
        ):
            raise PipelineInputError('Invalid edge neighborhood or surface-variation limit')


@dataclass(frozen=True, slots=True)
class PointEdgeFeatures:
    normal_valid: NDArray[np.bool_]
    neighbor_counts: NDArray[np.uint16]
    surface_variation: NDArray[np.float32]
    boundary_gap_degrees: NDArray[np.float32]
    crease_angle_degrees: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class PointEdgeResult:
    source_ids: NDArray[np.int64]
    xyz: NDArray[np.float64]
    features: PointEdgeFeatures
    neighborhood: EdgeNeighborhood
    source_sha256: str
    selected_ids_sha256: str
    elapsed_seconds: float
    cached: bool
    cache_dir: Path


def point_edge_features(
    xyz: NDArray[np.float64],
    neighborhood: EdgeNeighborhood = EdgeNeighborhood(),
    *,
    workers: int = 8,
    batch_size: int = 2048,
) -> PointEdgeFeatures:
    """Estimate local PCA and angular gaps; no point snapping or graph deletion.

    Neighborhoods use up to k other observations within the closed physical
    radius. This is an adaptive kNN estimator, not a complete radius graph.
    Sparse, collinear and volumetric neighborhoods have no reliable surface.
    Crease angle is the third-largest unoriented normal change from the
    dominant reliable neighboring surface normal. The query's own normal
    need not be reliable at a sharp corner; six reliable neighbors are needed.
    One anomalous normal cannot create an edge.
    RAM is O(N) plus O(batch_size * k); neighborhoods are re-queried.
    """
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or xyz.dtype != np.float64
        or not len(xyz)
        or not np.isfinite(xyz).all()
        or type(workers) is not int
        or workers < 1
        or type(batch_size) is not int
        or batch_size < 1
    ):
        raise PipelineInputError(
            'Edges require finite float64 XYZ and positive worker/batch counts'
        )
    count = len(xyz)
    tree = cKDTree(xyz)
    k = neighborhood.neighbors
    normals = np.zeros((count, 3), dtype=np.float64)
    valid = np.zeros(count, dtype=np.bool_)
    counts = np.zeros(count, dtype=np.uint16)
    variation = np.zeros(count, dtype=np.float32)
    gaps = np.zeros(count, dtype=np.float32)
    jumps = np.zeros(count, dtype=np.float32)

    def query(
        start: int, stop: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.bool_]]:
        _, ids = tree.query(
            xyz[start:stop],
            k=k + 1,
            distance_upper_bound=np.nextafter(neighborhood.radius, np.inf),
            workers=workers,
        )
        present = (ids < count) & (ids != np.arange(start, stop)[:, None])
        ids = np.minimum(ids, count - 1)
        delta = xyz[ids] - xyz[start:stop, None, :]
        present &= np.sum(delta * delta, axis=2) <= neighborhood.radius**2
        # In zero-distance ties the query itself may be absent: still keep only k others.
        present &= np.cumsum(present, axis=1) <= k
        return ids, delta, present

    progress = -1
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        _, delta, present = query(start, stop)
        number = present.sum(axis=1)
        counts[start:stop] = number
        divisor = np.maximum(number, 1)
        offset = np.where(present[:, :, None], delta, 0)
        center = offset.sum(axis=1) / divisor[:, None]
        centered = np.where(present[:, :, None], delta - center[:, None, :], 0)
        covariance = np.einsum('bki,bkj->bij', centered, centered) / divisor[:, None, None]
        eigenvalues, vectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0)
        total = eigenvalues.sum(axis=1)
        ratio = np.divide(eigenvalues[:, 0], total, out=np.zeros(len(total)), where=total > 0)
        usable = (
            (number >= neighborhood.min_neighbors)
            & (eigenvalues[:, 1] > eigenvalues[:, 2] * 0.02)
            & (total > 0)
            & (ratio <= neighborhood.max_surface_variation)
        )
        valid[start:stop] = usable
        variation[start:stop] = ratio
        normals[start:stop] = vectors[:, :, 0]
        # The other orthonormal eigenvectors span the tangent plane, without a viewpoint.
        u = np.einsum('bki,bi->bk', delta, vectors[:, :, 2])
        v = np.einsum('bki,bi->bk', delta, vectors[:, :, 1])
        directions = present & (u * u + v * v > 1e-24)
        angles = np.sort(np.where(directions, np.arctan2(v, u), np.inf), axis=1)
        directions_count = directions.sum(axis=1)
        angles[~np.isfinite(angles)] = 0
        differences = np.diff(angles, axis=1, prepend=angles[:, :1])
        differences[np.arange(k + 1)[None, :] >= directions_count[:, None]] = 0
        last = angles[np.arange(stop - start), np.maximum(directions_count - 1, 0)]
        wrap = 2 * np.pi - last + angles[:, 0]
        gap = np.maximum(differences.max(axis=1), wrap)
        gaps[start:stop] = np.where(
            usable & (directions_count >= neighborhood.min_neighbors),
            np.rad2deg(gap),
            0,
        )
        bucket = stop * 10 // count
        if bucket != progress:
            print(f'POINT_EDGES_STAGE pca {stop}/{count}', flush=True)
            progress = bucket

    progress = -1
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        ids, delta, present = query(start, stop)
        pair_valid = present & valid[ids] & (np.sum(delta * delta, axis=2) > 1e-24)
        nearby_normals = np.where(pair_valid[:, :, None], normals[ids], 0)
        # nn^T is unchanged by normal sign flips, including perpendicular faces.
        tensor = np.einsum('bki,bkj->bij', nearby_normals, nearby_normals)
        _, axes = np.linalg.eigh(tensor)
        dot = np.abs(np.einsum('bi,bki->bk', axes[:, :, 2], nearby_normals))
        angle = np.where(pair_valid, np.rad2deg(np.arccos(np.clip(dot, 0, 1))), 0)
        enough = pair_valid.sum(axis=1) >= max(6, neighborhood.min_neighbors)
        jumps[start:stop] = np.where(
            enough, np.partition(angle, angle.shape[1] - 3, axis=1)[:, -3], 0
        )
        bucket = stop * 10 // count
        if bucket != progress:
            print(f'POINT_EDGES_STAGE normal-change {stop}/{count}', flush=True)
            progress = bucket
    for value in (valid, counts, variation, gaps, jumps):
        value.setflags(write=False)
    return PointEdgeFeatures(valid, counts, variation, gaps, jumps)


def select_edges(
    features: PointEdgeFeatures,
    *,
    boundary_angle: float = 135.0,
    crease_angle: float = 35.0,
    include_boundary: bool = True,
    include_crease: bool = True,
) -> NDArray[np.uint8]:
    """Readonly original-order flags: bit 1 surface boundary, bit 2 normal change."""
    if (
        type(boundary_angle) not in (int, float)
        or not isfinite(boundary_angle)
        or not 0 < boundary_angle <= 360
        or type(crease_angle) not in (int, float)
        or not isfinite(crease_angle)
        or not 0 < crease_angle <= 90
        or type(include_boundary) is not bool
        or type(include_crease) is not bool
    ):
        raise PipelineInputError('Boundary/crease angles or type toggles invalid')
    boundary = features.normal_valid & (features.boundary_gap_degrees >= boundary_angle)
    crease = features.crease_angle_degrees >= crease_angle
    flags: NDArray[np.uint8] = np.zeros(len(features.normal_valid), dtype=np.uint8)
    flags[boundary & include_boundary] = np.uint8(1)
    flags[crease & include_crease] |= np.uint8(2)
    flags.setflags(write=False)
    return flags


def compute_point_edges(
    records_path: Path,
    source_ids_path: Path,
    cache_dir: Path,
    *,
    neighborhood: EdgeNeighborhood = EdgeNeighborhood(),
    workers: int = 8,
    batch_size: int = 2048,
) -> PointEdgeResult:
    """Cache edge features by original content, frozen LOF IDs and neighborhood."""
    started = perf_counter()
    if type(workers) is not int or workers < 1 or type(batch_size) is not int or batch_size < 1:
        raise PipelineInputError('Workers and batch size must be positive integers')
    source_hash, ids_hash = sha256(records_path), sha256(source_ids_path)
    try:
        records = np.load(records_path, mmap_mode='r', allow_pickle=False)
        ids = np.load(source_ids_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise PipelineInputError('Cannot load source records or frozen LOF IDs') from error
    if (
        records.ndim != 1
        or records.dtype.fields is None
        or not {'x', 'y', 'z'}.issubset(records.dtype.fields)
        or any(
            records.dtype[name].shape or records.dtype[name].kind not in 'biuf' for name in 'xyz'
        )
        or ids.ndim != 1
        or ids.dtype != np.int64
        or not len(ids)
        or ids[0] < 0
        or ids[-1] >= len(records)
        or np.any(ids[1:] <= ids[:-1])
    ):
        raise PipelineInputError('XYZ records and increasing unique int64 source IDs required')
    xyz = np.column_stack([records[name][ids] for name in 'xyz']).astype(np.float64)
    del records
    if not np.isfinite(xyz).all():
        raise PipelineInputError('Frozen LOF XYZ must be finite')
    metadata = {
        'algorithm': ALGORITHM_VERSION,
        'source_sha256': source_hash,
        'selected_ids_sha256': ids_hash,
        'selected_records': len(ids),
        'neighborhood': asdict(neighborhood),
    }
    identity = hash_bytes(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    owned = cache_dir / f'{ALGORITHM_VERSION}-{identity}'
    completion = owned / 'complete.json'
    cached = completion.exists()
    if cached:
        if completion.read_text(encoding='utf-8') != json.dumps(metadata, indent=2) + '\n':
            raise PipelineInputError('Completed edge metadata disagrees with input')
    else:
        owned.mkdir(parents=True, exist_ok=True)
        computed = point_edge_features(xyz, neighborhood, workers=workers, batch_size=batch_size)
        for field in fields(computed):
            np.save(owned / f'{field.name}.npy', getattr(computed, field.name))
        if sha256(records_path) != source_hash or sha256(source_ids_path) != ids_hash:
            raise PipelineInputError('Input changed during edge calculation')
        write_json(completion, metadata)
    arrays = [
        np.load(owned / f'{field.name}.npy', mmap_mode='r', allow_pickle=False)
        for field in fields(PointEdgeFeatures)
    ]
    dtypes = (np.bool_, np.uint16, np.float32, np.float32, np.float32)
    if any(
        value.shape != (len(ids),) or value.dtype != dtype or not np.isfinite(value).all()
        for value, dtype in zip(arrays, dtypes)
    ):
        raise PipelineInputError('Completed edge features have invalid shape, dtype or values')
    if (
        np.any(arrays[1] > neighborhood.neighbors)
        or np.any(arrays[2] < 0)
        or np.any(arrays[2] > 1 / 3 + 1e-7)
        or np.any(arrays[3] < 0)
        or np.any(arrays[3] > 360)
        or np.any(arrays[4] < 0)
        or np.any(arrays[4] > 90)
    ):
        raise PipelineInputError('Completed edge feature ranges invalid')
    if sha256(records_path) != source_hash or sha256(source_ids_path) != ids_hash:
        raise PipelineInputError('Input changed while loading edge features')
    ids.setflags(write=False)
    xyz.setflags(write=False)
    return PointEdgeResult(
        ids,
        xyz,
        PointEdgeFeatures(*arrays),
        neighborhood,
        source_hash,
        ids_hash,
        perf_counter() - started,
        cached,
        owned,
    )
