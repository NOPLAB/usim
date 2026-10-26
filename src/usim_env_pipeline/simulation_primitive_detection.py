"""Connected, normal-supported regional plane extraction from unmatched measurements."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .plane_types import PlaneConfig, plane_basis
from .planes import extract_planes, refit
from .storage import write_json


def detect_primitives(
    points: NDArray[np.float32], available: NDArray[np.bool_], folder: Path
) -> int:
    """Generate supported plane hypotheses, not bounding boxes around arbitrary clusters.

    Original IDs survive sampling and segmentation. Normal agreement is measured
    against the complete regional residual cloud, not only an already flattened
    inlier set. Connected support and footprint coverage reject porous candidates.
    """
    ids = np.flatnonzero(available)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    if not len(ids):
        write_json(folder / 'index.json', rows)
        return 0
    regions, groups = np.unique(
        np.floor(points[ids, :2] / 10).astype(np.int64), axis=0, return_inverse=True
    )
    order = np.argsort(groups, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(groups[order])) + 1, len(order)]
    config = PlaneConfig(
        voxel=0.1, tolerance=0.04, min_support=128, max_planes=6, iterations=250, seed=7
    )
    for region_number, region in enumerate(regions):
        members = ids[order[starts[region_number] : starts[region_number + 1]]]
        if len(members) < config.min_support:
            continue
        _, selected = np.unique(
            np.floor(points[members] / config.voxel), axis=0, return_index=True
        )
        samples = points[members[selected]].astype(np.float64)
        tree = cKDTree(samples)
        normals = np.zeros_like(samples)
        planar = np.zeros(len(samples), bool)
        for start in range(0, len(samples), 4096):
            distance, neighbors = tree.query(
                samples[start : start + 4096], k=16, distance_upper_bound=0.3, workers=4
            )
            present = neighbors < len(samples)
            local = samples[np.minimum(neighbors, len(samples) - 1)]
            count = present.sum(1)
            mean = (local * present[..., None]).sum(1) / np.maximum(count, 1)[:, None]
            delta = (local - mean[:, None]) * present[..., None]
            eigen, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', delta, delta))
            end = start + len(local)
            normals[start:end] = vectors[:, :, 0]
            planar[start:end] = (
                (count >= 8) & (eigen[:, 1] > 1e-8) & (eigen[:, 0] <= 0.05 * eigen.sum(1))
            )
        for candidate in extract_planes(points, members, config):
            u, v = plane_basis(candidate.normal)
            uv = np.column_stack((points[candidate.ids] @ u, points[candidate.ids] @ v))
            cells, first, inverse = np.unique(
                np.floor(uv / 0.1).astype(np.int64),
                axis=0,
                return_index=True,
                return_inverse=True,
            )
            pairs = cKDTree(uv[first]).query_pairs(0.25, output_type='ndarray')
            graph = coo_matrix(
                (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                shape=(len(cells), len(cells)),
            )
            count, labels = connected_components(graph, directed=False)
            for component in range(count):
                component_ids = candidate.ids[labels[inverse] == component]
                occupied = cells[labels == component]
                area = len(occupied) * 0.1**2
                bounds_area = np.prod(np.ptp(occupied, axis=0) * 0.1 + 0.1)
                if len(component_ids) < 128 or area < 0.5 or area / bounds_area < 0.45:
                    continue
                plane = refit(points, np.sort(component_ids))
                nearest = tree.query(points[plane.ids], workers=4)[1]
                coherent = planar[nearest] & (
                    np.abs(normals[nearest] @ plane.normal) >= np.cos(np.deg2rad(20))
                )
                if coherent.mean() < 0.8:
                    continue
                residual = np.abs(points[plane.ids] @ plane.normal + plane.offset)
                support_ids = plane.ids[residual <= config.tolerance]
                if len(support_ids) < config.min_support:
                    continue
                number = len(rows)
                np.savez_compressed(
                    folder / f'plane_{number:04}.npz',
                    equation=np.r_[plane.normal, plane.offset],
                    ids=support_ids,
                )
                rows.append(
                    {
                        'source_plane': number,
                        'region': region.tolist(),
                        'records': len(support_ids),
                        'normal_coherence': float(coherent.mean()),
                        'observed_area_m2': float(area),
                    }
                )
    write_json(folder / 'index.json', rows)
    return len(rows)
