"""CPU occupied-voxel point counts before and after a source-order SOR selection."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final, TypedDict
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError

VOXEL_SIZE_M: Final = 0.1


class CountSummary(TypedDict):
    min: int | None
    max: int | None
    mean: float | None
    median: float | None
    occupied_voxels: int
    total_points: int


@dataclass(frozen=True, slots=True)
class VoxelStatistics:
    """Sparse integer cell coordinates, counts, and exact count-frequency distribution."""

    indices: NDArray[np.int64]
    counts: NDArray[np.int64]
    point_counts: NDArray[np.int64]
    frequencies: NDArray[np.int64]
    summary: CountSummary


def occupied_voxels(
    records: NDArray[np.void] | NDArray[np.float64],
    keep: NDArray[np.bool_] | None = None,
) -> VoxelStatistics:
    """Count every selected row in floor(XYZ * 10) cells, including coincident rows.

    Accept structured source records with x/y/z fields or an (N, 3) XYZ array.
    Arithmetic uses source coordinates promoted to float64, without boundary snapping.
    Only occupied cells are returned; no dense array spans the world bounds.
    """
    if keep is not None:
        if keep.dtype != np.bool_ or keep.shape != (len(records),):
            raise PipelineInputError('Keep mask must be boolean and match source rows')
    if records.dtype.names is not None:
        if records.ndim != 1 or not {'x', 'y', 'z'}.issubset(records.dtype.names):
            raise PipelineInputError('Source records must contain scalar x/y/z fields')
        fields = records.dtype.fields
        assert fields is not None
        points = np.column_stack(
            [records.getfield(fields[name][0], fields[name][1]) for name in ('x', 'y', 'z')]
        )
    else:
        points = records
    points = np.array(points, dtype=np.float64, copy=True)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise PipelineInputError('XYZ must be a finite (N, 3) array')
    if keep is not None:
        points = points[keep]
    # Reciprocal multiplication keeps nominal decimal 0.3/0.7 boundaries exact.
    # In-place operations bound the working set and never mutate caller-owned XYZ.
    points *= 1.0 / VOXEL_SIZE_M
    np.floor(points, out=points)
    if np.any(points < -(2.0**63)) or np.any(points >= 2.0**63):
        raise PipelineInputError('Voxel indices exceed the int64 coordinate range')
    cells = points.astype(np.int64)
    total_points = len(points)
    del points
    indices, counts = np.unique(cells, axis=0, return_counts=True)
    del cells
    counts = counts.astype(np.int64, copy=False)
    point_counts, frequencies = np.unique(counts, return_counts=True)
    summary: CountSummary = {
        'min': int(counts.min()) if counts.size else None,
        'max': int(counts.max()) if counts.size else None,
        'mean': float(counts.mean()) if counts.size else None,
        'median': float(np.median(counts)) if counts.size else None,
        'occupied_voxels': len(counts),
        'total_points': total_points,
    }
    return VoxelStatistics(indices, counts, point_counts, frequencies, summary)


def plot_distribution(before: VoxelStatistics, after: VoxelStatistics, path: Path) -> None:
    """Render exact frequencies with a full-range log-log view and linear low-count view."""
    plot_populations([('Before SOR', before, '#0072B2'), ('After SOR', after, '#D55E00')], path)


def plot_populations(populations: Sequence[tuple[str, VoxelStatistics, str]], path: Path) -> None:
    """Compare staged selections on the same full and low-count axes."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    figure = Figure(figsize=(12, 5.5), constrained_layout=True)
    _ = FigureCanvasAgg(figure)
    full = figure.add_subplot(1, 2, 1)
    low = figure.add_subplot(1, 2, 2)
    maximum = max(stats.summary['max'] or 1 for _, stats, _ in populations)
    low_limit = min(maximum, 50)
    for label, stats, color in populations:
        full.scatter(
            stats.point_counts, stats.frequencies, s=18, alpha=0.75, color=color, label=label
        )
        selected = stats.point_counts <= low_limit
        low.plot(
            stats.point_counts[selected],
            stats.frequencies[selected],
            marker='o',
            markersize=4,
            color=color,
            label=label,
        )
    full.set_xscale('log')
    full.set_yscale('log')
    full.set_xlim(0.8, max(1.25, maximum * 1.2))
    highest_frequency = max(
        int(stats.frequencies.max()) if stats.frequencies.size else 1
        for _, stats, _ in populations
    )
    full.set_ylim(0.8, max(1.25, highest_frequency * 1.3))
    full.set_title('Full distribution (log-log; all occupied voxels)')
    low.set_title(f'Low-count detail (linear; 1-{low_limit} points)')
    low.set_xlim(0.5, low_limit + 0.5)
    low.set_ylim(bottom=0)
    low.xaxis.set_major_locator(MaxNLocator(integer=True))
    for axis in (full, low):
        axis.set_xlabel('Points per occupied 0.1 m voxel (records)')
        axis.set_ylabel('Occupied voxels (count)')
        axis.grid(alpha=0.2)
        axis.legend()
    if not any(stats.counts.size for _, stats, _ in populations):
        full.text(0.5, 0.5, 'No occupied voxels', transform=full.transAxes, ha='center')
    _ = figure.suptitle('Point density on the same source-origin grid')
    figure.savefig(path, dpi=160)


def export_comparison(
    filter_output: Path, destination: Path
) -> tuple[VoxelStatistics, VoxelStatistics]:
    """Read SOR source rows and decisions; export before.csv, after.csv, summary.json and PNG."""
    records: NDArray[np.void] = np.load(
        filter_output / 'records.npy', mmap_mode='r', allow_pickle=False
    )
    with np.load(filter_output / 'decisions.npz', allow_pickle=False) as decisions:
        keep: NDArray[np.bool_] = decisions['keep']
        ids: NDArray[np.int64] = decisions['ids']
        sor_keep: NDArray[np.bool_] | None = (
            decisions['sor_keep'] if 'sor_keep' in decisions else None
        )
    if ids.dtype != np.int64 or not np.array_equal(ids, np.arange(len(records))):
        raise PipelineInputError('SOR decisions must follow original source-row IDs')
    del ids
    before, after = occupied_voxels(records), occupied_voxels(records, keep)
    after_sor = occupied_voxels(records, sor_keep) if sor_keep is not None else None
    if sor_keep is not None and np.any(keep & ~sor_keep):
        raise PipelineInputError('Final selection must be a subset of the SOR selection')
    series = [('before', before), ('after', after)]
    if after_sor is not None:
        series.insert(1, ('after_sor', after_sor))
    destination.mkdir(parents=True, exist_ok=True)
    for name, stats in series:
        with (destination / f'{name}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('points_per_voxel', 'occupied_voxels'))
            writer.writerows(zip(stats.point_counts.tolist(), stats.frequencies.tolist()))
    summary = {
        'grid': {
            'voxel_size_m': VOXEL_SIZE_M,
            'origin_m': [0.0, 0.0, 0.0],
            'index_rule': 'floor((XYZ - origin_m) / voxel_size_m)',
            'cell_bounds': '[origin + index * size, origin + (index + 1) * size)',
            'coordinates': 'source XYZ in meters; float64 XYZ * 10; no boundary snapping',
            'population': 'nonempty voxels only, separately for each cloud',
            'duplicates': 'each source record counts as a point',
        },
        'before': before.summary,
        'after': after.summary,
    }
    if after_sor is not None:
        summary['after_sor'] = after_sor.summary
    with (destination / 'summary.json').open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        _ = stream.write('\n')
    if after_sor is None:
        plot_distribution(before, after, destination / 'distribution.png')
    else:
        plot_populations(
            [
                ('Original', before, '#0072B2'),
                ('After weak SOR', after_sor, '#D55E00'),
                ('After geometry', after, '#009E73'),
            ],
            destination / 'distribution.png',
        )
    return before, after


class Arguments(argparse.Namespace):
    filter_output: Path = Path()
    destination: Path = Path()


def main(argv: list[str] | None = None) -> int:
    """Standalone CPU CLI; matplotlib is needed only when exporting the chart."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument('--filter-output', required=True, type=Path)
    _ = parser.add_argument('--destination', required=True, type=Path)
    args = Arguments()
    _ = parser.parse_args(argv, namespace=args)
    try:
        _ = export_comparison(args.filter_output, args.destination)
    except (PipelineInputError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
