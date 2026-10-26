"""Compare independent inclusive voxel thresholds on the same CPU SOR population."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError
from .voxel_statistics import CountSummary, VOXEL_SIZE_M, VoxelStatistics, plot_populations

THRESHOLDS: Final = (5, 10, 20, 40)
MODE_COLORS: Final = (
    (1.0, 0.0, 0.0),
    (1.0, 0.5, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.4, 1.0),
)


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    """One threshold's original-order row selection and occupied-cell statistics."""

    threshold: int
    keep: NDArray[np.bool_]
    stats: VoxelStatistics


@dataclass(frozen=True, slots=True)
class ThresholdAnalysis:
    """Post-SOR cell populations mapped back to every original source row."""

    counts_per_record: NDArray[np.int64]
    sor_stats: VoxelStatistics
    selections: tuple[ThresholdSelection, ...]


def _statistics(indices: NDArray[np.int64], counts: NDArray[np.int64]) -> VoxelStatistics:
    point_counts, frequencies = np.unique(counts, return_counts=True)
    summary: CountSummary = {
        'min': int(counts.min()) if counts.size else None,
        'max': int(counts.max()) if counts.size else None,
        'mean': float(counts.mean()) if counts.size else None,
        'median': float(np.median(counts)) if counts.size else None,
        'occupied_voxels': len(counts),
        'total_points': int(counts.sum()),
    }
    return VoxelStatistics(indices, counts, point_counts, frequencies, summary)


def analyze_thresholds(
    records: NDArray[np.void], sor_keep: NDArray[np.bool_]
) -> ThresholdAnalysis:
    """Group SOR survivors once; remove whole cells with population <= each threshold.

    Coordinates follow occupied_voxels exactly: source origin zero, float64 XYZ * 10,
    floor without snapping. Duplicate records count separately, and no caller array changes.
    """
    if sor_keep.dtype != np.bool_ or sor_keep.shape != (len(records),):
        raise PipelineInputError('SOR mask must be boolean and match source rows')
    if (
        records.ndim != 1
        or records.dtype.names is None
        or not {'x', 'y', 'z'}.issubset(records.dtype.names)
    ):
        raise PipelineInputError('Source records must contain scalar x/y/z fields')
    fields = records.dtype.fields
    assert fields is not None
    points = np.column_stack(
        [records.getfield(fields[name][0], fields[name][1]) for name in ('x', 'y', 'z')]
    )
    points = np.array(points, dtype=np.float64, copy=True)
    if points.shape != (len(records), 3) or not np.isfinite(points).all():
        raise PipelineInputError('XYZ must be a finite (N, 3) array')
    points = points[sor_keep]
    points *= 1.0 / VOXEL_SIZE_M
    _ = np.floor(points, out=points)
    if np.any(points < -(2.0**63)) or np.any(points >= 2.0**63):
        raise PipelineInputError('Voxel indices exceed the int64 coordinate range')
    cells = points.astype(np.int64)
    del points
    indices, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    del cells
    counts = counts.astype(np.int64, copy=False)
    counts_per_record = np.zeros(len(records), dtype=np.int64)
    counts_per_record[sor_keep] = counts[inverse.reshape(-1)]
    sor_stats = _statistics(indices, counts)
    selections = tuple(
        ThresholdSelection(
            threshold,
            counts_per_record > threshold,
            _statistics(indices[counts > threshold], counts[counts > threshold]),
        )
        for threshold in THRESHOLDS
    )
    return ThresholdAnalysis(counts_per_record, sor_stats, selections)


def publish_thresholds(source_output: Path, output: Path) -> Path:
    """Copy the original source archive and publish five geometry-free conditions."""
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise PipelineInputError('Threshold output must be absent or empty; use a new directory')
    source_report = json.loads((source_output / 'report.json').read_text(encoding='utf-8'))
    records: NDArray[np.void] = np.load(
        source_output / 'records.npy', mmap_mode='r', allow_pickle=False
    )
    with np.load(source_output / 'decisions.npz', allow_pickle=False) as decisions:
        ids: NDArray[np.int64] = decisions['ids']
        sor_keep: NDArray[np.bool_] = (
            decisions['sor_keep'] if 'sor_keep' in decisions else decisions['keep']
        )
    if ids.dtype != np.int64 or not np.array_equal(ids, np.arange(len(records))):
        raise PipelineInputError('SOR decisions must follow original source-row IDs')
    analysis = analyze_thresholds(records, sor_keep)
    output.mkdir(parents=True, exist_ok=True)
    _ = shutil.copytree(source_output / 'source', output / 'source')
    _ = shutil.copy2(source_output / 'records.npy', output / 'records.npy')
    np.savez_compressed(
        output / 'decisions.npz',
        ids=ids,
        sor_keep=sor_keep,
        counts_per_record=analysis.counts_per_record,
        **{f'keep_{item.threshold}': item.keep for item in analysis.selections},
    )
    sor_retained = analysis.sor_stats.summary['total_points']
    populations = [('SOR', 0, analysis.sor_stats)] + [
        (f'Voxel{item.threshold}', item.threshold, item.stats) for item in analysis.selections
    ]
    modes = []
    for (name, threshold, stats), color in zip(populations, MODE_COLORS):
        modes.append(
            {
                'name': name,
                'threshold': threshold,
                'retained_records': stats.summary['total_points'],
                'voxel_removed_records': sor_retained - stats.summary['total_points'],
                **{
                    key: stats.summary[key]
                    for key in ('occupied_voxels', 'min', 'max', 'mean', 'median')
                },
                'color': list(color),
            }
        )
        with (output / f'{name}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('points_per_voxel', 'occupied_voxels'))
            writer.writerows(zip(stats.point_counts.tolist(), stats.frequencies.tolist()))
    report = {
        'schema': 1,
        'source': source_report['source'],
        'frame': source_report['frame'],
        'color': source_report['color'],
        'source_config': source_report['config'],
        'records': len(records),
        'sor_removed_records': len(records) - sor_retained,
        'modes': modes,
    }
    report_path = output / 'report.json'
    _ = report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )
    plot_populations(
        [
            (name, stats, '#%02x%02x%02x' % tuple(round(channel * 255) for channel in color))
            for (name, _, stats), color in zip(populations, MODE_COLORS)
        ],
        output / 'distribution.png',
    )
    return report_path


class Arguments(argparse.Namespace):
    source_output: Path = Path()
    output: Path = Path()


def main(argv: list[str] | None = None) -> int:
    """Standalone CPU CLI; plotting uses the existing five-curve distribution format."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument('--source-output', required=True, type=Path)
    _ = parser.add_argument('--output', required=True, type=Path)
    args = Arguments()
    _ = parser.parse_args(argv, namespace=args)
    try:
        _ = publish_thresholds(args.source_output, args.output)
    except (PipelineInputError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
