"""Publish weak-SOR and conservative geometric decisions in original source order."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .geometric_filter import DEFAULT_GEOMETRY, GeometricSelection, geometric_outliers
from .models import PipelineInputError
from .point_filter_retune import retune_filter
from .storage import write_json

WEAK_SOR_RATIO: Final = 4.0
CANDIDATE_RATIO: Final = 1.0


def publish_geometric_filter(previous: Path, output: Path) -> Path:
    """Preserve every source field and publish mutually exclusive red/yellow labels."""
    report = retune_filter(previous, output, WEAK_SOR_RATIO)
    records: NDArray[np.void] = np.load(output / 'records.npy', mmap_mode='r', allow_pickle=False)
    with np.load(output / 'decisions.npz', allow_pickle=False) as saved:
        ids: NDArray[np.int64] = saved['ids']
        sor_keep: NDArray[np.bool_] = saved['keep']
        means: NDArray[np.float64] = saved['mean_distances']
    candidate_threshold = report['global_mean_m'] + CANDIDATE_RATIO * report['global_std_m']
    candidates = sor_keep & (means > candidate_threshold)
    positions = np.column_stack([records[name] for name in ('x', 'y', 'z')]).astype(np.float64)
    print(f'GEOMETRY_QUERY candidates={int(candidates.sum())}', flush=True)
    result = geometric_outliers(positions, GeometricSelection(sor_keep, candidates))
    if (
        result.keep.shape != sor_keep.shape
        or np.any(result.geometry_rejected & ~sor_keep)
        or not np.array_equal(result.keep, sor_keep & ~result.geometry_rejected)
    ):
        raise PipelineInputError('Geometric decisions must partition the weak-SOR selection')
    np.savez_compressed(
        output / 'decisions.npz',
        ids=ids,
        keep=result.keep,
        sor_keep=sor_keep,
        geometry_rejected=result.geometry_rejected,
        mean_distances=means,
        evaluated=result.evaluated,
        supported_scales=result.supported_scales,
        consistent_scales=result.consistent_scales,
    )
    retained = int(result.keep.sum())
    report['retained_records'] = retained
    report['removed_records'] = len(records) - retained
    report['removed_fraction'] = (len(records) - retained) / len(records)
    report['execution'] = 'weak SOR then multiscale local plane/line consistency; CPU; no meshes'
    combined = {
        **report,
        'geometry': {
            'config': asdict(DEFAULT_GEOMETRY),
            'candidate_std_ratio': CANDIDATE_RATIO,
            'candidate_distance_threshold_m': candidate_threshold,
            'evaluated_records': int(result.evaluated.sum()),
            'sor_retained_records': int(sor_keep.sum()),
            'sor_removed_records': int((~sor_keep).sum()),
            'geometry_removed_records': int(result.geometry_rejected.sum()),
            'retained_records': retained,
            'colors': {'sor': [1, 0, 0], 'geometry': [1, 1, 0], 'retained': [0.45] * 3},
            'policy': 'keep any consistent scale or insufficient evidence; no density-only removal',
        },
    }
    write_json(output / 'report.json', combined)
    print(
        f'GEOMETRY_FILTER_PASSED sor_removed={int((~sor_keep).sum())} '
        f'geometry_removed={int(result.geometry_rejected.sum())} retained={retained}',
        flush=True,
    )
    from .point_filter_usd import publish_comparison

    return publish_comparison(output)


class Arguments(argparse.Namespace):
    previous: Path = Path()
    output: Path = Path()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument('--previous', required=True, type=Path)
    _ = parser.add_argument('--output', required=True, type=Path)
    args = Arguments()
    _ = parser.parse_args(namespace=args)
    _ = publish_geometric_filter(args.previous, args.output)
    print('GEOMETRY_COMPARISON_READY', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
