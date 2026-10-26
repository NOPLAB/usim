"""Reuse exact global neighbor distances to adjust only the SOR multiplier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError
from .point_filter import SORConfig
from .point_filter_output import FilterReport
from .storage import write_json


def retune_filter(previous: Path, output: Path, std_ratio: float) -> FilterReport:
    """Preserve source records and distances in a new, non-overwriting selection."""
    config = SORConfig(std_ratio=std_ratio)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise PipelineInputError('SOR output must be absent or empty; use a new directory')
    report: FilterReport = json.loads((previous / 'report.json').read_text(encoding='utf-8'))
    if report['schema'] != 1 or report['status'] != 'complete':
        raise PipelineInputError('Retuning requires a complete schema-1 filter output')
    records: NDArray[np.void] = np.load(
        previous / 'records.npy', mmap_mode='r', allow_pickle=False
    )
    with np.load(previous / 'decisions.npz', allow_pickle=False) as decisions:
        ids: NDArray[np.int64] = decisions['ids']
        means: NDArray[np.float64] = decisions['mean_distances']
    if (
        ids.dtype != np.int64
        or not np.array_equal(ids, np.arange(len(records), dtype=np.int64))
        or means.dtype != np.float64
        or means.shape != ids.shape
        or not np.isfinite(means).all()
        or report['records'] != len(records)
        or len(means) < 2
    ):
        raise PipelineInputError('Retuning requires finite distances for every source row')
    mean = float(means.mean())
    deviation = float(means.std(ddof=1))
    if mean != report['global_mean_m'] or deviation != report['global_std_m']:
        raise PipelineInputError('Saved distance statistics disagree with the filter report')
    threshold = mean + config.std_ratio * deviation
    keep = means <= threshold
    output.mkdir(parents=True, exist_ok=True)
    _ = shutil.copytree(previous / 'source', output / 'source')
    _ = shutil.copy2(previous / 'records.npy', output / 'records.npy')
    np.savez_compressed(output / 'decisions.npz', ids=ids, keep=keep, mean_distances=means)
    retained = int(np.count_nonzero(keep))
    report['config']['std_ratio'] = config.std_ratio
    report['retained_records'] = retained
    report['removed_records'] = len(records) - retained
    report['removed_fraction'] = (len(records) - retained) / len(records)
    report['threshold_m'] = threshold
    report['execution'] = (
        f'reused exact global CPU kNN distances from {previous.resolve()}; '
        'only multiplier changed; no iterative filtering or reconstruction'
    )
    write_json(output / 'report.json', report)
    print(f'SOR_RETUNED retained={retained} removed={len(records) - retained}', flush=True)
    return report


class Arguments(argparse.Namespace):
    """CLI paths and multiplier populated by required arguments."""

    previous: Path = Path()
    output: Path = Path()
    std_ratio: float = 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument('--previous', required=True, type=Path)
    _ = parser.add_argument('--output', required=True, type=Path)
    _ = parser.add_argument('--std-ratio', required=True, type=float)
    args = Arguments()
    _ = parser.parse_args(namespace=args)
    _ = retune_filter(args.previous, args.output, args.std_ratio)
    from .point_filter_usd import publish_comparison

    _ = publish_comparison(args.output)
    print('SOR_COMPARISON_READY', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
