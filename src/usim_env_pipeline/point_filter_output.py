"""Archive a source and publish SOR decisions without invoking downstream geometry."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict

import numpy as np

from .models import PipelineInputError, Source, SourceMetadata
from .point_filter import DEFAULT_SOR, SORConfig, statistical_outliers
from .storage import archive_source, write_json


class FilterParameters(TypedDict):
    neighbors: int
    std_ratio: float
    batch_size: int
    workers: int


class FilterReport(TypedDict):
    schema: int
    status: str
    source: SourceMetadata
    input_path: str
    frame: str
    color: str
    config: FilterParameters
    records: int
    retained_records: int
    removed_records: int
    removed_fraction: float
    global_mean_m: float
    global_std_m: float
    threshold_m: float
    distance_quantile_levels: list[float]
    distance_quantiles_m: list[float]
    distance_histogram_edges_m: list[float]
    distance_histogram_counts: list[int]
    neighbor_policy: str
    execution: str


def filter_source(source: Source, output: Path, config: SORConfig = DEFAULT_SOR) -> FilterReport:
    """Keep immutable source bytes and all fields; publish only a working-point selection."""
    root = output.resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise PipelineInputError('SOR output must be absent or empty; use a new directory')
    (root / 'source').mkdir(parents=True, exist_ok=True)
    print(f'SOR_READING source={source.path}', flush=True)
    cloud = archive_source(source, root)
    records = cloud.records
    np.save(root / 'records.npy', records, allow_pickle=False)
    positions = np.column_stack([records[name] for name in ('x', 'y', 'z')]).astype(np.float64)
    print(f'SOR_QUERY records={len(records)} neighbors={config.neighbors}', flush=True)
    result = statistical_outliers(positions, config)
    ids = np.arange(len(records), dtype=np.int64)
    np.savez_compressed(
        root / 'decisions.npz', ids=ids, keep=result.keep, mean_distances=result.mean_distances
    )
    retained = int(np.count_nonzero(result.keep))
    quantiles = [0.0, 0.01, 0.5, 0.9, 0.95, 0.99, 1.0]
    counts, edges = np.histogram(result.mean_distances, bins=64)
    report: FilterReport = {
        'schema': 1,
        'status': 'complete',
        'source': cloud.metadata,
        'input_path': str(source.path.resolve()),
        'frame': 'source XYZ; meters; Z-up',
        'color': cloud.color_description,
        'config': {
            'neighbors': config.neighbors,
            'std_ratio': config.std_ratio,
            'batch_size': config.batch_size,
            'workers': config.workers,
        },
        'records': len(records),
        'retained_records': retained,
        'removed_records': len(records) - retained,
        'removed_fraction': (len(records) - retained) / len(records),
        'global_mean_m': result.global_mean,
        'global_std_m': result.global_std,
        'threshold_m': result.threshold,
        'distance_quantile_levels': quantiles,
        'distance_quantiles_m': np.quantile(result.mean_distances, quantiles).tolist(),
        'distance_histogram_edges_m': edges.tolist(),
        'distance_histogram_counts': counts.tolist(),
        'neighbor_policy': 'k other records; coincident observations counted separately',
        'execution': 'single pass; exact global CPU kNN; no normals, meshes or reconstruction',
    }
    write_json(root / 'report.json', report)
    print(f'SOR_FILTER_PASSED retained={retained} removed={len(records) - retained}', flush=True)
    return report


class Arguments(argparse.Namespace):
    """Typed argparse namespace; required path flags must be supplied before use."""

    source: Path = Path()
    output: Path = Path()
    neighbors: int = DEFAULT_SOR.neighbors
    std_ratio: float = DEFAULT_SOR.std_ratio
    batch_size: int = DEFAULT_SOR.batch_size
    workers: int = DEFAULT_SOR.workers
    license: str = 'unspecified'
    attribution: str = ''
    source_url: str = ''
    source_catalog: str = ''
    sha256: str | None = None
    comparison: bool = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument('--source', required=True, type=Path)
    _ = parser.add_argument('--output', required=True, type=Path)
    _ = parser.add_argument('--neighbors', type=int, default=20)
    _ = parser.add_argument('--std-ratio', type=float, default=3.0)
    _ = parser.add_argument('--batch-size', type=int, default=32768)
    _ = parser.add_argument('--workers', type=int, default=4)
    _ = parser.add_argument('--license', default='unspecified')
    _ = parser.add_argument('--attribution', default='')
    _ = parser.add_argument('--source-url', default='')
    _ = parser.add_argument('--source-catalog', default='')
    _ = parser.add_argument('--sha256')
    _ = parser.add_argument('--comparison', action='store_true')
    args = Arguments()
    _ = parser.parse_args(argv, namespace=args)
    try:
        _ = filter_source(
            Source(
                args.source,
                license=args.license,
                attribution=args.attribution,
                url=args.source_url,
                catalog=args.source_catalog,
                expected_sha256=args.sha256,
            ),
            args.output,
            SORConfig(args.neighbors, args.std_ratio, args.batch_size, args.workers),
        )
        if args.comparison:
            from .point_filter_usd import publish_comparison

            _ = publish_comparison(args.output)
    except PipelineInputError as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
