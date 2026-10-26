"""Point-only, source-ID-preserving views of a completed statistical filter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .cloud import colors
from .models import PipelineInputError
from .pcd import xyz
from .usd import Geometry, LayerMetadata, author_geometry, stage_base


def publish_comparison(output: Path) -> Path:
    """Publish Before, After and gray/red Removed views; return comparison.usda."""
    from pxr import UsdGeom

    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    if report['schema'] != 1 or report['status'] != 'complete':
        raise PipelineInputError('Comparison requires a complete schema-1 filter output')
    records = np.load(output / 'records.npy', mmap_mode='r', allow_pickle=False)
    with np.load(output / 'decisions.npz', allow_pickle=False) as decisions:
        ids = decisions['ids']
        keep = decisions['keep']
        distances = decisions['mean_distances']
        sor_keep = decisions['sor_keep'] if 'sor_keep' in decisions else keep
        geometry_rejected = (
            decisions['geometry_rejected']
            if 'geometry_rejected' in decisions
            else np.zeros(len(records), dtype=np.bool_)
        )
    if (
        ids.dtype != np.int64
        or ids.shape != (len(records),)
        or not np.array_equal(ids, np.arange(len(records), dtype=np.int64))
        or keep.dtype != np.bool_
        or keep.shape != ids.shape
        or distances.dtype != np.float64
        or distances.shape != ids.shape
        or report['records'] != len(records)
        or report['retained_records'] != int(keep.sum())
        or report['removed_records'] != int((~keep).sum())
        or sor_keep.dtype != np.bool_
        or sor_keep.shape != keep.shape
        or geometry_rejected.dtype != np.bool_
        or geometry_rejected.shape != keep.shape
        or np.any(geometry_rejected & ~sor_keep)
        or not np.array_equal(keep, sor_keep & ~geometry_rejected)
    ):
        raise PipelineInputError('Filter records, counts and source-order decisions disagree')
    source = report['source']
    metadata = LayerMetadata(
        source['license'],
        source['attribution'],
        source['official_url'],
        report['frame'],
        report['color'],
    )
    positions = xyz(records)
    source_colors = colors(records)
    # Stable sorting groups tiles without changing the source IDs or order within a tile.
    keys = np.floor(positions[:, :2].astype(np.float64) / 30.0).astype(np.int64)
    order = np.lexsort((ids, keys[:, 1], keys[:, 0]))
    ordered_keys = keys[order]
    boundaries = np.flatnonzero(np.any(np.diff(ordered_keys, axis=0), axis=1)) + 1
    tiles = np.split(order, boundaries) if len(order) else []
    comparison_path = output / 'comparison.usda'
    comparison = stage_base(comparison_path, '/World', metadata)
    stages = []
    for mode in ('Before', 'After', 'Removed'):
        folder = output / 'comparison-points' / mode.lower()
        folder.mkdir(parents=True, exist_ok=True)
        view = stage_base(output / f'{mode.lower()}.usda', '/World', metadata)
        root = f'/World/{mode}'
        UsdGeom.Xform.Define(view, root)
        for tile_number, tile_ids in enumerate(tiles):
            selected = tile_ids[keep[tile_ids]] if mode == 'After' else tile_ids
            if not len(selected):
                continue
            display = source_colors[selected]
            if mode == 'Removed':
                display = np.full((len(selected), 3), 0.45, dtype=np.float32)
                display[~sor_keep[selected]] = (1.0, 0.0, 0.0)
                display[geometry_rejected[selected]] = (1.0, 1.0, 0.0)
            name = f't_{tile_number}'
            payload = folder / f'{name}.usdc'
            author_geometry(
                payload, Geometry(positions[selected], ids[selected], display), metadata
            )
            prim = UsdGeom.Xform.Define(view, f'{root}/{name}').GetPrim()
            prim.GetPayloads().AddPayload(
                f'./comparison-points/{mode.lower()}/{name}.usdc', '/Tile'
            )
        group = UsdGeom.Xform.Define(comparison, root)
        group.GetPrim().GetReferences().AddReference(f'./{mode.lower()}.usda', root)
        group.CreateVisibilityAttr(
            UsdGeom.Tokens.inherited if mode == 'After' else UsdGeom.Tokens.invisible
        )
        stages.append(view)
    for stage in [*stages, comparison]:
        data = dict(stage.GetRootLayer().customLayerData)
        data['sourceMetadata'] = json.dumps(source, sort_keys=True)
        data['filterReport'] = json.dumps(report, sort_keys=True)
        stage.GetRootLayer().customLayerData = data
        stage.GetRootLayer().Save()
    return comparison_path
