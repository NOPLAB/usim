"""Shared original geometry with five independently colored removal overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError
from .pcd import xyz
from .usd import Geometry, LayerMetadata, author_geometry, stage_base


def publish_threshold_comparison(output: Path) -> Path:
    """Share source positions/IDs between modes; vary only the removal colors."""
    from pxr import UsdGeom, Vt

    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    records: NDArray[np.void] = np.load(output / 'records.npy', mmap_mode='r', allow_pickle=False)
    with np.load(output / 'decisions.npz', allow_pickle=False) as saved:
        ids: NDArray[np.int64] = saved['ids']
        sor_keep: NDArray[np.bool_] = saved['sor_keep']
        counts: NDArray[np.int64] = saved['counts_per_record']
    if (
        ids.dtype != np.int64
        or not np.array_equal(ids, np.arange(len(records)))
        or sor_keep.dtype != np.bool_
        or sor_keep.shape != ids.shape
        or counts.shape != ids.shape
        or counts.dtype != np.int64
        or np.any(counts[~sor_keep] != 0)
        or np.any(counts[sor_keep] < 1)
    ):
        raise PipelineInputError('Threshold comparison requires original-order SOR voxel counts')
    source = report['source']
    metadata = LayerMetadata(
        source['license'],
        source['attribution'],
        source['official_url'],
        report['frame'],
        'retained gray; SOR rejected red; independent threshold rejected condition color',
    )
    positions = xyz(records)
    keys = np.floor(positions[:, :2].astype(np.float64) / 30).astype(np.int64)
    order = np.lexsort((ids, keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(np.any(np.diff(sorted_keys, axis=0), axis=1)) + 1
    tiles = np.split(order, boundaries) if len(order) else []
    base = output / 'comparison-points/base'
    base.mkdir(parents=True, exist_ok=True)
    for number, tile in enumerate(tiles):
        author_geometry(
            base / f't_{number}.usdc',
            Geometry(positions[tile], ids[tile], np.full((len(tile), 3), 0.45, dtype=np.float32)),
            metadata,
        )
    comparison_path = output / 'comparison.usda'
    comparison = stage_base(comparison_path, '/World', metadata)
    for mode in report['modes']:
        name, threshold = mode['name'], mode['threshold']
        keep = sor_keep & (counts > threshold)
        if int(keep.sum()) != mode['retained_records']:
            raise PipelineInputError(f'Threshold count mismatch: {name}')
        view = stage_base(output / f'{name.lower()}.usda', '/World', metadata)
        root = f'/World/{name}'
        UsdGeom.Xform.Define(view, root)
        for number, tile in enumerate(tiles):
            path = f'{root}/t_{number}'
            prim = UsdGeom.Xform.Define(view, path).GetPrim()
            prim.GetPayloads().AddPayload(f'./comparison-points/base/t_{number}.usdc', '/Tile')
            display = np.full((len(tile), 3), 0.45, dtype=np.float32)
            display[~sor_keep[tile]] = (1, 0, 0)
            display[sor_keep[tile] & ~keep[tile]] = mode['color']
            points = UsdGeom.Points.Define(view, f'{path}/Points')
            points.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(
                Vt.Vec3fArray.FromNumpy(display)
            )
        view.GetRootLayer().Save()
        group = UsdGeom.Xform.Define(comparison, root)
        group.GetPrim().GetReferences().AddReference(f'./{name.lower()}.usda', root)
        group.CreateVisibilityAttr(
            UsdGeom.Tokens.inherited if name == 'SOR' else UsdGeom.Tokens.invisible
        )
    comparison.GetRootLayer().Save()
    print('VOXEL_THRESHOLD_COMPARISON_READY', flush=True)
    return comparison_path


class Arguments(argparse.Namespace):
    output: Path = Path()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument('--output', type=Path, required=True)
    args = Arguments()
    _ = parser.parse_args(namespace=args)
    _ = publish_threshold_comparison(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
