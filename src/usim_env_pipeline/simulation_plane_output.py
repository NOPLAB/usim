"""Publish closed source-supported residual slabs without inventing plane envelopes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .simulation_output import SimulationTile
from .simulation_planes import PlanarProxy
from .storage import write_json
from .usd import Geometry, LayerMetadata, author_geometry


def publish_planes(
    output: Path, proxies: list[PlanarProxy], metadata: LayerMetadata
) -> list[SimulationTile]:
    from pxr import Sdf, Usd

    folder = output / 'simulation-models/planes'
    folder.mkdir(parents=True, exist_ok=True)
    entries: list[SimulationTile] = []
    rows = []
    for number, proxy in enumerate(proxies):
        name = f'plane_{number:04}'
        path = folder / f'{name}.usdc'
        author_geometry(
            path,
            Geometry(
                proxy.positions,
                np.empty(0, dtype=np.int64),
                np.tile(
                    np.array([[0.64, 0.63, 0.58]], dtype=np.float32), (len(proxy.positions), 1)
                ),
                proxy.faces,
            ),
            metadata,
        )
        np.savez_compressed(
            path.with_suffix('.npz'),
            source_plane=np.asarray(proxy.source_plane),
            support_ids=proxy.support_ids,
            normal=proxy.normal,
            offset=np.asarray(proxy.offset),
            u=proxy.u,
            v=proxy.v,
            keys=proxy.keys,
            positions=proxy.positions,
            faces=proxy.faces,
            source_path=np.asarray(proxy.source_path.relative_to(output).as_posix()),
            observed_keys=proxy.observed_keys,
            completed=np.asarray(proxy.completed),
        )
        stage = Usd.Stage.Open(str(path))
        prim = stage.GetPrimAtPath('/Tile/Surface')
        prim.RemoveProperty('source:recordIds')
        prim.CreateAttribute('simulation:kind', Sdf.ValueTypeNames.String, custom=True).Set(
            'planes'
        )
        prim.CreateAttribute('source:provenance', Sdf.ValueTypeNames.Asset, custom=True).Set(
            Sdf.AssetPath('./' + name + '.npz')
        )
        stage.GetRootLayer().Save()
        entries.append(
            {
                'path': f'/World/SimulationPlanes/{name}',
                'payload': f'./simulation-models/planes/{name}.usdc',
                'kind': 'planes',
                'faces': len(proxy.faces),
            }
        )
        rows.append(
            {'name': name, 'source_plane': proxy.source_plane, 'records': len(proxy.support_ids)}
        )
    write_json(folder / 'index.json', rows)
    return entries
