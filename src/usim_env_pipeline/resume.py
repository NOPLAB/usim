"""Validate immutable inputs/settings before reusing completed regional checkpoints."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .models import MapConfig, PipelineInputError, RegionReport, Source
from .pcd import sha256
from .regions import RegionJob
from .storage import tile_name


def validate(output: Path, source: Source, config: MapConfig) -> None:
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    expected_support = json.loads(json.dumps(asdict(config.support)))
    if (
        manifest['tile_size'] != config.tile_size
        or manifest['halo'] != config.halo
        or manifest['support'] != expected_support
        or manifest['normal_policy'] != config.normal_policy
        or manifest['frame'] != config.frame
        or manifest['primary_source'] != source.key
        or set(manifest['sources']) != {s.key for s in (source, *config.references)}
    ):
        raise PipelineInputError('Resume source keys/frame/geometry settings differ')
    for item in (source, *config.references):
        stored = manifest['sources'][item.key]
        digest = sha256(item.path)
        if (
            digest != stored['sha256']
            or sha256(output / stored['path']) != digest
            or (item.expected_sha256 is not None and digest != item.expected_sha256)
            or stored['license'] != item.license
            or stored['attribution'] != item.attribution
            or stored['official_url'] != item.url
        ):
            raise PipelineInputError('Resume original bytes/provenance differ')


def completed(job: RegionJob) -> RegionReport | None:
    name = tile_name(job.key)
    path = job.output / 'reports' / f'{name}.json'
    if not path.exists():
        # An interrupted region has no completion report; regenerate only its own files.
        for folder, suffix in (('points', '.usdc'), ('meshes', '.usdc'), ('meshes', '.npz')):
            job.output.joinpath(folder, name + suffix).unlink(missing_ok=True)
        return None
    report: RegionReport = json.loads(path.read_text(encoding='utf-8'))
    with np.load(job.output / 'records' / f'{name}.npz') as own:
        if len(own['ids']) != report['records'] or report['key'] != list(job.key):
            raise PipelineInputError('Regional record checkpoint differs')
    with np.load(job.output / 'meshes' / f'{name}.npz') as mesh:
        if (
            len(mesh['ids']) != report['mesh_vertices']
            or len(mesh['faces']) != report['triangles']
        ):
            raise PipelineInputError('Regional mesh checkpoint differs')
    if not job.output.joinpath('points', name + '.usdc').is_file() or (
        report['triangles'] and not job.output.joinpath('meshes', name + '.usdc').is_file()
    ):
        raise PipelineInputError('Completed regional USD checkpoint is missing')
    return report
