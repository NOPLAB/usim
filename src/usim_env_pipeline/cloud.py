"""Decode point clouds without filtering records or rewriting the source file."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import assert_never

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError, SourceMetadata
from .pcd import open_records, xyz


@dataclass(frozen=True, slots=True)
class Cloud:
    records: NDArray[np.void]
    metadata: SourceMetadata
    color_description: str


class CloudFormat(Enum):
    PCD = '.pcd'
    PLY = '.ply'


def colors(records: NDArray[np.void]) -> NDArray[np.float32]:
    names = records.dtype.names or ()
    if all(f in names for f in ('r', 'g', 'b')):
        rgb = np.column_stack([records[f] for f in ('r', 'g', 'b')]).astype(np.float32)
        if records.dtype['r'].kind in 'iu':
            rgb /= 255
    elif 'rgb' in names or 'rgba' in names:
        field = 'rgb' if 'rgb' in names else 'rgba'
        packed = np.ascontiguousarray(records[field])
        if packed.dtype.itemsize != 4:
            raise PipelineInputError('Packed PCD RGB must be a 32-bit scalar')
        bits = packed.view(np.uint32)
        rgb = np.column_stack([(bits >> shift) & 255 for shift in (16, 8, 0)])
        rgb = rgb.astype(np.float32) / 255
    elif 'intensity' in names:
        rgb = np.repeat((records['intensity'].astype(np.float32) / 255)[:, None], 3, axis=1)
    else:
        rgb = np.full((len(records), 3), 0.65, dtype=np.float32)
    if not np.isfinite(rgb).all():
        raise PipelineInputError('Nonfinite display color/intensity')
    return rgb


def read_cloud(path: Path) -> Cloud:
    """Accept PCD and PLY; source files remain the authoritative lossless archive."""
    metadata: SourceMetadata
    try:
        format = CloudFormat(path.suffix.lower())
    except ValueError as error:
        raise PipelineInputError('Input must be a .pcd or .ply point cloud') from error
    match format:
        case CloudFormat.PCD:
            records, metadata = open_records(path)
        case CloudFormat.PLY:
            import open3d as o3d

            cloud = o3d.io.read_point_cloud(
                str(path), remove_nan_points=False, remove_infinite_points=False
            )
            points = np.asarray(cloud.points)
            fields = [('x', '<f8'), ('y', '<f8'), ('z', '<f8')]
            if cloud.has_normals():
                fields += [(f, '<f8') for f in ('normal_x', 'normal_y', 'normal_z')]
            if cloud.has_colors():
                fields += [(f, '<f8') for f in ('r', 'g', 'b')]
            records = np.empty(len(points), dtype=np.dtype(fields))
            for i, f in enumerate(('x', 'y', 'z')):
                records[f] = points[:, i]
            if cloud.has_normals():
                for i, f in enumerate(('normal_x', 'normal_y', 'normal_z')):
                    records[f] = np.asarray(cloud.normals)[:, i]
            if cloud.has_colors():
                for i, f in enumerate(('r', 'g', 'b')):
                    records[f] = np.asarray(cloud.colors)[:, i]
            metadata = {
                'points': len(records),
                'bytes': path.stat().st_size,
                'fields': list(records.dtype.names or ()),
                'dtype': str(records.dtype.descr),
            }
        case unreachable:
            assert_never(unreachable)
    if not len(records) or not all(f in (records.dtype.names or ()) for f in ('x', 'y', 'z')):
        raise PipelineInputError('Require a nonempty cloud with scalar XYZ fields')
    for f in ('x', 'y', 'z'):
        if records.dtype[f].shape:
            raise PipelineInputError('XYZ fields must be scalar')
    for f in ('normal_x', 'normal_y', 'normal_z', 'r', 'g', 'b', 'rgb', 'rgba', 'intensity'):
        if f in (records.dtype.names or ()) and records.dtype[f].shape:
            raise PipelineInputError(f'Normal/color field must be scalar: {f}')
    if not np.isfinite(xyz(records)).all():
        raise PipelineInputError('Nonfinite/out-of-float32-range XYZ is not supported')
    names = records.dtype.names or ()
    description = 'constant gray; no inferred photographic color'
    if all(f in names for f in ('r', 'g', 'b')) or 'rgb' in names or 'rgba' in names:
        description = 'source RGB'
    elif 'intensity' in names:
        description = 'source intensity grayscale, not RGB'
    colors(records)
    return Cloud(records, metadata, description)
