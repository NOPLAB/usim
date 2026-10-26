"""Exact CPU populations for a variable source-origin grid over fixed SOR survivors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .models import PipelineInputError


def _validate_cell_size(cell_size: float) -> None:
    if (
        isinstance(cell_size, (bool, np.bool_))
        or not isinstance(cell_size, (int, float, np.integer, np.floating))
        or not np.isfinite(cell_size)
        or not 0.001 <= cell_size <= 10.0
    ):
        raise PipelineInputError('Cell size must be finite and between 0.001 and 10.0 meters')


def _validate_threshold(threshold: int) -> None:
    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, (int, np.integer))
        or not 0 <= threshold <= 100
    ):
        raise PipelineInputError('Threshold must be an integer between 0 and 100')


@dataclass(frozen=True, slots=True)
class LiveVoxelParameters:
    """Validated live controls; cells with population <= threshold are removed."""

    cell_size: float
    threshold: int

    def __post_init__(self) -> None:
        _validate_cell_size(self.cell_size)
        _validate_threshold(self.threshold)


class LiveVoxelCounter:
    """Cache owned, read-only float64 SOR XYZ and original selected row IDs once.

    Each count call owns its working arrays, so independent worker calls share no
    mutable cache. Retain a returned count array for threshold-only updates via
    select; no regrouping is performed by that method. Geometry decisions are not
    inputs: the population is exclusively the supplied SOR selection.
    """

    def __init__(
        self,
        records: NDArray[np.void] | NDArray[np.float64],
        sor_keep: NDArray[np.bool_],
    ) -> None:
        if records.dtype.names is not None:
            if records.ndim != 1 or not {'x', 'y', 'z'}.issubset(records.dtype.names):
                raise PipelineInputError('Source records must contain scalar x/y/z fields')
            fields = records.dtype.fields
            assert fields is not None
            if any(
                fields[name][0].shape or fields[name][0].kind not in 'biuf'
                for name in ('x', 'y', 'z')
            ):
                raise PipelineInputError('Source records must contain scalar numeric x/y/z')
            points = np.column_stack(
                [records.getfield(fields[name][0], fields[name][1]) for name in ('x', 'y', 'z')]
            )
        else:
            points = records
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or points.dtype.kind not in 'biuf'
            or not np.isfinite(points).all()
        ):
            raise PipelineInputError('XYZ must be a finite numeric (N, 3) array')
        if sor_keep.dtype != np.bool_ or sor_keep.shape != (len(points),):
            raise PipelineInputError('SOR mask must be boolean and match source rows')
        self._row_count: int = len(points)
        self._selected_ids: NDArray[np.int64] = np.flatnonzero(sor_keep).astype(
            np.int64, copy=False
        )
        self._xyz: NDArray[np.float64] = np.array(
            points[self._selected_ids], dtype=np.float64, copy=True
        )
        if not np.isfinite(self._xyz).all():
            raise PipelineInputError('XYZ must be representable as finite float64')
        self._selected_ids.setflags(write=False)
        self._xyz.setflags(write=False)

    def count(self, cell_size: float) -> NDArray[np.int64]:
        """Return original-order post-SOR populations; SOR rejects have count zero.

        Floor float64 XYZ * (1 / size) before translating integer cells solely
        for encoding. Per-axis bit widths make the packed scalar key injective.
        Worlds requiring more than 63 bits or out-of-int64 indices are rejected;
        memory is proportional to source rows, never to world volume.
        """
        _validate_cell_size(cell_size)
        result = np.zeros(self._row_count, dtype=np.int64)
        if not self._selected_ids.size:
            return result
        with np.errstate(over='ignore'):
            points: NDArray[np.float64] = self._xyz * (1.0 / cell_size)
        _ = np.floor(points, out=points)
        if np.any(points < -(2.0**63)) or np.any(points >= 2.0**63):
            raise PipelineInputError('Voxel indices exceed the int64 coordinate range')
        cells: NDArray[np.int64] = points.astype(np.int64)
        del points
        minimum: NDArray[np.int64] = cells.min(axis=0)
        maximum: NDArray[np.int64] = cells.max(axis=0)
        # Python integer subtraction avoids overflow even across signed int64 bounds.
        widths = [(int(maximum[axis]) - int(minimum[axis])).bit_length() for axis in range(3)]
        if sum(widths) > 63:
            raise PipelineInputError('Voxel coordinate spans exceed 63-bit packed key capacity')
        cells -= minimum
        keys = np.zeros(len(cells), dtype=np.int64)
        shift = 0
        for axis, width in enumerate(widths):
            if width:
                keys |= cells[:, axis] << shift
                shift += width
        del cells
        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        result[self._selected_ids] = counts[inverse.reshape(-1)]
        return result

    def select(self, counts: NDArray[np.int64], threshold: int) -> NDArray[np.bool_]:
        """Apply inclusive removal to a previous count result without regrouping."""
        _validate_threshold(threshold)
        if counts.dtype != np.int64 or counts.shape != (self._row_count,):
            raise PipelineInputError('Counts must be int64 and match source rows')
        return counts > threshold
