"""One CPU worker, cached grid counts, and latest-request-only publication."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import anyio
from anyio import to_thread
import numpy as np
from numpy.typing import NDArray

from .live_voxel_counts import LiveVoxelParameters


class CountSource(Protocol):
    def count(self, cell_size: float) -> NDArray[np.int64]: ...


@dataclass(frozen=True, slots=True)
class LiveVoxelUpdate:
    revision: int
    parameters: LiveVoxelParameters
    counts: NDArray[np.int64]
    count_seconds: float
    cached: bool


class LatestVoxelUpdates:
    """Mutable request mailbox; callbacks and publication stay on the event-loop thread."""

    def __init__(
        self, counter: CountSource, publish: Callable[[LiveVoxelUpdate], Awaitable[None]]
    ) -> None:
        self.counter = counter
        self.publish = publish
        self.revision = 0
        self.requested: LiveVoxelParameters | None = None
        self.wake = anyio.Event()

    def request(self, parameters: LiveVoxelParameters) -> int:
        self.revision += 1
        self.requested = parameters
        self.wake.set()
        return self.revision

    def is_current(self, revision: int) -> bool:
        return revision == self.revision

    async def run(self) -> None:
        """Reuse counts for threshold changes and discard stale grid results."""
        size: float | None = None
        counts: NDArray[np.int64] | None = None
        published_revision = 0
        while True:
            await self.wake.wait()
            self.wake = anyio.Event()
            if self.revision == published_revision:
                continue
            requested = self.requested
            assert requested is not None
            cached = size == requested.cell_size
            started = perf_counter()
            if not cached:
                counts = await to_thread.run_sync(self.counter.count, requested.cell_size)
                size = requested.cell_size
            elapsed = perf_counter() - started
            latest = self.requested
            assert latest is not None and counts is not None
            if latest.cell_size != size:
                continue
            update = LiveVoxelUpdate(self.revision, latest, counts, elapsed, cached)
            await self.publish(update)
            published_revision = update.revision
