"""Event-driven regressions for stale grid work and cached threshold changes."""

import anyio
from anyio import from_thread
import numpy as np

from usim_env_pipeline.live_voxel_counts import LiveVoxelParameters
from usim_env_pipeline.live_voxel_updates import LatestVoxelUpdates


def test_latest_grid_wins_and_threshold_does_not_recount():
    # Given a real worker blocked on an exact event, not a timing delay.
    async def scenario():
        entered = anyio.Event()
        release = anyio.Event()
        published = anyio.Event()
        calls = []
        updates = []

        class Counter:
            def count(self, cell_size: float):
                calls.append(cell_size)
                if cell_size == 0.1:
                    from_thread.run_sync(entered.set)
                    from_thread.run(release.wait)
                return np.array([0, 6, 20], dtype=np.int64)

        async def publish(update):
            updates.append(update)
            published.set()

        controller = LatestVoxelUpdates(Counter(), publish)
        async with anyio.create_task_group() as group:
            group.start_soon(controller.run)
            controller.request(LiveVoxelParameters(0.1, 5))
            await entered.wait()
            # When newer grid and threshold requests arrive during the earlier work.
            controller.request(LiveVoxelParameters(0.2, 10))
            controller.request(LiveVoxelParameters(0.2, 20))
            release.set()
            await published.wait()
            # Then the old grid never publishes, and latest threshold is selected.
            assert calls == [0.1, 0.2]
            assert len(updates) == 1
            assert updates[0].parameters == LiveVoxelParameters(0.2, 20)
            published = anyio.Event()
            controller.request(LiveVoxelParameters(0.2, 0))
            await published.wait()
            assert calls == [0.1, 0.2]
            assert updates[-1].cached
            assert updates[-1].parameters.threshold == 0
            group.cancel_scope.cancel()

    async def bounded():
        with anyio.fail_after(10):
            await scenario()

    anyio.run(bounded)
