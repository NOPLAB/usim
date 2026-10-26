"""Bound full-detail static world payload residency around a moving robot."""

from __future__ import annotations

import json
import math
from pathlib import Path


class WorldResidency:
    """Index bounds are metre-scale world coordinates; only the prim root is remapped."""

    def __init__(self, stage, index: Path, environment_prim_path: str, radius: float):
        entries = json.loads(index.read_text(encoding='utf-8'))
        self.entries = []
        for entry in entries:
            path = entry['path']
            if not path.startswith('/World/'):
                raise ValueError(f'world index payload must be below /World: {path}')
            low, high = entry['min'], entry['max']
            if (
                len(low) != 3
                or len(high) != 3
                or not all(math.isfinite(value) for value in (*low, *high))
                or any(a > b for a, b in zip(low, high))
            ):
                raise ValueError(f'invalid world index bounds: {path}')
            self.entries.append((environment_prim_path + path[len('/World') :], low, high))
        self.stage = stage
        self.radius = radius
        self.threshold = min(5.0, radius / 6)
        self.resident: set[str] = set()
        self.center: tuple[float, float] | None = None
        # The supplied wrapper already deactivates these payloads before reference
        # composition. Session edits never alter that wrapper or original assets.
        for path, _, _ in self.entries:
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError(f'world index payload is missing from stage: {path}')
            prim.SetActive(False)

    def update(self, position) -> bool:
        """Change native geometry between physics steps, with spatial hysteresis."""
        point = (float(position[0]), float(position[1]))
        if self.center is not None and math.dist(point, self.center) < self.threshold:
            return False
        desired = set()
        for path, low, high in self.entries:
            radius = self.radius + (2 * self.threshold if path in self.resident else 0)
            if all(
                low[axis] <= point[axis] + radius and high[axis] >= point[axis] - radius
                for axis in (0, 1)
            ):
                desired.add(path)
        removed, added = self.resident - desired, desired - self.resident
        for path in sorted(removed):
            self.stage.GetPrimAtPath(path).SetActive(False)
        for path in sorted(added):
            self.stage.GetPrimAtPath(path).SetActive(True)
            # USD payload composition is synchronous. Do not Unload live RTX
            # payloads: native_view uses activation for the same reason.
            self.stage.Load(path)
        self.resident = desired
        self.center = point
        if added or removed:
            print(
                f'USIM_ISAAC_RESIDENCY added={len(added)} removed={len(removed)} '
                f'resident={len(desired)} center={point}',
                flush=True,
            )
        return bool(added or removed)
