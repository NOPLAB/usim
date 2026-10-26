"""VTC City Hall layout recipes over shared original mesh primitives."""

from math import ceil

from usim_env_pipeline.geometry import GeometryRecipe


class _Recipe(GeometryRecipe):
    def rails(self, prefix: str = 'Rail', balcony: bool = False) -> None:
        # A U-shaped guard leaves its fourth side open for circulation.
        t = min(0.06, 0.07 / max(min(self.size[:2]), 0.01))
        base = 0.16 if balcony else 0
        for side in range(3):
            axis = 0 if side < 2 else 1
            fixed = 0 if side == 0 else 1 - t
            length = self.size[1 - axis]
            count = max(2, ceil(length / 1.5))
            for index in range(count + 1):
                along = index / count * (1 - t)
                low = [along, along, base]
                high = [along + t, along + t, 1]
                low[axis], high[axis] = fixed, fixed + t
                self.box(f'{prefix}_{side}_Post_{index}', low, high, 'metal', False)
            low, high = [0, 0, 0.88], [1.0, 1.0, 1.0]
            low[axis], high[axis] = fixed, fixed + t
            self.box(f'{prefix}_{side}_Top', low, high, 'metal', False)

    def facade(self, upper: bool = False) -> None:
        """Window bays on four perimeter faces, with two unfilled ground portals."""
        t = min(0.04, 0.22 / max(min(self.size[:2]), 0.01))
        h = self.size[2]
        # Walls spans both levels; 1F_noentry supplies only the lower envelope.
        levels: list[tuple[float, float]] = [(0, min(3.7 / h, 1))]
        if upper and h > 5.4:
            levels.append((5.4 / h, 1))
        for side in range(4):
            axis = side // 2
            fixed = 0 if side % 2 == 0 else 1 - t
            along_axis = 1 - axis
            count = max(3, ceil(self.size[along_axis] / 4))

            def panel(
                label: str,
                a: float,
                b: float,
                z0: float,
                z1: float,
                material: str = 'stone',
                collision: bool = True,
            ) -> None:
                low, high = [0, 0, z0], [1, 1, z1]
                low[axis], high[axis] = fixed, fixed + t
                low[along_axis], high[along_axis] = a, b
                self.box(f'Facade_{side}_{label}', low, high, material, collision)

            for level, (z0, z1) in enumerate(levels):
                for bay in range(count):
                    a, b = bay / count, (bay + 1) / count
                    # East portal matches entrance Y; south portal admits the vestibule.
                    portal = level == 0 and (
                        (side == 0 and a < 0.80 and b > 0.68)
                        or (side == 3 and a < 0.60 and b > 0.48)
                    )
                    label = f'{level}_{bay}'
                    height = z1 - z0
                    if portal:
                        panel(label + '_Lintel', a, b, z0 + height * 0.88, z1)
                        continue
                    frame = min(0.01, (b - a) * 0.12)
                    panel(label + '_Pier', a, a + frame, z0, z1)
                    panel(label + '_Sill', a + frame, b, z0, z0 + height * 0.24)
                    panel(label + '_Header', a + frame, b, z0 + height * 0.86, z1)
                    panel(
                        label + '_Glass',
                        a + frame,
                        b,
                        z0 + height * 0.24,
                        z0 + height * 0.86,
                        'glass',
                        False,
                    )
                    middle = (a + b) / 2
                    panel(
                        label + '_Mullion',
                        middle,
                        middle + frame / 2,
                        z0 + height * 0.24,
                        z0 + height * 0.86,
                        'metal',
                        False,
                    )

    def portal(self) -> None:
        # Choose the thin axis for the wall plane; the other axis has a walk-through.
        axis = 0 if self.size[0] < self.size[1] else 1
        for index, (a, b) in enumerate(((0, 0.14), (0.86, 1))):
            low, high = [a, a, 0], [b, b, 1]
            low[axis], high[axis] = 0, min(1, 0.18 / self.size[axis])
            self.box(f'Portal_Jamb_{index}', low, high, 'metal')
        low, high = [0, 0, 0.88], [1.0, 1.0, 1.0]
        high[axis] = min(1, 0.18 / self.size[axis])
        self.box('Portal_Lintel', low, high, 'metal')

    def desk(self, reception: bool = False) -> None:
        self.box('Worktop', (0, 0, 0.9), (1, 1, 1), 'wood')
        self.box('Front', (0, 0, 0), (1, 0.12, 0.9), 'stone')
        for index, x in enumerate((0.05, 0.88)):
            self.box(f'Support_{index}', (x, 0.15, 0), (x + 0.07, 0.85, 0.9), 'metal')
        if reception:
            self.box('Return', (0, 0.1, 0.75), (0.15, 1, 0.9), 'wood')
