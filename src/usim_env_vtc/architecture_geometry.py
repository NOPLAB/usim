"""Original mesh primitives and facade recipes for authored VTC architecture."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, cos, pi, sin
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pxr import Usd, UsdShade


_COLORS: Final = {
    'stone': (0.68, 0.72, 0.66),
    'floor': (0.34, 0.40, 0.43),
    'metal': (0.12, 0.23, 0.27),
    'glass': (0.24, 0.57, 0.64),
    'wood': (0.52, 0.27, 0.13),
    'seat': (0.22, 0.36, 0.48),
    'orange': (0.95, 0.24, 0.035),
    'white': (0.92, 0.93, 0.88),
    'rock': (0.40, 0.43, 0.37),
}


class _Recipe:
    def __init__(
        self,
        stage: Usd.Stage,
        path: str,
        minimum: tuple[float, float, float],
        maximum: tuple[float, float, float],
    ) -> None:
        from pxr import UsdGeom

        self.stage = stage
        self.path = path
        self.minimum = minimum
        self.size = tuple(hi - lo for lo, hi in zip(minimum, maximum))
        self.root = UsdGeom.Xform.Define(stage, path).GetPrim()
        self.materials: dict[str, UsdShade.Material] = {}

    def mesh(
        self,
        name: str,
        points: Sequence[Sequence[float]],
        faces: Sequence[Sequence[int]],
        material: str = 'stone',
        collision: bool = True,
    ) -> None:
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

        mesh = UsdGeom.Mesh.Define(self.stage, f'{self.path}/{name}')
        vertices = [
            tuple(lo + value * size for lo, value, size in zip(self.minimum, point, self.size))
            for point in points
        ]
        mesh.CreatePointsAttr(vertices)
        mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
        mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateExtentAttr(
            [
                tuple(min(point[axis] for point in vertices) for axis in range(3)),
                tuple(max(point[axis] for point in vertices) for axis in range(3)),
            ]
        )
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*_COLORS[material])])
        if material not in self.materials:
            mat = UsdShade.Material.Define(self.stage, f'{self.path}/Looks/{material}')
            shader = UsdShade.Shader.Define(self.stage, f'{mat.GetPath()}/Surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(_COLORS[material])
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.7)
            if material == 'glass':
                shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(0.32)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
            self.materials[material] = mat
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.materials[material])
        if collision:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            approximation = 'none' if len(faces) == 1 else 'convexHull'
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
                approximation
            )

    def box(
        self,
        name: str,
        low: Sequence[float],
        high: Sequence[float],
        material: str = 'stone',
        collision: bool = True,
    ) -> None:
        x, y, z = low
        a, b, c = high
        self.mesh(
            name,
            [
                (x, y, z),
                (a, y, z),
                (a, b, z),
                (x, b, z),
                (x, y, c),
                (a, y, c),
                (a, b, c),
                (x, b, c),
            ],
            [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
            material,
            collision,
        )

    def round(
        self,
        name: str,
        low: Sequence[float],
        high: Sequence[float],
        material: str = 'stone',
        cone: bool = False,
        sphere: bool = False,
        rock: bool = False,
    ) -> None:
        """Independently tessellated elliptical solids, never source mesh samples."""
        center = tuple((a + b) / 2 for a, b in zip(low, high))
        radius = tuple((b - a) / 2 for a, b in zip(low, high))
        points = []
        sections = []
        rings = 8 if sphere or rock else 1
        for row in range(rings + 1):
            fraction = row / rings
            pole = ((sphere or rock) and row in (0, rings)) or (cone and row == rings)
            r = 0 if pole else sin(pi * fraction) if sphere or rock else 1
            section = []
            for column in range(1 if pole else 16):
                angle = column * 2 * pi / 16
                irregular = 0.86 + 0.12 * sin(3 * angle + row) if rock else 1
                section.append(len(points))
                points.append(
                    (
                        center[0] + radius[0] * r * cos(angle) * irregular,
                        center[1] + radius[1] * r * sin(angle) * irregular,
                        center[2] - radius[2] * cos(pi * fraction)
                        if sphere or rock
                        else low[2] + fraction * (high[2] - low[2]),
                    )
                )
            sections.append(section)
        faces = []
        if len(sections[0]) > 1:
            faces.append(tuple(reversed(sections[0])))
        if len(sections[-1]) > 1:
            faces.append(tuple(sections[-1]))
        for bottom, top in zip(sections, sections[1:]):
            for column in range(16):
                a, b = bottom[column % len(bottom)], bottom[(column + 1) % len(bottom)]
                c, d = top[(column + 1) % len(top)], top[column % len(top)]
                faces.append(tuple(dict.fromkeys((a, b, c, d))))
        self.mesh(name, points, faces, material)

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
