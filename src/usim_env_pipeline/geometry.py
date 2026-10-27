"""Reusable original mesh primitives and prop recipes."""

from __future__ import annotations

from collections.abc import Sequence
from math import cos, pi, sin
from typing import TYPE_CHECKING, Final, Literal

from typing_extensions import assert_never

if TYPE_CHECKING:
    from pxr import Usd, UsdShade

PropKind = Literal[
    'chair',
    'table',
    'roadcone',
    'cube',
    'cone',
    'cylinder',
    'sphere',
    'rock',
    'plane',
    'pyramid',
    'mannequin',
]

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


class GeometryRecipe:
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

    def prop(self, kind: PropKind) -> None:
        """Author an original generic prop within this recipe's local envelope."""
        match kind:
            case 'chair':
                self.box('Seat', (0, 0, 0.46), (1, 1, 0.55), 'seat')
                self.box('Back', (0, 0.88, 0.55), (1, 1, 1), 'seat')
                for index, (x, y) in enumerate(
                    ((0.08, 0.08), (0.84, 0.08), (0.08, 0.84), (0.84, 0.84))
                ):
                    self.box(f'Leg_{index}', (x, y, 0), (x + 0.08, y + 0.08, 0.46), 'metal')
            case 'table':
                self.box('Tabletop', (0, 0, 0.92), (1, 1, 1), 'wood')
                for index, (x, y) in enumerate(
                    ((0.06, 0.06), (0.86, 0.06), (0.06, 0.86), (0.86, 0.86))
                ):
                    self.box(f'Leg_{index}', (x, y, 0), (x + 0.08, y + 0.08, 0.92), 'metal')
            case 'roadcone':
                self.box('Base', (0, 0, 0), (1, 1, 0.08), 'metal')
                self.round('Cone', (0.08, 0.08, 0.08), (0.92, 0.92, 1), 'orange', cone=True)
                self.round('Reflector', (0.235, 0.235, 0.40), (0.765, 0.765, 0.47), 'white')
            case 'cube':
                self.box('Cube', (0, 0, 0), (1, 1, 1))
            case 'cone' | 'cylinder' | 'sphere' | 'rock':
                self.round(
                    'Solid',
                    (0, 0, 0),
                    (1, 1, 1),
                    'rock' if kind == 'rock' else 'stone',
                    cone=kind == 'cone',
                    sphere=kind == 'sphere',
                    rock=kind == 'rock',
                )
            case 'plane':
                self.mesh(
                    'Plane', [(0, 0, 0.5), (1, 0, 0.5), (1, 1, 0.5), (0, 1, 0.5)], [(0, 1, 2, 3)]
                )
            case 'pyramid':
                self.mesh(
                    'Pyramid',
                    [(0, 0, 0), (1, 0, 0), (0.5, 1, 0), (0.5, 0.4, 1)],
                    [(2, 1, 0), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
                )
            case 'mannequin':
                self.round('Head', (0.40, 0.1, 0.84), (0.60, 0.9, 1), 'seat', sphere=True)
                self.box('Torso', (0.30, 0.1, 0.46), (0.70, 0.9, 0.84), 'seat')
                for index, (a, b) in enumerate(((0, 0.3), (0.7, 1))):
                    self.box(f'Arm_{index}', (a, 0.3, 0.65), (b, 0.7, 0.77), 'metal')
                for index, (a, b) in enumerate(((0.32, 0.46), (0.54, 0.68))):
                    self.box(f'Leg_{index}', (a, 0.2, 0), (b, 0.8, 0.46), 'metal')
            case unreachable:
                assert_never(unreachable)
