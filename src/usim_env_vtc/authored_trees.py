"""Original, texture-free vegetation; batched leaf surfaces keep prototypes inexpensive."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin
from random import Random
from typing import TYPE_CHECKING, Literal, assert_never

if TYPE_CHECKING:
    from pxr import Gf, Usd


@dataclass(frozen=True, slots=True)
class _Surface:
    """Accumulate disconnected closed wood hulls or individually shaped leaf surfaces."""

    points: list[Gf.Vec3f] = field(default_factory=list)
    triangles: list[tuple[int, int, int]] = field(default_factory=list)
    colors: list[Gf.Vec3f] = field(default_factory=list)

    def tube(self, start: Gf.Vec3f, end: Gf.Vec3f, radius: float, sides: int = 8) -> None:
        """Append a tapered, capped hull with longitudinal bark ridges."""
        from pxr import Gf

        axis = (end - start).GetNormalized()
        reference = Gf.Vec3f(0, 0, 1) if abs(axis[2]) < 0.9 else Gf.Vec3f(1, 0, 0)
        u = Gf.Cross(axis, reference).GetNormalized()
        v = Gf.Cross(axis, u)
        offset = len(self.points)
        for center, scale in ((start, 1.0), (end, 0.57)):
            for i in range(sides):
                angle = 2 * pi * i / sides
                ridge = 1 + 0.09 * sin(i * 2.4)
                self.points.append(
                    center + radius * scale * ridge * (cos(angle) * u + sin(angle) * v)
                )
                shade = 0.72 + 0.22 * (i % 3)
                self.colors.append(Gf.Vec3f(0.24 * shade, 0.145 * shade, 0.075 * shade))
        self.points.extend((start, end))
        self.colors.extend((Gf.Vec3f(0.36, 0.23, 0.12), Gf.Vec3f(0.51, 0.36, 0.19)))
        for i in range(sides):
            a, b = offset + i, offset + (i + 1) % sides
            c, d = a + sides, b + sides
            self.triangles.extend(
                ((a, b, d), (a, d, c), (offset + 2 * sides, b, a), (offset + 2 * sides + 1, c, d))
            )

    def leaf(self, center: Gf.Vec3f, length: float, rng: Random, needle: bool) -> None:
        """Append a pointed, folded leaf with a raised midrib and varied vertex pigment."""
        from pxr import Gf

        azimuth, tilt = rng.uniform(0, 2 * pi), rng.uniform(-0.8, 0.8)
        u = Gf.Vec3f(cos(azimuth), sin(azimuth), tilt).GetNormalized()
        v = Gf.Cross(u, Gf.Vec3f(0, 0, 1)).GetNormalized()
        n = Gf.Cross(u, v)
        width = length * (0.055 if needle else 0.36)
        outline = ((-1, 0), (-0.45, -1), (0.45, -0.8), (1, 0), (0.45, 0.8), (-0.45, 1))
        offset = len(self.points)
        shade = rng.uniform(0.65, 1.25)
        pigment = Gf.Vec3f(0.055, 0.19, 0.075) if needle else Gf.Vec3f(0.14, 0.34, 0.055)
        for x, y in outline:
            self.points.append(center + x * length * u + y * width * v)
            self.colors.append(pigment * shade * (0.82 + 0.18 * (x + 1) / 2))
        self.points.append(center + n * width * 0.22)
        self.colors.append(pigment * shade * 1.12)
        for i in range(6):
            self.triangles.append((offset + 6, offset + i, offset + (i + 1) % 6))


def _write_surface(stage: Usd.Stage, path: str, surface: _Surface, foliage: bool) -> None:
    """Declare topology, flat face normals, bounds, vertex colors and local material."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(surface.points)
    mesh.CreateFaceVertexCountsAttr([3] * len(surface.triangles))
    mesh.CreateFaceVertexIndicesAttr([i for triangle in surface.triangles for i in triangle])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    mesh.CreateDoubleSidedAttr(foliage)
    mesh.CreateExtentAttr(
        [
            Gf.Vec3f(*(min(p[i] for p in surface.points) for i in range(3))),
            Gf.Vec3f(*(max(p[i] for p in surface.points) for i in range(3))),
        ]
    )
    normals = []
    for a, b, c in surface.triangles:
        normal = Gf.Cross(
            surface.points[b] - surface.points[a], surface.points[c] - surface.points[a]
        ).GetNormalized()
        normals.extend((normal, normal, normal))
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(surface.colors)
    material = UsdShade.Material.Define(stage, path + '/Material')
    shader = UsdShade.Shader.Define(stage, path + '/Material/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.85)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0)
    reader = UsdShade.Shader.Define(stage, path + '/Material/Pigment')
    reader.CreateIdAttr('UsdPrimvarReader_float3')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('displayColor')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(
        reader.CreateOutput('result', Sdf.ValueTypeNames.Float3)
    )
    material.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    if not foliage:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')


def author_tree(
    stage: Usd.Stage,
    path: str,
    kind: Literal['broadleaf', 'pine', 'bush', 'stump'],
    seed: int = 0,
) -> None:
    """Author a reusable original prototype in XY [-1, 1], Z [0, 1], without root ops.

    Wood contains separate tapered closed hulls, not a canopy collision approximation.
    Leaves are thousands of disconnected folded surfaces batched into one USD mesh.
    All pigments and PreviewSurface nodes are local, with no asset dependencies.
    """
    from pxr import Gf, UsdGeom

    rng = Random(seed)
    wood, leaves = _Surface(), _Surface()

    def branch(start: Gf.Vec3f, direction: Gf.Vec3f, radius: float, depth: int) -> None:
        """Grow three recursive twig forks and attach leaves to the terminal wood."""
        end = start + direction
        wood.tube(start, end, radius)
        if depth:
            for i in range(3):
                angle = rng.uniform(0, 2 * pi)
                delta = Gf.Vec3f(cos(angle), sin(angle), rng.uniform(0.1, 0.6))
                child = direction * 0.48 + delta * direction.GetLength() * 0.42
                branch(start + direction * (0.55 + 0.2 * i), child, radius * 0.48, depth - 1)
        else:
            for i in range(36):
                anchor = start + direction * rng.uniform(0.15, 1.08)
                spread = Gf.Vec3f(*(rng.uniform(-0.09, 0.09) for _ in range(3)))
                leaves.leaf(anchor + spread, rng.uniform(0.035, 0.065), rng, False)

    match kind:
        case 'broadleaf' | 'bush':
            bush = kind == 'bush'
            wood.tube(Gf.Vec3f(0), Gf.Vec3f(0.035, -0.02, 0.8), 0.075 if bush else 0.09, 16)
            for i in range(12):
                angle = i * 2.399963 + rng.uniform(-0.2, 0.2)
                z = 0.12 + i * 0.045 if bush else 0.34 + i * 0.035
                length = rng.uniform(0.38, 0.58) * (1 - 0.025 * i)
                branch(
                    Gf.Vec3f(0.035 * z, -0.02 * z, z),
                    Gf.Vec3f(cos(angle) * length, sin(angle) * length, 0.2 + i * 0.008),
                    0.024 if bush else 0.033,
                    2,
                )
        case 'pine':
            wood.tube(Gf.Vec3f(0), Gf.Vec3f(0.015, 0, 1.05), 0.06, 16)
            for layer in range(11):
                z = 0.18 + layer * 0.073
                reach = 0.53 * (1 - layer / 12)
                for arm in range(7):
                    angle = arm * 2 * pi / 7 + layer * 0.78 + rng.uniform(-0.12, 0.12)
                    start = Gf.Vec3f(0, 0, z)
                    delta = Gf.Vec3f(cos(angle) * reach, sin(angle) * reach, 0.025)
                    wood.tube(start, start + delta, 0.013 * (1 - layer / 14))
                    for fork in range(3):
                        anchor = start + delta * (0.35 + fork * 0.24)
                        twig_angle = angle + (-0.65 if fork % 2 else 0.65)
                        twig = Gf.Vec3f(cos(twig_angle), sin(twig_angle), 0.35) * reach * 0.42
                        wood.tube(anchor, anchor + twig, 0.004)
                        for station in range(6):
                            center = anchor + twig * (station + 1) / 6
                            for _ in range(4):
                                jitter = Gf.Vec3f(*(rng.uniform(-0.018, 0.018) for _ in range(3)))
                                leaves.leaf(center + jitter, rng.uniform(0.035, 0.055), rng, True)
        case 'stump':
            for i in range(8):
                angle = i * pi / 4
                wood.tube(
                    Gf.Vec3f(0),
                    Gf.Vec3f(cos(angle) * 0.33, sin(angle) * 0.33, 0.035),
                    0.12,
                    16,
                )
            for i in range(6):
                wood.tube(
                    Gf.Vec3f(0, 0, i * 0.06),
                    Gf.Vec3f(0.008, 0, (i + 1) * 0.06),
                    0.23 - i * 0.015,
                    32,
                )
        case unreachable:
            assert_never(unreachable)

    points = wood.points + leaves.points
    radius = max(max(abs(p[0]), abs(p[1])) for p in points)
    low, high = min(p[2] for p in points), max(p[2] for p in points)
    for surface in (wood, leaves):
        surface.points[:] = [
            Gf.Vec3f(p[0] / radius, p[1] / radius, (p[2] - low) / (high - low))
            for p in surface.points
        ]
    UsdGeom.Xform.Define(stage, path)
    _write_surface(stage, path + '/Wood', wood, False)
    if leaves.points:
        _write_surface(stage, path + '/Leaves', leaves, True)
