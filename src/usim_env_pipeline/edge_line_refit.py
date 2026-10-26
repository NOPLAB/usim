"""Constrained edge refitting from unions of unchanged measured face observations."""

from dataclasses import replace
from itertools import permutations
from typing import assert_never

import numpy as np
from numpy.typing import NDArray

from .edge_line_types import EdgeLineConfig, MeasuredEdge
from .plane_types import plane_basis
from .planes import refit


def refit_edge_union(
    xyz: NDArray[np.float64], edges: list[MeasuredEdge], config: EdgeLineConfig
) -> MeasuredEdge | None:
    """Reuse unchanged fitted data; refit only a genuinely new support union."""
    seed = edges[0]
    if seed.kind == 2:
        unchanged = True
        for edge in edges[1:]:
            matched = False
            for order in permutations(range(2)):
                identical = True
                for face, other in enumerate(order):
                    equation = edge.equations[other]
                    if equation[:3] @ seed.equations[face, :3] < 0:
                        equation = -equation
                    first = (seed.face_members or seed.line.witnesses)[face]
                    second = (edge.face_members or edge.line.witnesses)[other]
                    if not np.array_equal(equation, seed.equations[face]) or not (
                        first is second or np.array_equal(np.sort(first), np.sort(second))
                    ):
                        identical = False
                        break
                if identical:
                    matched = True
                    break
            if not matched:
                unchanged = False
                break
        if unchanged:
            return seed
    return refit_edge_models(xyz, edges, config)


def refit_edge_models(
    xyz: NDArray[np.float64], edges: list[MeasuredEdge], config: EdgeLineConfig
) -> MeasuredEdge | None:
    """Refit compatible original supports; never average prediction coordinates."""
    seed = edges[0]
    count = len(seed.line.witnesses)
    mappings = [
        max(
            permutations(range(count)),
            key=lambda order: sum(
                abs(seed.equations[face, :3] @ edge.equations[other, :3])
                for face, other in enumerate(order)
            ),
        )
        for edge in edges
    ]
    equations = seed.equations.copy()
    faces = []
    cosine = np.cos(np.deg2rad(config.max_refit_angle))
    for face in range(count):
        groups = [
            (edge.face_members or edge.line.witnesses)[order[face]]
            for edge, order in zip(edges, mappings, strict=True)
        ]
        ids = np.unique(np.concatenate(groups))
        match seed.kind:
            case 2:
                residual = np.abs(xyz[ids] @ equations[:2, :3].T + equations[:2, 3])
                ids = ids[residual[:, face] + config.normal_advantage < residual[:, 1 - face]]
            case 1:
                if not all(edge.face_members for edge in edges):
                    faces.append(ids)
                    continue
            case unreachable:
                assert_never(unreachable)
        if len(ids) < config.support.min_points:
            return None
        plane = refit(xyz, ids)
        for _ in range(2):
            ids = ids[np.abs(xyz[ids] @ plane.normal + plane.offset) <= config.plane_tolerance]
            if len(ids) < config.support.min_points:
                return None
            plane = refit(xyz, ids)
        if len(ids) < config.support.min_points or any(
            abs(plane.normal @ edge.equations[order[face], :3]) < cosine
            for edge, order in zip(edges, mappings, strict=True)
        ):
            return None
        equations[face] = np.r_[plane.normal, plane.offset]
        faces.append(ids)
    line = seed.line
    match seed.kind:
        case 1:
            members = np.unique(np.concatenate([edge.line.witnesses[0] for edge in edges]))
            normal, offset = equations[0, :3], equations[0, 3]
            u, v = plane_basis(normal)
            basis = np.column_stack((u, v))
            center = xyz[members].mean(axis=0)
            center = center - (center @ normal + offset) * normal
            uv = (xyz[members] - center) @ basis
            tangent = np.linalg.eigh(uv.T @ uv)[1][:, -1]
            direction = basis @ tangent
            if direction[np.argmax(np.abs(direction))] < 0:
                direction = -direction
            t = (line.endpoints - center) @ direction
            line = replace(line, endpoints=center + t[:, None] * direction)
        case 2:
            pass
        case unreachable:
            assert_never(unreachable)
    return replace(seed, line=line, equations=equations, face_members=tuple(faces))
