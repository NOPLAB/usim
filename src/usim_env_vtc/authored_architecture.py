"""Original parametric architecture, using prototype bounds only as placement metadata.

These deliberately approximate recipes do not reproduce source topology, UVs or
materials. Coordinates are Z-up and normalized to the supplied local envelope.
City Hall floor apertures align with the inventoried stair envelope; thin slabs,
open portals and perimeter facades replace the original component artwork.
"""

from __future__ import annotations

from math import ceil, isfinite
from typing import TYPE_CHECKING, Literal, TypeAlias, assert_never

from .architecture_geometry import _Recipe

if TYPE_CHECKING:
    from pxr import Usd


ModelKind: TypeAlias = Literal[
    'facade',
    'portal',
    'elevator',
    'floor',
    '2F_floor',
    'roof',
    'rails',
    'balcony',
    'stairs',
    'pillar',
    'desk',
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


class UnsupportedModelError(ValueError):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f'Unsupported authored architecture category: {name}')


class ModelBoundsError(ValueError):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f'Invalid prototype bounds for {name}')


def author_architecture(
    stage: Usd.Stage,
    path: str,
    name: str,
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
) -> None:
    """Author original bounded geometry below a transform-free prototype Xform.

    Supported names are full inventory names, not fuzzy unknown-category matches.
    Architecture is an approximation with traversable openings, not a survey.
    """
    key = name.rsplit('/', 1)[-1]
    basic: dict[str, ModelKind] = {
        'Engine/BasicShapes/Cube': 'cube',
        'Engine/BasicShapes/Cone': 'cone',
        'Engine/BasicShapes/Cylinder': 'cylinder',
        'Engine/BasicShapes/Plane': 'plane',
        'Engine/BasicShapes/Sphere': 'sphere',
        'Engine/EngineMeshes/Cylinder': 'cylinder',
        'Engine/EngineMeshes/Sphere': 'sphere',
        'Engine/ArtTools/RenderToTexture/Meshes/S_1_Unit_Plane': 'plane',
        'StarterContent/Shapes/Shape_Cylinder': 'cylinder',
        'StarterContent/Shapes/Shape_TriPyramid': 'pyramid',
        'ParagonProps/Agora/Props/Meshes/Rock_Formation_Strip_C': 'rock',
        'Mannequin/Character/Mesh/SK_Mannequin': 'mannequin',
        'TCAssets/CityHallChair': 'chair',
        'TCAssets/CityHallTable1': 'table',
        'TCAssets/CityHallTable2': 'table',
        'TCAssets/RoadCone': 'roadcone',
    }
    category = basic.get(name)
    if name.startswith('TCAssets/CityHall/City_Hall_'):
        part = key.removeprefix('City_Hall_')
        if part in {'1F_noentry', 'walls', 'south_room', 'EV'}:
            category = 'facade' if part != 'EV' else 'elevator'
        elif part in {'floor', '2F_floor'}:
            category = 'floor' if part == 'floor' else '2F_floor'
        elif part in {'east_roof', 'south_roof', 'top_roof'}:
            category = 'roof'
        elif part in {'east_entrance', 'south_entrance'}:
            category = 'portal'
        elif part in {'2F_handrail_1', '2F_handrail_2', 'east_handrail', 'west_handrail'}:
            category = 'rails'
        elif part == 'balcony':
            category = 'balcony'
        elif part in {'counter_1', 'counter_2', 'reception'}:
            category = 'desk'
        elif part == 'Stairs':
            category = 'stairs'
        elif part.startswith('Pillar_') and part[7:] in {
            '01',
            '10',
            '11',
            '20',
            '21',
            '30',
            '31',
            '40',
            '41',
            '50',
            '51',
            '60',
            '61',
            '70',
            '71',
            *'ABCDEFGHIJKLMNO',
        }:
            category = 'pillar'
    if category is None:
        raise UnsupportedModelError(name)
    if (
        len(minimum) != 3
        or len(maximum) != 3
        or not all(isfinite(v) for v in (*minimum, *maximum))
        or any(hi < lo for lo, hi in zip(minimum, maximum))
        or any(maximum[i] == minimum[i] for i in range(2))
        or (category != 'plane' and maximum[2] == minimum[2])
    ):
        raise ModelBoundsError(name)
    r = _Recipe(stage, path, minimum, maximum)
    r.root.SetCustomDataByKey('authoredRecipe', category)
    r.root.SetCustomDataByKey('boundsOnlyApproximation', True)
    match category:
        case 'facade':
            r.facade(upper=key == 'City_Hall_walls')
        case 'portal':
            r.portal()
        case 'elevator':
            # Three-sided shaft with an open front, rather than a solid obstruction.
            r.box('Shaft_Back', (0, 0.94, 0), (1, 1, 1))
            r.box('Shaft_Left', (0, 0, 0), (0.06, 0.94, 1))
            r.box('Shaft_Right', (0.94, 0, 0), (1, 0.94, 1))
            r.box('Shaft_Header', (0.06, 0, 0.95), (0.94, 0.06, 1))
        case 'floor':
            thickness = min(0.18 / r.size[2], 1)
            r.box('Slab', (0, 0, 1 - thickness), (1, 1, 1), 'floor')
        case '2F_floor':
            # Metadata-only aperture matches Stairs X/Y, leaving its top landing exposed.
            x0, x1 = 20.163705825805664 / 94.0999984741211, 36.094017028808594 / 94.0999984741211
            y0, y1 = (65 - 21.25) / 65, (65 - 12.949999809265137) / 65
            thickness = min(0.18 / r.size[2], 0.2)
            for index, (a, b) in enumerate(
                [
                    ((0, 0), (x0, 1)),
                    ((x1, 0), (1, 1)),
                    ((x0, 0), (x1, y0)),
                    ((x0, y1), (x1, 1)),
                ]
            ):
                r.box(f'Slab_{index}', (*a, 1 - thickness), (*b, 1), 'floor')
        case 'roof':
            thickness = min(0.16 / r.size[2], 1)
            r.box('Roof_Slab', (0, 0, 1 - thickness), (1, 1, 1), 'floor')
        case 'rails' | 'balcony':
            if category == 'balcony':
                r.box('Balcony_Slab', (0, 0, 0), (1, 1, 0.16), 'floor')
            r.rails(balcony=category == 'balcony')
        case 'stairs':
            # Solid individual treads rise along +X; final tread meets upper-floor Z.
            count = max(2, ceil(r.size[2] / 0.18 - 1e-6))
            width = min(2.0 / r.size[1], 1)
            for index in range(count):
                r.box(
                    f'Tread_{index:03}',
                    (index / count, 0, 0),
                    ((index + 1) / count, width, (index + 1) / count),
                    'floor',
                )
        case 'pillar':
            r.box('Shaft', (0.15, 0.15, 0.04), (0.85, 0.85, 0.96))
            r.box('Foot', (0, 0, 0), (1, 1, 0.04))
            r.box('Capital', (0, 0, 0.96), (1, 1, 1))
        case 'desk':
            r.desk(reception=key == 'City_Hall_reception')
        case 'chair':
            r.box('Seat', (0, 0, 0.46), (1, 1, 0.55), 'seat')
            r.box('Back', (0, 0.88, 0.55), (1, 1, 1), 'seat')
            for index, (x, y) in enumerate(
                ((0.08, 0.08), (0.84, 0.08), (0.08, 0.84), (0.84, 0.84))
            ):
                r.box(f'Leg_{index}', (x, y, 0), (x + 0.08, y + 0.08, 0.46), 'metal')
        case 'table':
            r.box('Tabletop', (0, 0, 0.92), (1, 1, 1), 'wood')
            for index, (x, y) in enumerate(
                ((0.06, 0.06), (0.86, 0.06), (0.06, 0.86), (0.86, 0.86))
            ):
                r.box(f'Leg_{index}', (x, y, 0), (x + 0.08, y + 0.08, 0.92), 'metal')
        case 'roadcone':
            r.box('Base', (0, 0, 0), (1, 1, 0.08), 'metal')
            r.round('Cone', (0.08, 0.08, 0.08), (0.92, 0.92, 1), 'orange', cone=True)
            r.round('Reflector', (0.235, 0.235, 0.40), (0.765, 0.765, 0.47), 'white')
        case 'cube':
            r.box('Cube', (0, 0, 0), (1, 1, 1))
        case 'cone' | 'cylinder' | 'sphere' | 'rock':
            r.round(
                'Solid',
                (0, 0, 0),
                (1, 1, 1),
                'rock' if category == 'rock' else 'stone',
                cone=category == 'cone',
                sphere=category == 'sphere',
                rock=category == 'rock',
            )
        case 'plane':
            r.mesh('Plane', [(0, 0, 0.5), (1, 0, 0.5), (1, 1, 0.5), (0, 1, 0.5)], [(0, 1, 2, 3)])
        case 'pyramid':
            r.mesh(
                'Pyramid',
                [(0, 0, 0), (1, 0, 0), (0.5, 1, 0), (0.5, 0.4, 1)],
                [(2, 1, 0), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
            )
        case 'mannequin':
            r.round('Head', (0.40, 0.1, 0.84), (0.60, 0.9, 1), 'seat', sphere=True)
            r.box('Torso', (0.30, 0.1, 0.46), (0.70, 0.9, 0.84), 'seat')
            for index, (a, b) in enumerate(((0, 0.3), (0.7, 1))):
                r.box(f'Arm_{index}', (a, 0.3, 0.65), (b, 0.7, 0.77), 'metal')
            for index, (a, b) in enumerate(((0.32, 0.46), (0.54, 0.68))):
                r.box(f'Leg_{index}', (a, 0.2, 0), (b, 0.8, 0.46), 'metal')
        case unreachable:
            assert_never(unreachable)
