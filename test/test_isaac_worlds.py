"""Validate that the owned Gazebo pilot geometry survives conversion parsing."""

import unittest
import json
from pathlib import Path

import pytest

from usim_isaacsim.sim import IsaacSimulator
from usim_isaacsim.world_residency import WorldResidency
from usim_isaacsim.worlds import parse_boxes


ROOT = Path(__file__).resolve().parents[1]


def test_indexed_residency_remaps_paths_and_retains_boundary_payloads(tmp_path):
    index = tmp_path / 'index.json'
    index.write_text(
        json.dumps(
            [
                {'path': '/World/A', 'min': [-2, -2, -1], 'max': [2, 2, 1]},
                {'path': '/World/B', 'min': [65, -2, -1], 'max': [70, 2, 1]},
            ]
        )
    )
    events = []

    class Prim:
        def __init__(self, path):
            self.path = path

        def SetActive(self, active):
            events.append((self.path, active))

    class Stage:
        def GetPrimAtPath(self, path):
            return Prim(path)

        def Load(self, path):
            events.append(('load', path))

    residency = WorldResidency(Stage(), index, '/Scene/Map', 60.0)
    assert events == [('/Scene/Map/A', False), ('/Scene/Map/B', False)]
    assert residency.update((0, 0, 0))
    assert residency.resident == {'/Scene/Map/A'}
    assert events[-2:] == [('/Scene/Map/A', True), ('load', '/Scene/Map/A')]
    events.clear()
    assert not residency.update((4, 0, 100))
    assert not events
    assert residency.update((5, 0, 0))
    assert residency.resident == {'/Scene/Map/A', '/Scene/Map/B'}
    assert events == [('/Scene/Map/B', True), ('load', '/Scene/Map/B')]
    assert not residency.update((65, 0, 0))
    assert residency.update((75, 0, 0))
    assert residency.resident == {'/Scene/Map/B'}
    assert events[-1] == ('/Scene/Map/A', False)
    assert residency.update((0, 0, 0))
    assert residency.resident == {'/Scene/Map/A', '/Scene/Map/B'}
    assert residency.update((-10, 0, 0))
    assert residency.resident == {'/Scene/Map/A'}


@pytest.mark.parametrize('radius', [0, -1, float('nan'), float('inf')])
def test_residency_radius_must_be_finite_positive(radius):
    with pytest.raises(ValueError, match='residency_radius'):
        IsaacSimulator(residency_radius=radius)


@pytest.mark.parametrize(
    'entry',
    [
        {'path': '/Elsewhere/A', 'min': [0, 0, 0], 'max': [1, 1, 1]},
        {'path': '/World/A', 'min': [2, 0, 0], 'max': [1, 1, 1]},
        {'path': '/World/A', 'min': [0, 0], 'max': [1, 1, 1]},
    ],
)
def test_indexed_residency_rejects_invalid_index(tmp_path, entry):
    index = tmp_path / 'index.json'
    index.write_text(json.dumps([entry]))
    with pytest.raises(ValueError):
        WorldResidency(None, index, '/Scene/Map', 60.0)


class WorldTest(unittest.TestCase):
    def test_all_owned_worlds_have_matching_visual_and_collision_boxes(self):
        expected = {'corridor': 3, 'junction': 3, 'weave': 3}
        for scene, count in expected.items():
            with self.subTest(scene=scene):
                boxes = parse_boxes(ROOT / 'examples' / 'worlds' / f'{scene}.world')
                self.assertEqual(len(boxes), count)
                self.assertTrue(all(len(item['pose']) == 6 for item in boxes))


if __name__ == '__main__':
    unittest.main()
