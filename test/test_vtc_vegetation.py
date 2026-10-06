"""Derived canopy models must not turn buildings into trees or rewrite LiDAR."""

import numpy as np
import pytest
import hashlib
import json
import sys

pytest.importorskip('scipy')

from usim_env_vtc.vegetation import infer_trees


def crown(center):
    angle = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    latitude = np.linspace(-1.2, 1.2, 30)
    a, b = np.meshgrid(angle, latitude)
    return np.column_stack(
        (
            (center + 3 * np.cos(a) * np.cos(b)).ravel(),
            (3 * np.sin(a) * np.cos(b)).ravel(),
            (7 + 3 * np.sin(b)).ravel(),
        )
    )


def ground():
    x, y = np.meshgrid(np.arange(-6, 19, 0.2), np.arange(-6, 7, 0.2))
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))


def test_separate_volumetric_crowns_retain_height_and_original_points():
    # Given two canopy shells above measured ground, in an offset source Z frame.
    points = np.concatenate((ground(), crown(0), crown(12))).astype(np.float64)
    points[:, 2] -= 8
    original = points.copy()
    # When interpreting the canopy support.
    trees = infer_trees(points)
    # Then both trees use ground-relative height, without changing the source.
    assert len(trees) == 2
    assert sorted(tree.x for tree in trees) == pytest.approx([0.5, 12.5], abs=1)
    assert all(8 < tree.height < 11 and 2 < tree.radius < 4 for tree in trees)
    assert np.array_equal(points, original)


def test_flat_roof_does_not_become_a_canopy():
    # Given an elevated planar roof and the ground around it.
    roof = ground()
    roof[:, 2] = 12
    points = np.concatenate((ground(), roof)).astype(np.float64)
    # When detecting supported crowns.
    trees = infer_trees(points)
    # Then the constant-height roof supplies no tree.
    assert trees == []


def test_vertical_facade_does_not_become_a_canopy():
    # Given a tall, narrow plane beside measured ground.
    x, z = np.meshgrid(np.arange(-3, 4, 0.1), np.arange(0, 16, 0.1))
    wall = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    points = np.concatenate((ground(), wall)).astype(np.float64)
    # When detecting supported crowns.
    trees = infer_trees(points)
    # Then a facade's planar support is not accepted as vegetation.
    assert trees == []


def test_authored_negative_coordinate_tree_is_grounded_and_source_is_preserved(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip('pxr')
    from PIL import Image
    from pxr import Usd, UsdGeom
    from usim_env_vtc import vegetation
    from usim_env_vtc.pcd import DTYPE, FIELDS

    # Given a real PCD crown, a centered prototype, and negative VTC X coordinates.
    points = np.concatenate((ground(), crown(0))).astype(np.float32)
    points[:, 0] -= 45
    points[:, 1] += 40
    records = np.zeros(len(points), dtype=DTYPE)
    for axis in range(3):
        records[FIELDS[axis]] = points[:, axis]
    source = tmp_path / 'source.pcd'
    header = (
        'VERSION 0.7\nFIELDS '
        + ' '.join(FIELDS)
        + '\nSIZE 4 4 4 4 4 4 4 4\nTYPE F F F F F F F F\n'
        + f'COUNT 1 1 1 1 1 1 1 1\nWIDTH {len(records)}\nHEIGHT 1\n'
        + f'POINTS {len(records)}\nDATA binary\n'
    ).encode('ascii')
    source.write_bytes(header + records.tobytes())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setitem(vegetation.PUBLIC_HASHES, 'map_tc19_o085_f-04_t05.pcd', digest)
    registration = tmp_path / 'registration.json'
    registration.write_text(
        json.dumps(
            {
                'source19_to_vtc': np.eye(4).tolist(),
                'source_sha256': digest,
            }
        )
    )
    base = Usd.Stage.CreateNew(str(tmp_path / 'base.usda'))
    base.SetDefaultPrim(UsdGeom.Xform.Define(base, '/World').GetPrim())
    UsdGeom.Xform.Define(base, '/World/Structures/VTC')
    base.GetRootLayer().Save()
    prototype = Usd.Stage.CreateNew(str(tmp_path / 'tree.usda'))
    model = UsdGeom.Xform.Define(prototype, '/Model')
    cube = UsdGeom.Cube.Define(prototype, '/Model/Geometry')
    cube.CreateSizeAttr(2)
    UsdGeom.Xformable(cube).AddScaleOp().Set((5, 5, 10))
    prototype.SetDefaultPrim(model.GetPrim())
    prototype.GetRootLayer().Save()
    index = tmp_path / 'index.json'
    index.write_text(
        json.dumps(
            [
                {
                    'path': '/World/Terrain/TC_x0_y0',
                    'min': [-100, 0, -1],
                    'max': [0, 100, 1],
                }
            ]
        )
    )
    textures = tmp_path / 'terrain/source-textures/TC_x0_y0'
    textures.mkdir(parents=True)
    np.save(textures / 'height_uint16.npy', np.full((1009, 1009), 32768, dtype=np.uint16))
    masks = tmp_path / 'terrain/shader-masks'
    masks.mkdir()
    for number in range(2):
        Image.fromarray(np.zeros((1009, 1009, 4), dtype=np.uint8)).save(
            masks / f'TC_x0_y0-{number}.png'
        )
    out = tmp_path / 'result'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'vegetation.py',
            '--source',
            str(source),
            '--registration',
            str(registration),
            '--world',
            str(tmp_path / 'base.usda'),
            '--index',
            str(index),
            '--terrain',
            str(tmp_path / 'terrain'),
            '--prototype',
            str(tmp_path / 'tree.usda'),
            '--bounds',
            '-100',
            '0',
            '0',
            '100',
            '--out',
            str(out),
        ],
    )
    # When generating and reopening the actual composed USD tree layer.
    vegetation.main()
    actual = Usd.Stage.Open(str(out / 'world.usda'))
    tree = actual.GetPrimAtPath('/World/Vegetation/t_m2_1/Tree_0')
    bounds = (
        UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_],
        )
        .ComputeWorldBound(tree)
        .ComputeAlignedRange()
    )
    # Then its instance exists, its base is grounded, and source bytes are untouched.
    assert tree.IsInstance()
    assert bounds.GetMin()[2] == pytest.approx(0, abs=1e-5)
    assert 8 < bounds.GetMax()[2] < 11
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
