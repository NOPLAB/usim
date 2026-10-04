"""Gazebo resource URI resolution and mounts."""

from pathlib import Path
from dataclasses import replace
import xml.etree.ElementTree as ET

import pytest

from usim.ports.gazebo import GazeboAssetConfig
from usim_gazebo.assets import prepare_assets
from usim.simulation import SimulationConfig


def prepared(
    config: SimulationConfig, directory: Path
) -> tuple[ET.Element, list[tuple[Path, str]]]:
    mounts = prepare_assets(config, directory, GazeboAssetConfig('/private/run_test/cmd_vel'))
    return ET.parse(directory / 'robot.urdf').getroot(), mounts


def test_relative_mesh_and_adjacent_texture_mounts_are_real(
    config: SimulationConfig, tmp_path: Path
) -> None:
    mesh_dir = tmp_path / 'geometry'
    mesh_dir.mkdir()
    mesh = mesh_dir / 'body.stl'
    mesh.write_text('solid body\nendsolid body\n')
    (tmp_path / 'textures').mkdir()
    (tmp_path / 'textures' / 'color.png').write_bytes(b'fixture')
    root = ET.parse(config.robot_urdf).getroot()
    geometry = root.findall('link/visual/geometry')[0]
    geometry.clear()
    ET.SubElement(geometry, 'mesh', filename='geometry/body.stl')
    ET.ElementTree(root).write(config.robot_urdf)
    robot, mounts = prepared(config, tmp_path / 'prepared')
    filename = robot.findall('link/visual/geometry/mesh')[0].get('filename')
    host, target = next((host, target) for host, target in mounts if host == tmp_path)
    assert filename == f'file://{target}/geometry/body.stl'
    assert (host / 'geometry' / '..' / 'textures' / 'color.png').is_file()
    assert (host / 'geometry' / 'body.stl').read_bytes() == mesh.read_bytes()


@pytest.mark.parametrize(
    'change', ['base', 'joint', 'joint_type', 'mesh', 'package', 'xml', 'world']
)
def test_malformed_or_unresolved_assets_fail_before_launch(
    config: SimulationConfig, tmp_path: Path, change: str
) -> None:
    if change == 'base':
        config = replace(config, base_link='absent')
    elif change == 'joint':
        config = replace(config, left_joint='absent')
    elif change == 'joint_type':
        config.robot_urdf.write_text(config.robot_urdf.read_text().replace('continuous', 'fixed'))
    elif change in ('mesh', 'package'):
        root = ET.parse(config.robot_urdf).getroot()
        ET.SubElement(
            root.findall('link/visual/geometry')[0],
            'mesh',
            filename=('missing.stl' if change == 'mesh' else 'package://not_installed/mesh.stl'),
        )
        ET.ElementTree(root).write(config.robot_urdf)
    elif change == 'xml':
        config.robot_urdf.write_text('<robot')
    else:
        config.world.write_text('<sdf><model name="not_a_world"/></sdf>')
    with pytest.raises(ValueError):
        prepared(config, tmp_path / 'prepared')
    assert not (tmp_path / 'prepared').exists()


def test_package_mesh_resolves_from_own_manifest(config: SimulationConfig, tmp_path: Path) -> None:
    (tmp_path / 'package.xml').write_text('<package><name>my_robot</name></package>')
    (tmp_path / 'mesh.stl').write_text('solid x\nendsolid x\n')
    root = ET.parse(config.robot_urdf).getroot()
    ET.SubElement(
        root.findall('link/visual/geometry')[0], 'mesh', filename='package://my_robot/mesh.stl'
    )
    ET.ElementTree(root).write(config.robot_urdf)
    robot, mounts = prepared(config, tmp_path / 'prepared')
    mesh = robot.findall('link/visual/geometry/mesh')[0]
    assert mesh.get('filename') == 'file:///assets/0/mesh.stl'
    assert mounts == [(tmp_path, '/assets/0')]


def test_local_model_uri_mounts_sibling_models_directory(
    config: SimulationConfig, tmp_path: Path
) -> None:
    checkout = tmp_path / 'checkout'
    worlds = checkout / 'worlds'
    models = checkout / 'models' / 'warehouse_shelf'
    worlds.mkdir(parents=True)
    models.mkdir(parents=True)
    (models / 'model.config').write_text('<model><name>warehouse_shelf</name></model>')
    (models / 'model.sdf').write_text('<sdf/>')
    world = worlds / 'warehouse.sdf'
    world.write_text(
        '<sdf version="1.6"><world name="warehouse">'
        '<include><uri>model://warehouse_shelf</uri></include></world></sdf>'
    )
    _, mounts = prepared(replace(config, world=world), tmp_path / 'prepared')
    output = ET.parse(tmp_path / 'prepared' / 'world.sdf').getroot()
    assert output.findtext('world/include/uri') == 'model://warehouse_shelf'
    assert mounts == [(worlds, '/assets/0'), (checkout / 'models', '/assets/1')]


def test_public_topic_cannot_be_used_as_private_drive(
    config: SimulationConfig, tmp_path: Path
) -> None:
    assert config.ros is not None
    with pytest.raises(ValueError):
        prepare_assets(config, tmp_path / 'prepared', GazeboAssetConfig(config.ros.cmd_vel_topic))


def test_unrelated_ros_package_directory_cannot_resolve_wrong_package(
    config: SimulationConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / 'mesh.stl').write_text('solid x')
    monkeypatch.setenv('ROS_PACKAGE_PATH', str(tmp_path))
    monkeypatch.delenv('AMENT_PREFIX_PATH', raising=False)
    root = ET.parse(config.robot_urdf).getroot()
    ET.SubElement(
        root.findall('link/visual/geometry')[0],
        'mesh',
        filename='package://wrong_package/mesh.stl',
    )
    ET.ElementTree(root).write(config.robot_urdf)
    with pytest.raises(ValueError):
        prepared(config, tmp_path / 'prepared')
