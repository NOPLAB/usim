"""Exercise installed console commands outside checkouts, without source PYTHONPATH."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def executable(name: str, bin_dir: Path | None) -> str:
    if bin_dir is None:
        found = shutil.which(name)
        if found:
            return str(Path(found).absolute())
    else:
        for suffix in ('.exe', '') if os.name == 'nt' else ('',):
            candidate = bin_dir / (name + suffix)
            if candidate.is_file():
                # A venv's python is a symlink; resolving it would escape the environment.
                return str(candidate.absolute())
    raise FileNotFoundError(f'installed executable not found: {name}')


def invoke(argv: list[str], cwd: Path, env: dict, expected: int = 0):
    result = subprocess.run(
        argv, cwd=cwd, env=env, capture_output=True, text=True, encoding='utf-8', timeout=60
    )
    if result.returncode != expected:
        raise RuntimeError(
            f'{argv}: exit {result.returncode}, expected {expected}\n'
            f'{result.stdout}\n{result.stderr}'
        )
    return result


def check_robot(path: Path) -> None:
    robot = ET.parse(path).getroot()
    assert robot.attrib['name'] == 'portable_smoke'
    boxes = [
        list(map(float, box.attrib['size'].split()))
        for box in robot.findall('.//collision/geometry/box')
    ]
    assert [0.8, 0.5, 0.3] in boxes
    wheels = [
        joint for joint in robot.findall('joint') if joint.attrib.get('type') == 'continuous'
    ]
    assert len(wheels) == 2
    positions = []
    for joint in wheels:
        child = joint.find('child').attrib['link']
        link = robot.find(f"link[@name='{child}']")
        for shape in ('./visual/geometry/cylinder', './collision/geometry/sphere'):
            assert math.isclose(float(link.find(shape).attrib['radius']), 0.11)
        positions.append(list(map(float, joint.find('origin').attrib['xyz'].split())))
    assert math.isclose(abs(positions[0][1] - positions[1][1]), 0.64)
    assert math.isclose(
        sum(float(mass.attrib['value']) for mass in robot.findall('./link/inertial/mass')), 14
    )
    camera_joints = [
        joint for joint in robot.findall('joint') if 'camera' in joint.find('child').attrib['link']
    ]
    assert any(
        list(map(float, joint.find('origin').attrib['xyz'].split())) == [0.2, 0.1, 0.6]
        for joint in camera_joints
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=Path)
    args = parser.parse_args()
    bins = args.bin_dir.resolve() if args.bin_dir else None
    usim = executable('usim', bins)
    python = executable('python' if os.name == 'nt' else 'python3', bins or Path(usim).parent)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONUTF8'] = '1'
    with tempfile.TemporaryDirectory(prefix='usim-installed-') as directory:
        root = Path(directory)
        invoke([usim, '--help'], root, env)
        invoke([usim, 'run', '--help'], root, env)
        invoke([usim, 'backends'], root, env)
        invoke([usim, 'simulate', '--help'], root, env)
        invoke([usim, 'convert-world', '--help'], root, env)
        robot = root / 'robot.urdf'
        invoke(
            [
                usim,
                'create-robot',
                '--out',
                str(robot),
                '--name',
                'portable_smoke',
                '--wheel-radius',
                '0.11',
                '--wheel-separation',
                '0.64',
                '--body-size',
                '0.8',
                '0.5',
                '0.3',
                '--mass',
                '14',
                '--camera-offset',
                '0.2',
                '0.1',
                '0.6',
            ],
            root,
            env,
        )
        check_robot(robot)
        invalid = root / 'invalid.urdf'
        invoke(
            [usim, 'create-robot', '--out', str(invalid), '--wheel-radius', '0'],
            root,
            env,
            expected=2,
        )
        assert not invalid.exists()
        invoke(
            [usim, 'simulate', '--world', str(root / 'missing.usd'), '--robot-urdf', str(robot)],
            root,
            env,
            expected=2,
        )
        invoke(
            [
                python,
                '-c',
                'import importlib.util,usim; '
                'assert importlib.util.find_spec("isaac_rvln") is None; '
                'assert importlib.util.find_spec("isaac_r2r") is None; '
                'from usim import list_simulators; list_simulators()',
            ],
            root,
            env,
        )
        print(
            json.dumps(
                {
                    'status': 'passed',
                    'create_robot': 'passed',
                    'invalid_geometry_exit': 2,
                    'missing_world_exit': 2,
                    'application_adapters': 'absent',
                }
            )
        )


if __name__ == '__main__':
    main()
