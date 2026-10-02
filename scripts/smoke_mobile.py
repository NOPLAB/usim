"""Generate a primitive robot/world and exercise real standard ROS control and cameras."""

from __future__ import annotations

import argparse
from importlib.util import find_spec
import os
from pathlib import Path
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET

from usim.robot import MobileRobot, render_robot
from usim.simulation import SimulationConfig


WORLD = """<?xml version="1.0"?>
<sdf version="1.6"><world name="pilot">
  <physics type="ode"><max_step_size>0.001</max_step_size>
    <real_time_update_rate>1000</real_time_update_rate></physics>
  <scene><ambient>0.5 0.5 0.5 1</ambient><shadows>false</shadows></scene>
  <light name="sun" type="directional"><pose>0 0 10 0 0 0</pose>
    <diffuse>0.8 0.8 0.8 1</diffuse><direction>-0.3 -0.2 -1</direction></light>
  <model name="floor"><static>true</static><link name="floor">
    <collision name="floor"><geometry><plane><normal>0 0 1</normal>
      <size>20 20</size></plane></geometry></collision>
    <visual name="floor"><geometry><plane><normal>0 0 1</normal>
      <size>20 20</size></plane></geometry><material>
      <ambient>0.5 0.5 0.5 1</ambient><diffuse>0.5 0.5 0.5 1</diffuse>
    </material></visual></link></model>
  <model name="marker"><static>true</static><pose>2 0 0.5 0 0 0</pose><link name="box">
    <collision name="box"><geometry><box><size>0.2 2 1</size></box></geometry></collision>
    <visual name="box"><geometry><box><size>0.2 2 1</size></box></geometry><material>
      <ambient>0.8 0.04 0.04 1</ambient><diffuse>0.8 0.04 0.04 1</diffuse>
    </material></visual></link></model>
</world></sdf>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('gazebo', 'isaac'), required=True)
    parser.add_argument('--image', default='usim-gazebo:local')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'smoke.json').unlink(missing_ok=True)
    robot = args.out_dir / 'robot.urdf'
    robot.write_text(render_robot(MobileRobot()), encoding='utf-8')
    world = args.out_dir / 'pilot.sdf'
    world.write_text(WORLD, encoding='utf-8')
    if args.backend == 'isaac':
        # The USD converter creates its own ground; retain the same marker geometry.
        sdf = ET.fromstring(WORLD)
        sdf.find('world').remove(sdf.find("world/model[@name='floor']"))
        ET.ElementTree(sdf).write(world, encoding='utf-8')
        usd = args.out_dir / 'pilot.usd'
        subprocess.run(
            [
                sys.executable,
                '-m',
                'usim.cli',
                'convert-world',
                '--world',
                str(world),
                '--out',
                str(usd),
            ],
            check=True,
            timeout=60,
        )
        world = usd
    config = SimulationConfig(
        world=world,
        robot_urdf=robot,
        headless=True,
        camera_width=320,
        camera_height=240,
        max_seconds=60,
    )
    if args.backend == 'gazebo':
        from usim.ports.gazebo import GazeboSimulator

        GazeboSimulator(image=args.image).smoke(config, args.out_dir)
    else:
        from dataclasses import replace
        from usim.bridges.smoke import probe
        from usim.ports.isaac import IsaacSimulator

        # Discover bundled ROS without bootstrapping Kit/Torch in the probe parent.
        sdk = find_spec('isaacsim')
        if sdk is None or sdk.origin is None:
            raise RuntimeError('Isaac SDK is not installed in this interpreter')
        os.environ['ISAAC_PATH'] = str(Path(sdk.origin).parent)
        config = replace(
            config, headless=args.headless, max_seconds=120, camera_width=640, camera_height=480
        )
        stop = threading.Event()
        errors = []

        def measure():
            try:
                probe(config, args.out_dir)
            except Exception as error:
                errors.append(error)
            finally:
                stop.set()

        thread = threading.Thread(target=measure, name='usim-isaac-probe', daemon=True)
        thread.start()
        try:
            IsaacSimulator(contact_out=args.out_dir / 'contacts.json').run(config, stop=stop)
        finally:
            thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError('Isaac probe did not finish')
        if errors:
            raise errors[0]


if __name__ == '__main__':
    main()
