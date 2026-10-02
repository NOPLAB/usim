"""Exercise the standalone Isaac execution loop through a deterministic engine fixture."""

import contextlib
import io
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from usim.ports.isaac import IsaacSimulator, SimulationConfig
from usim.ports.isaac.sim import _run
from usim.simulation import Ros2Config, Velocity
from dataclasses import replace


class StandaloneIsaacTest(unittest.TestCase):
    def test_custom_robot_runs_without_ros_and_closes(self):
        actions = []
        references = []
        imports = []
        settings = []
        camera_poses = []
        robot = NS(
            get_dof_index=lambda name: {'drive_left': 0, 'drive_right': 1}[name],
            apply_action=actions.append,
            get_world_pose=lambda: (np.array([1.0, 2.0, 3.0]), np.array([1.0, 0, 0, 0])),
            get_linear_velocity=lambda: np.zeros(3),
            get_angular_velocity=lambda: np.zeros(3),
            get_joint_velocities=lambda: np.zeros(2),
        )

        class App:
            def __init__(self, configuration):
                self.running = True
                settings.append(configuration)

            def is_running(self):
                result = self.running
                self.running = False
                return result

            def close(self):
                actions.append('close')

        class Prim:
            def __init__(self, path, name, api):
                self.path, self.name, self.api = path, name, api

            def GetPath(self):
                return NS(pathString=self.path)

            def GetName(self):
                return self.name

            def HasAPI(self, api, *args):
                return api == self.api

            def GetVariantSet(self, name):
                return NS(SetVariantSelection=lambda selection: None)

        prims = [
            Prim('/World/Environment/Wall', 'Wall', 'collision'),
            Prim('/World/Custom/Base', 'Base', 'articulation'),
            Prim('/World/Custom/drive_left', 'drive_left', 'drive'),
            Prim('/World/Custom/drive_right', 'drive_right', 'drive'),
        ]
        stage = NS(
            GetPrimAtPath=lambda path: prims[1], Load=lambda: None, Traverse=lambda: iter(prims)
        )
        world = NS(
            scene=NS(add=lambda item: item),
            current_time=1.0,
            reset=lambda: None,
            step=lambda **kw: actions.append(('step', kw)),
        )
        drive = NS(
            GetStiffnessAttr=lambda: NS(Set=lambda value: None),
            GetDampingAttr=lambda: NS(Set=lambda value: None),
            GetMaxForceAttr=lambda: NS(Set=lambda value: None),
        )
        drive_api = NS(Get=lambda joint, kind: drive)
        for prim in prims[2:]:
            prim.api = drive_api
        camera = NS(
            authoring_object=NS(set_world_poses=lambda **kw: camera_poses.append(kw)),
            get_data=lambda name: (
                np.zeros((3, 4, 4), dtype=np.uint8) if name == 'rgb' else np.ones((3, 4)),
                None,
            ),
        )
        values = {
            'isaacsim': dict(SimulationApp=App),
            'isaacsim.core.experimental.utils.app': dict(play=lambda **kw: None),
            'omni.replicator.core': dict(orchestrator=NS(step=lambda **kw: None)),
            'omni.usd': dict(get_context=lambda: NS(get_stage=lambda: stage)),
            'isaacsim.asset.importer.urdf': dict(
                URDFImporterConfig=lambda **kw: imports.append(kw) or NS(**kw),
                URDFImporter=lambda config: NS(import_urdf=lambda: str(config.urdf_path)),
            ),
            'isaacsim.core.utils.extensions': dict(enable_extension=lambda name: None),
            'isaacsim.core.api': dict(World=lambda **kw: world),
            'isaacsim.core.prims': dict(SingleArticulation=lambda **kw: robot),
            'isaacsim.core.utils.stage': dict(
                add_reference_to_stage=lambda source, target: references.append(target)
            ),
            'isaacsim.core.utils.types': dict(ArticulationAction=lambda **kw: NS(**kw)),
            'isaacsim.sensors.experimental.rtx': dict(CameraSensor=lambda *a, **kw: camera),
            'pxr': dict(
                Usd=NS(),
                UsdGeom=NS(
                    Tokens=NS(z='z'),
                    GetStageUpAxis=lambda stage: 'z',
                    GetStageMetersPerUnit=lambda stage: 1,
                    Camera=NS(Define=lambda stage, path: None),
                ),
                UsdPhysics=NS(
                    CollisionAPI='collision',
                    ArticulationRootAPI='articulation',
                    DriveAPI=drive_api,
                ),
            ),
        }
        modules = {}
        for name, attributes in values.items():
            parts = name.split('.')
            for count in range(1, len(parts) + 1):
                key = '.'.join(parts[:count])
                if key not in modules:
                    modules[key] = types.ModuleType(key)
                    modules[key].__path__ = []
            modules[name].__dict__.update(attributes)
        for name, module in modules.items():
            parent, _, child = name.rpartition('.')
            if parent:
                setattr(modules[parent], child, module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world_path, robot_path = root / 'room.usd', root / 'robot.urdf'
            world_path.write_bytes(b'fixture')
            robot_path.write_bytes(b'fixture')
            config = SimulationConfig(
                world_path,
                robot_path,
                ros=None,
                robot_name='custom',
                left_joint='drive_left',
                right_joint='drive_right',
                camera_offset=(0.3, 0.2, 0.5),
                camera_width=4,
                camera_height=3,
            )
            output = io.StringIO()
            with (
                patch.dict(sys.modules, {**modules, 'rclpy': None, 'rvln_msgs': None}),
                patch('usim.ports.isaac.sim.time.monotonic', return_value=1.0),
                contextlib.redirect_stdout(output),
            ):
                _run(config, IsaacSimulator(robot_prim_path='/World/Custom'))
                cancelled = threading.Event()
                cancelled.set()
                cancelled_output = io.StringIO()
                with contextlib.redirect_stdout(cancelled_output):
                    _run(config, IsaacSimulator(robot_prim_path='/World/Custom'), stop=cancelled)
            bridge = NS(
                gate=NS(sample=Velocity),
                command_count=0,
                motor_requests=0,
                publish_state=lambda state: 'stamp',
                publish_color=lambda *args: NS(header='header'),
                destroy_node=lambda: actions.append('destroy_bridge'),
            )
            ros = types.ModuleType('rclpy')
            ros.init = lambda: actions.append('ros_init')
            ros.shutdown = lambda: actions.append('ros_shutdown')
            ros.spin_once = lambda *args, **kwargs: None
            camera.get_data = lambda name: (
                np.zeros((3, 4, 4)) if name == 'rgb' else np.empty(0),
                None,
            )
            with (
                patch.dict(sys.modules, {**modules, 'rclpy': ros}),
                patch('usim.ports.isaac.sim._load_ros_python'),
                patch('usim.bridges.ros2.Bridge', return_value=bridge),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, 'depth camera frame unavailable'),
            ):
                _run(
                    replace(config, ros=Ros2Config()),
                    IsaacSimulator(robot_prim_path='/World/Custom'),
                )
            self.assertEqual(actions[-3:], ['destroy_bridge', 'ros_shutdown', 'close'])
        report = json.loads(
            [line for line in output.getvalue().splitlines() if line.startswith('{')][-1]
        )
        self.assertEqual(
            (report['status'], report['physics_steps'], report['camera_frames']),
            ('finished', 1, 1),
        )
        self.assertEqual(report['command_count'], 0)
        self.assertEqual(references, ['/World/Environment', '/World/Custom'] * 3)
        self.assertEqual(
            imports[0]['joint_target_type'], {'^(drive_left|drive_right)$': 'velocity'}
        )
        np.testing.assert_allclose(camera_poses[0]['positions'], [[1.3, 2.2, 3.5]])
        self.assertEqual(actions[-1], 'close')
        cancellation = json.loads(
            [line for line in cancelled_output.getvalue().splitlines() if line.startswith('{')][-1]
        )
        self.assertEqual((cancellation['status'], cancellation['physics_steps']), ('stopped', 0))
