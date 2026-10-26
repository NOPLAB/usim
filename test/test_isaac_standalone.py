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
from usim_isaacsim.sim import _lidar_points, _lidar_ranges, _pace, _run
from usim.simulation import Ros2Config, Velocity
from dataclasses import replace


class StandaloneIsaacTest(unittest.TestCase):
    def test_multi_elevation_returns_keep_terrain_hits_and_follow_body_pitch(self):
        config = SimulationConfig(
            Path('world'),
            Path('robot'),
            lidar_samples=3,
            lidar_angle_min=0,
            lidar_angle_max=np.pi,
            lidar_offset=(1.0, 0.0, 0.5),
            lidar_vertical_angles=(0.0, np.pi / 6),
        )
        rays = []

        def cast(origin, direction, distance, report):
            rays.append((origin, direction))
            report(NS(collision='/Robot/body', rigid_body='', distance=0.01))
            if len(rays) in (1, 4):
                report(NS(collision='/World/Ground', rigid_body='', distance=2.0))
                report(NS(collision='/World/Wall', rigid_body='', distance=4.0))
            elif len(rays) == 2:
                report(NS(collision='/World/Wall', rigid_body='', distance=0.01))
            elif len(rays) == 5:
                report(NS(collision='/World/Wall', rigid_body='', distance=np.inf))

        with patch.dict(sys.modules, {'carb': NS(Float3=lambda *xyz: xyz)}):
            points = _lidar_points(
                NS(raycast_all=cast),
                np.array([2, 3, 4]),
                np.array([2**-0.5, 0, 2**-0.5, 0]),
                config,
                '/Robot',
            )
        self.assertEqual(len(rays), 6)
        np.testing.assert_allclose(rays[0][0], [2.5, 3, 3], atol=1e-15)
        np.testing.assert_allclose(rays[0][1], [0, 0, -1], atol=1e-15)
        np.testing.assert_allclose(rays[3][1], [0.5, 0, -(3**0.5) / 2], atol=1e-15)
        np.testing.assert_allclose(points, [[2, 0, 0], [3**0.5, 0, 1]])
        self.assertTrue(np.isfinite(points).all())

    def test_point_cloud_loop_does_not_publish_a_planar_scan(self):
        self.exercise_robot(variant='points')

    def test_pacing_uses_absolute_clock_and_bounded_cancel_wait(self):
        waits = []
        stop = NS(wait=waits.append)
        with patch('usim_isaacsim.sim.time.monotonic', return_value=11.0):
            _pace(10.0, 2.0, stop, 0)
            _pace(10.0, 20.0, stop, 3.0)
            _pace(10.0, 0.5, stop, 0)
        self.assertEqual(waits, [1.0, 2.0])

    def test_raycast_filters_self_selects_nearest_and_rotates_physical_offset(self):
        config = SimulationConfig(
            Path('world'),
            Path('robot'),
            lidar_samples=3,
            lidar_angle_min=0,
            lidar_angle_max=np.pi,
            lidar_offset=(1.0, 0.0, 0.5),
        )
        rays = []

        def cast(origin, direction, distance, report):
            rays.append((origin, direction, distance))
            hits = [
                NS(collision='/Robot/body', rigid_body='/Robot', distance=0.01),
                NS(collision='/wall/far', rigid_body='', distance=8.0),
                NS(collision='/wall/near', rigid_body='', distance=2.0),
            ]
            if len(rays) == 2:
                hits.append(NS(collision='/RobotOther', rigid_body='', distance=0.02))
            if len(rays) == 3:
                hits = hits[:1]
            for hit in hits:
                self.assertTrue(report(hit))

        with patch.dict(sys.modules, {'carb': NS(Float3=lambda *xyz: xyz)}):
            ranges = _lidar_ranges(
                NS(raycast_all=cast),
                np.array([2.0, 3.0, 4.0]),
                np.array([2**-0.5, 0, 0, 2**-0.5]),
                config,
                '/Robot',
            )
        self.assertEqual(ranges, [2.0, float('inf'), float('inf')])
        np.testing.assert_allclose(rays[0][0], [2, 4, 4.5])
        np.testing.assert_allclose(rays[0][1], [0, 1, 0], atol=1e-15)
        self.assertEqual(rays[0][2], 30.0)

    def test_visible_physics_spawn_and_lidar(self):
        self.exercise_robot(variant='navigation')

    def test_indexed_residency_loads_before_reset_and_follows_robot(self):
        self.exercise_robot(variant='residency')

    def test_visible_camera_samples_at_configured_cadence_after_completed_render(self):
        self.exercise_robot(variant='camera_lifetime')

    def test_completed_camera_frame_timestamp_is_preserved(self):
        self.exercise_robot(variant='camera_ros')

    def test_low_preview_rate_preserves_physics_and_completed_camera_frames(self):
        normal, normal_result = self.exercise_robot(variant='camera_ros')
        reduced, reduced_result = self.exercise_robot(variant='camera_ros', preview_hz=1.0)
        self.assertEqual(normal, [True] * 4)
        self.assertEqual(reduced, [True, False, True, False])
        self.assertEqual(normal_result, reduced_result)

    def test_preview_rate_defaults_to_30_and_rejects_invalid_values(self):
        self.assertEqual(IsaacSimulator().preview_hz, 30.0)
        for rate in (0, -1, float('inf'), float('-inf'), float('nan')):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, 'preview_hz'):
                IsaacSimulator(preview_hz=rate)

    def test_stale_completed_camera_frames_fail_instead_of_republishing(self):
        self.exercise_robot(variant='camera_stale')

    def test_custom_robot_runs_without_ros_and_closes(self):
        self.exercise_robot()

    def test_paired_wheels_receive_synchronized_actions(self):
        self.exercise_robot(paired=True)

    def test_paired_wheel_selection_rejects_missing_duplicate_and_aliased_joints(self):
        for invalid in ('missing', 'duplicate', 'index'):
            with self.subTest(invalid=invalid):
                self.exercise_robot(paired=True, invalid=invalid)

    def exercise_robot(self, paired=False, invalid=None, variant=None, preview_hz=None):
        actions = []
        references = []
        imports = []
        settings = []
        camera_poses = []
        spawns, views, lights, reads, rays = [], [], [], [], []
        camera_settings = []
        world_settings = []
        camera_ready = NS(initialized=False, depth=False, active=True)
        cached_frame: dict[str, object] = {
            'rgb': None,
            'distance_to_image_plane': None,
            'rendering_time': 0.0,
        }
        borrowed_rgb = np.empty((3, 4, 4), dtype=np.uint8)
        rendered_frames = [0]
        wall_clock = [1.0]
        left_names = ('drive_left', 'rear_left') if paired else ('drive_left',)
        right_names = ('drive_right', 'rear_right') if paired else ('drive_right',)
        names = left_names + right_names
        indices = dict(zip(names, (2, 0, 3, 1) if paired else (0, 1)))
        if invalid == 'index':
            indices['rear_right'] = indices['rear_left']
        robot = NS(
            get_dof_index=indices.__getitem__,
            apply_action=actions.append,
            get_world_pose=lambda: (np.array([1.0, 2.0, 3.0]), np.array([1.0, 0, 0, 0])),
            get_linear_velocity=lambda: np.zeros(3),
            get_angular_velocity=lambda: np.zeros(3),
            get_joint_velocities=lambda: np.arange(len(names), dtype=float),
        )

        class App:
            def __init__(self, configuration):
                self.remaining = (
                    35
                    if variant == 'camera_stale'
                    else 4
                    if variant in ('camera_lifetime', 'camera_ros')
                    else 1
                )
                settings.append(configuration)

            def is_running(self):
                self.remaining -= 1
                return self.remaining >= 0

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
            *[Prim(f'/World/Custom/{name}', name, 'drive') for name in names],
        ]
        if invalid == 'missing':
            prims.pop()
        elif invalid == 'duplicate':
            prims.append(Prim('/World/Custom/Extra/rear_right', 'rear_right', 'drive'))
        stage = NS(
            GetPrimAtPath=lambda path: prims[1],
            Load=lambda: None,
            Traverse=lambda: iter(prims),
            GetSessionLayer=lambda: 'session',
            SetEditTarget=lambda layer: actions.append(('edit_target', layer)),
        )

        def step_world(**kw):
            actions.append(('step', kw))
            if variant in ('camera_lifetime', 'camera_ros', 'camera_stale'):
                wall_clock[0] += 0.25
                world.current_time += 0.25
            if (
                kw['render']
                and camera_ready.initialized
                and camera_ready.depth
                and camera_ready.active
            ):
                rendered_frames[0] += 1
                if variant != 'camera_stale' or cached_frame['rgb'] is None:
                    borrowed_rgb.fill(rendered_frames[0])
                    cached_frame.update(
                        rgb=borrowed_rgb,
                        distance_to_image_plane=np.ones((3, 4), dtype=np.float32),
                        rendering_time=world.current_time - 0.125,
                    )

        world = NS(
            scene=NS(add=lambda item: item),
            current_time=1.0,
            reset=lambda: actions.append('reset'),
            step=step_world,
        )
        drive = NS(
            GetStiffnessAttr=lambda: NS(Set=lambda value: None),
            GetDampingAttr=lambda: NS(Set=lambda value: None),
            GetMaxForceAttr=lambda: NS(Set=lambda value: None),
        )
        drive_api = NS(Get=lambda joint, kind: drive)
        for prim in prims[2:]:
            prim.api = drive_api

        def camera_data(*, clone):
            reads.append(clone)
            self.assertIs(clone, True)
            return {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in cached_frame.items()
            }

        def cast(origin, direction, distance, report):
            rays.append((origin, direction, distance))
            report(NS(collision='/World/Environment/Wall', rigid_body='', distance=2.0))

        camera = NS(
            initialize=lambda: setattr(camera_ready, 'initialized', True),
            add_distance_to_image_plane_to_frame=lambda: setattr(camera_ready, 'depth', True),
            pause=lambda: setattr(camera_ready, 'active', False),
            resume=lambda: setattr(camera_ready, 'active', True),
            set_world_pose=lambda **kw: camera_poses.append(kw),
            get_current_frame=camera_data,
        )
        values = {
            'isaacsim': dict(SimulationApp=App),
            'isaacsim.core.experimental.utils.app': dict(play=lambda **kw: None),
            'omni.usd': dict(get_context=lambda: NS(get_stage=lambda: stage)),
            'isaacsim.asset.importer.urdf': dict(
                URDFImporterConfig=lambda **kw: imports.append(kw) or NS(**kw),
                URDFImporter=lambda config: NS(import_urdf=lambda: str(config.urdf_path)),
            ),
            'isaacsim.core.utils.extensions': dict(enable_extension=lambda name: None),
            'isaacsim.core.api': dict(World=lambda **kw: world_settings.append(kw) or world),
            'isaacsim.core.prims': dict(
                SingleArticulation=lambda **kw: spawns.append(kw) or robot
            ),
            'isaacsim.core.utils.viewports': dict(set_camera_view=lambda **kw: views.append(kw)),
            'carb': dict(Float3=lambda *xyz: xyz),
            'omni.physx': dict(get_physx_scene_query_interface=lambda: NS(raycast_all=cast)),
            'isaacsim.core.utils.stage': dict(
                add_reference_to_stage=lambda source, target: references.append(target)
            ),
            'isaacsim.core.utils.types': dict(ArticulationAction=lambda **kw: NS(**kw)),
            'isaacsim.sensors.camera': dict(
                Camera=lambda *a, **kw: camera_settings.append(kw) or camera
            ),
            'builtin_interfaces.msg': dict(Time=NS),
            'pxr': dict(
                Usd=NS(),
                UsdLux=NS(
                    LightAPI='light',
                    DomeLight=NS(
                        Define=lambda stage, path: NS(
                            CreateIntensityAttr=lambda value: lights.append((path, value))
                        )
                    ),
                ),
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
                left_joints=left_names if paired else None,
                right_joints=right_names if paired else None,
                camera_offset=(0.3, 0.2, 0.5),
                camera_width=4,
                camera_height=3,
            )
            if variant in ('camera_lifetime', 'camera_ros', 'camera_stale'):
                config = replace(config, camera_hz=2.0)
            if variant in ('navigation', 'points', 'residency'):
                config = replace(
                    config,
                    initial_position=(4.0, 5.0, 0.1),
                    initial_yaw=np.pi / 2,
                    camera_enabled=False,
                    lidar_enabled=True,
                    lidar_samples=3,
                )
                if variant == 'points':
                    config = replace(config, lidar_vertical_angles=(0.0, np.pi / 6))
            if variant in ('points', 'camera_ros'):
                from test.test_isaac_runtime import runtime_modules

                modules.update(runtime_modules())
                modules['rclpy'].init = lambda: None
                modules['rclpy'].shutdown = lambda: None
                modules['rclpy'].spin_once = lambda *a, **kw: None
            output = io.StringIO()
            if invalid:
                with (
                    patch.dict(sys.modules, modules),
                    contextlib.redirect_stdout(output),
                    self.assertRaisesRegex(
                        RuntimeError, 'wheel drive missing|wheel joints missing'
                    ),
                ):
                    _run(config, IsaacSimulator(robot_prim_path='/World/Custom'))
                self.assertEqual(actions[-1], 'close')
                self.assertEqual(json.loads(output.getvalue().splitlines()[-1])['status'], 'error')
                return
            with (
                patch.dict(
                    sys.modules,
                    modules if variant in ('points', 'camera_ros') else {**modules, 'rclpy': None},
                ),
                patch('usim_isaacsim.sim.time.monotonic', side_effect=lambda: wall_clock[0]),
                contextlib.redirect_stdout(output),
            ):
                if variant in ('points', 'camera_ros'):
                    from usim.bridges.ros2 import Bridge

                    config = replace(config, ros=Ros2Config())
                    bridge = Bridge(
                        camera_enabled=variant == 'camera_ros',
                        lidar_enabled=variant == 'points',
                        lidar_points=variant == 'points',
                    )
                    bridge.destroy_node = lambda: None
                    with (
                        patch('usim_isaacsim.sim._load_ros_python'),
                        patch('usim.bridges.ros2.Bridge', return_value=bridge),
                    ):
                        port = IsaacSimulator(
                            robot_prim_path='/World/Custom',
                            preview_hz=30.0 if preview_hz is None else preview_hz,
                        )
                        _run(config, port)
                    if variant == 'points':
                        self.assertNotIn('/scan_raw', bridge.publishers)
                        assert bridge.points_pub is not None
                        cloud = bridge.points_pub.messages[0]
                        self.assertEqual(cloud.width, 6)
                        self.assertTrue(np.isfinite(np.frombuffer(cloud.data, dtype='<f4')).all())
                    else:
                        self.assertEqual(
                            [
                                (image.header.stamp.sec, image.header.stamp.nanosec)
                                for image in bridge.publishers['/camera/color/image_raw'].messages
                            ],
                            [(1, 125_000_000), (1, 625_000_000)],
                        )
                else:
                    port = IsaacSimulator(robot_prim_path='/World/Custom')
                    if variant == 'camera_stale':
                        with self.assertRaises(RuntimeError):
                            _run(config, port)
                        self.assertEqual(
                            sum(isinstance(item, tuple) and item[0] == 'step' for item in actions),
                            32,
                        )
                        self.assertEqual(actions[-1], 'close')
                        return
                    if variant == 'residency':
                        port.world_index = root / 'index.json'
                        with patch('usim_isaacsim.world_residency.WorldResidency') as resident:
                            resident.return_value.update.side_effect = lambda point: (
                                actions.append(('residency', tuple(point)))
                            )
                            _run(config, port)
                        assert actions.index(('residency', (4.0, 5.0, 0.1))) < actions.index(
                            'reset'
                        )
                        assert actions.index(('residency', (1.0, 2.0, 3.0))) > actions.index(
                            'reset'
                        )
                        assert len(views) == 2
                    else:
                        _run(config, port)
                if variant:
                    result = json.loads(output.getvalue().splitlines()[-1])
                    self.assertEqual(lights, [('/World/UsimDefaultLight', 1000.0)])
                    np.testing.assert_allclose(views[0]['eye'], [5.0, -2.0, 6.0])
                    rendered = [
                        item[1]['render']
                        for item in actions
                        if isinstance(item, tuple) and item[0] == 'step'
                    ]
                    if variant in ('navigation', 'points', 'residency'):
                        self.assertTrue(all(rendered))
                        np.testing.assert_allclose(spawns[0]['position'], [4, 5, 0.1])
                        np.testing.assert_allclose(
                            spawns[0]['orientation'], [2**-0.5, 0, 0, 2**-0.5]
                        )
                        self.assertEqual(result['scan_frames'], 1)
                        self.assertEqual(len(rays), 6 if variant == 'points' else 3)
                        self.assertEqual(result['camera_frames'], 0)
                    else:
                        self.assertEqual(
                            world_settings,
                            [
                                dict(
                                    stage_units_in_meters=1.0,
                                    physics_dt=1 / 60,
                                    rendering_dt=1 / 30,
                                )
                            ],
                        )
                        self.assertEqual(len(rendered), 4)
                        if preview_hz is None:
                            self.assertEqual(rendered, [True] * 4)
                        self.assertEqual(
                            (result['physics_steps'], result['camera_frames']), (4, 2)
                        )
                        self.assertEqual(reads, [True, True])
                        self.assertEqual(rendered_frames[0], 2)
                        self.assertEqual(camera_settings[0]['annotator_device'], 'cpu')
                        self.assertEqual(camera_settings[0]['resolution'], (4, 3))
                    return rendered, result
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
            camera.get_current_frame = lambda **kw: {
                'rgb': np.zeros((3, 4, 4)),
                'distance_to_image_plane': np.empty(0),
                'rendering_time': 1.0,
            }
            with (
                patch.dict(sys.modules, {**modules, 'rclpy': ros}),
                patch('usim_isaacsim.sim._load_ros_python'),
                patch('usim.bridges.ros2.Bridge', return_value=bridge),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, 'depth camera frame unavailable'),
            ):
                _run(
                    replace(config, ros=Ros2Config()),
                    IsaacSimulator(robot_prim_path='/World/Custom'),
                )
            self.assertEqual(actions[-3:], ['destroy_bridge', 'ros_shutdown', 'close'])
            bridge.gate.sample = lambda: Velocity(0.2, 0.5)
            driven_output = io.StringIO()
            with (
                patch.dict(sys.modules, {**modules, 'rclpy': ros}),
                patch('usim_isaacsim.sim._load_ros_python'),
                patch('usim.bridges.ros2.Bridge', return_value=bridge),
                contextlib.redirect_stdout(driven_output),
            ):
                _run(
                    replace(config, ros=Ros2Config(), camera_enabled=False),
                    IsaacSimulator(robot_prim_path='/World/Custom'),
                )
            action = next(item for item in reversed(actions) if hasattr(item, 'joint_indices'))
            np.testing.assert_array_equal(action.joint_indices, [indices[name] for name in names])
            np.testing.assert_allclose(
                action.joint_velocities, [1.5] * len(left_names) + [3.5] * len(right_names)
            )
            driven_report = json.loads(driven_output.getvalue().splitlines()[-1])
            self.assertEqual(driven_report['driven_steps'], 1)
            self.assertEqual(driven_report['wheel_velocities'], [indices[name] for name in names])
            self.assertEqual(
                driven_report['peak_wheel_velocities'], [indices[name] for name in names]
            )
        report = json.loads(
            [line for line in output.getvalue().splitlines() if line.startswith('{')][-1]
        )
        self.assertEqual(
            (report['status'], report['physics_steps'], report['camera_frames']),
            ('finished', 1, 1),
        )
        self.assertEqual(report['command_count'], 0)
        first_action = next(item for item in actions if hasattr(item, 'joint_indices'))
        np.testing.assert_array_equal(
            first_action.joint_indices, [indices[name] for name in names]
        )
        np.testing.assert_array_equal(first_action.joint_velocities, np.zeros(len(names)))
        self.assertEqual(references, ['/World/Environment', '/World/Custom'] * 4)
        self.assertEqual(
            imports[0]['joint_target_type'], {'^(' + '|'.join(names) + ')$': 'velocity'}
        )
        np.testing.assert_allclose(camera_poses[0]['position'], [1.3, 2.2, 3.5])
        self.assertEqual(camera_poses[0]['camera_axes'], 'usd')
        self.assertEqual(actions[-1], 'close')
        cancellation = json.loads(
            [line for line in cancelled_output.getvalue().splitlines() if line.startswith('{')][-1]
        )
        self.assertEqual((cancellation['status'], cancellation['physics_steps']), ('stopped', 0))
