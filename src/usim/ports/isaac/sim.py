"""Isaac Sim ROS 2 runtime for configured differential-drive mobile robots.

Requires an expanded mobile robot URDF and a collidable, Z-up USD world. This
module is kept separate from ROS launch because Isaac uses Python 3.12.
"""

from __future__ import annotations

import json
import math
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

from threading import Event

from usim.simulation import RobotState, SimulationConfig, Velocity, wheel_velocities
from usim.ports.isaac.contacts import ObstacleContactLog


def _load_ros_python() -> None:
    """Compatibility entry point for the ROS runtime loader."""
    from usim.bridges.ros_runtime import load_ros_python

    load_ros_python()


class IsaacSimulator:
    """Blocking Isaac lifecycle; engine-specific scene paths remain constructor options."""

    def __init__(
        self,
        *,
        robot_prim_path: str = '/World/Robot',
        environment_prim_path: str = '/World/Environment',
        camera_prim_path: str = '/World/Mobile_Camera',
        ground_name: str = 'Ground',
        contact_out: Path | None = None,
        optical_frame: str = 'camera_link',
        python: str = sys.executable,
    ):
        paths = (robot_prim_path, environment_prim_path, camera_prim_path)
        for path in paths:
            if not re.fullmatch(r'/[A-Za-z_][A-Za-z_0-9]*(/[A-Za-z_][A-Za-z_0-9]*)+', path):
                raise ValueError(f'invalid absolute USD prim path: {path}')
        for index, path in enumerate(paths):
            if any(
                path == other or path.startswith(other + '/') or other.startswith(path + '/')
                for other in paths[index + 1 :]
            ):
                raise ValueError('robot, environment and camera prim paths must not overlap')
        self.robot_prim_path = robot_prim_path
        self.environment_prim_path = environment_prim_path
        self.camera_prim_path = camera_prim_path
        self.ground_name = ground_name
        self.contact_out = contact_out
        self.optical_frame = optical_frame
        self.python = python

    def run(self, configuration: SimulationConfig, *, stop: Event | None = None) -> None:
        if stop is not None and stop.is_set():
            return
        for path in (configuration.world, configuration.robot_urdf):
            if not path.is_file():
                raise ValueError(f'missing asset: {path}')
        from usim.ports.isaac.runner import run

        run(configuration, self, stop=stop)


def _run(
    args: SimulationConfig,
    port: IsaacSimulator,
    *,
    stop: Event | None = None,
    asset_directory: Path | None = None,
) -> None:
    import faulthandler

    print('USIM_ISAAC_IMPORT', flush=True)
    # A stalled native bootstrap reports where it is blocked instead of hanging silently.
    faulthandler.dump_traceback_later(120)
    from isaacsim import SimulationApp

    faulthandler.cancel_dump_traceback_later()
    print('USIM_ISAAC_IMPORTED', flush=True)

    robot_assets = None
    contact_log = None
    contact_subscription = None
    contacts_saved = False
    bridge = None
    ros_started = False
    app = SimulationApp(
        {
            'headless': args.headless,
            'multi_gpu': False,
            'create_new_stage': False,
            'enable_crashreporter': False,
            'width': args.camera_width,
            'height': args.camera_height,
            'samples_per_pixel_per_frame': 1,
        }
    )
    print('USIM_ISAAC_APP_READY', flush=True)
    try:
        import numpy as np
        import isaacsim.core.experimental.utils.app as app_utils
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension('isaacsim.asset.importer.urdf')
        if args.ros is not None:
            _load_ros_python()
            import rclpy
            from usim.bridges.ros2 import Bridge
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.experimental.rtx import CameraSensor
        from pxr import Usd, UsdGeom, UsdPhysics

        robot_assets = tempfile.TemporaryDirectory(prefix='urdf-', dir=asset_directory)
        wheel_joints = f'^({re.escape(args.left_joint)}|{re.escape(args.right_joint)})$'
        import_config = URDFImporterConfig(
            urdf_path=str(args.robot_urdf.resolve()),
            usd_path=robot_assets.name,
            merge_fixed_joints=True,
            fix_base=False,
            collision_from_visuals=False,
            joint_target_type={wheel_joints: 'velocity'},
            joint_drive_type={wheel_joints: 'force'},
            override_joint_stiffness={wheel_joints: 0.0},
            override_joint_damping={wheel_joints: 1000.0},
        )
        robot_usd = URDFImporter(import_config).import_urdf()
        print('USIM_ISAAC_ROBOT_IMPORTED', flush=True)
        if not Path(robot_usd).is_file():
            raise RuntimeError('robot URDF import failed')

        world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30)
        add_reference_to_stage(str(args.world.resolve()), port.environment_prim_path)
        add_reference_to_stage(str(robot_usd), port.robot_prim_path)
        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath(port.robot_prim_path)
        robot_prim.GetVariantSet('Physics').SetVariantSelection('physx')
        stage.Load()
        if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
            raise RuntimeError('Isaac world must be Z-up')
        if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
            raise RuntimeError('Isaac world must use metres')
        if not any(
            prim.HasAPI(UsdPhysics.CollisionAPI)
            for prim in stage.Traverse()
            if prim.GetPath().pathString.startswith(port.environment_prim_path)
        ):
            raise RuntimeError('world USD has no collision geometry')

        robot_prefix = port.robot_prim_path.rstrip('/') + '/'
        articulation_roots = [
            prim
            for prim in stage.Traverse()
            if prim.GetPath().pathString.startswith(robot_prefix)
            and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        if len(articulation_roots) != 1:
            raise RuntimeError(
                f'expected one robot articulation root, got {len(articulation_roots)}'
            )
        for name in (args.left_joint, args.right_joint):
            joints = [
                prim
                for prim in stage.Traverse()
                if prim.GetPath().pathString.startswith(robot_prefix) and prim.GetName() == name
            ]
            if len(joints) != 1 or not joints[0].HasAPI(UsdPhysics.DriveAPI, 'angular'):
                raise RuntimeError(f'wheel drive missing from imported URDF: {name}')
            drive = UsdPhysics.DriveAPI.Get(joints[0], 'angular')
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(10000.0)
            drive.GetMaxForceAttr().Set(1000.0)
        robot = world.scene.add(
            SingleArticulation(
                prim_path=articulation_roots[0].GetPath().pathString, name=args.robot_name
            )
        )
        if port.contact_out is not None:
            from omni.physx import get_physx_simulation_interface
            from pxr import PhysicsSchemaTools, PhysxSchema

            environment = stage.GetPrimAtPath(port.environment_prim_path)
            obstacles = {
                child.GetName()
                for child in environment.GetChildren()
                if child.GetName() != port.ground_name
                and any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(child))
            }
            contact_log = ObstacleContactLog(
                obstacles,
                robot_prim_path=port.robot_prim_path,
                scene_prim_path=port.environment_prim_path,
                ground_name=port.ground_name,
            )
            bodies = [
                prim
                for prim in stage.Traverse()
                if prim.GetPath().pathString.startswith(robot_prefix)
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ]
            if not bodies:
                raise RuntimeError('imported robot has no rigid bodies for contact reports')
            for prim in bodies:
                PhysxSchema.PhysxContactReportAPI.Apply(prim)

            def on_contacts(headers, _data):
                for header in headers:
                    if header.num_contact_data < 1:
                        continue
                    paths = [
                        str(PhysicsSchemaTools.intToSdfPath(value))
                        for value in (
                            header.actor0,
                            header.actor1,
                            header.collider0,
                            header.collider1,
                        )
                    ]
                    contact_log.record(*paths, world.current_time, header.num_contact_data)

            contact_subscription = (
                get_physx_simulation_interface().subscribe_contact_report_events(on_contacts)
            )
        world.reset()
        camera = None
        camera_prim = None
        if args.camera_enabled:
            UsdGeom.Camera.Define(stage, port.camera_prim_path)
            camera = CameraSensor(
                port.camera_prim_path,
                resolution=(args.camera_height, args.camera_width),
                annotators=['rgb', 'distance_to_image_plane'],
            )
            camera_prim = camera.authoring_object
            app_utils.play(commit=True)
            rep.orchestrator.step(rt_subframes=2, pause_timeline=False)
        left = robot.get_dof_index(args.left_joint)
        right = robot.get_dof_index(args.right_joint)
        if left == right or min(left, right) < 0:
            raise RuntimeError('wheel joints missing from imported URDF')

        if args.ros is not None:
            rclpy.init()
            ros_started = True
            bridge = Bridge(
                camera_enabled=camera is not None,
                configuration=args.ros,
                base_link=args.base_link,
                optical_frame=port.optical_frame,
            )
        begin = time.monotonic()
        print('USIM_ISAAC_RUNNING', flush=True)
        last_camera = 0.0
        steps = 0
        camera_frames = 0
        empty_camera_frames = 0
        driven_steps = 0
        peak_joint_velocity = [0.0, 0.0]
        position, orientation = robot.get_world_pose()
        while (
            (stop is None or not stop.is_set())
            and app.is_running()
            and (args.max_seconds == 0 or time.monotonic() - begin < args.max_seconds)
        ):
            if bridge is not None:
                rclpy.spin_once(bridge, timeout_sec=0)
            motion = bridge.gate.sample() if bridge is not None else Velocity()
            command = (motion.linear, motion.angular)
            if command != (0.0, 0.0):
                driven_steps += 1
            velocity = wheel_velocities(*command, args.wheel_radius, args.wheel_separation)
            robot.apply_action(
                ArticulationAction(
                    joint_velocities=np.asarray(velocity), joint_indices=np.asarray([left, right])
                )
            )
            previous_position, previous_orientation = robot.get_world_pose()
            previous_yaw = math.atan2(
                2
                * (
                    previous_orientation[0] * previous_orientation[3]
                    + previous_orientation[1] * previous_orientation[2]
                ),
                1 - 2 * (previous_orientation[2] ** 2 + previous_orientation[3] ** 2),
            )
            if camera is not None:
                w, x, y, z = previous_orientation
                camera_orientation = (
                    np.array([w - x + y + z, w + x - y + z, -w + x + y + z, -w - x - y + z]) * 0.5
                )
                camera_prim.set_world_poses(
                    positions=np.asarray(
                        [
                            previous_position
                            + np.array(
                                [
                                    args.camera_offset[0] * math.cos(previous_yaw)
                                    - args.camera_offset[1] * math.sin(previous_yaw),
                                    args.camera_offset[0] * math.sin(previous_yaw)
                                    + args.camera_offset[1] * math.cos(previous_yaw),
                                    args.camera_offset[2],
                                ]
                            )
                        ]
                    ),
                    orientations=np.asarray([camera_orientation]),
                )
            # Render only steps whose frame is read: buffers of unread rendered frames are
            # recycled, and a later read faults with CUDA error 700.
            capture = camera is not None and time.monotonic() - last_camera >= 1 / args.camera_hz
            world.step(render=capture)
            if command != (0.0, 0.0):
                joint_velocity = robot.get_joint_velocities()[[left, right]]
                peak_joint_velocity = [
                    max(old, abs(float(new)))
                    for old, new in zip(peak_joint_velocity, joint_velocity)
                ]
            steps += 1
            if contact_log is not None:
                contact_log.tick()
            position, orientation = robot.get_world_pose()
            yaw = math.atan2(
                2 * (orientation[0] * orientation[3] + orientation[1] * orientation[2]),
                1 - 2 * (orientation[2] ** 2 + orientation[3] ** 2),
            )
            linear_world = robot.get_linear_velocity()
            angular_world = robot.get_angular_velocity()
            stamp = None
            if bridge is not None:
                state = RobotState(
                    sim_time=world.current_time,
                    position=tuple(float(v) for v in position),
                    orientation_xyzw=tuple(float(orientation[i]) for i in (1, 2, 3, 0)),
                    velocity=Velocity(
                        float(linear_world[0] * math.cos(yaw) + linear_world[1] * math.sin(yaw)),
                        float(angular_world[2]),
                    ),
                )
                stamp = bridge.publish_state(state)
            now = time.monotonic()
            if camera is not None and capture:
                color, _ = camera.get_data('rgb')
                if color is None:
                    rep.orchestrator.step(rt_subframes=2, pause_timeline=False)
                    color, _ = camera.get_data('rgb')
                rgba = (
                    color.numpy()
                    if color is not None and hasattr(color, 'numpy')
                    else np.asarray(color)
                    if color is not None
                    else np.empty(0)
                )
                if rgba.size == 0 or rgba.shape == ():
                    empty_camera_frames += 1
                    if empty_camera_frames >= 30:
                        raise RuntimeError(
                            'Isaac camera produced no RGB frame after 30 capture attempts'
                        )
                    continue
                if rgba.shape not in (
                    (args.camera_height, args.camera_width, 3),
                    (args.camera_height, args.camera_width, 4),
                ):
                    raise RuntimeError(f'unexpected camera frame: {rgba.shape}')
                if bridge is not None:
                    image = bridge.publish_color(stamp, rgba)
                depth, _ = camera.get_data('distance_to_image_plane')
                depth = (
                    depth.numpy()
                    if depth is not None and hasattr(depth, 'numpy')
                    else np.asarray(depth)
                )
                if depth.shape == (args.camera_height, args.camera_width, 1):
                    depth = depth[:, :, 0]
                if depth.shape != (args.camera_height, args.camera_width):
                    raise RuntimeError('depth camera frame unavailable')
                if bridge is not None:
                    bridge.publish_depth(image.header, depth)
                last_camera = now
                camera_frames += 1
        if contact_log is not None:
            contact_payload = contact_log.save(port.contact_out)
            contacts_saved = True
        complete = steps and (not args.camera_enabled or camera_frames)
        status = 'finished' if complete else 'incomplete'
        if stop is not None and stop.is_set():
            status = 'stopped'
        print(
            json.dumps(
                {
                    'status': status,
                    'physics_steps': steps,
                    'camera_frames': camera_frames,
                    'physics_only': not args.camera_enabled,
                    'command_count': bridge.command_count if bridge is not None else 0,
                    'motor_requests': bridge.motor_requests if bridge is not None else 0,
                    'driven_steps': driven_steps,
                    'final_position': [float(v) for v in position],
                    'wheel_velocities': [
                        float(v) for v in robot.get_joint_velocities()[[left, right]]
                    ],
                    'peak_wheel_velocities': peak_joint_velocity,
                    'collisions': contact_payload['collisions']
                    if contact_log is not None
                    else None,
                }
            ),
            flush=True,
        )
        if status == 'incomplete':
            raise RuntimeError('Isaac exited before producing camera frames')
    except Exception as error:
        # Kit may exit during close() before Python reports the re-raised traceback.
        traceback.print_exc()
        print(
            json.dumps({'status': 'error', 'error': f'{type(error).__name__}: {error}'}),
            flush=True,
        )
        raise
    finally:
        try:
            if contact_log is not None and not contacts_saved:
                contact_log.save(port.contact_out)
        finally:
            if bridge is not None:
                bridge.destroy_node()
            if ros_started:
                rclpy.shutdown()
            del contact_subscription
            try:
                app.close()
            finally:
                if robot_assets is not None:
                    robot_assets.cleanup()
