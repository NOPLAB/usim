# Generic Gazebo Classic port

Gazebo Classic 11 runs in the ROS 2 Humble image. No robot-specific workspace,
policy model, or host ROS installation is needed. Docker Desktop must use Linux
containers. Classic is end-of-life; this port deliberately targets its existing
Humble plugins rather than claiming compatibility with modern Gazebo.

```sh
docker build -f docker/Dockerfile.gazebo -t usim-gazebo:local .
PYTHONPATH=src python scripts/smoke_mobile.py --backend gazebo \
  --image usim-gazebo:local --out-dir runs/gazebo-smoke
```

The host library is Python 3.12. Humble's `rclpy` runs under the container's
Python 3.10, **not** the host interpreter. Each invocation explicitly copies
`usim/__init__.py`, `robot.py`, `simulation.py`, package initializers, the two
ROS bridge source modules, and the Gazebo port into a temporary read-only mount.
It does not install the Python 3.12 wheel in the image. This also means source
changes do not require an image rebuild. The image supplies native dependencies.

## Library use

```python
from pathlib import Path
from threading import Event
from usim.simulation import SimulationConfig, Ros2Config
from usim.ports.gazebo import GazeboSimulator

stop = Event()  # Another thread may call stop.set().
GazeboSimulator(image='usim-gazebo:local').run(
    SimulationConfig(
        world=Path('my_world.sdf'),
        robot_urdf=Path('my_robot.urdf'),
        robot_name='my_robot',
        base_link='chassis',
        left_joint='left_axle',
        right_joint='right_axle',
        wheel_radius=0.1,
        wheel_separation=0.4,
        camera_offset=(0.15, 0.0, 0.3),
        headless=True,
        max_seconds=30,
        ros=Ros2Config(),
    ),
    stop=stop,
)
```

`run` blocks until cancellation, a wall-time deadline measured after
spawning, or a failure. `max_seconds=0` has no run-time limit. The owned
container has a unique `usim-gazebo-...` name and contains Xvfb, gzserver,
spawn_entity, and (for smoke) the ROS probe. Spawning must complete successfully.
Unexpected exits propagate as errors. Native process groups receive termination
and then a bounded forced kill; the host always removes its container and
staging directory. `ros=None` omits the drive and camera ROS plugins and public
motor interface while still running physics and any requested camera sensor.

Headless cameras use an actual Xvfb display and software OpenGL; the port never
silently turns off cameras. Non-headless use requires a reachable `DISPLAY` and,
on Linux, access to the X11 socket. GUI use on Windows was not verified.

## Assets and control

Supply one SDF world and a physical URDF with the configured base and two
rotating wheel joints. The port does not invent mass, collision geometry, wheel
axes, or friction for arbitrary robots. The generated primitive robot in the
smoke script is one known-working example.

The source files are immutable. Only a temporary URDF receives the diff-drive
and depth camera plugins. Existing robot plugins are stripped so an old
controller cannot bypass the motor gate; geometry and other Gazebo extensions
are preserved. Wheel diameter is twice the configured radius, wheel separation
and joint names are explicit, and odometry uses Gazebo world truth, the `odom`
frame, and the configured base frame.

The public command topic goes only to a native `CommandGate`. The diff-drive
plugin receives a unique private topic, never public `/cmd_vel`. Motors start
disabled. Every motor toggle clears stored commands; disabling publishes zero
immediately. The gate publishes at 50 Hz, clamps to 0.4 m/s and 1 rad/s, and
publishes zero after the shared 0.5-second wall-clock command watchdog expires.
The motor service uses `std_srvs/SetBool`.

Color and depth are real Gazebo depth-camera outputs: `sensor_msgs/Image` with
`rgb8` and `32FC1` (metres), the configured dimensions/rate/offset, and the
configured ROS topics. The optical frame is `usim_camera_optical_frame`.

Relative mesh and world file references are resolved against their original
source files, rewritten for the container, and mounted read-only. Relative
resource trees retain sibling texture directories. `package://` references
resolve from an enclosing matching `package.xml`, `AMENT_PREFIX_PATH`, or
`ROS_PACKAGE_PATH`; unresolved packages fail before Docker is launched.
`model://` world includes remain Gazebo-native and must exist in the image or
next to the supplied world. Automatic online model downloads are disabled.
Absolute mesh files preserve their containing directory; additional dependencies
outside that directory should use a package or relative resource tree.

Docker uses host networking for ROS. External clients need compatible DDS
networking and `ROS_DOMAIN_ID`; Docker Desktop host networking may need enabling.
The smoke probe runs inside the same container, avoiding both host DDS and Python
ABI assumptions. Run one Gazebo session at a time per network namespace: the
native Gazebo master and default ROS topics are shared.

## Verification and evidence

```sh
PYTHONPATH=src python -m pytest -q test/test_gazebo.py test/test_simulation.py
```

The smoke subscribes before commanding and waits for actual odometry/image
updates with bounded timeouts. It proves disabled immobility, enable/forward,
turn, disable/settle, watchdog/settle, advancing simulation time, and meaningful
RGB/depth payloads. There are no fixed startup or motion sleeps.

Artifacts: `smoke.json`, `trajectory.json`, `rgb.ppm`, and little-endian
`depth.f32`. A failed probe exits nonzero; it never writes a passing report.
For reproducibility, retain the generated `robot.urdf`, `pilot.sdf`, and stdout.

On 2026-10-01 the real Docker Desktop/Gazebo 11.10.2 run passed with the default
primitive robot and with different joint/base names, radius 0.11 m, separation
0.38 m, custom ROS topics, and a 160x120 camera. The default run advanced from
0.466 to 4.956 simulation seconds, moved 0.1514 m forward and turned 0.3560 rad;
disabled drift was below 0.8 mm. RGB was 230400 bytes and depth was 307200 bytes
at 320x240. Separate event-driven lifecycle exercises verified `ros=None`
deadlines, cancellation, and injected gzserver failure with container removal.

`scripts/smoke_mobile.py --backend isaac` exercises the same ROS probe in the
Isaac runtime. The reusable `usim.bridges.smoke.probe(config, out_dir)` uses only the
shared `Ros2Client` and simulation contracts, not Gazebo topics.
