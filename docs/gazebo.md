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

The host library supports Python 3.10-3.14. Humble's `rclpy` runs under the container's
Python 3.10, **not** the host interpreter. Each invocation explicitly copies
`usim/__init__.py`, `robot.py`, `simulation.py`, package initializers, the two
ROS bridge source modules, and the Gazebo port into a temporary read-only mount.
It does not install the host's wheel in the image. This also means source
changes do not require an image rebuild. The image supplies native dependencies.

## Library use

```python
from pathlib import Path
from threading import Event
from usim.simulation import SimulationConfig, Ros2Config
from usim_gazebo import GazeboSimulator

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

`GazeboSimulator(startup_timeout=90.0)` bounds world loading, robot spawning,
and the first odometry message. For a large collision world, set a larger
finite positive deadline; the CLI forwards `--startup-timeout` to this same
setting and the native `spawn_entity` service wait. This does not extend
`max_seconds`, which starts after spawning.

### Container engine

`GazeboSimulator(image, *, engine='docker')` drives the `docker` CLI by default.
Pass `engine='podman'` to issue the same `create`, `start --attach`, `stop`,
`ps` and `rm --force` calls through `podman`; the parameter is typed as
`Literal['docker', 'podman']`. The image must be built or pulled with the selected engine. Podman runs use the
same host networking, read-only bind mounts and `--init`; hosts that enforce
SELinux labels on bind mounts were not verified.

### WSLg GPU rendering

For a GUI on a WSLg Docker host, `GazeboSimulator(wslg_gpu=True)` or the installed
CLI's `--wslg-gpu` passes `/dev/dxg` and the host's `/usr/lib/wsl` libraries to
the container. This requires `headless=False` and the existing `DISPLAY`
connection. Missing host devices or libraries are engine errors, not a silent
software-rendering fallback. Set `MESA_D3D12_DEFAULT_ADAPTER_NAME` to select an
adapter, for example `NVIDIA`; the value is forwarded only in this mode.

### ROS transport configuration

The adapter forwards explicitly set `ROS_DOMAIN_ID`, `ROS_LOCALHOST_ONLY`,
`RMW_IMPLEMENTATION`, and `FASTDDS_BUILTIN_TRANSPORTS` to native ROS.
`network=` selects the container network; an application can put its stack and
the simulator on the same owned bridge without changing the simulator core.

For Fast DDS, pass `fastdds_profile=Path('udp_only.xml')` to `GazeboSimulator`.
The caller's XML is copied unchanged into the private read-only staging mount
and selected through `FASTRTPS_DEFAULT_PROFILES_FILE` inside the container.
This works with Humble versions that do not recognize
`FASTDDS_BUILTIN_TRANSPORTS`. Profile contents and paths to additional resources
inside that XML are the caller's native container configuration. An unreadable
profile fails before an engine call. Omitting it leaves the middleware defaults
unchanged; the public core gains no ROS or middleware dependency.

### Lidar

```python
from usim_gazebo import GazeboLidarConfig, GazeboSimulator

simulator = GazeboSimulator(
    'usim-gazebo:local',
    engine='podman',
    lidar=GazeboLidarConfig(link_name='laser_link', frame_name='laser_link', topic='/scan'),
)
```

`GazeboLidarConfig(link_name, frame_name, topic, update_rate=10.0,
horizontal_samples=720, min_angle=-pi, max_angle=pi, range_min=0.1,
range_max=10.0, vertical_samples=1, min_vertical_angle=0.0,
max_vertical_angle=0.0, output_type='LaserScan')` adds a physical Gazebo `ray` sensor named `usim_lidar`
to the temporary URDF after source plugins are stripped. The sensor is attached
to `link_name`, which must exist in the URDF; place a dedicated link to control
its mounting pose. With ROS enabled, a `libgazebo_ros_ray_sensor.so` plugin
publishes `sensor_msgs/LaserScan` on the absolute `topic` with header frame
`frame_name`. `ros=None` keeps the ray sensor but omits the ROS plugin. The
lidar is independent of `camera_enabled`. Construction raises `ConfigurationError` for blank names,
relative or whitespace topics, non-finite values, a nonpositive rate, fewer than
one sample, `max_angle <= min_angle`, a span above 2*pi, a nonpositive
`range_min`, and `range_max <= range_min`. Range resolution is 0.01 m; no noise
model is added. Caller URDF and world files are never modified.

For a 3D cloud, select `output_type='PointCloud2'`, `vertical_samples >= 2`
and increasing vertical bounds in radians within `[-pi/2, pi/2]`. More than
one vertical row requires PointCloud2 so the LaserScan plugin cannot silently
discard elevation rows. A single row requires both vertical bounds to remain
zero; the omitted `<vertical>` element preserves existing planar behavior.
Both sample counts must be positive integers (not booleans).
The native ray plugin publishes `sensor_msgs/PointCloud2` directly, with
FLOAT32 `x`, `y`, `z`, and `intensity`. XYZ comes from actual ranges and each
ray's azimuth/elevation, not a planar scan padded with zero Z. Intensity is
Gazebo's native return value, not calibrated reflectivity. Invalid ranges are
omitted, so the cloud is unordered (`height=1`) and its point count can be less
than `horizontal_samples * vertical_samples`. The selected frame must match
the sensor link axes; its mounting transform remains caller-owned.

For example, downstream code can approximate the Mid-360 nominal 360-degree
horizontal and -7 to 52-degree vertical field of view using this generic grid:

```sh
usim simulate --backend gazebo --world room.sdf --robot-urdf robot.urdf \
  --headless --physics-only \
  --lidar-link livox_frame --lidar-frame livox_frame --lidar-topic /livox/lidar \
  --lidar-output-type PointCloud2 --lidar-horizontal-samples 720 \
  --lidar-min-angle -3.141592653589793 --lidar-max-angle 3.141592653589793 \
  --lidar-vertical-samples 32 \
  --lidar-min-vertical-angle -0.12217304763960307 \
  --lidar-max-vertical-angle 0.9075712110370514
```

This uniformly sampled ray grid is not Livox's nonrepetitive scan pattern,
per-point timing, motion distortion, or IMU stream. Row count, rate, range,
topic and frame are caller choices, not a built-in device preset. Defaults
remain the existing planar LaserScan. Portable tests verify generated sensor
geometry and CLI forwarding; real Gazebo QA must verify finite, nonplanar
returns against scene geometry and the native image's plugin version.

Headless cameras use an actual Xvfb display and software OpenGL; the port never
silently turns off cameras. Non-headless use requires a reachable `DISPLAY` and,
on Linux, access to the X11 socket. GUI use on Windows was not verified.

## Assets and control

Supply one SDF world and a physical URDF with the configured base and
rotating wheel joints. The port does not invent mass, collision geometry, wheel
axes, or friction for arbitrary robots. The generated primitive robot in the
smoke script is one known-working example.

The source files are immutable. Only a temporary URDF receives the diff-drive
and depth camera plugins. Existing robot plugins are stripped so an old
controller cannot bypass the motor gate; geometry and other Gazebo extensions
are preserved. Wheel diameter is twice the configured radius, wheel separation
and joint names are explicit, and odometry uses Gazebo world truth, the `odom`
frame, and the configured base frame.

For skid steering, set `left_joints=('front_left', 'rear_left')` and
`right_joints=('front_right', 'rear_right')` in `SimulationConfig`, or pass
`--left-joints front_left rear_left --right-joints front_right rear_right` to
`usim simulate --backend gazebo`. Groups override the corresponding singular
`left_joint`/`right_joint` fields and CLI flags; omitting them keeps the original
one-wheel-per-side behavior. Each group must be nonempty, contain unique nonblank
names, be disjoint from the other side, and have the same number of joints.
The port emits `num_wheel_pairs` and repeated left/right joints, diameter and
separation for each pair. All wheels use the configured radius and separation.
Source joint axes must give the same forward rotation direction on each side.

The public command topic goes only to a native `CommandGate`. The diff-drive
plugin receives a unique private topic, never public `/cmd_vel`. Motors start
disabled. Every motor toggle clears stored commands; disabling publishes zero
immediately. The gate publishes at 50 Hz, clamps to 0.4 m/s and 1 rad/s, and
publishes zero after the shared 0.5-second wall-clock command watchdog expires.
The motor service uses `std_srvs/SetBool`.
The watchdog is not configurable in this port. A planner producing commands at
1 Hz must use a downstream command publisher or velocity smoother that refreshes
the configured command topic more frequently than every 0.5 seconds; for
example, a 20 Hz smoother. Its own input timeout must cover the planner cadence
and expected jitter. Connecting a 1 Hz planner directly causes zero commands
between updates; changing lidar settings does not change that motor contract.

Color and depth are real Gazebo depth-camera outputs: `sensor_msgs/Image` with
`rgb8` and `32FC1` (metres), the configured dimensions/rate/offset, and the
configured ROS topics. The optical frame is `usim_camera_optical_frame`.

Relative mesh and world file references are resolved against their original
source files, rewritten for the container, and mounted read-only. Relative
resource trees retain sibling texture directories. `package://` references
resolve from an enclosing matching `package.xml`, `AMENT_PREFIX_PATH`, or
`ROS_PACKAGE_PATH`; unresolved packages fail before Docker is launched.
`model://` world includes remain Gazebo-native and must exist in the image or
in a local `models/` directory adjacent to the world's parent directory;
these model roots are mounted and added to `GAZEBO_MODEL_PATH`. Automatic
online model downloads are disabled.
Absolute mesh files preserve their containing directory; additional dependencies
outside that directory should use a package or relative resource tree.

The container uses host networking for ROS. External clients need compatible DDS
networking and `ROS_DOMAIN_ID`; Docker Desktop host networking may need enabling.
When set on the host, `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION` and
`FASTDDS_BUILTIN_TRANSPORTS` (for example `UDPv4` to avoid shared-memory
transport across the container boundary) are forwarded into the container. The
last one takes effect only if the image's Fast DDS release honors it.
The smoke probe runs inside the same container, avoiding both host DDS and Python
ABI assumptions. Run one Gazebo session at a time per network namespace: the
native Gazebo master and default ROS topics are shared.

## Verification and evidence

```sh
uv run --no-sync pytest -q test/test_gazebo_lidar.py test/test_runtime_cli.py test/test_simulation.py
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
