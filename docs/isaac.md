# Isaac Sim port

The port uses Isaac Sim 6.1.0.0 and Python 3.12. The SDK and its dependencies
are optional; the base library and CLI help do not import them.

## Setup

```sh
uv sync --python 3.12 --extra isaac --group dev --frozen
uv run --python 3.12 --extra isaac usim --help
```

NVIDIA's [installation guide](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html)
contains supported hardware, drivers and license terms. Set
`OMNI_KIT_ACCEPT_EULA=YES` after accepting those terms.

Standalone USD conversion can use the `usd` extra instead. The `usd` and
`isaac` extras use incompatible NumPy versions and must not be combined.
Select the desired extra again when switching environments.

## Docker

`docker/Dockerfile.isaac` extends NVIDIA's Isaac Sim 6.1.0 container with the
usim source and bundled demonstration worlds. Build it from the repository
root:

```sh
docker build -f docker/Dockerfile.isaac -t usim-isaac:6.1 .
```

NVIDIA requires `ACCEPT_EULA=Y` when the container starts; set it only after
reviewing and accepting the license terms linked from the
[Isaac Sim container guide](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_container.html).
The image does not bake in EULA acceptance or opt in to telemetry. Run a
headless simulation with the NVIDIA Container Toolkit installed and a compatible
GPU:

```sh
docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -v "$PWD/assets:/workspace/usim/assets" \
  usim-isaac:6.1 simulate \
  --world /workspace/usim/worlds/corridor.usd \
  --robot-urdf /workspace/usim/assets/mobile.urdf \
  --headless --no-ros --max-seconds 10
```

The example assumes the USD world and URDF already exist in the mounted
`assets/` directory. Keep the mount writable if the run writes contact reports
or other outputs there. Isaac's container supports Python apps headlessly;
GPU access requires the host driver and NVIDIA Container Toolkit.

## Worlds and robots

```sh
uv run --python 3.12 --extra isaac usim convert-world \
  --world examples/worlds/corridor.world --out assets/corridor.usd
uv run usim create-robot --out assets/mobile.urdf
uv run --python 3.12 --extra isaac usim simulate \
  --world assets/corridor.usd --robot-urdf assets/mobile.urdf \
  --max-seconds 10 --contact-out runs/contacts.json
```

The converter handles only matching static visual/collision boxes. For other
geometry, supply a collidable, metre-scale Z-up USD world. URDF inputs need
physical collision/inertia and the configured rotating wheel joints. They are
imported into a temporary native asset directory without changing the source.
Generated robots use sphere wheel colliders, since imported cylinder colliders
sink and hop on the ground plane. Their body centre of mass sits between axle
and caster so the chassis doesn't rock forward.

The port configures wheel velocity drives, creates an RGB/depth camera at the
configured body offset, and publishes world-source state through the ROS bridge.
Contact reports count obstacle onsets, excluding ground and self contact.
The engine's scene paths remain optional `IsaacSimulator` constructor settings.

Skid robots can select synchronized groups with
`SimulationConfig(left_joints=('front_left', 'rear_left'),
right_joints=('front_right', 'rear_right'), ...)`, or CLI
`--left-joints front_left rear_left --right-joints front_right rear_right`.
Groups override the corresponding singular joint options; `None` retains their
one-joint defaults. Groups must be nonempty, unique, disjoint and equal in size.
Each selected imported joint must exist exactly once and have a velocity drive
and a distinct articulation index. Every joint on a side receives the same
angular speed computed from the configured radius and separation. Joint axes
must agree with that direction. Reports retain `wheel_velocities` and
`peak_wheel_velocities`, ordered as the left group followed by the right group;
the default configuration still reports two values.

```python
from pathlib import Path
from usim import SimulationConfig
from usim.ports.isaac import IsaacSimulator

IsaacSimulator().run(
    SimulationConfig(
        world=Path('assets/corridor.usd'),
        robot_urdf=Path('assets/mobile.urdf'),
        max_seconds=10,
        ros=None,
    )
)
```

## Worker process

### Indexed world residency

For a large static world, construct
`IsaacSimulator(world_index=Path('assets/world-index.json'), residency_radius=60.0)`.
The JSON index is a list of `{"path": "/World/...", "min": [x, y, z],
"max": [x, y, z]}` entries for independently activatable payload roots. Bounds
remain in the original metre-scale world coordinates; `/World` in each path maps
to `environment_prim_path`. Supply a Z-up USD wrapper that sublayers the complete
original world, retains its default prim and original asset references, and authors
every indexed payload inactive. This avoids an eager full-world load when the
wrapper is referenced. Do not nest indexed payload roots under another inactive
indexed root.

Before physics reset, the worker activates and loads payloads intersecting the
spawn's XY neighborhood. Between physics steps it updates after 5 metres of travel
(less for small radii), retaining existing payloads for an additional 10 metres
to avoid boundary churn. Only activation changes; source geometry, materials and
collision remain intact, and LiDAR still queries the live PhysX scene. No live
`Unload` is used. Choose a radius exceeding the sensor range plus the movement
threshold, and enough ahead of the robot for native collision cooking. Native
runtime validation is still required for a particular world's collider cooking
and GPU budget. A visible viewport follows the robot in this mode. Omitting
`world_index` preserves the ordinary eager world and fixed spawn view.

`IsaacSimulator.run` doesn't load the SDK in the calling process. It writes
`configuration.json` into a private temporary directory, starts
`python -u -m usim.ports.isaac.runner <configuration.json>` with the port's
`python` interpreter, and removes the directory on exit.

The worker's stdin is `DEVNULL`. On Windows, a pending pipe read blocked CRT
stdio setup inside native DLL loads and hung the SDK import. Cancellation
therefore works through a file: setting the `stop` event creates `stop` in
the temporary directory, and the worker polls for it.

The worker prints progress markers: `USIM_ISAAC_IMPORT`,
`USIM_ISAAC_IMPORTED`, `USIM_ISAAC_APP_READY`, `USIM_ISAAC_ROBOT_IMPORTED` and
`USIM_ISAAC_RUNNING`. If the SDK import stalls for 120 s, it dumps every
thread's traceback. The supervisor enforces these deadlines and kills the
worker when one expires:

| Phase | Limit |
| --- | --- |
| Initialization, until `USIM_ISAAC_RUNNING` | 300 s |
| Run, when `max_seconds` is set | `max_seconds` + 15 s |
| Shutdown after a result line | 30 s |
| Cancellation after the stop file | 30 s |

A run succeeds only when the worker exits with code zero and its last JSON
result line has status `finished` or `stopped`. Anything else raises
`RuntimeError`.

## ROS and rendering

Use compatible Python 3.12 ROS modules, not a sourced Python 3.10 Humble install.
The loader accepts a sourced ROS environment or the SDK's bundled Humble
modules located through `ISAAC_PATH`. The worker's SDK import sets it; a
separate ROS-only process sets it from the installed package location
(`importlib.util.find_spec('isaacsim')`) without starting the SDK.
Bundled ROS Python is appended to `sys.path`, not prepended, so the
environment's NumPy wins over the copy shipped with the bundle.
On Linux, add the SDK's
`exts/isaacsim.ros2.core/humble/lib` directory to `LD_LIBRARY_PATH` before
starting Python. Windows retains an explicit DLL-directory handle.

`--no-ros` omits application ROS imports; `--physics-only` omits the camera.
Visible mode renders every step, including `--physics-only`, adds a dome light
when the world has no lights, and points the viewport at the robot spawn.
Headless mode renders only camera captures. Every rendered camera frame is read
before another render; publishing remains limited by `camera_hz`. Reading after
unread rendered frames hits recycled buffers (CUDA error 700) in Isaac 6.1.
`--headless` requests native headless mode. Kit rendering is platform-dependent;
if its renderer fails to advance, use a visible window or Xvfb on Linux.

Optional `SimulationConfig` fields `initial_position=(x, y, z)` and
`initial_yaw` (radians) place the articulation root before reset. Omission
preserves the imported pose. `real_time=True` limits physics advancement to wall
time, with cancellable waits; slow frames do not skip physics steps.
The CLI equivalents are `--initial-position X Y Z --initial-yaw R --real-time`.

`lidar_enabled=True` enables instantaneous planar rays against live PhysX
colliders, not a map approximation. `lidar_offset` is a physical translation in
the reported body's frame; rays follow its full orientation. The defaults are
360 samples from `-pi` to `pi`, 10 Hz of simulation time, and ranges 0.05 to
30 metres. Configure `lidar_samples`, `lidar_angle_min`, `lidar_angle_max`,
`lidar_hz`, `lidar_min_range`, and `lidar_max_range` in `SimulationConfig`.
Self hits are excluded; the nearest external hit is selected, with infinity for
missing or too-close returns. `Ros2Config(scan_topic='/scan_raw')` selects the
topic, and `IsaacSimulator(lidar_frame='lidar_link')` selects the frame.
The ROS bridge publishes the matching body-to-LiDAR transform with each scan.
When robot_state_publisher owns the sensor chain, set
`Ros2Config(publish_lidar_tf=False)` to avoid giving `lidar_link` a second parent.
Scans use sensor-data best-effort QoS.

Set `SimulationConfig.lidar_vertical_angles` to a nonempty tuple of elevation
angles in radians (each between `-pi/2` and `pi/2`) to select physical 3D point
mode. Its default `None` preserves planar scans. Each elevation casts
`lidar_samples` azimuth rays at `lidar_hz`; all rays and the mounting offset
follow the body's full orientation. Only finite, in-range nearest external
PhysX hits become sensor-frame XYZ points, including terrain returns. Nothing
is ground-masked or substituted from a map.

Point mode publishes `sensor_msgs/PointCloud2` on
`Ros2Config(points_topic='/lidar/points')`, using XYZ float32 little-endian
fields and sensor-data best-effort QoS. It does not create a LaserScan
publisher, leaving downstream point-cloud filtering and scan conversion in
control of `/scan_raw`. The matching sensor transform is published before each
cloud unless `publish_lidar_tf=False`.

For example, an independently named sensor frame can feed a downstream
elevation-filter/pointcloud-to-laserscan pipeline:

```python
import math
from usim.simulation import Ros2Config

configuration = SimulationConfig(
    world=Path('assets/local-world.usd'),
    robot_urdf=Path('assets/mobile.urdf'),
    lidar_enabled=True,
    lidar_vertical_angles=(0.0, math.radians(10), math.radians(20)),
    lidar_samples=360,
    lidar_offset=(0.144, 0.0, 0.275),
    ros=Ros2Config(points_topic='/livox/lidar', publish_lidar_tf=True),
)
IsaacSimulator(lidar_frame='sim_mid360_link').run(configuration)
```

```python
IsaacSimulator().run(
    SimulationConfig(
        world=Path('assets/local-world.usd'),
        robot_urdf=Path('assets/raspicat.urdf'),
        base_link='base_footprint',
        wheel_radius=0.0762,
        wheel_separation=0.27918,
        initial_position=(300.0, -190.0, 0.2),
        initial_yaw=0.0,
        real_time=True,
        camera_enabled=False,
        lidar_enabled=True,
        lidar_offset=(0.144, 0.0, 0.105),
    )
)
```

The spawn height and LiDAR offset must match the imported articulation root,
which may change when fixed URDF joints merge. The CLI exposes
`--lidar-enabled --lidar-offset X Y Z --lidar-hz HZ --lidar-samples N --scan-topic TOPIC`;
the older `--lidar-link` options still configure Gazebo only.

```sh
uv run --python 3.12 --extra isaac python scripts/smoke_mobile.py \
  --backend isaac --out-dir runs/isaac-smoke
```

This exercises the same event-driven ROS probe as Gazebo: disabled motion,
enable/forward/turn, disable, watchdog, timestamps and real RGB/depth. Inspect
`smoke.json`, `trajectory.json`, `rgb.ppm`, `depth.f32` and contacts.
Do not accept native exit code alone: Kit can terminate with code zero after an
execution error, so a fresh passing report and sensor artifacts are required.
