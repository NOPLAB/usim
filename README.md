# usim

Mobile robot and manipulator simulation with a dependency-free Python core,
standard ROS 2 bridges, and Gazebo Classic, Isaac Sim, Genesis and ManiSkill.

`usim` owns the robot-neutral interfaces, configuration, CLI and bundled
CRANE-X7 model. Engines are separately installable workspace packages:
`usim_genesis`, `usim_isaacsim`, `usim_maniskill` and `usim_gazebo`.
See [engine setup and simulation workflows](docs/engines.md).

## Install and check

The library supports Python 3.10-3.12. Isaac Sim 6.1 requires Python 3.12.

```sh
uv sync --python 3.12 --group dev --frozen
uv run ruff check .
uv run ruff format --check .
uv run pytest -ra
uv build
```

The base wheel has no runtime dependencies. Native engines and ROS are optional.
`usim backends` lists installed providers and their execution capabilities
without initializing native engines.
See [Gazebo](docs/gazebo.md), [Isaac](docs/isaac.md), and
[the shared contract](docs/architecture.md).
For an optional downloaded world, see [Virtual Tsukuba Challenge](docs/vtc.md).

## Run a robot

Create a mesh-free physical differential-drive robot:

```sh
uv run usim create-robot --out assets/mobile.urdf --name delivery_robot \
  --wheel-radius 0.09 --wheel-separation 0.36 \
  --body-size 0.5 0.28 0.12 --mass 8 --camera-offset 0.15 0 0.35
```

Gazebo uses an SDF world and an isolated native Humble runtime:

```sh
docker build -f docker/Dockerfile.gazebo -t usim-gazebo:local .
uv run usim simulate --backend gazebo --headless \
  --world examples/worlds/corridor.world --robot-urdf assets/mobile.urdf \
  --wheel-radius 0.09 --wheel-separation 0.36 --camera-offset 0.15 0 0.35
```

Isaac uses collidable, metre-scale Z-up USD:

```sh
uv run --python 3.12 --extra isaac usim convert-world \
  --world examples/worlds/corridor.world --out assets/corridor.usd
uv run --python 3.12 --extra isaac usim simulate \
  --world assets/corridor.usd --robot-urdf assets/mobile.urdf \
  --wheel-radius 0.09 --wheel-separation 0.36 --camera-offset 0.15 0 0.35
```

For an Isaac Sim 6.1 container, build `docker/Dockerfile.isaac` from the
repository root. See [Isaac Docker setup](docs/isaac.md#docker) for GPU and
runtime license requirements.

External URDFs are accepted directly. Wheel joints, geometry, camera dimensions,
offset and ROS topics are explicit; inspect `usim simulate --help`. Authoring
and drive dimensions must agree. The box-world converter intentionally rejects
unsupported SDF geometry rather than dropping it.

For license-audited robot sources and additional indoor/outdoor worlds, see
[robots](docs/robots.md) and [worlds](docs/worlds.md). The
[Raspberry Pi Cat VTC SLAM/Nav2 example](examples/vtc_navigation/README.md)
shows how to map and navigate a fetched VTC world.

## Python library

```python
from pathlib import Path
from usim import SimulationConfig
from usim.ports.gazebo import GazeboSimulator

GazeboSimulator().run(
    SimulationConfig(
        world=Path('examples/worlds/corridor.world'),
        robot_urdf=Path('assets/mobile.urdf'),
        headless=True,
        max_seconds=30,
    )
)
```

Continuous runners implement `run(configuration, stop=event)` and own their lifecycle.
The core exports robot geometry, immutable configuration, velocity/state data,
wheel math and motor gating. Mobile engines do not expose a fictitious lockstep
`step()` API. Native episode providers expose `reset`, `step`, observation and
`close` through the same `usim` namespace. These are engine capabilities, not
separate mobile/arm categories. One external articulation can contain wheels,
an arm and a gripper.

Motion is controlled through standard ROS 2: `/cmd_vel`, `/motor_power`
(`std_srvs/SetBool`), `/odom`, `/clock`, RGB and depth images. Motors start
**disabled**; every motor toggle clears commands and the wall-clock watchdog
expires after 0.5 seconds. `usim.bridges.ros2.Ros2Client` provides bounded
readiness, motor requests, velocity commands and state observation in a
compatible ROS environment. `ros=None` runs passive physics/camera without
application ROS interfaces.

## Layout and smoke checks

- `src/usim/`: pure contracts, factory, CLI, bridges and robot descriptions.
- `src/usim/bridges/`: optional standard ROS I/O.
- `packages/{genesis,isaacsim,maniskill,gazebo}/`: independent engine distributions.
- `src/usim/ports/`: lightweight public compatibility facades for existing mobile clients.
- `examples/worlds/`: small owned SDF demonstration worlds.
- `test/` and `scripts/`: regressions and real control/sensor smokes.

```sh
uv run python scripts/smoke_mobile.py --backend gazebo --out-dir runs/gazebo-smoke
uv run --python 3.12 --extra isaac python scripts/smoke_mobile.py \
  --backend isaac --out-dir runs/isaac-smoke
```

Runtime assets and results under `assets/` and `runs/` are ignored. Supply
resource paths explicitly. Application policies and evaluation belong to
consumers. Source code is MIT licensed; native dependencies retain their licenses.
