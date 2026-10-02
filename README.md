# usim

Mobile robot simulation with a dependency-free Python core, a standard ROS 2
bridge, and Gazebo Classic / Isaac Sim ports.

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
See [Gazebo](docs/gazebo.md), [Isaac](docs/isaac.md), and
[the shared contract](docs/architecture.md).

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
  --world worlds/corridor.world --robot-urdf assets/mobile.urdf \
  --wheel-radius 0.09 --wheel-separation 0.36 --camera-offset 0.15 0 0.35
```

Isaac uses collidable, metre-scale Z-up USD:

```sh
uv run --python 3.12 --extra isaac usim convert-world \
  --world worlds/corridor.world --out assets/corridor.usd
uv run --python 3.12 --extra isaac usim simulate \
  --world assets/corridor.usd --robot-urdf assets/mobile.urdf \
  --wheel-radius 0.09 --wheel-separation 0.36 --camera-offset 0.15 0 0.35
```

External URDFs are accepted directly. Wheel joints, geometry, camera dimensions,
offset and ROS topics are explicit; inspect `usim simulate --help`. Authoring
and drive dimensions must agree. The box-world converter intentionally rejects
unsupported SDF geometry rather than dropping it.

## Python library

```python
from pathlib import Path
from usim import SimulationConfig
from usim.ports.gazebo import GazeboSimulator

GazeboSimulator().run(
    SimulationConfig(
        world=Path('worlds/corridor.world'),
        robot_urdf=Path('assets/mobile.urdf'),
        headless=True,
        max_seconds=30,
    )
)
```

Both ports implement `run(configuration, stop=event)` and own their lifecycle.
The core exports robot geometry, immutable configuration, velocity/state data,
wheel math and motor gating. There is no fictitious lockstep `step()` API.

Motion is controlled through standard ROS 2: `/cmd_vel`, `/motor_power`
(`std_srvs/SetBool`), `/odom`, `/clock`, RGB and depth images. Motors start
**disabled**; every motor toggle clears commands and the wall-clock watchdog
expires after 0.5 seconds. `usim.bridges.ros2.Ros2Client` provides bounded
readiness, motor requests, velocity commands and state observation in a
compatible ROS environment. `ros=None` runs passive physics/camera without
application ROS interfaces.

## Layout and smoke checks

- `src/usim/`: pure contracts and robot authoring.
- `src/usim/bridges/`: optional standard ROS I/O.
- `src/usim/ports/`: native engine execution.
- `worlds/`: small owned SDF demonstration worlds.
- `test/` and `scripts/`: regressions and real control/sensor smokes.

```sh
uv run python scripts/smoke_mobile.py --backend gazebo --out-dir runs/gazebo-smoke
uv run --python 3.12 --extra isaac python scripts/smoke_mobile.py \
  --backend isaac --out-dir runs/isaac-smoke
```

Runtime assets and results under `assets/` and `runs/` are ignored. Supply
resource paths explicitly. Application policies and evaluation belong to
consumers. Source code is MIT licensed; native dependencies retain their licenses.
