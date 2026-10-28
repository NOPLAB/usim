# Simulator containers

Build from the repository root. Each engine has a standalone Dockerfile;
`Dockerfile.all` contains the SDKs in isolated Python environments.

| Dockerfile | Runtime |
| --- | --- |
| `Dockerfile.genesis` | Genesis, Python 3.12 |
| `Dockerfile.maniskill` | ManiSkill, Python 3.12 |
| `Dockerfile.bullet` | Bullet CPU, Python 3.12 |
| `Dockerfile.mujoco` | MuJoCo CPU / Warp CUDA, Python 3.12 |
| `Dockerfile.isaac` | Official Isaac Sim image and bundled Python |
| `Dockerfile.gazebo` | ROS 2 Humble / Gazebo worker |
| `Dockerfile.all` | Isaac plus independent Genesis, ManiSkill, Bullet, MuJoCo, MuJoCo Warp and core environments |

```sh
docker build -f docker/Dockerfile.genesis -t usim-genesis:local .
docker build -f docker/Dockerfile.maniskill -t usim-maniskill:local .
docker build -f docker/Dockerfile.bullet -t usim-bullet:local .
docker build -f docker/Dockerfile.mujoco -t usim-mujoco:local .
docker build -f docker/Dockerfile.all -t usim-all:local .
```

Genesis and ManiSkill install the local core wheel and their own backend's
`native` extra, not the other engine or development dependencies. SDK versions
are pinned by the package manifests. These recipes resolve transitive dependencies
from those manifests rather than synchronizing the combined development workspace
lock. Native dependencies, including PyTorch and CUDA, make these large downloads.
The ManiSkill images use the same tested Torch 2.9.1 pin as Genesis. Builds use
two concurrent downloads, one installation/build at a time, and a shared uv
cache. PyBullet's source compiler invocations are serialized to limit memory use.

For large SDK images, use a dedicated builder with a 4 GiB RAM limit and two CPUs:

```sh
docker buildx create --name usim-bounded --driver docker-container \
  --driver-opt memory=4g,memory-swap=4g,cpu-period=100000,cpu-quota=200000
docker buildx build --builder usim-bounded --load \
  -f docker/Dockerfile.all -t usim-all:local .
docker buildx stop usim-bounded
```

Build one image at a time. The limit applies to the builder container, not to
other running containers or the Docker Engine image store.

If image loading overlaps with too much builder memory on WSL, export and load
in separate steps instead of using `--load`. Allow about 20 GiB for the archive:

```sh
docker buildx build --builder usim-bounded \
  --output type=docker,dest=usim-all.tar \
  -f docker/Dockerfile.all -t usim-all:local .
docker buildx stop usim-bounded
docker load --input usim-all.tar
```

## All-in-one image

The all-in-one image uses the official Isaac runtime and separate environments
at `/opt/usim/genesis`, `/opt/usim/maniskill`, `/opt/usim/bullet`,
`/opt/usim/mujoco`, `/opt/usim/mujoco-warp` and `/opt/usim/core`. It does not
combine their conflicting NumPy, Pillow or native SDK dependencies. Its launcher
parses the existing CLI and replaces itself with the selected interpreter;
no extra engine-selection argument is needed:

```sh
docker run --rm usim-all:local backends
docker run --rm \
  -e PYGLET_HEADLESS=1 -e PYOPENGL_PLATFORM=egl \
  --mount type=bind,src=/absolute/path/to/models,dst=/models,readonly \
  usim-all:local run --engine genesis --compute cpu \
  --robot /models/robot.urdf --joint-names hinge \
  --render-mode none --obs-mode state --steps 0
docker run --rm --gpus all -e ACCEPT_EULA=Y \
  --mount type=bind,src=/absolute/path/to/models,dst=/models,readonly \
  usim-all:local run --engine isaacsim --compute cpu \
  --robot /models/robot.usd --joint-names hinge \
  --render-mode none --obs-mode state --steps 0
```

Use `--engine maniskill` for ManiSkill. Isaac execution and world conversion
require `ACCEPT_EULA=Y` after accepting NVIDIA's terms; discovery and help do not
initialize Isaac. The native runtime images have the same GPU/display requirements
as their standalone equivalents.

Use `--engine bullet` or `--engine mujoco` for the CPU engines.
`--engine mujoco-warp --compute gpu` selects actual CUDA physics and requires
a Slurm GPU allocation and GPU access in the container. These engines expose
episodes, not continuous ROS runners; see [setup and controls](../docs/bullet-mujoco.md).

Gazebo still runs in its separately built worker image. The all-in-one image
contains its backend and Docker client, not a nested Docker daemon. Mount the
host socket and models so the runner can create and supervise the worker:

```sh
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --mount type=bind,src=/absolute/path/to/models,dst=/models,readonly \
  usim-all:local simulate --backend gazebo --engine docker \
  --world /models/room.sdf --robot-urdf /models/robot.urdf \
  --headless --no-ros --max-seconds 5
```

The host daemon must have `usim-gazebo:local`, built with `Dockerfile.gazebo`.
The bundled client uses Docker; standalone host execution can still select Podman.
The all-in-one image selects `USIM_GAZEBO_ASSET_TRANSFER=copy`: it transfers
prepared runtime files and model resource folders into its owned worker through
the Docker API, rather than asking the host to bind paths inside the caller
container. Smoke outputs are copied back before worker removal. Resource copies
use a temporary disk-backed archive, so allow space for the referenced folders.
Direct host execution retains read-only binds by default. The public
`GazeboSimulator(asset_transfer='copy')` option also supports remote daemons.

## Native CLI images

Both images enter `python -m usim.cli` and print help without arguments:

```sh
docker run --rm usim-genesis:local
docker run --rm usim-maniskill:local run --help
docker run --rm usim-genesis:local backends
```

See [simulator selection](../docs/cli.md) for the execution modes and public
APIs used by each command. External models use the grouped model API; registered
tasks retain their native episode action space.

The following commands reset and observe without stepping or guessing an action.
Genesis needs a caller-owned model and its ordered controlled joint names. Mount
the model's meshes and other referenced files with the model:

```sh
docker run --rm \
  -e PYGLET_HEADLESS=1 -e PYOPENGL_PLATFORM=egl \
  --mount type=bind,src=/absolute/path/to/models,dst=/models,readonly \
  usim-genesis:local run --engine genesis --compute cpu \
  --robot /models/robot.urdf --joint-names hinge \
  --joint-stiffness 100 --joint-damping 10 \
  --render-mode none --obs-mode state --steps 0
```

For ManiSkill, select an upstream registered task and supported robot explicitly:

```sh
docker run --rm --gpus all \
  usim-maniskill:local run --engine maniskill --compute gpu \
  --env-id PickCube-v1 --robot-uid panda \
  --render-mode none --obs-mode state --steps 0
```

Positive `--steps` requires `--action` in the selected engine's native control
space. Robot assets, custom task registrations, and policies remain caller-owned.
Upstream tasks may download their own assets on first use.

GPU runs require a supported host GPU, driver, and NVIDIA Container Toolkit.
Use `--gpus all` for Genesis GPU execution as well. The image requests compute,
utility, and graphics driver capabilities; ManiSkill/SAPIEN also needs a working
Vulkan device and host driver ICD, even for some state-only headless tasks.
CUDA visibility alone does not establish Vulkan support, especially on WSL.
`--compute cpu` selects CPU physics; it does not promise software rendering.
For Genesis without an X display, set `PYGLET_HEADLESS=1` and
`PYOPENGL_PLATFORM=egl`, as in the CPU examples. ManiSkill/SAPIEN still creates
render materials for authored visual geometry even with `--render-mode none`;
models with visuals therefore need a working rendering device. A collision-only
URDF can exercise CPU physics without one.
The recipes do not configure a desktop display or run native GPU computations
during the build. Build-time help and dependency checks do not prove native
reset/action/render execution.

## Gazebo worker

`Dockerfile.gazebo` is the existing ROS 2 Humble / Gazebo Classic worker image,
not one of these native CLI images. Its default command is a shell, and its
Python runtime belongs to ROS Humble rather than the Python 3.12 native
environment. Build it separately:

```sh
docker build -f docker/Dockerfile.gazebo -t usim-gazebo:local .
```

Run the installed host-side `usim simulate --backend gazebo` CLI to supervise
this worker, with explicit world and robot paths and the desired container
engine/image options. See [Gazebo](../docs/gazebo.md).

## Isaac runtime

Isaac uses `Dockerfile.isaac`, NVIDIA's official Isaac Sim image, and its bundled
Python launcher. Do not install Isaac into either standalone native environment or replace its
special runtime with a plain Python base image. Its entrypoint requires
`ACCEPT_EULA=Y` after accepting NVIDIA's terms; GPU and host setup are described
in [Isaac](../docs/isaac.md).

```sh
docker build -f docker/Dockerfile.isaac -t usim-isaac:local .
docker run --rm --gpus all -e ACCEPT_EULA=Y usim-isaac:local --help
```
