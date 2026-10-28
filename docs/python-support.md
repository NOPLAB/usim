# Python support

The dependency-free core and the six backend base packages support regular
CPython 3.10-3.14. CI runs generic CPU tests, builds all seven distributions and
exercises installed console commands on every version. Python 3.15 and
free-threaded builds are not in the supported matrix.

## Separate the host library from native SDKs

| Environment | Python | Boundary |
| --- | --- | --- |
| Core, CLI, backend discovery and contract tests | 3.10-3.14 | No native SDK is needed for discovery |
| Point-cloud pipeline and standalone USD tools | 3.10-3.14 | Version-specific CPU dependency pins |
| Genesis 1.4.3 / Torch 2.9.1 | Keep the verified 3.12 environment | 3.13 dependency resolution alone is not native certification; quadrants 1.3.3 lacks a 3.14 wheel |
| ManiSkill 3.0.1 / SAPIEN 3.0.3 | Keep the verified 3.12 environment | The pinned native stack requires NumPy <2 |
| Isaac Sim 6.1.0.0 | 3.12 | The SDK extra is conditional on Python 3.12 |
| Gazebo Humble worker | Container Python 3.10 | The host may use 3.10-3.14; ROS modules stay in the container |
| Bullet / MuJoCo native SDKs | Use separate Python 3.12 environments | Base-package portability is not native SDK certification |
| MuJoCo Warp | Use a separate Python 3.12 environment with CUDA | GPU computation requires a Slurm GPU allocation |

Base-package installation and fixture tests do not establish native physics,
rendering, GPU or ROS compatibility. Native SDK pins are unchanged by the
portable Python expansion. Do not combine pipeline dependencies with Isaac
native/USD or ManiSkill native extras; the resolver records these conflicts.

## Development and pipeline environments

On Debian/Ubuntu, install Open3D's shared-library dependencies before importing
it, including for CPU-only geometry tests:

```sh
sudo apt-get update
sudo apt-get install --yes --no-install-recommends libegl1 libgl1
```

CI installs these libraries on its Ubuntu runner. No GPU allocation or display
server is required for the geometry tests.

From a POSIX shell or Git Bash, select a separate environment explicitly:

```sh
UV_PROJECT_ENVIRONMENT=runs/dev-py314 uv sync --python 3.14 --group dev --frozen
UV_PROJECT_ENVIRONMENT=runs/dev-py314 uv run --no-sync pytest -ra

UV_PROJECT_ENVIRONMENT=runs/pipeline-py314 uv sync --python 3.14 \
  --group dev --extra pipeline --extra cad --frozen
UV_PROJECT_ENVIRONMENT=runs/pipeline-py314 uv run --no-sync pytest -ra

UV_PROJECT_ENVIRONMENT=runs/isaac-py312 uv sync --python 3.12 \
  --extra isaac --no-dev --frozen
```

In PowerShell, set `$env:UV_PROJECT_ENVIRONMENT` to the same directory before
running the corresponding `uv` command. This avoids replacing an existing
native SDK environment when switching the host Python.

The pipeline keeps the previous NumPy 1.26.4, SciPy 1.15.3, Open3D 0.19.0 and
USD 25.5.1 pins below Python 3.13. Python 3.13+ uses NumPy 2.5.4, SciPy 1.18.1,
Open3D 0.20.0 and USD 26.11; Python 3.14 uses scikit-image 0.26.0 instead of
0.25.2. CAD and native SDK versions are not upgraded. All pins are reachable
from `pyproject.toml`; the isolated pipeline requirements mirror its core
geometry dependency markers.

NumPy-derived record counts are converted to Python integers at the JSON
report boundary. Edge-quality tests compare every dataclass field separately,
so nested tuple values do not rely on pytest's dictionary approximation rules.
