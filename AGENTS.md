# Repository Guidelines

usim is a general-purpose mobile robot simulator.
`src/usim/` contains simulator-independent contracts and tooling;
`src/usim/ports/` owns Gazebo/Isaac execution and `src/usim/bridges/` owns
configurable standard ROS interfaces. Application-specific policies, evaluation,
robot descriptions and inference integrations belong to downstream consumers.
Keep the public core dependency-free. Share configuration, velocity/state data
and motor gating, but do not invent lockstep `step()` semantics across engines.

Use Python 3.12. Run `uv sync --group dev`, `uv run ruff check .`,
`uv run ruff format --check .`, `uv run pytest -ra`, and `uv build`.
Isaac and USD dependencies are optional and mutually exclusive; use the
matching extra documented in `docs/isaac.md`. Keep versions pinned.

Tests live in `test/`. Read existing tests before refactoring and preserve
ROS interfaces and physics semantics.
Python lines are limited to 99 characters; follow existing formatting.

Worlds are checkout resources. Resolve them through explicit arguments or
`USIM_ROOT`; never infer a consumer checkout from this package's parent directories.
Runtime `assets/`, `runs/`, weights and downloaded upstream data are not
committed.

Use concise Conventional Commits. Keep the simulator independent of application
inference dependencies.
