"""Lazy engine discovery by installed execution capability."""

from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Callable, Literal

from usim.interface import EpisodeSimulator
from usim.simulation import Simulator
from usim.types import SimulatorConfig

_SIMULATORS: dict[str, type[EpisodeSimulator]] = {}


def _providers(group: str) -> dict[str, EntryPoint]:
    return {entry.name: entry for entry in entry_points(group=group)}


def register_simulator(
    name: str,
) -> Callable[[type[EpisodeSimulator]], type[EpisodeSimulator]]:
    """Register a custom episode engine without importing native providers."""

    def register(cls: type[EpisodeSimulator]) -> type[EpisodeSimulator]:
        if name in _SIMULATORS:
            raise ValueError(f"Simulator '{name}' is already registered")
        _SIMULATORS[name] = cls
        return cls

    return register


def create_simulator(name: str, config: SimulatorConfig) -> EpisodeSimulator:
    """Construct only the selected installed native episode provider."""
    if name in _SIMULATORS:
        return _SIMULATORS[name](config)
    providers = _providers('usim.simulators')
    if name not in providers:
        raise ValueError(f"Unknown simulator: '{name}'. Available: {', '.join(list_simulators())}")
    return providers[name].load()(config)


def create_runner(name: str, **backend_options: str | Path | None) -> Simulator:
    """Construct a continuous runner; this does not launch its engine."""
    providers = _providers('usim.runners')
    if name not in providers:
        raise ValueError(f"Unknown runner: '{name}'. Available: {', '.join(providers)}")
    return providers[name].load()(**backend_options)


def list_simulators() -> list[str]:
    """List installed episode providers without importing or initializing engines."""
    return sorted(set(_providers('usim.simulators')) | set(_SIMULATORS))


def capabilities(name: str) -> frozenset[Literal['run', 'episode']]:
    """Describe supported execution models, not robot categories."""
    result: set[Literal['run', 'episode']] = set()
    if name in _providers('usim.runners'):
        result.add('run')
    if name in _providers('usim.simulators') or name in _SIMULATORS:
        result.add('episode')
    return frozenset(result)
