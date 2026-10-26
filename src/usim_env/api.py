"""Dependency-free contracts and explicit lazy environment providers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol

from usim_env_pipeline import MapConfig
from usim_env_pipeline.acquire import JSONValue
from usim_env_pipeline.models import PipelineInputError, Summary


@dataclass(frozen=True, slots=True)
class DownloadRequest:
    output: Path
    assets: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    source: Path
    output: Path
    config: MapConfig = MapConfig()
    reference_2018: Path | None = None
    authored: bool = True


class Environment(Protocol):
    name: str
    assets: Mapping[str, str]

    def download(self, request: DownloadRequest) -> tuple[Mapping[str, JSONValue], ...]: ...

    def generate(self, request: GenerateRequest) -> Summary: ...


_PROVIDERS = {
    'vtc': 'usim_env_vtc.environment',
    'tsudanuma': 'usim_env_tsudanuma.environment',
}
ENVIRONMENTS = tuple(_PROVIDERS)


def finish_generation(
    request: GenerateRequest,
    summary: Summary,
    *,
    infer_vegetation: bool = True,
    measured_ground: bool = False,
) -> Summary:
    """Apply the same original scene presentation for every dataset provider."""
    if request.authored:
        from usim_env_pipeline.scene import decorate

        decorate(
            request.output, infer_vegetation=infer_vegetation, measured_ground=measured_ground
        )
    return summary


def get_environment(name: str) -> Environment:
    """Resolve a known provider without importing its generation dependencies."""
    try:
        module = _PROVIDERS[name]
    except KeyError:
        raise PipelineInputError(
            f'Unknown environment {name!r}; choose {", ".join(ENVIRONMENTS)}'
        ) from None
    return import_module(module).EnvironmentProvider()
