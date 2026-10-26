"""CPU point-cloud map reconstruction; optional geometry dependencies load on demand."""

from pathlib import Path

from .models import MapConfig, Source, Summary

__all__ = ['MapConfig', 'Source', 'generate']


def generate(source: Source, output: Path, config: MapConfig = MapConfig()) -> Summary:
    """Generate a source-preserving streamed USD map in an empty output directory."""
    from .pipeline import generate as run

    return run(source, output, config)
