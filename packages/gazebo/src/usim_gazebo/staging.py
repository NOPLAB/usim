"""Package the private worker runtime and transfer caller-local inputs."""

from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

import usim


def prepare_runtime(directory: Path) -> None:
    """Copy the small Python-3.10-compatible runtime, without installing host wheels."""
    source = Path(usim.__file__).resolve().parent
    runtime = directory / 'runtime' / 'usim'
    for relative in (
        '__init__.py',
        'robot.py',
        'simulation.py',
        'interface.py',
        'types.py',
        'factory.py',
        'capability.py',
        'facets.py',
        'groups.py',
        'model.py',
        'model_io.py',
        'model_inspection.py',
        'runtime.py',
        'scene.py',
        'specs.py',
        'bridges/__init__.py',
        'bridges/ros2.py',
        'bridges/ros_runtime.py',
        'bridges/smoke.py',
    ):
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    shutil.copytree(
        Path(__file__).parent,
        runtime.parent / 'usim_gazebo',
        ignore=shutil.ignore_patterns('__pycache__'),
    )


def copy_inputs(engine: str, name: str, paths: list[tuple[Path, str]]) -> None:
    """Send a disk-backed tar archive; daemon-host paths need not match the caller."""
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode='w', dereference=True) as archive:
            for source, destination in paths:
                archive.add(source, arcname=destination.lstrip('/'))
        stream.seek(0)
        subprocess.run([engine, 'cp', '-', f'{name}:/'], stdin=stream, check=True, timeout=600)
