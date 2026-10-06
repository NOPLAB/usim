"""Supervised Docker/Podman port for Gazebo Classic; importing it needs no ROS installation."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal

import usim

from usim_gazebo.assets import GazeboAssetConfig, prepare_assets
from usim.simulation import ConfigurationError, SimulationConfig

__all__ = ['GazeboAssetConfig', 'GazeboLidarConfig', 'GazeboSimulator']

# Host settings that must match between external ROS clients and the container.
_PASSTHROUGH: Final = (
    'ROS_DOMAIN_ID',
    'ROS_LOCALHOST_ONLY',
    'RMW_IMPLEMENTATION',
    'FASTDDS_BUILTIN_TRANSPORTS',
)


@dataclass(frozen=True, slots=True)
class GazeboLidarConfig:
    """Physical planar Gazebo ray sensor attached to an existing robot link."""

    link_name: str
    frame_name: str
    topic: str
    update_rate: float = 10.0
    horizontal_samples: int = 720
    min_angle: float = -math.pi
    max_angle: float = math.pi
    range_min: float = 0.1
    range_max: float = 10.0

    def __post_init__(self) -> None:
        for name in ('link_name', 'frame_name', 'topic'):
            value = getattr(self, name)
            if not value.strip() or value != value.strip():
                raise ConfigurationError(f'lidar.{name}')
        if not self.topic.startswith('/') or any(char.isspace() for char in self.topic):
            raise ConfigurationError('lidar.topic')
        for name in ('update_rate', 'min_angle', 'max_angle', 'range_min', 'range_max'):
            if not math.isfinite(getattr(self, name)):
                raise ConfigurationError(f'lidar.{name}')
        if self.update_rate <= 0:
            raise ConfigurationError('lidar.update_rate')
        # bool satisfies an int annotation, so the type alone does not exclude it.
        if isinstance(self.horizontal_samples, bool) or self.horizontal_samples < 1:
            raise ConfigurationError('lidar.horizontal_samples')
        if self.max_angle <= self.min_angle or self.max_angle - self.min_angle > 2 * math.pi:
            raise ConfigurationError('lidar.max_angle')
        if self.range_min <= 0:
            raise ConfigurationError('lidar.range_min')
        if self.range_max <= self.range_min:
            raise ConfigurationError('lidar.range_max')


class GazeboSimulator:
    """Run native Humble Python inside an owned container, never in host Python."""

    def __init__(
        self,
        image: str = 'usim-gazebo:local',
        *,
        engine: Literal['docker', 'podman'] = 'docker',
        lidar: GazeboLidarConfig | None = None,
        network: str = 'host',
        fastdds_profile: Path | None = None,
    ) -> None:
        self.image = image
        self.engine = engine
        self.lidar = lidar
        self.network = network
        self.fastdds_profile = fastdds_profile

    def run(self, configuration: SimulationConfig, *, stop: threading.Event | None = None) -> None:
        self._run(configuration, stop=stop)

    def smoke(
        self,
        configuration: SimulationConfig,
        out_dir: Path,
        *,
        stop: threading.Event | None = None,
    ) -> None:
        """Run the same engine with a native-ABI ROS probe and persisted evidence."""
        if configuration.ros is None or not configuration.camera_enabled:
            raise ValueError('smoke requires ROS and the camera')
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'smoke.json').unlink(missing_ok=True)
        self._run(configuration, stop=stop, out_dir=out_dir.resolve())

    def _run(
        self,
        configuration: SimulationConfig,
        *,
        stop: threading.Event | None,
        out_dir: Path | None = None,
    ) -> None:
        if stop is not None and stop.is_set():
            return
        name = 'usim-gazebo-' + uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix='usim-gazebo-') as temporary:
            directory = Path(temporary)
            if self.fastdds_profile is not None:
                shutil.copyfile(self.fastdds_profile, directory / 'fastdds_profile.xml')
            private_topic = '/usim_private/run_' + uuid.uuid4().hex + '/cmd_vel'
            mounts = prepare_assets(
                configuration, directory, GazeboAssetConfig(private_topic, self.lidar)
            )
            # Explicit small source copy, compatible with both 3.10 and 3.12.
            # No install of the host's Python-3.12 wheel in a Humble container.
            source = Path(usim.__file__).resolve().parent
            runtime = directory / 'runtime' / 'usim'
            for relative in (
                '__init__.py',
                'robot.py',
                'simulation.py',
                'interface.py',
                'types.py',
                'factory.py',
                'ports/__init__.py',
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
            data = asdict(configuration)
            data.update(world='/run/usim/world.sdf', robot_urdf='/run/usim/robot.urdf')
            (directory / 'config.json').write_text(
                json.dumps(
                    {
                        'configuration': data,
                        'private_topic': private_topic,
                        'smoke_output': '/output' if out_dir else None,
                    }
                ),
                encoding='utf-8',
            )
            model_paths = [path for _, path in mounts]
            model_paths.append('/usr/share/gazebo-11/models')
            engine = self.engine
            command = [
                engine,
                'create',
                '--name',
                name,
                '--init',
                '--network',
                self.network,
                '--shm-size',
                '512m',
                '--env',
                'PYTHONPATH=/run/usim/runtime',
                '--env',
                'PYTHONDONTWRITEBYTECODE=1',
                '--env',
                'GAZEBO_MODEL_DATABASE_URI=',
                '--env',
                'GAZEBO_MODEL_PATH=' + ':'.join(model_paths),
            ]
            for variable in _PASSTHROUGH:
                if variable in os.environ:
                    command.extend(('--env', variable + '=' + os.environ[variable]))
            if self.fastdds_profile is not None:
                command.extend(
                    ('--env', 'FASTRTPS_DEFAULT_PROFILES_FILE=/run/usim/fastdds_profile.xml')
                )
            for host, container in [(directory, '/run/usim'), *mounts]:
                command.extend(('--mount', f'type=bind,source={host},target={container},readonly'))
            if out_dir is not None:
                command.extend(('--mount', f'type=bind,source={out_dir},target=/output'))
            if not configuration.headless:
                if not os.environ.get('DISPLAY'):
                    raise ValueError('GUI requires DISPLAY; use headless=True for Xvfb')
                command.extend(('--env', 'DISPLAY=' + os.environ['DISPLAY']))
                if Path('/tmp/.X11-unix').exists():
                    command.extend(
                        (
                            '--mount',
                            'type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix,readonly',
                        )
                    )
            command.extend(
                (
                    self.image,
                    'bash',
                    '-c',
                    'source /opt/ros/humble/setup.bash && '
                    'exec python3 -m usim_gazebo.runner /run/usim/config.json',
                )
            )
            process = None
            created = False
            try:
                subprocess.run(command, check=True, timeout=60)
                created = True
                process = subprocess.Popen([engine, 'start', '--attach', name])
                while True:
                    try:
                        status = process.wait(timeout=0.1)
                        break
                    except subprocess.TimeoutExpired:
                        if stop is not None and stop.is_set():
                            subprocess.run(
                                [engine, 'stop', '--time', '10', name], check=True, timeout=20
                            )
                            process.wait(timeout=20)
                            return
                if status:
                    raise RuntimeError(f'Gazebo container exited with status {status}')
            finally:
                # Force-removal also kills descendants if native graceful cleanup failed.
                # A successful create already proves ownership. Only an uncertain
                # creation needs an exact-name lookup before removal.
                if not created:
                    prefix = '^/' if engine == 'docker' else '^'
                    owned = subprocess.run(
                        [engine, 'ps', '--all', '--quiet', '--filter', f'name={prefix}{name}$'],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    created = bool(owned.stdout.strip())
                if created:
                    subprocess.run([engine, 'rm', '--force', name], check=True, timeout=30)
                if process is not None:
                    process.wait(timeout=15)
