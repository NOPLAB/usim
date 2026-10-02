"""Supervised Docker port for Gazebo Classic; importing it needs no ROS installation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import asdict
from pathlib import Path

from usim.simulation import SimulationConfig
from usim.ports.gazebo.assets import prepare_assets


class GazeboSimulator:
    """Run native Humble Python inside an owned container, never in host Python."""

    def __init__(self, image: str = 'usim-gazebo:local') -> None:
        self.image = image

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
            private_topic = '/usim_private/run_' + uuid.uuid4().hex + '/cmd_vel'
            mounts = prepare_assets(configuration, directory, private_topic)
            # Explicit small source copy, compatible with both 3.10 and 3.12.
            # No install of the host's Python-3.12 wheel in a Humble container.
            source = Path(__file__).resolve().parents[2]
            runtime = directory / 'runtime' / 'usim'
            for relative in (
                '__init__.py',
                'robot.py',
                'simulation.py',
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
                runtime / 'ports' / 'gazebo',
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
            command = [
                'docker',
                'create',
                '--name',
                name,
                '--init',
                '--network',
                'host',
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
            for variable in ('ROS_DOMAIN_ID', 'RMW_IMPLEMENTATION'):
                if variable in os.environ:
                    command.extend(('--env', variable + '=' + os.environ[variable]))
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
                    'exec python3 -m usim.ports.gazebo.runner /run/usim/config.json',
                )
            )
            process = None
            try:
                subprocess.run(command, check=True, timeout=60)
                process = subprocess.Popen(['docker', 'start', '--attach', name])
                while True:
                    try:
                        status = process.wait(timeout=0.1)
                        break
                    except subprocess.TimeoutExpired:
                        if stop is not None and stop.is_set():
                            subprocess.run(
                                ['docker', 'stop', '--time', '10', name], check=True, timeout=20
                            )
                            process.wait(timeout=20)
                            return
                if status:
                    raise RuntimeError(f'Gazebo container exited with status {status}')
            finally:
                # Force-removal also kills descendants if native graceful cleanup failed.
                owned = subprocess.run(
                    ['docker', 'ps', '--all', '--quiet', '--filter', 'name=^/' + name + '$'],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if owned.stdout.strip():
                    subprocess.run(['docker', 'rm', '--force', name], check=True, timeout=30)
                if process is not None:
                    process.wait(timeout=15)
