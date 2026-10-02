"""Supervise the native SDK so blocked rendering/shutdown cannot leak an engine."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import TYPE_CHECKING

from usim.simulation import Ros2Config, SimulationConfig

if TYPE_CHECKING:
    from usim.ports.isaac.sim import IsaacSimulator


def run(
    configuration: SimulationConfig, port: IsaacSimulator, *, stop: threading.Event | None = None
) -> None:
    with tempfile.TemporaryDirectory(prefix='usim-isaac-') as temporary:
        directory = Path(temporary)
        data = asdict(configuration)
        data.update(
            world=str(configuration.world.resolve()),
            robot_urdf=str(configuration.robot_urdf.resolve()),
        )
        options = {
            name: getattr(port, name)
            for name in (
                'robot_prim_path',
                'environment_prim_path',
                'camera_prim_path',
                'ground_name',
                'optical_frame',
            )
        }
        options['contact_out'] = str(port.contact_out.resolve()) if port.contact_out else None
        path = directory / 'configuration.json'
        stop_file = directory / 'stop'
        path.write_text(
            json.dumps(
                {
                    'configuration': data,
                    'port': options,
                    'asset_directory': str(directory),
                    'stop_file': str(stop_file),
                }
            ),
            encoding='utf-8',
        )
        process = subprocess.Popen(
            [port.python, '-u', '-m', __name__, str(path)],
            # A pending pipe read on Windows blocks CRT stdio setup inside native DLL loads,
            # so cancellation uses a stop file instead of stdin.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        ready = threading.Event()
        result_received = threading.Event()
        result = []

        def output():
            for line in process.stdout:
                print(line, end='', flush=True)
                if line.strip() == 'USIM_ISAAC_RUNNING':
                    ready.set()
                if line.startswith('{'):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get('status') in ('finished', 'stopped', 'incomplete', 'error'):
                        result.append(row)
                        result_received.set()

        reader = threading.Thread(target=output, name='usim-isaac-output', daemon=True)
        reader.start()
        started = time.monotonic()
        running_at = None
        closing_at = None
        cancelled_at = None
        try:
            while True:
                try:
                    status = process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    pass
                now = time.monotonic()
                if ready.is_set() and running_at is None:
                    running_at = now
                if result_received.is_set() and closing_at is None:
                    closing_at = now
                if stop is not None and stop.is_set() and cancelled_at is None:
                    stop_file.touch()
                    cancelled_at = now
                if running_at is None and now - started > 300:
                    raise TimeoutError('Isaac initialization exceeded 300 seconds')
                if (
                    running_at is not None
                    and configuration.max_seconds
                    and closing_at is None
                    and now - running_at > configuration.max_seconds + 15
                ):
                    raise TimeoutError('Isaac failed to stop at its run deadline')
                if closing_at is not None and now - closing_at > 30:
                    raise TimeoutError('Isaac shutdown exceeded 30 seconds')
                if cancelled_at is not None and now - cancelled_at > 30:
                    raise TimeoutError('Isaac cancellation exceeded 30 seconds')
            reader.join(timeout=5)
            if reader.is_alive():
                raise RuntimeError('Isaac output reader failed to stop')
            if status or not result or result[-1]['status'] not in ('finished', 'stopped'):
                raise RuntimeError(f'Isaac worker failed: exit={status}, result={result}')
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=15)
            reader.join(timeout=5)
            process.stdout.close()


def main() -> None:
    from usim.ports.isaac.sim import IsaacSimulator, _run

    data = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    # Kit must not interpret the supervisor's JSON path as a scene to open.
    sys.argv = [sys.argv[0]]
    fields = data['configuration']
    fields['world'], fields['robot_urdf'] = Path(fields['world']), Path(fields['robot_urdf'])
    fields['camera_offset'] = tuple(fields['camera_offset'])
    if fields['ros'] is not None:
        fields['ros'] = Ros2Config(**fields['ros'])
    options = data['port']
    if options['contact_out'] is not None:
        options['contact_out'] = Path(options['contact_out'])
    stop = threading.Event()
    stop_file = Path(data['stop_file'])

    def cancel():
        while not stop_file.exists():
            time.sleep(0.1)
        stop.set()

    threading.Thread(target=cancel, name='usim-isaac-cancel', daemon=True).start()
    _run(
        SimulationConfig(**fields),
        IsaacSimulator(**options),
        stop=stop,
        asset_directory=Path(data['asset_directory']),
    )


if __name__ == '__main__':
    main()
