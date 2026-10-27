"""Owned container and correlated, bounded JSON request acknowledgements."""

from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FutureTimeout
import json
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Protocol, TypeAlias
from uuid import uuid4

from usim.model import ModelError

Json: TypeAlias = None | bool | int | float | str | list['Json'] | dict[str, 'Json']


class RequestPort(Protocol):
    @property
    def ended(self) -> Future[None]: ...

    def request(self, operation: str, payload: dict[str, Json]) -> dict[str, Json]: ...

    def close(self) -> None: ...


class WireClient:
    """One reader owns stdout; request futures are completed only by matching replies."""

    def __init__(self, process: subprocess.Popen[str], timeout: float = 30.0) -> None:
        self.process = process
        self.timeout = timeout
        self._lock = threading.Lock()
        self._pending: dict[str, Future[dict[str, Json]]] = {}
        self._closed = False
        self.ended: Future[None] = Future()
        self._reader = threading.Thread(target=self._receive, name='gazebo-replies', daemon=True)
        self._reader.start()

    def _fail_pending(self, error: Exception) -> None:
        with self._lock:
            notify = not self._closed
            self._closed = True
            pending, self._pending = self._pending, {}
        for future in pending.values():
            future.set_exception(error)
        if notify:
            self.ended.set_result(None)

    def _receive(self) -> None:
        if self.process.stdout is None:
            self._fail_pending(ModelError('Worker has no stdout'))
            return
        try:
            for line in self.process.stdout:
                if not line.startswith('USIM_RESPONSE '):
                    continue
                response = json.loads(line.removeprefix('USIM_RESPONSE '))
                if not isinstance(response, dict) or not isinstance(response.get('id'), str):
                    raise ModelError('Malformed Gazebo acknowledgement')
                with self._lock:
                    future = self._pending.pop(response['id'], None)
                if future is None:
                    raise ModelError('Unexpected Gazebo acknowledgement ID')
                if response.get('ok') is True and isinstance(response.get('result'), dict):
                    future.set_result(response['result'])
                else:
                    future.set_exception(ModelError(str(response.get('error', 'Worker failed'))))
        except (OSError, ValueError) as error:
            self._fail_pending(error)
        finally:
            self._fail_pending(ModelError('Gazebo worker disconnected'))

    def request(self, operation: str, payload: dict[str, Json]) -> dict[str, Json]:
        identifier = uuid4().hex
        future: Future[dict[str, Json]] = Future()
        with self._lock:
            if self._closed or self.process.stdin is None:
                raise ModelError('Gazebo worker is closed')
            self._pending[identifier] = future
            try:
                self.process.stdin.write(
                    json.dumps(
                        {
                            'id': identifier,
                            'operation': operation,
                            'payload': payload,
                            'timeout': self.timeout,
                        },
                        allow_nan=False,
                    )
                    + '\n'
                )
                self.process.stdin.flush()
            except (OSError, ValueError):
                self._pending.pop(identifier)
                raise
        try:
            return future.result(timeout=self.timeout + 1)
        except FutureTimeout as error:
            self.close()
            raise TimeoutError(f'Gazebo {operation} acknowledgement timed out') from error

    def close(self) -> None:
        self._fail_pending(ModelError('Gazebo request cancelled by close'))
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if threading.current_thread() is not self._reader:
            self._reader.join(timeout=5)
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()


class ContainerPort:
    """Own exactly one UUID-named container, including partially failed creation."""

    def __init__(
        self,
        directory: Path,
        *,
        image: str = 'usim-gazebo:local',
        engine: str = 'docker',
        timeout: float = 30.0,
    ) -> None:
        if engine not in {'docker', 'podman'}:
            raise ModelError('Container engine must be docker or podman')
        self.engine, self.name = engine, 'usim-grouped-' + uuid4().hex
        self.directory = directory
        self._client: WireClient | None = None
        self._owned = False
        bundle = directory / '_usim_worker'
        bundle.mkdir()
        for module in ('worker.py', 'worker_runtime.py'):
            shutil.copyfile(Path(__file__).with_name(module), bundle / module)
        try:
            subprocess.run(
                [
                    engine,
                    'create',
                    '--name',
                    self.name,
                    '--init',
                    '--interactive',
                    '--network',
                    'none',
                    '--env',
                    'GAZEBO_MODEL_DATABASE_URI=',
                    '--mount',
                    f'type=bind,source={directory},target=/run/usim,readonly',
                    image,
                    'bash',
                    '-c',
                    'source /opt/ros/humble/setup.bash && '
                    'export PYTHONPATH=/run/usim:"${PYTHONPATH:-}" && '
                    'exec python3 -m _usim_worker.worker',
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self._owned = True
            process = subprocess.Popen(
                [engine, 'start', '--attach', '--interactive', self.name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._client = WireClient(process, timeout)
        finally:
            if self._client is None:
                if not self._owned:
                    prefix = '^/' if engine == 'docker' else '^'
                    result = subprocess.run(
                        [
                            engine,
                            'ps',
                            '--all',
                            '--quiet',
                            '--filter',
                            f'name={prefix}{self.name}$',
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    self._owned = bool(result.stdout.strip())
                self.close()

    def request(self, operation: str, payload: dict[str, Json]) -> dict[str, Json]:
        if self._client is None:
            raise ModelError('Container is closed')
        # All resources are staged under the session directory, including later edits.
        encoded = json.dumps(payload).replace(self.directory.as_uri(), 'file:///run/usim')
        return self._client.request(operation, json.loads(encoded))

    @property
    def ended(self) -> Future[None]:
        if self._client is None:
            raise ModelError('Container is closed')
        return self._client.ended

    def close(self) -> None:
        try:
            if self._owned:
                subprocess.run(
                    [self.engine, 'rm', '--force', self.name],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                self._owned = False
        finally:
            if self._client is not None:
                self._client.close()
                self._client = None
