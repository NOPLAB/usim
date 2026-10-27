"""Own Kit startup and its imported files as one explicit runtime context."""

from __future__ import annotations

import ctypes
import gc
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from types import TracebackType
from typing import ClassVar, TYPE_CHECKING

from usim.model import ModelError, ModelFormat
from usim.model_io import ModelIO
from usim_isaacsim.transports import XmlToUsd
from usim_isaacsim.usd_loader import UsdAsset, UsdLoader

if TYPE_CHECKING:
    from isaacsim import SimulationApp


class ImportRuntime:
    """Mutable Kit owner; converted USD files live until the owning model closes."""

    _active: ClassVar[ImportRuntime | None] = None

    def __init__(self, *, headless: bool = True, rendering: bool = False) -> None:
        self.headless = headless
        self.rendering = rendering
        self.app: SimulationApp | None = None
        self.resources = ExitStack()
        self.models: ModelIO[UsdAsset] = ModelIO.compose(
            (UsdLoader(self.require_running),),
            (
                XmlToUsd(ModelFormat.URDF, self.require_running),
                XmlToUsd(ModelFormat.MJCF, self.require_running),
            ),
        )

    def require_running(self) -> None:
        if self.app is None or not self.app.is_running():
            raise ModelError('Isaac imports require a live Backend.imports() or session context')

    def start(self) -> None:
        if self.app is not None:
            self.require_running()
            return
        if ImportRuntime._active is not None:
            raise ModelError('Kit is process-global; close the active Isaac runtime first')
        from isaacsim import SimulationApp

        ImportRuntime._active = self
        try:
            self.app = SimulationApp(
                {
                    'headless': self.headless,
                    'multi_gpu': False,
                    'enable_crashreporter': False,
                    'fast_shutdown': False,
                },
                experience=(
                    str(Path(__file__).with_name('headless.kit'))
                    if self.headless and not self.rendering
                    else ''
                ),
            )
        finally:
            if self.app is None:
                ImportRuntime._active = None

    def close(self) -> None:
        try:
            if self.app is not None:
                gc.collect()
                if sys.platform == 'linux':
                    import carb

                    mapped_paths = {
                        fields[5]
                        for line in Path('/proc/self/maps').read_text().splitlines()
                        if len(fields := line.split(maxsplit=5)) == 6
                    }
                    # Keep provider code mapped until Python finalizers have finished,
                    # matching the legacy adapter's native shutdown contract.
                    for plugin in carb.get_framework().get_plugins():
                        if plugin.libPath and str(Path(plugin.libPath).resolve()) in mapped_paths:
                            ctypes.CDLL(
                                plugin.libPath,
                                mode=os.RTLD_NOW
                                | os.RTLD_LOCAL
                                | os.RTLD_NOLOAD
                                | os.RTLD_NODELETE,
                            )
                app, self.app = self.app, None
                app.close()
        finally:
            if ImportRuntime._active is self and self.app is None:
                ImportRuntime._active = None
            self.resources.close()

    def __enter__(self) -> ImportRuntime:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
