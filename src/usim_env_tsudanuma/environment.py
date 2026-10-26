"""Tsudanuma provider for the common environment interface."""

from argparse import Namespace
from types import MappingProxyType

from usim_env import DownloadRequest, GenerateRequest
from usim_env.api import finish_generation
from usim_env_pipeline.acquire import JSONValue
from usim_env_pipeline.models import PipelineInputError, Summary

from .sources import ASSETS


class EnvironmentProvider:
    name = 'tsudanuma'
    assets = MappingProxyType({'map': ASSETS['map']['name'], 'sensors': ASSETS['sensor']['name']})

    def download(self, request: DownloadRequest) -> tuple[dict[str, JSONValue], ...]:
        keys = request.assets or tuple(self.assets)
        unknown = set(keys) - self.assets.keys()
        if unknown:
            raise PipelineInputError(f'Unknown Tsudanuma assets: {", ".join(sorted(unknown))}')
        from .download import download

        return tuple(
            download('sensor' if key == 'sensors' else key, request.output) for key in keys
        )

    def generate(self, request: GenerateRequest) -> Summary:
        if request.reference_2018 is not None:
            raise PipelineInputError('--reference-2018 is only supported by VTC')
        from .reconstruct import run

        summary = run(
            Namespace(
                source=request.source,
                output=request.output,
                tile_size=request.config.tile_size,
                halo=request.config.halo,
                workers=request.config.workers,
                resume=request.config.resume,
            )
        )
        # Unlabelled urban XYZI returns do not establish tree species or crown envelopes.
        # Retain the measured vegetation rather than covering buildings with guessed models.
        return finish_generation(request, summary, infer_vegetation=False, measured_ground=True)
