"""VTC provider for the common environment interface."""

from argparse import Namespace
from types import MappingProxyType

from usim_env import DownloadRequest, GenerateRequest
from usim_env.api import finish_generation
from usim_env_pipeline.acquire import JSONValue
from usim_env_pipeline.models import PipelineInputError, Summary

from .sources import ASSETS


class EnvironmentProvider:
    name = 'vtc'
    assets = MappingProxyType({key: asset['name'] for key, asset in ASSETS.items()})

    def download(self, request: DownloadRequest) -> tuple[dict[str, JSONValue], ...]:
        keys = request.assets or tuple(self.assets)
        unknown = set(keys) - self.assets.keys()
        if unknown:
            raise PipelineInputError(f'Unknown VTC assets: {", ".join(sorted(unknown))}')
        from .download import download

        return tuple(download(key, request.output) for key in keys)

    def generate(self, request: GenerateRequest) -> Summary:
        from .reconstruct import run

        summary = run(
            Namespace(
                source=request.source,
                reference_2018=request.reference_2018,
                output=request.output,
                tile_size=request.config.tile_size,
                halo=request.config.halo,
                workers=request.config.workers,
                resume=request.config.resume,
            )
        )
        return finish_generation(request, summary)
