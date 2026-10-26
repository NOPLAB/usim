# Environment interfaces

`usim_env` is the common dataset interface. `usim_env_pipeline` owns reusable
acquisition, point-cloud reconstruction, original geometry, feature inference and
scene presentation. Dataset packages own provenance and dataset-specific policy.

```sh
uv run --extra pipeline python -m usim_env list
uv run --extra pipeline python -m usim_env download vtc --output-dir assets/public-tsukuba 2019
uv run --extra pipeline python -m usim_env download tsudanuma --output-dir assets/tsudanuma/upstream map
uv run --extra pipeline python -m usim_env generate tsudanuma \
  --source assets/tsudanuma/upstream/map_tsudanuma20_cit.pcd \
  --output assets/tsudanuma/measured --workers 1
```

An omitted asset selection downloads all listed originals. VTC keys are `2018`,
`2019`; Tsudanuma keys are `map`, `sensors`. Failed acquisition is an error, not
a successful empty result. The original sensor archive remains separate from
the published point-cloud map; it is never silently fused.
For the requested Tsudanuma point-cloud-only delivery, explicitly select `map`
as above. The unavailable `sensors` asset is not required for generation or use.

Generation preserves all original records and produces the measured collision
surfaces, point archives and bounded streaming indices. The default common
interface also produces `authored.usda` and `authored-index.json`: original
procedural surface colors, explicitly synthetic window bands, supported original
tree models and narrow-column proxies. It does not invent missing ground or
move measured collision faces. Unsupported cores retain visible source points.
Use `--measured-only` to produce only the measured reconstruction.

For interrupted large jobs, `--resume` verifies original/archived hashes,
provenance, frame and geometry settings before reusing completed regional
reports. Incomplete regions are regenerated; unrelated input and completed
payloads are not overwritten. Worker count can change when resuming.

```sh
uv run --extra pipeline python -m usim_env generate tsudanuma \
  --source assets/tsudanuma/upstream/map_tsudanuma20_cit.pcd \
  --output assets/tsudanuma/measured --workers 1 --resume
```

## Python API

```python
from pathlib import Path
from usim_env import DownloadRequest, GenerateRequest, get_environment

environment = get_environment('tsudanuma')
environment.download(DownloadRequest(Path('assets/tsudanuma/upstream'), ('map',)))
summary = environment.generate(
    GenerateRequest(
        Path('assets/tsudanuma/upstream/map_tsudanuma20_cit.pcd'),
        Path('assets/tsudanuma/measured'),
    )
)
```

Place generation calls inside a `__main__` guard in multiprocessing programs.
The environment protocol has only `name`, `assets`, `download` and `generate`.
Providers are explicitly registered and loaded lazily. Listing/help/acquisition
do not import native geometry or simulation SDKs.

The existing VTC reconstruction, vegetation, authoring and download CLI entry
points remain available. They delegate reusable operations to the pipeline;
VTC registration, terrain masks, City Hall selection and independent source
verification remain dataset-specific.

CPU dependencies are declared in `pyproject.toml`'s `pipeline` extra. Native
simulation SDK environments remain separate. MIT covers software and original
artwork, not every data-derived map: VTC retains its source conditions;
Tsudanuma CIT data and derived geometry require CC-BY-4.0 attribution.

See [pipeline contracts](env-pipeline.md) and
[Tsudanuma provenance/status](tsudanuma.md).
