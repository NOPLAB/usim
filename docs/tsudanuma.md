# Tsudanuma 2020 CIT

Catalogue: <https://github.com/tsukubachallenge/tc-datasets/blob/832026ff389e94eca9f945a3fe07f5905d0646d7/README.md>

Provider: Chiba Institute of Technology. Original data license: CC-BY-4.0.
Attribution: Chiba Institute of Technology, Tsudanuma 2020 dataset.

## Published originals

The requested delivery uses the publicly released **point-cloud map only**.
The user explicitly excluded the original sensor archive on 2026-10-07.
No bag acquisition, raw-sensor replay or republication is needed for this map.

| Asset | Catalogue | Observed status |
| --- | --- | --- |
| `map_tsudanuma20_cit.pcd` | Approx. 490.8 MB; 13,583,284 points | Retrieved and validated: 491,004,034 bytes, 13,611,640 ASCII XYZI rows, no normals |
| `tsudanuma20_cit_compressed.bag` | Approx. 57 GB; 2020-08-27 17:43:12, 56:12 duration | Published endpoints return HTTP 404; browser reports file not found |

The map SHA256 is
`19fd7f0cf35e3d8a72f089e48081f572ff21d0a8fd92fcaf916f1bce8993a486`.
This is a fingerprint of the actually retrieved original, not a provider-published
checksum. The catalogue point count differs by 28,356 rows; the actual count and
catalogue discrepancy are recorded in its `.acquisition.json` receipt.
The WIP fuRo section has no downloadable original and is not a third invented asset.

Acquired originals live under `assets/tsudanuma/upstream/` and are ignored by Git.
The common downloader atomically publishes originals after content/header/row
validation and hashes them while streaming. Cached originals are revalidated.

The sensor archive has no acquired bytes, checksum or verified bag header.
All-data acquisition remains incomplete, but it is outside this delivery.
The published point-cloud map is independently usable; its generation does not
require fabricating or substituting sensor data.

### Web acquisition audit (2026-10-07)

The current catalogue, original [publication PR](https://github.com/tsukubachallenge/tc-datasets/pull/3)
and historical filename `CIT_2020_compressed.bag` all identify the same Drive
file, `1SBKNJ2NDwQlhbN75WJcMppTSwlJu1s4g`. Viewer, `uc` and
`drive.usercontent.google.com` download probes returned HTTP 404.
All 12 public catalogue forks were examined: the 11 containing this dataset
retain that same link, and the remaining fork has no Tsudanuma entry.
Catalogue issues/PRs, the CIT RDC lab page, SelectDataset, Hugging Face and
Internet Archive searches supplied no publicly downloadable matching copy.
No sensor bytes were acquired. Local C: and G: each had less than 3 GB free,
so a full 57 GB acquisition would also need a different storage destination.

The user subsequently selected `nop@pve1ubuntu:/mnt/nas/datasets` for storage
and `/mnt/workspace/nop/usim/` for work. Windows OpenSSH authenticated
successfully; Git Bash's SSH could not use the Windows authentication agent.
The NAS reported 22 TB available. The work directory was created, but creation
of `/mnt/nas/datasets/tsudanuma/upstream` was denied: its parent belongs to
root and is mode 0755, and `sudo -n` requires interactive authentication.
Using the existing CPU-only Ubuntu image, the new dedicated directory was
created with nop ownership. Its permissions were then finalized by nop.
`/mnt/nas/datasets/tsudanuma/upstream` is now writable, with 22 TB available.
The temporary empty fallback directories were removed; no existing datasets
were changed.
Viewer and both direct-download routes also returned HTTP 404 from pve1ubuntu,
so changing the host did not restore public access to the bag.
Hourly endpoint and catalogue-link monitors were tested, but a subsequent
stall check still found HTTP 404 and the unchanged original link. Both
monitors were stopped because no restoration or replacement is underway that
this task can observe. The user subsequently selected point-cloud-only use;
the unavailable bag no longer blocks delivery. The prepared republication
request has not been submitted.

A separate check for a missing Google Drive `resourcekey` found no matching
public key in search results. Wayback CDX timed out and its availability API
returned HTTP 429, so archived viewer-page recovery remains unverified.
These responses do not establish that the original file was deleted.

## Generation

Use the [common environment CLI](env.md). `usim_env_tsudanuma` supplies only
catalogue provenance, acquisition validation and raw Tsudanuma frame policy.
The shared pipeline estimates working normals, reconstructs supported source
surfaces, retains exact original records and produces collidable streaming cells.
Tsudanuma presentation retains measured buildings and vegetation without adding
unlabelled procedural tree proxies. No VTC geography, proprietary building asset
or survey pose is silently assigned to Tsudanuma.

The source XYZ frame is used in meters/Z-up. No geographic CRS or survey
registration is inferred. Synthetic colors/windows and original model proxies
are visualization/representation, not claims of measured facade or species labels.

### Server use

The point-cloud-only common CLI generation and independent verification passed
on this server on 2026-10-07. The NAS PCD SHA256 matches the acquired public map.
Verification checked all 13,611,640 records, 4,613,571 source-derived vertices
and 5,128,220 collision triangles across 143 occupied cells.

The public PCD and generated map are stored on `nop@pve1ubuntu`:

- Input: `/mnt/nas/datasets/tsudanuma/upstream/map_tsudanuma20_cit.pcd`
- USD scene: `/mnt/nas/datasets/tsudanuma/measured/authored.usda`
- Measured scene: `/mnt/nas/datasets/tsudanuma/measured/world.usda`
- Source snapshot and CPU environment: `/mnt/workspace/nop/usim`

Run from the server work directory. The environment uses the existing
`pyproject.toml` pipeline extra and Python 3.12, not native simulator dependencies.

```sh
cd /mnt/workspace/nop/usim
~/.local/bin/uv run --frozen --no-dev --extra pipeline --python 3.12 \
  python -m usim_env generate tsudanuma \
  --source /mnt/nas/datasets/tsudanuma/upstream/map_tsudanuma20_cit.pcd \
  --output /mnt/nas/datasets/tsudanuma/measured --workers 1 --resume
~/.local/bin/uv run --frozen --no-dev --extra pipeline --python 3.12 \
  python -m usim_env_pipeline.verify \
  --map /mnt/nas/datasets/tsudanuma/measured
```

## Generated map and checks

`assets/tsudanuma/measured/` contains 143 occupied cells, including 117 measured
surface cells with 5,128,220 triangles and 4,613,571 source-derived vertices.
The 26 cells without supported triangles remain point payloads. The original
13,611,640 records are retained independently, including records not used in a
supported triangle; surfaces do not assert complete or watertight coverage.
The corrected local `authored.usda` / `authored-index.json` contain no inferred
tree proxies. The earlier 308 inferred proxies were not semantic labels and
obscured the urban buildings; their prototype placement also doubled inferred
crown radii. Shared crown inference now rejects locally planar recessed facades,
and prototype placement uses the supported radius rather than its diameter.

The corrected scene retains all source point payloads, including unsupported
returns inside partially meshed cells. Its separate `/World/Ground` layer adds
833,389 source-vertex triangles, 568,135 original vertices and 74 ground cells,
covering 11,476.25 m² of projected ground. Ground reconstruction uses actual
lower-envelope representatives, XY triangulation, upward winding, measured
edge/interior support and slope limits. Coarse connectivity rejects detached
elevated floor regions; it does not generate faces across missing observations.
Measured BPA surfaces and original records remain unchanged. Ground meshes have
full triangle collision and source IDs; independent verification checks exact
source coordinates, winding, edge support, ownership, bounds and collisions.
This is supported terrain, not an extrapolated flat plane or complete coverage
under unobserved building interiors. Ground selection is geometric, not a
surveyed semantic classification. Twenty narrow-column proxies remain.

`verification.json` independently confirms every original record, every
source-derived vertex, all full-triangle collisions and conservative surface
bounds. `authored-scene.json` records unchanged measured positions/collision
faces and the additional ground counts.

The baseline Isaac Sim 6.1 captures of `authored.usda` use the real RTX viewport,
including a mesh-only view with `/World/PointRegions` hidden in the session
layer. That original authored scene remains available for point inspection.
The server verification above describes the earlier measured BPA map, not a
regeneration of this corrected local authored presentation.

### Planar presentation

The separate planar optimizer fits walls and horizontal/sloped surfaces from
the original point cloud, unifies compatible regional planes and connects
small coplanar discontinuities. Recess returns protect observed openings.
It retains source-identical residual geometry where planar replacement is not
supported. Original records, BPA meshes and the measured-ground layer are not
rewritten.

```sh
uv run --extra pipeline python -m usim_env_pipeline.planar \
  --map assets/tsudanuma/measured --workers 8
uv run --extra pipeline python -m usim_env_pipeline.planar_verify \
  --map assets/tsudanuma/measured
```

Open `assets/tsudanuma/measured/optimized.usda` to inspect the fitted version.
Its `planar-report.json` distinguishes observed support from bounded
interpolation; [the pipeline contract](env-pipeline.md#optional-planar-presentation)
documents tolerances, preservation and verification.

The local Gaussian result evaluates 9,538 nearby board pairs with eight workers
and accepts 42 compatible joins. It contains 883 fitted patches with 1,108,284
triangles and 1,981,995 residual triangles: 3,090,279 total, versus 5,961,609 in
the earlier BPA plus ground presentation. Original support covers 5,919,391
records; fitted area is 79,616.59 m², including 29,825.30 m² of interpolation.
That includes 3,002.17 m² of enclosed-interior completion and 88.29 m² of
Gaussian-supported bridges. Large floor gaps are no longer rejected because
of above-floor obstacle returns or the earlier near-point limit.
Maximum normal displacement of supporting observations is 0.08 m. The separate
verifier passed for all patch footprints, source coordinates and collisions.
These areas are fitted surfaces, not semantic building/ground area estimates.

The earlier local viewer opened this optimized version.
Its `completed_overview`, `completed_reverse`, `completed_detail`,
`completed_facade` and `completed_street` captures preserve the earlier
`planar_*` baseline images. Separate session-only cameras and a rendered-view
matrix check reject frames still in flight from the preceding pose.

### CAD replacement

The current local viewer opens `assets/tsudanuma/measured/cad.usda`.
Generate its authoritative CAD surfaces with:

```sh
uv run --extra pipeline --extra cad python -m usim_env_pipeline.cad \
  --map assets/tsudanuma/measured --workers 8
uv run --extra pipeline --extra cad python -m usim_env_pipeline.cad_verify \
  --map assets/tsudanuma/measured
```

Open `cad/model.step` in a CAD application, or use the meter-valued
`cad/model.brep`. The USD view contains only kernel-derived tessellation.
The source PCD, fitted-plane archives and earlier BPA/planar views remain
available separately. `cad_*` screenshots preserve the earlier `completed_*`
and `planar_*` captures. The [CAD contract](env-pipeline.md#cad-geometry-authority)
defines hole removal, near-parallel containment and exchange units.

### Building-only geometry models

The [building model program](building-models.md) replaces connected measured
facades with closed cuboid proxies in `buildings.usda`. It keeps the existing
ground separate and unchanged. Corner-supported building envelopes and
single-facade wall cuboids retain their measured support and explicitly label
inferred geometry. This is the building-specific alternative to replacing
every fitted surface with a CAD board; ground reconstruction remains separate.
