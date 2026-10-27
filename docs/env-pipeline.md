# Point-cloud map generation

`usim_env_pipeline` turns local PCD or PLY point clouds into regional USD
render/collision maps. It shares the source-preserving reconstruction previously
implemented for the full Tsukuba map. It has no VTC filenames, providers, year
roles, license assumptions, or geographic registration built into its engine.

## Generate a map

Run from the usim checkout with Python 3.10-3.14 in a dedicated CPU environment:

```sh
uv run --python 3.14 --extra pipeline python -m usim_env_pipeline \
  --source assets/survey.pcd --output assets/survey-map \
  --license "your-data-license" --attribution "your-provider" \
  --source-url "https://example.org/survey" --frame "local survey XYZ" \
  --workers 4
```

Use a new or empty output directory. Existing maps are never silently overwritten.
`--sha256 <digest>` optionally pins the input. A failed generation has no complete
manifest; use a new output directory when retrying it.

The input coordinates must already be **meters and Z-up**. The frame label records
your declared coordinate frame; it does not register a cloud, infer a CRS, or
establish survey accuracy. Do not implicitly combine scans in different frames.

PCD supports `DATA ascii` and little-endian `DATA binary`, scalar XYZ, arbitrary
additional fields and optional supplied normals/intensity/RGB. Packed 32-bit
`rgb`/`rgba`, byte-valued `r,g,b`, or normalized float `r,g,b` are accepted.
Binary-compressed PCD is rejected. Every PCD field and source row is retained
in regional archives. PLY uses Open3D's point-cloud decoder: positions,
normals and RGB are retained in regional records, while additional PLY properties
remain in the byte-identical original file. Empty clouds and nonfinite XYZ
are rejected rather than silently dropping records.

`--normal-policy auto` uses valid supplied normals and estimates missing/invalid
working normals from local neighborhoods. Derived normals are identified in the
manifest and quantitative reports; they never overwrite original fields.
`--normal-policy supplied` prohibits estimation: unsupported records remain
visible points without invented collision faces.

`--spacing` is nominal sample spacing in meters (default 0.1). It scales pivot
radii and edge/interior support limits. `--tile-size` defaults to 30 m,
`--halo` to 1.5 m; the halo must be at least twice the largest pivot radius and
smaller than the tile. Sparse surveys may need different spacing/halo settings.
Do not increase support limits merely to hide unsupported gaps.

## Library API

```python
from pathlib import Path
from usim_env_pipeline import MapConfig, Source, generate

summary = generate(
    Source(
        Path('assets/survey.ply'),
        license='CC-BY-4.0',
        attribution='Example survey provider',
    ),
    Path('assets/survey-map'),
    MapConfig(workers=4),
)
print(summary['triangles'])
```

Programs using multiprocessing must invoke `generate` beneath their normal
`if __name__ == '__main__':` guard, especially on Windows. CLI invocation already
does this. `MapConfig.references` accepts separately labeled sources for archival
only; references do not enter primary geometry.

Importing the package and requesting CLI help do not load the native geometry
libraries. The core usim wheel remains dependency-free. The `pipeline` extra pins
NumPy, SciPy, Open3D and USD in a CPU environment. It is deliberately incompatible
with the separately pinned Isaac native/USD and ManiSkill native extras; use
separate environments. Python 3.13+ selects newer NumPy, SciPy, Open3D and USD
pins; Python 3.14 additionally selects a newer scikit-image. See the
[Python support matrix](python-support.md).
For an isolated interpreter, install
`src/usim_env_pipeline/requirements.txt`. No GPU is used by generation.

## Generated contract

- `source/`: byte-identical originals, including archive-only references.
- `records/<tile>.npz`: original regional records and global source-row `int64` IDs.
- `points/<tile>.usdc`: every primary source point with its ID, without collision.
- `meshes/<tile>.npz`: global vertex IDs, local triangle indices and working normals.
- `meshes/<tile>.usdc`: supported triangle meshes with full triangle collision,
  no subdivision and no external artwork/texture dependency.
- `reports/`, `summary.json`: measured record/face counts, boundary edges,
  normal-estimation counts and point-only coverage.
- `world.usda`: both supported surfaces and all source points.
- `streamed.usda`, `points.usda`, `archival-points.usda`: separate surface/point roots.
- `index.json`, `point-index.json`, `world-index.json`: payload paths and measured
  bounds for residency/culling; combined index lists surfaces before points.
- `manifest.json`, `NOTICE`: input hashes, sources, licenses, frame labels,
  reconstruction settings and completion status.

Triangle vertices use source-derived float32 XYZ. Float32 PCD positions retain
their exact bits; higher-precision inputs are converted for USD geometry while
their original values and source bytes remain archived. There is no downsampling,
point omission, decimation, smoothing, fabricated ground or gap filling.
Half-open face-centroid ownership avoids cross-tile duplicate faces. Halo samples
support seams without changing global record IDs.

The output is an open measured-surface map, not a watertight model or a guarantee
of everywhere-drivable ground. Point-only regions have no invented collision.
Unseen building walls, object classes, photorealistic textures and semantic
tree/building models cannot be recovered reliably from arbitrary XYZ alone.

The common Tsudanuma provider additionally authors measured-ground payloads in
`authored/ground/`, without changing the archival BPA surfaces. These use a
connected lower envelope, exact source vertices, upward XY triangulation and
the measured edge/interior support limits. The authored index includes ground
cells and every point payload, even in partially meshed regions.
`authored-scene.json` reports the added ground counts and projected area.
Unlabelled Tsudanuma returns are not replaced with procedural tree proxies.
The independent verifier checks the additional ground source coordinates,
triangle collisions, winding, edge support, ownership and bounds.

## Optional planar presentation

For an authored PCD map, fit planar surfaces and connect bounded coplanar gaps
without changing the measured-surface contract above:

```sh
uv run --extra pipeline python -m usim_env_pipeline.planar --map assets/survey-map --workers 8
uv run --extra pipeline python -m usim_env_pipeline.planar_verify --map assets/survey-map
```

`optimized.usda` composes the authored scene with its original surface, ground
and point groups inactive. It substitutes flat fitted patches plus unchanged
residual source triangles; the original USD layers, source bytes and records
remain available independently. Generation and fitting run on CPU.

Regional RANSAC uses 0.12 m voxel samples only for fitting; original observations
refit each plane with a maximum 0.08 m normal displacement. Each board's nearby
candidate pairs are evaluated concurrently using native NumPy/SciPy operations.
The Gaussian distance cost is the negative log likelihood relative to its peak:

`C = 0.5 * ((d / 0.04)^2 + (g / 1.0)^2 + (angle / 3.0)^2)`

Here `d` is symmetric point-to-other-plane distance in meters, `g` is nearest
sampled boundary distance in meters, and `angle` is normal disagreement in
degrees. Costs rank candidate joins; `C <= 4.5` permits consideration, not
automatic fusion. A common least-squares plane must retain bounded source
support. Every actual boundary correspondence also passes the Gaussian cost.
Accepted measured endpoints seed explicit bridge segments in the common plane.

A 0.15 m plane-local grid uses 0.45 m-radius closing and fills enclosed
interiors up to 256 m². Local repair stays within 0.60 m of measured or
Gaussian-certified seed cells; enclosed completion is separately recorded and
does not use that near-point cap. Open outer boundaries are not extrapolated.
Returns 0.25-2 m below horizontal planes protect downward openings; objects
above a floor do not make that floor an opening. For walls, either side is
checked because the source supplies no reliable sensor-viewpoint orientation.
Depth evidence protects individual cells without rejecting an entire large
hole. This is geometric protection, not a reliable window/door classifier.

Coplanar cell runs become rectangles rather than dense, individually noisy
triangles. Generated corners have fitted-plane provenance, not original
vertex IDs. `optimized/planes/*.npz` records the source support IDs, equation,
occupied/repaired/protected/interior/bridge masks, measured bridge endpoint IDs,
original model IDs and generated geometry. `optimized/joins.npz` records all
candidate costs, original equations and accepted joins. `planar-report.json`
separates enclosed completion, Gaussian bridges and total interpolation.
The verifier independently checks Gaussian costs, source anchors, actual
observed cells, enclosed boundaries, local repair limits, flatness, winding,
area, exact residual vertices and triangle collisions.

## CAD geometry authority

The CAD presentation replaces raster triangles with analytic OpenCASCADE
planar B-Rep faces. It reconstructs oriented polygon boundaries from the fitted
plane archives, not from the old triangle mesh:

```sh
uv run --extra pipeline --extra cad python -m usim_env_pipeline.cad \
  --map assets/survey-map --workers 8
uv run --extra pipeline --extra cad python -m usim_env_pipeline.cad_verify \
  --map assets/survey-map
```

`cad/model.brep` is the canonical geometry in meters. `cad/model.step` exports
a scaled copy with millimeter units for CAD applications. These are trimmed
analytic surface faces, not triangle facets packaged as CAD and not a claim
that incomplete measurements define closed solids.

Internal wires enclosing less than 1 m² are removed irrespective of the old
per-cell protection mask. Larger voids are cut using the kernel's boolean
difference; a void touching the perimeter becomes a notch. Point-touching
raster cycles are separated before CAD construction. Components below 1 m²
are omitted from the CAD presentation and their area is reported.

Descending-area consolidation projects a smaller parallel face only when
normal disagreement is at most 3 degrees and **every boundary vertex** moves
at most 0.15 m. The kernel computes actual overlap, rather than treating
overlapping bounding boxes as containment. Contained aliases are removed;
overlapping faces are fused and same-domain internal boundaries unified.
Distinct floor levels remain separate.

`cad.usda` is a display/collision cache tessellated from those B-Reps. Original
BPA and point layers are inactive in this view but remain unchanged. Every
cache names its CAD authority and keeps the original plane/support IDs in its
sidecar. `cad-report.json` records removed holes, aliases, omitted fragments,
surface area and display triangles. Verification rereads STEP and B-Rep,
checks units, analytic surfaces, kernel validity, remaining hole areas,
source support, exact tessellation and collision APIs.

## VTC integration

`usim_env_vtc.reconstruct` is now a dataset adapter calling this pipeline, rather
than a second reconstruction implementation. Its existing direct-file and module
CLI arguments remain available:

```sh
uv run --extra pipeline python -m usim_env_vtc.reconstruct \
  --source assets/public-tsukuba/map_tc19_o085_f-04_t05.pcd \
  --reference-2018 assets/public-tsukuba/map_tc18_o085_f-04_t30.pcd \
  --output assets/tsukuba-pipeline --workers 4
uv run --extra pipeline python -m usim_env_vtc.verify_map \
  --output assets/tsukuba-pipeline
```

The adapter supplies official checksums/URLs, fuRo attribution and CC BY-NC-SA 4.0,
requires the strict binary XYZINormal schema, preserves supplied normals,
labels the raw 2019 frame and archives 2018 without fusion. Output layout is
compatible with the independent VTC verifier, registration/composition tools
and streaming viewer.

VTC composition, road-mask repair, LiDAR-supported vegetation placement and
[original artwork replacement](vtc/authored-models.md) remain downstream,
dataset-specific steps. The generic point-cloud engine does not pretend that
VTC terrain masks, its validated pose or City Hall model envelopes are available
for another survey.

## License boundary

MIT covers pipeline software, not every input or generated map. Generic maps
default to an **unspecified** data license until you supply one; the tool does
not grant permission or clear source conditions. Retain the generated NOTICE,
manifest, original attribution and applicable license documents when sharing.
VTC maps retain the dataset restrictions described in
[the source audit](vtc/tsukuba-full-map/LICENSES.md).

## Validation

The shared implementation passed 172 pipeline/VTC tests with the optional CPU
dependencies, including PCD field preservation, PLY RGB, normal estimation,
cross-core geometry, point-only regions and the independent VTC verifier.
The base environment passed 283 tests with 19 optional-runtime skips. Ruff lint,
formatting, the source audit and source/wheel builds passed.

Actual CLI runs generated a 36-point XYZ-only map (2 cores, 50 faces) and a
120-record strict VTC fixture (3 cores, 192 faces, 3 point-only records). The VTC
fixture passed independent exact-source-bit verification. Both generated USD
surfaces were reopened and inspected from three directions with CPU rendering.
This validates the shared generation paths; the archived 22-million-point
production map was not regenerated as part of this migration.

The session's editor LSP did not resolve new source-module paths or the optional
CPU SDK imports. Pinned-environment imports and execution passed without adding
type-error suppressions.
