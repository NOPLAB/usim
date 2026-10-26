# Measured building cuboid models

Building replacement is separate from ground reconstruction. It uses the original
PCD and the measured support IDs in `optimized/planes/`, not the previously filled
CAD face boundaries. Run the planar fitting stage first if these archives do not
exist.

```sh
uv run --extra pipeline --extra cad python -m usim_env_pipeline.building_models \
  --map assets/tsudanuma/measured
uv run --extra pipeline --extra cad python -m usim_env_pipeline.building_verify \
  --map assets/tsudanuma/measured
```

Open `assets/tsudanuma/measured/buildings.usda` in Isaac. Open
`building-models/model.step` in a CAD application. STEP uses millimeters; the
canonical `model.brep`, USD and fitting parameters use meters/Z-up.

## Fitting

- Only near-vertical measured planes are candidates. Connected support must span
  at least 3 m horizontally and 2.5 m vertically, with at least 100 points,
  12 square meters of facade extent and 25 percent occupied-cell coverage.
- Nearby coplanar fragments can share a rectangular facade. Disconnected support
  across a larger gap remains separate. `--gap` defaults to 1 m.
- Two approximately perpendicular facades must meet at their observed ends and
  overlap in height before they can support a building envelope. This is not a
  scene-wide bounding box or a transitive fusion of neighboring buildings.
- A single observed facade becomes a closed, bounded wall cuboid, not a building
  with an invented deep footprint. `--wall-thickness` defaults to 0.3 m, centered
  on the fitted facade plane.
- Ground, short objects and sparse components are excluded from fitting.
  These are geometric facade candidates, not semantic building labels.

The model intentionally closes facade holes, including openings inside its
rectangular envelope. Opposite walls, roofs, thickness and enclosed volume are
approximation, not claims of observed architecture. Interior navigation and
preserving entrances require explicit opening/footprint models.

## Composition and provenance

The generated layer inherits `authored.usda`. It never modifies ground payloads
or overrides `/World/Ground`. Only original near-vertical triangles covered by a
selected facade rectangle are replaced. Remaining original geometry is copied
exactly to `BuildingResidual` for affected tiles; unaffected tiles remain
inherited. Original PCD, BPA and earlier views are retained.

Each proxy is a closed six-face OpenCASCADE solid. The same kernel tessellation
is used for display and convex-hull collision. Source points are hidden in this
inspection view rather than deleted. This command does not make the existing
ground smooth or establish robot driveability.

For a separate terrain/building/obstacle simulation presentation without duplicate
archival surface collisions, use [supported simulation-map models](simulation-map.md).
This leaves the building-only authority and its original ground contract intact.

Artifacts beneath `building-models/` include:

- `model.brep` and `model.step`: compound of separate closed solids.
- `models.json` and per-model NPZ: box dimensions, pose, facade/source-plane
  references, exact original support IDs and inferred-geometry labels.
- `facades.json` and `facades/*.npz`: fitted rectangles, source IDs and observed
  coverage, before solid closure.
- `residual/*.npz`: original face indices retained in each replacement tile.
- `ground-source.json`: fingerprints of the original ground payloads.
- `verification.json`: independently measured point-to-proxy surface error and
  ground preservation result.

`building-report.json` records candidate/model counts and replaced versus retained
source triangles. The independent verifier checks source hash/count, facade
support ownership, closed solid boundaries, BRep/STEP volume and units, USD
geometry/collision equality, retained source triangles and unchanged ground.
