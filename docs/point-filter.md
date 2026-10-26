# Point-cloud noise filtering

The first-stage SOR command archives the original PCD/PLY bytes and every decoded
field, then selects working points. It never starts normal estimation,
reconstruction, fitting, physics, or the rest of the map pipeline.

```sh
uv sync --extra pipeline --group dev --frozen
uv run --extra pipeline python -m usim_env_pipeline.point_filter_output \
  --source assets/tsudanuma/measured/source/map_tsudanuma20_cit.pcd \
  --output assets/tsudanuma/point-filter-sor \
  --neighbors 20 --std-ratio 3.0 --workers 4 \
  --license CC-BY-4.0 \
  --attribution "Chiba Institute of Technology, Tsudanuma 2020 dataset" \
  --comparison
```

The output must be new or empty. Existing maps and inspection stages are not
overwritten.

## Decision

For each original record, compute the mean distance to its 20 nearest other
records using the complete cloud, including across tile boundaries. Keep a
record when its mean distance is at most the global mean plus three sample
standard deviations. Apply this rule once, not repeatedly to a shrinking cloud.
Queries use float64 coordinates and bounded CPU batches.

Coincident observations count separately, as in standard record-level SOR.
There is no voxel downsampling, smoothing, plane assumption, or coordinate
movement. Standard SOR can reject legitimate sparse structures and retain dense
noise clusters; the rejected overlay is evidence for reviewing the threshold,
not a ground-truth noise label.

## Saved data

- `source/`: byte-for-byte source archive with SHA256 provenance.
- `records.npy`: all decoded source records and attributes in original order.
- `decisions.npz`: original row IDs, Boolean `keep`, and mean neighbor distances.
- `report.json`: configuration, threshold, retained/rejected counts, quantiles,
  histogram, attribution, and source provenance.
- `comparison.usda`: switchable Before, After, and Removed point groups.
- `before.usda`, `after.usda`, `removed.usda`: independently inspectable views.

Use `records[ids[keep]]` for a later working selection; retain the original IDs.
No geometry pipeline consumes this selection automatically.

## Local Isaac comparison

Use the separate cached Isaac 6.1 environment, after accepting its EULA as
documented in `docs/isaac.md`:

```sh
.venv-isaac-view/Scripts/isaacsim.exe \
  --exec runs/tsudanuma-isaac-local/open_point_filter.py
```

The viewer captures matched overview, reverse, street, and detail viewpoints
for all three modes in `runs/tsudanuma-isaac-local/sor/`. The capture manifest
records camera matrices, visibility, resolution, source provenance, and PNG
paths. Mode-transition captures fence old in-flight camera frames.

Before and After share the source intensity/RGB colors and 0.055 m point width.
Removed displays retained points in gray and rejected points in red. Only the
selected mode's payloads are loaded. The GUI remains open on Removed, with mode
buttons for interactive comparison. Changes are session-only.

Override paths with `USIM_POINT_FILTER_OUTPUT` and `USIM_POINT_FILTER_CAPTURES`;
detail framing uses `USIM_POINT_FILTER_DETAIL_CENTER` (JSON XYZ) and
`USIM_POINT_FILTER_DETAIL_DISTANCE` (meters).

## Retuning and occupied-voxel statistics

Retuning reuses the original global neighbor distances, not a second pass over
retained points. A smaller multiplier strengthens removal; a larger distance
cutoff weakens it. Preserve the earlier output by selecting a new directory:

```sh
uv run --extra pipeline python -m usim_env_pipeline.point_filter_retune \
  --previous assets/tsudanuma/point-filter-sor \
  --output assets/tsudanuma/point-filter-sor-2 --std-ratio 2
```

For the 0.1 m voxel analysis, Before means all original source records and After
means the new selection. Both use the same source-frame origin `(0, 0, 0)` and
half-open cells indexed by `floor(XYZ / 0.1)`, including negative coordinates.
Only occupied cells contribute to each distribution. Coincident source records
count individually; this analysis does not downsample or alter the point cloud.

```sh
uv run --frozen --extra pipeline python -m usim_env_pipeline.voxel_statistics \
  --filter-output assets/tsudanuma/point-filter-sor-2 \
  --destination assets/tsudanuma/voxel-statistics-sor-2
```

The output includes `summary.json`, exact discrete-frequency `before.csv` and
`after.csv`, and `distribution.png`. The chart overlays Before and After using
shared axes: the full distribution uses log-log axes, while the second panel
shows the low-count range with linear axes. Each occupied voxel is one sample;
mean and median are not weighted by its number of points.

## Weak SOR followed by geometric consistency

```sh
uv run --frozen --extra pipeline python -m usim_env_pipeline.geometric_filter_output \
  --previous assets/tsudanuma/point-filter-sor \
  --output assets/tsudanuma/point-filter-geometry
uv run --frozen --extra pipeline python -m usim_env_pipeline.voxel_statistics \
  --filter-output assets/tsudanuma/point-filter-geometry \
  --destination assets/tsudanuma/voxel-statistics-geometry
```

The first stage reuses original neighbor distances with SOR multiplier 4.
Surviving records above the original mean plus one standard deviation are
geometric inspection candidates. Dense records are retained without another
inspection. Neighborhoods are taken from the immutable weak-SOR selection.

The second stage checks robust local plane and line fits at multiple scales.
Query points do not fit themselves. Any consistent scale protects the point;
underdetermined fits and insufficient evidence are retained. Only multiple
confident contradictory scales can reject a point. This is a conservative
experimental filter, not a ground-truth semantic noise detector.

`decisions.npz` adds original-order `sor_keep`, `geometry_rejected`, `evaluated`,
`supported_scales`, and `consistent_scales`. Final `keep` is exactly
`sor_keep & ~geometry_rejected`. The source archive and all original attributes
remain unchanged.

Removed displays SOR rejects in red, geometric rejects in yellow, and final
retained records in gray. The three labels are mutually exclusive and cover
every original row. The density chart retains the same two-panel layout,
overlaying Original, After weak SOR, and After geometry with shared axes.

## SOR and four occupied-voxel thresholds

```sh
uv run --frozen --extra pipeline python -m usim_env_pipeline.voxel_thresholds \
  --source-output assets/tsudanuma/point-filter-geometry \
  --output assets/tsudanuma/voxel-threshold-comparison
uv run --frozen --extra pipeline python -m usim_env_pipeline.voxel_threshold_usd \
  --output assets/tsudanuma/voxel-threshold-comparison
```

Only the saved `sor_keep` selection is used, never the geometric selection.
The four independent conditions remove whole occupied 0.1 m cells containing
5, 10, 20, or 40 points or fewer. Counts are computed once on the same SOR
population; conditions are not applied cumulatively. SOR rejects have count
zero in the original-order `counts_per_record` vector.

Five modes show the original locations: SOR only, Voxel5, Voxel10, Voxel20,
and Voxel40. SOR rejects are always red. Additional cell-population rejects
are orange, yellow, green, or blue respectively; retained points are gray.
Positions and original IDs share one USD payload set between all modes.
The five exact count-frequency CSVs and `distribution.png` use the same
full-range log-log and low-count linear chart format.

For the existing viewer, set `USIM_POINT_FILTER_MODES` to the JSON list
`["SOR","Voxel5","Voxel10","Voxel20","Voxel40"]`, final mode to `SOR` with
`USIM_POINT_FILTER_FINAL_MODE`, and provide a color legend through
`USIM_POINT_FILTER_LEGEND`. Point-filter output and capture directories remain
controlled by `USIM_POINT_FILTER_OUTPUT` and `USIM_POINT_FILTER_CAPTURES`.

## Live voxel sliders

```sh
.venv-isaac-view/Scripts/isaacsim.exe \
  --exec runs/tsudanuma-isaac-local/open_live_voxel.py
```

The live window has grid-size **0.001-10.0 m** and inclusive point-count
threshold **0-100** sliders. The adjacent numeric fields share the same bounded
models. Grid input is rounded to 0.001 m resolution. Threshold zero disables
additional voxel rejection but leaves SOR rejection enabled.

Colors remain red for SOR rejects, orange for live voxel rejects, and gray for
retained records. Geometry rejection is not applied. The controller runs one
CPU worker and publishes only the latest requested grid/threshold. Grid changes
use collision-free packed integer keys; threshold changes reuse cached counts.
Updating colors yields between tile batches so the viewport remains usable.
All edits live in the USD session layer; original source files and saved
threshold outputs are unchanged.

For the full 13.6 million-record source, the measured grid recount is about
1.6-1.9 seconds, plus about 0.8-1.4 seconds to reflect colors. Threshold-only
changes reflect in about 0.65-0.9 seconds. This is asynchronous interactive
updating, not a 60 Hz full-cloud recount. Current count and paint timings are
shown in the window and recorded in
`runs/tsudanuma-isaac-local/live-voxel/history.json`.

## Raw-point local density comparison (LOF)

LOF compares each observation's local reachability density with its neighbors,
instead of applying one absolute point-density cutoff to the whole map.
Compute it on **all original records**, without SOR, a voxel grid, downsampling,
coordinate movement or a forced removal percentage:

```sh
uv run --frozen --extra pipeline python -m usim_env_pipeline.live_lof \
  --records assets/tsudanuma/voxel-threshold-comparison/records.npy \
  --cache assets/tsudanuma/lof-comparison --k 20 40 80 \
  --batch-size 8192 --workers 8
.venv-isaac-view/Scripts/isaacsim.exe \
  --/app/window/title="Tsudanuma LOF Density Comparison" \
  --exec runs/tsudanuma-isaac-local/open_live_lof.py
```

The viewer has k=20/40/80 buttons and a 1.00-10.00 score cutoff, initially
1.50. A point is removed only when `LOF > cutoff`. Red marks LOF rejection;
gray marks retention. Optional cyan marks observations rejected by the saved
weak SOR but retained by LOF; it is a comparison, not a correctness label.
The original-cloud toggle restores every observation. "Hide rejected points"
actually subsets displayed source positions and IDs in the session layer.

Calculation uses exact Euclidean CPU queries in bounded batches, excluding the
query row while counting coincident observations separately. The density
denominator includes the conventional `1e-10` stabilizer. Queries are repeated
between density passes rather than persisting an N-by-k graph. The completed
cache saves float64 scores and float32 inspection distances, about 490 MB for
13,611,640 rows and three k values; inspection precision never affects LOF.
Source SHA256, algorithm version and ordered k values identify each cache.
Threshold and k switching reuse the completed scores without neighbor searches.

Launch the viewer separately from any existing editor. Original records,
attributes, source stages and previous outputs are unchanged; only the new
viewer session is edited. QA states and timings are under
`runs/tsudanuma-isaac-local/live-lof/`. Set `USIM_LIVE_LOF_QA=1` for native
model-callback checks and matched-state captures, including actual point hiding.

LOF can retain coherent sparse surfaces, but may reject valid density
transitions and retain a dense isolated cluster. Its relative-density score
is not a semantic noise or measurement-quality ground truth.

## Iterative LOF

Each iteration builds fresh neighborhoods on the preceding iteration's retained
observations. It does not apply the original scores repeatedly. The same k and
strict `LOF > threshold` rule apply throughout one run. Specify the requested
iteration count explicitly:

```sh
uv run --frozen --extra pipeline python -m usim_env_pipeline.iterative_lof \
  --records assets/tsudanuma/voxel-threshold-comparison/records.npy \
  --cache assets/tsudanuma/lof-comparison --k 20 --threshold 1.5 \
  --iterations 3 --batch-size 8192 --workers 8 \
  --output runs/tsudanuma-isaac-local/iterative-lof
.venv-isaac-view/Scripts/isaacsim.exe \
  --/app/window/title="Tsudanuma Iterative LOF Comparison" \
  --exec runs/tsudanuma-isaac-local/open_iterative_lof.py
```

The iteration count is not limited to three. Computation stops early when no
points are removed (`converged`) or when at most k points remain
(`insufficient_points`); it never silently lowers k. `result.json` reports the
requested count, actual passes, stop reason and each pass's additional removal.
`removed_at.npy` is an int32 array in **original record order**: zero means
retained, and a positive value is the iteration that first removed that point.
Original coordinates and attributes are untouched.

The viewer provides k, threshold and iteration-count fields and an explicit
Compute button. Changing fields alone does not reinterpret an older result.
Selecting pass zero shows the original cloud; selecting a later pass shows its
cumulative removals and highlights points newly removed in that pass.
Hide rejected points actually removes their displayed positions and IDs.
New calculations run on CPU while the native window remains responsive.
Uncached passes require neighbor searches; they are not instantaneous slider
updates. Cached scores are isolated by source hash, k, algorithm version and
survivor-ID hash. Changing a threshold therefore recomputes scores whenever it
changes the survivor neighborhoods.

Defaults are k=20, threshold=1.5 and three requested iterations. Override with
`USIM_ITERATIVE_LOF_K`, `USIM_ITERATIVE_LOF_THRESHOLD` and
`USIM_ITERATIVE_LOF_ITERATIONS`. Evidence defaults to
`runs/tsudanuma-isaac-local/iterative-lof`; override it with
`USIM_ITERATIVE_LOF_EVIDENCE`. Set `USIM_ITERATIVE_LOF_QA=1` for native
model-callback verification and captures. Launch separately to preserve any
existing editor and its unsaved session.

Iteration can expose additional outliers, but can also progressively erode
legitimate boundaries. More passes do not establish better measurement quality.

## Floating-island candidate comparison

This comparison is currently **on hold**. User trials found legitimate
disconnected structures (including outside the map area and indoors) among
its geometric removal candidates. Do not apply its removal mask to the
following edge-extraction input: use the completed iterative LOF selection only.

Use a frozen original-ID selection from a **completed** iterative LOF run.
The comparison does not modify original observations, restart model generation,
or apply triangle-model ground-contact rules to a sampled point cloud.

For the current Tsudanuma comparison, `prepare_point_islands.py` snapshots the
last completed applied settings in `iterative-lof-reopened/result.json` and
verifies their retained count and source fingerprint. It saves increasing
original IDs, source XYZ and `input.json` in
`runs/tsudanuma-isaac-local/point-islands/`. Changing the earlier LOF GUI later
does not change this input.

```sh
.venv-tsudanuma-fix/Scripts/python.exe \
  runs/tsudanuma-isaac-local/prepare_point_islands.py
uv run --frozen --extra pipeline python -m usim_env_pipeline.point_islands \
  --records assets/tsudanuma/voxel-threshold-comparison/records.npy \
  --source-ids runs/tsudanuma-isaac-local/point-islands/source_ids.npy \
  --cache assets/tsudanuma/point-islands --anchors 59854 94908 \
  --radius 0.2 --max-diameter 1.0 --min-body-gap 0.4 --workers 8 \
  --output runs/tsudanuma-isaac-local/point-islands
.venv-isaac-view/Scripts/isaacsim.exe \
  --/app/window/title="Tsudanuma Point Islands" \
  --exec runs/tsudanuma-isaac-local/open_point_islands.py
```

The graph compares closed Euclidean connection distances of 0.1, 0.2 and
0.4 m in source coordinates. It counts coincident records separately but does
not use point density or a minimum number of neighbors to reject observations.
Spatial cells have diagonal smaller than 0.1 m, so their observations are
guaranteed connected. Cross-cell edges are checked using the original float64
point-distance predicate `squared_distance <= radius_squared`.
Neighbor queries and native local-component merges are streamed; no full
radius adjacency matrix, neighbor-count truncation or cell-center approximation
is used. A count-only preflight adapts batch sizes to an explicit edge budget.
Radii, source SHA256, selected-ID SHA256 and algorithm version isolate caches.

The retained side starts from actual original observation IDs, not simply the
largest cluster or a flat ground plane spanning the map. IDs 59854 and 94908
are initial references from previously inspected low horizontal patches;
they are editable assumptions, not semantic ground-truth labels.

Classification:

- **Gray:** connected to a retained-side reference at the selected distance.
- **Red:** still unsupported at the largest distance, whose largest-distance
  parent component has bounding-box diagonal at most the maximum candidate
  diameter and exact point-to-body gap at least the minimum body gap.
- **Yellow:** not currently supported, but does not meet every candidate rule.
  This includes fine-distance fragments that reconnect at a larger distance,
  large disconnected objects, and optionally protected planar/linear components.

The planar guard uses at least 12 observations and smallest/middle covariance
eigenvalue ratio <= 0.04, excluding the linear case. The linear guard uses at
least six observations and middle/largest ratio <= 0.04. These counts establish
shape evidence; count alone never makes a component removable. The guards
operate on the largest-distance parent, so fragmentation cannot turn parts
of a larger object into small removal candidates.

Radius buttons use applied settings. Edit anchor IDs, maximum candidate
diameter, minimum body gap or shape protections, then press **Apply**.
Incomplete field edits do not change the last result. **Hide candidates**
removes only red points from the displayed session and preserves yellow.
**Original LOF subset** restores all frozen LOF survivors, not observations
already removed by LOF.

Inspect a component number or locate an original source ID to see its point
count, XYZ dimensions, bounds, bounds-diagonal diameter, exact nearest-body
distance, support at both selected/largest distances and shape ratios.
Selected points are cyan. **Add selected component to anchors** adds an actual
member source ID and recomputes support; it is not just a color override.

`USIM_POINT_ISLANDS_EVIDENCE` selects the evidence directory, defaulting to
`runs/tsudanuma-isaac-local/point-islands-live`. `USIM_POINT_ISLANDS_QA=1`
exercises native controls and exact ID/position/color correspondence with
matched-render camera captures. Original records, previous outputs and
existing editor sessions are not overwritten.

Red means a conservative geometric candidate, not proof that a measurement is
noise or physically unsupported. Review the comparison before making any
destructive export decision; this stage only classifies and hides points.

## Edge points from iterative LOF only

Freeze the last completed LOF selection and compute CPU edge features:

```sh
.venv-tsudanuma-fix/Scripts/python.exe \
  runs/tsudanuma-isaac-local/prepare_point_edges.py
.venv-isaac-view/Scripts/isaacsim.exe \
  --/app/window/title="Tsudanuma LOF Point Edges" \
  --exec runs/tsudanuma-isaac-local/open_point_edges.py
```

`point-edges/input.json` records the applied LOF settings, full source SHA256,
ordered selected-ID fingerprint and `island_filter_applied: false`.
The selection remains fixed when a different editor changes its controls.
`edge_source_ids.npy` identifies selected edge observations without moving,
averaging or deleting original observations; all original attributes remain
available by source ID. No SOR, voxel reduction or floating-island mask is added.

The default estimator uses up to 64 other observations inside a closed 0.3 m
radius, with at least eight neighbors. Local covariance in relative coordinates
estimates the tangent plane and an unoriented normal. Smallest-eigenvalue /
sum-eigenvalues <= 0.08 and middle/largest > 0.02 establish surface evidence.
Collinear, coincident, insufficient or volumetric neighborhoods do not establish
a reliable query surface. Supplied source normals are not assumed reliable.

- **Cyan:** maximum cyclic angular gap in tangent-plane neighbor directions
  is at least 135 degrees: a sampled surface boundary.
- **Orange:** neighboring reliable surface normals change by at least
  35 degrees relative to their dominant unoriented direction. The third-largest
  change is used and at least max(6, minimum support) reliable neighbors are
  required. Normal signs do not matter. The query's own normal can be unreliable
  at a sharp corner without suppressing the surrounding two-face evidence.
- **Magenta:** both feature conditions. **Gray:** remaining LOF observations.

These are geometric **edge points**, not fitted line segments, a connectivity
graph of architectural edges, or semantic noise labels. Sample holes and
occlusion boundaries may also have angular gaps. Vegetation and curved surfaces
may have normal changes. The neighborhood is explicitly capped kNN inside a
physical radius, not an exhaustive radius-neighbor graph.

Original LOF (gray), overlay and Edge only compare the same fixed source IDs.
The three class toggles select exclusive boundary-only, crease-only and both
classes. Angle Apply reuses cached features. Neighborhood Apply recomputes or
loads a distinct cache keyed by source, selected IDs, algorithm and all
neighborhood parameters. The previous result remains displayed during CPU work
and invalid input preserves it. Inspector reports original XYZ, support,
surface variation and both angle scores.

The CPU estimator streams queries twice with O(N) state and O(batch size * k)
temporary memory, rather than storing an N-by-k graph. Only the native display
uses the local GPU. Source stages and earlier editors remain untouched; the
new viewer edits its own USD session layer.

`USIM_POINT_EDGES_EVIDENCE` controls the evidence directory (default
`runs/tsudanuma-isaac-local/point-edges-live`). `USIM_POINT_EDGES_QA=1` exercises
original/overlay/edge-only modes, separate classes, stricter angles, valid
neighborhood changes/restoration, invalid inputs and diagnostic source IDs.
Each publication asserts actual USD IDs, unchanged positions and exact colors.
Capture requests fence against a distinct camera matrix before readback.

## Observed structural edge lines

The point-feature prototype above is not a finished line extraction. User
trials found scattered candidates, and asked for actual lines and cleaner
structure. The line stage fits observed plane boundaries and supported
intersections, rather than joining those candidate points indiscriminately.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-tsudanuma-fix/Scripts/python.exe \
  runs/tsudanuma-isaac-local/prepare_edge_lines.py
.venv-isaac-view/Scripts/isaacsim.exe \
  --/app/window/title="Tsudanuma LOF Edge Lines" \
  --exec runs/tsudanuma-isaac-local/open_point_edge_lines.py
```

On Windows, set those environment variables in PowerShell before the CPU
command. Native loading requires the completed default cache; it does not run
RANSAC inside Isaac. Input remains the full frozen LOF selection, without an
island removal mask.

All observations are visited in source-anchored 8 m fitting cores. Plane
discovery retains the 0.6 m halo; physical-normal context extends by
`max(halo, surface_radius + voxel)` (0.7 m by default). Normal probes are
restricted to the core, so their complete radius neighborhoods and voxel
margin fit inside the supplied context. Moving the ownership grid does not
change the plane-discovery partition. Plane
hypotheses use actual original representatives in 0.1 m cells and are refitted
and revalidated on original observations within 0.05 m. No original point is
averaged, moved, or removed. A plane also requires at least 1 m² of observed
footprint and physical-scale surface evidence from the complete local cloud.
Up to 128 deterministic plane probes use full 0.6 m neighborhoods of original
representatives, rather than a fixed nearest-24 cap. At least 45% must have
PCA surface variation <= 0.05, noncollinear support and normal agreement within
20 degrees. Uniform volumetric scatter cannot establish a face merely because
RANSAC found points in a thin slab.

The physical scale matters: a real facade's reliable fraction measured 18.75%
with nearest-24 neighborhoods but 82.81% with complete 0.6 m neighborhoods.
The small capped neighborhoods were measuring registration thickness rather
than the wall. `probe_face_scale.py` and `probe_line_gates.py` record the
diagnostic comparisons; they are not full-map quality proofs.

Intersection lines require a dihedral angle >= 25 degrees and independent
original point strips from both faces. Ambiguous points cannot count as both
faces. Source support lies within 0.15 m of the fitted line; each face must
occupy multiple transverse cells. Longitudinal bins and actual projected
observation gaps trim and split finite segments. No segment bridges an
unsupported span > 0.25 m. Original support coverage must be at least 75%.
Each v4 cache also stores the original longitudinal bin origin. Revalidating
only the saved finite witnesses must retain this phase, rather than reanchor
bins to their trimmed minimum; otherwise occupancy can change the endpoints.

Observed boundary lines verify proposal points against complete plane-local
0.3 m neighborhoods, with angular vacancy >= 150 degrees and boundary tangent
agreement within 15 degrees. Seeded 2D RANSAC and orthogonal refitting produce
straight segments backed by those original boundary observations. Convex
hulls, repaired footprints and bounding-box wireframes are not edge sources.
Complete halo neighborhoods and core ownership avoid treating crop perimeters
as observed structural boundaries.

The independent viewer draws **linear BasisCurves**, with cyan observed
boundaries and orange supported intersections. The gray point cloud keeps all
frozen LOF IDs even in lines-only mode (visibility only). Minimum length,
coverage, maximum gap and line width filter the cached evidence. Defaults are
1.5 m, 80%, 0.25 m and 0.06 m. These are visibility filters, not point deletion.
Inspection reports fitted endpoints, support counts, observed gaps, equations
and actual raw witness IDs; Frame selected centers a closeup.

Line endpoints are fitted derived geometry, not claimed unchanged original
observations. Source records and attributes remain the lossless reference.
Occlusion boundaries can resemble physical openings; curved surfaces and weak,
sparse or unobserved edges may be omitted. Unsupported corners are not snapped
or filled to complete a wireframe.

`USIM_POINT_EDGE_LINES_EVIDENCE` selects evidence (default
`runs/tsudanuma-isaac-local/point-edge-lines-live`).
`USIM_POINT_EDGE_LINES_QA=1` validates actual curve endpoints, two-vertex counts,
uniform colors/widths, line IDs, original source IDs/positions and visibility;
it captures original/overlay/lines-only, separate types, filters, widths,
inspection and several views. Automation supersession is recorded as a failed
QA run without terminating the interactive controller or restoring user edits.

The 2026-10-14 v4 evaluation retains 182 supported intersections, zero
boundaries and 115 default-visible lines. Source/finite-support audits and
native display checks pass, but quality targets fail: confirmed held-out
creases and boundaries have zero recall. Boundary extraction is not complete.
Use the fixed baseline/new tables and same-pose captures recorded in
`runs/tsudanuma-isaac-local/POINT_EDGE_LINES_RESULTS.md`, not the line count,
to assess the result.

The frozen CPU and test conditions require `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1`. `uv run --extra pipeline pytest -ra` supplies
the existing pipeline dependencies; dev-only pytest cannot collect all
pipeline tests. Unrestricted BLAS threading still fails the noisy
ownership-grid invariance regression and is recorded as an unresolved
numerical limitation, not silently treated as a passing check.
