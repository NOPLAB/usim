# Straight outlines and continuous curved shapes

Use the separate fitted presentation when the original measured detail is too
fragmented. The baseline `simulation.usda`, its one-mesh floor, original source
archives and density cleanup are preserved. In the fitted presentation, inferred
building cuboids are inactive: filled rectangles and unobserved faces can create
walls in empty space.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  .venv-tsudanuma-fix/Scripts/python.exe -m usim_env_pipeline.simulation_fit \
  --map assets/tsudanuma/measured
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  .venv-tsudanuma-fix/Scripts/python.exe -m usim_env_pipeline.simulation_fit_verify \
  --map assets/tsudanuma/measured
```

Open `assets/tsudanuma/measured/simulation-fitted.usda` for the fitted result.

## Straight planar outlines

Measured, depth-protected plane footprints are converted to exact cell unions.
Douglas-Peucker contour candidates replace stair-stepped boundaries with fewer
straight segments. Simplification stays within 1.5 cells in both continuous
outline directions: 0.15 m at the current resolution. Original opening centers,
disconnected components and their contact relationships are preserved. Small
already-straight triangular/rectangular rings are retained.

Native constrained Delaunay triangulation respects holes. The resulting polygons
are extruded as closed 0.16 m slabs. Fitted area, deviation, mesh and original
authority hashes are recorded separately.

## Curved residual shapes

The occupied residual cells support a smooth implicit envelope, rather than
millions of individually rounded cube faces. A distance field is truncated at
0.4 m, Gaussian-fitted at a 0.1 m scale, and contoured at a 0.2 m envelope radius.
Marching cubes creates the continuous curved shape; area-weighted vertex normals
make the rendered curvature continuous.

Fifteen-metre core tiles have overlapping support halos. Global float64 edge
interpolation makes shared vertices agree. Empty neighboring tiles are considered
when a supported envelope crosses a boundary, avoiding clipped outer surfaces.
After conversion to USD float32 coordinates, identical vertices are welded and
zero-area triangles are removed before publishing render and collision geometry.
Concave smoothing can expand a cavity boundary beyond the nominal radius.
Vertices beyond 0.299 m from the nearest occupied-cell center are projected back
to that support radius, leaving a 1 mm margin for float32 publication within the
verified 0.3 m bound. The same bound is applied to both render and collision.
Equidistant supporting centers are selected in coordinate order, independent
of tile-local search-tree ordering, to preserve shared seam coordinates.
Large unsupported gaps remain open; small gaps in dense detail can merge as part
of the approximation. This is not exact leaf/wire geometry or a semantic tree fit.

## Presentation and verification

Building facades instead use the occupied 0.1 m cells projected from their
original source records, with bounded contour fitting and 0.16 m slab thickness.
Empty regions and openings are retained rather than filled from a bounding
rectangle. No roof or opposite wall is inferred from a measured corner.
Run `python -m usim_env_pipeline.simulation_facades --map assets/tsudanuma/measured`
to update just these facades without refitting the unchanged planes and curves.

The fitted layer inherits the unchanged floor. The earlier
voxel surfaces, raw residual triangles and pixel-outline slabs are inactive only
in this presentation, so they do not obscure or duplicate the fitted shapes.
The original presentation remains available for comparison.

Render and static triangle collision use the same fitted mesh. The verifier
checks original fingerprints, baseline preservation, the unchanged floor,
supported polygon fitting, positive slab volume, curve support distance and
occupied-cell coverage through independent signed-distance queries.
For facades it reconstructs occupied keys independently from original point IDs
and checks published vertices, triangle centers and edge midpoints against the
measured footprint. Inferred building render and collision are both inactive.
USD payloads are loaded and unloaded individually during verification. Run this
full audit before starting the local Isaac viewer, rather than alongside it.

Optional pipeline dependencies are pinned in `pyproject.toml`: Shapely performs
topology-preserving contour fitting/triangulation; scikit-image performs
isosurface extraction. All geometry work is CPU-only.

For local paired camera captures use `USIM_SIMULATION_STAGE=simulation-fitted.usda`
and `USIM_CAPTURE_PREFIX=fitted` with `runs/tsudanuma-isaac-local/open_simulation.py`.
For actual physics checks use the same stage selector and
`USIM_PHYSICS_RESULT=fitted-physics.json` with `probe_simulation_physics.py`.

## Remove floating components

Create an additional presentation containing only fitted components connected to
the actual finite ground mesh:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-tsudanuma-fix/Scripts/python.exe -m usim_env_pipeline.simulation_grounded \
  --map assets/tsudanuma/measured
```

Open `assets/tsudanuma/measured/simulation-grounded.usda`. The unfiltered
`simulation-fitted.usda`, original payloads, source archives and floor are retained.
Run this CPU job before opening the local viewer.

The filter labels disconnected islands inside each payload, welds identical
published world coordinates across tile boundaries, and follows actual triangle
surface intersections from the floor. Indirect chains such as floor, wall,
column, roof and branch survive. Coplanar overlap and contact inside a large
triangle are tested by Open3D's native intersection predicate. Spatial bounds
only select candidates; proximity alone does not create a connection.

Connectivity is defined on triangle surfaces, with no deliberate gap allowance
or volumetric containment rule. This differs from density cleanup, which can
preserve spatially supported but disconnected foliage. The ground is its finite
mesh, not an infinite plane or a half-space under the terrain.

The chosen contact allowance is **0 m** beyond the native triangle-intersection
predicate's floating-point arithmetic. Fitted tile seams already share identical
published coordinates; the 0.1 m source-cell size and 0.3 m fitting envelope are
geometry approximation bounds, not permission to bridge a real air gap. This
keeps a nearby but disconnected object from becoming a false support. Regressions
cover exact negative-coordinate seams, positive gaps, interior and coplanar
contacts, indirect support, and separate islands in one payload.

Original triangles are kept whole. Partial payloads contain only retained
components, preserving coordinates, winding, normals, colors and transforms.
Rendering and static collision use these same triangles. Small spatial chunks
and bounded caches avoid loading the entire USD scene at once.

`grounded-models/report.json` records removed component and triangle counts.
`connections.npz` records floor-rooted contact witnesses, and `verification.json`
records checks of source fingerprints, exact retained triangles, whole-component
selection, the one unchanged floor, and render/collision equality. Repeat the
publication audit with:

```sh
.venv-tsudanuma-fix/Scripts/python.exe -m usim_env_pipeline.simulation_grounded_verify \
  --map assets/tsudanuma/measured
```

For local captures and physics probes, select
`USIM_SIMULATION_STAGE=simulation-grounded.usda`.

Keep an already open editor and its unsaved session unchanged. Launch a separate
Isaac process after the CPU audit, with a distinct capture prefix so previous
evidence is retained. For the local runtime with its already accepted license:

```powershell
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$env:USIM_SIMULATION_STAGE = 'simulation-grounded.usda'
$env:USIM_CAPTURE_PREFIX = 'floating_final_20261013'
$env:USIM_QA_KEEP_OPEN = '1'
.venv-isaac-view/Scripts/isaacsim.exe `
  --/app/window/title='Tsudanuma Floor Connected Models' `
  --/persistent/physics/cooking/ujitsoCookingMaxProcessCount=1 `
  --/physics/cooking/ujitsoCollisionCooking=true `
  --/persistent/physics/numThreads=2 `
  --exec runs/tsudanuma-isaac-local/open_simulation.py
```

The helper captures overview, opposite overview, detail, facade, street and
opposing building-close views. Its cameras and lighting are session-layer edits,
and the result remains open at the overview camera. It does not save or replace
the stage in another running editor.

## Angular presentation without FittedPlanes

The angular view replaces the retained residual curves with constrained low-poly
polyhedra and flat face normals. It deactivates `FittedPlanes` for both rendering
and collision, preserves measured `FittedFacades` and the one-mesh floor, and keeps
the original point cloud under the initially inactive `PointRegions` root.

```sh
.venv-tsudanuma-fix/Scripts/python.exe -m usim_env_pipeline.simulation_angular \
  --map assets/tsudanuma/measured --cell 0.3 --name polyhedral
.venv-tsudanuma-fix/Scripts/python.exe -m usim_env_pipeline.simulation_angular_verify \
  --map assets/tsudanuma/measured --name polyhedral
.venv-tsudanuma-fix/Scripts/python.exe -m usim_env_pipeline.simulation_grounded \
  --map assets/tsudanuma/measured --angular --angular-name polyhedral
```

Open `assets/tsudanuma/measured/simulation-polyhedral-grounded.usda` in Isaac Sim.
The last command recomputes floor reachability against the actual angular view,
without treating the excluded planes as supports. The original fitted, grounded,
point-cloud and angular source layers remain unchanged. Repeat the final audit:

```sh
.venv-tsudanuma-fix/Scripts/python.exe -m usim_env_pipeline.simulation_grounded_verify \
  --map assets/tsudanuma/measured --angular --angular-name polyhedral
```

Geometry reduction uses MeshLab's topology-preserving quadric edge collapse.
Immutable contact triangles are excluded from each movable patch, and patch
boundary simplification is disabled. The target triangle count follows patch
area and the requested face scale; constraints can prevent reaching that count.
Vertices are chosen from the existing geometry rather than moved to QEM optima.
Tile boundary vertices, saved contact triangles, and triangles incident to exact
cross-payload shared vertices remain fixed. Components whose candidates change
topology, seams, contact triangles or closed-volume orientation retain their
reference geometry. Uniform face normals accompany actual triangle reduction.
Existing nonmanifold neighborhoods are also fixed. If a native candidate
introduces an exceptional edge, the corresponding original neighborhood is
added to the fixed patch before another reduction attempt. Constraints only
grow, so exceptional local detail does not force an otherwise safe large body
to revert wholesale.
The separate `polyhedral` artifact name preserves earlier `angular` stages and
any open candidate GUI. The native filter options are documented in the
[PyMeshLab filter reference](https://pymeshlab.readthedocs.io/en/latest/filter_list.html).

`polyhedral-models/verification.json` records actual triangle reduction, unchanged
contact patches, seams, component topology, source hashes, triangle collisions and
the availability of original point payloads. It does not certify absence of global
self-intersections or preservation of continuous clearance. The subsequent
`polyhedral-grounded-models/verification.json` certifies floor-connected triangle
support after excluding the planes. Facade gaps are not filled with convex hulls.

To compare the measurement, activate `/World/PointRegions` in the Isaac Stage
tree. Its point payloads are already present; no import is required. It starts
deactivated, does not participate in triangle collision, and can be deactivated
again after inspection. Leave `FittedPlanes` deactivated.

For final captures and physics probes, use
`USIM_SIMULATION_STAGE=simulation-polyhedral-grounded.usda` with the local scripts
described above.
