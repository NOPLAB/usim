# Supported simulation-map models

`buildings.usda` preserves the original ground and unrelated BPA surfaces. Use
`simulation.usda` for the derived simulation presentation: independent terrain,
closed building models, source-backed planar slabs and occupied obstacle volumes.
No original PCD, BPA, authored-ground or building-authority file is rewritten.

```sh
uv run --extra pipeline python -m usim_env_pipeline.simulation_map \
  --map assets/tsudanuma/measured --voxel-cell 0.1 \
  --cleanup-radius 0.4 --cleanup-min-cells 5
uv run --extra pipeline python -m usim_env_pipeline.simulation_verify \
  --map assets/tsudanuma/measured
```

Generate the measured ground and independent building models first. Geometry
generation is CPU-only. Dependencies remain in the existing `pyproject.toml`
pipeline extra; there are no new native CAD or GPU requirements for this step.

## Ground

The derived ground is one continuous USD triangle mesh with shared vertices on
a 0.5 m lattice, upward vertex normals for smooth shading, and identical static
triangle collision. It is not split into 30 m output tiles. Its footprint consists
of lattice cells whose centers lie inside the convex XY hull of the authoritative
measured ground's occupied cells. This includes every measured boundary cell.
Sampling gaps and disconnected measured patches inside that
footprint are filled; the exterior is still bounded by the measured floor domain.
The rasterized exterior can differ from the hull by up to one cell diagonal.

Local least-squares neighborhoods use up to 48 nearest source-ground samples
within 0.75 m. Trusted anchors lie in their local support hull, within 0.6 m of
measured support, with at least six noncollinear samples, maximum 0.12 m residual,
gradient at most 0.35 and compatible one-sided predictions. Incompatible local
levels do not become anchors, but no longer cause missing floor cells.

A screened harmonic solve minimizes anchor-height error and neighboring
residual-height differences relative to a fitted reference plane. This preserves
the measured overall slope, smooths sampling noise and level transitions, and
interpolates missing nodes. Inferred nodes have only `-1` support IDs: they are
generated estimates, not new measurements. Their count and the maximum distance
from triangle centers to measured ground are reported. The entire mesh must be
connected, have one exterior boundary and no interior holes. Excessive slopes
increase the global smoothing strength in powers of four until every triangle
satisfies the 0.35 slope bound, rather than deleting cells. The selected strength
and maximum anchor-height adjustment are recorded. A measured reference plane
itself exceeding that bound fails generation.

This intentionally approximates a continuous simulation floor, including gaps
under buildings and smoothing across small measured ground steps. It does not
certify the inferred floor as measured or all-site driveability. Original
ground and source archives are unchanged. Authoritative floor records and their
bounded measured-floor neighbors remain ground after smoothing; they cannot
reappear as voxel obstacles or raw floor triangles. A neighbor must be within
0.6 m in XY and 0.08 m in elevation of its nearest original ground sample.
Other records can be assigned to the floor within 0.08 m of the new triangular
height field. The maximum measured-to-smoothed height change is reported, rather
than falsely claiming exact measured elevations. Building ownership can still
supersede floor ownership inside its bounded volume.

## Buildings and remaining objects

Existing closed OpenCASCADE building models are reused unchanged. A source point
must be inside the actual oriented cuboid or within the declared 0.2 m Euclidean
approximation tolerance; its expanded axis-aligned bounds are not enough.

Other measured plane supports become 0.16 m closed polygonal slabs over 0.1 m UV
footprints. They use original plane equations and source IDs, not old inferred
archive masks. Bounded local completion closes small sampling defects, while
depth-supported openings and large passages remain empty. Additional residual
planes are detected with distance, normal, coverage and connectivity gates.
Ground-like horizontal planes are excluded; low horizontal structures need a
nearby measured lower-ground reference and more than 0.25 m clearance.
See [supported shape approximation](shape-approximation.md) for the adopted
WingEarth workflow ideas and public algorithm references.

Remaining source returns retain the authority of occupied 0.1 m cells with only
exterior faces, one quarter of the previous 0.4 m width. The displayed/collidable
surface is locally Taubin-smoothed, with fixed tile seams and a 0.02 m vertex
movement cap; its original connectivity and holes are preserved.
Before slab or obstacle
ownership, candidate cells with fewer than five distinct occupied cells within
0.4 m are removed as isolated noise candidates. The queried cell counts as one.
Counts use the original candidate population in a single pass, not iterative
erosion. Repeated returns in one cell cannot inflate spatial density. Disconnected
leaves can support one another, and single-return cells survive when their
neighborhood is dense enough.

`--voxel-cell`, `--cleanup-radius` and `--cleanup-min-cells` control those values.
The density radius is in meters, independent of voxel width. A minimum of one
retains all candidate cells. This is a density heuristic, not a semantic foliage
classifier; genuinely sparse objects can also be classified as isolated.
No component convex hull, semantic tree inference or extrusion to the floor is
used. Bounded planar completion is explicitly recorded; the irregular occupied
cell set is not filled. Its smoothed approximation can extend by up to
`sqrt(3)*0.1 + 0.02`, about 0.193 m, from a source return; narrow
passages near that resolution are not certified navigable.

Every original BPA triangle is separately accounted for. Original floor triangles
whose vertices all belong to the ground authority are replaced by the smooth
floor deliberately. Other replacement requires its whole footprint to be bounded
by a model, not merely its vertices or centroid.
Triangles touching a discarded noise record are separately marked as
removed noise, so the archival BPA cannot reintroduce a cleaned block.
Other uncovered triangles remain collidable, with exact original vertices
and IDs, under `SimulationUnresolved`. Consequently the output is not a claim
that every uncertain shape has been reconstructed or that all terrain is drivable.

## Composition and evidence

Only the new simulation layer deactivates `Regions`, `PointRegions`, original
`Ground`, `BuildingResidual`, `InferredStructures` and duplicate vegetation.
Their archival files remain accessible. New geometry uses the same mesh for
rendering and static triangle collision; obstacle holes are not filled by
convex-hull collision.

Artifacts under `simulation-models/` include:

- `terrain.npz`: continuous lattice, filled cells, measured-anchor IDs and explicit inferred nodes.
- `planes/*.npz`: plane archive ownership, observed/filled footprints and closed slabs.
- `primitive-planes/`: separately detected source-backed primitive hypotheses.
- `approximation.json`: repair parameters and inferred planar cells.
- `obstacles.npz`: occupied coordinates, source IDs and counts.
- `cleanup.npz`: original candidate cells, local occupied-cell counts and retained flags.
- `source-partition.npz`: one category per original record: unresolved, ground,
  building, occupied-volume, measured-plane model or removed noise.
- `face-accounting/*.npz`: one replacement, retention or noise-removal reason per BPA triangle.
- `residual/*.npz`: exact source-face and vertex ownership for retained geometry.
- `source-hashes.json`: unchanged original archives and authorities.
- `verification.json`: independently checked support, counts, volume and collisions.

The independent verifier checks source fingerprints, disjoint/exhaustive
ownership, recomputed noise density, hole-free connected ground topology, measured
anchor provenance and the independent smoothing equation, original occupied
obstacle volume, bounded surface smoothing, protected planar openings, closed slab
boundaries, exact retained triangles and composed USD/collision equality.
Voxel edge/corner contacts are reported separately; watertight occupied volume
does not imply a manifold surface suitable for downstream CAD operations.

Verification can run headlessly; the completed result is displayed in an Isaac
Sim window and left open. The local verification helper
`runs/tsudanuma-isaac-local/open_simulation.py` saves matched-camera PNGs for the
full scene or a separate ground/model inspection. Set `USIM_QA_VIEW=ground` or
`models` for the latter. `USIM_QA_GROUND_FIRST=1` captures the isolated continuous
floor before opening the full completion scene. `select_simulation_probes.py` and
`probe_simulation_physics.py` verify actual terrain contacts with rigid bodies
over a fixed simulated interval, including filled inferred floor cells, and a
falling body outside the supported map.
Generated images, probe reports and the repair plan remain in ignored `runs/`.

For the locally cached Isaac runtime whose license was already accepted:

```sh
OMNI_KIT_ACCEPT_EULA=YES .venv-isaac-view/Scripts/isaacsim.exe --no-window \
  --/app/file/ignoreUnsavedOnExit=true \
  --exec runs/tsudanuma-isaac-local/open_simulation.py
```

For the completion window, omit `--no-window` and set `USIM_QA_KEEP_OPEN=1`:

```sh
OMNI_KIT_ACCEPT_EULA=YES USIM_QA_KEEP_OPEN=1 USIM_QA_GROUND_FIRST=1 \
  .venv-isaac-view/Scripts/isaacsim.exe \
  --/persistent/physics/cooking/ujitsoCookingMaxProcessCount=1 \
  --/physics/cooking/ujitsoCollisionCooking=true \
  --/persistent/physics/numThreads=2 \
  --exec runs/tsudanuma-isaac-local/open_simulation.py
```

The fine measured map has about 52 million rendered triangles. Limiting collision
cooking to one process avoids the parallel-cooking memory exhaustion observed
when older Isaac scenes remain open. These runtime settings leave the authored
geometry and triangle-collision settings unchanged.

The exit flag avoids an unsaved-session confirmation when temporary inspection
cameras are added. It does not save the session or modify original map layers.
