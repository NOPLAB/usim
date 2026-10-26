# Supported shape approximation

The remaining measured cloud is not one object. Simple structures can use fitted
faces, while irregular residual geometry needs a bounded surface approximation.
`simulation_map` applies this after ground/building ownership and density cleanup.
Original measurements and building authorities are not rewritten.

## References and what is adopted

- [WingEarth's vendor documentation](https://www.aisantec-geo.jp/service/wingearth/)
  documents point grouping, wall/floor/ceiling face fitting and 3D mesh generation.
  It does not publish the internal reconstruction algorithm. These are workflow
  references, not a claim that this implementation reproduces its private engine.
- [CGAL Shape Detection](https://doc.cgal.org/latest/Shape_detection/index.html)
  establishes distance/normal-supported primitive detection and connected support.
- [PolyFit](https://github.com/LiangliangNan/PolyFit) and
  [PrimFit](https://github.com/xiaowuga/PrimFit) use fitted primitives, supported
  face hypotheses and selection rather than converting every point to a box.
  Their full closed-manifold arrangement/solver pipelines are not imported.
- [Open3D surface reconstruction](https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html)
  and [mesh smoothing](https://www.open3d.org/docs/release/tutorial/geometry/mesh.html)
  provide the local surface and Taubin-smoothing reference. Open3D is already in
  the pinned pipeline extra. All work here uses CPU geometry APIs.

No vendor code or GPL reconstruction implementation is copied into the project.

## Structural faces

Existing measured plane supports are reused first. Unmatched samples then undergo
seeded regional RANSAC in 10 m XY regions, using the existing Open3D extraction
routine. Neighbors within 0.3 m establish local normal and planarity evidence.
Accepted components need at least 128 source records, 0.5 m2 observed footprint,
45% footprint coverage, and 80% planar-normal agreement within 20 degrees.
Point residuals are at most 0.04 m. Disconnected support is split before modeling.

The generated source archives are separate from `optimized/planes`. Original
record IDs remain attached. An explicit generation manifest excludes stale files.
Low horizontal structures can be modeled with a nearby measured-ground reference
and more than 0.25 m clearance; ground-like support stays excluded.

Plane-local repair uses a 0.2 m closing radius and an enclosed-hole limit of
0.16 m2. Original returns behind a missing wall/floor patch protect observed
openings. Inferred cells remain within the declared repair distance of support;
large openings and separated passages are retained. `observed_keys`, filled
`keys`, source archive paths and completion flags record what was approximated.

The declared cell union is then converted to a compact closed polygonal slab.
Coplanar runs merge; shared edges are split at neighboring rectangle corners so
there are no T-junction cracks. Front/back polygon fans and only exposed side
edges preserve the exact filled footprint. A 10,000-cell rectangle becomes
16 triangles rather than thousands of per-cell faces. Thickness is 0.16 m.

## Irregular residual surfaces

Unmatched occupied volumes retain their source ownership and topology. Each
output tile applies four Taubin pairs with lambda 0.5 and mu -0.53 using uniform
edge-neighbor averaging. This is inspired by Open3D, not its distance-weighted
implementation. Total vertex movement is capped at 0.2 voxel widths: 0.02 m at
the default 0.1 m resolution.

Tile-boundary and nonmanifold-contact vertices are fixed. Face count, connectivity,
vertex ordering, and holes do not change. Orientation checks prevent collapsed
or inverted triangles. Smooth vertex normals are authored. Rendered geometry and
static triangle collision remain identical; no separate convex collision hull
fills vegetation gaps or space below objects.

This is a geometric approximation, not a semantic vegetation classifier.
Unsupported shapes remain residual geometry rather than being forced into a
guessed box, tree crown or Poisson surface.

## Evidence

`simulation-models/approximation.json` records inferred planar cells and detected
planes. The local QA probe manifest records a source-backed low-structure
inspection center. Plane NPZs retain
source paths and observation/approximation footprints. The verifier checks source
ownership, recomputed completion/protected openings, closed volume and exact USD
collision equality. Residual surface positions are independently compared with
the declared smoothing operation and displacement bound. The original occupied
volume and smoothed rendered volume are reported separately.

The completion viewer captures a close view of a detected low structure when
one is available, as well as ground-only and full-scene camera pairs. Physics
probes exclude nearby fitted slabs and include inferred floor cells. Completed
results are displayed in the Isaac GUI and left open.
