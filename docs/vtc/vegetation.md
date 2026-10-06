# LiDAR-supported park vegetation

`assets/vtc-vegetation/world.usda` adds derived trees over the road-corrected
local VTC scene. It references original tree artwork and the existing corrected
world; original terrain, point records, buildings and artwork are not rewritten.

The verified official 2019 PCD and its validated
[2019-to-VTC pose](tsukuba-full-map/vtc-registration.json) supply canopy positions
and dimensions. The inspected area is VTC X -340 to -20 m, Y 30 to 290 m.
It includes the park, its southern wooded area and nearby rows of trees.
Unregistered 2018 points are not used.

## Derived models

The initial extraction found 317 volumetric canopy candidates. Building
footprints, flat roofs and narrow planar facades are rejected. Candidates on
the original paved mask or without usable terrain are excluded, leaving
**249 tree instances in 75 streamed 30 m cells**.

Height is measured relative to the local LiDAR ground envelope. The original
VTC terrain and LiDAR ground do not agree everywhere in Z; each tree is grounded
on the preserved VTC terrain while retaining its measured relative height.
Crown spread controls scale. The existing broadleaf `HillTree_Tall_02` model,
including branches and leaves, is referenced as an instance instead of copying
its geometry for every tree.

These are visualization models, not an exact census, trunk survey or species
classification. Stem centers and separation of overlapping crowns are inferred;
tree shape and yaw are artistic approximations. No hidden or unmeasured plant
geometry is claimed as a measured surface.

## Generate

Run in the isolated CPU environment from
[the full-map instructions](tsukuba-full-map/README.md):

```powershell
& $cpu src/usim_env_vtc/vegetation.py `
  --source assets/tsukuba-full-oriented/source/map_tc19_o085_f-04_t05.pcd `
  --registration docs/vtc/tsukuba-full-map/vtc-registration.json `
  --world assets/vtc-road-repair/world.usda `
  --index assets/vtc-road-repair/world.index.json `
  --terrain assets/vtc-full/terrain `
  --prototype assets/vtc-full/meshes/KiteDemo_Environments_Trees_HillTree_Tall_02_HillTree_Tall_02.usdc `
  --out assets/vtc-vegetation
```

Outputs are `world.usda`, `world.index.json`, individual tree-cell payloads,
and `vegetation.json` containing placements, sizes and supporting point counts.
All remain local runtime assets, outside the MIT Python distribution.
This scene references mixed-license VTC artwork; the public PCD's license does
not grant redistribution rights to those models. See the existing
[asset audit](tsukuba-full-map/LICENSES.md).

## View

```powershell
& $isaac src/usim_env_vtc/view.py `
  --world assets/vtc-vegetation/world.usda `
  --index assets/vtc-vegetation/world.index.json `
  --out runs/vtc-vegetation/view --region 2 5 --oblique
```

The same native viewer culls tree cells with terrain by frustum and distance.
Original point layers remain hidden. Actual GPU captures verified trees in the
park; a same-process pan moved from 21 resident payloads to 16, loading the
southern wooded area without restarting the GUI.

Four CPU regressions cover separated crowns, measured relative height, rejected
roofs/facades, and actual USD instancing/grounding at negative coordinates.
Native capture evidence is in `runs/vtc-vegetation/view/`.

## Additional structures

The original VTC's static furniture, walls, stairs and railings are already
represented by its recovered placements; they are not duplicated.

With `--with-structures`, narrow point clusters outside canopy footprints can
also supply **seven unidentified vertical-structure proxies**, approximately
3.0 to 11.7 m high. They have at least 60 supporting points, five occupied
vertical levels, and less than 0.8 m horizontal extent. Their semantic identity
is not established: they are not asserted to be streetlights, signs or benches.
Simple cylindrical geometry represents the measured envelope, not hidden detail.

Each proxy uses the actual composed, road-corrected terrain mesh for grounding
and has static cylinder collision. Separate bounded payloads participate in the
same native culling as trees and terrain. Original point records and VTC objects
stay unchanged.

To generate trees and these additional structures together, run the command
above with `--out assets/vtc-detailed --with-structures`. View the combined
result with:

```powershell
& $isaac src/usim_env_vtc/view.py `
  --world assets/vtc-detailed/world.usda `
  --index assets/vtc-detailed/world.index.json `
  --out runs/vtc-detailed/view --region 2 5 --oblique
```

`assets/vtc-detailed/vegetation.json` records all tree and structure dimensions,
positions, supporting point counts and derived-geometry limitations.
The structural regressions check column detection, exclusion of existing tree
trunks, negative-coordinate payloads, grounding, collision and conservative
bounds. The combined tree, structure, road and viewer regression set passed
12 CPU tests.

An independent reopen of the final scene verified all 249 trees and seven
structure proxies: every added object is grounded, all tree references remain
instances, all posts retain collision, and all 81 added payload bounds contain
their geometry. The report is `runs/vtc-detailed/verification.json`.
The combined scene passed same-process native traversal (22 to 17 resident
payloads); the park capture and live wooded-area GUI were inspected.
The pan-probe PNG can precede terrain GPU readiness, so the live GUI, not its
early black-background snapshot, was used to verify the wooded-area ground.
Repository lint, formatting, 256 base tests (10 optional-environment skips),
and source/wheel builds also passed. Optional CPU/USD tests ran separately.
