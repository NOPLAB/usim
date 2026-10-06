# Road surface repair

The composed VTC stage (`assets/vtc-full/with-public-survey.usda`) shows spikes
and cliffs on paved roads. This note records what was found and how a derived,
smoothed terrain copy is produced. The original terrain is never modified.

## Findings

- At the worst sites, the archived source meshes equal the composed points.
  The defects come from the source terrain, not from conversion or composition.
- The largest adjacent step on paved samples in the source, on a 10 cm
  lattice, is 3.44140625 m.
- The registered 2019 public survey points don't support a blanket
  replacement of source heights.
- No 2018 data is fused. The 2018 to 2019 registration failed its fusion gate
  (see [the full map notes](tsukuba-full-map/README.md)).

## Derived correction

Run from the usim checkout root in the CPU environment described in
[the full map notes](tsukuba-full-map/README.md):

```powershell
& $cpu src/usim_env_vtc/repair_terrain.py `
  --terrain assets/vtc-full/terrain `
  --manifest runs/vtc-full/terrain-conversion/terrain-manifest.json `
  --world assets/vtc-full/with-public-survey.usda `
  --index assets/vtc-full/with-public-survey.index.json `
  --out assets/vtc-road-repair
```

How it works:

- Heights are sampled on a 10 cm lattice.
- A sample counts as paved when the summed weight of `Asphalt`,
  `Asphalt_White` and `Tile` is at least 128. Samples within 3 lattice steps
  of a material border are excluded from edge detection.
- An adjacent step above 8 cm on paved samples marks a defect.
- A normalized Gaussian (sigma 1.5 m) smooths heights only within 4.5 m of a
  defect. Neighborhood boundaries are blended into the surrounding surface.
- Unpaved values and ordinary grades are kept.
- No triangles or vertices are removed. Source assets stay unchanged.

`road-correction.usdc` authors local mesh attributes on the full
`/World/Terrain/...` paths, beneath the stronger repaired root layer. Merely
replacing leaf payloads was insufficient: the original parent terrain reference
still supplied stronger point opinions. The composed-world regression test
covers this distinction.

Altered heights in the derived copy are **not** exact original source values,
and they carry no survey accuracy claim.

## Verification

The corrected world changes 5,742,027 unique samples across 817 mesh
components, including 5,836,027 component vertices with shared border copies.
The largest single correction is 2.2505648136 m.
`runs/vtc-full/road-inspection/road-before-after.png` was inspected: pavement
spikes and cliffs are visibly softened. Three regression tests for the repair
passed, including actual USD parent-reference composition.

An independent reopen of the final composed world checked all 4,096 components,
66,064,384 vertices and 130,056,192 triangles. XY positions, triangle topology,
placements, triangle colliders and unpaved samples are unchanged; all 112 tile
seams match exactly. Paved-interior edges exceeding 8 cm per 10 cm sample fell
from 174,299 to zero. The maximum paved edge including material boundaries
fell from 3.44140625 m to 2.1208426952 m: material boundaries can retain steep
terrain, and this is not a claim that every curb or bank is drivable.
The report is `runs/vtc-full/road-inspection/verification.json`.

Final repository checks passed: Ruff lint and formatting, 256 base tests with
8 skipped, and source/wheel builds. The three optional CPU/USD repair tests
also passed. The wheel contains all 16 VTC tooling files and no downloaded
map assets; direct-file and module CLI checks passed in the declared CPU
environment. The base editor lacks the optional SciPy, Open3D, Pillow and USD
packages, so its unresolved-import diagnostics are environment limitations.

Native capture uses the viewport's asynchronous completion future. A redundant
synchronous renderer fence blocked GUI updates after a capture and was removed
from both initial rendering and the pan probe. Capture also awaits an actual
frame delivered to the viewport, rather than treating USD load completion as
GPU frame completion.

Native Isaac QA passed using the local GPU authorization already given for this
VTC work. Both sides of the affected road and the park were inspected in actual
GPU captures, within one GUI process (PID 42576). Moving to the park culled the
four previous terrain tiles and loaded six new tiles without restarting the
process. Camera pose errors were below 6e-14 and streaming was idle after
readback. Captures and the camera/residency record are in
`runs/vtc-road-repair/view/` (`road-ne.png`, `road-sw.png`, `park.png` and
`road-gui-qa.json`). The GUI is left open at the park.

## View the repaired world

```powershell
& $isaac src/usim_env_vtc/view.py `
  --world assets/vtc-road-repair/world.usda `
  --index assets/vtc-road-repair/world.index.json `
  --out runs/vtc-road-repair/view --region 2 5 --oblique
```

The unrepaired composed world still opens with the command in
[the full map notes](tsukuba-full-map/README.md). Native point layers remain
excluded from the GUI, and mesh culling is unchanged.
