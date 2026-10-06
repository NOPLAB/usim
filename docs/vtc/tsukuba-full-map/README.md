# Full old-course Tsukuba map

This map is owned by the `usim` workflow, not a downstream application.
It reconstructs measured surfaces from official fuRo point clouds. It is an
unofficial derived map, not an official survey, a 2026 course, or a guarantee
that every hidden building surface was measured.

## License boundary

**usim software is MIT. Downloaded point clouds and derived map assets are
separately licensed CC BY-NC-SA 4.0.** They are not included in the MIT Python
wheel. A free download, an academic institution, or local use does not by
itself establish that a use is NonCommercial under that license.

The distributed map variant uses the directly licensed official PCD files and
owned visualization code. It does not include VTC landscape heightmaps,
City Hall models, reference-map images, Epic StarterContent, KiteDemo, Paragon,
Megascans/Quixel textures, CagePlugin models, or the earlier Blender PLY.
Do not copy the old mixed-art `assets/vtc-full/` directory into this artifact.
See [the asset-by-asset licensing audit](LICENSES.md) for those distinctions.

The data license permits NonCommercial reproduction and adaptation. Sharing
requires retained attribution, license/source links and modification notices;
shared adaptations must meet ShareAlike. Converting a format alone is not
automatically an adaptation; surface reconstruction and derived-map
contributions are deliberately distributed under CC BY-NC-SA 4.0.
The MIT converter remains separate software, rather than being relicensed CC.
The local inspection composition described below retains existing VTC artwork
without copying it into the public map. That composition is **not** covered by
the public map's redistribution decision.

Controlling terms:

- [Official dataset-specific declarations](https://github.com/tsukubachallenge/tc-datasets).
- [CC BY-NC-SA 4.0 legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en),
  especially sections 1, 2(a), 3 and 4.
- [MIT software license](../../../LICENSE).

## Sources

Provider: **fuRo**. Acquired and independently hashed on 2026-10-05.

| Dataset | Original points | Original bytes | SHA256 |
| --- | ---: | ---: | --- |
| `map_tc18_o085_f-04_t30.pcd` | 17,002,094 | 544,071,104 | `6927c661dfcbc50317f3ad61907f4a20088a8bc3111767f0413e02af33a75f4e` |
| `map_tc19_o085_f-04_t05.pcd` | 22,356,688 | 715,418,112 | `119250f3b153bd0948b05e441cee30459bed45ce05cdf5749267bc2fc703309d` |

Original downloads:

- [2018 fuRo PCD](https://drive.google.com/file/d/1c7Vd4vkMudAHyxc0ZOZCbTgx8ZFZ_Slx/view).
- [2019 fuRo PCD](https://drive.google.com/file/d/1mH20dXpnBBlQ6hMKJZqdVhphrffsvWK_/view).

Both contain XYZ, intensity, normals and curvature as eight little-endian
float32 fields. They have no photographic RGB. All source records and their
original field values are retained; gray intensity visualization is not
photographic color. The catalog describes moving-object-removed occupancy
voxel mapping using 3D Cartographer.

The catalog requests this academic citation:

Yoshitaka Hara and Masahiro Tomono, "Moving Object Removal and Surface Mesh
Mapping for Path Planning on 3D Terrain", Advanced Robotics 34(6), 375-387,
2020. <https://doi.org/10.1080/01691864.2020.1717375>.

Associated sensor recordings are dated 2018-09-15 and 2019-09-14. These are
not asserted as the map-file creation dates. The catalog does not declare a
geographic CRS or origin for these SLAM maps; a PCD VIEWPOINT is not proof of
geographic registration.

The catalog's 83,178,268-point GNSS+INS map was investigated but unavailable:
its official binary and viewer returned HTTP 404, and provider relocation,
repository-fork and archive searches found no provenance-backed replacement.
It is not counted as acquired or used. A changed-course map is not substituted.

## Coordinate and reconstruction contract

The primary frame is the original 2019 point frame. Source arrays are never
rewritten by registration. Analysis samples may estimate a transform; the
shipped map retains complete source records and exact source-derived vertices.
Cross-year registration is not proof of temporal identity. Unverified poses
must not merge years or produce duplicate facades.

Region payloads retain native detail instead of a decimated whole-map proxy.
Triangle vertices come from original records. Unsupported gaps and unseen
surfaces are not filled with guessed boxes or flattened ground. The coverage
manifest distinguishes measured points, reconstructed faces and unresolved
boundaries. Open surfaces are not sealed buildings.

The validated 2019-to-VTC pose is recorded in
[`vtc-registration.json`](vtc-registration.json). Held-out nearest-point
distances have median 0.0000117 m and 99th percentile 0.0000317 m, consistent
with the same underlying point geometry and a coordinate conversion. These
numbers are **not** geographic survey accuracy.

Global structural-height registration followed by robust point-to-plane ICP
between 2018 and 2019 gave median 0.254 m, 90th percentile 0.827 m and 56.7%
within 0.3 m. This failed the fusion gate (median below 0.15 m and at least
70% within 0.3 m). The 17,002,094 original 2018 records remain a separate
reference PCD; they are not claimed as aligned or rendered in the 2019 world.

## Generate and verify

Run from the **usim checkout root**. The examples use PowerShell; on Linux,
use `bin/python` instead of `Scripts/python.exe`.

```powershell
uv venv runs/tsukuba-map/cpu --python 3.12
$cpu = "runs/tsukuba-map/cpu/Scripts/python.exe"
uv pip install --python $cpu -r src/usim_env_vtc/requirements.txt
& $cpu src/usim_env_vtc/download.py --output-dir assets/public-tsukuba 2018 2019
& $cpu src/usim_env_vtc/reconstruct.py `
  --source assets/public-tsukuba/map_tc19_o085_f-04_t05.pcd `
  --reference-2018 assets/public-tsukuba/map_tc18_o085_f-04_t30.pcd `
  --output assets/tsukuba-full-oriented --workers 4
& $cpu src/usim_env_vtc/verify_map.py --output assets/tsukuba-full-oriented
```

The downloader streams to a temporary file and publishes only after the
original byte count and SHA256 match. An existing original is verified, not
silently replaced. No GPU is used for acquisition, registration or conversion.
`src/usim_env_vtc/register.py --primary <2019.pcd> --reference <2018.pcd> --out <report.json>`
reproduces the cross-year registration experiment; its analysis sampling never
reduces the shipped source.

The independent verifier reopens every USD tile, compares every source field
and vertex's float32 bits, checks face ownership, tangent/edge/midpoint/centroid
support, indices, collisions and the complete default rendering index.

The current full-resolution result has:

| Measure | Result |
| --- | ---: |
| Primary records retained and rendered | 22,356,688 |
| Occupied 30 m regional cores | 447 |
| Cores containing supported triangles | 424 |
| Triangles, without decimation | 5,551,343 |
| Serialized mesh vertices, including halos | 7,241,737 |
| Distinct primary records used by a core mesh | 7,204,800 |
| Records remaining point-only | 15,151,888 |
| Per-tile open boundary edges | 7,765,357 |
| Nonmanifold edges | 0 |

Ball pivoting uses supplied normal **plane directions**, normalized with the
largest absolute component oriented positive for working geometry only.
Original normal fields are unchanged. Support limits intentionally reject
unsupported faces rather than bridging them. Many points therefore remain
point glyphs, not connected collision surfaces. This is not a watertight or
everywhere-drivable reconstruction.

`world.usda` combines all original point payloads with supported meshes;
`world-index.json` covers all 871 payloads. The older mesh-only candidate
`assets/tsukuba-full/` is not the completed artifact. Loading only
`streamed.usda` would again hide the point-only records.

## View the entire map in Isaac

The inspection viewer was exercised with Python 3.12 and Isaac Sim 6.0.1.0,
using the isolated pin below. This is separate from the simulator backend's
6.1.0.0 pin in [the main Isaac setup](../../isaac.md). That backend version is not
claimed as tested by these map captures.

```powershell
uv venv runs/tsukuba-map/isaac --python 3.12
$isaac = "runs/tsukuba-map/isaac/Scripts/python.exe"
uv pip install --python $isaac --extra-index-url https://pypi.nvidia.com `
  -r src/usim_env_vtc/requirements-isaac.txt
& $isaac src/usim_env_vtc/view.py `
  --world assets/tsukuba-full-oriented/world.usda `
  --index assets/tsukuba-full-oriented/world-index.json `
  --out runs/tsukuba-map/view --region 2 5 --oblique
```

Accept the native runtime's terms through its normal setup before starting it.
The generated world is a checkout resource; paths are explicit, with no
downstream-application directory inferred by the tools.

The native window's camera controls allow panning throughout the measured
bounds. The GUI now excludes survey point payloads, the old point-completion
overlay and the fragmented public-survey reconstruction overlay; originals
remain archived unchanged. It uses **mesh frustum and distance
culling in the same window/process**. Off-view payload prims become inactive,
so GPU residency does not grow indefinitely while traversing the map.
The full-detail mesh vertices are not decimated.

The resident neighborhood is bounded by `meshResidencyRadiusMeters` in the
capture report (about 94 m for this map/grid). Distant mesh tiles may be culled
when zooming far out; move the camera across the map to view those tiles.
This is not a claim that every full-resolution mesh fits in VRAM simultaneously.
Camera position and projection changes both update culling.
The inspection layer explicitly retains Z-up and meter units. Missing this
root-layer metadata had made navigation use the USD Y-up/centimeter defaults;
this was corrected after horizontal drag behaved like roll.
Leaving measured bounds does not close the GUI; outside coverage remains empty.
`--probe-region 2 5` checks real camera/residency changes without restarting.
Headless full-data capture retains point layers unless `--mesh-only` is supplied.

For local inspection with the **original VTC terrain, placement and materials**
plus the newly reconstructed public geometry, compose references without
changing or copying the original assets:

```powershell
& $cpu src/usim_env_vtc/compose_vtc.py `
  --vtc-world assets/vtc-full/completed.usda `
  --vtc-index assets/vtc-full/streaming-index.json `
  --survey-world assets/tsukuba-full-oriented/world.usda `
  --survey-index assets/tsukuba-full-oriented/world-index.json `
  --registration docs/vtc/tsukuba-full-map/vtc-registration.json `
  --out assets/vtc-full/with-public-survey.usda
& $isaac src/usim_env_vtc/view.py `
  --world assets/vtc-full/with-public-survey.usda `
  --index assets/vtc-full/with-public-survey.index.json `
  --out runs/vtc-full/view --region 2 5 --oblique
```

This preserves the 95 original indexed VTC payloads and adds 871 public-map
payloads under a separate transform. The combined **966-payload local stage**
inherits the mixed-license restrictions in [LICENSES.md]; it must not be
redistributed as the CC-only map. Original VTC primitive buildings remain
primitive where the source contains no detailed model. Public measured points
add facade/vegetation evidence, not invented textured architecture.
Its existing material port uses source-derived USD Preview Surface. Unreal
procedural/layered/animated shader behavior and tangent-normal blending are
not claimed as exact Isaac equivalents; original shader and texture sources
remain archived. No new RGB textures are inferred from intensity-only PCDs.

Paved surfaces in this composed stage have abrupt height steps. A derived,
smoothed copy is described in [road surface repair](../road-repair.md).

## Full-area native evidence

Capture every cell in an 8 by 8 view grid, rather than testing only City Hall:

```powershell
& $cpu src/usim_env_vtc/capture_all.py --python $isaac `
  --world assets/vtc-full/with-public-survey.usda `
  --index assets/vtc-full/with-public-survey.index.json `
  --out runs/vtc-full/composed-all-regions
```

Every region uses a fresh native process. Capture waits on the actual native
frame and USD streaming state, then checks the camera and streaming state
again after readback. `coverage.json` requires the union of loaded payloads to
equal the entire index. Original 1600 by 1000 PNGs and per-region evidence are
retained; `all-regions.png` is only a reduced contact sheet for visual review.
The grid overlay is disabled for capture and its prior setting is restored.

The completed run at `runs/vtc-full/composed-all-regions/coverage.json` passed
**64/64 regions and 966/966 indexed payloads**, with streaming idle after every
readback and no geometry decimation. `all-regions.png` was visually reviewed
across the full footprint, including its source-limited exterior. Front/rear
views and a park view passed under `runs/vtc-full/oblique-qa/`.
The earlier restart-based GUI was replaced after repeated window disappearance
was reported. An unbounded same-window cache then reached GPU memory exhaustion:
14,921 MiB of 16,380 MiB was in use, followed by
`ERROR_OUT_OF_DEVICE_MEMORY` on `Survey2019/Unsplit/Part_14/Points` and
`ERROR_DEVICE_LOST`. Python was inside the native `SimulationApp.update`.

The revised `runs/vtc-full/mesh-culling-probe/` retained PID 1182492, loaded
24 new meshes, culled 32, and reduced the resident set from 37 to 29.
`pan-probe.json` and its actual GPU image record this run.
The GUI measurement with points disabled and mesh culling enabled used
6,275 MiB GPU memory instead of 14,921 MiB. After a real wide-view mouse-wheel
operation the final clean-mesh GUI remained responsive at 7,499 MiB.
These are whole-GPU measurements,
not isolated process allocation figures.

Full-area RTX mesh preload was also tested and still exhausted VRAM.
The point-free complete stage has 82,122,847 active mesh vertices and
142,822,798 triangle equivalents (shared instance proxies not expanded).
`--all-regions` is a diagnostic option, not a recommended 16 GB GUI mode.

## Actual point count

`runs/vtc-full/point-count.json` was counted from real USD point attributes,
not inferred from filenames:

| Map layer | Point primitives | Stored points |
| --- | ---: | ---: |
| VTC 2019 split survey | 8 | 22,222,867 |
| VTC 2019 unsplit survey | 23 | 22,356,688 |
| Public 2019 regional survey | 447 | 22,356,688 |
| Total in the composed map | 478 | 66,936,243 |
| Active point primitives/points in the GUI inspection | 0 | 0 |

The three 2019 representations overlap. This sum measures stored/renderable
copies, not distinct physical locations. The separate 17,002,094-point 2018
PCD is archived and is not part of this composed-stage count.
Reproduce the audit with `src/usim_env_vtc/count_points.py --world <composed-world> --index
<composed-index> --inspection <mesh-inspection.usda> --out <report.json>`
in the CPU environment.

Repository checks completed: dependency sync, Ruff lint and formatting,
**231 passing tests with 7 skipped**, and source/wheel builds. A packaging
audit found no downloaded map assets or runtime artifacts in either build.
The optional native/CPU SDK dependencies are unavailable to the base editor
language server; actual separate-environment reconstruction, independent USD
verification and native Isaac execution passed. No commit or push was made.

## Packaging and redistribution

Keep generated maps under ignored `assets/` and validation output under
`runs/`. Keep downloaded originals byte-for-byte. The map's NOTICE and
provenance manifest must accompany redistributed generated geometry and point
data; do not remove the NC/SA conditions or describe it as commercially
unrestricted. Retain fuRo attribution, original source and license links,
hashes, modifications and transform information.

Native Isaac Sim has its own runtime agreement; installing or running it
does not relicense this map. Conversion and verification use a separate CPU
environment so USD and Isaac native dependencies do not conflict.
