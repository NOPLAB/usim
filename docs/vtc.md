# Virtual Tsukuba Challenge world

The fetch script downloads an allowlisted subset of
[Field-Robotics-Japan/vtc_world_blender](https://github.com/Field-Robotics-Japan/vtc_world_blender)
at commit `2e6087aea800876e673e1e713a720728481e3f84`, verifies every downloaded
file's SHA256 and size, and converts it locally. No upstream asset bytes are
included in usim. Downloads and conversions belong under ignored `assets/` or
`runs/`.

## Provenance and exclusions

Upstream license: Apache-2.0. Attribution:
`Copyright [2020] Ryodo Tanaka groadpg@gmail.com`.
The script writes the upstream `LICENSE`, a `NOTICE` describing attribution and
modifications, and `provenance.json` beside the generated worlds. Retain these
when redistributing converted assets. usim's MIT license applies to the tooling,
not to downloaded geometry.

Only eight FBX files under `Environment/Terrain/fbx/` and `LICENSE` are fetched:
`002-ParkingArea.fbx`, `005-ParkingArea.fbx`, `006-RoadMark.fbx`, `ParkArea.fbx`,
`ParkingArea.fbx`, `ParkingArea2.fbx`, `SevenElevenArea.fbx`, and
`Tsukuba-Terrain.fbx`. The geometry
hashes are the SHA256 object IDs in the pinned upstream Git LFS pointers; the
license hash is computed from the regular Git blob. The manifest in
`scripts/vtc/fetch_vtc_world.py` records sizes and digests.

City Hall (`cityhall.blend`, `CityHall.blend`, or `City Hall.fbx`) is excluded.
Although the Blender README calls it Apache-2.0, its source in
[furo-org/VTC](https://github.com/furo-org/VTC) identifies City Hall as
CC-BY-NC-SA-4.0. The assembly is not downloaded because it links this asset.
No `tsukubachallenge/tc-datasets`, furo-org/VTC voxel or terrain files,
point-cloud data, photographs, or texture files are fetched. This world is a
terrain and road-mark subset, not the complete assembly: trees, fences,
railroad props, and buildings are not included.
Blender construction scenes are also excluded: inspecting them found hidden
boolean cutters and City Hall-named objects. The audited FBX exports avoid that
scene dependency. Conversion rejects any imported object named City Hall,
including case, space, underscore, and hyphen variants.

## Fetch and convert

Docker Desktop must run Linux containers on an amd64 host. No host Blender
installation is needed. Blender runs headless, with networking and automatic
Python execution disabled, in `linuxserver/blender:4.5.3`, pinned to its amd64
manifest digest
`sha256:16b3e174154bc89b9ae56edcfb8f5076cb3b05a6bea78156eced17073b851b7d`.
The converter mounts only the output directory and its own read-only script;
`docker run --rm` removes the conversion container.

```sh
uv run python scripts/vtc/fetch_vtc_world.py --out assets/vtc

# Optional Isaac-compatible USD, using the separate USD environment:
uv run --extra usd python scripts/vtc/fetch_vtc_world.py --out assets/vtc --usd
```

The USD extra pins `usd-core==25.5` and `numpy<2`; it is mutually exclusive with
the Isaac runtime extra. To use generated USD in Isaac, switch to that runtime's
environment after conversion.

Outputs:

- `world.sdf`: Gazebo Classic SDF 1.6, static COLLADA mesh visuals and collisions.
- `meshes/*.dae`: triangulated, metre-scale, Z-up geometry with baked transforms.
- `world.usdc` with `--usd`: Z-up, metres, static triangle-mesh collisions.
- `geometry.json`: intermediate vertices, triangles, and colors shared by exporters.
- `origin.json`: source-coordinate spawn origin and fallback ground height.
- `upstream/`: hash-verified source files, retained for reproducibility.
- `LICENSE`, `NOTICE`, `manifest.json`, `provenance.json`: license and conversion metadata.

Blender's FBX importer converts the source unit/axis metadata to metres and Z-up.
Object transforms are baked into vertices. External textures are deliberately
not searched for or downloaded; the exporter
uses one untextured diffuse color per object. All meshes are translated together
to put an interior horizontal triangle of the first parking export at the default
robot spawn `(0, 0, 0)`. Relative geometry is preserved; `origin.json` records the
translation. Both worlds include a 2 km square fallback ground slab 0.1 m below
the lowest exported vertex, leaving the actual terrain surfaces unobstructed.

Existing downloaded files are reverified, not silently replaced. Delete a corrupt
cache file before rerunning. A failed download or hash check never promotes a
partial file to the cache.

## Use in Gazebo or Isaac

```sh
uv run usim create-robot --out runs/vtc/robot.urdf
uv run usim simulate --backend gazebo --headless --max-seconds 5 \
  --world assets/vtc/world.sdf --robot-urdf runs/vtc/robot.urdf

uv run --extra isaac usim simulate --headless --max-seconds 5 \
  --world assets/vtc/world.usdc --robot-urdf runs/vtc/robot.urdf
```

Build `usim-gazebo:local` as described in [Gazebo](gazebo.md). The port resolves
relative `meshes/` URIs and mounts their resource directory read-only. See
[Isaac](isaac.md) for the optional native runtime. Conversion needs no native
Isaac runtime or GPU.
