# Additional navigation worlds

`scripts/worlds/fetch_worlds.py` installs six reviewed starter sources into the
ignored checkout directory `assets/worlds/`. No upstream models, textures, or
archives are committed. The machine-readable allowlist is
`scripts/worlds/sources.json`; revisions, archive sizes, and SHA-256 values are
fixed there. A mismatch fails before extraction. Downloads require only Python's
standard library.

## Fetch

From a usim checkout:

```sh
python scripts/worlds/fetch_worlds.py --list
python scripts/worlds/fetch_worlds.py aws-racetrack sonoma-raceway
python scripts/worlds/fetch_worlds.py aws-small-house aws-bookstore aws-small-warehouse willow-garage
```

Alternatively, use `--all` to select exactly those six sources. `--root /path/to/usim`
or `USIM_ROOT` selects the checkout root explicitly; the default is the checkout
containing this script. Unknown names and duplicate selections are rejected
before any network access. Existing installations are verified against their
pinned source record and file hashes, then reused without network access.
Changed or incomplete installations fail rather than being overwritten; to
reinstall those, move the corresponding world directory aside yourself. Completed
archives are cached in `assets/worlds/_archives/` by SHA-256 and verified again
on reuse. Partial installations never become the final world directory.

## Sources and rights

| Local name | Immutable revision | Rights | Archive bytes |
| --- | --- | --- | ---: |
| `aws-racetrack` | `ea90fdd9bac48344198f7cb3031dcb418a741064` | MIT-0, Amazon.com, Inc. or its affiliates (2019) | 94,501,660 |
| `sonoma-raceway` | Fuel model version **3** | CC0-1.0, OpenRobotics; authors Ian Chen and Cole Biesemeyer | 8,546,577 |
| `aws-small-house` | `dd44be6205b2e6577bf641300dacc1699cb2ca81` | MIT-0, Amazon.com, Inc. or its affiliates (2019) | 109,467,605 |
| `aws-bookstore` | `ed542eb41809d5c5f1c67f32d4ba24e0e90886b3` | MIT-0, Amazon.com, Inc. or its affiliates (2019) | 39,791,360 |
| `aws-small-warehouse` | `3c23a698bf0b4e366ddf8b084af507c519bd3483` | MIT-0, Amazon.com, Inc. or its affiliates | 10,214,085 |
| `willow-garage` | `8163eb4b5e7e21985c6591d1c0bfb56468c0093f` | CC-BY-3.0, Copyright 2012 Nathan Koenig; author Nate Koenig | 3,344,881 |

AWS repositories were archived when RoboMaker support ended. These commits
contain the original assets from their former `ros1` branches, not the
replacement archival README-only branches:

- [AWS racetrack](https://github.com/aws-robotics/aws-robomaker-racetrack-world/tree/ea90fdd9bac48344198f7cb3031dcb418a741064)
- [AWS small house](https://github.com/aws-robotics/aws-robomaker-small-house-world/tree/dd44be6205b2e6577bf641300dacc1699cb2ca81)
- [AWS bookstore](https://github.com/aws-robotics/aws-robomaker-bookstore-world/tree/ed542eb41809d5c5f1c67f32d4ba24e0e90886b3)
- [AWS small warehouse](https://github.com/aws-robotics/aws-robomaker-small-warehouse-world/tree/3c23a698bf0b4e366ddf8b084af507c519bd3483)

The four pinned AWS LICENSE files contain the
[MIT No Attribution terms](https://spdx.org/licenses/MIT-0.html).
The installer retains LICENSE, README, models, and worlds, excluding launch
code, maps, routes, photos, repository automation, and documentation images.
It retains all files beneath the selected model/world roots, including any
nested rights notices. MIT-0 does not require attribution; the original notice
and source attribution are nevertheless preserved.

[Sonoma Raceway](https://fuel.gazebosim.org/1.0/OpenRobotics/models/Sonoma%20Raceway)
is a Fuel **model**, not a ready-made world. Its version-3 archive has no license
file, so the reviewed Fuel rights metadata is recorded separately in the
manifest and installed as `LICENSE-METADATA.json`. The review on 2026-10-02
observed version 3, `license_id: 1`, and
`Creative Commons Zero v1.0 Universal`, with the
[CC0 deed](https://creativecommons.org/publicdomain/zero/1.0/).
The independently retrieved
[CC0 legal code](https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt)
is SHA-256 pinned and retained as `LICENSE-CC0.txt`. Installation never consults
Fuel's mutable model metadata or a `tip`/`latest` asset URL. The snapshot is
evidence of the reviewed version, not a promise about future Fuel uploads.

[Willow Garage offices](https://github.com/osrf/gazebo_models/tree/8163eb4b5e7e21985c6591d1c0bfb56468c0093f/willowgarage)
uses the repository's
[CC-BY-3.0 LICENSE](https://github.com/osrf/gazebo_models/blob/8163eb4b5e7e21985c6591d1c0bfb56468c0093f/LICENSE),
with author attribution also retained in `model.config`.
Only its 14 model files and that LICENSE are requested, each at the exact commit
with a fixed SHA-256 and size. The entire Gazebo model database is **not**
downloaded. These 15 files are sorted into a reproducible, uncompressed ZIP
(1980-01-01 timestamp, Unix regular-file mode 0644); that locally assembled
subset archive also has a fixed SHA-256. When redistributing Willow Garage,
preserve `LICENSE`, `ATTRIBUTION.txt`, the source and license links, and identify
your modifications, as required by
[CC-BY-3.0](https://creativecommons.org/licenses/by/3.0/).
The supplied geometry and textures are unchanged; only a wrapper and provenance
notices are added.

Each completed installation includes `PROVENANCE.json`, containing the complete
source record, installed file hashes/sizes, retained attribution, entrypoints,
and a modification statement. Its `installed_size` excludes the provenance file
itself. Redistribution should carry the provenance and all retained notices.
Citysim, PLATEAU, Fuel hospital furniture, and other unreviewed assets are outside
this allowlist.

The six cached archives total **265,866,168 bytes**. Selected upstream resources
occupy about **270 MB** uncompressed, plus notices/provenance and the archive
cache; allow about **0.8 GB** for temporary extraction and installation.

## Use

The installed Gazebo Classic model paths are:

```sh
export USIM_ROOT=/path/to/usim
export GAZEBO_MODEL_PATH="$USIM_ROOT/assets/worlds/aws-racetrack/models:$USIM_ROOT/assets/worlds/sonoma-raceway/models:$USIM_ROOT/assets/worlds/aws-small-house/models:$USIM_ROOT/assets/worlds/aws-bookstore/models:$USIM_ROOT/assets/worlds/aws-small-warehouse/models:$USIM_ROOT/assets/worlds/willow-garage/models${GAZEBO_MODEL_PATH:+:$GAZEBO_MODEL_PATH}"
gazebo "$USIM_ROOT/assets/worlds/aws-racetrack/worlds/racetrack_day_empty.world"
```

Use only model paths for sources you installed. Other available entrypoints are
`racetrack_day.world`, `racetrack_night.world`, `small_house.world`,
`bookstore.world`, `small_warehouse.world`, and `no_roof_small_warehouse.world`
under their respective `worlds/` directories. Sonoma and Willow Garage have
generated `worlds/sonoma_raceway.world` and `worlds/willowgarage.world` wrappers.
Their wrappers include Gazebo's standard `sun` and `ground_plane` models, which
must already be provided by your simulator installation; the fetcher does not
download them.

These are upstream SDF/mesh assets, not robot deployments or Isaac USD stages.
Gazebo loading, spawn poses, collision/physics behavior, and Isaac conversion
remain **pending runtime validation**. No Gazebo or Isaac execution is implied
by successful fetching or the offline tests.

## Offline checks

```sh
uv run pytest -q test/test_world_sources.py
uv run ruff check scripts/worlds/fetch_worlds.py test/test_world_sources.py
uv run ruff format --check scripts/worlds/fetch_worlds.py test/test_world_sources.py
```

Tests cover the exact six-source boundary, immutable metadata, license evidence,
hash verification, subset determinism, traversal/link rejection, excluded
content, and installed provenance without downloading external assets.
