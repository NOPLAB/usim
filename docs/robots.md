# Audited external robot descriptions

These six selections are upstream assets, not MIT-licensed usim source. Fetching
downloads immutable archives into ignored `assets/robots/`; it does not install
ROS packages, expand xacro, convert meshes, or prove a Gazebo/Isaac import or
physics result. The ROSbot and PMB2 profiles also fetch their audited
Apache-2.0 xacro helper packages. No other robot is selectable. The source
audit date is 2026-10-02.

## Fetch and provenance

From the checkout root, with Python 3.12:

```sh
uv run python scripts/robots/fetch.py --list
uv run python scripts/robots/fetch.py
# Or fetch only one selection (the two Clearpath selections share one archive):
uv run python scripts/robots/fetch.py jackal husky
```

No branch tips are used. `scripts/robots/sources.json` pins the archive URL,
commit, compressed byte size, SHA-256, LICENSE hash, extraction allowlist, joint
groups and metre-scale geometry. The fetcher checks the archive before extraction,
rejects traversal, links, special files and duplicate selected paths, preserves
upstream files byte-for-byte, and checks each upstream LICENSE hash. It extracts
only descriptions, meshes and their license evidence, not hardware drivers or
other Clearpath platforms. Archives necessarily contain the original repository
contents, but unselected descriptions are not extracted or registered.

The generated `assets/robots/provenance.json` describes the selections from the
most recent invocation; each `sources/<source>-<commit>/provenance.json` also
contains the source URL, commit, archive hash, license evidence and an inventory
of extracted paths, byte sizes and SHA-256 hashes. `import_verified` is false.
Cached archives are rechecked, and an existing source must match the freshly
extracted inventory. A modified source or corrupt cache fails instead of being
silently overwritten. Store local adaptations outside `sources/`.

| Source ID | Repository | Immutable commit | Archive bytes | License |
| --- | --- | --- | ---: | --- |
| `turtlebot3` | [ROBOTIS-GIT/turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3) | `90a68bd2e3c61c12966779da89d8eeaec82730e9` | 8,862,240 | Apache-2.0 |
| `rosbot` | [husarion/rosbot_ros](https://github.com/husarion/rosbot_ros) | `5a42ca181fbac0b5eb1020795282785561a473f4` | 5,801,431 | Apache-2.0 |
| `pmb2` | [pal-robotics/pmb2_robot](https://github.com/pal-robotics/pmb2_robot) | `e6584136f15356b8f4944c25d70fb8689e3355af` | 2,977,359 | Apache-2.0 |
| `orne-box` | [open-rdc/orne-box](https://github.com/open-rdc/orne-box) | `4d9815f1f363dbb2ea837cc51f8d96a87b6a60bb` | 3,710,896 | BSD-2-Clause |
| `clearpath` (`jackal`, `husky`) | [clearpathrobotics/clearpath_common](https://github.com/clearpathrobotics/clearpath_common) | `5c5ec97ee0245d8543aeb29115e01d3c0f38900e` | 20,252,477 | BSD-3-Clause |
| `husarion_components_description` (ROSbot xacro dependency) | [husarion/husarion_components_description](https://github.com/husarion/husarion_components_description) | `5f783f89961bb16098184f5381b1a76058cec19e` | 3,572,783 | Apache-2.0; retains `NOTICE.md` with embedded BSD-3 terms |
| `pal_urdf_utils` (PMB2 xacro dependency) | [pal-robotics/pal_urdf_utils](https://github.com/pal-robotics/pal_urdf_utils) | `171a1fe521e42eaf2b6cff71d5339551feaa0191` | 7,387,273 | Apache-2.0 |

Retain each downloaded `LICENSE`, embedded copyright/license headers and any
upstream NOTICE when redistributing source, derived URDFs or meshes. Apache-2.0
assets remain Apache-2.0, including its attribution/NOTICE requirements; they do
not become MIT by being loaded by usim. The five robot-description archives have
no standalone upstream NOTICE. The Husarion helper archive includes
`NOTICE.md`; the fetcher retains it because it includes BSD-3-derived URDFs.

Clearpath additionally has a distinct full BSD-3-Clause header in
`clearpath_platform_description/urdf/common.urdf.xacro`, copyright 2023 Clearpath
Robotics, Inc., with Luis Camero and Roni Kreinin named as authors. That file is
retained intact alongside the repository LICENSE (copyright 2023
clearpathrobotics). Neither Clearpath Robotics nor its contributors may be used
to endorse or promote derived products without specific prior written
permission. ORNE-box's LICENSE retains the 2014 Robot Design and Control Lab
copyright and the BSD-2 conditions/disclaimer; it has no non-endorsement clause.

## Geometry and drive joints

All names below assume an empty namespace/prefix, with `base_link` as the body
frame. Dimensions are from the audited wheel macros/parameters, not inferred
from mesh bounds.

| Selection / variant | Wheel radius (m) | Left-right separation (m) | Axle spacing (m) | Left drive joints | Right drive joints |
| --- | ---: | ---: | ---: | --- | --- |
| ROBOTIS TurtleBot3 Burger | 0.033 | 0.160 | one axle | `wheel_left_joint` | `wheel_right_joint` |
| ROBOTIS TurtleBot3 Waffle / Waffle Pi | 0.033 | 0.287 | one axle | `wheel_left_joint` | `wheel_right_joint` |
| Husarion ROSbot 2, skid only | 0.0425 | 0.192 | 0.106 | `fl_wheel_joint`, `rl_wheel_joint` | `fr_wheel_joint`, `rr_wheel_joint` |
| Husarion ROSbot XL, skid only | 0.048 | 0.248 | 0.170 | `fl_wheel_joint`, `rl_wheel_joint` | `fr_wheel_joint`, `rr_wheel_joint` |
| PAL PMB2 | 0.0985 | 0.4044 | one axle | `wheel_left_joint` | `wheel_right_joint` |
| ORNE-box | 0.145 | 0.4615 | one axle | `left_wheel_joint` | `right_wheel_joint` |
| Clearpath Jackal J100 | 0.098 | 0.37559 | 0.262 | `front_left_wheel_joint`, `rear_left_wheel_joint` | `front_right_wheel_joint`, `rear_right_wheel_joint` |
| Clearpath Husky A200 | 0.1651 | 0.555 | 0.512 | `front_left_wheel_joint`, `rear_left_wheel_joint` | `front_right_wheel_joint`, `rear_right_wheel_joint` |

All listed drive joints are continuous. TurtleBot3 casters are fixed/passive;
PMB2 has four passive casters, wheel width 0.040 m, caster radius 0.025 m and
caster spacing x=0.343 m, y=0.204 m. ORNE-box has two passive caster swivel/wheel
pairs at `(0.362, +/-0.150, 0.108)` m; its drive wheel width is 0.045 m and the
drive origins are y=+/-0.23075 m. Jackal's wheel width is 0.040 m; Husky's wheel
length/width is 0.1143 m.

For each skid platform, command both left joints at the same angular speed and
both right joints at the same angular speed, using track as separation. This
is a differential approximation with axle slip, not a validated friction model.
Use `SimulationConfig.left_joints` and `right_joints`, or the
`--left-joints` / `--right-joints` CLI options, to drive both axles. Every
paired wheel must use the configured common radius and separation and have a
joint axis that produces the same forward direction. The fetcher does not
silently add mimic joints or modify upstream geometry. ROSbot must always
expand with `mecanum:=false`; mecanum geometry/control is not a selection in
this manifest even though the original upstream package contains its files.

## Preparing descriptions

Paths here are relative to `assets/robots/sources/<source>-<commit>/`.
Use xacro with package lookup configured for the downloaded description tree
and the named dependencies. For ROS 2, `$(find ...)` uses the sourced ament
package index; merely setting `ROS_PACKAGE_PATH` may not register a package.
Install/register these source trees in your ROS workspace or provide an
equivalent xacro package resolver. Package registration and optional dependencies
are environment preparation, not work performed by the fetcher.

### TurtleBot3

Expand `turtlebot3_description/urdf/turtlebot3_burger.urdf` (or
`turtlebot3_waffle.urdf` / `turtlebot3_waffle_pi.urdf`) with xacro and
`namespace:=''`: although named `.urdf`, these files use xacro properties.
Only `urdf`/`xacro` and this package are needed. Geometry constants come from the
audited `turtlebot3_node/param` files; the original archive retains that evidence.
LDS and IMU links are present; Waffle adds R200 frames and Waffle Pi a camera
frame. STL/DAE meshes are retained. The Humble URDF files have no Gazebo plugin
block.

### ROSbot 2 / XL

Expand `rosbot_description/urdf/rosbot.urdf.xacro` or
`rosbot_description/urdf/rosbot_xl.urdf.xacro` with `mecanum:=false`,
`namespace:=''`, `use_sim:=false`; for XL also use `configuration:=basic` and
`include_camera_mount:=false`. Do not select manipulation configurations.
Both entrypoints unconditionally include `husarion_components_description`;
their body/wheel macros are self-contained, but the default entrypoints are not.
The audited helper dependency is
[husarion/husarion_components_description](https://github.com/husarion/husarion_components_description)
at `5f783f89961bb16098184f5381b1a76058cec19e`, Apache-2.0 (`LICENSE.txt`).
The fetcher downloads its pinned package resources and retains `NOTICE.md`, which
includes BSD-3 terms for adapted Universal Robots and Kinova xacros in that
package. Its other ROS dependencies may still need installation to build or
register the package. Base-only wrappers can instead call the `rosbot` /
`rosbot_xl` macros without sensor composition. Remove hardware/Gazebo/ros2_control
plugins from a derived engine input as needed. Supply both wheel groups for
skid-drive simulation.

### PMB2

Expand `pmb2_description/robots/pmb2.urdf.xacro` with
`laser_model:=no-laser camera_model:=no-camera has_sonars:=false`
`has_microphone:=false add_on_module:=no-add-on namespace:=''`.
Its materials/degree conversion require
[pal-robotics/pal_urdf_utils](https://github.com/pal-robotics/pal_urdf_utils)
at audited commit `171a1fe521e42eaf2b6cff71d5339551feaa0191`, Apache-2.0
(`LICENSE`). The fetcher includes its pinned xacro helper package. The ROS 2
control xacro is in `pmb2_description`. Base-only settings avoid optional
laser/camera packages whose meshes were not fully audited. Keep all four casters
passive and prepare the engine-specific control/plugin input separately.

### ORNE-box

Expand `orne_box_description/urdf/orne_box.urdf.xacro`. Only this description
package and xacro are needed. Geometry uses primitives, with no meshes.
IMU and MID360 links are included; Hokuyo and the main Gazebo include are
commented out. The wheel transmissions use ROS 1
`hardware_interface/VelocityJointInterface`; remove them in a derived input if
the engine importer rejects them. Leave caster swivel/wheel joints passive.

### Jackal / Husky

The retained files `clearpath_platform_description/urdf/j100/j100.urdf.xacro`
and `urdf/a200/a200.urdf.xacro` define macros, not standalone instantiated
robots. The usim wrappers in `examples/robots/clearpath_jackal.urdf.xacro`
and `clearpath_husky.urdf.xacro` include the common materials and instantiate
the corresponding platform macro:

```xml
<?xml version="1.0"?>
<robot name="jackal" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="$(find clearpath_platform_description)/urdf/common.urdf.xacro"/>
  <xacro:include filename="$(find clearpath_platform_description)/urdf/j100/j100.urdf.xacro"/>
  <xacro:j100/>
</robot>
```

Expand with `namespace:='' prefix:='' is_sim:=false`
`use_platform_controllers:=false gazebo_controllers:=''` (Husky does not require
`prefix`). This disables the upstream platform controller and avoids its default
`clearpath_control` configuration lookup. The retained package is a resource
subset, not a complete Clearpath workspace; register it for xacro lookup rather
than assuming the original package build/launch files are present. Keep the
common file's full BSD header with any derived wrapper/material distribution.
Husky has DAE visual meshes and cylinder wheel collisions; Jackal uses STL
chassis/wheel visuals. Check the target importer before any mesh conversion.
Ignition wheel-slip blocks remain upstream and need engine-specific handling.
No optional Clearpath sensor packages are fetched; Jackal itself includes
IMU/GPS links and Ignition sensor blocks in this pinned platform macro.

## Runtime selection

Supply a prepared physical URDF with resolved meshes to the existing
`--robot-urdf` CLI option or `SimulationConfig.robot_urdf`. Xacro is preparation,
not a native simulator input. For example, after expanding the Burger into
`assets/robots/prepared/burger.urdf`:

```python
from pathlib import Path
from usim.simulation import SimulationConfig
from usim.ports.gazebo import GazeboSimulator

GazeboSimulator(image='usim-gazebo:local').run(
    SimulationConfig(
        world=Path('worlds/corridor.world'),
        robot_urdf=Path('assets/robots/prepared/burger.urdf'),
        base_link='base_link',
        left_joint='wheel_left_joint',
        right_joint='wheel_right_joint',
        wheel_radius=0.033,
        wheel_separation=0.160,
        headless=True,
        max_seconds=10,
    )
)
```

Use the corresponding variant's metre-scale constants and joint names, or the
same fields in `SimulationConfig`, with the target engine. For skid platforms,
finish paired-drive preparation first. Preserve upstream attribution separately
from runtime plugins; xacro may discard XML comments during expansion.
Resolve `package://` mesh URLs through registered packages or explicit local
resource paths before import. Gazebo's external-resource behavior is documented
in [gazebo.md](gazebo.md), and Isaac setup in [isaac.md](isaac.md).
None of these downloads has been asserted to pass real import, collision,
inertia, drive, motor-gate, or sensor checks.
