# Raspberry Pi Cat VTC SLAM and Nav2 example

This optional ROS 2 Humble example builds on the standalone `usim-gazebo`
image. It fetches RT Corporation's Raspberry Pi Cat ROS packages at the commits
in `raspicat.repos`; none of those sources or binary assets are copied into the
usim repository. The simulator core remains robot-independent.

The upstream Raspberry Pi Cat packages and their meshes are Apache-2.0. Their
license files are retained in the fetched source workspace. The example
preserves RT Corporation's copyright notices and does not use the robot name
to imply endorsement.

## Build and run

First fetch and convert the VTC road/terrain subset as described in
[`docs/vtc.md`](../../docs/vtc.md). Then build this optional ROS overlay from
the usim repository root (the first build creates the standalone Gazebo base):

```sh
docker build -f docker/Dockerfile.gazebo -t usim-gazebo:local .
docker build -f examples/vtc_navigation/Dockerfile \
  -t usim-vtc-navigation:local .
mkdir -p runs/vtc-navigation
docker run --rm --network host \
  --shm-size 512m \
  --name usim-vtc-navigation \
  -v "$PWD/assets/vtc:/assets/vtc:ro" \
  -v "$PWD/runs/vtc-navigation:/output" \
  usim-vtc-navigation:local
```

The run performs two real Gazebo sessions:

1. Launches the VTC SDF world, the pinned Raspberry Pi Cat Gazebo package and
   the upstream `raspicat_slam` SLAM Toolbox configuration. A bounded odometry
   controller drives a short square survey, requires live `/scan` and `/map`
   data, then saves the occupancy map to `runs/vtc-navigation/vtc_map.yaml`
   and `.pgm`.
2. Restarts Gazebo in the same world with the saved map, Nav2's standard
   `bringup_launch.py`, and the upstream `raspicat_navigation` parameter file.
   It publishes an initial pose and sends a `NavigateToPose` action to a point
   in the mapped area. The container exits nonzero unless Nav2 reports success
   and the observed final pose is within the configured goal tolerance.

The example uses ROS domain 42 by default and chooses a free Gazebo master port
for each session. Set `ROS_DOMAIN_ID` explicitly to use another domain.

A verified headless run completed all four survey waypoints, built a 720 by 576
occupancy grid with 20,149 known cells, and completed `NavigateToPose` with
status 4 (`SUCCEEDED`) and 0.039 m final localization error. The run writes its
exact measurements to `exploration.json` and `navigation.json`.

Inspect `exploration.json`, `navigation.json`, `slam.log`, `nav.log`, the map
image and `map.yaml`. The world, robot sources, meshes and generated map stay
outside Git under `assets/` and `runs/`.
