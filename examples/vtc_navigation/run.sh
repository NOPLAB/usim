#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /opt/raspicat_ws/install/setup.bash
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

mkdir -p /output
slam_pid=

gazebo_port() {
  python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}

stop_launch() {
  if [[ -n "$slam_pid" ]] && kill -0 "$slam_pid" 2>/dev/null; then
    kill -TERM -- "-$slam_pid"
    wait "$slam_pid" || true
  fi
}

trap stop_launch EXIT INT TERM

export GAZEBO_MASTER_URI="http://localhost:$(gazebo_port)"
setsid ros2 launch /opt/vtc_navigation/launch/slam.launch.py \
  world:=/assets/vtc/world.sdf > /output/slam.log 2>&1 &
slam_pid=$!

python3 /opt/vtc_navigation/scripts/explore_and_save.py \
  --output /output/vtc_map

timeout 30s ros2 run nav2_map_server map_saver_cli \
  -f /output/vtc_map \
  --ros-args -p use_sim_time:=true

stop_launch
slam_pid=

# Use a fresh master port in case the previous server is still shutting down.
export GAZEBO_MASTER_URI="http://localhost:$(gazebo_port)"

setsid ros2 launch /opt/vtc_navigation/launch/nav.launch.py \
  world:=/assets/vtc/world.sdf \
  map:=/output/vtc_map.yaml > /output/nav.log 2>&1 &
slam_pid=$!

python3 /opt/vtc_navigation/scripts/navigate_goal.py \
  --map /output/vtc_map.yaml \
  --result /output/navigation.json

stop_launch
slam_pid=
trap - EXIT INT TERM
