#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
set -u

assets="${ROBOT_ASSETS:-/robots}"
output="${ROBOT_OUTPUT_DIR:-$assets/prepared}"
overlay=/tmp/usim-robot-description-overlay
share="$overlay/share"

register_package() {
  local name="$1"
  local source="$2"
  mkdir -p "$share/ament_index/resource_index/packages" "$share"
  touch "$share/ament_index/resource_index/packages/$name"
  ln -s "$source" "$share/$name"
}

register_package turtlebot3_description \
  "$assets/sources/turtlebot3-90a68bd2e3c61c12966779da89d8eeaec82730e9/turtlebot3_description"
register_package rosbot_description \
  "$assets/sources/rosbot-5a42ca181fbac0b5eb1020795282785561a473f4/rosbot_description"
register_package husarion_components_description \
  "$assets/sources/husarion_components_description-5f783f89961bb16098184f5381b1a76058cec19e"
register_package pmb2_description \
  "$assets/sources/pmb2-e6584136f15356b8f4944c25d70fb8689e3355af/pmb2_description"
register_package pal_urdf_utils \
  "$assets/sources/pal_urdf_utils-171a1fe521e42eaf2b6cff71d5339551feaa0191"
register_package orne_box_description \
  "$assets/sources/orne-box-4d9815f1f363dbb2ea837cc51f8d96a87b6a60bb/orne_box_description"
register_package clearpath_platform_description \
  "$assets/sources/clearpath-5c5ec97ee0245d8543aeb29115e01d3c0f38900e/clearpath_platform_description"
mkdir -p /tmp/clearpath_control
register_package clearpath_control /tmp/clearpath_control
export AMENT_PREFIX_PATH="$overlay:/opt/ros/humble${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
mkdir -p "$output"

xacro "$assets/sources/turtlebot3-90a68bd2e3c61c12966779da89d8eeaec82730e9/turtlebot3_description/urdf/turtlebot3_burger.urdf" namespace:= -o "$output/turtlebot3_burger.urdf"
xacro "$assets/sources/rosbot-5a42ca181fbac0b5eb1020795282785561a473f4/rosbot_description/urdf/rosbot.urdf.xacro" mecanum:=false namespace:= use_sim:=false -o "$output/rosbot.urdf"
xacro "$assets/sources/rosbot-5a42ca181fbac0b5eb1020795282785561a473f4/rosbot_description/urdf/rosbot_xl.urdf.xacro" configuration:=basic include_camera_mount:=false mecanum:=false namespace:= use_sim:=false -o "$output/rosbot_xl.urdf"
xacro "$assets/sources/pmb2-e6584136f15356b8f4944c25d70fb8689e3355af/pmb2_description/robots/pmb2.urdf.xacro" laser_model:=no-laser camera_model:=no-camera has_sonars:=false has_microphone:=false add_on_module:=no-add-on namespace:= -o "$output/pmb2.urdf"
xacro "$assets/sources/orne-box-4d9815f1f363dbb2ea837cc51f8d96a87b6a60bb/orne_box_description/urdf/orne_box.urdf.xacro" -o "$output/orne_box.urdf"
xacro /wrappers/clearpath_jackal.urdf.xacro namespace:= prefix:= is_sim:=false use_platform_controllers:=false gazebo_controllers:= -o "$output/jackal.urdf"
xacro /wrappers/clearpath_husky.urdf.xacro namespace:= is_sim:=false use_platform_controllers:=false gazebo_controllers:= -o "$output/husky.urdf"

for robot in turtlebot3_burger rosbot rosbot_xl pmb2 orne_box jackal husky; do
  test -s "$output/$robot.urdf"
  printf '%s\n' "Expanded $robot: $output/$robot.urdf"
done
