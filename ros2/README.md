# usim ROS 2 simulation bridge

`usim.bridges.simulation_ros2` in the core package owns execution. The canonical
`ros2/src/usim_sim` colcon package publishes its `main` as `usim_sim_node`.
Install the core and one independent backend into the Python environment used by
ROS, then include `usim/ros2/src` in `colcon build --base-paths`.

```bash
python -m pip install -e /path/to/usim -e "/path/to/usim/packages/maniskill[native]"
ros2 run usim_sim usim_sim_node --ros-args \
  -p env_id:=PickPlace-CRANE-X7 -p robot_uid:=CRANE-X7 \
  -p simulator:=maniskill -p backend:=cpu
```

The core bridge calls the lazy `usim.create_simulator` factory; it does not import
training code, engines directly, or mutate `sys.path`. Backends are independent
`usim-maniskill`, `usim-genesis`, `usim-isaacsim` and `usim-gazebo` distributions.
Use `usim`, `usim.types`, `usim.interface` and `usim.factory` for shared APIs.
ROS dependencies (`rclpy`, `rcl_interfaces`, `cv_bridge` and message packages)
come from the ROS installation rather than the dependency-free core.

Supply `env_id` and `robot_uid`. Parameters select `robot_path`, joint names,
`initial_qpos`, `camera_uid`, `camera_frame`, `obs_mode`, `actuator_profile`,
control/render modes, simulation rate, episode length, auto-reset, topic names
and service names. The node is named `usim_sim_node`. Default input `/usim/action`
is `std_msgs/Float32MultiArray` in the selected engine's action representation.
Outputs are RGB images (`sensor_msgs/Image`) on `/camera/color/image_raw`,
joint angles (`sensor_msgs/JointState`) on `/joint_states`, episode completion
(`std_msgs/Bool`) on `/usim/episode_done`, and status (`std_msgs/String`) on
`/usim/task_info`. `/usim/reset` uses `std_srvs/Trigger`; `/usim/pause` uses
`std_srvs/SetBool`.

Downstream `crane_x7_bringup` owns CRANE-specific configuration and
`usim.launch.py`, `usim_logger.launch.py`, `usim_vla.launch.py`. Its configuration
sets `/vla/predicted_action` as input and `hand_camera_link` as the image frame.
No legacy package or node aliases are retained. ROS, rendering and native physics
execution require separate runtime validation.
