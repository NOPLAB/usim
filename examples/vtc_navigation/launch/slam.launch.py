"""Launch the pinned Raspberry Pi Cat in the generated VTC world with SLAM."""

import os
from collections.abc import Mapping

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def include(
    package: str,
    launch_file: str,
    arguments: Mapping[str, str],
) -> IncludeLaunchDescription:
    """Include an upstream launch file with the given package arguments."""
    path = os.path.join(get_package_share_directory(package), 'launch', launch_file)
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(path),
        launch_arguments=arguments.items(),
    )


def generate_launch_description() -> LaunchDescription:
    """Start Gazebo, Raspberry Pi Cat and the upstream asynchronous SLAM node."""
    world = LaunchConfiguration('world')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')

    return LaunchDescription(
        [
            DeclareLaunchArgument('world', default_value='/assets/vtc/world.sdf'),
            DeclareLaunchArgument('x_pose', default_value='0.0'),
            DeclareLaunchArgument('y_pose', default_value='0.0'),
            include(
                'raspicat_gazebo',
                'raspicat_with_iscas_museum.launch.py',
                {
                    'world': world,
                    'gui': 'false',
                    'rviz': 'false',
                    'use_sim_time': 'true',
                    'x_pose': x_pose,
                    'y_pose': y_pose,
                },
            ),
            include(
                'raspicat_slam',
                'raspicat_slam_toolbox.launch.py',
                {'use_sim_time': 'true', 'use_rviz': 'false'},
            ),
        ]
    )
