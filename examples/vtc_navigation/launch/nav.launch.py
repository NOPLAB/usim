"""Launch the pinned Raspberry Pi Cat and Nav2 on a saved VTC occupancy map."""

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
    """Start Gazebo, localization and the upstream Nav2 stack."""
    world = LaunchConfiguration('world')
    map_file = LaunchConfiguration('map')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    params_file = os.path.join(
        get_package_share_directory('raspicat_navigation'),
        'config',
        'param',
        'nav2.param.yaml',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('world', default_value='/assets/vtc/world.sdf'),
            DeclareLaunchArgument('map', default_value='/output/vtc_map.yaml'),
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
                'nav2_bringup',
                'bringup_launch.py',
                {
                    'namespace': '',
                    'use_namespace': 'False',
                    'slam': 'False',
                    'use_sim_time': 'True',
                    'map': map_file,
                    'params_file': params_file,
                    'autostart': 'True',
                    'use_composition': 'False',
                    'use_respawn': 'False',
                },
            ),
        ]
    )
