"""Load sourced or Isaac-bundled ROS and retain its Windows DLL handle."""

from __future__ import annotations

import os
import sys
from pathlib import Path


_ROS_DLL_DIRECTORY = None


def load_ros_python() -> None:
    """Use a sourced ROS environment or Isaac's bundled ROS 2 Python modules."""
    try:
        import rclpy  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    distro = os.environ.setdefault('ROS_DISTRO', 'humble')
    os.environ.setdefault('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')
    isaac_path = Path(os.environ['ISAAC_PATH'])
    bundle = isaac_path / 'exts' / 'isaacsim.ros2.core' / distro
    if not bundle.is_dir():
        bundle = isaac_path / 'exts' / 'isaacsim.ros2.bridge' / distro
    python = bundle / 'rclpy'
    libraries = bundle / 'lib'
    if not python.is_dir() or not libraries.is_dir():
        raise RuntimeError(f'Isaac bundled ROS {distro} is unavailable')
    # Bundled ROS also contains NumPy; retain the environment's installed libraries.
    sys.path.append(str(python))
    if sys.platform == 'win32':
        os.environ['PATH'] += os.pathsep + str(libraries)
        global _ROS_DLL_DIRECTORY
        _ROS_DLL_DIRECTORY = os.add_dll_directory(str(libraries))
    elif str(libraries) not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
        raise RuntimeError(
            f'set LD_LIBRARY_PATH={libraries}:$LD_LIBRARY_PATH '
            'before starting Isaac to load bundled ROS libraries'
        )
    import rclpy  # noqa: F401
