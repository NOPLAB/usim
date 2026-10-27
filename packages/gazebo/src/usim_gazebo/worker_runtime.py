"""Gazebo process configuration, independent of ROS and the host usim wheel."""

from __future__ import annotations

import os
from pathlib import Path
import select
import subprocess
import sys
import time
from typing import Protocol
import xml.etree.ElementTree as ET


SERVICE_TYPES = {
    '/spawn_entity': ('gazebo_msgs.srv', 'SpawnEntity'),
    '/delete_entity': ('gazebo_msgs.srv', 'DeleteEntity'),
    '/get_entity_state': ('gazebo_msgs.srv', 'GetEntityState'),
    '/set_entity_state': ('gazebo_msgs.srv', 'SetEntityState'),
    '/get_model_properties': ('gazebo_msgs.srv', 'GetModelProperties'),
    '/get_joint_properties': ('gazebo_msgs.srv', 'GetJointProperties'),
    '/set_joint_properties': ('gazebo_msgs.srv', 'SetJointProperties'),
    '/get_link_properties': ('gazebo_msgs.srv', 'GetLinkProperties'),
    '/set_link_properties': ('gazebo_msgs.srv', 'SetLinkProperties'),
    '/set_model_configuration': ('gazebo_msgs.srv', 'SetModelConfiguration'),
    '/apply_joint_effort': ('gazebo_msgs.srv', 'ApplyJointEffort'),
    '/clear_joint_forces': ('gazebo_msgs.srv', 'JointRequest'),
    '/pause_physics': ('std_srvs.srv', 'Empty'),
    '/unpause_physics': ('std_srvs.srv', 'Empty'),
}


class DisplayOwner(Protocol):
    deadline: float
    display: subprocess.Popen[bytes] | None


def start_display(owner: DisplayOwner) -> None:
    read_fd, write_fd = os.pipe()
    try:
        owner.display = subprocess.Popen(
            [
                'Xvfb',
                '-displayfd',
                str(write_fd),
                '-screen',
                '0',
                '1280x720x24',
                '-nolisten',
                'tcp',
                '-ac',
            ],
            pass_fds=(write_fd,),
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
        os.close(write_fd)
        write_fd = -1
        if not select.select([read_fd], [], [], max(0.0, owner.deadline - time.monotonic()))[0]:
            raise TimeoutError('Xvfb did not acknowledge a ready display')
        display = os.read(read_fd, 64).decode().strip()
        if not display.isdigit():
            raise RuntimeError('Xvfb exited without a display')
        os.environ['DISPLAY'] = ':' + display
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    finally:
        os.close(read_fd)
        if write_fd != -1:
            os.close(write_fd)


def prepare_world(directory: Path, xml: str) -> Path:
    root = ET.fromstring(xml)
    worlds = root.findall('world')
    if len(worlds) != 1:
        raise ValueError('Expected exactly one world')
    world = worlds[0]
    physics = world.find('physics')
    if physics is not None and physics.get('type', 'ode') != 'ode':
        raise ValueError('Grouped Gazebo joint control requires the ODE physics engine')
    for plugin_name in ('state', 'properties'):
        plugin = ET.SubElement(
            world,
            'plugin',
            name='usim_' + plugin_name,
            filename=f'libgazebo_ros_{plugin_name}.so',
        )
        ET.SubElement(ET.SubElement(plugin, 'ros'), 'namespace').text = '/'
    world_path = directory / 'world.sdf'
    ET.ElementTree(root).write(world_path, encoding='unicode')
    return world_path
