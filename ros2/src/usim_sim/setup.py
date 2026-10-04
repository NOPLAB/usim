# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 nop

from setuptools import find_packages, setup

setup(
    name='usim_sim',
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/usim_sim']),
        ('share/usim_sim', ['package.xml']),
    ],
    install_requires=['setuptools', 'usim==0.1.0'],
    zip_safe=True,
    maintainer='nop',
    maintainer_email='noplab90@gmail.com',
    description='Configurable ROS 2 execution bridge for usim engines',
    license='MIT',
    entry_points={'console_scripts': ['usim_sim_node = usim.bridges.simulation_ros2:main']},
)
