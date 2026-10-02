"""Exercise geometry authoring without Raspicat, ROS or external mesh assets."""

import math
import unittest
import xml.etree.ElementTree as ET

from usim.robot import MobileRobot, RobotGeometryError, render_robot


class MobileRobotTest(unittest.TestCase):
    def test_custom_geometry_survives_urdf_generation(self):
        configuration = MobileRobot(
            name='delivery_robot',
            wheel_radius=0.09,
            wheel_separation=0.36,
            body_size=(0.5, 0.28, 0.12),
            mass=8.0,
            camera_offset=(0.15, 0.0, 0.35),
        )
        root = ET.fromstring(render_robot(configuration))
        links = {link.attrib['name']: link for link in root.findall('link')}
        joints = {joint.attrib['name']: joint for joint in root.findall('joint')}
        self.assertEqual(root.attrib['name'], 'delivery_robot')
        self.assertEqual(
            links['base_link'].find('collision/geometry/box').attrib['size'], '0.5 0.28 0.12'
        )
        for side, expected_y in (('left', 0.18), ('right', -0.18)):
            wheel = links[f'{side}_wheel_link']
            self.assertEqual(float(wheel.find('visual/geometry/cylinder').attrib['radius']), 0.09)
            self.assertEqual(float(wheel.find('collision/geometry/sphere').attrib['radius']), 0.09)
            position = [
                float(value)
                for value in joints[f'{side}_wheel_joint'].find('origin').attrib['xyz'].split()
            ]
            self.assertEqual(position, [0.0, expected_y, 0.09])
            self.assertEqual(joints[f'{side}_wheel_joint'].find('axis').attrib['xyz'], '0 1 0')
        self.assertEqual(joints['camera_joint'].find('origin').attrib['xyz'], '0.15 0.0 0.35')
        self.assertFalse(list(root.iter('mesh')))
        self.assertAlmostEqual(
            sum(
                float(link.find('inertial/mass').attrib['value'])
                for link in links.values()
                if link.find('inertial') is not None
            ),
            8.0,
        )
        # Statically stable tripod: body mass lies between the wheel axle and the caster.
        body_x = float(links['base_link'].find('inertial/origin').attrib['xyz'].split()[0])
        caster_x = float(joints['caster_joint'].find('origin').attrib['xyz'].split()[0])
        self.assertLess(caster_x, body_x)
        self.assertLess(body_x, 0.0)

    def test_rejects_invalid_geometry_before_rendering(self):
        for radius in (0.0, -0.1, math.inf, math.nan):
            with self.subTest(radius=radius), self.assertRaises(RobotGeometryError):
                MobileRobot(wheel_radius=radius)
        with self.assertRaises(RobotGeometryError):
            MobileRobot(wheel_separation=0.1)
        with self.assertRaises(RobotGeometryError):
            MobileRobot(camera_offset=(0.0, math.nan, 0.0))
