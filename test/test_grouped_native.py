"""Opt-in CPU tests of the grouped model-first API against installed native engines."""

import os

import numpy as np
import pytest

from usim import (
    AgentSpec,
    ControlSpec,
    ControlTargets,
    EntitySpec,
    ExecutionSpec,
    ImportRequirements,
    ModelSource,
    Pose,
    SceneSpec,
    SensorSpec,
    create_backend,
)


@pytest.mark.skipif(
    os.environ.get('USIM_GROUPED_NATIVE') not in {'genesis', 'maniskill'},
    reason='Select an installed CPU engine with USIM_GROUPED_NATIVE=genesis|maniskill',
)
@pytest.mark.parametrize(
    'format_name',
    [
        'urdf',
        'mjcf',
        pytest.param(
            'usd',
            marks=pytest.mark.skipif(
                os.environ.get('USIM_GROUPED_NATIVE') != 'genesis',
                reason='ManiSkill has no native USD loader',
            ),
        ),
    ],
)
def test_real_model_import_and_named_control(tmp_path, format_name):
    # Given real URDF or MJCF input, not a native SDK fixture.
    engine = os.environ['USIM_GROUPED_NATIVE']
    source_text = {
        'urdf': (
            '<robot name="robot"><link name="base"/>'
            '<link name="arm"><inertial><mass value="1"/>'
            '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>'
            '</inertial><collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>'
            '</link><joint name="hinge" type="revolute"><parent link="base"/>'
            '<child link="arm"/><origin xyz="0 0 0.5"/><axis xyz="0 0 1"/>'
            '<limit lower="-1" upper="1" effort="10" velocity="2"/></joint></robot>'
        ),
        'mjcf': (
            '<mujoco><compiler angle="radian"/><worldbody><body name="base">'
            '<geom type="box" size="0.05 0.05 0.05" mass="1"/>'
            '<body name="arm" pos="0 0 0.5"><joint name="hinge" type="hinge" '
            'axis="0 0 1" limited="true" range="-1 1"/>'
            '<inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>'
            '<geom type="box" size="0.05 0.05 0.05" mass="1"/>'
            '</body></body></worldbody></mujoco>'
        ),
    }
    if format_name == 'usd':
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        path = tmp_path / 'robot.usda'
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, '/Robot').GetPrim()
        stage.SetDefaultPrim(root)
        UsdPhysics.ArticulationRootAPI.Apply(root)
        for name, height in (('base', 0.0), ('arm', 0.5)):
            link = UsdGeom.Xform.Define(stage, '/Robot/' + name)
            link.AddTranslateOp().Set(Gf.Vec3d(0, 0, height))
            UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
            mass = UsdPhysics.MassAPI.Apply(link.GetPrim())
            mass.CreateMassAttr(1.0)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.01))
            cube = UsdGeom.Cube.Define(stage, '/Robot/' + name + '/shape')
            cube.CreateSizeAttr(0.1)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        anchor = UsdPhysics.FixedJoint.Define(stage, '/Robot/fixed_base')
        anchor.CreateBody1Rel().SetTargets(['/Robot/base'])
        joint = UsdPhysics.RevoluteJoint.Define(stage, '/Robot/hinge')
        joint.CreateBody0Rel().SetTargets(['/Robot/base'])
        joint.CreateBody1Rel().SetTargets(['/Robot/arm'])
        joint.CreateAxisAttr('Z')
        joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.5))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLowerLimitAttr(-57.2958)
        joint.CreateUpperLimitAttr(57.2958)
        stage.GetRootLayer().Save()
        stage = None
    else:
        path = tmp_path / ('robot.urdf' if format_name == 'urdf' else 'robot.xml')
        path.write_text(source_text[format_name], encoding='utf-8')
    accepted_loss = (
        frozenset({'inertia'}) if engine == 'maniskill' and format_name == 'mjcf' else frozenset()
    )
    requirements = ImportRequirements(
        preserve=frozenset({'joint_structure', 'inertia', 'collision'}) - accepted_loss,
        allow_loss=accepted_loss,
        base='fixed' if format_name == 'urdf' else 'authored',
    )
    spec = SceneSpec(
        agents=(
            AgentSpec(
                'robot',
                ModelSource.from_path(path),
                ControlSpec(
                    joint_names=('hinge',),
                    position_joints=('hinge',),
                    initial_qpos=(0.0,),
                    joint_stiffness=(100.0,),
                    joint_damping=(10.0,),
                    joint_force_limits=(10.0,),
                ),
                requirements=requirements,
            ),
        )
    )
    with create_backend(engine).open(spec, ExecutionSpec()) as session:
        initial = np.asarray(session.state.for_entity('robot').get_qpos(), dtype=float).copy()
        # When native physics executes previously supplied physical targets.
        session.control.for_entity('robot').apply(ControlTargets(positions={'hinge': 0.2}))
        for _ in range(8):
            session.execution.step()
        final = np.asarray(session.state.for_entity('robot').get_qpos(), dtype=float)
        # Then imported joints move physically and remain finite.
        assert np.all(np.isfinite(final))
        assert np.max(np.abs(final - initial)) > 1e-5


@pytest.mark.skipif(
    os.environ.get('USIM_GAZEBO_GROUPED_NATIVE') != '1',
    reason='Native Gazebo rendering is opt-in',
)
def test_real_gazebo_sensor_attachments_and_native_frames(tmp_path):
    from usim_gazebo.backend import Backend

    # Given a real lit world, a sensor carrier and an obstacle in front of its camera.
    world = tmp_path / 'world.sdf'
    world.write_text(
        '<sdf version="1.6"><world name="test"><scene>'
        '<ambient>1 1 1 1</ambient><background>0 0 0 1</background>'
        '</scene></world></sdf>',
        encoding='utf-8',
    )
    carrier = tmp_path / 'carrier.sdf'
    carrier.write_text(
        '<sdf version="1.6"><model name="carrier"><static>true</static>'
        '<link name="base"><inertial><mass>1</mass><inertia><ixx>0.01</ixx>'
        '<iyy>0.01</iyy><izz>0.01</izz></inertia></inertial></link></model></sdf>',
        encoding='utf-8',
    )
    target = tmp_path / 'target.sdf'
    target.write_text(
        '<sdf version="1.6"><model name="target"><static>true</static><link name="box">'
        '<collision name="shape"><geometry><box><size>0.3 0.3 0.3</size></box></geometry>'
        '</collision><visual name="shape"><geometry><box><size>0.3 0.3 0.3</size></box>'
        '</geometry><material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse>'
        '</material></visual></link></model></sdf>',
        encoding='utf-8',
    )
    spec = SceneSpec(
        world=ModelSource.from_path(world),
        agents=(
            AgentSpec(
                'robot',
                ModelSource.from_path(carrier),
                sensors=(
                    SensorSpec(
                        'camera',
                        'rgbd',
                        parent='base',
                        pose=Pose(position=(0, 0, 0.2)),
                        width=64,
                        height=48,
                    ),
                    SensorSpec(
                        'scan', 'lidar', parent='base', pose=Pose(position=(0, 0, 0.2)), width=65
                    ),
                ),
            ),
        ),
        entities=(
            EntitySpec(
                'target',
                ModelSource.from_path(target),
                Pose(position=(1, 0, 0.2)),
                ImportRequirements(role='rigid_body'),
            ),
        ),
    )
    options = ExecutionSpec(mode='continuous', render_mode='rgb_array', obs_mode='rgbd')
    with Backend().open(spec, options) as session:
        # When native physics/rendering starts and subscribed sensor messages arrive.
        session.execution.resume()
        observation = session.state.for_entity('robot').get_observation()
        session.execution.pause()
        # Then real pixels and ranges see the authored obstacle, not synthetic placeholders.
        rgb = np.asarray(observation.rgb_image)
        depth = np.asarray(observation.depth_image, dtype=float)
        assert rgb.shape == (48, 64, 3)
        assert rgb.dtype == np.uint8
        assert rgb.max() > 0
        assert depth.shape[:2] == (48, 64)
        assert np.any(np.isfinite(depth) & (depth > 0) & (depth < 2))
        lidar_layout = next(
            item
            for item in session.state.for_entity('robot').schema.sensors
            if item.kind == 'lidar'
        )
        ranges = observation.extra['sensors'][lidar_layout.name]
        assert ranges.shape == (65,)
        assert np.any(np.isfinite(ranges) & (ranges > 0) & (ranges < 2))
