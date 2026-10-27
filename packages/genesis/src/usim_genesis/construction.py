"""Preflight and construction of independently owned native Genesis scenes."""

from contextlib import ExitStack
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Any

import numpy as np

from usim.model import ModelError, ModelSource
from usim.model_io import ModelIO, OwnedModel
from usim.specs import (
    ControlSpec,
    EntitySpec,
    ExecutionSpec,
    RegisteredAgentRef,
    SceneSpec,
    SensorSpec,
)

from .models import ParsedAsset
from .lidar import BoundLidar, add_lidar
from .native import BoundEntity, add_entity, bind, resolve_joint_name
from .sensors import BoundCamera, CAMERA_KINDS


@dataclass(frozen=True, slots=True)
class PreparedScene:
    specs: tuple[EntitySpec, ...]
    assets: tuple[OwnedModel[ParsedAsset], ...]
    controls: tuple[tuple[str, ControlSpec], ...]
    sensors: tuple[tuple[str, SensorSpec], ...]


def prepare(spec: SceneSpec, models: ModelIO[ParsedAsset], resources: ExitStack) -> PreparedScene:
    entities = list(spec.entities)
    controls = []
    sensors = []
    if spec.world is not None:
        if any(item.name == '__world__' for item in spec.entities + spec.agents):
            raise ModelError('The __world__ entity name is reserved when a world is supplied')
        entities.insert(0, EntitySpec('__world__', spec.world))
    for agent in spec.agents:
        match agent.model:
            case RegisteredAgentRef(backend=backend, uid=uid):
                raise ModelError(
                    f'Genesis requires a direct ModelSource; registered agent {backend}:{uid} '
                    'belongs to a backend registry and is not a portable asset'
                )
            case ModelSource():
                entities.append(
                    EntitySpec(agent.name, agent.model, agent.pose, agent.requirements)
                )
        for sensor in agent.sensors:
            if sensor.kind not in CAMERA_KINDS | {'lidar'}:
                raise ModelError(f'Unsupported Genesis sensor kind: {sensor.kind}')
            sensors.append((agent.name, sensor))
        controls.append((agent.name, agent.control))
    assets = tuple(
        resources.enter_context(models.load(models.plan(entity.source, entity.requirements)))
        for entity in entities
    )
    return PreparedScene(tuple(entities), assets, tuple(controls), tuple(sensors))


def initialize(options: ExecutionSpec):
    """Initialize Genesis once, after dependency-free request preflight."""
    if options.mode != 'episode':
        raise ModelError('Genesis exposes episode execution, not a continuous driver')
    if options.backend not in {'cpu', 'gpu', 'cuda'}:
        raise ModelError('Genesis backend must be cpu, gpu, or cuda')
    if options.render_mode not in {'none', 'human', 'rgb_array'} or options.obs_mode not in {
        'state',
        'rgb',
        'rgbd',
    }:
        raise ModelError(
            'Grouped Genesis currently exposes state observations and optional viewer'
        )
    if options.native_task != 'JointControl':
        raise ModelError('Grouped Genesis builds declared scenes, not native registered tasks')
    gs = import_module('genesis')
    backend = gs.cpu if options.backend == 'cpu' else gs.cuda
    if not gs._initialized:
        gs.init(backend=backend)
    elif gs.backend != backend:
        raise ModelError('Genesis is already initialized with a different backend')
    return gs


@dataclass(frozen=True, slots=True)
class BuiltScene:
    # This boundary owns dynamically typed optional Genesis SDK instances.
    native: Any
    entities: dict[str, BoundEntity]
    controllers: dict[str, ControlSpec]
    cameras: dict[str, tuple[BoundCamera, ...]]
    lidars: dict[str, tuple[BoundLidar, ...]]


def build(prepared: PreparedScene, options: ExecutionSpec, resources: ExitStack) -> BuiltScene:
    if options.obs_mode != 'state' and not any(
        sensor.kind in CAMERA_KINDS for _, sensor in prepared.sensors
    ):
        raise ModelError('Image observations require a declared camera SensorSpec')
    gs = initialize(options)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / options.sim_rate),
        vis_options=gs.options.VisOptions(
            split_envs=True, rendered_envs_idx=list(range(options.n_envs))
        ),
        show_viewer=options.render_mode == 'human',
    )
    resources.callback(scene.destroy)
    natives = tuple(
        add_entity(scene, asset.value, spec)
        for spec, asset in zip(prepared.specs, prepared.assets)
    )
    cameras: dict[str, tuple[BoundCamera, ...]] = {}
    lidars: dict[str, tuple[BoundLidar, ...]] = {}
    native_by_name = {spec.name: native for spec, native in zip(prepared.specs, natives)}
    for owner, sensor in prepared.sensors:
        if sensor.kind == 'lidar':
            lidar = add_lidar(scene, native_by_name[owner], (owner, sensor))
            lidars[owner] = lidars.get(owner, ()) + (lidar,)
            continue
        if sensor.kind == 'camera':
            sensor = replace(sensor, kind='rgbd' if options.obs_mode == 'rgbd' else 'rgb')
        camera = scene.add_camera(res=(sensor.width, sensor.height), fov=69, GUI=False)
        cameras[owner] = cameras.get(owner, ()) + (BoundCamera(camera, sensor, owner),)
    scene.build(n_envs=options.n_envs)
    entities = {
        spec.name: bind(native, spec, options.n_envs)
        for spec, native in zip(prepared.specs, natives)
    }
    for owner in cameras.keys() | lidars.keys():
        bound = entities[owner]
        for camera in cameras.get(owner, ()):
            camera.attach(bound.native)
        attached = cameras.get(owner, ()) + lidars.get(owner, ())
        entities[owner] = replace(
            bound,
            schema=replace(
                bound.schema,
                sensors=tuple(layout for camera in attached for layout in camera.layouts),
            ),
        )
    controllers = {}
    for name, control in prepared.controls:
        bound = entities[name]
        scalar = tuple(
            joint.name
            for joint in bound.schema.joints
            if joint.qpos_range[1] - joint.qpos_range[0] == 1
            and joint.qvel_range[1] - joint.qvel_range[0] == 1
        )
        groups = control.position_joints + control.velocity_joints + control.effort_joints
        aliases = {resolve_joint_name(bound, name): name for name in groups}
        names = control.joint_names or tuple(aliases.get(name, name) for name in scalar)
        native_names = tuple(resolve_joint_name(bound, name) for name in names)
        if not set(native_names) <= set(scalar) or len(set(native_names)) != len(native_names):
            raise ModelError('Controlled joints must exist and have exactly one DOF')
        if control.initial_qpos is not None and len(control.initial_qpos) != len(names):
            raise ModelError('Initial joint positions must match mapped joints')
        if groups:
            selected = dict(zip(native_names, names))
            remapped = []
            for group in (control.position_joints, control.velocity_joints, control.effort_joints):
                native_group = tuple(resolve_joint_name(bound, name) for name in group)
                if not set(native_group) <= set(selected):
                    raise ModelError('Control groups must belong to the resolved joint order')
                remapped.append(tuple(selected[name] for name in native_group))
            control = replace(
                control,
                position_joints=remapped[0],
                velocity_joints=remapped[1],
                effort_joints=remapped[2],
            )
        if not groups:
            match control.native_mode:
                case 'pd_joint_pos':
                    control = replace(control, position_joints=names)
                case 'pd_joint_vel':
                    control = replace(control, velocity_joints=names)
                case 'joint_force':
                    control = replace(control, effort_joints=names)
                case _:
                    raise ModelError(f'Unsupported grouped control mode: {control.native_mode}')
        resolved = replace(control, joint_names=names)
        controllers[name] = resolved
        indices = [bound.native.get_joint(joint).dofs_idx_local[0] for joint in native_names]
        if indices:
            bound.native.set_dofs_kp(np.full(len(indices), 100.0), indices)
            bound.native.set_dofs_kv(np.full(len(indices), 10.0), indices)
    return BuiltScene(scene, entities, controllers, cameras, lidars)
