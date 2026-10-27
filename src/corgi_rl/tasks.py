"""Small stand and velocity-tracking tasks built against upstream mjlab."""

import math
import mujoco
import torch
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg

from .model import mjcf
from .spec import RobotSpec


def actor_observation(env, velocity_scale: float):
    data = env.scene["robot"].data
    return torch.cat((
        data.root_link_ang_vel_b,
        data.projected_gravity_b,
        env.command_manager.get_command("twist"),
        data.joint_pos - data.default_joint_pos,
        data.joint_vel * velocity_scale,
        env.action_manager.action,
    ), dim=-1)


def tracking_xy(env):
    d = env.scene["robot"].data
    cmd = env.command_manager.get_command("twist")
    return torch.exp(-((d.root_link_lin_vel_b[:, :2] - cmd[:, :2]) ** 2).sum(-1) / .09)


def tracking_yaw(env):
    d = env.scene["robot"].data
    cmd = env.command_manager.get_command("twist")
    return torch.exp(-(d.root_link_ang_vel_b[:, 2] - cmd[:, 2]) ** 2 / .25)


def upright(env):
    return torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 1)


def height(env, target: float):
    z = env.scene["robot"].data.root_link_pos_w[:, 2]
    return torch.exp(-((z - target) / .05) ** 2)


def pose_error(env):
    d = env.scene["robot"].data
    return ((d.joint_pos - d.default_joint_pos) ** 2).sum(-1)


def vertical_speed(env):
    return env.scene["robot"].data.root_link_lin_vel_b[:, 2].square()


def make_env_cfg(spec: RobotSpec, task="stand", num_envs=32, seed=42):
    if task not in ("stand", "walk"):
        raise ValueError("Task must be stand or walk")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    stand = task == "stand"
    robot = EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_string(mjcf(spec, scene=False)),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0, 0, spec.spawn_height_m),
            joint_pos={j.name: j.home_rad for j in spec.joints},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=tuple(j.name for j in spec.joints)),),
        ),
    )
    observations = ObservationGroupCfg(
        terms={"state": ObservationTermCfg(
            func=actor_observation,
            params={"velocity_scale": spec.joint_velocity_observation_scale},
        )}, concatenate_terms=True, enable_corruption=False,
    )
    return ManagerBasedRlEnvCfg(
        seed=seed,
        scene=SceneCfg(entities={"robot": robot}, num_envs=num_envs, extent=2,
                       terrain=TerrainEntityCfg(terrain_type="plane", collisions=(CollisionCfg(
                           geom_names_expr=(".*",),
                           contype=1, conaffinity=1, priority=0,
                           friction=(spec.floor_friction, .005, .0001), condim=3,
                       ),))),
        sim=SimulationCfg(nconmax=64, njmax=256,
                          mujoco=MujocoCfg(timestep=spec.physics_dt_s,
                                           iterations=10, ls_iterations=20,
                                           integrator="implicitfast", cone="pyramidal")),
        decimation=spec.control_decimation,
        episode_length_s=10.0,
        observations={"actor": observations, "critic": observations},
        actions={"joints": JointPositionActionCfg(
            entity_name="robot", actuator_names=(".*",),
            scale={j.name: j.action_scale_rad for j in spec.joints},
            use_default_offset=True,
            clip={j.name: (j.lower_rad, j.upper_rad) for j in spec.joints},
        )},
        commands={"twist": UniformVelocityCommandCfg(
            entity_name="robot", resampling_time_range=(3, 5),
            rel_standing_envs=1.0 if stand else .1,
            rel_heading_envs=0, rel_forward_envs=0,
            heading_command=False, debug_vis=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0., 0.) if stand else (-.2, .4),
                lin_vel_y=(0., 0.) if stand else (-.15, .15),
                ang_vel_z=(0., 0.) if stand else (-.5, .5),
            ),
        )},
        events={
            "reset_base": EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset",
                                       params={"pose_range": {}, "velocity_range": {}}),
            "reset_joints": EventTermCfg(func=mdp.reset_joints_by_offset, mode="reset",
                                         params={"position_range": (-.02, .02),
                                                 "velocity_range": (0., 0.)}),
        },
        rewards={
            "track_xy": RewardTermCfg(func=tracking_xy, weight=2),
            "track_yaw": RewardTermCfg(func=tracking_yaw, weight=.5),
            "upright": RewardTermCfg(func=upright, weight=1),
            "height": RewardTermCfg(func=height, weight=.5,
                                     params={"target": spec.spawn_height_m - .01}),
            "pose": RewardTermCfg(func=pose_error, weight=-.2 if stand else -.02),
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-.02),
            "vertical_speed": RewardTermCfg(func=vertical_speed, weight=-.1),
            "joint_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1),
        },
        terminations={
            "timeout": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "fall": TerminationTermCfg(func=mdp.bad_orientation,
                                        params={"limit_angle": math.pi / 3}),
            "low_body": TerminationTermCfg(func=mdp.root_height_below_minimum,
                                            params={"minimum_height": .08}),
            "nonfinite": TerminationTermCfg(func=mdp.nan_detection),
        },
    )
