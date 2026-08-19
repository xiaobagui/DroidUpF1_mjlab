"""E1 21-DOF flat-ground velocity-commanded locomotion AMP task."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from src.assets.e1_21dof import get_e1_21dof_robot_cfg
from src.tasks.amp import mdp
from src.tasks.amp.constants import AMP_KEY_BODY_NAMES, MJLAB_JOINT_NAMES
from src.tasks.amp.mdp import UniformVelocityCommandCfg


def e1_21dof_walk_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the local E1 walk-and-run AMP task."""
  all_joints = SceneEntityCfg(
    "robot", joint_names=MJLAB_JOINT_NAMES, preserve_order=True
  )
  torso = SceneEntityCfg("robot", body_names=("torso_link",))
  feet = SceneEntityCfg(
    "robot",
    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
    preserve_order=True,
  )
  key_bodies = SceneEntityCfg(
    "robot", body_names=AMP_KEY_BODY_NAMES, preserve_order=True
  )
  ankle_roll = SceneEntityCfg(
    "robot",
    joint_names=("left_ankle_roll_joint", "right_ankle_roll_joint"),
    preserve_order=True,
  )
  hip_yaw = SceneEntityCfg(
    "robot",
    joint_names=("left_hip_yaw_joint", "right_hip_yaw_joint"),
    preserve_order=True,
  )
  hip_roll = SceneEntityCfg(
    "robot",
    joint_names=("left_hip_roll_joint", "right_hip_roll_joint"),
    preserve_order=True,
  )
  waist_yaw = SceneEntityCfg("robot", joint_names=("waist_yaw_joint",))

  observations = {
    "actor": ObservationGroupCfg(
      terms={
        "frame": ObservationTermCfg(
          func=mdp.actor_frame,
          params={"joint_cfg": all_joints, "torso_cfg": torso},
          clip=(-100.0, 100.0),
        )
      },
      concatenate_terms=True,
      enable_corruption=True,
      history_length=5,
      flatten_history_dim=True,
      nan_policy="sanitize",
    ),
    "critic": ObservationGroupCfg(
      terms={
        "frame": ObservationTermCfg(
          func=mdp.critic_frame,
          params={
            "joint_cfg": all_joints,
            "torso_cfg": torso,
            "feet_contact_sensor_name": "feet_ground_contact",
          },
          clip=(-100.0, 100.0),
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
      history_length=5,
      flatten_history_dim=True,
      nan_policy="sanitize",
    ),
    "amp": ObservationGroupCfg(
      terms={
        "state": ObservationTermCfg(
          func=mdp.amp_state,
          params={"joint_cfg": all_joints, "key_body_cfg": key_bodies},
          clip=(-100.0, 100.0),
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
      nan_policy="sanitize",
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(10.0, 10.0),
      rel_standing_envs=0.1,
      rel_turning_envs=0.1,
      rel_lateral_envs=0.1,
      turning_deadband=0.4,
      lateral_deadband=0.15,
      debug_vis=False,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 0.8),
        lin_vel_y=(-0.3, 0.3),
        ang_vel_z=(-0.8, 0.8),
        pure_turn_ang_vel_z=(-1.0, 1.0),
        pure_lateral_lin_vel_y=(-0.3, 0.3),
      ),
    ),
  }

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "yaw": (-math.pi, math.pi),
        },
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.5, 0.5),
          "roll": (-0.5, 0.5),
          "pitch": (-0.5, 0.5),
          "yaw": (-0.5, 0.5),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (-0.5, 0.5),
        "asset_cfg": all_joints,
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(7.0, 10.0),
      params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    ),
    "body_inertia": EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "alpha_range": (-0.05, 0.05),
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": torso,
        "operation": "add",
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
    "pd_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.8, 1.2),
        "kd_range": (0.8, 1.2),
        "operation": "scale",
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=r"^(left|right)_foot[1-7]_collision$"
        ),
        "operation": "abs",
        "ranges": (0.3, 1.0),
        "shared_random": True,
      },
    ),
  }

  rewards = {
    "track_lin_vel_xy": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=3.0,
      params={"std": 0.5, "command_name": "twist", "asset_cfg": all_joints},
    ),
    "track_ang_vel_z": RewardTermCfg(
      func=mdp.track_yaw_velocity,
      weight=2.0,
      params={"std": 0.5, "command_name": "twist", "asset_cfg": all_joints},
    ),
    "alive": RewardTermCfg(func=mdp.is_alive, weight=0.1),
    "lin_vel_z": RewardTermCfg(
      func=mdp.vertical_velocity_l2, weight=-2.0, params={"asset_cfg": all_joints}
    ),
    "ang_vel_xy": RewardTermCfg(
      func=mdp.base_angular_velocity_xy_l2,
      weight=-0.05,
      params={"asset_cfg": all_joints},
    ),
    "joint_vel": RewardTermCfg(
      func=mdp.joint_vel_l2, weight=-1.0e-3, params={"asset_cfg": all_joints}
    ),
    "joint_acc": RewardTermCfg(
      func=mdp.joint_acc_l2, weight=-5.0e-7, params={"asset_cfg": all_joints}
    ),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "action_smoothness": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.01),
    "joint_limits": RewardTermCfg(
      func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": all_joints}
    ),
    "energy": RewardTermCfg(
      func=mdp.energy_l2, weight=-2.0e-5, params={"asset_cfg": all_joints}
    ),
    "flat_orientation": RewardTermCfg(
      func=mdp.flat_orientation_l2,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "torso_orientation": RewardTermCfg(
      func=mdp.torso_orientation_l2, weight=-2.0, params={"asset_cfg": torso}
    ),
    "torso_ang_vel_xy": RewardTermCfg(
      func=mdp.torso_ang_vel_xy_l2, weight=-0.2, params={"asset_cfg": torso}
    ),
    "torso_roll": RewardTermCfg(
      func=mdp.torso_roll_l2, weight=-5.0, params={"asset_cfg": torso}
    ),
    "torso_roll_ang_vel": RewardTermCfg(
      func=mdp.torso_roll_ang_vel_l2, weight=-0.3, params={"asset_cfg": torso}
    ),
    "stand_torso_pitch": RewardTermCfg(
      func=mdp.stand_torso_pitch_l2, weight=-20.0, params={"asset_cfg": torso}
    ),
    "stand_hip_yaw": RewardTermCfg(
      func=mdp.zero_command_joint_l2, weight=-5.0, params={"asset_cfg": hip_yaw}
    ),
    "stand_waist_yaw": RewardTermCfg(
      func=mdp.zero_command_joint_l2, weight=-2.0, params={"asset_cfg": waist_yaw}
    ),
    "stand_feet_heading": RewardTermCfg(
      func=mdp.stand_feet_heading_l2,
      weight=-2.0,
      params={"feet_cfg": feet},
    ),
    "stable_stand_hip_roll": RewardTermCfg(
      func=mdp.stable_stand_hip_roll_norm,
      weight=-2.0,
      params={
        "asset_cfg": hip_roll,
        "torso_cfg": torso,
        "sensor_name": "feet_ground_contact",
      },
    ),
    "stable_stand_feet_fore_aft": RewardTermCfg(
      func=mdp.stable_stand_feet_fore_aft_l1,
      weight=-2.0,
      params={
        "feet_cfg": feet,
        "torso_cfg": torso,
        "sensor_name": "feet_ground_contact",
      },
    ),
    "stable_stand_feet_lateral_center": RewardTermCfg(
      func=mdp.stable_stand_feet_lateral_center_l1,
      weight=-5.0,
      params={
        "feet_cfg": feet,
        "torso_cfg": torso,
        "sensor_name": "feet_ground_contact",
      },
    ),
    "feet_slide": RewardTermCfg(
      func=mdp.feet_slide,
      weight=-0.25,
      params={"sensor_name": "feet_ground_contact", "feet_cfg": feet},
    ),
    "feet_crossing": RewardTermCfg(
      func=mdp.feet_crossing,
      weight=-2.0,
      params={"minimum_distance": 0.16, "feet_cfg": feet},
    ),
    "feet_spacing": RewardTermCfg(
      func=mdp.feet_spacing_non_lateral,
      weight=-1.0,
      params={
        "minimum_distance": 0.18,
        "maximum_distance": 0.28,
        "lateral_command_threshold": 0.1,
        "feet_cfg": feet,
      },
    ),
    "feet_support_roll": RewardTermCfg(
      func=mdp.feet_support_roll_l2,
      weight=-2.0,
      params={"feet_cfg": feet, "sensor_name": "feet_ground_contact"},
    ),
    "ankle_roll_action_rate": RewardTermCfg(
      func=mdp.ankle_roll_action_rate_l2,
      weight=-0.04,
      params={
        "asset_cfg": ankle_roll,
        "sensor_name": "feet_ground_contact",
      },
    ),
    "ankle_roll": RewardTermCfg(
      func=mdp.ankle_roll_l2, weight=-0.2, params={"asset_cfg": ankle_roll}
    ),
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_contacts,
      weight=-1.0,
      params={"sensor_name": "illegal_ground_contact", "threshold": 1.0},
    ),
    "termination": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "illegal_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": "illegal_ground_contact", "threshold": 1.0},
    ),
    "base_height": TerminationTermCfg(
      func=mdp.root_height_below_minimum,
      params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("robot")},
    ),
    "nan": TerminationTermCfg(func=mdp.nan_detection),
  }

  curriculum = {
    "command_velocity": CurriculumTermCfg(
      func=mdp.command_velocity_stages,
      params={
        "command_name": "twist",
        "stages": (
          {
            "step": 0,
            "lin_vel_x": (-0.5, 0.8),
            "lin_vel_y": (-0.3, 0.3),
            "ang_vel_z": (-0.8, 0.8),
            "pure_turn_ang_vel_z": (-1.0, 1.0),
            "pure_lateral_lin_vel_y": (-0.3, 0.3),
            "turning_deadband": 0.4,
            "lateral_deadband": 0.15,
          },
          {
            "step": 5_000 * 24,
            "lin_vel_x": (-0.7, 1.1),
            "lin_vel_y": (-0.4, 0.4),
            "ang_vel_z": (-1.2, 1.2),
            "pure_turn_ang_vel_z": (-1.4, 1.4),
            "pure_lateral_lin_vel_y": (-0.45, 0.45),
            "turning_deadband": 0.5,
            "lateral_deadband": 0.15,
          },
          {
            "step": 10_000 * 24,
            "lin_vel_x": (-0.8, 1.4),
            "lin_vel_y": (-0.5, 0.5),
            "ang_vel_z": (-1.5, 1.5),
            "pure_turn_ang_vel_z": (-1.8, 1.8),
            "pure_lateral_lin_vel_y": (-0.55, 0.55),
            "turning_deadband": 0.6,
            "lateral_deadband": 0.2,
          },
        ),
      },
    )
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1 if play else 4096,
      env_spacing=2.5,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=80,
      njmax=500,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
  cfg.scene.entities = {
    "robot": get_e1_21dof_robot_cfg(action_delay_range=(0, 0) if play else (0, 2))
  }
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(
        mode="subtree",
        pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      history_length=4,
      track_air_time=True,
    ),
    ContactSensorCfg(
      name="illegal_ground_contact",
      primary=ContactMatch(
        mode="body",
        pattern=(
          r"^(left_knee_link|right_knee_link|left_shoulder_roll_link|"
          r"right_shoulder_roll_link|left_elbow_link|right_elbow_link|"
          r"pelvis|torso_link)$"
        ),
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      history_length=4,
    ),
  )

  if play:
    cfg.episode_length_s = int(1.0e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.clear()
    cfg.curriculum = {}

  return cfg
