"""E1 21-DOF motion mimic task configuration.

This is a re-implementation of BeyondMimic (https://beyondmimic.github.io/).

Based on https://github.com/HybridRobotics/whole_body_tracking
Commit: f8e20c880d9c8ec7172a13d3a88a65e3a5a88448
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from src import DATASET_PATH
from src.assets.e1_21dof import E1_21DOF_ACTION_SCALE, get_e1_21dof_robot_cfg
from src.tasks.mimic import mdp
from src.tasks.mimic.mdp import MotionCommandCfg

VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}


def e1_21dof_flat_mimic_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the flat-terrain motion tracking task for E1 21-DOF."""

  ##
  # Observations
  ##

  actor_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.25, n_max=0.25),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity_from_sensor,
      params={"sensor_name": "robot/imu_upvector"},
      noise=NoiseModelWithAdditiveBiasCfg(
        noise_cfg=Unoise(n_min=-0.05, n_max=0.05),
        bias_noise_cfg=Unoise(n_min=-0.02, n_max=0.02, operation="abs"),
      ),
      delay_min_lag=0,
      delay_max_lag=1,
      delay_hold_prob=0.9,
      delay_update_period=5,
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=NoiseModelWithAdditiveBiasCfg(
        noise_cfg=Unoise(n_min=-0.2, n_max=0.2),
        bias_noise_cfg=Unoise(n_min=-0.05, n_max=0.05, operation="abs"),
      ),
      delay_min_lag=0,
      delay_max_lag=1,
      delay_hold_prob=0.9,
      delay_update_period=5,
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
      delay_min_lag=0,
      delay_max_lag=1,
      delay_hold_prob=0.9,
      delay_update_period=5,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
      delay_min_lag=0,
      delay_max_lag=1,
      delay_hold_prob=0.9,
      delay_update_period=5,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "motion": MotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      # Override in robot cfg.
      motion_file="",
      anchor_body_name="",
      body_names=(),
    )
  }

  ##
  # Events
  ##

  events: dict[str, EventTermCfg] = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": VELOCITY_RANGE},
    ),
    "body_inertia": EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        # e^(2 * alpha) gives approximately 0.95--1.05 mass/inertia scaling.
        "alpha_range": (-0.025, 0.025),
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set in robot cfg.
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "pd_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "effort_limits": EventTermCfg(
      mode="startup",
      func=dr.effort_limits,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "effort_limit_range": (0.9, 1.0),
        "operation": "scale",
      },
    ),
    "joint_damping": EventTermCfg(
      mode="startup",
      func=dr.joint_damping,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (0.8, 1.2),
        "operation": "scale",
      },
    ),
    "joint_friction": EventTermCfg(
      mode="startup",
      func=dr.joint_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (0.8, 1.2),
        "operation": "scale",
      },
    ),
    "joint_armature": EventTermCfg(
      mode="startup",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards: dict[str, RewardTermCfg] = {
    "motion_global_root_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_global_root_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": (),  # Set per-robot.
      },
    ),
  }

  ##
  # Assemble and return
  ##

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )

  cfg.scene.entities = {
    "robot": get_e1_21dof_robot_cfg(
      action_delay_range=(0, 0) if play else (0, 4)
    )
  }
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = E1_21DOF_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = str(
    # DATASET_PATH / "e1_21dof" / "mimic/dance_npz/MJ_dance.npz"
    # DATASET_PATH / "e1_21dof" / "mimic/dance_npz/victory1.npz"
    # DATASET_PATH / "e1_21dof" / "mimic/dance_npz/victory2.npz"
    DATASET_PATH / "e1_21dof" / "mimic/backflip/robot_backflip_soma.npz"
  )
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_pitch_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_pitch_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
  )
  cfg.viewer.body_name = "torso_link"

  if not has_state_estimation:
    actor_terms_without_state_estimation = {
      name: term
      for name, term in cfg.observations["actor"].terms.items()
      if name not in ("motion_anchor_pos_b", "base_lin_vel")
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=actor_terms_without_state_estimation,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    for term in cfg.observations["actor"].terms.values():
      term.delay_min_lag = 0
      term.delay_max_lag = 0
      term.delay_hold_prob = 0.0
      term.delay_update_period = 0
    cfg.events.clear()
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "start"

  return cfg
