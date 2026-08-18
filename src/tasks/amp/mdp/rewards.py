"""Task rewards accompanying the adversarial walking-style reward."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  command = env.command_manager.get_command(name)
  assert command is not None
  return command


def moving_mask(
  env: ManagerBasedRlEnv, command_name: str = "twist", threshold: float = 0.1
) -> torch.Tensor:
  command = _command(env, command_name)
  return (
    torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  ) > threshold


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  error = _command(env, command_name)[:, :2] - asset.data.root_link_lin_vel_b[:, :2]
  return torch.exp(-torch.sum(torch.square(error), dim=1) / std**2)


def track_yaw_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  error = _command(env, command_name)[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
  return torch.exp(-torch.square(error) / std**2)


def vertical_velocity_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def base_angular_velocity_xy_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def energy_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  power = torch.abs(
    asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
    * asset.data.joint_vel[:, asset_cfg.joint_ids]
  )
  return torch.linalg.vector_norm(power, dim=1)


def torso_orientation_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  gravity_b = quat_apply_inverse(quat, asset.data.gravity_vec_w)
  return torch.sum(torch.square(gravity_b[:, :2]), dim=1)


def torso_ang_vel_xy_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  return torch.sum(torch.square(quat_apply_inverse(quat, vel_w)[:, :2]), dim=1)


def torso_roll_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  gravity_b = quat_apply_inverse(quat, asset.data.gravity_vec_w)
  return torch.square(gravity_b[:, 1])


def torso_roll_ang_vel_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  return torch.square(quat_apply_inverse(quat, vel_w)[:, 0])


def stand_torso_pitch_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str = "twist",
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  gravity_b = quat_apply_inverse(quat, asset.data.gravity_vec_w)
  return torch.square(gravity_b[:, 0]) * (~moving_mask(env, command_name)).float()


def zero_command_joint_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str = "twist",
  threshold: float = 0.05,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  error = asset.data.joint_pos[:, asset_cfg.joint_ids] - default[:, asset_cfg.joint_ids]
  command = _command(env, command_name)
  magnitude = torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return torch.sum(torch.square(error), dim=1) * (magnitude < threshold).float()


def _feet_positions_b(asset: Entity, feet_cfg: SceneEntityCfg) -> torch.Tensor:
  pos_w = asset.data.body_link_pos_w[:, feet_cfg.body_ids, :]
  relative = pos_w - asset.data.root_link_pos_w.unsqueeze(1)
  quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, len(feet_cfg.body_ids), -1)
  return quat_apply_inverse(quat, relative)


def feet_crossing(
  env: ManagerBasedRlEnv,
  minimum_distance: float,
  feet_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[feet_cfg.name]
  feet = _feet_positions_b(asset, feet_cfg)
  signed_distance = feet[:, 0, 1] - feet[:, 1, 1]
  return torch.relu(minimum_distance - signed_distance)


def feet_spacing_non_lateral(
  env: ManagerBasedRlEnv,
  minimum_distance: float,
  maximum_distance: float,
  lateral_command_threshold: float,
  feet_cfg: SceneEntityCfg,
  command_name: str = "twist",
) -> torch.Tensor:
  asset: Entity = env.scene[feet_cfg.name]
  feet = _feet_positions_b(asset, feet_cfg)
  distance = feet[:, 0, 1] - feet[:, 1, 1]
  outside = torch.relu(minimum_distance - distance) + torch.relu(
    distance - maximum_distance
  )
  non_lateral = torch.abs(_command(env, command_name)[:, 1]) < lateral_command_threshold
  return outside * non_lateral.float()


def stand_feet_heading_l2(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  command_name: str = "twist",
  threshold: float = 0.05,
) -> torch.Tensor:
  """Keep both feet aligned with the pelvis while standing still."""
  asset: Entity = env.scene[feet_cfg.name]
  feet_quat_w = asset.data.body_link_quat_w[:, feet_cfg.body_ids, :]
  foot_forward_b = torch.zeros(
    (*feet_quat_w.shape[:-1], 3), device=env.device, dtype=feet_quat_w.dtype
  )
  foot_forward_b[..., 0] = 1.0
  foot_forward_w = quat_apply(feet_quat_w, foot_forward_b)
  pelvis_quat_w = asset.data.root_link_quat_w.unsqueeze(1).expand_as(feet_quat_w)
  foot_forward_pelvis = quat_apply_inverse(pelvis_quat_w, foot_forward_w)
  heading_error = torch.atan2(
    foot_forward_pelvis[..., 1], foot_forward_pelvis[..., 0]
  )
  command = _command(env, command_name)
  magnitude = torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return torch.sum(torch.square(heading_error), dim=1) * (magnitude < threshold).float()


def feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  feet_cfg: SceneEntityCfg,
  threshold: float = 1.0,
) -> torch.Tensor:
  asset: Entity = env.scene[feet_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force_history is not None
  force = torch.linalg.vector_norm(sensor.data.force_history, dim=-1).amax(dim=2)
  contact = force > threshold
  speed = torch.linalg.vector_norm(
    asset.data.body_link_lin_vel_w[:, feet_cfg.body_ids, :2], dim=-1
  )
  return torch.sum(speed * contact.float(), dim=1)


def undesired_contacts(
  env: ManagerBasedRlEnv, sensor_name: str, threshold: float = 1.0
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force_history is not None
  force = torch.linalg.vector_norm(sensor.data.force_history, dim=-1).amax(dim=2)
  return torch.sum((force > threshold).float(), dim=1)


def _foot_force(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force_history is not None
  return torch.linalg.vector_norm(sensor.data.force_history, dim=-1).mean(dim=2)


def _stable_stand_weight(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  torso_cfg: SceneEntityCfg,
  command_name: str,
  command_threshold: float,
  planar_speed_scale: float,
  torso_ang_vel_scale: float,
  contact_force_threshold: float,
) -> torch.Tensor:
  asset: Entity = env.scene[torso_cfg.name]
  command = _command(env, command_name)
  command_magnitude = torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(
    command[:, 2]
  )
  planar_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
  quat = asset.data.body_link_quat_w[:, torso_cfg.body_ids, :].squeeze(1)
  ang_vel_w = asset.data.body_link_ang_vel_w[:, torso_cfg.body_ids, :].squeeze(1)
  torso_ang_speed = torch.linalg.vector_norm(
    quat_apply_inverse(quat, ang_vel_w)[:, :2], dim=1
  )
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force_history is not None
  peak_force = torch.linalg.vector_norm(sensor.data.force_history, dim=-1).amax(dim=2)
  double_support = torch.all(peak_force > contact_force_threshold, dim=1)
  motion_weight = torch.exp(-torch.square(planar_speed / planar_speed_scale))
  motion_weight *= torch.exp(-torch.square(torso_ang_speed / torso_ang_vel_scale))
  return (
    (command_magnitude < command_threshold).float()
    * double_support.float()
    * motion_weight
  )


def stable_stand_hip_roll_norm(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  sensor_name: str,
  command_name: str = "twist",
  command_threshold: float = 0.1,
  planar_speed_scale: float = 0.2,
  torso_ang_vel_scale: float = 0.5,
  contact_force_threshold: float = 5.0,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  error = asset.data.joint_pos[:, asset_cfg.joint_ids] - default[:, asset_cfg.joint_ids]
  weight = _stable_stand_weight(
    env,
    sensor_name,
    torso_cfg,
    command_name,
    command_threshold,
    planar_speed_scale,
    torso_ang_vel_scale,
    contact_force_threshold,
  )
  return torch.linalg.vector_norm(error, dim=1) * weight


def stable_stand_feet_fore_aft_l1(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  sensor_name: str,
  position_deadband: float = 0.02,
  command_name: str = "twist",
  command_threshold: float = 0.1,
  planar_speed_scale: float = 0.2,
  torso_ang_vel_scale: float = 0.5,
  contact_force_threshold: float = 5.0,
) -> torch.Tensor:
  asset: Entity = env.scene[feet_cfg.name]
  feet = _feet_positions_b(asset, feet_cfg)
  error = torch.sum(torch.relu(torch.abs(feet[:, :, 0]) - position_deadband), dim=1)
  weight = _stable_stand_weight(
    env,
    sensor_name,
    torso_cfg,
    command_name,
    command_threshold,
    planar_speed_scale,
    torso_ang_vel_scale,
    contact_force_threshold,
  )
  return error * weight


def stable_stand_feet_lateral_center_l1(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  sensor_name: str,
  center_deadband: float = 0.01,
  command_name: str = "twist",
  command_threshold: float = 0.1,
  planar_speed_scale: float = 0.2,
  torso_ang_vel_scale: float = 0.5,
  contact_force_threshold: float = 5.0,
) -> torch.Tensor:
  asset: Entity = env.scene[feet_cfg.name]
  feet = _feet_positions_b(asset, feet_cfg)
  center = 0.5 * (feet[:, 0, 1] + feet[:, 1, 1])
  error = torch.relu(torch.abs(center) - center_deadband)
  weight = _stable_stand_weight(
    env,
    sensor_name,
    torso_cfg,
    command_name,
    command_threshold,
    planar_speed_scale,
    torso_ang_vel_scale,
    contact_force_threshold,
  )
  return error * weight


def feet_support_roll_l2(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  sensor_name: str,
  velocity_command_name: str = "twist",
  contact_force_threshold: float = 5.0,
) -> torch.Tensor:
  asset: Entity = env.scene[feet_cfg.name]
  quat = asset.data.body_link_quat_w[:, feet_cfg.body_ids, :]
  gravity = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, len(feet_cfg.body_ids), -1)
  roll_error = torch.square(quat_apply_inverse(quat, gravity)[:, :, 1])
  support = (_foot_force(env, sensor_name) > contact_force_threshold).float()
  return (
    torch.sum(roll_error * support, dim=1)
    * moving_mask(env, velocity_command_name).float()
  )


def ankle_roll_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  error = asset.data.joint_pos[:, asset_cfg.joint_ids] - default[:, asset_cfg.joint_ids]
  return torch.linalg.vector_norm(error, dim=1)


def ankle_roll_action_rate_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  sensor_name: str,
  contact_force_threshold: float = 20.0,
  velocity_command_name: str = "twist",
) -> torch.Tensor:
  delta = (
    env.action_manager.action[:, asset_cfg.joint_ids]
    - env.action_manager.prev_action[:, asset_cfg.joint_ids]
  )
  loaded = (_foot_force(env, sensor_name) > contact_force_threshold).float()
  return (
    torch.sum(torch.square(delta) * loaded, dim=1)
    * moving_mask(env, velocity_command_name).float()
  )
