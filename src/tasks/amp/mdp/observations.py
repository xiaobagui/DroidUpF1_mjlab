"""Policy, critic, and discriminator observations for E1 locomotion AMP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from src.tasks.amp.constants import ACTOR_FRAME_DIM, AMP_OBS_DIM, CRITIC_FRAME_DIM

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _ordered_joint_state(
  asset: Entity, asset_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  default = asset.data.default_joint_pos
  assert default is not None
  ids = asset_cfg.joint_ids
  return (
    asset.data.joint_pos[:, ids],
    asset.data.joint_vel[:, ids],
    default[:, ids],
  )


def _torso_imu(
  asset: Entity, torso_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  quat_w = asset.data.body_link_quat_w[:, torso_cfg.body_ids, :].squeeze(1)
  ang_vel_w = asset.data.body_link_ang_vel_w[:, torso_cfg.body_ids, :].squeeze(1)
  ang_vel_b = quat_apply_inverse(quat_w, ang_vel_w)
  gravity_b = quat_apply_inverse(quat_w, asset.data.gravity_vec_w)
  return ang_vel_b, gravity_b


def _actor_frame(
  env: ManagerBasedRlEnv,
  joint_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  velocity_command_name: str,
  corrupt: bool,
) -> torch.Tensor:
  asset: Entity = env.scene[joint_cfg.name]
  ang_vel_b, gravity_b = _torso_imu(asset, torso_cfg)
  command = env.command_manager.get_command(velocity_command_name)
  assert command is not None
  joint_pos, joint_vel, default = _ordered_joint_state(asset, joint_cfg)
  action = env.action_manager.action

  frame = torch.cat(
    (
      0.2 * ang_vel_b,
      gravity_b,
      command,
      joint_pos - default,
      0.05 * joint_vel,
      action,
    ),
    dim=1,
  )
  assert frame.shape[1] == ACTOR_FRAME_DIM

  if corrupt:
    # Match the original Isaac Lab task's per-field uniform noise.  Keeping the
    # whole frame in one observation term preserves frame-major history order.
    scale = torch.zeros(ACTOR_FRAME_DIM, device=env.device)
    scale[0:3] = 0.3 * 0.2
    scale[3:6] = 0.05
    scale[9:30] = 0.02
    scale[30:51] = 1.5 * 0.05
    frame = frame + (2.0 * torch.rand_like(frame) - 1.0) * scale
  return frame


def actor_frame(
  env: ManagerBasedRlEnv,
  joint_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  velocity_command_name: str = "twist",
) -> torch.Tensor:
  corrupt = env.cfg.observations["actor"].enable_corruption
  return _actor_frame(
    env,
    joint_cfg,
    torso_cfg,
    velocity_command_name,
    corrupt=corrupt,
  )


def critic_frame(
  env: ManagerBasedRlEnv,
  joint_cfg: SceneEntityCfg,
  torso_cfg: SceneEntityCfg,
  feet_contact_sensor_name: str,
  velocity_command_name: str = "twist",
) -> torch.Tensor:
  asset: Entity = env.scene[joint_cfg.name]
  actor = _actor_frame(
    env,
    joint_cfg,
    torso_cfg,
    velocity_command_name,
    corrupt=False,
  )
  sensor: ContactSensor = env.scene[feet_contact_sensor_name]
  assert sensor.data.found is not None
  found = sensor.data.found
  contact = (found > 0).float()
  if contact.ndim > 2:
    contact = contact.flatten(start_dim=2).any(dim=2).float()
  frame = torch.cat((actor, asset.data.root_link_lin_vel_b, contact), dim=1)
  assert frame.shape[1] == CRITIC_FRAME_DIM
  return frame


def amp_state(
  env: ManagerBasedRlEnv,
  joint_cfg: SceneEntityCfg,
  key_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return the discriminator state used by the expert NPZ loader."""
  asset: Entity = env.scene[joint_cfg.name]
  joint_pos, joint_vel, _ = _ordered_joint_state(asset, joint_cfg)
  root_pos = asset.data.root_link_pos_w
  root_quat = asset.data.root_link_quat_w
  key_pos = asset.data.body_link_pos_w[:, key_body_cfg.body_ids, :]
  relative = key_pos - root_pos.unsqueeze(1)
  root_quat_repeated = root_quat.unsqueeze(1).expand(-1, relative.shape[1], -1)
  key_pos_b = quat_apply_inverse(root_quat_repeated, relative)
  key_quat_w = asset.data.body_link_quat_w[:, key_body_cfg.body_ids, :]
  _, key_quat_b = subtract_frame_transforms(
    root_pos.unsqueeze(1).expand_as(key_pos),
    root_quat_repeated,
    key_pos,
    key_quat_w,
  )
  key_ori_b = matrix_from_quat(key_quat_b)[..., :2]
  state = torch.cat(
    (
      asset.data.root_link_lin_vel_b,
      asset.data.root_link_ang_vel_b,
      joint_pos,
      joint_vel,
      key_pos_b.flatten(start_dim=1),
      key_ori_b.flatten(start_dim=1),
    ),
    dim=1,
  )
  assert state.shape[1] == AMP_OBS_DIM
  return state
