"""Left-right policy symmetry in the target mjlab joint order."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from src.tasks.amp.constants import (
  ACTOR_FRAME_DIM,
  AMP_KEY_BODY_ORI_DIM,
  AMP_KEY_BODY_POS_DIM,
  AMP_OBS_DIM,
  CRITIC_FRAME_DIM,
  MJLAB_JOINT_NAMES,
)

_MIRROR_NAME = {
  name: name.replace("left_", "right_")
  if name.startswith("left_")
  else name.replace("right_", "left_")
  if name.startswith("right_")
  else name
  for name in MJLAB_JOINT_NAMES
}
_JOINT_MIRROR = tuple(
  MJLAB_JOINT_NAMES.index(_MIRROR_NAME[name]) for name in MJLAB_JOINT_NAMES
)
_JOINT_SIGN = tuple(
  -1.0 if any(axis in name for axis in ("roll", "yaw")) else 1.0
  for name in MJLAB_JOINT_NAMES
)
_POLAR_SIGN = (1.0, -1.0, 1.0)
_AXIAL_SIGN = (-1.0, 1.0, -1.0)
_COMMAND_SIGN = (1.0, -1.0, -1.0)


def _mirror_joints(value: torch.Tensor) -> torch.Tensor:
  index = torch.tensor(_JOINT_MIRROR, device=value.device)
  sign = value.new_tensor(_JOINT_SIGN)
  return value[..., index] * sign


def _mirror_actor(value: torch.Tensor) -> torch.Tensor:
  if value.shape[-1] % ACTOR_FRAME_DIM:
    raise ValueError(f"Actor width {value.shape[-1]} is not frame-aligned.")
  frames = value.reshape(*value.shape[:-1], -1, ACTOR_FRAME_DIM)
  out = frames.clone()
  out[..., 0:3] *= frames.new_tensor(_AXIAL_SIGN)
  out[..., 3:6] *= frames.new_tensor(_POLAR_SIGN)
  out[..., 6:9] *= frames.new_tensor(_COMMAND_SIGN)
  out[..., 9:30] = _mirror_joints(frames[..., 9:30])
  out[..., 30:51] = _mirror_joints(frames[..., 30:51])
  out[..., 51:72] = _mirror_joints(frames[..., 51:72])
  return out.reshape_as(value)


def _mirror_critic(value: torch.Tensor) -> torch.Tensor:
  if value.shape[-1] % CRITIC_FRAME_DIM:
    raise ValueError(f"Critic width {value.shape[-1]} is not frame-aligned.")
  frames = value.reshape(*value.shape[:-1], -1, CRITIC_FRAME_DIM)
  out = frames.clone()
  out[..., :ACTOR_FRAME_DIM] = _mirror_actor(
    frames[..., :ACTOR_FRAME_DIM]
  )
  out[..., 72:75] *= frames.new_tensor(_POLAR_SIGN)
  out[..., 75:77] = frames[..., (76, 75)]
  return out.reshape_as(value)


def _mirror_amp(value: torch.Tensor) -> torch.Tensor:
  if value.shape[-1] != AMP_OBS_DIM:
    raise ValueError(f"AMP width must be {AMP_OBS_DIM}, got {value.shape[-1]}.")
  out = value.clone()
  out[..., 0:3] *= value.new_tensor(_POLAR_SIGN)
  out[..., 3:6] *= value.new_tensor(_AXIAL_SIGN)
  out[..., 6:27] = _mirror_joints(value[..., 6:27])
  out[..., 27:48] = _mirror_joints(value[..., 27:48])
  key_pos_end = 48 + AMP_KEY_BODY_POS_DIM
  key_ori_end = key_pos_end + AMP_KEY_BODY_ORI_DIM
  key = value[..., 48:key_pos_end].reshape(*value.shape[:-1], 6, 3)
  key = key[..., (1, 0, 3, 2, 5, 4), :] * value.new_tensor(_POLAR_SIGN)
  out[..., 48:key_pos_end] = key.reshape(
    *value.shape[:-1], AMP_KEY_BODY_POS_DIM
  )
  key_ori = value[..., key_pos_end:key_ori_end].reshape(
    *value.shape[:-1], 6, 3, 2
  )
  key_ori = key_ori[..., (1, 0, 3, 2, 5, 4), :, :]
  # A sagittal reflection transforms a relative rotation as M R M, where
  # M=diag(1,-1,1). Only the first two columns are stored by the 6D encoding.
  ori_sign = value.new_tensor(((1.0, -1.0), (-1.0, 1.0), (1.0, -1.0)))
  key_ori = key_ori * ori_sign
  out[..., key_pos_end:key_ori_end] = key_ori.reshape(
    *value.shape[:-1], AMP_KEY_BODY_ORI_DIM
  )
  return out


def data_augmentation_func(
  env,
  obs: TensorDict | None = None,
  actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """Append mirrored samples using the RSL-RL 5.4 TensorDict interface."""
  del env
  obs_aug = None
  if obs is not None:
    mirrored = {}
    for key, value in obs.items():
      if key == "actor":
        mirrored[key] = _mirror_actor(value)
      elif key == "critic":
        mirrored[key] = _mirror_critic(value)
      elif key == "amp":
        mirrored[key] = _mirror_amp(value)
      else:
        mirrored[key] = value.clone()
    obs_aug = TensorDict(
      {key: torch.cat((obs[key], mirrored[key]), dim=0) for key in obs.keys()},
      batch_size=[2 * obs.batch_size[0]],
      device=obs.device,
    )

  actions_aug = None
  if actions is not None:
    actions_aug = torch.cat((actions, _mirror_joints(actions)), dim=0)
  return obs_aug, actions_aug
