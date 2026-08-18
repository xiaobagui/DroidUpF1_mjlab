"""Load XML-order E1 locomotion NPZ files as AMP discriminator states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_inv, quat_mul

from src.tasks.amp.constants import (
  AMP_KEY_BODY_NAMES,
  AMP_OBS_DIM,
  MJLAB_JOINT_NAMES,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _quat_inverse(quaternion: torch.Tensor) -> torch.Tensor:
  return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def _quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
  scalar, xyz = quaternion[..., :1], quaternion[..., 1:]
  cross = 2.0 * torch.linalg.cross(xyz, vector)
  return vector + scalar * cross + torch.linalg.cross(xyz, cross)


class MotionLoader:
  def __init__(
    self,
    motion_files: tuple[str, ...],
    device: str,
    transition_dt: float,
    preload_transitions: int,
    velocity_threshold: float,
    slow_weights: tuple[float, ...],
    fast_weights: tuple[float, ...],
  ):
    self.device = device
    self.transition_dt = transition_dt
    paths = self._resolve_files(motion_files)
    self.motion_names = tuple(path.stem for path in paths)
    self.motion_labels = torch.tensor(
      [0.0 if name == "walk" else 1.0 for name in self.motion_names],
      device=device,
      dtype=torch.float32,
    )
    self.trajectories: list[torch.Tensor] = []
    self.frame_dt: list[float] = []
    self.velocity_threshold = float(velocity_threshold)
    if not np.isfinite(self.velocity_threshold):
      raise ValueError("AMP motion velocity threshold must be finite.")
    self.slow_weights = self._validate_weights(
      slow_weights, len(paths), "slow"
    )
    self.fast_weights = self._validate_weights(
      fast_weights, len(paths), "fast"
    )
    self.last_fast_fraction = 0.0
    self.last_motion_fractions = torch.zeros(len(paths), device=device)

    print("\n========= AMP locomotion motion files ========")
    for path in paths:
      trajectory, frame_dt = self._load(path)
      self.trajectories.append(trajectory)
      self.frame_dt.append(frame_dt)
      print(
        f"{path.name}: {len(trajectory)} frames, "
        f"{1.0 / frame_dt:.1f} fps"
      )
    print("=============================================\n")

    self.preloaded_state: list[torch.Tensor] | None = None
    self.preloaded_next_state: list[torch.Tensor] | None = None
    if preload_transitions > 0:
      print(f"Preloading {preload_transitions} AMP expert transitions...")
      self.preloaded_state = []
      self.preloaded_next_state = []
      per_motion, remainder = divmod(preload_transitions, len(self.trajectories))
      for trajectory_id, name in enumerate(self.motion_names):
        count = per_motion + int(trajectory_id < remainder)
        state, next_state = self._sample_trajectory(trajectory_id, count)
        self.preloaded_state.append(state)
        self.preloaded_next_state.append(next_state)
        print(f"  {name}: {count} transitions")

  def _validate_weights(
    self,
    values: tuple[float, ...],
    motion_count: int,
    label: str,
  ) -> torch.Tensor:
    weights = torch.tensor(values, device=self.device, dtype=torch.float32)
    if weights.shape != (motion_count,):
      raise ValueError(
        f"AMP {label} weights contain {len(weights)} values; "
        f"expected {motion_count}."
      )
    if not torch.isfinite(weights).all() or torch.any(weights < 0.0):
      raise ValueError(f"AMP {label} weights must be finite and non-negative.")
    if weights.sum() <= 0.0:
      raise ValueError(f"AMP {label} weights need a positive sum.")
    return weights / weights.sum()

  @staticmethod
  def _resolve_files(entries):
    paths: list[Path] = []
    for entry in entries:
      path = Path(entry).expanduser()
      if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
      matches = sorted(path.glob("*.npz")) if path.is_dir() else [path]
      for match in matches:
        if not match.exists():
          raise FileNotFoundError(f"AMP motion file not found: {match}")
        paths.append(match)
    if not paths:
      raise ValueError("No AMP motion files were configured.")
    return paths

  def _load(self, path: Path) -> tuple[torch.Tensor, float]:
    with np.load(path, allow_pickle=False) as data:
      required = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "key_body_pos",
        "key_body_quat_w",
        "fps",
      )
      missing = [name for name in required if name not in data.files]
      if missing:
        raise ValueError(f"{path} is missing AMP fields: {missing}")
      for name in required[:-1]:
        if not np.isfinite(data[name]).all():
          raise ValueError(f"{path} contains non-finite values in '{name}'.")
      if data["joint_pos"].shape[1] != len(MJLAB_JOINT_NAMES):
        raise ValueError(f"{path} does not contain 21-DOF E1 joint data.")
      if data["key_body_pos"].shape[1] != 3 * len(AMP_KEY_BODY_NAMES):
        raise ValueError(f"{path} does not contain six AMP key bodies.")
      if data["key_body_quat_w"].shape[1:] != (
        len(AMP_KEY_BODY_NAMES),
        4,
      ):
        raise ValueError(f"{path} has invalid AMP key-body orientations.")
      if "joint_names" in data.files:
        stored_joint_names = data["joint_names"].astype(str).tolist()
        if stored_joint_names != list(MJLAB_JOINT_NAMES):
          raise ValueError(
            f"{path} joint order does not match E1_21dof.xml:\n"
            f"NPZ: {stored_joint_names}\nXML: {list(MJLAB_JOINT_NAMES)}"
          )
      if "key_body_names" in data.files:
        stored_key_bodies = data["key_body_names"].astype(str).tolist()
        if stored_key_bodies != list(AMP_KEY_BODY_NAMES):
          raise ValueError(f"{path} has an unexpected AMP key-body order.")

      joint_pos = torch.as_tensor(
        data["joint_pos"], device=self.device, dtype=torch.float32
      )
      joint_vel = torch.as_tensor(
        data["joint_vel"], device=self.device, dtype=torch.float32
      )
      root_pos = torch.as_tensor(
        data["body_pos_w"][:, 0], device=self.device, dtype=torch.float32
      )
      root_quat = torch.as_tensor(
        data["body_quat_w"][:, 0], device=self.device, dtype=torch.float32
      )
      lin_vel_w = torch.as_tensor(
        data["body_lin_vel_w"][:, 0], device=self.device, dtype=torch.float32
      )
      ang_vel_w = torch.as_tensor(
        data["body_ang_vel_w"][:, 0], device=self.device, dtype=torch.float32
      )
      key_pos_w = torch.as_tensor(
        data["key_body_pos"], device=self.device, dtype=torch.float32
      ).reshape(-1, len(AMP_KEY_BODY_NAMES), 3)
      key_quat_w = torch.as_tensor(
        data["key_body_quat_w"], device=self.device, dtype=torch.float32
      )
      fps = float(np.asarray(data["fps"]).reshape(-1)[0])

    if len(joint_pos) < 2:
      raise ValueError(f"{path} must contain at least two motion frames.")
    if not np.isfinite(fps) or fps <= 0.0:
      raise ValueError(f"{path} has an invalid fps value: {fps}.")
    max_lin_vel = torch.linalg.vector_norm(lin_vel_w, dim=1).max().item()
    max_ang_vel = torch.linalg.vector_norm(ang_vel_w, dim=1).max().item()
    if max_lin_vel > 10.0 or max_ang_vel > 20.0:
      raise ValueError(
        f"{path} has implausible root velocity peaks "
        f"({max_lin_vel:.2f} m/s, {max_ang_vel:.2f} rad/s)."
      )

    inverse = _quat_inverse(root_quat)
    lin_vel_b = _quat_apply(inverse, lin_vel_w)
    ang_vel_b = _quat_apply(inverse, ang_vel_w)
    relative = key_pos_w - root_pos.unsqueeze(1)
    inverse_repeated = inverse.unsqueeze(1).expand(-1, len(AMP_KEY_BODY_NAMES), -1)
    key_pos_b = _quat_apply(inverse_repeated, relative).flatten(start_dim=1)
    root_quat_inverse = quat_inv(root_quat).unsqueeze(1).expand_as(key_quat_w)
    key_quat_b = quat_mul(root_quat_inverse, key_quat_w)
    key_ori_b = matrix_from_quat(key_quat_b)[..., :2].flatten(start_dim=1)
    trajectory = torch.cat(
      (lin_vel_b, ang_vel_b, joint_pos, joint_vel, key_pos_b, key_ori_b),
      dim=1,
    )
    if trajectory.shape[1] != AMP_OBS_DIM:
      raise RuntimeError(
        f"Expected {AMP_OBS_DIM} AMP values, got {trajectory.shape[1]}."
      )
    return trajectory, 1.0 / fps

  def _sample_trajectory(
    self, trajectory_id: int, count: int
  ) -> tuple[torch.Tensor, torch.Tensor]:
    trajectory = self.trajectories[trajectory_id]
    expert_dt = self.frame_dt[trajectory_id]
    duration = (len(trajectory) - 1) * expert_dt
    available = max(duration - self.transition_dt, 0.0)
    times = torch.rand(count, device=self.device) * available
    state = self._interpolate(trajectory, times, expert_dt)
    next_state = self._interpolate(
      trajectory, times + self.transition_dt, expert_dt
    )
    return state, next_state

  def _draw_trajectory_ids(self, motion_label: torch.Tensor) -> torch.Tensor:
    motion_label = motion_label.to(device=self.device, dtype=torch.float32).flatten()
    fast = motion_label > 0.5
    weights = torch.where(
      fast.unsqueeze(1),
      self.fast_weights.unsqueeze(0),
      self.slow_weights.unsqueeze(0),
    )
    trajectory_ids = torch.multinomial(weights, num_samples=1).squeeze(1)
    self.last_fast_fraction = fast.float().mean().item()
    self.last_motion_fractions = torch.bincount(
      trajectory_ids, minlength=len(self.trajectories)
    ).float() / max(len(trajectory_ids), 1)
    return trajectory_ids

  def _sample(
    self, trajectory_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    count = len(trajectory_ids)
    state = torch.empty(count, AMP_OBS_DIM, device=self.device)
    next_state = torch.empty_like(state)
    for trajectory_id in torch.unique(trajectory_ids).tolist():
      mask = trajectory_ids == trajectory_id
      sample_count = int(mask.sum().item())
      sampled_state, sampled_next = self._sample_trajectory(
        trajectory_id, sample_count
      )
      state[mask] = sampled_state
      next_state[mask] = sampled_next
    return state, next_state

  @staticmethod
  def _interpolate(
    trajectory: torch.Tensor, times: torch.Tensor, frame_dt: float
  ) -> torch.Tensor:
    """Linearly sample a feature trajectory at arbitrary times."""
    frame_position = torch.clamp(
      times / frame_dt, min=0.0, max=float(len(trajectory) - 1)
    )
    low = torch.floor(frame_position).long()
    high = torch.clamp(low + 1, max=len(trajectory) - 1)
    blend = (frame_position - low).unsqueeze(1)
    return torch.lerp(trajectory[low], trajectory[high], blend)

  def sample(
    self, motion_label: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    trajectory_ids = self._draw_trajectory_ids(motion_label)
    if self.preloaded_state is None or self.preloaded_next_state is None:
      state, next_state = self._sample(trajectory_ids)
    else:
      count = len(trajectory_ids)
      state = torch.empty(count, AMP_OBS_DIM, device=self.device)
      next_state = torch.empty_like(state)
      for trajectory_id in torch.unique(trajectory_ids).tolist():
        mask = trajectory_ids == trajectory_id
        sample_count = int(mask.sum().item())
        source = self.preloaded_state[trajectory_id]
        source_next = self.preloaded_next_state[trajectory_id]
        index = torch.randint(len(source), (sample_count,), device=self.device)
        state[mask] = source[index]
        next_state[mask] = source_next[index]
    labels = self.motion_labels[trajectory_ids]
    return state, next_state, labels
