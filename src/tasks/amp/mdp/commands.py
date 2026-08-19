"""Velocity commands for the standalone AMP locomotion task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, yaw_quat
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


COMMAND_MODE_STANDING = 0
COMMAND_MODE_TURNING = 1
COMMAND_MODE_LATERAL = 2
COMMAND_MODE_MIXED = 3


class UniformVelocityCommand(CommandTerm):
  """Sample body-frame planar velocity and yaw-rate commands."""

  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.is_standing_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.command_mode = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["standing_fraction"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["turning_fraction"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["lateral_fraction"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["mixed_fraction"] = torch.zeros(
      self.num_envs, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_steps = self.cfg.resampling_time_range[1] / self._env.step_dt
    episode_steps = self._env.max_episode_length
    self.metrics["error_vel_xy"] += torch.linalg.vector_norm(
      self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=1
    ) / max_steps
    self.metrics["error_vel_yaw"] += torch.abs(
      self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]
    ) / max_steps
    self.metrics["standing_fraction"] += (
      self.command_mode == COMMAND_MODE_STANDING
    ).float() / episode_steps
    self.metrics["turning_fraction"] += (
      self.command_mode == COMMAND_MODE_TURNING
    ).float() / episode_steps
    self.metrics["lateral_fraction"] += (
      self.command_mode == COMMAND_MODE_LATERAL
    ).float() / episode_steps
    self.metrics["mixed_fraction"] += (
      self.command_mode == COMMAND_MODE_MIXED
    ).float() / episode_steps

  def _sample_outside_deadband(
    self,
    count: int,
    value_range: tuple[float, float],
    deadband: float,
  ) -> torch.Tensor:
    """Sample a signed command while avoiding ineffective near-zero values."""
    low, high = value_range
    if low >= -deadband or high <= deadband:
      return torch.empty(count, device=self.device).uniform_(low, high)
    negative_span = -deadband - low
    positive_span = high - deadband
    positive_probability = positive_span / (negative_span + positive_span)
    positive = torch.rand(count, device=self.device) < positive_probability
    magnitude = torch.rand(count, device=self.device)
    negative_value = low + magnitude * (-deadband - low)
    positive_value = deadband + magnitude * (high - deadband)
    return torch.where(positive, positive_value, negative_value)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    # Advanced indexing returns a copy, so ``tensor[env_ids, i].uniform_()``
    # would leave the command buffer unchanged. Sample temporary tensors and
    # assign them back explicitly.
    self.vel_command_b[env_ids, 0] = torch.empty(
      count, device=self.device
    ).uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = torch.empty(
      count, device=self.device
    ).uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = torch.empty(
      count, device=self.device
    ).uniform_(*self.cfg.ranges.ang_vel_z)
    draws = torch.rand(count, device=self.device)
    standing_end = self.cfg.rel_standing_envs
    turning_end = standing_end + self.cfg.rel_turning_envs
    lateral_end = turning_end + self.cfg.rel_lateral_envs
    standing = draws < standing_end
    turning = (draws >= standing_end) & (draws < turning_end)
    lateral = (draws >= turning_end) & (draws < lateral_end)
    mixed = draws >= lateral_end

    self.is_standing_env[env_ids] = standing
    self.command_mode[env_ids] = COMMAND_MODE_MIXED
    self.command_mode[env_ids[standing]] = COMMAND_MODE_STANDING
    self.command_mode[env_ids[turning]] = COMMAND_MODE_TURNING
    self.command_mode[env_ids[lateral]] = COMMAND_MODE_LATERAL
    self.command_mode[env_ids[mixed]] = COMMAND_MODE_MIXED

    self.vel_command_b[env_ids[standing]] = 0.0
    turning_ids = env_ids[turning]
    self.vel_command_b[turning_ids, :2] = 0.0
    self.vel_command_b[turning_ids, 2] = self._sample_outside_deadband(
      len(turning_ids),
      self.cfg.ranges.pure_turn_ang_vel_z,
      self.cfg.turning_deadband,
    )
    lateral_ids = env_ids[lateral]
    self.vel_command_b[lateral_ids, 0] = 0.0
    self.vel_command_b[lateral_ids, 2] = 0.0
    self.vel_command_b[lateral_ids, 1] = self._sample_outside_deadband(
      len(lateral_ids),
      self.cfg.ranges.pure_lateral_lin_vel_y,
      self.cfg.lateral_deadband,
    )

  def _update_command(self) -> None:
    self.vel_command_b[self.is_standing_env] = 0.0

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw the commanded planar velocity as a green world-frame arrow."""
    for env_idx in visualizer.get_env_indices(self.num_envs):
      start = self.robot.data.root_link_pos_w[env_idx].detach().cpu().numpy().copy()
      start[2] += 0.15
      command_b = torch.zeros(3, device=self.device)
      command_b[:2] = self.vel_command_b[env_idx, :2]
      command_w = quat_apply(
        yaw_quat(self.robot.data.root_link_quat_w[env_idx]), command_b
      )
      end = start + command_w.detach().cpu().numpy()
      visualizer.add_arrow(
        start=np.asarray(start),
        end=np.asarray(end),
        color=(0.1, 1.0, 0.1, 1.0),
        width=0.02,
        label="command_velocity",
      )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  rel_standing_envs: float = 0.0
  rel_turning_envs: float = 0.0
  rel_lateral_envs: float = 0.0
  turning_deadband: float = 0.2
  lateral_deadband: float = 0.1

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    pure_turn_ang_vel_z: tuple[float, float]
    pure_lateral_lin_vel_y: tuple[float, float]

  ranges: Ranges

  def __post_init__(self) -> None:
    probabilities = (
      self.rel_standing_envs,
      self.rel_turning_envs,
      self.rel_lateral_envs,
    )
    if any(value < 0.0 or value > 1.0 for value in probabilities):
      raise ValueError("Command-mode probabilities must lie in [0, 1].")
    if sum(probabilities) > 1.0:
      raise ValueError("Command-mode probabilities must sum to at most 1.")
    if self.turning_deadband < 0.0 or self.lateral_deadband < 0.0:
      raise ValueError("Command deadbands must be non-negative.")

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)
