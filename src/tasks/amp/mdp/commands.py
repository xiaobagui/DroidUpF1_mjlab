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
    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_steps = self.cfg.resampling_time_range[1] / self._env.step_dt
    self.metrics["error_vel_xy"] += torch.linalg.vector_norm(
      self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=1
    ) / max_steps
    self.metrics["error_vel_yaw"] += torch.abs(
      self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]
    ) / max_steps

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
    self.is_standing_env[env_ids] = draws < self.cfg.rel_standing_envs
    self.vel_command_b[env_ids[self.is_standing_env[env_ids]]] = 0.0

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

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]

  ranges: Ranges

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)
