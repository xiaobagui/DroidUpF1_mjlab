"""Curriculum terms for E1 AMP locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from .commands import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class VelocityStage(TypedDict, total=False):
  step: int
  lin_vel_x: tuple[float, float]
  lin_vel_y: tuple[float, float]
  ang_vel_z: tuple[float, float]
  pure_turn_ang_vel_z: tuple[float, float]
  pure_lateral_lin_vel_y: tuple[float, float]
  turning_deadband: float
  lateral_deadband: float


def command_velocity_stages(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: tuple[VelocityStage, ...],
) -> dict[str, torch.Tensor]:
  """Expand command ranges at fixed environment-step thresholds."""
  del env_ids
  term = env.command_manager.get_term(command_name)
  cfg = cast(UniformVelocityCommandCfg, term.cfg)
  for stage in stages:
    if env.common_step_counter < stage["step"]:
      break
    if "lin_vel_x" in stage:
      cfg.ranges.lin_vel_x = stage["lin_vel_x"]
    if "lin_vel_y" in stage:
      cfg.ranges.lin_vel_y = stage["lin_vel_y"]
    if "ang_vel_z" in stage:
      cfg.ranges.ang_vel_z = stage["ang_vel_z"]
    if "pure_turn_ang_vel_z" in stage:
      cfg.ranges.pure_turn_ang_vel_z = stage["pure_turn_ang_vel_z"]
    if "pure_lateral_lin_vel_y" in stage:
      cfg.ranges.pure_lateral_lin_vel_y = stage["pure_lateral_lin_vel_y"]
    if "turning_deadband" in stage:
      cfg.turning_deadband = stage["turning_deadband"]
    if "lateral_deadband" in stage:
      cfg.lateral_deadband = stage["lateral_deadband"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
    "pure_turn_ang_vel_z_min": torch.tensor(
      cfg.ranges.pure_turn_ang_vel_z[0]
    ),
    "pure_turn_ang_vel_z_max": torch.tensor(
      cfg.ranges.pure_turn_ang_vel_z[1]
    ),
    "pure_lateral_lin_vel_y_min": torch.tensor(
      cfg.ranges.pure_lateral_lin_vel_y[0]
    ),
    "pure_lateral_lin_vel_y_max": torch.tensor(
      cfg.ranges.pure_lateral_lin_vel_y[1]
    ),
    "turning_deadband": torch.tensor(cfg.turning_deadband),
    "lateral_deadband": torch.tensor(cfg.lateral_deadband),
  }
