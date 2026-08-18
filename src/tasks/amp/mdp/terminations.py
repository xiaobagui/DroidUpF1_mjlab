"""Termination conditions for E1 flat-ground locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(
  env: ManagerBasedRlEnv, sensor_name: str, threshold: float = 1.0
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force_history is not None
  force = torch.linalg.vector_norm(sensor.data.force_history, dim=-1)
  return torch.any(force > threshold, dim=(1, 2))

