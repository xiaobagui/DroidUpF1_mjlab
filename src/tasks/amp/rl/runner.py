"""Checkpoint export support for AMP locomotion policies."""

from __future__ import annotations

import numpy as np
import torch

from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner

from src.tasks.amp.constants import ACTOR_FRAME_DIM


class AmpOnPolicyRunner(MjlabOnPolicyRunner):
  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    load_iteration = load_cfg is None or load_cfg.get("iteration", False)
    if load_iteration:
      unwrapped = self.env.unwrapped
      unwrapped.common_step_counter = (
        self.current_learning_iteration * self.cfg["num_steps_per_env"]
      )
      unwrapped.curriculum_manager.compute()
      env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device)
      unwrapped.command_manager.reset(env_ids)
    return infos

  def save(self, path: str, infos=None) -> None:
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      metadata = get_base_metadata(self.env.unwrapped, "local")
      metadata.update(
        {
          "task_type": "velocity_commanded_locomotion_amp",
          "actor_history_length": 5,
          "actor_frame_dim": ACTOR_FRAME_DIM,
        }
      )
      attach_metadata_to_onnx(str(onnx_path), metadata)
    except Exception as error:
      print(f"[WARN] AMP ONNX export failed (training continues): {error}")
