"""Concatenate right- and left-turn AMP NPZ recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np


FRAME_FIELDS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
  "key_body_pos",
  "key_body_quat_w",
)
META_FIELDS = ("fps", "joint_names", "key_body_names")


def _load(path: Path) -> dict[str, np.ndarray]:
  with np.load(path, allow_pickle=False) as source:
    required = (*FRAME_FIELDS, *META_FIELDS)
    missing = [name for name in required if name not in source.files]
    if missing:
      raise ValueError(f"{path} is missing fields: {missing}")
    return {name: np.asarray(source[name]) for name in source.files}


def _check_compatible(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> None:
  for name in META_FIELDS:
    if name == "fps":
      if not np.isclose(float(first[name]), float(second[name]), rtol=1e-5):
        raise ValueError(f"FPS mismatch: {first[name]} vs {second[name]}")
    elif not np.array_equal(first[name], second[name]):
      raise ValueError(f"Metadata mismatch in {name}")
  for name in FRAME_FIELDS:
    if first[name].ndim != second[name].ndim or first[name].shape[1:] != second[name].shape[1:]:
      raise ValueError(f"Frame field shape mismatch in {name}")


def concatenate_turns(right_path: Path, left_path: Path, output_path: Path) -> None:
  right = _load(right_path)
  left = _load(left_path)
  _check_compatible(right, left)

  # The first two frames in both exports contain finite-difference boundary
  # velocities from the source replay and are not physical motion frames.
  right_start = 2
  left_start = 2
  output = {
    name: np.concatenate((right[name][right_start:], left[name][left_start:]), axis=0)
    for name in FRAME_FIELDS
  }
  output.update({name: right[name] for name in META_FIELDS})
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output_path, **output)
  fps = float(np.asarray(output["fps"]).reshape(-1)[0])
  print(
    f"[INFO] {right_path.name}: {len(right['joint_pos'])} -> "
    f"{len(right['joint_pos']) - right_start} frames"
  )
  print(
    f"[INFO] {left_path.name}: {len(left['joint_pos'])} -> "
    f"{len(left['joint_pos']) - left_start} frames"
  )
  print(
    f"[INFO] Wrote {output_path} with {len(output['joint_pos'])} frames "
    f"({len(output['joint_pos']) / fps:.2f}s at {fps:.3f} Hz)"
  )


if __name__ == "__main__":
  root = Path("dataset/e1_21dof/amp")
  concatenate_turns(root / "turn_r.npz", root / "turn_l.npz", root / "turn_rl.npz")
