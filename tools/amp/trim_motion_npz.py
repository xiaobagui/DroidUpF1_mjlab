"""Trim inactive lead-in and tail frames from AMP motion NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _quat_apply_inverse(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
  q_vec = quat[:, 1:]
  q_w = quat[:, :1]
  cross = np.cross(q_vec, vector)
  return vector - 2.0 * q_w * cross + 2.0 * np.cross(q_vec, cross)


def _active_mask(data: np.lib.npyio.NpzFile, mode: str) -> np.ndarray:
  root_quat = data["body_quat_w"][:, 0]
  lin_vel_b = _quat_apply_inverse(root_quat, data["body_lin_vel_w"][:, 0])
  ang_vel_b = _quat_apply_inverse(root_quat, data["body_ang_vel_w"][:, 0])
  if mode == "turn":
    return (np.abs(ang_vel_b[:, 2]) > 0.2) & (
      np.linalg.norm(lin_vel_b[:, :2], axis=1) < 0.25
    )
  if mode == "side":
    return (
      (np.abs(lin_vel_b[:, 1]) > 0.1)
      & (np.abs(lin_vel_b[:, 0]) < 0.2)
      & (np.abs(ang_vel_b[:, 2]) < 0.4)
    )
  raise ValueError(f"Unsupported motion mode: {mode}")


def _trim_bounds(active: np.ndarray, window: int = 50, ratio: float = 0.4):
  if len(active) < window:
    raise ValueError("Motion is shorter than the activity window.")
  kernel = np.ones(window, dtype=np.float32) / window
  activity = np.convolve(active.astype(np.float32), kernel, mode="same")
  active_window = activity >= ratio
  indices = np.flatnonzero(active_window)
  if len(indices) == 0:
    raise ValueError("No sustained active motion segment was found.")
  start = int(indices[0])
  end = int(indices[-1]) + 1
  if end - start < window:
    raise ValueError("The sustained active motion segment is too short.")
  return start, end


def trim_motion(input_path: Path, output_path: Path, mode: str) -> None:
  with np.load(input_path, allow_pickle=False) as data:
    active = _active_mask(data, mode)
    start, end = _trim_bounds(active)
    trimmed = {
      key: (value[start:end] if value.ndim > 0 and value.shape[0] == len(active) else value)
      for key, value in ((key, data[key]) for key in data.files)
    }
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output_path, **trimmed)
  fps = float(np.asarray(trimmed["fps"]).reshape(-1)[0])
  print(
    f"{input_path.name}: frames {len(active)} -> {end - start}, "
    f"trim [{start}, {end}), duration {(end - start) / fps:.2f}s"
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/e1_21dof/amp"))
  args = parser.parse_args()
  trim_motion(args.dataset_dir / "turn.npz", args.dataset_dir / "turn_trim.npz", "turn")
  trim_motion(args.dataset_dir / "side.npz", args.dataset_dir / "side_trim.npz", "side")


if __name__ == "__main__":
  main()
