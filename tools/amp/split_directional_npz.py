"""Split combined turn and lateral AMP recordings by direction."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[str, np.ndarray]:
  if not path.is_file():
    raise FileNotFoundError(path)
  with np.load(path, allow_pickle=False) as source:
    data = {name: np.asarray(source[name]) for name in source.files}
  frame_count = len(data["joint_pos"])
  for name, value in data.items():
    if value.ndim > 0 and value.shape[0] == frame_count:
      if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
        raise ValueError(f"{path} contains non-finite values in {name!r}")
  return data


def _slice_frames(
  data: dict[str, np.ndarray], start: int, end: int
) -> dict[str, np.ndarray]:
  frame_count = len(data["joint_pos"])
  if not 0 <= start < end <= frame_count:
    raise ValueError(f"Invalid frame range [{start}, {end}) for {frame_count} frames")
  return {
    name: value[start:end]
    if value.ndim > 0 and value.shape[0] == frame_count
    else value
    for name, value in data.items()
  }


def _atomic_save(path: Path, data: dict[str, np.ndarray]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(path.name + ".tmp.npz")
  np.savez_compressed(temporary, **data)
  with np.load(temporary, allow_pickle=False) as check:
    if len(check["joint_pos"]) != len(data["joint_pos"]):
      raise RuntimeError(f"Failed to validate temporary output {temporary}")
  os.replace(temporary, path)


def main() -> None:
  root = Path("dataset/e1_21dof/amp").resolve()
  turn_path = root / "turn_rl.npz"
  side_path = root / "side_trim.npz"
  turn = _load(turn_path)
  side = _load(side_path)

  if len(turn["joint_pos"]) != 2211:
    raise ValueError("turn_rl.npz no longer has the expected 2211 frames")
  if len(side["joint_pos"]) != 3965:
    raise ValueError("side_trim.npz no longer has the expected 3965 frames")

  outputs = {
    root / "turn_r.npz": _slice_frames(turn, 0, 1041),
    root / "turn_l.npz": _slice_frames(turn, 1041, 2211),
    root / "side_l.npz": _slice_frames(side, 0, 2045),
    root / "side_r.npz": _slice_frames(side, 2208, 3965),
  }
  for path, data in outputs.items():
    _atomic_save(path, data)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    print(
      f"[INFO] Wrote {path.name}: {len(data['joint_pos'])} frames, "
      f"{len(data['joint_pos']) / fps:.2f}s at {fps:.3f} Hz"
    )
  print("[INFO] Discarded side_trim.npz transition frames [2045, 2208).")


if __name__ == "__main__":
  main()
