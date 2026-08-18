"""Convert an E1 21-DOF motion PKL into the XML-order AMP NPZ format.

The input ``dof_pos`` is expected to use the exact non-free joint order in
``E1_21dof.xml``.  The output preserves that order without remapping and stores
the joint names so the training loader can validate the contract.

Example:
  python tools/amp/pkl_to_npz.py \
    --input dataset/e1_21dof/mimic/dance_pkl/MJ_dance.pkl \
    --output dataset/e1_21dof/amp/MJ_dance.npz
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from src.tasks.amp.constants import (
  AMP_KEY_BODY_NAMES,
  MJLAB_JOINT_NAMES,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "src/assets/e1_21dof/mjcf/E1_21dof.xml"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Convert an E1 21-DOF PKL to an AMP expert-motion NPZ."
  )
  parser.add_argument("--input", "-i", type=Path, required=True)
  parser.add_argument(
    "--output",
    "-o",
    type=Path,
    default=None,
    help="Output file or directory. Defaults to beside each input PKL.",
  )
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument("--fps", type=float, default=None, help="Override PKL FPS.")
  return parser.parse_args()


def _array(
  source: dict[str, Any], key: str, shape_tail: tuple[int, ...]
) -> np.ndarray:
  value = np.asarray(source[key], dtype=np.float32)
  if value.ndim != len(shape_tail) + 1 or value.shape[1:] != shape_tail:
    raise ValueError(f"{key} has shape {value.shape}, expected (T, {shape_tail})")
  if not np.isfinite(value).all():
    raise ValueError(f"{key} contains NaN or infinity")
  return value


def _differentiate(value: np.ndarray, fps: float) -> np.ndarray:
  if len(value) < 2:
    raise ValueError("At least two motion frames are required")
  velocity = np.zeros_like(value, dtype=np.float32)
  velocity[1:] = (value[1:] - value[:-1]) * fps
  velocity[0] = velocity[1]
  return velocity


def _normalize_quaternions_xyzw(value: np.ndarray) -> np.ndarray:
  norm = np.linalg.norm(value, axis=1, keepdims=True)
  if np.any(norm < 1.0e-8):
    raise ValueError("root_rot contains a zero-length quaternion")
  value = value / norm
  for frame in range(1, len(value)):
    if np.dot(value[frame - 1], value[frame]) < 0.0:
      value[frame] *= -1.0
  return value.astype(np.float32)


def _rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
  result = np.empty_like(vector, dtype=np.float32)
  for frame in range(len(vector)):
    rotated = np.empty(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(
      rotated,
      vector[frame].astype(np.float64),
      quaternion[frame].astype(np.float64),
    )
    result[frame] = rotated
  return result


def _quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
  w1, x1, y1, z1 = left
  w2, x2, y2, z2 = right
  return np.asarray(
    (
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ),
    dtype=np.float32,
  )


def _angular_velocity_world(root_quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
  velocity_local = np.zeros((len(root_quat_wxyz), 3), dtype=np.float32)
  for frame in range(len(root_quat_wxyz) - 1):
    conjugate = root_quat_wxyz[frame].copy()
    conjugate[1:] *= -1.0
    delta = _quaternion_multiply_wxyz(
      conjugate, root_quat_wxyz[frame + 1]
    )
    if delta[0] < 0.0:
      delta *= -1.0
    angle = 2.0 * np.arccos(np.clip(delta[0], -1.0, 1.0))
    sin_half = np.sin(0.5 * angle)
    if abs(sin_half) > 1.0e-8:
      velocity_local[frame] = delta[1:] / sin_half * angle * fps
  velocity_local[-1] = velocity_local[-2]
  return _rotate_wxyz(root_quat_wxyz, velocity_local)


def convert(
  input_path: Path,
  output_path: Path,
  model_path: Path,
  fps_override: float | None = None,
) -> None:
  input_path = input_path.expanduser().resolve()
  output_path = output_path.expanduser().resolve()
  model_path = model_path.expanduser().resolve()
  if not input_path.is_file():
    raise FileNotFoundError(input_path)
  if not model_path.is_file():
    raise FileNotFoundError(model_path)

  # Pickle can execute code while loading. Only use trusted local motion files.
  with input_path.open("rb") as file:
    source = pickle.load(file)
  if not isinstance(source, dict):
    raise TypeError(f"Expected a dict, got {type(source).__name__}")
  missing = [key for key in ("root_pos", "root_rot", "dof_pos") if key not in source]
  if missing:
    raise KeyError(f"Missing required PKL fields: {missing}")

  fps = float(fps_override if fps_override is not None else source.get("fps", 50.0))
  if not np.isfinite(fps) or fps <= 0.0:
    raise ValueError(f"FPS must be positive, got {fps}")

  root_pos = _array(source, "root_pos", (3,))
  root_rot_xyzw = _normalize_quaternions_xyzw(_array(source, "root_rot", (4,)))
  source_joint_pos = _array(source, "dof_pos", (len(MJLAB_JOINT_NAMES),))
  frame_count = len(source_joint_pos)
  if len(root_pos) != frame_count or len(root_rot_xyzw) != frame_count:
    raise ValueError("Root and joint arrays must contain the same number of frames")

  joint_pos = source_joint_pos
  # Match the original AMP converter: velocities are regenerated from the
  # exported poses instead of trusting potentially filtered PKL velocity data.
  joint_vel = _differentiate(joint_pos, fps)
  root_lin_vel_w = _differentiate(root_pos, fps)

  root_quat_wxyz = root_rot_xyzw[:, (3, 0, 1, 2)]
  meta = source.get("meta", {}) or {}
  if meta.get("root_rot_convention", "xyzw") != "xyzw":
    raise ValueError("This converter expects root_rot in xyzw order")
  root_ang_vel_w = _angular_velocity_world(root_quat_wxyz, fps)

  model = mujoco.MjModel.from_xml_path(str(model_path))
  data = mujoco.MjData(model)
  qpos_addresses = []
  for name in MJLAB_JOINT_NAMES:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
      raise ValueError(f"Joint {name!r} was not found in {model_path}")
    qpos_addresses.append(int(model.jnt_qposadr[joint_id]))

  key_body_ids = []
  for name in AMP_KEY_BODY_NAMES:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
      raise ValueError(f"AMP key body {name!r} was not found in {model_path}")
    key_body_ids.append(body_id)

  key_body_pos = np.empty(
    (frame_count, 3 * len(key_body_ids)), dtype=np.float32
  )
  key_body_quat_w = np.empty(
    (frame_count, len(key_body_ids), 4), dtype=np.float32
  )
  for frame in range(frame_count):
    data.qpos[:3] = root_pos[frame]
    data.qpos[3:7] = root_quat_wxyz[frame]
    data.qpos[qpos_addresses] = joint_pos[frame]
    mujoco.mj_forward(model, data)
    key_body_pos[frame] = data.xpos[key_body_ids].reshape(-1)
    # MuJoCo exposes body orientations in scalar-first (wxyz) order.
    key_body_quat_w[frame] = data.xquat[key_body_ids]
    if (frame + 1) % 500 == 0 or frame + 1 == frame_count:
      print(f"[INFO] FK {frame + 1}/{frame_count}")

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    output_path,
    joint_pos=joint_pos.astype(np.float32),
    joint_vel=joint_vel.astype(np.float32),
    body_pos_w=root_pos[:, None, :].astype(np.float32),
    body_quat_w=root_quat_wxyz[:, None, :].astype(np.float32),
    body_lin_vel_w=root_lin_vel_w[:, None, :].astype(np.float32),
    body_ang_vel_w=root_ang_vel_w[:, None, :].astype(np.float32),
    key_body_pos=key_body_pos,
    key_body_quat_w=key_body_quat_w,
    fps=np.asarray(fps, dtype=np.float64),
    joint_names=np.asarray(MJLAB_JOINT_NAMES),
    key_body_names=np.asarray(AMP_KEY_BODY_NAMES),
  )
  print(f"[INFO] Saved {frame_count} frames at {fps:g} FPS to: {output_path}")
  print(f"[INFO] AMP/XML joint order: {list(MJLAB_JOINT_NAMES)}")


def main() -> None:
  args = _parse_args()
  input_path = args.input.expanduser().resolve()
  inputs = sorted(input_path.glob("*.pkl")) if input_path.is_dir() else [input_path]
  if not inputs:
    raise FileNotFoundError(f"No PKL files found under {input_path}")

  for path in inputs:
    if args.output is None:
      output = path.with_suffix(".npz")
    elif len(inputs) == 1 and args.output.suffix == ".npz":
      output = args.output
    else:
      output = args.output / path.with_suffix(".npz").name
    convert(path, output, args.model, args.fps)


if __name__ == "__main__":
  main()
