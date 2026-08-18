"""Convert an E1 21-DOF GMR pickle into an mjlab tracking motion NPZ.

The input ``dof_pos`` and ``dof_vel`` are assumed to already use the exact
non-free joint order of the E1 MJCF. No joint remapping is performed.

Example:
  python tools/mimic/pkl_to_npz.py \
    --input dataset/e1_21dof/mimic/dance_pkl/MJ_dance.pkl \
    --output dataset/e1_21dof/mimic/dance_npz/MJ_dance.npz
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "src/assets/e1_21dof/mjcf/E1_21dof.xml"

REQUIRED_KEYS = (
  "root_pos",
  "root_rot",
  "dof_pos",
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Convert an E1 21-DOF PKL motion to mjlab's NPZ format."
  )
  parser.add_argument("--input", type=Path, required=True, help="Input .pkl file.")
  parser.add_argument("--output", type=Path, default=None, help="Output .npz file.")
  parser.add_argument(
    "--model",
    type=Path,
    default=DEFAULT_MODEL,
    help="E1 MuJoCo XML used for forward kinematics.",
  )
  parser.add_argument(
    "--fps",
    type=float,
    default=None,
    help="Override input FPS. By default the PKL's fps field is used.",
  )
  return parser.parse_args()


def _as_float32(data: dict[str, Any], key: str, shape_tail: tuple[int, ...]) -> np.ndarray:
  value = np.asarray(data[key], dtype=np.float32)
  if value.ndim != len(shape_tail) + 1 or value.shape[1:] != shape_tail:
    raise ValueError(
      f"{key} must have shape (frames, {', '.join(map(str, shape_tail))}), "
      f"got {value.shape}"
    )
  if not np.isfinite(value).all():
    raise ValueError(f"{key} contains NaN or infinity")
  return value


def _differentiate(values: np.ndarray, fps: float) -> np.ndarray:
  if len(values) < 2:
    return np.zeros_like(values)
  return np.gradient(values, 1.0 / fps, axis=0, edge_order=1).astype(np.float32)


def _normalize_quaternions_xyzw(quaternions: np.ndarray) -> np.ndarray:
  norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
  if np.any(norms < 1.0e-8):
    raise ValueError("root_rot contains a zero-length quaternion")
  quaternions = quaternions / norms
  # q and -q encode the same rotation. Keeping adjacent samples in one
  # hemisphere prevents discontinuities in exported pose sequences.
  for frame in range(1, len(quaternions)):
    if np.dot(quaternions[frame - 1], quaternions[frame]) < 0.0:
      quaternions[frame] *= -1.0
  return quaternions.astype(np.float32)


def _quat_apply_inverse_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
  result = np.empty(3, dtype=np.float64)
  conjugate = quat.astype(np.float64).copy()
  conjugate[1:] *= -1.0
  mujoco.mju_rotVecQuat(result, vector.astype(np.float64), conjugate)
  return result.astype(np.float32)


def _model_names(model: mujoco.MjModel) -> tuple[list[str], list[str], list[int], list[int]]:
  joint_names: list[str] = []
  joint_qpos_addresses: list[int] = []
  joint_dof_addresses: list[int] = []
  for joint_id in range(model.njnt):
    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
      raise ValueError("The converter currently expects only hinge joints after the free joint")
    joint_names.append(
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    )
    joint_qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
    joint_dof_addresses.append(int(model.jnt_dofadr[joint_id]))

  body_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    for body_id in range(1, model.nbody)
  ]
  return joint_names, body_names, joint_qpos_addresses, joint_dof_addresses


def convert(input_path: Path, output_path: Path, model_path: Path, fps_override: float | None) -> None:
  input_path = input_path.expanduser().resolve()
  output_path = output_path.expanduser().resolve()
  model_path = model_path.expanduser().resolve()

  if not input_path.is_file():
    raise FileNotFoundError(input_path)
  if not model_path.is_file():
    raise FileNotFoundError(model_path)

  # Pickle files can execute code while loading. This tool is intended only
  # for trusted, locally generated motion files.
  with input_path.open("rb") as file:
    source = pickle.load(file)
  if not isinstance(source, dict):
    raise TypeError(f"Expected a dict in {input_path}, got {type(source).__name__}")
  missing = [key for key in REQUIRED_KEYS if key not in source]
  if missing:
    raise KeyError(f"Missing required PKL fields: {missing}")

  model = mujoco.MjModel.from_xml_path(str(model_path))
  sim_data = mujoco.MjData(model)
  joint_names, body_names, joint_qpos_adr, joint_dof_adr = _model_names(model)

  root_pos = _as_float32(source, "root_pos", (3,))
  root_rot_xyzw = _normalize_quaternions_xyzw(
    _as_float32(source, "root_rot", (4,))
  )
  joint_pos = _as_float32(source, "dof_pos", (len(joint_names),))
  frame_count = len(joint_pos)
  for key, value in (("root_pos", root_pos), ("root_rot", root_rot_xyzw)):
    if len(value) != frame_count:
      raise ValueError(f"{key} has {len(value)} frames, expected {frame_count}")

  fps = float(fps_override if fps_override is not None else source.get("fps", 50.0))
  if not np.isfinite(fps) or fps <= 0.0:
    raise ValueError(f"FPS must be positive, got {fps}")

  if "dof_vel" in source:
    joint_vel = _as_float32(source, "dof_vel", (len(joint_names),))
  else:
    joint_vel = _differentiate(joint_pos, fps)
  if len(joint_vel) != frame_count:
    raise ValueError(f"dof_vel has {len(joint_vel)} frames, expected {frame_count}")

  root_lin_vel_w = (
    _as_float32(source, "root_vel", (3,))
    if "root_vel" in source
    else _differentiate(root_pos, fps)
  )
  root_ang_vel = (
    _as_float32(source, "root_rot_vel", (3,))
    if "root_rot_vel" in source
    else np.zeros((frame_count, 3), dtype=np.float32)
  )
  if len(root_lin_vel_w) != frame_count or len(root_ang_vel) != frame_count:
    raise ValueError("Root velocity frame count does not match dof_pos")

  meta = source.get("meta", {})
  quat_convention = meta.get("root_rot_convention", "xyzw")
  if quat_convention != "xyzw":
    raise ValueError(f"Expected root_rot_convention='xyzw', got {quat_convention!r}")
  ang_vel_space = meta.get("root_ang_vel_space", "local")
  if ang_vel_space not in ("local", "world"):
    raise ValueError(f"Unsupported root_ang_vel_space: {ang_vel_space!r}")

  body_count = len(body_names)
  body_pos_w = np.empty((frame_count, body_count, 3), dtype=np.float32)
  body_quat_w = np.empty((frame_count, body_count, 4), dtype=np.float32)
  body_lin_vel_w = np.empty((frame_count, body_count, 3), dtype=np.float32)
  body_ang_vel_w = np.empty((frame_count, body_count, 3), dtype=np.float32)

  for frame in range(frame_count):
    root_quat_wxyz = root_rot_xyzw[frame, [3, 0, 1, 2]]
    sim_data.qpos[:3] = root_pos[frame]
    sim_data.qpos[3:7] = root_quat_wxyz
    sim_data.qpos[joint_qpos_adr] = joint_pos[frame]

    sim_data.qvel[:3] = root_lin_vel_w[frame]
    if ang_vel_space == "local":
      sim_data.qvel[3:6] = root_ang_vel[frame]
    else:
      sim_data.qvel[3:6] = _quat_apply_inverse_wxyz(
        root_quat_wxyz, root_ang_vel[frame]
      )
    sim_data.qvel[joint_dof_adr] = joint_vel[frame]

    mujoco.mj_forward(model, sim_data)
    body_pos_w[frame] = sim_data.xpos[1:]
    body_quat_w[frame] = sim_data.xquat[1:]

    # This is the same cvel-to-link-velocity conversion used by mjlab's
    # EntityData.body_link_vel_w property.
    angular_w = sim_data.cvel[1:, :3]
    subtree_com = sim_data.subtree_com[1]
    offset = subtree_com[None, :] - sim_data.xpos[1:]
    body_lin_vel_w[frame] = sim_data.cvel[1:, 3:6] - np.cross(
      angular_w, offset
    )
    body_ang_vel_w[frame] = angular_w

    if (frame + 1) % 500 == 0 or frame + 1 == frame_count:
      print(f"[INFO] FK {frame + 1}/{frame_count}")

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    output_path,
    fps=np.asarray([fps], dtype=np.float64),
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
    joint_names=np.asarray(joint_names),
    body_names=np.asarray(body_names),
  )

  print(f"[INFO] Joint order ({len(joint_names)}): {joint_names}")
  print(f"[INFO] Body order ({len(body_names)}): {body_names}")
  print(f"[INFO] Saved {frame_count} frames at {fps:g} FPS to: {output_path}")


def main() -> None:
  args = _parse_args()
  output = args.output or args.input.with_suffix(".npz")
  convert(args.input, output, args.model, args.fps)


if __name__ == "__main__":
  main()
