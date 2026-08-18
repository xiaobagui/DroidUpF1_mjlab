"""Kinematically replay an mjlab tracking NPZ in MuJoCo.

Example:
  python tools/mimic/replay_npz.py \
    --motion dataset/e1_21dof/mimic/dance_npz/MJ_dance.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "src/assets/e1_21dof/mjcf/E1_21dof.xml"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Replay an E1 mjlab motion NPZ.")
  parser.add_argument("--motion", type=Path, required=True, help="Motion .npz file.")
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="E1 MJCF.")
  parser.add_argument("--start-frame", type=int, default=0)
  parser.add_argument("--speed", type=float, default=1.0)
  parser.add_argument("--no-loop", action="store_true", help="Stop after one pass.")
  parser.add_argument(
    "--headless",
    action="store_true",
    help="Validate every frame without opening the viewer.",
  )
  return parser.parse_args()


def _model_names(model: mujoco.MjModel) -> tuple[list[str], list[str], list[int], list[int]]:
  joint_names: list[str] = []
  joint_qpos_adr: list[int] = []
  joint_dof_adr: list[int] = []
  for joint_id in range(model.njnt):
    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    joint_names.append(
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    )
    joint_qpos_adr.append(int(model.jnt_qposadr[joint_id]))
    joint_dof_adr.append(int(model.jnt_dofadr[joint_id]))
  body_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    for body_id in range(1, model.nbody)
  ]
  return joint_names, body_names, joint_qpos_adr, joint_dof_adr


def _load_motion(path: Path, model: mujoco.MjModel) -> dict[str, np.ndarray | float]:
  path = path.expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError(path)
  source = np.load(path, allow_pickle=False)
  required = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  )
  missing = [key for key in required if key not in source]
  if missing:
    raise KeyError(f"Missing NPZ fields: {missing}")

  joint_names, body_names, _, _ = _model_names(model)
  joint_pos = np.asarray(source["joint_pos"], dtype=np.float64)
  frame_count = len(joint_pos)
  expected_shapes = {
    "joint_pos": (frame_count, len(joint_names)),
    "joint_vel": (frame_count, len(joint_names)),
    "body_pos_w": (frame_count, len(body_names), 3),
    "body_quat_w": (frame_count, len(body_names), 4),
    "body_lin_vel_w": (frame_count, len(body_names), 3),
    "body_ang_vel_w": (frame_count, len(body_names), 3),
  }
  motion: dict[str, np.ndarray | float] = {}
  for key, shape in expected_shapes.items():
    value = np.asarray(source[key], dtype=np.float64)
    if value.shape != shape:
      raise ValueError(f"{key} has shape {value.shape}, expected {shape}")
    if not np.isfinite(value).all():
      raise ValueError(f"{key} contains NaN or infinity")
    motion[key] = value

  if "joint_names" in source:
    stored = source["joint_names"].astype(str).tolist()
    if stored != joint_names:
      raise ValueError(f"NPZ/MJCF joint order mismatch:\nNPZ: {stored}\nMJCF: {joint_names}")
  if "body_names" in source:
    stored = source["body_names"].astype(str).tolist()
    if stored != body_names:
      raise ValueError(f"NPZ/MJCF body order mismatch:\nNPZ: {stored}\nMJCF: {body_names}")

  fps = float(np.asarray(source["fps"]).reshape(-1)[0])
  if not np.isfinite(fps) or fps <= 0.0:
    raise ValueError(f"Invalid FPS: {fps}")
  motion["fps"] = fps
  return motion


def _write_frame(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  motion: dict[str, np.ndarray | float],
  frame: int,
  joint_qpos_adr: list[int],
  joint_dof_adr: list[int],
) -> None:
  joint_pos = motion["joint_pos"]
  joint_vel = motion["joint_vel"]
  body_pos_w = motion["body_pos_w"]
  body_quat_w = motion["body_quat_w"]
  body_lin_vel_w = motion["body_lin_vel_w"]
  body_ang_vel_w = motion["body_ang_vel_w"]
  assert isinstance(joint_pos, np.ndarray)
  assert isinstance(joint_vel, np.ndarray)
  assert isinstance(body_pos_w, np.ndarray)
  assert isinstance(body_quat_w, np.ndarray)
  assert isinstance(body_lin_vel_w, np.ndarray)
  assert isinstance(body_ang_vel_w, np.ndarray)

  root_quat_wxyz = body_quat_w[frame, 0]
  data.qpos[:3] = body_pos_w[frame, 0]
  data.qpos[3:7] = root_quat_wxyz
  data.qpos[joint_qpos_adr] = joint_pos[frame]
  data.qvel[:3] = body_lin_vel_w[frame, 0]

  root_ang_vel_local = np.empty(3, dtype=np.float64)
  conjugate = root_quat_wxyz.copy()
  conjugate[1:] *= -1.0
  mujoco.mju_rotVecQuat(root_ang_vel_local, body_ang_vel_w[frame, 0], conjugate)
  data.qvel[3:6] = root_ang_vel_local
  data.qvel[joint_dof_adr] = joint_vel[frame]
  data.time = frame / float(motion["fps"])
  mujoco.mj_forward(model, data)


def replay(args: argparse.Namespace) -> None:
  model_path = args.model.expanduser().resolve()
  if not model_path.is_file():
    raise FileNotFoundError(model_path)
  if args.speed <= 0.0:
    raise ValueError("--speed must be positive")

  model = mujoco.MjModel.from_xml_path(str(model_path))
  data = mujoco.MjData(model)
  joint_names, body_names, joint_qpos_adr, joint_dof_adr = _model_names(model)
  motion = _load_motion(args.motion, model)
  joint_pos = motion["joint_pos"]
  assert isinstance(joint_pos, np.ndarray)
  frame_count = len(joint_pos)
  if not 0 <= args.start_frame < frame_count:
    raise ValueError(f"--start-frame must be in [0, {frame_count - 1}]")

  print(f"[INFO] Frames: {frame_count}, FPS: {motion['fps']}")
  print(f"[INFO] Joint order ({len(joint_names)}): {joint_names}")
  print(f"[INFO] Body order ({len(body_names)}): {body_names}")

  if args.headless:
    for frame in range(frame_count):
      _write_frame(model, data, motion, frame, joint_qpos_adr, joint_dof_adr)
    print(f"[INFO] Headless validation passed for all {frame_count} frames")
    return

  from mujoco import viewer as mj_viewer

  frame = args.start_frame
  frame_period = 1.0 / (float(motion["fps"]) * args.speed)
  with mj_viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      start = time.perf_counter()
      _write_frame(model, data, motion, frame, joint_qpos_adr, joint_dof_adr)
      viewer.sync()

      frame += 1
      if frame >= frame_count:
        if args.no_loop:
          break
        frame = 0

      remaining = frame_period - (time.perf_counter() - start)
      if remaining > 0.0:
        time.sleep(remaining)


def main() -> None:
  replay(_parse_args())


if __name__ == "__main__":
  main()
