"""Kinematically replay or validate an E1 21-DOF AMP NPZ in MuJoCo.

Example:
  python tools/amp/replay_npz.py \
    --motion-file dataset/e1_21dof/amp/walk_s1_1_e1_21dof.npz

Use ``--headless`` to validate every frame without opening a viewer.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from src.tasks.amp.constants import AMP_KEY_BODY_NAMES, MJLAB_JOINT_NAMES


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "src/assets/e1_21dof/mjcf/E1_21dof.xml"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Replay an E1 AMP NPZ in MuJoCo.")
  parser.add_argument(
    "--motion-file",
    "--motion_file",
    "--motion",
    dest="motion_file",
    type=Path,
    required=True,
  )
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument(
    "--height-offset", "--height_offset", dest="height_offset", type=float, default=0.0
  )
  parser.add_argument(
    "--start-frame", "--start_frame", dest="start_frame", type=int, default=0
  )
  parser.add_argument("--speed", type=float, default=1.0)
  parser.add_argument("--no-loop", action="store_true")
  parser.add_argument(
    "--headless", action="store_true", help="Validate all frames without a viewer."
  )
  return parser.parse_args()


def _load_motion(path: Path) -> dict[str, np.ndarray | float | list[str]]:
  path = path.expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError(path)
  with np.load(path, allow_pickle=False) as source:
    required = (
      "joint_pos",
      "joint_vel",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
      "key_body_pos",
      "key_body_quat_w",
      "fps",
    )
    missing = [name for name in required if name not in source.files]
    if missing:
      raise KeyError(f"Missing AMP NPZ fields: {missing}")
    motion: dict[str, np.ndarray | float | list[str]] = {
      name: np.asarray(source[name], dtype=np.float64) for name in required[:-1]
    }
    motion["fps"] = float(np.asarray(source["fps"]).reshape(-1)[0])
    motion["joint_names"] = (
      source["joint_names"].astype(str).tolist()
      if "joint_names" in source.files
      else list(MJLAB_JOINT_NAMES)
    )
    if "key_body_names" in source.files:
      stored_key_bodies = source["key_body_names"].astype(str).tolist()
      if stored_key_bodies != list(AMP_KEY_BODY_NAMES):
        raise ValueError(
          f"Unexpected AMP key-body order: {stored_key_bodies}"
        )

  joint_pos = motion["joint_pos"]
  joint_vel = motion["joint_vel"]
  body_pos = motion["body_pos_w"]
  body_quat = motion["body_quat_w"]
  body_lin = motion["body_lin_vel_w"]
  body_ang = motion["body_ang_vel_w"]
  key_body_pos = motion["key_body_pos"]
  key_body_quat = motion["key_body_quat_w"]
  joint_names = motion["joint_names"]
  assert isinstance(joint_pos, np.ndarray)
  assert isinstance(joint_vel, np.ndarray)
  assert isinstance(body_pos, np.ndarray)
  assert isinstance(body_quat, np.ndarray)
  assert isinstance(body_lin, np.ndarray)
  assert isinstance(body_ang, np.ndarray)
  assert isinstance(key_body_pos, np.ndarray)
  assert isinstance(key_body_quat, np.ndarray)
  assert isinstance(joint_names, list)
  frame_count = len(joint_pos)
  if body_pos.ndim != 3 or body_pos.shape[0] != frame_count or body_pos.shape[2] != 3:
    raise ValueError(f"body_pos_w has invalid shape {body_pos.shape}")
  body_count = body_pos.shape[1]
  if body_count < 1:
    raise ValueError("body_pos_w must contain at least the root body")
  expected = {
    "joint_pos": (frame_count, len(joint_names)),
    "joint_vel": (frame_count, len(joint_names)),
    "body_pos_w": (frame_count, body_count, 3),
    "body_quat_w": (frame_count, body_count, 4),
    "body_lin_vel_w": (frame_count, body_count, 3),
    "body_ang_vel_w": (frame_count, body_count, 3),
    "key_body_pos": (frame_count, 3 * len(AMP_KEY_BODY_NAMES)),
    "key_body_quat_w": (frame_count, len(AMP_KEY_BODY_NAMES), 4),
  }
  for name, shape in expected.items():
    value = motion[name]
    assert isinstance(value, np.ndarray)
    if value.shape != shape:
      raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
    if not np.isfinite(value).all():
      raise ValueError(f"{name} contains NaN or infinity")
  quat_norm = np.linalg.norm(key_body_quat, axis=-1)
  if not np.allclose(quat_norm, 1.0, atol=1.0e-4):
    max_error = float(np.max(np.abs(quat_norm - 1.0)))
    raise ValueError(
      f"key_body_quat_w contains non-unit quaternions "
      f"(maximum norm error {max_error:.3e})"
    )
  fps = float(motion["fps"])
  if not np.isfinite(fps) or fps <= 0.0:
    raise ValueError(f"Invalid FPS: {fps}")
  if len(set(joint_names)) != len(joint_names):
    raise ValueError("joint_names contains duplicates")
  if joint_names != list(MJLAB_JOINT_NAMES):
    raise ValueError(
      f"NPZ joint order does not match E1_21dof.xml:\n"
      f"NPZ: {joint_names}\nXML: {list(MJLAB_JOINT_NAMES)}"
    )
  return motion


class NPZReplayer:
  def __init__(
    self, motion_path: Path, model_path: Path, height_offset: float = 0.0
  ):
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
      raise FileNotFoundError(model_path)
    self.model = mujoco.MjModel.from_xml_path(str(model_path))
    self.data = mujoco.MjData(self.model)
    self.motion = _load_motion(motion_path)
    self.height_offset = height_offset

    joint_names = self.motion["joint_names"]
    assert isinstance(joint_names, list)
    self.qpos_addresses: list[int] = []
    self.dof_addresses: list[int] = []
    for name in joint_names:
      joint_id = mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_JOINT, name
      )
      if joint_id < 0:
        raise ValueError(f"Joint {name!r} was not found in {model_path}")
      self.qpos_addresses.append(int(self.model.jnt_qposadr[joint_id]))
      self.dof_addresses.append(int(self.model.jnt_dofadr[joint_id]))

    self.key_body_ids: list[int] = []
    for name in AMP_KEY_BODY_NAMES:
      body_id = mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_BODY, name
      )
      if body_id < 0:
        raise ValueError(f"AMP key body {name!r} was not found in {model_path}")
      self.key_body_ids.append(body_id)

    joint_pos = self.motion["joint_pos"]
    assert isinstance(joint_pos, np.ndarray)
    self.frame_count = len(joint_pos)
    self.fps = float(self.motion["fps"])
    print(
      f"[INFO] Loaded {self.frame_count} frames, {self.fps:g} FPS, "
      f"{len(joint_names)} joints"
    )
    print(f"[INFO] Joint order: {joint_names}")

  def set_frame(self, frame: int) -> None:
    joint_pos = self.motion["joint_pos"]
    joint_vel = self.motion["joint_vel"]
    body_pos = self.motion["body_pos_w"]
    body_quat = self.motion["body_quat_w"]
    body_lin = self.motion["body_lin_vel_w"]
    body_ang = self.motion["body_ang_vel_w"]
    assert isinstance(joint_pos, np.ndarray)
    assert isinstance(joint_vel, np.ndarray)
    assert isinstance(body_pos, np.ndarray)
    assert isinstance(body_quat, np.ndarray)
    assert isinstance(body_lin, np.ndarray)
    assert isinstance(body_ang, np.ndarray)

    root_quat_wxyz = body_quat[frame, 0]
    self.data.qpos[:3] = body_pos[frame, 0]
    self.data.qpos[2] += self.height_offset
    self.data.qpos[3:7] = root_quat_wxyz
    self.data.qpos[self.qpos_addresses] = joint_pos[frame]
    self.data.qvel[:3] = body_lin[frame, 0]

    root_ang_vel_local = np.empty(3, dtype=np.float64)
    conjugate = root_quat_wxyz.copy()
    conjugate[1:] *= -1.0
    mujoco.mju_rotVecQuat(root_ang_vel_local, body_ang[frame, 0], conjugate)
    self.data.qvel[3:6] = root_ang_vel_local
    self.data.qvel[self.dof_addresses] = joint_vel[frame]
    self.data.time = frame / self.fps
    mujoco.mj_forward(self.model, self.data)

  def validate(self) -> None:
    stored_pos = self.motion["key_body_pos"]
    stored_quat = self.motion["key_body_quat_w"]
    assert isinstance(stored_pos, np.ndarray)
    assert isinstance(stored_quat, np.ndarray)
    stored_pos = stored_pos.reshape(
      self.frame_count, len(self.key_body_ids), 3
    )
    max_position_error = 0.0
    max_orientation_error = 0.0
    for frame in range(self.frame_count):
      self.set_frame(frame)
      position_error = np.linalg.norm(
        self.data.xpos[self.key_body_ids] - stored_pos[frame], axis=1
      )
      max_position_error = max(max_position_error, float(position_error.max()))

      # q and -q encode the same rotation, so compare their absolute dot.
      dots = np.abs(
        np.sum(self.data.xquat[self.key_body_ids] * stored_quat[frame], axis=1)
      )
      orientation_error = 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))
      max_orientation_error = max(
        max_orientation_error, float(orientation_error.max())
      )

    # Quaternion values are stored as float32.  The acos conversion amplifies
    # their rounding error near an exact dot product of one.
    if max_position_error > 1.0e-5 or max_orientation_error > 1.0e-3:
      raise ValueError(
        "Stored key-body FK does not match the XML replay: "
        f"position error={max_position_error:.3e} m, "
        f"orientation error={max_orientation_error:.3e} rad"
      )
    print(
      f"[INFO] Headless validation passed for all {self.frame_count} frames; "
      f"key-body FK max errors: {max_position_error:.3e} m, "
      f"{max_orientation_error:.3e} rad"
    )

  def run(self, start_frame: int, speed: float, loop: bool) -> None:
    if not 0 <= start_frame < self.frame_count:
      raise ValueError(f"--start-frame must be in [0, {self.frame_count - 1}]")
    if speed <= 0.0:
      raise ValueError("--speed must be positive")
    from mujoco import viewer as mj_viewer

    frame = start_frame
    frame_period = 1.0 / (self.fps * speed)
    with mj_viewer.launch_passive(self.model, self.data) as viewer:
      viewer.cam.distance = 3.0
      viewer.cam.azimuth = 45.0
      viewer.cam.elevation = -20.0
      while viewer.is_running():
        start = time.perf_counter()
        self.set_frame(frame)
        viewer.cam.lookat[:] = self.data.qpos[:3]
        viewer.sync()
        frame += 1
        if frame >= self.frame_count:
          if not loop:
            break
          frame = 0
        remaining = frame_period - (time.perf_counter() - start)
        if remaining > 0.0:
          time.sleep(remaining)


def main() -> None:
  args = _parse_args()
  replayer = NPZReplayer(args.motion_file, args.model, args.height_offset)
  if args.headless:
    replayer.validate()
  else:
    replayer.run(args.start_frame, args.speed, not args.no_loop)


if __name__ == "__main__":
  main()
