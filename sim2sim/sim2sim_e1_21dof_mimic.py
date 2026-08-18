"""MuJoCo sim-to-sim for the E1 21-DOF mjlab no-state mimic policy.

This runner is intentionally specific to the actor used by
``Tracking-Flat-E1-21DOF-No-State-Estimation``.  Its observation is:

  motion command (joint_pos + joint_vel)  42
  torso IMU projected gravity              3
  torso IMU angular velocity               3
  joint position relative to default      21
  joint velocity                          21
  previous raw policy action              21
                                             ---
                                             111

The exported mimic ONNX contains the reference motion.  Consequently this
script does not load a separate NPZ and cannot accidentally use a motion whose
joint order differs from the policy.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnx
from onnx import numpy_helper


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_POLICY_PATH = SCRIPT_DIR / "policy" / "policy1.onnx"
DEFAULT_XML_PATH = (
  PROJECT_ROOT / "src" / "assets" / "e1_21dof" / "mjcf" / "E1_21dof.xml"
)

EXPECTED_OBSERVATIONS = (
  "command",
  "projected_gravity",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
  "actions",
)
OBSERVATION_SIZE = 111
ACTION_SIZE = 21


def _csv_strings(metadata: dict[str, str], key: str) -> tuple[str, ...]:
  value = metadata.get(key)
  if value is None:
    raise ValueError(f"ONNX metadata is missing {key!r}")
  return tuple(item.strip() for item in value.split(",") if item.strip())


def _csv_floats(metadata: dict[str, str], key: str) -> np.ndarray:
  try:
    return np.asarray([float(x) for x in _csv_strings(metadata, key)], dtype=np.float32)
  except ValueError as exc:
    raise ValueError(f"Invalid floating-point metadata in {key!r}") from exc


def _tensor_shape(value_info: onnx.ValueInfoProto) -> tuple[int | None, ...]:
  dims: list[int | None] = []
  for dim in value_info.type.tensor_type.shape.dim:
    dims.append(dim.dim_value if dim.HasField("dim_value") else None)
  return tuple(dims)


def _initializer_feeding_output(
  model: onnx.ModelProto, output_name: str
) -> np.ndarray:
  """Return the constant gathered by one of the bundled-motion outputs."""
  initializers = {item.name: item for item in model.graph.initializer}
  for node in model.graph.node:
    if output_name in node.output and node.op_type == "Gather":
      source_name = node.input[0]
      if source_name not in initializers:
        break
      array = numpy_helper.to_array(initializers[source_name])
      return np.asarray(array, dtype=np.float32)
  raise ValueError(f"Could not find bundled motion tensor for ONNX output {output_name!r}")


class OnnxPolicy:
  """Small inference adapter with an ONNX Runtime fast path and local fallback."""

  def __init__(self, path: Path, model: onnx.ModelProto) -> None:
    self.backend: str
    self._session: Any
    try:
      import onnxruntime as ort

      self._session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
      )
      self.backend = f"onnxruntime {ort.__version__}"
    except ImportError:
      # ONNX's reference evaluator is fast enough for this small MLP and keeps
      # the script runnable in the project's default environment.
      from onnx.reference import ReferenceEvaluator

      self._session = ReferenceEvaluator(model)
      self.backend = "onnx.reference.ReferenceEvaluator"

  def __call__(self, observation: np.ndarray, frame: int) -> np.ndarray:
    feeds = {
      "obs": observation.reshape(1, OBSERVATION_SIZE).astype(np.float32),
      "time_step": np.asarray([[frame]], dtype=np.float32),
    }
    action = self._session.run(["actions"], feeds)[0]
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (ACTION_SIZE,):
      raise RuntimeError(f"Policy returned action shape {action.shape}, expected (21,)")
    if not np.all(np.isfinite(action)):
      raise FloatingPointError("Policy returned a non-finite action")
    return action


class E1NoStateSim2Sim:
  def __init__(
    self,
    policy_path: Path,
    xml_path: Path,
    sim_dt: float,
    decimation: int,
    start_frame: int,
    loop_motion: bool,
  ) -> None:
    if not policy_path.is_file():
      raise FileNotFoundError(f"Policy does not exist: {policy_path}")
    if not xml_path.is_file():
      raise FileNotFoundError(f"MJCF does not exist: {xml_path}")
    if sim_dt <= 0.0:
      raise ValueError("sim_dt must be positive")
    if decimation <= 0:
      raise ValueError("decimation must be positive")

    self.onnx_model = onnx.load(str(policy_path))
    onnx.checker.check_model(self.onnx_model)
    self.metadata = {item.key: item.value for item in self.onnx_model.metadata_props}
    self._validate_onnx_interface()

    self.joint_names = _csv_strings(self.metadata, "joint_names")
    self.default_joint_pos = _csv_floats(self.metadata, "default_joint_pos")
    self.stiffness = _csv_floats(self.metadata, "joint_stiffness")
    self.damping = _csv_floats(self.metadata, "joint_damping")
    metadata_action_scale = _csv_floats(self.metadata, "action_scale")

    for name, array in (
      ("default_joint_pos", self.default_joint_pos),
      ("joint_stiffness", self.stiffness),
      ("joint_damping", self.damping),
      ("action_scale", metadata_action_scale),
    ):
      if array.shape != (ACTION_SIZE,):
        raise ValueError(f"ONNX {name} has shape {array.shape}, expected (21,)")

    self.motion_joint_pos = _initializer_feeding_output(
      self.onnx_model, "joint_pos"
    )
    self.motion_joint_vel = _initializer_feeding_output(
      self.onnx_model, "joint_vel"
    )
    self.motion_body_pos_w = _initializer_feeding_output(
      self.onnx_model, "body_pos_w"
    )
    self.motion_body_quat_w = _initializer_feeding_output(
      self.onnx_model, "body_quat_w"
    )
    self.motion_body_lin_vel_w = _initializer_feeding_output(
      self.onnx_model, "body_lin_vel_w"
    )
    self.motion_body_ang_vel_w = _initializer_feeding_output(
      self.onnx_model, "body_ang_vel_w"
    )
    self.motion_frames = self.motion_joint_pos.shape[0]
    self._validate_motion()

    self.body_names = _csv_strings(self.metadata, "body_names")
    try:
      self.root_motion_body_index = self.body_names.index("pelvis")
    except ValueError as exc:
      raise ValueError("ONNX bundled body_names does not contain 'pelvis'") from exc

    self.model = mujoco.MjModel.from_xml_path(str(xml_path))
    self.data = mujoco.MjData(self.model)
    self.model.opt.timestep = sim_dt
    self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    # Match the E1 task's MujocoCfg rather than the standalone XML defaults.
    self.model.opt.iterations = 10
    self.model.opt.ls_iterations = 20

    self.qpos_ids: list[int] = []
    self.qvel_ids: list[int] = []
    self.actuator_ids: list[int] = []
    for joint_name in self.joint_names:
      joint_id = mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
      )
      if joint_id < 0:
        raise ValueError(f"MJCF is missing policy joint {joint_name!r}")
      self.qpos_ids.append(int(self.model.jnt_qposadr[joint_id]))
      self.qvel_ids.append(int(self.model.jnt_dofadr[joint_id]))

      matching_actuators = np.flatnonzero(
        self.model.actuator_trnid[:, 0] == joint_id
      )
      if matching_actuators.size != 1:
        raise ValueError(
          f"Expected exactly one actuator for {joint_name!r}, "
          f"found {matching_actuators.size}"
        )
      self.actuator_ids.append(int(matching_actuators[0]))

    self.qpos_ids_np = np.asarray(self.qpos_ids, dtype=np.int32)
    self.qvel_ids_np = np.asarray(self.qvel_ids, dtype=np.int32)
    self.actuator_ids_np = np.asarray(self.actuator_ids, dtype=np.int32)
    self.torque_ranges = self.model.actuator_ctrlrange[self.actuator_ids_np].copy()
    effort_limits = np.minimum(
      np.abs(self.torque_ranges[:, 0]), np.abs(self.torque_ranges[:, 1])
    ).astype(np.float32)

    # The exporter prints three decimals. Reconstruct the exact training scale
    # from action_scale = 0.25 * effort_limit / stiffness and cross-check it.
    self.action_scale = 0.25 * effort_limits / self.stiffness
    if not np.allclose(self.action_scale, metadata_action_scale, atol=6e-4, rtol=0.0):
      raise ValueError(
        "Action scale reconstructed from MJCF torque limits disagrees with ONNX metadata"
      )

    self._require_sensor("imu_upvector", 3)
    self._require_sensor("imu_ang_vel", 3)

    self.policy = OnnxPolicy(policy_path, self.onnx_model)
    self.decimation = decimation
    self.start_frame = start_frame % self.motion_frames
    self.loop_motion = loop_motion
    self.control_step = 0
    self.previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)
    self.target_joint_pos = self.default_joint_pos.copy()

    self.reset_to_motion_frame(self.start_frame)

  @property
  def control_dt(self) -> float:
    return self.model.opt.timestep * self.decimation

  def _validate_onnx_interface(self) -> None:
    inputs = {item.name: _tensor_shape(item) for item in self.onnx_model.graph.input}
    outputs = {item.name: _tensor_shape(item) for item in self.onnx_model.graph.output}
    if inputs.get("obs") != (1, OBSERVATION_SIZE):
      raise ValueError(
        f"This runner requires ONNX input obs=(1,111), got {inputs.get('obs')}"
      )
    if inputs.get("time_step") != (1, 1):
      raise ValueError(
        f"This runner requires ONNX input time_step=(1,1), got {inputs.get('time_step')}"
      )
    if outputs.get("actions") != (1, ACTION_SIZE):
      raise ValueError(
        f"This runner requires actions=(1,21), got {outputs.get('actions')}"
      )
    observations = _csv_strings(self.metadata, "observation_names")
    if observations != EXPECTED_OBSERVATIONS:
      raise ValueError(
        "This runner only supports the no-state projected-gravity policy. "
        f"Expected observations {EXPECTED_OBSERVATIONS}, got {observations}."
      )
    if self.metadata.get("anchor_body_name") != "torso_link":
      raise ValueError(
        "This runner requires an E1 policy whose anchor_body_name is 'torso_link'"
      )

  def _validate_motion(self) -> None:
    arrays = {
      "joint_pos": self.motion_joint_pos,
      "joint_vel": self.motion_joint_vel,
      "body_pos_w": self.motion_body_pos_w,
      "body_quat_w": self.motion_body_quat_w,
      "body_lin_vel_w": self.motion_body_lin_vel_w,
      "body_ang_vel_w": self.motion_body_ang_vel_w,
    }
    for name, array in arrays.items():
      if array.shape[0] != self.motion_frames:
        raise ValueError(
          f"Bundled motion {name} has {array.shape[0]} frames, "
          f"expected {self.motion_frames}"
        )
      if not np.all(np.isfinite(array)):
        raise ValueError(f"Bundled motion {name} contains non-finite values")
    if self.motion_joint_pos.shape[1:] != (ACTION_SIZE,):
      raise ValueError(
        f"Bundled joint_pos shape is {self.motion_joint_pos.shape}, expected (T,21)"
      )
    if self.motion_joint_vel.shape[1:] != (ACTION_SIZE,):
      raise ValueError(
        f"Bundled joint_vel shape is {self.motion_joint_vel.shape}, expected (T,21)"
      )

  def _require_sensor(self, name: str, dimension: int) -> None:
    sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
      raise ValueError(f"MJCF is missing required torso IMU sensor {name!r}")
    if int(self.model.sensor_dim[sensor_id]) != dimension:
      raise ValueError(
        f"Sensor {name!r} has dimension {self.model.sensor_dim[sensor_id]}, "
        f"expected {dimension}"
      )
    site_id = -1
    if self.model.sensor_objtype[sensor_id] == mujoco.mjtObj.mjOBJ_SITE:
      site_id = int(self.model.sensor_objid[sensor_id])
    elif self.model.sensor_reftype[sensor_id] == mujoco.mjtObj.mjOBJ_SITE:
      site_id = int(self.model.sensor_refid[sensor_id])
    if site_id < 0:
      raise ValueError(f"Sensor {name!r} is not attached to an IMU site")
    body_id = int(self.model.site_bodyid[site_id])
    body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    if body_name != "torso_link":
      raise ValueError(
        f"Sensor {name!r} is attached to {body_name!r}, expected 'torso_link'"
      )

  @staticmethod
  def _quat_rotate_inverse(quat_wxyz: np.ndarray, vector_w: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the quaternion's local frame."""
    w = float(quat_wxyz[0])
    xyz = np.asarray(quat_wxyz[1:4], dtype=np.float64)
    vector = np.asarray(vector_w, dtype=np.float64)
    # Conjugate-quaternion vector rotation, avoiding a scipy dependency.
    return (
      vector * (2.0 * w * w - 1.0)
      - 2.0 * w * np.cross(xyz, vector)
      + 2.0 * xyz * np.dot(xyz, vector)
    )

  def reset_to_motion_frame(self, frame: int) -> None:
    frame %= self.motion_frames
    root = self.root_motion_body_index
    mujoco.mj_resetData(self.model, self.data)

    root_quat = self.motion_body_quat_w[frame, root].astype(np.float64).copy()
    norm = np.linalg.norm(root_quat)
    if norm <= 0.0:
      raise ValueError(f"Motion frame {frame} has an invalid root quaternion")
    root_quat /= norm

    self.data.qpos[0:3] = self.motion_body_pos_w[frame, root]
    self.data.qpos[3:7] = root_quat
    self.data.qpos[self.qpos_ids_np] = self.motion_joint_pos[frame]
    self.data.qvel[0:3] = self.motion_body_lin_vel_w[frame, root]
    self.data.qvel[3:6] = self._quat_rotate_inverse(
      root_quat, self.motion_body_ang_vel_w[frame, root]
    )
    self.data.qvel[self.qvel_ids_np] = self.motion_joint_vel[frame]
    self.data.ctrl[:] = 0.0
    mujoco.mj_forward(self.model, self.data)

    self.previous_action.fill(0.0)
    self.target_joint_pos = self.motion_joint_pos[frame].copy()
    self.control_step = 0

  def motion_frame(self) -> int:
    # mjlab resets the robot to start_frame, then CommandManager.compute()
    # advances MotionCommand once before producing the first actor observation.
    frame = self.start_frame + self.control_step + 1
    if self.loop_motion:
      return frame % self.motion_frames
    return min(frame, self.motion_frames - 1)

  def get_observation(self, frame: int) -> np.ndarray:
    command = np.concatenate(
      (self.motion_joint_pos[frame], self.motion_joint_vel[frame])
    )
    # mjlab's projected_gravity_from_sensor negates the framezaxis up-vector.
    projected_gravity = -self.data.sensor("imu_upvector").data.copy()
    torso_ang_vel = self.data.sensor("imu_ang_vel").data.copy()
    joint_pos = self.data.qpos[self.qpos_ids_np].copy()
    joint_vel = self.data.qvel[self.qvel_ids_np].copy()

    observation = np.concatenate(
      (
        command,
        projected_gravity,
        torso_ang_vel,
        joint_pos - self.default_joint_pos,
        joint_vel,
        self.previous_action,
      )
    ).astype(np.float32)
    if observation.shape != (OBSERVATION_SIZE,):
      raise RuntimeError(
        f"Constructed observation shape {observation.shape}, expected (111,)"
      )
    if not np.all(np.isfinite(observation)):
      raise FloatingPointError("Constructed observation contains non-finite values")
    return observation

  def update_policy(self) -> tuple[int, np.ndarray]:
    # Refresh site-frame sensors from the current post-integration state, just
    # as mjlab forward()+sense() does before computing actor observations.
    mujoco.mj_forward(self.model, self.data)
    frame = self.motion_frame()
    observation = self.get_observation(frame)
    action = self.policy(observation, frame)
    self.previous_action = action.copy()
    self.target_joint_pos = self.default_joint_pos + action * self.action_scale
    self.control_step += 1
    return frame, observation

  def physics_step(self) -> np.ndarray:
    joint_pos = self.data.qpos[self.qpos_ids_np]
    joint_vel = self.data.qvel[self.qvel_ids_np]
    torque = self.stiffness * (self.target_joint_pos - joint_pos)
    torque -= self.damping * joint_vel
    torque = np.clip(torque, self.torque_ranges[:, 0], self.torque_ranges[:, 1])
    if not np.all(np.isfinite(torque)):
      raise FloatingPointError("PD controller produced a non-finite torque")
    self.data.ctrl[self.actuator_ids_np] = torque
    mujoco.mj_step(self.model, self.data)
    return torque


def run(args: argparse.Namespace) -> None:
  runner = E1NoStateSim2Sim(
    policy_path=args.policy.resolve(),
    xml_path=args.xml.resolve(),
    sim_dt=args.sim_dt,
    decimation=args.decimation,
    start_frame=args.start_frame,
    loop_motion=args.loop_motion,
  )

  total_physics_steps = math.ceil(args.duration / runner.model.opt.timestep)
  torso_id = mujoco.mj_name2id(
    runner.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
  )
  print("[INFO] E1 21-DOF no-state projected-gravity sim2sim")
  print(f"[INFO] Policy: {args.policy.resolve()}")
  print(f"[INFO] MJCF: {args.xml.resolve()}")
  print(f"[INFO] Inference backend: {runner.policy.backend}")
  print(
    f"[INFO] Motion: {runner.motion_frames} frames, "
    f"control_dt={runner.control_dt:.3f}s ({1.0 / runner.control_dt:.1f} Hz)"
  )
  print(f"[INFO] Observation/action: {OBSERVATION_SIZE}/{ACTION_SIZE}")

  if args.headless:
    viewer_context: Any = contextlib.nullcontext(None)
  else:
    from mujoco import viewer as mujoco_viewer

    viewer_context = mujoco_viewer.launch_passive(runner.model, runner.data)

  last_log_time = -float("inf")
  current_frame = runner.start_frame
  last_torque = np.zeros(ACTION_SIZE, dtype=np.float32)
  wall_start = time.perf_counter()

  with viewer_context as viewer:
    if viewer is not None:
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      viewer.cam.trackbodyid = torso_id
      viewer.cam.distance = 2.8
      viewer.cam.azimuth = 120.0
      viewer.cam.elevation = -5.0

    for physics_step in range(total_physics_steps):
      step_start = time.perf_counter()
      if physics_step % runner.decimation == 0:
        current_frame, _ = runner.update_policy()

      last_torque = runner.physics_step()

      if viewer is not None:
        if not viewer.is_running():
          break
        viewer.sync()

      if runner.data.time - last_log_time >= args.log_interval:
        joint_pos = runner.data.qpos[runner.qpos_ids_np]
        print(
          f"\r[SIM] t={runner.data.time:7.2f}s frame={current_frame:4d} "
          f"height={runner.data.qpos[2]:.3f} "
          f"|q|={np.linalg.norm(joint_pos):.3f} "
          f"|tau|max={np.max(np.abs(last_torque)):.2f}    ",
          end="",
          flush=True,
        )
        last_log_time = runner.data.time

      if args.realtime and not args.headless:
        remaining = runner.model.opt.timestep - (time.perf_counter() - step_start)
        if remaining > 0.0:
          time.sleep(remaining)

  print()
  elapsed = time.perf_counter() - wall_start
  simulated = float(runner.data.time)
  print(
    f"[INFO] Finished: simulated={simulated:.3f}s, wall={elapsed:.3f}s, "
    f"real-time factor={simulated / max(elapsed, 1e-9):.2f}x"
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Run the mjlab E1 21-DOF No-State-Estimation 111D ONNX policy in native MuJoCo."
    )
  )
  parser.add_argument(
    "--policy",
    type=Path,
    default=DEFAULT_POLICY_PATH,
    help="Mimic ONNX policy with bundled motion.",
  )
  parser.add_argument(
    "--xml",
    type=Path,
    default=DEFAULT_XML_PATH,
    help="Standalone E1 MuJoCo XML.",
  )
  parser.add_argument("--duration", type=float, default=300.0)
  parser.add_argument("--sim-dt", type=float, default=0.005)
  parser.add_argument("--decimation", type=int, default=4)
  parser.add_argument("--start-frame", type=int, default=0)
  parser.add_argument(
    "--loop-motion",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Loop the ONNX-bundled motion after its final frame.",
  )
  parser.add_argument(
    "--realtime",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Pace GUI simulation to wall-clock time.",
  )
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--log-interval", type=float, default=1.0)
  args = parser.parse_args()
  if args.duration <= 0.0:
    parser.error("--duration must be positive")
  if args.log_interval <= 0.0:
    parser.error("--log-interval must be positive")
  return args


if __name__ == "__main__":
  run(parse_args())
