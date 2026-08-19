"""Run the mjlab E1 21-DOF velocity-commanded AMP policy in MuJoCo.

The runner reproduces the 360-D actor observation used by
``AMP-Walk-Flat-E1-21DOF``:

  torso angular velocity * 0.2       3
  torso projected gravity            3
  body-frame velocity command        3
  joint position relative to default 21
  joint velocity * 0.05              21
  previous raw policy action         21
                                      --
  one frame                          72
  five frames, oldest to newest     360

Joint states, actions and PD gains use the exact E1_21dof.xml joint order.
The exported ONNX metadata is treated as the source of truth and is checked
against the standalone MuJoCo model before simulation starts.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import math
import os
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnx


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_XML_PATH = (
  PROJECT_ROOT / "src" / "assets" / "e1_21dof" / "mjcf" / "E1_21dof.xml"
)

ACTION_SIZE = 21
FRAME_SIZE = 72
HISTORY_LENGTH = 5
OBSERVATION_SIZE = FRAME_SIZE * HISTORY_LENGTH

ANGULAR_VELOCITY_SCALE = 0.2
JOINT_VELOCITY_SCALE = 0.05
OBSERVATION_CLIP = (-100.0, 100.0)
ACTION_CLIP = (-100.0, 100.0)

# Final training command ranges from e1_21dof_walk_amp_env_cfg().
MAX_FORWARD_VELOCITY = 1.4
MAX_BACKWARD_VELOCITY = 0.8
MAX_LATERAL_VELOCITY = 0.6
MAX_YAW_VELOCITY = 1.5
KEYBOARD_COMMAND_STEP = 0.1


class KeyboardCommand:
  """Thread-safe accumulated velocity command controlled by terminal keys."""

  def __init__(self) -> None:
    self._command = np.zeros(3, dtype=np.float32)
    self._running = True
    self._lock = threading.Lock()

  @property
  def running(self) -> bool:
    with self._lock:
      return self._running

  def on_key(self, key: int) -> None:
    """Handle MuJoCo viewer key codes (ASCII codes for letter keys)."""
    try:
      char = chr(key).lower()
    except (TypeError, ValueError):
      return
    with self._lock:
      if char == "w":
        self._command[0] += KEYBOARD_COMMAND_STEP
      elif char == "s":
        self._command[0] -= KEYBOARD_COMMAND_STEP
      elif char == "d":
        self._command[1] += KEYBOARD_COMMAND_STEP
      elif char == "a":
        self._command[1] -= KEYBOARD_COMMAND_STEP
      elif char == "j":
        self._command[2] += KEYBOARD_COMMAND_STEP
      elif char == "l":
        self._command[2] -= KEYBOARD_COMMAND_STEP
      elif char == "r":
        self._command.fill(0.0)
      elif char == "q":
        self._running = False
        return
      self._command[0] = np.clip(
        self._command[0], -MAX_BACKWARD_VELOCITY, MAX_FORWARD_VELOCITY
      )
      self._command[1] = np.clip(
        self._command[1], -MAX_LATERAL_VELOCITY, MAX_LATERAL_VELOCITY
      )
      self._command[2] = np.clip(
        self._command[2], -MAX_YAW_VELOCITY, MAX_YAW_VELOCITY
      )

  def get_command(self) -> np.ndarray:
    with self._lock:
      return self._command.copy()


class TerminalKeyboard:
  """Read single keystrokes from the launching terminal without pressing Enter."""

  def __init__(self, command: KeyboardCommand) -> None:
    self.command = command
    self.fd = sys.stdin.fileno()
    self._saved_attributes: list[Any] | None = None

  def __enter__(self) -> "TerminalKeyboard":
    if not sys.stdin.isatty():
      raise RuntimeError("--keyboard requires a TTY terminal as stdin")
    self._saved_attributes = termios.tcgetattr(self.fd)
    tty.setcbreak(self.fd)
    return self

  def poll(self) -> None:
    while select.select([self.fd], [], [], 0.0)[0]:
      value = os.read(self.fd, 1)
      if not value:
        return
      self.command.on_key(value[0])

  def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
    if self._saved_attributes is not None:
      termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved_attributes)


def _tensor_shape(value_info: onnx.ValueInfoProto) -> tuple[int | None, ...]:
  shape: list[int | None] = []
  for dim in value_info.type.tensor_type.shape.dim:
    shape.append(dim.dim_value if dim.HasField("dim_value") else None)
  return tuple(shape)


def _csv_strings(metadata: dict[str, str], key: str) -> tuple[str, ...]:
  value = metadata.get(key)
  if value is None:
    raise ValueError(f"ONNX metadata is missing {key!r}")
  return tuple(item.strip() for item in value.split(",") if item.strip())


def _csv_floats(metadata: dict[str, str], key: str) -> np.ndarray:
  try:
    values = [float(item) for item in _csv_strings(metadata, key)]
  except ValueError as error:
    raise ValueError(f"Invalid numeric ONNX metadata in {key!r}") from error
  return np.asarray(values, dtype=np.float32)


def _expand_per_joint(values: np.ndarray, name: str) -> np.ndarray:
  if values.shape == (1,):
    return np.full(ACTION_SIZE, values.item(), dtype=np.float32)
  if values.shape != (ACTION_SIZE,):
    raise ValueError(
      f"ONNX {name} has shape {values.shape}; expected scalar or ({ACTION_SIZE},)"
    )
  return values


class AmpOnnxPolicy:
  """Validated ONNX Runtime adapter for the AMP actor."""

  def __init__(self, policy_path: Path) -> None:
    if not policy_path.is_file():
      raise FileNotFoundError(f"Policy does not exist: {policy_path}")

    self.model = onnx.load(str(policy_path))
    onnx.checker.check_model(self.model)
    self.metadata = {item.key: item.value for item in self.model.metadata_props}

    inputs = {item.name: _tensor_shape(item) for item in self.model.graph.input}
    outputs = {item.name: _tensor_shape(item) for item in self.model.graph.output}
    if inputs.get("obs") != (1, OBSERVATION_SIZE):
      raise ValueError(
        f"AMP policy requires obs=(1,{OBSERVATION_SIZE}), got {inputs.get('obs')}"
      )
    if outputs.get("actions") != (1, ACTION_SIZE):
      raise ValueError(
        f"AMP policy requires actions=(1,{ACTION_SIZE}), "
        f"got {outputs.get('actions')}"
      )
    if self.metadata.get("task_type") != "velocity_commanded_locomotion_amp":
      raise ValueError(
        "The ONNX file is not tagged as a velocity-commanded AMP policy"
      )
    if self.metadata.get("actor_history_length") != str(HISTORY_LENGTH):
      raise ValueError("ONNX actor history length does not match this runner")
    if self.metadata.get("actor_frame_dim") != str(FRAME_SIZE):
      raise ValueError("ONNX actor frame dimension does not match this runner")

    try:
      import onnxruntime as ort

      self.session: Any = ort.InferenceSession(
        str(policy_path), providers=["CPUExecutionProvider"]
      )
      self.backend = f"onnxruntime {ort.__version__}"
    except ImportError:
      from onnx.reference import ReferenceEvaluator

      self.session = ReferenceEvaluator(self.model)
      self.backend = "onnx.reference.ReferenceEvaluator"

  def __call__(self, observation: np.ndarray) -> np.ndarray:
    action = self.session.run(
      ["actions"], {"obs": observation.reshape(1, -1).astype(np.float32)}
    )[0]
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (ACTION_SIZE,):
      raise RuntimeError(
        f"Policy returned action shape {action.shape}, expected ({ACTION_SIZE},)"
      )
    if not np.all(np.isfinite(action)):
      raise FloatingPointError("Policy returned a non-finite action")
    return action


class E1AmpSim2Sim:
  def __init__(
    self,
    policy_path: Path,
    xml_path: Path,
    sim_dt: float,
    decimation: int,
    initial_height: float,
  ) -> None:
    if not xml_path.is_file():
      raise FileNotFoundError(f"MJCF does not exist: {xml_path}")
    if sim_dt <= 0.0:
      raise ValueError("sim_dt must be positive")
    if decimation <= 0:
      raise ValueError("decimation must be positive")

    self.policy = AmpOnnxPolicy(policy_path)
    metadata = self.policy.metadata
    self.joint_names = _csv_strings(metadata, "joint_names")
    if len(self.joint_names) != ACTION_SIZE:
      raise ValueError(
        f"ONNX contains {len(self.joint_names)} joints, expected {ACTION_SIZE}"
      )
    self.default_joint_pos = _expand_per_joint(
      _csv_floats(metadata, "default_joint_pos"), "default_joint_pos"
    )
    self.stiffness = _expand_per_joint(
      _csv_floats(metadata, "joint_stiffness"), "joint_stiffness"
    )
    self.damping = _expand_per_joint(
      _csv_floats(metadata, "joint_damping"), "joint_damping"
    )
    self.action_scale = _expand_per_joint(
      _csv_floats(metadata, "action_scale"), "action_scale"
    )

    self.model = mujoco.MjModel.from_xml_path(str(xml_path))
    self.data = mujoco.MjData(self.model)
    self.model.opt.timestep = sim_dt
    self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    self.model.opt.iterations = 10
    self.model.opt.ls_iterations = 20

    qpos_ids: list[int] = []
    qvel_ids: list[int] = []
    actuator_ids: list[int] = []
    for joint_name in self.joint_names:
      joint_id = mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
      )
      if joint_id < 0:
        raise ValueError(f"MJCF is missing ONNX joint {joint_name!r}")
      qpos_ids.append(int(self.model.jnt_qposadr[joint_id]))
      qvel_ids.append(int(self.model.jnt_dofadr[joint_id]))
      matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == joint_id)
      if matches.size != 1:
        raise ValueError(
          f"Expected one actuator for {joint_name!r}, found {matches.size}"
        )
      actuator_ids.append(int(matches[0]))

    self.qpos_ids = np.asarray(qpos_ids, dtype=np.int32)
    self.qvel_ids = np.asarray(qvel_ids, dtype=np.int32)
    self.actuator_ids = np.asarray(actuator_ids, dtype=np.int32)
    self.torque_ranges = self.model.actuator_ctrlrange[self.actuator_ids].copy()

    self._require_torso_sensor("imu_upvector", 3)
    self._require_torso_sensor("imu_ang_vel", 3)

    self.decimation = decimation
    self.command = np.zeros(3, dtype=np.float32)
    self.previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)
    self.target_joint_pos = self.default_joint_pos.copy()
    self.history = np.zeros((HISTORY_LENGTH, FRAME_SIZE), dtype=np.float32)
    self._reset(initial_height)

  @property
  def control_dt(self) -> float:
    return float(self.model.opt.timestep * self.decimation)

  def _require_torso_sensor(self, name: str, dimension: int) -> None:
    sensor_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_SENSOR, name
    )
    if sensor_id < 0:
      raise ValueError(f"MJCF is missing torso IMU sensor {name!r}")
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
    body_name = mujoco.mj_id2name(
      self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
    )
    if body_name != "torso_link":
      raise ValueError(
        f"Sensor {name!r} is on {body_name!r}, expected 'torso_link'"
      )

  def _reset(self, initial_height: float) -> None:
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[0:3] = (0.0, 0.0, initial_height)
    self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    self.data.qpos[self.qpos_ids] = self.default_joint_pos
    self.data.qvel[:] = 0.0
    self.data.ctrl[:] = 0.0
    self.previous_action.fill(0.0)
    self.target_joint_pos = self.default_joint_pos.copy()
    mujoco.mj_forward(self.model, self.data)

    # CircularBuffer backfills its entire history with the first observation.
    initial_frame = self._actor_frame()
    self.history[:] = initial_frame

  def set_command(self, command: np.ndarray) -> None:
    command = np.asarray(command, dtype=np.float32)
    if command.shape != (3,) or not np.all(np.isfinite(command)):
      raise ValueError("Velocity command must contain three finite values")
    self.command[:] = command

  def _actor_frame(self) -> np.ndarray:
    # mjlab projected_gravity_from_sensor negates this framezaxis up-vector.
    projected_gravity = -self.data.sensor("imu_upvector").data.copy()
    angular_velocity = self.data.sensor("imu_ang_vel").data.copy()
    joint_pos = self.data.qpos[self.qpos_ids]
    joint_vel = self.data.qvel[self.qvel_ids]
    frame = np.concatenate(
      (
        ANGULAR_VELOCITY_SCALE * angular_velocity,
        projected_gravity,
        self.command,
        joint_pos - self.default_joint_pos,
        JOINT_VELOCITY_SCALE * joint_vel,
        self.previous_action,
      )
    ).astype(np.float32)
    if frame.shape != (FRAME_SIZE,):
      raise RuntimeError(
        f"Constructed frame shape {frame.shape}, expected ({FRAME_SIZE},)"
      )
    if not np.all(np.isfinite(frame)):
      raise FloatingPointError("Actor observation contains non-finite values")
    return np.clip(frame, *OBSERVATION_CLIP)

  def update_policy(self) -> np.ndarray:
    mujoco.mj_forward(self.model, self.data)
    frame = self._actor_frame()
    self.history[:-1] = self.history[1:]
    self.history[-1] = frame
    observation = self.history.reshape(-1)
    action = self.policy(observation)
    action = np.clip(action, *ACTION_CLIP)
    self.previous_action = action.copy()
    self.target_joint_pos = (
      self.default_joint_pos + self.action_scale * self.previous_action
    )
    return observation

  def physics_step(self) -> np.ndarray:
    joint_pos = self.data.qpos[self.qpos_ids]
    joint_vel = self.data.qvel[self.qvel_ids]
    torque = self.stiffness * (self.target_joint_pos - joint_pos)
    torque -= self.damping * joint_vel
    torque = np.clip(
      torque, self.torque_ranges[:, 0], self.torque_ranges[:, 1]
    )
    if not np.all(np.isfinite(torque)):
      raise FloatingPointError("PD controller produced a non-finite torque")
    self.data.ctrl[self.actuator_ids] = torque
    mujoco.mj_step(self.model, self.data)
    return torque

def _gamepad_command(gamepad: Any) -> np.ndarray:
  pad_x, pad_y, pad_yaw = gamepad.get_commands()
  x_scale = MAX_FORWARD_VELOCITY if pad_x >= 0.0 else MAX_BACKWARD_VELOCITY
  return np.asarray(
    (
      pad_x * x_scale,
      np.clip(pad_y, -MAX_LATERAL_VELOCITY, MAX_LATERAL_VELOCITY),
      pad_yaw * MAX_YAW_VELOCITY,
    ),
    dtype=np.float32,
  )


def run(args: argparse.Namespace) -> None:
  runner = E1AmpSim2Sim(
    policy_path=args.policy.resolve(),
    xml_path=args.xml.resolve(),
    sim_dt=args.sim_dt,
    decimation=args.decimation,
    initial_height=args.initial_height,
  )
  fixed_command = np.asarray(args.command, dtype=np.float32)
  runner.set_command(fixed_command)
  keyboard_command = KeyboardCommand() if args.keyboard else None
  if args.gamepad and not args.keyboard:
    try:
      from sim2sim.gamepad_controller import GamepadController
    except ModuleNotFoundError:
      from gamepad_controller import GamepadController

    gamepad = GamepadController(deadzone=args.gamepad_deadzone)
  else:
    gamepad = None

  torso_id = mujoco.mj_name2id(
    runner.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
  )
  if args.duration is None:
    physics_steps = itertools.count()
  else:
    physics_steps = range(
      math.ceil(args.duration / runner.model.opt.timestep)
    )
  print("[INFO] E1 21-DOF mjlab AMP sim2sim")
  print(f"[INFO] Policy: {args.policy.resolve()}")
  print(f"[INFO] MJCF: {args.xml.resolve()}")
  print(f"[INFO] Inference backend: {runner.policy.backend}")
  print(f"[INFO] Joint order: {', '.join(runner.joint_names)}")
  print(
    f"[INFO] Observation/action: {OBSERVATION_SIZE}/{ACTION_SIZE}, "
    f"control={1.0 / runner.control_dt:.1f} Hz, "
    f"physics={1.0 / args.sim_dt:.1f} Hz"
  )
  if keyboard_command is not None:
    print(
      "[INFO] Terminal keyboard: W/S=forward/back, A/D=lateral, "
      "J/L=yaw, R=zero command, Q=exit"
    )
  elif gamepad is None:
    print(
      f"[INFO] Fixed command: vx={fixed_command[0]:+.2f}, "
      f"vy={fixed_command[1]:+.2f}, wz={fixed_command[2]:+.2f}"
    )
  else:
    print("[INFO] Gamepad: L stick=XY, R stick=Yaw, LT+B=exit")

  if args.headless:
    viewer_context: Any = contextlib.nullcontext(None)
  else:
    from mujoco import viewer as mujoco_viewer

    # Keep viewer shortcuts separate from terminal command input.
    viewer_context = mujoco_viewer.launch_passive(runner.model, runner.data)

  last_log_time = -float("inf")
  last_torque = np.zeros(ACTION_SIZE, dtype=np.float32)
  wall_start = time.perf_counter()
  keyboard_context: Any = (
    TerminalKeyboard(keyboard_command)
    if keyboard_command is not None
    else contextlib.nullcontext()
  )

  with keyboard_context:
    with viewer_context as viewer:
      if viewer is not None:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = torso_id
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 120.0
        viewer.cam.elevation = -5.0

      for physics_step in physics_steps:
        if keyboard_command is not None:
          keyboard_context.poll()
          if not keyboard_command.running:
            print("[INFO] Q: exit")
            break
        step_start = time.perf_counter()
        if physics_step % runner.decimation == 0:
          if keyboard_command is not None:
            runner.set_command(keyboard_command.get_command())
          elif gamepad is not None:
            runner.set_command(_gamepad_command(gamepad))
            if gamepad.get_button_b() and gamepad.get_button_lt():
              print("[INFO] LT+B: exit")
              break
          runner.update_policy()

        last_torque = runner.physics_step()

        if viewer is not None:
          if not viewer.is_running():
            break
          viewer.sync()

        if runner.data.time - last_log_time >= args.log_interval:
          print(
            f"\r[SIM] t={runner.data.time:7.2f}s "
            f"cmd=({runner.command[0]:+.2f},{runner.command[1]:+.2f},"
            f"{runner.command[2]:+.2f}) height={runner.data.qpos[2]:.3f} "
            f"|tau|max={np.max(np.abs(last_torque)):.2f}    ",
            end="",
            flush=True,
          )
          last_log_time = float(runner.data.time)

        if args.realtime and not args.headless:
          remaining = runner.model.opt.timestep - (time.perf_counter() - step_start)
          if remaining > 0.0:
            time.sleep(remaining)

  print()
  elapsed = time.perf_counter() - wall_start
  simulated = float(runner.data.time)
  print(
    f"[INFO] Finished: simulated={simulated:.3f}s, wall={elapsed:.3f}s, "
    f"real-time factor={simulated / max(elapsed, 1.0e-9):.2f}x"
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run an E1 21-DOF mjlab AMP ONNX policy in native MuJoCo."
  )
  parser.add_argument("--policy", type=Path, required=True)
  parser.add_argument(
    "--xml",
    type=Path,
    default=DEFAULT_XML_PATH,
    help="Standalone E1 MuJoCo XML.",
  )
  parser.add_argument(
    "--command",
    type=float,
    nargs=3,
    metavar=("VX", "VY", "WZ"),
    default=(0.5, 0.0, 0.0),
    help="Fixed body-frame command used when --no-gamepad is selected.",
  )
  parser.add_argument(
    "--gamepad",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use the gamepad instead of the fixed --command.",
  )
  parser.add_argument(
    "--keyboard",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Read accumulated commands from the launching terminal keyboard.",
  )
  parser.add_argument("--gamepad-deadzone", type=float, default=0.15)
  parser.add_argument(
    "--duration",
    type=float,
    default=None,
    help="Optional simulated duration in seconds; omitted means run forever.",
  )
  parser.add_argument("--sim-dt", type=float, default=0.005)
  parser.add_argument("--decimation", type=int, default=4)
  parser.add_argument("--initial-height", type=float, default=0.75)
  parser.add_argument(
    "--realtime",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Pace GUI simulation to wall-clock time.",
  )
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--log-interval", type=float, default=1.0)
  args = parser.parse_args()
  if args.duration is not None and args.duration <= 0.0:
    parser.error("--duration must be positive")
  if args.sim_dt <= 0.0:
    parser.error("--sim-dt must be positive")
  if args.decimation <= 0:
    parser.error("--decimation must be positive")
  if args.initial_height <= 0.0:
    parser.error("--initial-height must be positive")
  if args.log_interval <= 0.0:
    parser.error("--log-interval must be positive")
  return args


if __name__ == "__main__":
  run(parse_args())
