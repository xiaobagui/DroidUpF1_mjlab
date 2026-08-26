"""mjlab configuration for the F1 1-robot model."""

from copy import deepcopy
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


F1_1_XML = Path(__file__).resolve().parent / "mjcf" / "f1_1.xml"
assert F1_1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  """Load the standalone F1 MJCF as a robot spec."""
  spec = mujoco.MjSpec.from_file(str(F1_1_XML))
  spec.meshdir = str((F1_1_XML.parent.parent / "meshes").resolve())

  for camera in list(spec.cameras):
    spec.delete(camera)
  for light in list(spec.lights):
    spec.delete(light)

  return spec


# Effort limits and gains are aligned with the existing F1 scene XML.
F1_HIP_PITCH_EFFORT = 75.0
F1_HIP_ROLL_EFFORT = 75.0
F1_HIP_YAW_EFFORT = 75.0
F1_KNEE_EFFORT = 75.0
F1_ANKLE_PITCH_EFFORT = 75.0
F1_ANKLE_ROLL_EFFORT = 75.0
F1_WAIST_YAW_EFFORT = 50.0
F1_WAIST_ROLL_EFFORT = 50.0
F1_SHOULDER_PITCH_EFFORT = 25.0
F1_SHOULDER_ROLL_EFFORT = 25.0
F1_SHOULDER_YAW_EFFORT = 25.0
F1_ELBOW_EFFORT = 25.0
F1_WRIST_ROLL_EFFORT = 25.0
F1_WRIST_YAW_EFFORT = 5.0
F1_WRIST_PITCH_EFFORT = 5.0
F1_ARMATURE = 0.01


def _position_actuator(
  target_names_expr: tuple[str, ...],
  stiffness: float,
  damping: float,
  effort_limit: float,
) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=stiffness,
    damping=damping,
    effort_limit=effort_limit,
    armature=F1_ARMATURE,
  )


F1_HIP_PITCH_ACTUATOR = _position_actuator(
  (".*_hip_pitch_joint",), 150.0, 2.0, F1_HIP_PITCH_EFFORT
)
F1_HIP_ROLL_ACTUATOR = _position_actuator(
  (".*_hip_roll_joint",), 150.0, 2.0, F1_HIP_ROLL_EFFORT
)
F1_HIP_YAW_ACTUATOR = _position_actuator(
  (".*_hip_yaw_joint",), 150.0, 2.0, F1_HIP_YAW_EFFORT
)
F1_KNEE_ACTUATOR = _position_actuator(
  (".*_knee_joint",), 150.0, 2.0, F1_KNEE_EFFORT
)
F1_ANKLE_PITCH_ACTUATOR = _position_actuator(
  (".*_ankle_pitch_joint",), 30.0, 2.0, F1_ANKLE_PITCH_EFFORT
)
F1_ANKLE_ROLL_ACTUATOR = _position_actuator(
  (".*_ankle_roll_joint",), 30.0, 2.0, F1_ANKLE_ROLL_EFFORT
)
F1_WAIST_YAW_ACTUATOR = _position_actuator(
  ("waist_yaw_joint",), 150.0, 2.0, F1_WAIST_YAW_EFFORT
)
F1_WAIST_ROLL_ACTUATOR = _position_actuator(
  ("waist_roll_joint",), 150.0, 2.0, F1_WAIST_ROLL_EFFORT
)
F1_SHOULDER_PITCH_ACTUATOR = _position_actuator(
  (".*_shoulder_pitch_joint",), 30.0, 2.0, F1_SHOULDER_PITCH_EFFORT
)
F1_SHOULDER_ROLL_ACTUATOR = _position_actuator(
  (".*_shoulder_roll_joint",), 30.0, 2.0, F1_SHOULDER_ROLL_EFFORT
)
F1_SHOULDER_YAW_ACTUATOR = _position_actuator(
  (".*_shoulder_yaw_joint",), 30.0, 2.0, F1_SHOULDER_YAW_EFFORT
)
F1_ELBOW_ACTUATOR = _position_actuator(
  (".*_elbow_joint",), 30.0, 2.0, F1_ELBOW_EFFORT
)
F1_WRIST_ROLL_ACTUATOR = _position_actuator(
  (".*_wrist_roll_joint",), 30.0, 2.0, F1_WRIST_ROLL_EFFORT
)
F1_WRIST_YAW_ACTUATOR = _position_actuator(
  (".*_wrist_yaw_joint",), 20.0, 1.0, F1_WRIST_YAW_EFFORT
)
F1_WRIST_PITCH_ACTUATOR = _position_actuator(
  (".*_wrist_pitch_joint",), 20.0, 1.0, F1_WRIST_PITCH_EFFORT
)


F1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    F1_HIP_PITCH_ACTUATOR,
    F1_HIP_ROLL_ACTUATOR,
    F1_HIP_YAW_ACTUATOR,
    F1_KNEE_ACTUATOR,
    F1_ANKLE_PITCH_ACTUATOR,
    F1_ANKLE_ROLL_ACTUATOR,
    F1_WAIST_YAW_ACTUATOR,
    F1_WAIST_ROLL_ACTUATOR,
    F1_SHOULDER_PITCH_ACTUATOR,
    F1_SHOULDER_ROLL_ACTUATOR,
    F1_SHOULDER_YAW_ACTUATOR,
    F1_ELBOW_ACTUATOR,
    F1_WRIST_ROLL_ACTUATOR,
    F1_WRIST_YAW_ACTUATOR,
    F1_WRIST_PITCH_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)


HOME_KEY_FRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.873),
  joint_pos={
    ".*_hip_pitch_joint": -0.10,
    ".*_knee_joint": 0.23,
    ".*_ankle_pitch_joint": -0.13,
    ".*_shoulder_roll_joint": 0.0,
    ".*_shoulder_pitch_joint": 0.0,
    ".*_elbow_joint": 1.0,
  },
  joint_vel={".*": 0.0},
)


FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)


FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)


FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)


def get_f1_robot_cfg(
  action_delay_range: tuple[int, int] = (0, 0),
) -> EntityCfg:
  """Return a fresh mjlab entity configuration for the F1 robot."""
  delay_min_lag, delay_max_lag = action_delay_range
  if delay_min_lag < 0 or delay_max_lag < delay_min_lag:
    raise ValueError(f"Invalid action_delay_range={action_delay_range}")

  articulation = deepcopy(F1_ARTICULATION)
  for actuator in articulation.actuators:
    actuator.delay_min_lag = delay_min_lag
    actuator.delay_max_lag = delay_max_lag
    actuator.delay_hold_prob = 0.9 if delay_max_lag > 0 else 0.0
    actuator.delay_update_period = 4 if delay_max_lag > 0 else 0

  return EntityCfg(
    init_state=HOME_KEY_FRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=articulation,
  )


F1_ACTION_SCALE: dict[str, float] = {}
for actuator in F1_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  assert actuator.effort_limit is not None
  for name_expr in actuator.target_names_expr:
    F1_ACTION_SCALE[name_expr] = 0.25 * actuator.effort_limit / actuator.stiffness


# Backward-compatible aliases for older code.
X3_HIP_PITCH_ACTUATOR = F1_HIP_PITCH_ACTUATOR
X3_HIP_ROLL_ACTUATOR = F1_HIP_ROLL_ACTUATOR
X3_HIP_YAW_ACTUATOR = F1_HIP_YAW_ACTUATOR
X3_KNEE_ACTUATOR = F1_KNEE_ACTUATOR
X3_ANKLE_PITCH_ACTUATOR = F1_ANKLE_PITCH_ACTUATOR
X3_ANKLE_ROLL_ACTUATOR = F1_ANKLE_ROLL_ACTUATOR
X3_WAIST_YAW_ACTUATOR = F1_WAIST_YAW_ACTUATOR
X3_WAIST_ROLL_ACTUATOR = F1_WAIST_ROLL_ACTUATOR
X3_SHOULDER_PITCH_ACTUATOR = F1_SHOULDER_PITCH_ACTUATOR
X3_SHOULDER_ROLL_ACTUATOR = F1_SHOULDER_ROLL_ACTUATOR
X3_SHOULDER_YAW_ACTUATOR = F1_SHOULDER_YAW_ACTUATOR
X3_ELBOW_ACTUATOR = F1_ELBOW_ACTUATOR
X3_WRIST_ROLL_ACTUATOR = F1_WRIST_ROLL_ACTUATOR
X3_WRIST_YAW_ACTUATOR = F1_WRIST_YAW_ACTUATOR
X3_WRIST_PITCH_ACTUATOR = F1_WRIST_PITCH_ACTUATOR
X3_ARTICULATION = F1_ARTICULATION
X3_ACTION_SCALE = F1_ACTION_SCALE


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_f1_robot_cfg())
  viewer.launch(robot.spec.compile())
