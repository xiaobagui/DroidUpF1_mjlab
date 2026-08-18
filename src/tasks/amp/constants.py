"""Canonical XML ordering shared by E1 AMP observations, data, and symmetry."""

# Non-free joints in their exact E1_21dof.xml definition order.  Policy
# observations, actions, expert NPZ files, the discriminator, and symmetry all
# use this one ordering without a boundary permutation.
MJLAB_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
)

AMP_KEY_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_shoulder_yaw_link",
  "right_shoulder_yaw_link",
  "left_elbow_link",
  "right_elbow_link",
)

NUM_JOINTS = len(MJLAB_JOINT_NAMES)
ACTOR_FRAME_DIM = 3 + 3 + 3 + 3 * NUM_JOINTS
CRITIC_FRAME_DIM = ACTOR_FRAME_DIM + 3 + 2
AMP_KEY_BODY_POS_DIM = 3 * len(AMP_KEY_BODY_NAMES)
AMP_KEY_BODY_ORI_DIM = 6 * len(AMP_KEY_BODY_NAMES)
AMP_OBS_DIM = (
  3
  + 3
  + 2 * NUM_JOINTS
  + AMP_KEY_BODY_POS_DIM
  + AMP_KEY_BODY_ORI_DIM
)
# A one-hot conditioning vector is appended internally for AMP training. The
# four classes are command/style categories and are not actor observations.
AMP_LABEL_NAMES = ("walk", "run", "turn", "side")
AMP_LABEL_DIM = len(AMP_LABEL_NAMES)
AMP_DISCRIMINATOR_STATE_DIM = AMP_OBS_DIM + AMP_LABEL_DIM
