"""F1-specific ordering and dimensions for AMP locomotion tasks."""

F1_JOINT_NAMES = (
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
  "waist_roll_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
)

F1_AMP_KEY_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_shoulder_yaw_link",
  "right_shoulder_yaw_link",
  "left_elbow_link",
  "right_elbow_link",
  "left_wrist_pitch_link",
  "right_wrist_pitch_link",
)

F1_NUM_JOINTS = len(F1_JOINT_NAMES)
F1_ACTOR_FRAME_DIM = 3 + 3 + 3 + 3 * F1_NUM_JOINTS
F1_CRITIC_FRAME_DIM = F1_ACTOR_FRAME_DIM + 3 + 2
F1_AMP_OBS_DIM = 6 + 2 * F1_NUM_JOINTS + 9 * len(F1_AMP_KEY_BODY_NAMES)
F1_AMP_LABEL_NAMES = ("walk", "run", "turn", "side")
F1_AMP_LABEL_DIM = len(F1_AMP_LABEL_NAMES)
F1_AMP_DISCRIMINATOR_STATE_DIM = F1_AMP_OBS_DIM + F1_AMP_LABEL_DIM
