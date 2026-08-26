"""Self-contained RSL-RL 5.4.2 configuration for F1 locomotion AMP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlModelCfg

from .constants import (
  F1_AMP_DISCRIMINATOR_STATE_DIM,
  F1_AMP_KEY_BODY_NAMES,
  F1_AMP_LABEL_NAMES,
  F1_AMP_OBS_DIM,
  F1_JOINT_NAMES,
)


@dataclass
class AmpPpoAlgorithmCfg:
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  learning_rate: float = 1.0e-3
  schedule: str = "adaptive"
  gamma: float = 0.99
  lam: float = 0.95
  entropy_coef: float = 0.01
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0
  value_loss_coef: float = 1.0
  use_clipped_value_loss: bool = True
  clip_param: float = 0.2
  normalize_advantage_per_mini_batch: bool = False
  optimizer: str = "adam"
  share_cnn_encoders: bool = False
  rnd_cfg: None = None
  symmetry_cfg: None = None
  amp_joint_names: tuple[str, ...] = F1_JOINT_NAMES
  amp_key_body_names: tuple[str, ...] = F1_AMP_KEY_BODY_NAMES
  amp_label_names: tuple[str, ...] = F1_AMP_LABEL_NAMES
  amp_obs_dim: int = F1_AMP_OBS_DIM
  amp_discriminator_state_dim: int = F1_AMP_DISCRIMINATOR_STATE_DIM
  amp_motion_files: tuple[str, ...] = (
    "dataset/f1/amp/walk.npz",
    "dataset/f1/amp/run.npz",
    "dataset/f1/amp/run_mirror.npz",
    "dataset/f1/amp/turn_l.npz",
    "dataset/f1/amp/turn_r.npz",
    "dataset/f1/amp/side_l.npz",
    "dataset/f1/amp/side_r.npz",
  )
  amp_motion_velocity_threshold: float = 0.8
  amp_motion_weights: tuple[float, ...] = (1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
  amp_motion_labels: tuple[str, ...] = (
    "walk", "run", "run", "turn", "turn", "side", "side"
  )
  amp_reward_coefficient: float = 0.4
  amp_task_reward_lerp: float = 0.7
  amp_discriminator_hidden_dims: tuple[int, ...] = (1024, 512, 256)
  amp_replay_buffer_size: int = 100_000
  amp_preload_transitions: int = 500_000
  amp_transition_dt: float = 0.02
  amp_gradient_penalty: float = 10.0
  minimum_action_std: tuple[float, ...] = (0.05,) * len(F1_JOINT_NAMES)
  class_name: str = "src.tasks.amp.rl.algorithm:AmpPPO"


@dataclass
class F1WalkAmpRunnerCfg:
  seed: int = 42
  num_steps_per_env: int = 24
  max_iterations: int = 50_000
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)}
  )
  save_interval: int = 100
  experiment_name: str = "f1_walk_run_amp"
  run_name: str = ""
  logger: Literal["tensorboard"] = "tensorboard"
  resume: bool = False
  load_run: str = ".*"
  load_checkpoint: str = "model_.*.pt"
  clip_actions: float | None = None
  upload_model: bool = False
  class_name: str = "OnPolicyRunner"
  actor: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "rsl_rl.modules:GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
        "std_range": (0.05, 2.0),
      },
      class_name="rsl_rl.models:MLPModel",
    )
  )
  critic: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      class_name="rsl_rl.models:MLPModel",
    )
  )
  algorithm: AmpPpoAlgorithmCfg = field(default_factory=AmpPpoAlgorithmCfg)


def f1_walk_amp_runner_cfg() -> F1WalkAmpRunnerCfg:
  return F1WalkAmpRunnerCfg()
