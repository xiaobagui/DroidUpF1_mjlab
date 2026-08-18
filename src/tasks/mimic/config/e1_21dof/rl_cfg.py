"""Self-contained RSL-RL configuration for E1 21-DOF tracking."""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg


@dataclass
class E121DofOnPolicyRunnerCfg:
  """TensorBoard-only runner configuration maintained by the E1 task."""

  seed: int = 42
  num_steps_per_env: int = 24
  max_iterations: int = 30_000
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)}
  )
  save_interval: int = 500
  experiment_name: str = "external_e1_21dof_tracking"
  run_name: str = ""
  logger: Literal["tensorboard"] = "tensorboard"
  resume: bool = False
  load_run: str = ".*"
  load_checkpoint: str = "model_.*.pt"
  clip_actions: float | None = None
  class_name: str = "OnPolicyRunner"
  actor: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    )
  )
  critic: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    )
  )
  algorithm: RslRlPpoAlgorithmCfg = field(
    default_factory=lambda: RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    )
  )


def e1_21dof_mimic_ppo_runner_cfg() -> E121DofOnPolicyRunnerCfg:
  """Return a fresh E1 PPO configuration."""
  return E121DofOnPolicyRunnerCfg()
