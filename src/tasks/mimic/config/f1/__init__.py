"""Register the F1 motion-tracking tasks."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.mimic.rl import MimicOnPolicyRunner

from .rl_cfg import f1_mimic_ppo_runner_cfg
from .tracking_env_cfg import f1_flat_mimic_env_cfg


register_mjlab_task(
  task_id="Tracking-Flat-F1",
  env_cfg=f1_flat_mimic_env_cfg(),
  play_env_cfg=f1_flat_mimic_env_cfg(play=True),
  rl_cfg=f1_mimic_ppo_runner_cfg(),
  runner_cls=MimicOnPolicyRunner,
)

register_mjlab_task(
  task_id="Tracking-Flat-F1-No-State-Estimation",
  env_cfg=f1_flat_mimic_env_cfg(has_state_estimation=False),
  play_env_cfg=f1_flat_mimic_env_cfg(
  has_state_estimation=False,
  play=True,
  ),
  rl_cfg=f1_mimic_ppo_runner_cfg(),
  runner_cls=MimicOnPolicyRunner,
)
