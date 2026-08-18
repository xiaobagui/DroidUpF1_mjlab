"""Register the E1 walk-and-run AMP locomotion task."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.amp.rl import AmpOnPolicyRunner

from .env_cfg import e1_21dof_walk_amp_env_cfg
from .rl_cfg import e1_21dof_walk_amp_runner_cfg


register_mjlab_task(
  task_id="AMP-Walk-Flat-E1-21DOF",
  env_cfg=e1_21dof_walk_amp_env_cfg(),
  play_env_cfg=e1_21dof_walk_amp_env_cfg(play=True),
  rl_cfg=e1_21dof_walk_amp_runner_cfg(),
  runner_cls=AmpOnPolicyRunner,
)
