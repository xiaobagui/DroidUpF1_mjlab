"""Register the F1 walk-and-run AMP locomotion task."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.amp.rl import AmpOnPolicyRunner

from .env_cfg import f1_walk_amp_env_cfg
from .rl_cfg import f1_walk_amp_runner_cfg


register_mjlab_task(
  task_id="AMP-Walk-Flat-F1",
  env_cfg=f1_walk_amp_env_cfg(),
  play_env_cfg=f1_walk_amp_env_cfg(play=True),
  rl_cfg=f1_walk_amp_runner_cfg(),
  runner_cls=AmpOnPolicyRunner,
)
