"""Play external mjlab motion-tracking tasks."""

import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago
from src.tasks.mimic.mdp import MotionCommandCfg


@dataclass(frozen=True)
class MimicPlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  training_env: bool = False
  """Use the randomized training environment instead of the deterministic play cfg."""
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions."""


def run_play(task_id: str, cfg: MimicPlayConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=not cfg.training_env)
  agent_cfg = load_rl_cfg(task_id)
  dummy_mode = cfg.agent in {"zero", "random"}
  trained_mode = not dummy_mode

  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  motion_cfg = env_cfg.commands.get("motion")
  if not isinstance(motion_cfg, MotionCommandCfg):
    raise TypeError(f"Task {task_id!r} does not contain a mimic motion command.")
  if cfg.motion_file is not None:
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.is_file():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cfg.motion_file = str(motion_path)
  elif not motion_cfg.motion_file or not Path(motion_cfg.motion_file).is_file():
    raise FileNotFoundError(
      "Mimic playback requires a local motion file. Set --motion-file /path/motion.npz"
    )
  print(f"[INFO]: Using local motion file: {motion_cfg.motion_file}")

  resume_path: Path | None = None
  log_dir: Path | None = None
  if trained_mode:
    if cfg.checkpoint_file is None:
      raise ValueError("Trained playback requires --checkpoint-file /path/model.pt")
    resume_path = Path(cfg.checkpoint_file).expanduser().resolve()
    if not resume_path.is_file():
      raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    log_dir = resume_path.parent
    print(f"[INFO]: Loading checkpoint: {resume_path.name}")

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  elif cfg.training_env:
    env_cfg.scene.num_envs = 4
  if cfg.training_env:
    print(
      "[INFO]: Using training resets, observation noise, delays, and "
      f"domain randomization ({env_cfg.scene.num_envs} envs)"
    )
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (trained_mode and cfg.video) else None
  if cfg.video and dummy_mode:
    print("[WARN] Video recording is disabled for dummy agents.")
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if trained_mode and cfg.video:
    assert log_dir is not None
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  checkpoint_manager: CheckpointManager | None = None
  if dummy_mode:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape

    def policy(obs) -> torch.Tensor:
      del obs
      if cfg.agent == "zero":
        return torch.zeros(action_shape, device=env.unwrapped.device)
      return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

  else:
    assert resume_path is not None
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

    def reload_policy(path: str):
      runner.load(path, load_cfg={"actor": True}, strict=True, map_location=device)
      return runner.get_inference_policy(device=device)

    checkpoint_dir = resume_path.parent

    def fetch_available_local() -> list[tuple[str, str]]:
      now = time.time()
      entries: list[tuple[str, str, int]] = []
      for checkpoint in sorted(checkpoint_dir.glob("*.pt")):
        try:
          step = int(checkpoint.stem.split("_")[1])
        except (IndexError, ValueError):
          step = 0
        age = format_time_ago(int(now - checkpoint.stat().st_mtime))
        entries.append(
          (checkpoint.name, age, step)
        )
      entries.sort(key=lambda item: item[2])
      return [(name, age) for name, age, _ in entries]

    checkpoint_manager = CheckpointManager(
      current_name=resume_path.name,
      fetch_available=fetch_available_local,
      load_checkpoint=lambda name: reload_policy(str(checkpoint_dir / name)),
    )

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = cfg.viewer
  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  else:
    ViserPlayViewer(env, policy, checkpoint_manager=checkpoint_manager).run()
  env.close()


def main() -> None:
  maybe_print_top_level_help("play_mimic")
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  mimic_tasks = [task for task in list_tasks() if task.startswith("Tracking-")]
  if not mimic_tasks:
    raise RuntimeError("No external mimic tasks are registered.")
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(mimic_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    MimicPlayConfig,
    args=remaining_args,
    default=MimicPlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
