"""使用 RSL-RL 训练 RL Agent 的启动脚本。"""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import tyro  # 用于命令行参数解析的库

# 导入 mjlab 环境与 RL 封装模块
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from src.tasks.mimic.mdp import MotionCommandCfg


# 训练配置数据类
@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg   # 环境配置对象
  # 任务注册表提供独立于机器人的 Runner 参数类，使用 Any 保持动态性
  agent: Any                  # RL 算法/Agent 配置对象 (如 F1OnPolicyRunnerCfg)
  video: bool = False         # 是否启用训练过程中的视频录制
  video_length: int = 200     # 录制视频的步数长度 (Frames/Steps)
  video_interval: int = 2000  # 视频录制间隔（每 2000 步录制一次）
  enable_nan_guard: bool = False # 是否开启仿真数值异常 (NaN) 监控保护
  log_root: str = "logs/rsl_rl"
  """实验日志保存的根目录。"""
  torchrunx_log_dir: str | None = None  # 多卡训练框架 torchrunx 的日志输出路径
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0]) # 指定使用的 GPU 设备 ID

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    """根据注册的 task_id 加载默认的环境和算法配置。"""
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  """单进程训练的主函数，由单卡直接调用或多卡 torchrunx 分布式调起。"""
  
  # 1. 环境变量与设备/种子初始化
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    # 从分布式环境变量获取当前进程的 local_rank 和 global rank
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # 将 MuJoCo EGL 渲染设备 ID 设置为与当前 CUDA GPU 相配
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # 给不同进程设置不同的随机种子，以增加探索多样性
    seed = cfg.agent.seed + rank

  # 配置 PyTorch 的计算后端（例如使能 TensorCore、TF32 等加速选项）
  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # 2. 检查是否为动作模仿/跟踪任务，并校验参考动作数据文件是否存在
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")
    else:
      raise ValueError(
        "Tracking requires a local motion file. Set "
        "--env.commands.motion.motion-file /path/to/motion.npz"
      )

  # 3. 开启数值 NaN 保护检查
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  # 4. 创建基础仿真环境
  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # 上级实验总目录

  # 5. 如果设置了 resume，获取加载检查点 (.pt) 的文件路径
  resume_path: Path | None = None
  if cfg.agent.resume:
    resume_path = get_checkpoint_path(
      log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
    )

  # 6. 仅在主进程 (rank 0) 上挂载 VideoRecorder 录制训练视频（防止多卡写入文件冲突）
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  # 7. 使用 RSL-RL 的向量化环境包装器包覆原生环境
  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  # 加载该任务特定的 Runner 类（若无指定则使用默认的 MjlabOnPolicyRunner）
  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  # 8. 仅在主进程保存 YAML 超参数配置文件 (环境配置 env.yaml 和算法配置 agent.yaml)
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  # 9. 实例化 RSL-RL Runner 训练器
  runner = runner_cls(env, agent_cfg, str(log_dir), device)

  # 记录当前代码 Git 信息到日志中
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # 10. 开始训练循环
  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  # 训练完成后关闭环境
  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  """启动训练任务的统一入口，负责创建目录、解析 GPU 配置并分配单/多卡启动流程。"""
  args = args or TrainConfig.from_task(task_id)

  # 创建带时间戳的实验日志保存目录
  log_root_path = (Path(args.log_root) / args.agent.experiment_name).resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # 根据命令行传参及系统环境选择使用的 GPU 列表及数量
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # 显式设置 CUDA_VISIBLE_DEVICES 环境变量
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))

  # 根据可见 GPU 数量选择启动方式
  if num_gpus <= 1:
    # CPU 或单卡 GPU：直接调用 run_train 运行
    run_train(task_id, args, log_dir)
  else:
    # 多 GPU：通过 torchrunx 启动多进程分布式训练
    import torchrunx

    logging.basicConfig(level=logging.INFO)

    # 配置 torchrunx 的日志输出目录
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    # 使用 torchrunx.Launcher 启动多进程并发调用 run_train 函数
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # 让 RSL-RL 内部初始化 PyTorch ProcessGroup
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",), # 透传 MuJoCo 环境变量
    ).run(run_train, task_id, args, log_dir)


def main():
  """CLI 命令行入口。"""
  maybe_print_top_level_help("train")

  # 导入所有任务模块充实任务注册表
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  # 第一步：解析用户指定的任务名称（如 f1_flat_mimic）
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # 第二步：根据剩余命令行参数覆盖生成最终的 TrainConfig 参数结构
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  # 第三步：正式启动训练
  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()