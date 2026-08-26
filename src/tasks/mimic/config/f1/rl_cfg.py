"""用于 F1 动作跟踪任务的独立 RSL-RL 训练配置。"""

from dataclasses import dataclass, field  # 导入 dataclass 用于声明配置数据类
from typing import Literal                # 导入 类型检查工具

# 导入 mjlab 封装的 RSL-RL 模型与 PPO 算法配置类
from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg


@dataclass
class F1OnPolicyRunnerCfg:
  """针对 F1 跟踪任务维护的仅使用 TensorBoard 记录日志的 On-Policy Runner 配置。"""

  # 1. 基础运行参数
  seed: int = 42                  # 随机种子，保证实验可复现
  num_steps_per_env: int = 24     # 每个环境在单次 PPO 迭代中采样的步数 (Rollout Horizon)
  max_iterations: int = 30_000   # 最大的训练迭代次数 (Total Iterations)
  
  # 观测组映射：定义策略网络 (Actor) 和价值网络 (Critic) 分别使用哪个环境观测组
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)}
  )
  
  save_interval: int = 500       # 模型检查点 (Checkpoint) 的保存间隔（每 500 次迭代保存一次）
  experiment_name: str = "external_f1_tracking"  # 实验名称（决定日志保存目录）
  run_name: str = ""              # 具体运行的名称后缀（为空时通常以时间戳自动命名）
  logger: Literal["tensorboard"] = "tensorboard"  # 使用 TensorBoard 记录训练曲线与指标
  
  # 恢复训练 / 权重加载配置
  resume: bool = False            # 是否从之前保存的检查点恢复训练
  load_run: str = ".*"            # 加载检查点时的运行文件夹匹配正则
  load_checkpoint: str = "model_.*.pt" # 加载模型文件的正则（如匹配最新的模型权重）
  
  clip_actions: float | None = None    # 动作输出截断边界（None 表示不显式截断，交由策略控制）
  class_name: str = "OnPolicyRunner"   # RSL-RL 对应的底层 Runner 类名

  # 2. Actor (策略网络) 模型结构配置
  actor: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),  # 3 层 MLP 隐藏层神经元数量
      activation="elu",            # 激活函数选用 ELU
      obs_normalization=True,      # 开启 Actor 输入观测数据的在线均值/标准差归一化
      distribution_cfg={
        "class_name": "GaussianDistribution",  # 连续动作空间，输出高斯分布
        "init_std": 1.0,                       # 策略初始高斯标准差 (Standard Deviation)
        "std_type": "scalar",                  # 可学习的标准差类型（标量/所有动作维度共享初始标准差）
      },
    )
  )

  # 3. Critic (价值网络) 模型结构配置
  critic: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),  # 3 层 MLP 隐藏层神经元数量（与 Actor 结构保持相同）
      activation="elu",            # 激活函数选用 ELU
      obs_normalization=True,      # 开启 Critic 输入（特权信息/无噪状态）的在线归一化
    )
  )

  # 4. PPO 算法核心超参数配置
  algorithm: RslRlPpoAlgorithmCfg = field(
    default_factory=lambda: RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,         # 价值函数损失 (Value Loss) 的权重系数
      use_clipped_value_loss=True, # 开启 Critic 的 Value Loss 截断，防止价值估计波动过大
      clip_param=0.2,              # PPO 策略概率比率的截断范围 ε (Clip parameter, 1±0.2)
      entropy_coef=0.005,          # 策略熵正则化系数 (鼓励探索，防止策略过早收敛)
      num_learning_epochs=5,       # 每次 Rollout 收集数据后，使用当前数据重复训练的 Epoch 数
      num_mini_batches=4,          # 每个 Epoch 将采样的数据切分成多少个 Mini-batch 进行梯度更新
      learning_rate=1.0e-3,        # 初始学习率 (1e-3)
      schedule="adaptive",         # 自适应学习率调度器（基于 KL 散度自动调整学习率）
      gamma=0.99,                  # 折扣因子 (Discount Factor γ)，重视长远回报
      lam=0.95,                    # GAE (Generalized Advantage Estimation) 的 λ 参数，平衡方差与偏差
      desired_kl=0.01,             # 自适应学习率的目标 KL 散度阈值
      max_grad_norm=1.0,           # 梯度裁剪阈值 (Grad Clipping)，防止梯度爆炸
    )
  )


def f1_mimic_ppo_runner_cfg() -> F1OnPolicyRunnerCfg:
  """返回一个新的 F1 PPO 训练器配置对象实例。"""
  return F1OnPolicyRunnerCfg()