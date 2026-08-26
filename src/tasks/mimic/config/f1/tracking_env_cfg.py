"""F1 motion mimic task configuration.

这是 BeyondMimic (https://beyondmimic.github.io/) 算法的重构实现。

参考代码库: https://github.com/HybridRobotics/whole_body_tracking
提交 Hash: f8e20c880d9c8ec7172a13d3a88a65e3a5a88448
"""

# 导入 mjlab 环境框架相关的核心模块
from mjlab.envs import ManagerBasedRlEnvCfg  # 基于管理器的强化学习环境配置基类
from mjlab.envs.mdp import dr  # Domain Randomization (域随机化) 相关 MDP 函数
from mjlab.envs.mdp.actions import JointPositionActionCfg  # 关节位置控制动作配置
from mjlab.managers.action_manager import ActionTermCfg  # 动作项配置
from mjlab.managers.command_manager import CommandTermCfg  # 目标指令/参考轨迹项配置
from mjlab.managers.event_manager import EventTermCfg  # 事件（扰动/随机化）项配置
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg  # 观测项配置
from mjlab.managers.reward_manager import RewardTermCfg  # 奖励函数项配置
from mjlab.managers.scene_entity_config import SceneEntityCfg  # 场景实体（如机器人指定部件/关节）选择器配置
from mjlab.managers.termination_manager import TerminationTermCfg  # Episode 终止条件配置
from mjlab.scene import SceneCfg  # 场景配置（地面、环境数等）
from mjlab.sensor import ContactMatch, ContactSensorCfg  # 碰撞传感器配置
from mjlab.sim import MujocoCfg, SimulationCfg  # MuJoCo 物理仿真参数配置
from mjlab.terrains import TerrainEntityCfg  # 地形配置
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg  # 包含加性偏置的噪声模型
from mjlab.utils.noise import UniformNoiseCfg as Unoise  # 均匀分布噪声模型
from mjlab.viewer import ViewerConfig  # 可视化 Viewer 配置
from src import DATASET_PATH  # 数据集根路径
from src.assets.f1 import F1_ACTION_SCALE, get_f1_robot_cfg  # F1 机器人资产配置与动作缩放系数
from src.tasks.mimic import mdp  # 动作模仿专属的马尔可夫决策过程 (MDP) 函数库
from src.tasks.mimic.mdp import MotionCommandCfg  # 动作模仿目标指令配置

# 定义环境重置或随机推拽机器人时的线速度与角速度变化范围
VELOCITY_RANGE = {
  "x": (-0.5, 0.5),      # x 方向线速度 (m/s)
  "y": (-0.5, 0.5),      # y 方向线速度 (m/s)
  "z": (-0.2, 0.2),      # z 方向线速度 (m/s)
  "roll": (-0.52, 0.52),  # 翻滚角速度 (rad/s)
  "pitch": (-0.52, 0.52),# 俯仰角速度 (rad/s)
  "yaw": (-0.78, 0.78),  # 偏航角速度 (rad/s)
}


def f1_flat_mimic_env_cfg(
  has_state_estimation: bool = True,  # 是否开启理想的状态估计（如基座线速度/位置）
  play: bool = False,                 # 是否为测试/演示模式（训练为 False，评估/部署为 True）
) -> ManagerBasedRlEnvCfg:
  """创建 F1 人形机器人在平坦地面上的动作跟踪/模仿训练环境配置。"""

  ##
  # 1. 观测空间配置 (Observations)
  ##

  # Actor (策略网络) 的观测项 - 通常包含含噪传感器数据及延迟
  actor_terms = {
    # 动作模仿的目标指令（包含未来帧的参考姿态等信息）
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    # 基座坐标系下运动锚点 (Anchor Body) 的位置偏差（注入 [-0.25, 0.25] 均匀噪声）
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.25, n_max=0.25),
    ),
    # 机器人 IMU 测得的重力投影向量（包含噪声、零偏 bias 以及信号传输延迟模拟）
    "projected_gravity": ObservationTermCfg(
      # F1_1.xml does not define E1's imu_upvector sensor. Compute the
      # gravity projection directly from the robot root orientation instead.
      func=mdp.projected_gravity,
      params={"asset_cfg": SceneEntityCfg("robot")},
      noise=NoiseModelWithAdditiveBiasCfg(
        noise_cfg=Unoise(n_min=-0.05, n_max=0.05),
        bias_noise_cfg=Unoise(n_min=-0.02, n_max=0.02, operation="abs"),
      ),
      delay_min_lag=0,           # 最小延迟步数
      delay_max_lag=1,           # 最大延迟步数
      delay_hold_prob=0.9,       # 延迟保持概率
      delay_update_period=5,     # 延迟更新周期
    ),
    # 基座线速度（IMU 内置传感器输入，带 [-0.5, 0.5] 噪声）
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    # 基座角速度（带噪声、漂移和延迟模拟）
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=NoiseModelWithAdditiveBiasCfg(
        noise_cfg=Unoise(n_min=-0.2, n_max=0.2),
        bias_noise_cfg=Unoise(n_min=-0.05, n_max=0.05, operation="abs"),
      ),
      delay_min_lag=0,
      delay_max_lag=1,
      delay_hold_prob=0.9,
      delay_update_period=5,
    ),
    # 相对关节位置（带编码器偏置 noise 及延迟）
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
      delay_min_lag=0,
      delay_max_lag=1,
      delay_hold_prob=0.9,
      delay_update_period=5,
    ),
    # 相对关节速度（带噪声及延迟）
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
      delay_min_lag=0,
      delay_max_lag=1,
      delay_hold_prob=0.9,
      delay_update_period=5,
    ),
    # 机器人上一步执行的动作 (Last Action)
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  # Critic (价值网络) 的观测项 - 非对称 Actor-Critic (Asymmetric AC)，Critic 拥有理想/无噪的完整状态
  critic_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    # 无噪的运动锚点位置
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    # 无噪的运动锚点姿态（四元数/旋转矩阵）
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    # 无噪的机器人全身各 Link 相对位置
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    # 无噪的机器人全身各 Link 相对姿态
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
    # 无噪的真实线速度
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    # 无噪的真实角速度
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    # 无噪的真实关节位置
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    # 无噪的真实关节速度
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    # 上一步动作
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  # 将 Actor 与 Critic 的观测打包成组
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,    # 拼接为单个一维 Tensor
      enable_corruption=True,    # 允许注入噪声与延迟
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,   # Critic 观测不加任何噪声污染
    ),
  }

  ##
  # 2. 动作空间配置 (Actions)
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),   # 作用于机器人所有关节
      scale=0.5,               # 动作输出映射到关节位置目标时的缩放比例（后续会被覆盖）
      use_default_offset=True, # 叠加机器人的默认立正/站立姿姿态偏移量
    )
  }

  ##
  # 3. 目标指令配置 (Commands - 运动追踪数据源)
  ##

  commands: dict[str, CommandTermCfg] = {
    "motion": MotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),  # 极长重采样时间（即不重新采样，顺着参考动作一直播放）
      debug_vis=True,                        # 在渲染界面显示参考轨迹调试小球/姿态
      # 重置时姿态的微小扰动范围
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      # 以下字段后续在函数尾部填充
      motion_file="",
      anchor_body_name="",
      body_names=(),
    )
  }

  ##
  # 4. 域随机化与随机事件配置 (Events / Domain Randomization)
  ##

  events: dict[str, EventTermCfg] = {
    # 定期给机器人施加随机推力（设置线速度/角速度）
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),  # 每 1~3 秒推一次
      params={"velocity_range": VELOCITY_RANGE},
    ),
    # 随机化刚体质量和转动惯量
    "body_inertia": EventTermCfg(
      mode="startup",  # 环境启动时运行
      func=dr.pseudo_inertia,
      params={
        "alpha_range": (-0.025, 0.025), # 缩放系数，大概相当于 0.95~1.05 倍缩放
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    # 随机化基座质心 (CoM) 位置偏移
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # 具体 link 名在下文设置
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025), # X 轴偏移范围 (m)
          1: (-0.05, 0.05),   # Y 轴偏移范围 (m)
          2: (-0.05, 0.05),   # Z 轴偏移范围 (m)
        },
      },
    ),
    # 随机化 PD 控制器的增益 (Kp, Kd)
    "pd_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.9, 1.1),  # 90% ~ 110% 缩放
        "kd_range": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    # 随机化电机最大输出扭矩极限
    "effort_limits": EventTermCfg(
      mode="startup",
      func=dr.effort_limits,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "effort_limit_range": (0.9, 1.0), # 90% ~ 100%
        "operation": "scale",
      },
    ),
    # 随机化关节阻尼 (Damping)
    "joint_damping": EventTermCfg(
      mode="startup",
      func=dr.joint_damping,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (0.8, 1.2),  # 80% ~ 120%
        "operation": "scale",
      },
    ),
    # 随机化关节静摩擦力 (Friction)
    "joint_friction": EventTermCfg(
      mode="startup",
      func=dr.joint_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (0.8, 1.2),
        "operation": "scale",
      },
    ),
    # 随机化关节电枢电感/等效惯量 (Armature)
    "joint_armature": EventTermCfg(
      mode="startup",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    # 随机化关节角度编码器的零偏 (Encoder Bias)
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01), # rad
      },
    ),
    # 随机化脚底碰撞几何体的摩擦系数
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs",
        "ranges": (0.3, 1.2), # 摩擦系数 0.3 ~ 1.2
        "shared_random": True,  # 左右脚共享同一个随机生成的摩擦系数
      },
    ),
  }

  ##
  # 5. 奖励函数配置 (Rewards - 驱动机器人准确模仿运动)
  ##

  rewards: dict[str, RewardTermCfg] = {
    # 1. 全局根节点（Torso）位置追踪奖励 (指数衰减型: exp(-err / std))
    "motion_global_root_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.3},
    ),
    # 2. 全局根节点姿态（旋转方向）追踪奖励
    "motion_global_root_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    # 3. 机器人全身各个 Link 的相对位置追踪奖励（核心运动模仿项）
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    # 4. 机器人全身各个 Link 的相对姿态追踪奖励
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    # 5. 各 Link 的线速度匹配奖励
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    # 6. 各 Link 的角速度匹配奖励
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    # 7. 惩罚动作剧烈变化 (Action Rate L2 Penalty)，使动作更平滑
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
    # 8. 惩罚超过关节物理限位的行为
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # 9. 惩罚机器人自身发生的非预期碰撞（如左右腿互碰）
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }

  ##
  # 6. Episode 终止条件 (Terminations)
  ##

  terminations: dict[str, TerminationTermCfg] = {
    # 达到最大时间步，正常超时重置
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # 锚点（Torso）Z轴高度偏差太大时早期终止（如机器人摔倒）
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},  # 偏离超过 25cm
    ),
    # 锚点姿态偏差过大终止
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    # 末端执行器 (双手手腕、双脚脚踝) 位置偏差过大时终止
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": (),  # 下文具体设置
      },
    ),
  }

  ##
  # 7. 组装整体环境配置对象 (Assemble Base Config)
  ##

  cfg = ManagerBasedRlEnvCfg(
    # 平坦地面场景，1 个并行环境（训练时可在外部通过 num_envs 覆盖）
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    # 可视化相机跟踪配置
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # 下文设置为 torso_link
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    # MuJoCo 物理引擎解算器设置
    sim=SimulationCfg(
      # F1 has seven collision capsules per foot plus body collision geoms.
      # Keep enough room for simultaneous foot contacts and self-collisions.
      nconmax=80,
      njmax=500,
      mujoco=MujocoCfg(
        timestep=0.005,      # 物理仿真步长: 200 Hz (1 / 0.005)
        iterations=10,       # 解算器迭代次数
        ls_iterations=20,    # 线搜索迭代次数
      ),
    ),
    decimation=4,           # 控制频率降采样率 (控制频率 = 200Hz / 4 = 50Hz)
    episode_length_s=10.0,  # 训练时每回合最长 10 秒
  )

  ##
  # 8. 结合 F1 机器人资产参数与动作数据进行具体填充
  ##

  # 加载 F1 机器人配置，如果是测试/部署模式 (play=True) 则无动作延迟，训练时模拟 0~4 步(0~80ms)的延迟
  cfg.scene.entities = {
    "robot": get_f1_robot_cfg(
      action_delay_range=(0, 0) if play else (0, 4)
    )
  }
  # 配置用于检测自碰撞的接触传感器 (监测 pelvis 盆骨子树下部件的相互碰撞)
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
  )

  # 覆盖动作空间的缩放系数，使用预设的 F1 比例
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = F1_ACTION_SCALE

  # 配置参考动作轨迹文件和全身的关键 Link/Body 列表
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  # 加载参考姿态数据集路径 (.npz 格式)
  motion_cmd.motion_file = str(
    DATASET_PATH / "f1" / "mimic/default/f1_motion.npz"
  )
  motion_cmd.anchor_body_name = "torso_link"  # 选取躯干为核心对齐锚点
  # 参与姿态/运动追踪比较的机器人所有 Link 列表
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_yaw_link",
    "left_wrist_pitch_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_yaw_link",
    "right_wrist_pitch_link",
  )

  # 正则匹配双脚的 7 个碰撞几何体，用于脚底摩擦力随机化
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  # 质心偏移施加于躯干
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  # 监控双脚踝和双手腕的位置偏差作为早停条件
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
  )
  # 视线绑定躯干
  cfg.viewer.body_name = "torso_link"

  ##
  # 9. 特殊模式处理（无状态估计模式 / Play 推理模式）
  ##

  # 如果没有基座状态估计器（真实部署时难以直接获得精确的 Base 位置和线速度），则从 Actor 观测中移除它们
  if not has_state_estimation:
    actor_terms_without_state_estimation = {
      name: term
      for name, term in cfg.observations["actor"].terms.items()
      if name not in ("motion_anchor_pos_b", "base_lin_vel")
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=actor_terms_without_state_estimation,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # 如果是 Play/测试展示模式
  if play:
    cfg.episode_length_s = int(1e9)  # 回合不限时
    cfg.observations["actor"].enable_corruption = False  # 关闭观测噪声污染
    # 清空所有观测延迟
    for term in cfg.observations["actor"].terms.values():
      term.delay_min_lag = 0
      term.delay_max_lag = 0
      term.delay_hold_prob = 0.0
      term.delay_update_period = 0
    cfg.events.clear()  # 禁用所有域随机化和推力扰动
    # 将重置时的姿态/速度扰动全部置零，确保从轨迹起点精准开始
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "start"  # 从动作序列的第一帧开始播放

  return cfg
