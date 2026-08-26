# DroidUp F1 mjlab

<p align="center"><a href="#中文">中文</a> | <a href="#english">English</a></p>

<a id="中文"></a>

## 中文

用 mjlab (MuJoCo + RL) 训练人形机器人的运动控制策略，两条技术路线：
AMP (Adversarial Motion Prior)：判别器学专家动作的"风格"，策略跟随速度指令走路/跑步/转向。适合做可遥控的行走。
Mimic (motion tracking)：直接跟踪一条参考轨迹，逐帧对齐关键 link 位姿。适合做舞蹈、后空翻这种特定动作复现。
训练完导出 ONNX，再用 sim2sim/ 在纯 MuJoCo 里验证，为上真机做准备。
使用 MJLab 在仿真中训练 DroidUp F1 人形机器人的运动策略，支持 AMP（Adversarial Motion Prior）速度指令运动和 Mimic 动作跟踪。F1 模型包含 28 个受控关节。

### 任务

| 任务 ID | 说明 |
| --- | --- |
| `Tracking-Flat-F1` | F1 28-DOF 动作跟踪 |
| `Tracking-Flat-F1-No-State-Estimation` | 使用重力投影、无状态估计的动作跟踪 |
| `AMP-Walk-Flat-F1` | 支持 walk、run、turn 和 side motion 的速度指令 AMP |

查看所有已注册环境：

```bash
python scripts/list_envs.py
```

### 安装

项目要求 Python 3.11–3.13，并使用 `uv` 管理依赖：

```bash
git clone <repository-url>
cd DroidUpF1_mjlab
uv sync
source .venv/bin/activate
```

环境已经装好了，直接激活：

cd /home/msr/a/DroidUpF1_mjlab
source .venv/bin/activate

激活环境后可以直接运行 `python`，无需使用 `uv run`。主要依赖包括 MJLab 1.5.3、MuJoCo 3.10.0、PyTorch 2.13.0 和 RSL-RL 5.4.2。本地 `rsl_rl/` 以 editable 方式安装。

训练日志默认保存在 `logs/rsl_rl/`：

```bash
tensorboard --logdir logs/rsl_rl
```

### 数据集

训练前请将本地动作数据放入以下目录：

```text
dataset/f1/
├── amp/
│   ├── walk.npz
│   ├── run.npz
│   ├── run_mirror.npz
│   ├── turn_l.npz
│   ├── turn_r.npz
│   ├── side_l.npz
│   └── side_r.npz
└── mimic/
    └── default/
        └── f1_motion.npz
```

`.npz`、`.pkl` 和 `.pt` 文件不会提交到 Git，请在本地自行准备。AMP 默认加载上面的 7 个动作文件；Mimic 默认加载 `dataset/f1/mimic/default/f1_motion.npz`，也可以通过命令行指定其他动作。

### 训练和回放 AMP

训练 AMP（专家数据默认读 dataset/f1/amp/ 下的 walk/run/run_mirror/turn_l/turn_r/side_l/side_r 七个 npz，日志进 logs/rsl_rl/f1_walk_run_amp/）：

python scripts/train.py AMP-Walk-Flat-F1 --env.scene.num-envs 1024 --gpu-ids '[0]'

```bash
python scripts/train.py AMP-Walk-Flat-F1 \
  --env.scene.num-envs 4096 --gpu-ids '[0]'

python scripts/play_amp.py AMP-Walk-Flat-F1 \
  --checkpoint-file logs/rsl_rl/f1_walk_run_amp/<run>/model_6000.pt \
  --lin-vel-x 0.5 --lin-vel-y 0.0 --ang-vel-z 0.0 \
  --num-envs 1 --device cuda:0 --viewer native
```

训练结果默认写入 `logs/rsl_rl/f1_walk_run_amp/`。从 checkpoint 继续训练：

```bash
python scripts/train.py AMP-Walk-Flat-F1 \
  --agent.resume True --agent.load-run <run> \
  --agent.load-checkpoint model_6000.pt --agent.max-iterations 50000 \
  --env.scene.num-envs 4096 --gpu-ids '[0]'
```

### 训练和回放 Mimic

训练 Mimic（参考轨迹默认 dataset/f1/mimic/default/f1_motion.npz，日志进 logs/rsl_rl/external_f1_tracking/，默认 30000 迭代）：


python scripts/train.py Tracking-Flat-F1-No-State-Estimation --env.scene.num-envs 1024 --gpu-ids '[0]'
换轨迹加 --env.commands.motion.motion-file dataset/f1/mimic/backflip/robot_backflip_soma.npz。

回放已有 checkpoint，logs/rsl_rl/external_f1_tracking/2026-08-21_17-08-21/ 里有训到 3000 步的模型：


python scripts/play_mimic.py Tracking-Flat-F1-No-State-Estimation \
  --checkpoint-file logs/rsl_rl/external_f1_tracking/2026-08-21_17-08-21/model_3000.pt \
  --motion-file dataset/f1/mimic/default/f1_motion.npz \
  --num-envs 1 --viewer native


使用默认动作训练：

```bash
python scripts/train.py Tracking-Flat-F1-No-State-Estimation \
  --env.scene.num-envs 4096 --gpu-ids '[0]'
```

指定其他动作训练：

```bash
python scripts/train.py Tracking-Flat-F1-No-State-Estimation \
  --env.commands.motion.motion-file dataset/f1/mimic/dance_npz/f1_motion.npz \
  --env.scene.num-envs 4096 --gpu-ids '[0]'
```

Mimic 训练结果默认写入 `logs/rsl_rl/external_f1_tracking/`。回放命令如下：

```bash
python scripts/play_mimic.py Tracking-Flat-F1-No-State-Estimation \
  --checkpoint-file logs/rsl_rl/external_f1_tracking/<run>/model_5000.pt \
  --motion-file dataset/f1/mimic/dance_npz/f1_motion.npz \
  --num-envs 1 --device cuda:0 --viewer native
```

### 项目结构

- `src/assets/f1/`：F1 MJCF、URDF、USD 和网格资源
- `src/tasks/amp/config/f1/`：F1 AMP 环境与训练配置
- `src/tasks/mimic/config/f1/`：F1 Mimic 环境与训练配置
- `dataset/f1/`：本地 AMP/Mimic 动作数据
- `scripts/`：环境列表、训练和策略回放入口
- `tools/`：动作数据处理与 MuJoCo 回放工具
- `sim2sim/`：ONNX sim2sim 实验脚本和策略
- `rsl_rl/`：项目使用的本地 RSL-RL 5.4.2

> 注意：仓库中的 `sim2sim_e1_21dof_*` 脚本和部分数据转换工具仍使用 E1 21-DOF 模型及其关节顺序，不能直接用于 F1 策略。F1 训练使用 `src/assets/f1/mjcf/f1_1.xml` 和 F1 28-DOF 关节顺序。

<p align="right"><a href="#english">English</a></p>

<a id="english"></a>

## English

Train DroidUp F1 humanoid motion policies in simulation with MJLab. The project supports velocity-commanded AMP (Adversarial Motion Prior) locomotion and Mimic motion tracking for the 28-DOF F1 model.

### Tasks

| Task ID | Description |
| --- | --- |
| `Tracking-Flat-F1` | F1 28-DOF motion tracking |
| `Tracking-Flat-F1-No-State-Estimation` | Motion tracking with projected gravity and no state estimator |
| `AMP-Walk-Flat-F1` | Velocity-commanded AMP for walk, run, turn, and side motions |

List all registered environments with `python scripts/list_envs.py`.

### Installation

Python 3.11–3.13 and `uv` are required:

```bash
git clone <repository-url>
cd DroidUpF1_mjlab
uv sync
source .venv/bin/activate
```

After activation, commands can be run directly with `python`; `uv run` is not required. The main dependencies are MJLab 1.5.3, MuJoCo 3.10.0, PyTorch 2.13.0, and RSL-RL 5.4.2. The local `rsl_rl/` checkout is installed in editable mode. View training logs with `tensorboard --logdir logs/rsl_rl`.

### Datasets

Prepare the following files locally before training:

```text
dataset/f1/
├── amp/
│   ├── walk.npz
│   ├── run.npz
│   ├── run_mirror.npz
│   ├── turn_l.npz
│   ├── turn_r.npz
│   ├── side_l.npz
│   └── side_r.npz
└── mimic/default/f1_motion.npz
```

Files ending in `.npz`, `.pkl`, and `.pt` are excluded from Git and must be supplied locally. AMP loads the seven files above by default. Mimic uses `dataset/f1/mimic/default/f1_motion.npz` unless another motion is provided on the command line.

### AMP training and playback

```bash
python scripts/train.py AMP-Walk-Flat-F1 \
  --env.scene.num-envs 4096 --gpu-ids '[0]'

python scripts/play_amp.py AMP-Walk-Flat-F1 \
  --checkpoint-file logs/rsl_rl/f1_walk_run_amp/<run>/model_6000.pt \
  --lin-vel-x 0.5 --lin-vel-y 0.0 --ang-vel-z 0.0 \
  --num-envs 1 --device cuda:0 --viewer native
```

AMP runs are stored under `logs/rsl_rl/f1_walk_run_amp/`. Resume training with:

```bash
python scripts/train.py AMP-Walk-Flat-F1 \
  --agent.resume True --agent.load-run <run> \
  --agent.load-checkpoint model_6000.pt --agent.max-iterations 50000 \
  --env.scene.num-envs 4096 --gpu-ids '[0]'
```

### Mimic training and playback

```bash
python scripts/train.py Tracking-Flat-F1-No-State-Estimation \
  --env.scene.num-envs 4096 --gpu-ids '[0]'

python scripts/train.py Tracking-Flat-F1-No-State-Estimation \
  --env.commands.motion.motion-file dataset/f1/mimic/dance_npz/f1_motion.npz \
  --env.scene.num-envs 4096 --gpu-ids '[0]'

python scripts/play_mimic.py Tracking-Flat-F1-No-State-Estimation \
  --checkpoint-file logs/rsl_rl/external_f1_tracking/<run>/model_5000.pt \
  --motion-file dataset/f1/mimic/dance_npz/f1_motion.npz \
  --num-envs 1 --device cuda:0 --viewer native
```

Mimic runs are stored under `logs/rsl_rl/external_f1_tracking/`.

### Project layout

- `src/assets/f1/`: F1 MJCF, URDF, USD, and mesh assets
- `src/tasks/amp/config/f1/`: F1 AMP environment and training configuration
- `src/tasks/mimic/config/f1/`: F1 Mimic environment and training configuration
- `dataset/f1/`: local AMP and Mimic motion data
- `scripts/`: environment listing, training, and policy playback entry points
- `tools/`: motion processing and MuJoCo replay utilities
- `sim2sim/`: experimental ONNX sim2sim runners and policies
- `rsl_rl/`: local RSL-RL 5.4.2 checkout

> Note: the bundled `sim2sim_e1_21dof_*` scripts and some data conversion tools still depend on the E1 21-DOF model and joint order; they cannot be used directly with F1 policies. F1 training uses `src/assets/f1/mjcf/f1_1.xml` and the F1 28-DOF joint order.

<p align="right"><a href="#中文">中文</a></p>
