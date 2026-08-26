"""基于 mjlab 构建的外部机器人资产（Assets）与任务（Tasks）主包。"""

from pathlib import Path  # 导入面向对象的路径处理库 Path

# 1. 获取当前脚本文件 (__file__) 的绝对路径并解析符号链接，指向 `src` 目录
SRC_PATH = Path(__file__).resolve().parent

# 2. 获取 `src` 目录的上一级目录，即整个项目的根目录 (Project Root)
PROJECT_ROOT = SRC_PATH.parent

# 3. 拼接定义数据集 (Dataset) 的全局绝对路径（位于项目根目录下的 dataset/ 文件夹）
# 在上文的 f1 运动模仿配置中，DATASET_PATH 被用于定位运动捕捉数据 (.npz 文件)
DATASET_PATH = PROJECT_ROOT / "dataset"