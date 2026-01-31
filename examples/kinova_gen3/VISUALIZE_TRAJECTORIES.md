# Kinova Gen3 轨迹可视化工具

## 概述

提供两个轨迹可视化脚本，用于可视化 Kinova Gen3 机械臂末端执行器的运动轨迹。

## 脚本列表

### 1. visualize_trajectories_from_dataset.py

从 LeRobot 格式数据集可视化轨迹。

**功能**:
- 读取 `~/.cache/huggingface/lerobot/kinova_gen3_dataset/` 数据
- 提取关节角度序列（state 字段的前 7 维）
- 使用 URDF 正运动学计算末端位置
- 绘制 3D 空间轨迹

**快速开始**:
```bash
cd examples/kinova_gen3

# 基本使用
uv run python visualize_trajectories_from_dataset.py

# 处理前 10 个 episodes
uv run python visualize_trajectories_from_dataset.py --max-episodes 10

# 生成单独图片 + 统计信息
uv run python visualize_trajectories_from_dataset.py --max-episodes 5 --separate-plots --show-stats
```

### 2. visualize_trajectories_from_logs.py

从控制日志可视化轨迹。

**功能**:
- 读取 `logs/control_log_*.jsonl` 日志文件
- 提取 state.joint_position 或 action.joint_position
- 使用 URDF 正运动学计算末端位置
- 绘制 3D 空间轨迹

**快速开始**:
```bash
cd examples/kinova_gen3

# 可视化所有日志
uv run python visualize_trajectories_from_logs.py

# 处理单个日志文件
uv run python visualize_trajectories_from_logs.py --log-file ../../logs/control_log_2026_01_30_16-51-00.jsonl

# 处理前 5 个日志
uv run python visualize_trajectories_from_logs.py --max-logs 5 --separate-plots --show-stats

# 使用 action 位置而不是 state 位置
uv run python visualize_trajectories_from_logs.py --use-action
```

## 图片类型

### 从数据集生成的图片
- `kinova_all_trajectories.png` - 所有 episodes 的综合图
- `kinova_trajectories_3d.png` - 部分 episodes 的综合图
- `episode_XXX_trajectory.png` - 单个 episode 的轨迹图

### 从日志生成的图片
- `kinova_trajectories_from_logs.png` - 所有日志的综合图
- `kinova_logs_trajectories_topN.png` - 前 N 个日志的综合图
- `control_log_YYYY_MM_DD_HH-MM-SS_trajectory.png` - 单个日志的轨迹图

## 可视化说明

### 颜色标记
- **综合图**: 不同颜色代表不同的轨迹（episode 或日志）
- **单独图**: 渐变色表示时间进程（深蓝→黄色）
- **绿色圆点**: 起点
- **红色方块**: 终点

### 坐标系
所有轨迹都在基坐标系（`base_link`）中绘制，单位为米。

## 主要功能

### 1. 读取数据

**从数据集**:
- LeRobot Parquet 格式
- 状态维度: 32（前 7 维是关节角度）

**从日志**:
- JSONL 格式
- 每行一个 JSON 对象
- 包含 `state.joint_position` 和 `action.joint_position`

### 2. 正运动学

使用 `URDFKinematics` 类：
- URDF 文件: `GEN3_URDF_V12_with_dampint.urdf`
- 基坐标系: `base_link`
- 末端坐标系: `end_effector_link`

### 3. 统计分析

使用 `--show-stats` 显示：
- 轨迹长度（米）
- 点数
- 工作空间范围（X, Y, Z）
- 曲折度（轨迹长度 / 直线距离）

## 命令行参数对比

### visualize_trajectories_from_dataset.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data-dir` | 数据目录路径 | `~/.cache/huggingface/lerobot/kinova_gen3_dataset/data/chunk-000` |
| `--max-episodes` | 最多处理的 episode 数量 | 全部 |
| `--separate-plots` | 为每个 episode 生成单独图片 | False |
| `--show-stats` | 显示统计信息 | False |

### visualize_trajectories_from_logs.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--log-file` | 单个日志文件路径 | None |
| `--logs-dir` | 日志目录路径 | `../../logs` |
| `--max-logs` | 最多处理的日志文件数量 | 全部 |
| `--separate-plots` | 为每个日志生成单独图片 | False |
| `--show-stats` | 显示统计信息 | False |
| `--use-action` | 使用 action 而不是 state | False |

### 公共参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--urdf-path` | URDF 文件路径 | `./GEN3_URDF_V12_with_dampint.urdf` |
| `--output-dir` | 输出目录 | `./saved_images` |
| `--output-filename` | 输出文件名 | 各脚本默认不同 |

## 使用场景

### 数据集可视化
用于：
- 检查训练数据的空间覆盖
- 分析数据集的多样性
- 验证数据转换是否正确

### 日志可视化
用于：
- 检查实际执行轨迹
- 对比规划（action）与实际（state）
- 调试控制问题
- 评估任务执行质量

## State vs Action

### State (默认)
- `state.joint_position`: 机器人的实际关节位置
- 反映真实执行轨迹
- 包含控制误差和动力学影响

### Action (使用 `--use-action`)
- `action.joint_position`: 策略输出的目标关节位置
- 反映规划轨迹
- 理想情况下的期望路径

对比两者可以评估控制精度。

## 输出示例

### 从数据集

```
[INFO] 找到 55 个 episode 文件
[INFO] 处理 episode 1/55: episode_000000.parquet
[INFO]   时间步数: 195
[INFO]   轨迹长度: 0.515m
[INFO]   工作空间范围: X[0.088, 0.242], Y[-0.049, 0.010], Z[0.438, 0.596]
...
```

### 从日志

```
[INFO] 找到 10 个日志文件
[INFO] 处理日志 1/10: control_log_2026_01_30_16-51-00.jsonl
[INFO]   日志条目数: 248
[INFO]   轨迹长度: 0.392m
[INFO]   工作空间范围: X[0.085, 0.222], Y[-0.136, -0.045], Z[0.595, 0.651]
...
```

## 技术栈

- `URDFKinematics`: 正运动学计算
- `matplotlib`: 3D 可视化
- `pandas + pyarrow`: Parquet 数据读取
- `numpy`: 数值计算
- `json`: JSONL 日志解析

## 依赖项

```python
matplotlib
numpy
pandas
pyarrow  # 用于读取 Parquet 文件
```

已包含在项目的依赖中，使用 `uv run` 会自动管理。

## 故障排除

### 轨迹长度为 0
某些日志中机器人可能没有移动（例如测试或调试日志），这是正常的。

### 数据目录不存在
确认数据已转换为 LeRobot 格式：
```bash
ls ~/.cache/huggingface/lerobot/kinova_gen3_dataset/data/chunk-000/
```

### 日志目录不存在
确认日志已生成：
```bash
ls logs/control_log_*.jsonl
```

### 中文字体警告
不影响图片生成，只是图例中的中文可能显示为方块。可以忽略。

## 相关文件

- `urdf_kinematics.py` - URDF 正运动学实现
- `GEN3_URDF_V12_with_dampint.urdf` - Kinova Gen3 URDF 模型
- `log_utils.py` - 日志记录工具
- `convert_to_lerobot.py` - 数据格式转换
- `collect_data.py` - 数据收集
- `main.py` - 策略推理主程序（生成日志）
