# Kinova Gen3 轨迹可视化工具

本目录包含两个轨迹可视化脚本及其生成的图片。

## 脚本说明

### 1. `visualize_trajectories_from_dataset.py`
从 LeRobot 数据集可视化轨迹

**数据来源**: `~/.cache/huggingface/lerobot/kinova_gen3_dataset/`

**使用示例**:
```bash
cd examples/kinova_gen3

# 可视化所有 episodes
uv run python visualize_trajectories_from_dataset.py

# 处理前 10 个 episodes
uv run python visualize_trajectories_from_dataset.py --max-episodes 10

# 生成单独图片 + 统计信息
uv run python visualize_trajectories_from_dataset.py --max-episodes 5 --separate-plots --show-stats
```

### 2. `visualize_trajectories_from_logs.py`
从控制日志可视化轨迹

**数据来源**: `logs/control_log_*.jsonl`

**使用示例**:
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

## 技术细节

### 数据提取
- **从数据集**: 读取 `state` 字段的前 7 维（关节角度）
- **从日志**: 读取 `state.joint_position` 字段（默认）或 `action.joint_position`（使用 `--use-action`）

### 正运动学
- URDF 文件: `GEN3_URDF_V12_with_dampint.urdf`
- 基坐标系: `base_link`
- 末端坐标系: `end_effector_link`
- 使用 `URDFKinematics` 类计算末端位置

### 统计信息
使用 `--show-stats` 可以显示：
- 轨迹长度（米）
- 点数
- 工作空间范围（X, Y, Z）
- 曲折度（轨迹长度 / 直线距离）

## 常见问题

### 轨迹长度为 0
某些日志中机器人可能没有移动（例如测试或调试日志），这是正常的。

### 中文字体警告
不影响图片生成，只是图例中的中文可能显示为方块。可以忽略。

### 数据格式
- **数据集格式**: Parquet 文件，状态维度 32（前 7 维是关节角度）
- **日志格式**: JSONL 文件，每行一个 JSON 对象

## 目录结构

```
saved_images/
├── README.md                                          # 本文件
├── kinova_all_trajectories.png                        # 数据集：所有 episodes
├── kinova_trajectories_3d.png                         # 数据集：部分 episodes
├── episode_XXX_trajectory.png                         # 数据集：单个 episode
├── kinova_trajectories_from_logs.png                  # 日志：综合图
├── kinova_logs_trajectories_topN.png                  # 日志：前 N 个
└── control_log_YYYY_MM_DD_HH-MM-SS_trajectory.png    # 日志：单个日志
```

## 参考

详细文档请查看:
- `VISUALIZE_TRAJECTORIES.md` - 完整使用指南
- `urdf_kinematics.py` - URDF 正运动学实现
- `log_utils.py` - 日志记录工具
