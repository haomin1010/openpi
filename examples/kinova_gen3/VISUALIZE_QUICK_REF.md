# 轨迹可视化快速参考

## 两个脚本的区别

| 特性 | from_dataset.py | from_logs.py |
|------|-----------------|--------------|
| **数据源** | LeRobot 数据集 | 控制日志 JSONL |
| **数据格式** | Parquet | JSONL |
| **数据字段** | state (32维) | state/action.joint_position |
| **主要用途** | 检查训练数据 | 检查执行轨迹 |
| **特色功能** | - | 可对比 state vs action |

## 快速命令

```bash
cd examples/kinova_gen3

# 从数据集 - 基本
uv run python visualize_trajectories_from_dataset.py

# 从数据集 - 完整
uv run python visualize_trajectories_from_dataset.py --max-episodes 10 --separate-plots --show-stats

# 从日志 - 基本
uv run python visualize_trajectories_from_logs.py

# 从日志 - 单个文件
uv run python visualize_trajectories_from_logs.py --log-file ../../logs/control_log_2026_01_30_16-51-00.jsonl

# 从日志 - 对比规划与实际
uv run python visualize_trajectories_from_logs.py --log-file ../../logs/control_log_2026_01_30_16-51-00.jsonl --show-stats
uv run python visualize_trajectories_from_logs.py --log-file ../../logs/control_log_2026_01_30_16-51-00.jsonl --use-action --output-filename planned.png
```

## 输出位置

所有图片保存在: `examples/kinova_gen3/saved_images/`

## 帮助

```bash
uv run python visualize_trajectories_from_dataset.py --help
uv run python visualize_trajectories_from_logs.py --help
```
