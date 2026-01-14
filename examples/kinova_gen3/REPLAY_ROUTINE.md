# Kinova Gen3 轨迹回放指南

本文档介绍如何使用 `replay_routine.py` 脚本回放 Kinova Gen3 机械臂的轨迹数据，实现轨迹的精确复现。

## 📋 功能概述

`replay_routine.py` 用于从保存的轨迹数据文件中读取机器人轨迹，并使用 KinovaRobotEnv 精确复现机器人的动作序列。

**主要特性：**
- ✅ **平滑轨迹回放**（默认）：使用速度控制和平滑滤波，实现更平滑的运动
- ✅ **原始位置回放**：使用绝对位置模式精确复现轨迹（可选）
- ✅ 支持回放速度控制（加速/减速）
- ✅ 支持部分轨迹回放（指定起始和结束步数）
- ✅ 自动时间同步（使用原始采集频率）
- ✅ 支持键盘中断（Ctrl+C）
- ✅ 灵活的轨迹文件选择（自动查找、指定文件、指定 session）

## 🔍 数据要求

### 数据格式

回放功能使用 **回放数据（replay_data格式）**，而非训练数据（libero_format格式）。

**必需的数据文件：**
- 文件格式：`.npz` 文件（压缩 numpy 数组）
- 文件位置：`data/{session_name}/replay_data/episode_*_replay_*.npz`
- 文件命名：`episode_{count:03d}_replay_{timestamp}.npz`

**必需的数据字段：**
- `joint_positions`: (N, 7) 关节位置数组（弧度）
- `gripper_pos`: (N,) 夹爪状态数组，0.0=张开，1.0=闭合（二值动作）

**可选的数据字段：**
- `timestamp`: (N,) 时间戳数组（用于精确控制时序）
- `step`: (N,) 步数数组
- `eef_pose`: (N, 7) 末端执行器位姿 **[平滑回放必需]** - 格式为 [x, y, z, qx, qy, qz, qw]
- `action`: (N, 7) 动作数组
- `collection_frequency`: (标量) 采集频率（Hz）**[平滑回放推荐]** - 用于使用原始采集频率进行回放

### 数据来源

回放数据由 `collect_data.py` 脚本在数据采集时自动生成。如果数据目录结构如下：

```
data/
└── General_manipulation_task_20260114_093000/
    ├── libero_format/              # 训练数据（不使用）
    │   └── episode_001_libero_*.npz
    └── replay_data/                # 回放数据（使用）
        └── episode_001_replay_*.npz
```

## 🛠️ 准备工作

### 1. 硬件连接

确保以下设备已正确连接：
- ✅ Kinova Gen3 机械臂已连接并处于正常状态
- ✅ Arduino 夹爪控制器已连接（可选，用于夹爪控制）
- ✅ 相机已连接（可选，仅用于初始化，回放过程中不使用）

### 2. 环境配置

确保已安装以下依赖：

```bash
# 在项目根目录
cd <项目根目录>

# 安装基础依赖（如果还没安装）
GIT_LFS_SKIP_SMUDGE=1 uv sync

# 安装 Kinova SDK
pip install kortex_api-2.6.0.post3-py3-none-any.whl

# 安装其他依赖
uv pip install pyrealsense2 pynput
```

### 3. 网络配置

确保以下设备在同一局域网内：
- Kinova 机械臂（默认 IP: `192.168.1.10`）
- Arduino 夹爪控制器（默认 IP: `192.168.1.43`）

## 🚀 使用方法

### 基本用法

#### 方式 1：回放最新的轨迹（推荐）

自动在所有 session 中查找最新的轨迹文件并回放：

```bash
cd examples/kinova_gen3
python replay_routine.py
```

#### 方式 2：指定轨迹文件路径

直接指定要回放的轨迹文件：

```bash
python replay_routine.py \
    --data-path data/General_manipulation_task_20260114_093000/replay_data/episode_001_replay_20260114_093159.npz
```

#### 方式 3：指定 session 目录

指定 session 目录，自动使用该 session 中最新的轨迹文件：

```bash
python replay_routine.py \
    --session data/General_manipulation_task_20260114_093000
```

### 完整示例

#### 示例 1：使用默认设置回放最新轨迹

```bash
python replay_routine.py
```

#### 示例 2：指定机器人 IP 和夹爪 IP

```bash
python replay_routine.py \
    --robot-ip 192.168.1.10 \
    --gripper-ip 192.168.1.43
```

#### 示例 3：控制回放速度（0.5倍速，更慢更安全）

```bash
python replay_routine.py \
    --playback-speed 0.5
```

#### 示例 4：回放部分轨迹（从第 100 步到第 500 步）

```bash
python replay_routine.py \
    --start-step 100 \
    --end-step 500
```

#### 示例 5：使用平滑回放（默认，推荐）

```bash
python replay_routine.py \
    --playback-speed 1.0 \
    --smoothing-window-size 5
```

#### 示例 6：禁用平滑回放（使用原始位置控制）

```bash
python replay_routine.py \
    --no-smooth
```

#### 示例 7：组合使用多个参数

```bash
python replay_routine.py \
    --data-path data/MyTask_20260114_093000/replay_data/episode_002_replay_20260114_093500.npz \
    --robot-ip 192.168.1.10 \
    --gripper-ip 192.168.1.43 \
    --playback-speed 0.75 \
    --start-step 50 \
    --end-step 300 \
    --smoothing-window-size 7
```

## 📝 命令行参数

### 数据选择参数（互斥，只能选一个）

| 参数 | 类型 | 说明 |
|------|------|------|
| `--data-path` | str | 直接指定轨迹数据文件路径（.npz 文件） |
| `--session` | str | 指定 session 目录路径（将使用该 session 中最新的轨迹文件） |
| （无参数） | - | 默认行为：在所有 session 中自动查找最新的轨迹文件 |

**优先级：** `--data-path` > `--session` > 默认（最新轨迹）

### 机器人连接参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--robot-ip` | `192.168.1.10` | Kinova 机械臂 IP 地址 |
| `--gripper-ip` | `192.168.1.43` | Arduino 夹爪控制器 IP 地址 |
| `--external-camera-serial` | None | 外部相机序列号（可选，仅用于初始化） |
| `--wrist-camera-serial` | None | 腕部相机序列号（可选，仅用于初始化） |

### 回放控制参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--playback-speed` | `1.0` | 回放速度倍数<br>- `1.0`: 原始速度<br>- `2.0`: 2倍速（更快）<br>- `0.5`: 0.5倍速（更慢） |
| `--start-step` | `0` | 起始步数（从第几步开始回放） |
| `--end-step` | None | 结束步数（回放到第几步，默认到轨迹末尾） |
| `--smooth` | `True` (默认) | 使用平滑回放（速度控制和平滑滤波） |
| `--no-smooth` | - | 禁用平滑回放（使用原始位置控制模式） |
| `--smoothing-window-size` | `5` | 平滑窗口大小（用于轨迹平滑的点数，仅平滑回放时有效） |

### 其他参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `脚本目录/data` | 数据根目录（当使用默认查找时） |

## 🔧 实现原理

### 回放流程

1. **数据加载**：从 `.npz` 文件加载轨迹数据（关节位置、夹爪状态、时间戳等）
2. **时间计算**：根据时间戳或固定频率（60Hz）计算每步之间的时间间隔
3. **初始定位**：机器人先移动到轨迹的起始位置
4. **逐步执行**：按时间间隔逐步执行轨迹，每次移动到下一个目标位置
5. **清理资源**：回放完成后断开机器人连接、关闭相机等

### 关键技术

- **绝对位置模式**：使用 `ActionMode.ABSOLUTE` 直接指定目标关节位置，确保精确复现
- **时间同步**：优先使用时间戳保持原始时序，否则使用固定频率（60Hz，与采集频率一致）
- **速度控制**：通过调整时间间隔实现回放速度控制

### 数据流转

```
.npz 文件（replay_data格式）
    ↓
load_trajectory()  →  trajectory 字典
    ↓
replay_trajectory()  →  时间间隔计算
    ↓
移动到起始位置 (step 0)
    ↓
循环执行 (step 1 到 step N-1)
    ├─ 构造动作 [joint_pos(7), gripper(1)]
    ├─ env.step(action)  →  Kinova API
    └─ time.sleep(wait_time)  →  时序控制
```

## ⚠️ 注意事项

1. **安全第一**
   - 首次使用建议使用较慢的回放速度（`--playback-speed 0.5`）
   - 确保回放区域无障碍物
   - 准备随时按 `Ctrl+C` 中断回放

2. **数据格式**
   - 只能使用 `replay_data` 目录中的数据文件
   - 不能使用 `libero_format` 目录中的训练数据文件
   - 确保轨迹文件完整且未损坏

3. **网络连接**
   - 确保机器人 IP 地址正确
   - 如果夹爪未连接，可以省略 `--gripper-ip` 参数（但夹爪控制可能失败）

4. **轨迹完整性**
   - 如果轨迹数据不完整或时间戳异常，程序会回退到固定频率（60Hz）
   - 部分轨迹回放时，确保起始和结束步数有效

5. **回放速度**
   - 过快的回放速度可能导致机器人无法及时到达目标位置
   - 过慢的回放速度会延长回放时间
   - 建议范围：0.3 - 2.0

## 🐛 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| **无法连接机器人** | IP 地址错误或网络问题 | 检查网络连接，使用 `ping` 测试机器人 IP |
| **找不到轨迹文件** | 数据目录不存在或文件路径错误 | 使用 `--data-path` 直接指定文件路径，或检查数据目录结构 |
| **回放中断** | 机器人到达限位或急停触发 | 检查机器人状态，按 `ESC` 或 `q` 键清除急停，重新开始 |
| **回放不准确** | 时间戳缺失或数据不完整 | 程序会自动使用固定频率（60Hz），这是正常的 |
| **夹爪不动作** | 夹爪控制器未连接或 IP 错误 | 检查夹爪控制器连接，确认 IP 地址正确 |
| **步数超出范围** | `--start-step` 或 `--end-step` 参数无效 | 检查轨迹文件的实际步数，使用有效范围 |

## 📊 回放进度

程序会在回放过程中显示进度信息：

- 每 50 步显示一次进度百分比
- 格式：`回放进度: 50.0% (250/500)`
- 回放完成后显示：`轨迹回放完成！`

## 🔗 相关文档

- [数据采集指南](DATA_COLLECTION.md)：了解如何采集轨迹数据
- [推理脚本 README](README.md)：了解如何使用策略进行推理
- [Kinova 机器人环境](kinova_env.py)：了解机器人环境的实现

## 📚 参考

- [OpenPI 项目](https://github.com/Physical-Intelligence/openpi)
- [Kinova Gen3 文档](https://www.kinovarobotics.com/resources)
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)

## 💡 使用建议

1. **测试新轨迹**：首次回放新轨迹时，建议使用较慢速度（0.3-0.5倍速）进行测试
2. **部分回放**：如果只需要验证部分轨迹，使用 `--start-step` 和 `--end-step` 参数
3. **速度调整**：根据任务需求调整回放速度，精细操作使用慢速，快速演示使用快速
4. **数据备份**：回放前建议备份原始轨迹数据，防止意外修改
