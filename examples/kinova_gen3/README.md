cd /home/kinova/qyh/openpi_kinova/openpi
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_kinova \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_base \
  --port=8000


cd /home/kinova/qyh/openpi_kinova/openpi/examples/kinova_gen3
source ../../.venv/bin/activate  # 或者使用 examples/kinova_gen3/.venv
python main.py \
  --remote-host 127.0.0.1 \
  --remote-port 8000 \
  --robot-ip 192.168.1.10 \
  --gripper-ip 192.168.1.43 \
  --external-serial <外部相机序列号> \
  --wrist-serial <腕部相机序列号>





# Kinova Gen3 OpenPI 策略推理

使用 OpenPI 服务器对 Kinova Gen3 机械臂进行 VLA（Vision-Language-Action）策略推理。

## 硬件配置

| 组件 | 型号/规格 |
|------|----------|
| 机械臂 | Kinova Gen3 7DOF |
| 夹爪 | Arduino UDP 控制 |
| 外部相机 | Intel RealSense D435i（左侧视角） |
| 腕部相机 | Intel RealSense D435i |

## 文件结构

```
examples/kinova_gen3/
├── control_gripper.py     # UDP 夹爪控制
├── kinova_env.py          # KinovaRobotEnv 机器人环境
├── realsense_camera.py    # 双 RealSense 相机封装
├── main.py                # 推理入口脚本
└── README.md              # 本文档
```

## 依赖安装

### 1. 安装 kortex_api

从 Kinova 官网下载 kortex_api wheel 包并安装：

```bash
pip install kortex_api-2.6.0.post3-py3-none-any.whl
```

### 2. 安装其他依赖

```bash
pip install pyrealsense2 pynput moviepy tyro numpy pillow tqdm
```

### 3. 安装 openpi-client

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

## 配置

### 1. 查找相机序列号

运行以下命令列出所有连接的 RealSense 相机：

```bash
python realsense_camera.py --list
```

记下两个相机的序列号，分别对应外部相机和腕部相机。

### 2. 网络配置

确保以下设备在同一局域网内：
- Kinova 机械臂（默认 IP: `192.168.1.10`）
- Arduino 夹爪控制器（默认 IP: `192.168.1.43`）
- 策略服务器（运行 OpenPI 的 GPU 机器）

## 使用方法

### 步骤 1: 启动策略服务器

👋 **数据采集指南**：如果你需要采集训练数据，请查看 [数据采集文档](DATA_COLLECTION.md)。

在有 GPU 的机器上启动 OpenPI 策略服务器：

```bash
# 使用预训练的 pi0.5 DROID 模型
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

或者使用其他模型：

```bash
# pi0-FAST DROID
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

### 步骤 2: 运行推理脚本

在机器人控制电脑上运行：

**基本用法（使用默认配置）：**

```bash
python main.py \
    --robot-ip 192.168.1.10 \
    --gripper-ip 192.168.1.43 \
    --external-serial <外部相机序列号> \
    --wrist-serial <腕部相机序列号> \
    --remote-host <策略服务器IP> \
    --remote-port 8000
```

**使用平滑控制和安全检测（默认已启用）：**

```bash
python main.py \
    --robot-ip 192.168.1.10 \
    --gripper-ip 192.168.1.43 \
    --external-serial <外部相机序列号> \
    --wrist-serial <腕部相机序列号> \
    --remote-host <策略服务器IP> \
    --smooth \
    --safety \
    --safety-mode soft
```

**自定义平滑和安全参数：**

```bash
python main.py \
    --robot-ip 192.168.1.10 \
    --gripper-ip 192.168.1.43 \
    --external-serial <外部相机序列号> \
    --wrist-serial <腕部相机序列号> \
    --remote-host <策略服务器IP> \
    --smoothing-window-size 7 \
    --max-linear-velocity 0.03 \
    --safety-joints "2 3 4 5 6 7 8"
```

### 步骤 3: 输入任务指令

程序启动后，输入自然语言任务指令，例如：
- "pick up the red cup"
- "put the bottle on the table"
- "open the drawer"

## 命令行参数

### 硬件配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--robot-ip` | `192.168.1.10` | Kinova 机械臂 IP 地址 |
| `--gripper-ip` | `192.168.1.43` | 夹爪控制器 IP 地址 |
| `--external-serial` | None | 外部相机序列号（左侧视角） |
| `--wrist-serial` | None | 腕部相机序列号 |

### 策略服务器配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--remote-host` | `0.0.0.0` | 策略服务器 IP 地址 |
| `--remote-port` | `8000` | 策略服务器端口 |

### 推理配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-timesteps` | `600` | 最大执行步数 |
| `--open-loop-horizon` | `8` | 开环执行步数（从预测的 action chunk 中执行多少个动作后再查询服务器） |
| `--action-mode` | `absolute` | 动作模式：<br>- `absolute`: 绝对位置模式（默认）<br>- `delta`: 增量模式<br>- `velocity`: 速度模式 |

### 平滑控制参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--smooth` | `True` | 是否启用平滑控制（速度控制和平滑滤波，默认启用） |
| `--no-smooth` | - | 禁用平滑控制（使用原始位置控制模式） |
| `--smoothing-window-size` | `5` | 平滑窗口大小（用于轨迹平滑的点数） |
| `--max-linear-velocity` | `0.05` | 最大线速度 (m/s)，默认 5 cm/s |
| `--max-angular-velocity` | `0.5` | 最大角速度 (rad/s) |
| `--position-gain` | `2.0` | 位置控制增益（比例控制器增益） |
| `--orientation-gain` | `1.0` | 姿态控制增益 |

**平滑控制说明：**
- 平滑控制使用速度控制和平滑滤波实现更平滑的运动
- 适用于精细操作和需要平滑轨迹的任务
- 禁用平滑控制时，使用原始关节位置控制模式

### 安全检测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--safety` | `True` | 是否启用安全检测（默认启用） |
| `--no-safety` | - | 禁用安全检测 |
| `--safety-mode` | `soft` | 安全模式：<br>- `soft`: 软急停，超出安全区时忽略指令（默认）<br>- `hard`: 硬急停，超出安全区时触发急停并锁死机械臂 |
| `--safety-urdf` | `脚本目录/GEN3_URDF_V12_with_dampint.urdf` | URDF 文件路径（用于正运动学计算） |
| `--safety-bbox` | `脚本目录/boundingbox.txt` | boundingbox.txt 文件路径（定义安全区域） |
| `--safety-joints` | `2-8` | 要监督的关节编号，用空格分隔<br>- `1-7`: 对应关节 Actuator1 到 Actuator7<br>- `8`: 末端执行器位置<br>- 默认值：`2-8`（监督关节2-7和末端，不监督关节1）<br>- 示例：`"2 3 4 8"` 表示只监督关节2、3、4和末端 |

**安全检测说明：**
- 安全检测基于 URDF 正运动学计算各关节和末端在基座坐标系中的位置
- 使用 `boundingbox.txt` 定义的安全区域进行检测
- 详细说明请参考 [安全检测文档](SAFETY_MONITOR.md)

## 急停功能

- **键盘急停**: 按 `ESC` 或 `q` 键触发急停
- 急停后机械臂会立即停止运动
- 需要手动清除急停状态后才能继续操作

## 动作空间

| 维度 | 内容 | 范围 |
|------|------|------|
| 0-6 | 7 个关节位置 | 弧度 |
| 7 | 夹爪状态 | 0.0=张开，1.0=闭合（二值动作，非连续角度值） |

**动作执行模式：**

- **平滑控制模式（默认）**：模型输出的关节角度通过 URDF 正运动学转换为末端位置，使用速度控制实现平滑运动
- **原始位置控制模式**（`--no-smooth`）：策略输出的动作直接作为目标关节位置执行
- 动作模式（`--action-mode`）：
  - `absolute`: 绝对位置模式（默认），动作直接作为目标关节位置
  - `delta`: 增量模式，动作是相对当前位置的增量
  - `velocity`: 速度模式，动作是关节速度

## 观察空间

| 字段 | 格式 | 说明 |
|------|------|------|
| `observation/exterior_image_1_left` | (224, 224, 3) uint8 | 外部相机 RGB 图像 |
| `observation/wrist_image_left` | (224, 224, 3) uint8 | 腕部相机 RGB 图像 |
| `observation/joint_position` | (7,) float | 关节位置（弧度） |
| `observation/gripper_position` | (1,) float | 夹爪状态：0.0=张开，1.0=闭合（二值动作） |

## 测试

### 测试相机

```bash
python realsense_camera.py --external-serial <序列号1> --wrist-serial <序列号2> --save
```

### 测试机器人环境

```bash
python kinova_env.py --robot-ip 192.168.1.10 --gripper-ip 192.168.1.43
```

### 测试夹爪

```bash
# 闭合夹爪
python control_gripper.py --host 192.168.1.43 --speed 20.0 --angle 1872

# 张开夹爪
python control_gripper.py --host 192.168.1.43 --speed -20.0 --angle 1872
```

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| 无法连接机械臂 | 检查网络连接和 IP 地址，确保机械臂已开机 |
| 无法检测相机 | 检查 USB 连接，运行 `realsense-viewer` 确认相机正常 |
| 策略服务器超时 | 检查网络连接，尝试使用有线网络减少延迟 |
| 夹爪无响应 | 检查 Arduino 是否在线，使用 `ping` 测试连通性 |
| 急停后无法恢复 | 调用 `env.clear_estop()` 或重启程序 |

## 注意事项

1. **安全第一**: 
   - 首次测试时请降低速度（`--max-linear-velocity 0.03`），确保周围无障碍物
   - **强烈建议启用安全检测**（`--safety`），防止机械臂超出安全区域
   - 软急停模式（默认）会在超出安全区时停止当前动作，如果后续动作能回到安全范围会继续执行
   - 硬急停模式会在超出安全区时立即触发急停，终止整个任务

2. **平滑控制**:
   - 平滑控制（默认启用）提供更平滑的运动，适合精细操作
   - 如需更快的响应速度，可以使用 `--no-smooth` 禁用平滑控制
   - 平滑窗口大小影响平滑程度：较大的窗口（如 7）更平滑但响应更慢，较小的窗口（如 3）响应更快但可能不够平滑

3. **网络延迟**: 建议使用有线网络连接策略服务器，延迟 0.5-1 秒是正常的

4. **相机位置**: 确保外部相机和腕部相机能够清晰看到操作场景

5. **关节限位**: 注意 Kinova Gen3 的关节角度限制，避免超限（安全检测可帮助防止超限）

6. **安全区域配置**: 使用 `tools/get_kinova_state.py --detail` 获取当前关节位置，用于配置 `boundingbox.txt`

## 参考

- [OpenPI 项目](https://github.com/Physical-Intelligence/openpi)
- [Kinova Gen3 文档](https://www.kinovarobotics.com/resources)
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)

