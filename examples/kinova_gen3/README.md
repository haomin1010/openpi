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

```bash
python main.py \
    --robot-ip 192.168.1.10 \
    --gripper-ip 192.168.1.43 \
    --external-serial <外部相机序列号> \
    --wrist-serial <腕部相机序列号> \
    --remote-host <策略服务器IP> \
    --remote-port 8000
```

### 步骤 3: 输入任务指令

程序启动后，输入自然语言任务指令，例如：
- "pick up the red cup"
- "put the bottle on the table"
- "open the drawer"

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--robot-ip` | `192.168.1.10` | Kinova 机械臂 IP |
| `--gripper-ip` | `192.168.1.43` | 夹爪控制器 IP |
| `--external-serial` | None | 外部相机序列号 |
| `--wrist-serial` | None | 腕部相机序列号 |
| `--remote-host` | `0.0.0.0` | 策略服务器 IP |
| `--remote-port` | `8000` | 策略服务器端口 |
| `--max-timesteps` | `600` | 最大执行步数 |
| `--open-loop-horizon` | `8` | 开环执行步数 |

## 急停功能

- **键盘急停**: 按 `ESC` 或 `q` 键触发急停
- 急停后机械臂会立即停止运动
- 需要手动清除急停状态后才能继续操作

## 动作空间

| 维度 | 内容 | 范围 |
|------|------|------|
| 0-6 | 7 个关节位置 | 弧度 |
| 7 | 夹爪状态 | 0.0=张开，1.0=闭合（二值动作，非连续角度值） |

注意：本实现使用**关节位置控制**模式，策略输出的动作会被解释为目标关节位置。

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

1. **安全第一**: 首次测试时请降低速度，确保周围无障碍物
2. **网络延迟**: 建议使用有线网络连接策略服务器，延迟 0.5-1 秒是正常的
3. **相机位置**: 确保外部相机和腕部相机能够清晰看到操作场景
4. **关节限位**: 注意 Kinova Gen3 的关节角度限制，避免超限

## 参考

- [OpenPI 项目](https://github.com/Physical-Intelligence/openpi)
- [Kinova Gen3 文档](https://www.kinovarobotics.com/resources)
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)

