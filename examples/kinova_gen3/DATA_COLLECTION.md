# Kinova Gen3 数据采集指南

本文档介绍如何使用 `collect_data.py` 脚本采集 Kinova Gen3 机械臂的演示数据。采集的数据格式兼容 LIBERO 和 OpenPi 训练流程。

## 📋 功能概述

该脚本支持以下功能：
*   **多模态采集**：同步记录外部相机图像、腕部相机图像、机械臂状态（关节/末端）和夹爪状态。
*   **固定频率**：支持以可调频率（默认 60Hz）进行稳定采样。
*   **示教模式**：支持记录手动拖动示教（Teaching mode）的过程。
*   **数据格式**：直接生成 `.npz` 文件，包含训练所需的 observation 和 action。

## 🛠️ 准备工作

1.  **硬件连接**：
    *   Kinova 机械臂已连接并处于正常状态。
    *   Arduino 夹爪控制器已连接。
    *   两台 RealSense 相机（外部和腕部）已连接 USB。

2.  **环境配置**：

    数据采集需要以下依赖，请确保已正确安装：

    **基础依赖（已在项目根目录通过 `uv sync` 安装）：**
    - `kortex_api`：Kinova SDK（需单独从官网下载安装）
    - `pyrealsense2`：RealSense 相机驱动
    - `numpy`, `tyro`：基础库

    **数据采集特有依赖（需要额外安装）：**
    - `scipy`：用于四元数转换
    - `opencv-python`：用于图像处理（已在项目依赖中，但需确认已安装）

    **安装步骤：**

    ```bash
    # 在项目根目录（openpi/）
    cd <项目根目录>

    # 1. 安装项目基础依赖（如果还没安装）
    GIT_LFS_SKIP_SMUDGE=1 uv sync

    # 2. 安装数据采集特有依赖
    uv pip install scipy

    # 3. 安装其他必需依赖（如果还没安装）
    uv pip install pyrealsense2 pynput

    # 4. 安装 kortex_api（从 Kinova 官网下载）
    pip install kortex_api-2.6.0.post3-py3-none-any.whl
    ```

    更多依赖安装说明请参考主 [`README.md`](../README.md)。

## 🚀 使用方法

### 1. 启动采集程序

在项目根目录运行：

```bash
# 方式 1：使用 uv run（推荐）
uv run python examples/kinova_gen3/collect_data.py

# 方式 2：激活虚拟环境后运行
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows
python examples/kinova_gen3/collect_data.py
deactivate
```

脚本启动后会初始化机器人和相机，并打印控制说明。

### 2. 操作控制

脚本通过键盘监听进行交互控制：

| 按键 | 功能 | 说明 |
| :--- | :--- | :--- |
| **`Enter`** | **开始/停止录制** | 按下开始录制，再次按下停止并保存数据 |
| **`O`** | 张开夹爪 | Open Gripper |
| **`P`** | 闭合夹爪 | Close Gripper |
| **`R`** | 复位机器人 | 机械臂回到初始 Home 位置 |
| **`H`** | 打印状态 | 显示当前关节角度、夹爪状态和录制进度 |
| **`ESC`** | 退出 | 退出程序 |

### 3. 采集流程示例（手动示教）

1.  **准备**：
    *   将机械臂设置为**示教模式/零力模式**（通过手柄或 Web App 设置，使机械臂可以被手动拖动）。
    *   将机械臂移动到任务起始位置。

2.  **开始**：
    *   在终端按 **`Enter`** 键。
    *   看到 `🎬 录制开始!` 提示。

3.  **演示**：
    *   手动拖动机械臂完成任务（如抓取物体、移动、放置）。
    *   如果是遥操作控制夹爪，使用键盘 **`O`** (张开) 和 **`P`** (闭合)。脚本会记录夹爪的开闭状态。

4.  **结束**：
    *   任务完成后，再次按 **`Enter`** 键。
    *   脚本显示 `⏹️ 录制停止` 并自动保存数据。
    *   如果数据有效（长度足够），会提示 `📋 Episode saved!`。

5.  **重复**：
    *   重复上述步骤，直到采集够所需的演示数量（默认 10 条）。

## 📂 数据输出

数据保存在脚本中配置的目录下（代码中默认路径，可在 `collect_data.py` 中修改 `self.data_dir` 变量），按任务名和时间戳组织：

```
data/
└── General_manipulation_task_20240113_120000/
    ├── session_info.json          # 会话元数据
    ├── collection_summary.json    # 采集摘要
    ├── libero_format/             # 【训练用】LIBERO 格式数据
    │   ├── episode_001_libero_....npz
    │   └── ...
    └── replay_data/               # 【调试用】详细回放数据
        ├── episode_001_replay_....npz
        └── ...
```

**注意**：数据保存路径可在 `collect_data.py` 中修改 `self.data_dir` 变量。

*   **`libero_format/*.npz`**：包含 `agent_images`, `wrist_images`, `states` (8D), `actions` (7D) 等字段，可直接用于 OpenPi 数据转换。

## 🔄 数据格式转换

采集到的 `.npz` 数据需要转换为 LeRobot 数据集格式，才能用于 OpenPi 训练。

使用提供的转换脚本：

```bash
uv run examples/kinova_gen3/convert_to_lerobot.py \
    --data-dir examples/kinova_gen3/data/<任务目录>/libero_format \
    --repo-name kinova_gen3_dataset
```

### 参数说明

*   `--data-dir`: **[必填]** 包含 `.npz` 文件的目录路径 (通常是采集数据下的 `libero_format` 目录)。
*   `--repo-name`: 输出数据集名称，默认为 `kinova_gen3_dataset`。
*   `--force-override`: 如果输出目录已存在，强制覆盖。
*   `--push-to-hub`: 转换后自动上传到 Hugging Face Hub (需要登录)。

### 输出路径

转换后的数据集默认保存在 LeRobot 缓存目录：

```
~/.cache/huggingface/lerobot/<repo_name>
```

例如默认情况下为：`~/.cache/huggingface/lerobot/kinova_gen3_dataset`。

在 OpenPi 训练配置中，你可以直接使用该 `repo_name` 作为数据集 ID。

## ⚠️ 注意事项

*   **采集频率**：默认为 60Hz。如需调整，可修改脚本中 `self.collection_frequency` 的值。
*   **有效性检查**：录制时间过短（< 5 帧）的数据会被自动丢弃。
*   **相机对应**：脚本默认按检测顺序分配外部相机和腕部相机。如需指定，请在代码中设置 `external_camera_serial` 和 `wrist_camera_serial`。
*   **数据目录**：数据保存路径在 `collect_data.py` 的 `self.data_dir` 变量中配置，默认为代码中硬编码的绝对路径。建议修改为相对路径（如 `./data` 或 `../data`）或根据实际需求设置。
*   **虚拟环境**：建议使用项目根目录的统一虚拟环境（通过 `uv sync` 创建），不要在 `kinova_gen3` 目录中创建独立的虚拟环境。
