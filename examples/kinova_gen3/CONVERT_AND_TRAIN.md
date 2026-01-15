# Kinova Gen3 数据转换与 pi05_base 微调指南

本文整理 `examples/kinova_gen3/convert_to_lerobot.py` 的使用方式，以及使用转换后的数据集微调 `pi05_base` 的流程（包含极轻量测试性微调与正式微调）。

## 一、数据转换（npz → LeRobot）

### 1) 脚本功能与输入规则

`convert_to_lerobot.py` 会把 `collect_data.py` 生成的 `*_libero_*.npz` 转成 LeRobot 数据集。  
其默认会在 `data/**/libero_format/` 下自动搜集所有 `*_libero_*.npz`。

支持三种输入方式：

```bash
# 1. 不传 data_dir：自动搜索 data/**/libero_format
uv run examples/kinova_gen3/convert_to_lerobot.py

# 2. 指定目录
uv run examples/kinova_gen3/convert_to_lerobot.py --data-dir /path/to/your/data/libero_format

# 3. 指定单个文件
uv run examples/kinova_gen3/convert_to_lerobot.py --data-dir /path/to/your/data/libero_format/episode_000_libero_xxx.npz
```

### 2) Prompt 设置

脚本新增 `--prompt` 参数，用来统一设置数据集里的任务指令：

```bash
# 默认值就是 "Grab the target object"
uv run examples/kinova_gen3/convert_to_lerobot.py --prompt "Grab the target object"
```

规则：
- 默认 prompt 为 `"Grab the target object"`
- 你传入 `--prompt "xxx"` 时，所有帧都会写入该任务指令
- 若想使用 npz 内的 `task` 字段，传 `--prompt None`（tyro 会解析为 Python 的 `None`）

### 3) 输出位置与 repo_id

输出路径由 `repo_name` 与 `hub_username` 决定：

- `repo_name` 默认：`kinova_gen3_dataset`
- 若提供 `hub_username`，则 repo_id 形如 `username/kinova_gen3_dataset`

输出实际位置会在日志中打印：
```
数据集保存在: <HF_LEROBOT_HOME>/<repo_id>
```

`repo_id` 需要在后续训练配置里保持一致。

---

## 二、微调 `pi05_base`（JAX 训练）

训练逻辑在 `scripts/train.py`，配置在 `src/openpi/training/config.py`。
项目中已内置 `pi05_kinova_selfcollect` 配置，可直接用于自采 Kinova 数据集的 smoke test / 微调；你也可以基于它复制出自己的配置继续改。

### 1) 配置建议（必须修改的关键项）

建议新增配置名：`pi05_kinova_selfcollect`，核心修改点：

- `data.repo_id`: 指向你转换后的数据集 repo_id
- `model.action_dim`: 设为 **32**（与本项目 Kinova 转换脚本保持一致：8 维有效动作 + padding 到 32）
- `model.action_horizon`: 设为与你推理时的 `open-loop-horizon` 一致（如 8 或 15；仅做 smoke test 时可先设为 1）
- `weight_loader`: 使用 `pi05_base` 作为初始权重  
  `gs://openpi-assets/checkpoints/pi05_base/params`

> 说明：`collect_data.py` 的原始 `states/actions` 是 **8 维**（7关节 + 1夹爪），但 `convert_to_lerobot.py` 会把 `state/actions` **pad 到 32 维**写入 LeRobot 数据集，因此训练侧 `action_dim` 应与 32 对齐（有效维度在前 8，后 24 为 padding）。

---

## 三、GPU OOM（图像 dtype 转换）原因与本项目修复

### 1) 现象
在 GPU 上跑 `scripts/train.py` 时，可能在**取第一批数据**阶段就 OOM，堆栈通常指向：
- `src/openpi/training/data_loader.py`：把 batch 变成 JAX device array（搬到 GPU）
- `src/openpi/models/model.py::Observation.from_dict`：对 `uint8` 图像执行 `astype(float32)` 并归一化

### 2) 根因
当图像仍是 `uint8` 时，`Observation.from_dict` 会执行 `uint8 -> float32` 转换。  
如果此时 batch 已经在 GPU 上，这个转换会在 GPU 上产生**额外的 float32 临时 buffer**（峰值显存≈原图像+转换后的图像+中间临时），在多视角/较大 batch 时很容易触发 OOM。

### 3) 本项目的改动（方案一：把转换前移到 CPU）
为避免在 GPU 上做 dtype 转换，本项目把 “`uint8` → `float32[-1,1]`” 前移到了数据 transforms（CPU 侧）：
- 新增 transform：`src/openpi/transforms.py::ConvertImagesToFloat32Minus1To1`
- 并在 `src/openpi/training/config.py::ModelTransformFactory` 中将其插入到 `ResizeImages` **之后**（仍在 CPU 上进行 resize），在 batch 搬到 GPU 之前完成 dtype/归一化转换

这样 GPU 侧拿到的图像已经是 `float32[-1,1]`，不会再触发 `Observation.from_dict` 的 GPU `astype` 峰值分配，从而规避该类 OOM。

---

## 四、极轻量测试性微调（快速 sanity check）

用于验证数据管线是否能正常跑通，推荐设置很小的训练步数和 batch size。

### 建议配置
- `num_train_steps`: 1~5（只验证能前向/反向/保存 checkpoint）
- `batch_size`: 尽量小（例如 1 或 2；注意需满足 `batch_size % jax.device_count() == 0`）
- `wandb_enabled`: `False`（可选）
- `save_interval`: 1（方便立刻看到 checkpoint）

### 步骤

1. 计算归一化统计（可加 `--max-frames` 限制计算量）：
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_kinova_selfcollect --max-frames 16
```

2. 启动训练：
```bash
# 如需在单卡上跑（例如另一张卡正在跑 server），可用 CUDA_VISIBLE_DEVICES 限制设备
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
uv run scripts/train.py pi05_kinova_selfcollect --exp-name=smoke_test --overwrite
```

如果能顺利跑到保存 checkpoint，就说明数据与训练链路基本可用。

---

## 五、正式微调（完整训练）

在确认训练链路无误后，切换到正式配置：

### 推荐配置
- `num_train_steps`: 20k~100k（视数据量与收敛情况调整）
- `batch_size`: 32~256（视显存）
- `save_interval`: 1000
- `ema_decay`: 0.999（可保留 `pi05_libero` 默认）

### 正式训练命令
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_kinova_selfcollect

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_kinova_selfcollect --exp-name=finetune_v1 --overwrite
```

训练完成后，checkpoint 会保存在：
```
checkpoints/<config_name>/<exp_name>/<step>/
```

---

## 六、评估 / 推理（最小化验证链路）

这里给两种“只验证能推理”的方式：

### 方式 A：启动 policy server + simple_client（不依赖真机）

1) 启动 server（把 `--policy.dir` 指向你训练出的 step 目录）：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_kinova_selfcollect \
  --policy.dir=checkpoints/pi05_kinova_selfcollect/<exp_name>/<step>
```

2) 用 simple_client 连上去打通一次推理（LIBERO 输入格式）：

```bash
uv run examples/simple_client/main.py --env LIBERO --host 127.0.0.1 --port 8000
```

### 方式 B：上真机评估（Kinova Gen3）

真机推理脚本参考 `examples/kinova_gen3/README.md` 与 `examples/kinova_gen3/main.py`（DROID 输入格式）。  
如果你希望把“自采数据微调后的 checkpoint”用于真机 `main.py`，需要确保 server 侧的 `policy.config`（输入 key、action_horizon、动作空间解释方式）与你的真机脚本一致；建议基于 `pi05_kinova`/`pi05_kinova_selfcollect` 进一步整理一份专用的 inference config。

---

## 七、常见问题

- **Prompt 相关报错**  
  训练时若提示缺少 prompt，检查：
  1) 是否在转换时写入了 `task`  
  2) `prompt_from_task` 是否设为 `True`  
  3) 是否在转换时使用了 `--prompt` 参数  

- **Normalization stats 缺失**  
  必须先运行 `scripts/compute_norm_stats.py`。

- **动作维度不匹配**  
  训练侧 `action_dim` 必须与 LeRobot 数据一致：本项目 Kinova 转换脚本会把动作 pad 到 **32**（前 8 有效）。

---

如需我补充：
- 配置模板直接写进 `config.py`
- 适配 Kinova 关节动作（delta/absolute）
- 训练后部署推理 server 的命令

可以继续告诉我你的具体需求。
