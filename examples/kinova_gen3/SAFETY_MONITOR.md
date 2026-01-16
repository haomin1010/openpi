# SAFETY_MONITOR 使用说明

本文档说明如何使用 `boundingbox.txt` 设置 `safety_box`，以及在机械臂运行过程中进行安全检测。

## 1. 通过 boundingbox.txt 设置 safety_box

`boundingbox.txt` 文件位于 `examples/kinova_gen3/` 目录，包含若干行数据，每行格式如下：

```
[x, y, z]
```

说明：
- `x, y, z`：空间坐标（米），表示安全区域中的一个点
- **每行必须且仅包含3个参数**（x, y, z）
- 文件可以包含任意数量的点，用于定义安全区域的边界

`safety_box` 会根据每个点的 x/y/z 取最小值和最大值，形成安全长方体：

```
x_min = min(x)
x_max = max(x)
y_min = min(y)
y_max = max(y)
z_min = min(z)
z_max = max(z)
```

解析逻辑位于：
- `safety_box.py` 中的 `load_safety_box()`

后续可能会考虑使用更加复杂的方法，采集多个点形成包络面

## 2. 在机械臂运行过程中使用 safety_box 进行安全检测

安全检测通过 `SafetyMonitor` 完成，核心流程如下：

1. **读取 URDF**（默认 `GEN3_URDF_V12_with_dampint.urdf`）
2. **读取 boundingbox.txt** 并生成 `safety_box`
3. 运行过程中：
   - 根据关节角计算各关节与末端在基座坐标系的位置
   - 判断是否全部在安全长方体内
   - 超出时按模式触发软急停或硬急停

### 2.1 安全检测模块

`SafetyMonitor` 位于 `safety_monitor.py`，支持两种模式：

- `soft`：软急停，超出安全区时忽略控制指令
- `hard`：硬急停，超出安全区时触发急停并锁死机械臂

默认监督关节 2-8（即关节 2-7 和末端位置），不监督第一个关节（`Actuator1`），因为它通常固定不动。

### 2.2 在回放中启用安全检测

在 `replay_routine.py` 中使用以下参数启用：

```
python replay_routine.py --safety --safety-mode soft
```

硬急停：

```
python replay_routine.py --safety --safety-mode hard
```

指定 URDF 与 boundingbox：

```
python replay_routine.py --safety \
  --safety-urdf /home/kinova/qyh/openpi_kinova/openpi/examples/kinova_gen3/GEN3_URDF_V12_with_dampint.urdf \
  --safety-bbox /home/kinova/qyh/openpi_kinova/openpi/examples/kinova_gen3/boundingbox.txt
```

#### 2.2.1 指定要监督的关节（推荐）

使用 `--safety-joints` 参数可以精确指定要监督哪些关节位置：

```
# 监督关节 2、3、4 和末端位置（8）
python replay_routine.py --safety --safety-joints "2 3 4 8"

# 监督所有关节和末端（1-8）
python replay_routine.py --safety --safety-joints "1 2 3 4 5 6 7 8"

# 只监督末端位置
python replay_routine.py --safety --safety-joints "8"

# 监督关节 3-7 和末端（不监督关节 1 和 2）
python replay_routine.py --safety --safety-joints "3 4 5 6 7 8"
```

**关节编号说明：**
- `1-7`：对应关节 Actuator1 到 Actuator7
- `8`：末端执行器位置

**默认行为：**
- 如果不指定 `--safety-joints`，默认监督关节 2-8（即不监督关节 1）

#### 2.2.2 旧接口（已废弃，建议使用 --safety-joints）

启用第一个关节检测（旧方法）：

```
python replay_routine.py --safety --safety-check-first-joint
```

注意：此方法已废弃，建议使用 `--safety-joints "1 2 3 4 5 6 7 8"` 来监督所有关节。

### 2.3 在其他控制流程中复用

在任意控制循环中，调用示例：

**方法 1：使用 monitored_joints 参数（推荐）**

```python
from safety_monitor import SafetyMonitor

# 监督关节 2-8（默认行为）
safety_monitor = SafetyMonitor(
    urdf_path="GEN3_URDF_V12_with_dampint.urdf",
    boundingbox_path="boundingbox.txt",
    mode="soft",
    monitored_joints=[2, 3, 4, 5, 6, 7, 8],  # 监督关节 2-7 和末端
)

# 或者监督所有关节和末端
safety_monitor = SafetyMonitor(
    urdf_path="GEN3_URDF_V12_with_dampint.urdf",
    boundingbox_path="boundingbox.txt",
    mode="soft",
    monitored_joints=[1, 2, 3, 4, 5, 6, 7, 8],  # 监督所有关节和末端
)

obs = env.get_observation()
is_safe = safety_monitor.check_and_handle(env, obs["robot_state"]["joint_positions"])
if not is_safe:
    # 软急停：忽略本次控制指令
    return
```

**方法 2：使用 ignore_joint_names 参数（旧接口，已废弃）**

```python
from safety_monitor import SafetyMonitor

safety_monitor = SafetyMonitor(
    urdf_path="GEN3_URDF_V12_with_dampint.urdf",
    boundingbox_path="boundingbox.txt",
    mode="soft",
    ignore_joint_names=["Actuator1"],  # 忽略第一个关节
)

obs = env.get_observation()
is_safe = safety_monitor.check_and_handle(env, obs["robot_state"]["joint_positions"])
if not is_safe:
    # 软急停：忽略本次控制指令
    return
```

**参数说明：**
- `monitored_joints`：要监督的关节编号列表（1-7 表示关节，8 表示末端位置）
- `ignore_joint_names`：要忽略的关节名称列表（旧接口，已废弃，建议使用 `monitored_joints`）

## 3. 常见问题

1. **boundingbox.txt 文件格式要求？**  
   - 每行必须且仅包含3个参数：`[x, y, z]`
   - 不支持包含旋转角度（rx, ry, rz）的格式
   - 安全区仅使用 x/y/z 坐标

2. **安全检测是否依赖真实机器人反馈？**  
   需要 `env.get_observation()` 的关节角反馈，才能正确计算关节位置。

3. **URDF 文件需要和真实机械臂一致吗？**  
   需要一致，否则计算的关节位置会偏差，导致误判。
