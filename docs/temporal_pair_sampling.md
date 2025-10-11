# 时间配对采样 (Temporal Pair Sampling)

## 概述

时间配对采样是一种特殊的数据加载策略，它在同一个batch中同时加载来自同一轨迹中相差固定时间步的观测和动作对。这对于某些需要学习时间关系的任务特别有用。

## 工作原理

当启用时间配对采样时：
- 每个batch中的样本会成对出现
- 每对样本来自同一条轨迹
- 两个样本之间相差固定的时间步数（默认为5）

### 示例

假设 `batch_size=32` 且 `temporal_pair_offset=5`：

- Batch包含32个样本
- 这32个样本来自16条不同的轨迹
- 样本组织方式：
  - 样本0和样本1：来自轨迹A，相差5个时间步
  - 样本2和样本3：来自轨迹B，相差5个时间步
  - ...
  - 样本30和样本31：来自轨迹P，相差5个时间步

## 使用方法

### 方法1：通过配置文件

在训练配置中添加以下字段：

```python
TrainConfig(
    name="my_config",
    # ... 其他配置 ...
    use_temporal_pairs=True,        # 启用时间配对采样
    temporal_pair_offset=5,          # 时间偏移量（时间步数）
)
```

### 方法2：通过命令行参数

```bash
python scripts/train.py pi0_libero \
    --exp_name my_experiment \
    --use_temporal_pairs True \
    --temporal_pair_offset 5
```

### 方法3：在代码中直接调用

```python
from openpi.training import data_loader, config

# 创建配置
train_config = config.get_config("pi0_libero")

# 创建数据加载器
loader = data_loader.create_data_loader(
    train_config,
    shuffle=True,
    use_temporal_pairs=True,
    temporal_pair_offset=5,
)

# 迭代数据
for observation, actions in loader:
    # observation和actions的shape为 (batch_size, ...)
    # 其中batch_size必须是偶数
    # 索引0和1来自同一轨迹，相差5个时间步
    # 索引2和3来自同一轨迹，相差5个时间步
    # ...
    pass
```

## 注意事项

1. **Batch Size必须是偶数**：因为样本是成对的，所以batch_size必须是偶数。

2. **数据集要求**：
   - 目前仅支持使用LeRobot格式的数据集（通过`create_torch_data_loader`）
   - 不支持RLDS格式的数据加载器（会自动回退到标准采样）

3. **分布式训练**：目前时间配对采样与PyTorch DDP的DistributedSampler不兼容。如果尝试同时使用会抛出错误。

4. **轨迹长度**：如果某条轨迹的长度小于 `temporal_pair_offset`，该轨迹的样本将不会出现在训练中。

5. **采样效率**：由于只能使用满足时间间隔要求的样本对，实际可用的训练样本数可能会减少。

## 示例：训练一个Libero模型

```bash
# 使用默认配置（不使用时间配对采样）
python scripts/train.py pi0_libero --exp_name libero_baseline

# 使用时间配对采样，偏移量为5
python scripts/train.py pi0_libero \
    --exp_name libero_temporal_5 \
    --use_temporal_pairs True \
    --temporal_pair_offset 5

# 使用时间配对采样，偏移量为10
python scripts/train.py pi0_libero \
    --exp_name libero_temporal_10 \
    --use_temporal_pairs True \
    --temporal_pair_offset 10
```

## 实现细节

时间配对采样通过 `TemporalPairSampler` 类实现，该类位于 `src/openpi/training/data_loader.py`。

### 高效采样策略

主要步骤：
1. **预构建阶段（仅执行一次）**：
   - 分析数据集，找出每个episode（轨迹）的边界
   - 记录所有可以作为"起始点"的有效样本索引
   - 条件：样本后面还有至少 `temporal_pair_offset` 个样本

2. **采样阶段（每个batch执行）**：
   - 从有效起始样本中随机采样 `batch_size/2` 个样本
   - 对于每个起始样本 `idx`，动态计算配对样本 `idx + temporal_pair_offset`
   - 组织成交错序列：`[idx1, idx1+offset, idx2, idx2+offset, ...]`

### 性能优势

相比预先构建所有配对的方法，这种动态配对策略：
- ✅ **内存效率高**：只存储起始索引，内存占用从 O(n×pairs) 降到 O(n)
- ✅ **初始化快**：不需要预先生成所有配对
- ✅ **代码简洁**：逻辑更清晰，易于维护
- ✅ **灵活性强**：可以轻松支持动态偏移等扩展功能

## 常见问题

### Q: 为什么我的数据集没有找到任何有效的配对？

A: 可能的原因：
- `temporal_pair_offset` 设置得太大，超过了大部分轨迹的长度
- 数据集中的轨迹太短

解决方法：尝试减小 `temporal_pair_offset` 的值。

### Q: 我可以在推理时使用时间配对采样吗？

A: 不建议。时间配对采样主要用于训练时学习时间关系。在推理时，策略应该能够处理单个时间步的输入。

### Q: 如何验证时间配对采样是否正常工作？

A: 检查日志输出，应该会看到类似以下的信息：
```
找到 XXXX 个有效的时间配对样本 (时间偏移=5)
```

此外，可以在训练开始时检查第一个batch，验证相邻的样本对是否真的来自同一轨迹。

