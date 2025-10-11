# 时间配对采样功能说明

## 功能概述

我已经为OpenPI项目添加了**时间配对采样**功能。这个功能允许在同一个batch中同时加载来自同一轨迹中相差固定时间步的观测和动作对。

### 核心特性

- ✅ 在同一batch中配对加载同一轨迹的样本
- ✅ 可自定义时间偏移量（默认5个时间步）
- ✅ 支持LeRobot格式数据集
- ✅ 完全向后兼容（默认不启用）

## 修改的文件

### 1. `src/openpi/training/data_loader.py`
- **新增**：`TemporalPairSampler` 类 - 实现时间配对采样逻辑
- **修改**：`create_data_loader` 函数 - 添加 `use_temporal_pairs` 和 `temporal_pair_offset` 参数
- **修改**：`create_torch_data_loader` 函数 - 集成时间配对采样器

### 2. `src/openpi/training/config.py`
- **新增**：`TrainConfig.use_temporal_pairs` - 是否启用时间配对采样
- **新增**：`TrainConfig.temporal_pair_offset` - 时间偏移量配置

### 3. `scripts/train.py`
- **修改**：调用 `create_data_loader` 时传递配置参数

### 4. 新增文档
- `docs/temporal_pair_sampling.md` - 详细的使用文档
- `test_temporal_pair_sampling.py` - 功能测试脚本
- `TEMPORAL_PAIR_SAMPLING_README.md` - 本文件

## 使用示例

### 基本使用

```bash
# 训练时启用时间配对采样（偏移量为5）
python scripts/train.py pi0_libero \
    --exp_name my_experiment \
    --use_temporal_pairs True \
    --temporal_pair_offset 5
```

### 在配置中启用

```python
TrainConfig(
    name="my_config",
    # ... 其他配置 ...
    batch_size=32,              # 必须是偶数
    use_temporal_pairs=True,    # 启用时间配对采样
    temporal_pair_offset=5,     # 时间偏移量
)
```

### 效果说明

当 `batch_size=32` 且 `temporal_pair_offset=5` 时：

```
Batch中的样本组织：
├─ 样本 0: 轨迹A，时刻t
├─ 样本 1: 轨迹A，时刻t+5  ← 与样本0配对
├─ 样本 2: 轨迹B，时刻t
├─ 样本 3: 轨迹B，时刻t+5  ← 与样本2配对
├─ ...
├─ 样本30: 轨迹P，时刻t
└─ 样本31: 轨迹P，时刻t+5  ← 与样本30配对

总共：32个样本，来自16条不同轨迹
```

## 测试

运行测试脚本验证功能：

```bash
python test_temporal_pair_sampling.py
```

预期输出：
```
================================================================================
时间配对采样功能测试
================================================================================

开始测试标准采样（不使用时间配对）...
数据加载器创建成功！
成功获取一个batch
标准采样测试成功！

开始测试时间配对采样...
配置：batch_size=4, use_temporal_pairs=True, temporal_pair_offset=5
找到 XXXX 个有效的时间配对样本 (时间偏移=5)
数据加载器创建成功！
成功获取一个batch
✓ Batch size正确：4
✓ Batch size是偶数
测试成功！时间配对采样功能正常工作。

================================================================================
✓ 所有测试通过！
================================================================================
```

## 实现细节

### TemporalPairSampler 类

主要方法：
- `__init__`: 初始化采样器，验证batch_size为偶数
- `_build_episode_mapping`: 构建episode到样本索引的映射，找出所有有效的样本对
- `__iter__`: 生成batch中的样本索引
- `__len__`: 返回总样本数

核心逻辑：
1. 解析数据集的 `episode_data_index`，获取每个episode的起止索引
2. 对于每个episode，生成所有满足时间间隔的样本对
3. 在迭代时，按batch_size组织样本对，并展开成连续索引

### 兼容性

- ✅ 向后兼容：默认不启用，不影响现有代码
- ✅ 与现有transforms兼容
- ⚠️ 不支持RLDS数据加载器（会回退到标准采样）
- ⚠️ 不支持PyTorch DDP的DistributedSampler（会抛出错误）

## 应用场景

时间配对采样适用于：

1. **时序对比学习**：学习同一轨迹中不同时刻的状态表示
2. **未来预测**：给定当前状态，预测未来N步后的状态或动作
3. **轨迹一致性学习**：确保策略在同一轨迹的不同时刻产生连贯的行为
4. **时间因果关系建模**：学习动作和结果之间的时间延迟关系

## 注意事项

1. **Batch Size**：必须设置为偶数
2. **数据集格式**：目前仅支持LeRobot格式
3. **轨迹长度**：确保数据集中的轨迹长度 > `temporal_pair_offset`
4. **训练样本数**：实际可用样本数可能减少（取决于有效配对数量）

## 未来改进方向

可能的扩展：
- [ ] 支持RLDS数据加载器
- [ ] 支持分布式训练（与DistributedSampler兼容）
- [ ] 支持多个时间偏移量（例如同时采样t+5和t+10）
- [ ] 支持负时间偏移（采样过去的状态）
- [ ] 动态时间偏移（根据轨迹长度自适应调整）

## 常见问题

### Q: 为什么需要batch_size是偶数？
A: 因为样本是成对的（每对2个样本），所以总数必须是偶数。

### Q: 如果我的轨迹很短怎么办？
A: 减小 `temporal_pair_offset` 的值，或者增加数据集中的轨迹长度。

### Q: 这会影响训练速度吗？
A: 轻微影响。采样器需要在初始化时构建索引映射，但这只在训练开始时执行一次。

### Q: 可以与数据增强一起使用吗？
A: 可以！时间配对采样在数据加载层面工作，不影响后续的数据transforms。

## 贡献

如果您有任何问题或建议，欢迎：
1. 查看详细文档：`docs/temporal_pair_sampling.md`
2. 运行测试脚本：`test_temporal_pair_sampling.py`
3. 提交Issue或Pull Request

## 总结

时间配对采样是一个简单但强大的功能，可以帮助模型更好地学习时序关系。通过在batch中配对来自同一轨迹的样本，模型可以更有效地学习动作的时序效果和状态演变规律。

**开始使用：**
```bash
python scripts/train.py <your_config> \
    --exp_name temporal_pair_experiment \
    --use_temporal_pairs True \
    --temporal_pair_offset 5
```

祝训练顺利！🚀

