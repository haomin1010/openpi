# Pi0 sample_actions 测试脚本说明

## 概述

提供了两个测试脚本来验证 `Pi0.sample_actions()` 方法的功能：

1. **test_sample_actions_simple.py** - 快速测试版本（推荐首次运行）
2. **test_sample_actions.py** - 完整测试套件

## 快速开始

### 1. 简单测试（推荐）

```bash
cd /path/to/openpi
python test_sample_actions_simple.py
```

**测试内容：**
- ✅ 模型初始化
- ✅ 观察数据创建
- ✅ 动作采样
- ✅ 输出形状验证
- ✅ 数值有效性检查

**预期输出：**
```
============================================================
快速测试 sample_actions
============================================================
✓ 配置创建完成
  action_dim=14, action_horizon=10

正在初始化模型...
✓ 模型初始化完成: Pi0

正在创建观察数据...
✓ 观察数据创建完成
  batch_size=2
  state shape: (2, 14)
  ...

正在采样动作...
  num_steps=10

✓ 采样完成!
  输出形状: (2, 12, 14)
  期望形状: (2, 12, 14)
  形状匹配: True
  值范围: [...]
  均值: ...

============================================================
✅ 所有检查通过!
============================================================
```

---

### 2. 完整测试套件

```bash
python test_sample_actions.py
```

**包含以下测试：**

#### 测试 1: 基础功能测试
- 验证基本的采样功能
- 检查输出形状和数据类型
- 打印详细的统计信息

#### 测试 2: 自定义噪声测试
- 使用自定义初始噪声进行采样
- 验证模型能正确处理外部提供的噪声

#### 测试 3: Pi05 模式测试
- 测试 Pi0.5 版本（使用 adaRMS）
- 验证两种模式的兼容性

#### 测试 4: 不同采样步数测试
- 测试 num_steps = [1, 5, 10, 20]
- 分析不同步数对结果的影响

#### 测试 5: 确定性测试
- 验证相同随机种子产生相同结果
- 验证不同随机种子产生不同结果

---

## 测试参数说明

### 配置参数

```python
Pi0Config(
    action_dim=14,          # 动作维度
    action_horizon=10,      # 动作序列长度
    pi05=False,            # 是否使用 Pi0.5 模式
)
```

### sample_actions 参数

```python
model.sample_actions(
    rng,                   # JAX 随机数生成器
    observation,           # 观察数据
    num_steps=10,          # 扩散采样步数（默认10）
    noise=None,            # 可选：自定义初始噪声
)
```

### 观察数据结构

```python
Observation(
    images={
        "base_0_rgb": (batch, 224, 224, 3),
        "left_wrist_0_rgb": (batch, 224, 224, 3),
        "right_wrist_0_rgb": (batch, 224, 224, 3),
    },
    image_masks={...},                    # 每个图像的有效性掩码
    state=(batch, action_dim),            # 机器人状态
    tokenized_prompt=(batch, max_len),    # 可选：语言指令
    tokenized_prompt_mask=(batch, max_len),
)
```

### 输出格式

```python
actions.shape = (batch_size, action_horizon + 2, action_dim)
```

注意：输出包含 `action_horizon + 2` 个时间步（比配置多2步）。

---

## 常见问题

### Q: 为什么输出是 action_horizon + 2 而不是 action_horizon?

A: 这是代码中的设计，在 `sample_actions` 中噪声初始化为：
```python
noise = jax.random.normal(rng, (batch_size, self.action_horizon + 2, self.action_dim))
```
多出的2步可能用于提供额外的上下文或冗余。

### Q: Pi05 和非 Pi05 模式的区别？

A: 主要区别在于：
- **非 Pi05**: 使用 state token + 时间信息通过 MLP 与动作拼接
- **Pi05**: 使用 adaRMS 机制注入时间信息，没有独立的 state token

### Q: num_steps 应该设置多少？

A: 
- 更少的步数（1-5）：速度快，但质量可能较低
- 更多的步数（10-20）：质量更好，但速度慢
- 默认值 10 是一个平衡点

### Q: 如何使用自定义噪声？

A: 
```python
custom_noise = jax.random.normal(
    key, 
    (batch_size, action_horizon + 2, action_dim)
)
actions = model.sample_actions(
    rng, 
    observation, 
    noise=custom_noise
)
```

---

## 性能提示

### 加速建议

1. **使用 JIT 编译**：
```python
sample_fn = jax.jit(model.sample_actions)
actions = sample_fn(rng, observation, num_steps=10)
```

2. **批处理**：同时处理多个观察以提高 GPU 利用率

3. **减少采样步数**：在推理时使用较少的步数（如 5）

### 内存使用

- 每个图像：~600KB (224x224x3 float32)
- 模型参数：~2GB（取决于配置）
- 中间激活：取决于 batch_size 和序列长度

---

## 调试提示

### 检查数值稳定性

```python
assert jnp.isfinite(actions).all(), "输出包含 NaN 或 Inf"
```

### 打印中间状态

在 `pi0.py` 的 `sample_actions` 方法中添加：
```python
print(f"Step {step}: x_t range [{x_t.min():.4f}, {x_t.max():.4f}]")
```

### 可视化采样过程

```python
# 记录每一步的输出
trajectory = []
# 修改 step 函数记录 x_t
```

---

## 扩展测试

### 添加自定义测试

```python
def test_my_scenario():
    config = Pi0Config(...)
    model = config.create(jax.random.key(0))
    
    # 创建特定场景的观察数据
    observation = ...
    
    # 测试
    actions = model.sample_actions(...)
    
    # 验证
    assert ...
```

### 性能基准测试

```python
import time

start = time.time()
for _ in range(100):
    actions = model.sample_actions(rng, observation)
elapsed = time.time() - start
print(f"平均时间: {elapsed/100:.4f} 秒/次")
```

---

## 故障排除

| 错误 | 可能原因 | 解决方案 |
|------|---------|---------|
| 形状不匹配 | 观察数据格式错误 | 检查 `observation.state.shape[0]` 是否一致 |
| CUDA OOM | batch_size 太大 | 减小 batch_size 或使用 CPU |
| NaN 输出 | 数值不稳定 | 检查输入数据范围，使用 bfloat16 |
| 速度慢 | 未使用 JIT | 使用 `jax.jit` 编译函数 |

---

## 参考

- Pi0 论文: [链接]
- OpenPI 文档: [链接]
- JAX 文档: https://jax.readthedocs.io/

