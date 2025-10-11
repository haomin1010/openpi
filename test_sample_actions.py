"""测试 Pi0 模型的 sample_actions 函数"""

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models.pi0_config import Pi0Config
from openpi.models.model import Observation


def create_fake_observation(config: Pi0Config, batch_size: int = 2):
    """创建假的观察数据用于测试"""
    # 创建假的图像数据 (batch_size, 224, 224, 3)，值在 [-1, 1]
    fake_images = {
        "base_0_rgb": jnp.ones((batch_size, 224, 224, 3), dtype=jnp.float32) * 0.5,
        "left_wrist_0_rgb": jnp.ones((batch_size, 224, 224, 3), dtype=jnp.float32) * 0.3,
        "right_wrist_0_rgb": jnp.ones((batch_size, 224, 224, 3), dtype=jnp.float32) * -0.2,
    }
    
    # 创建图像掩码（全部有效）
    fake_image_masks = {
        "base_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
        "left_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
        "right_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
    }
    
    # 创建假的状态数据
    fake_state = jnp.ones((batch_size, config.action_dim), dtype=jnp.float32) * 0.1
    
    # 创建假的提示词（可选）
    fake_tokenized_prompt = jnp.ones((batch_size, config.max_token_len), dtype=jnp.int32) * 100
    fake_tokenized_prompt_mask = jnp.ones((batch_size, config.max_token_len), dtype=jnp.bool_)
    
    observation = Observation(
        images=fake_images,
        image_masks=fake_image_masks,
        state=fake_state,
        tokenized_prompt=fake_tokenized_prompt,
        tokenized_prompt_mask=fake_tokenized_prompt_mask,
    )
    
    return observation


def test_sample_actions_basic():
    """基础测试：验证 sample_actions 能正常运行并返回正确形状的输出"""
    print("=" * 80)
    print("测试 1: 基础功能测试")
    print("=" * 80)
    
    # 创建配置
    config = Pi0Config(
        action_dim=14,
        action_horizon=10,
        pi05=False,  # 测试非 pi05 模式
    )
    
    print(f"配置: action_dim={config.action_dim}, action_horizon={config.action_horizon}, pi05={config.pi05}")
    
    # 创建模型
    rng = jax.random.key(42)
    print("正在创建模型...")
    model = config.create(rng)
    print(f"模型创建成功: {type(model).__name__}")
    
    # 创建假观察数据
    batch_size = 2
    observation = create_fake_observation(config, batch_size=batch_size)
    print(f"\n观察数据形状:")
    print(f"  - 图像数量: {len(observation.images)}")
    for key, img in observation.images.items():
        print(f"    {key}: {img.shape}")
    print(f"  - 状态: {observation.state.shape}")
    print(f"  - 提示词: {observation.tokenized_prompt.shape if observation.tokenized_prompt is not None else 'None'}")
    
    # 调用 sample_actions
    print("\n正在采样动作...")
    rng_sample = jax.random.key(123)
    num_steps = 5
    actions = model.sample_actions(rng_sample, observation, num_steps=num_steps)
    
    print(f"\n✓ 采样成功!")
    print(f"  - 输出形状: {actions.shape}")
    print(f"  - 期望形状: ({batch_size}, {config.action_horizon + 2}, {config.action_dim})")
    print(f"  - 数据类型: {actions.dtype}")
    print(f"  - 值范围: [{jnp.min(actions):.4f}, {jnp.max(actions):.4f}]")
    print(f"  - 均值: {jnp.mean(actions):.4f}")
    print(f"  - 标准差: {jnp.std(actions):.4f}")
    
    # 验证形状
    expected_shape = (batch_size, config.action_horizon + 2, config.action_dim)
    assert actions.shape == expected_shape, f"形状不匹配: {actions.shape} vs {expected_shape}"
    
    print("\n✓ 测试通过!")
    return actions


def test_sample_actions_with_custom_noise():
    """测试：使用自定义噪声"""
    print("\n" + "=" * 80)
    print("测试 2: 自定义噪声测试")
    print("=" * 80)
    
    config = Pi0Config(action_dim=14, action_horizon=10, pi05=False)
    rng = jax.random.key(42)
    model = config.create(rng)
    
    batch_size = 1
    observation = create_fake_observation(config, batch_size=batch_size)
    
    # 创建自定义噪声
    custom_noise = jnp.ones((batch_size, config.action_horizon + 2, config.action_dim)) * 2.0
    print(f"自定义噪声形状: {custom_noise.shape}")
    print(f"自定义噪声值: 全部为 2.0")
    
    # 使用自定义噪声采样
    rng_sample = jax.random.key(456)
    actions = model.sample_actions(rng_sample, observation, num_steps=10, noise=custom_noise)
    
    print(f"\n✓ 采样成功!")
    print(f"  - 输出形状: {actions.shape}")
    print(f"  - 值范围: [{jnp.min(actions):.4f}, {jnp.max(actions):.4f}]")
    
    print("\n✓ 测试通过!")
    return actions


def test_sample_actions_pi05_mode():
    """测试：Pi05 模式"""
    print("\n" + "=" * 80)
    print("测试 3: Pi05 模式测试")
    print("=" * 80)
    
    config = Pi0Config(
        action_dim=14,
        action_horizon=10,
        pi05=True,  # 使用 pi05 模式
    )
    
    print(f"配置: action_dim={config.action_dim}, action_horizon={config.action_horizon}, pi05={config.pi05}")
    
    rng = jax.random.key(42)
    print("正在创建 Pi05 模型...")
    model = config.create(rng)
    print(f"模型创建成功: {type(model).__name__}")
    
    batch_size = 2
    observation = create_fake_observation(config, batch_size=batch_size)
    
    # 采样动作
    rng_sample = jax.random.key(789)
    actions = model.sample_actions(rng_sample, observation, num_steps=8)
    
    print(f"\n✓ 采样成功!")
    print(f"  - 输出形状: {actions.shape}")
    print(f"  - 期望形状: ({batch_size}, {config.action_horizon + 2}, {config.action_dim})")
    print(f"  - 值范围: [{jnp.min(actions):.4f}, {jnp.max(actions):.4f}]")
    
    expected_shape = (batch_size, config.action_horizon + 2, config.action_dim)
    assert actions.shape == expected_shape, f"形状不匹配: {actions.shape} vs {expected_shape}"
    
    print("\n✓ 测试通过!")
    return actions


def test_sample_actions_different_num_steps():
    """测试：不同的采样步数"""
    print("\n" + "=" * 80)
    print("测试 4: 不同采样步数测试")
    print("=" * 80)
    
    config = Pi0Config(action_dim=14, action_horizon=10, pi05=False)
    rng = jax.random.key(42)
    model = config.create(rng)
    
    batch_size = 1
    observation = create_fake_observation(config, batch_size=batch_size)
    
    step_counts = [1, 5, 10, 20]
    results = {}
    
    for num_steps in step_counts:
        print(f"\n测试 num_steps={num_steps}...")
        rng_sample = jax.random.key(100 + num_steps)
        actions = model.sample_actions(rng_sample, observation, num_steps=num_steps)
        results[num_steps] = actions
        print(f"  - 形状: {actions.shape}")
        print(f"  - 值范围: [{jnp.min(actions):.4f}, {jnp.max(actions):.4f}]")
        print(f"  - 均值: {jnp.mean(actions):.4f}")
    
    print("\n✓ 所有步数测试通过!")
    
    # 比较不同步数的结果差异
    print("\n结果差异分析:")
    for i, steps1 in enumerate(step_counts[:-1]):
        steps2 = step_counts[i + 1]
        diff = jnp.mean(jnp.abs(results[steps1] - results[steps2]))
        print(f"  num_steps={steps1} vs {steps2}: 平均绝对差异 = {diff:.4f}")
    
    return results


def test_determinism():
    """测试：相同种子是否产生相同结果"""
    print("\n" + "=" * 80)
    print("测试 5: 确定性测试")
    print("=" * 80)
    
    config = Pi0Config(action_dim=14, action_horizon=10, pi05=False)
    rng = jax.random.key(42)
    model = config.create(rng)
    
    batch_size = 1
    observation = create_fake_observation(config, batch_size=batch_size)
    
    # 使用相同的种子采样两次
    seed = 999
    actions1 = model.sample_actions(jax.random.key(seed), observation, num_steps=5)
    actions2 = model.sample_actions(jax.random.key(seed), observation, num_steps=5)
    
    # 检查是否完全相同
    is_same = jnp.allclose(actions1, actions2, rtol=1e-5, atol=1e-5)
    max_diff = jnp.max(jnp.abs(actions1 - actions2))
    
    print(f"相同种子两次采样:")
    print(f"  - 结果是否相同: {is_same}")
    print(f"  - 最大差异: {max_diff:.10f}")
    
    if is_same:
        print("\n✓ 确定性测试通过! (相同种子产生相同结果)")
    else:
        print("\n⚠ 警告: 相同种子产生了不同结果")
    
    # 使用不同的种子
    actions3 = model.sample_actions(jax.random.key(seed + 1), observation, num_steps=5)
    is_different = not jnp.allclose(actions1, actions3, rtol=1e-3, atol=1e-3)
    
    print(f"\n不同种子两次采样:")
    print(f"  - 结果是否不同: {is_different}")
    
    if is_different:
        print("✓ 不同种子产生了不同结果 (符合预期)")
    
    return actions1, actions2, actions3


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "Pi0 sample_actions 测试脚本" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")
    
    try:
        # 测试 1: 基础功能
        test_sample_actions_basic()
        
        # 测试 2: 自定义噪声
        test_sample_actions_with_custom_noise()
        
        # 测试 3: Pi05 模式
        test_sample_actions_pi05_mode()
        
        # 测试 4: 不同采样步数
        test_sample_actions_different_num_steps()
        
        # 测试 5: 确定性
        test_determinism()
        
        # 总结
        print("\n" + "=" * 80)
        print("🎉 所有测试通过!")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ 测试失败: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

