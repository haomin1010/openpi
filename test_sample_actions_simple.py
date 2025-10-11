"""简化版测试脚本 - 快速验证 sample_actions"""

import jax
import jax.numpy as jnp
from openpi.models.pi0_config import Pi0Config


def quick_test():
    """快速测试 sample_actions 功能"""
    print("=" * 60)
    print("快速测试 sample_actions")
    print("=" * 60)
    
    # 1. 创建配置和模型
    config = Pi0Config(
        action_dim=14,
        action_horizon=10,
        pi05=False,
    )
    print(f"✓ 配置创建完成")
    print(f"  action_dim={config.action_dim}, action_horizon={config.action_horizon}")
    
    # 2. 初始化模型
    print("\n正在初始化模型...")
    model = config.create(jax.random.key(0))
    print(f"✓ 模型初始化完成: {type(model).__name__}")
    
    # 3. 创建假数据
    print("\n正在创建观察数据...")
    batch_size = 2
    observation = config.fake_obs(batch_size=batch_size)
    print(f"✓ 观察数据创建完成")
    print(f"  batch_size={batch_size}")
    print(f"  state shape: {observation.state.shape}")
    for key in observation.images:
        print(f"  {key}: {observation.images[key].shape}")
    
    # 4. 采样动作
    print("\n正在采样动作...")
    print(f"  num_steps=10")
    actions = model.sample_actions(
        jax.random.key(42),
        observation,
        num_steps=10
    )
    
    # 5. 验证结果
    print(f"\n✓ 采样完成!")
    print(f"  输出形状: {actions.shape}")
    expected_shape = (batch_size, config.action_horizon + 2, config.action_dim)
    print(f"  期望形状: {expected_shape}")
    print(f"  形状匹配: {actions.shape == expected_shape}")
    print(f"  值范围: [{jnp.min(actions):.4f}, {jnp.max(actions):.4f}]")
    print(f"  均值: {jnp.mean(actions):.4f} ± {jnp.std(actions):.4f}")
    
    assert actions.shape == expected_shape, "形状不匹配!"
    assert jnp.isfinite(actions).all(), "包含 NaN 或 Inf!"
    
    print("\n" + "=" * 60)
    print("✅ 所有检查通过!")
    print("=" * 60)
    
    return model, observation, actions


if __name__ == "__main__":
    model, observation, actions = quick_test()

