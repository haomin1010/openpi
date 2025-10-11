"""
时间配对采样的实际使用示例

这个脚本展示了如何在实际训练中使用时间配对采样功能。
"""

import jax
import jax.numpy as jnp
from openpi.training import config, data_loader
from openpi.models import pi0_config
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_usage():
    """示例1：基本使用 - 使用假数据测试"""
    logger.info("\n" + "="*80)
    logger.info("示例1：基本使用")
    logger.info("="*80)
    
    # 创建配置
    train_config = config.TrainConfig(
        name="example_temporal",
        exp_name="test_temporal_pairs",
        model=pi0_config.Pi0Config(),
        data=config.FakeDataConfig(),
        batch_size=8,  # 必须是偶数
        use_temporal_pairs=True,    # 启用时间配对采样
        temporal_pair_offset=5,      # 相差5个时间步
        wandb_enabled=False,
    )
    
    logger.info(f"配置：")
    logger.info(f"  - batch_size: {train_config.batch_size}")
    logger.info(f"  - use_temporal_pairs: {train_config.use_temporal_pairs}")
    logger.info(f"  - temporal_pair_offset: {train_config.temporal_pair_offset}")
    
    # 创建数据加载器
    loader = data_loader.create_data_loader(
        train_config,
        shuffle=True,
        skip_norm_stats=True,
    )
    
    # 获取一个batch
    observation, actions = next(iter(loader))
    
    logger.info(f"\nBatch信息：")
    logger.info(f"  - Actions shape: {actions.shape}")
    logger.info(f"  - 样本配对：")
    logger.info(f"    * 样本0和样本1来自同一轨迹，相差5步")
    logger.info(f"    * 样本2和样本3来自同一轨迹，相差5步")
    logger.info(f"    * ...")
    logger.info(f"    * 样本{train_config.batch_size-2}和样本{train_config.batch_size-1}来自同一轨迹，相差5步")
    

def example_2_with_loss_function():
    """示例2：在训练中使用 - 自定义损失函数利用时间配对"""
    logger.info("\n" + "="*80)
    logger.info("示例2：自定义损失函数使用时间配对")
    logger.info("="*80)
    
    # 创建配置
    train_config = config.TrainConfig(
        name="example_temporal",
        exp_name="test_temporal_pairs",
        model=pi0_config.Pi0Config(),
        data=config.FakeDataConfig(),
        batch_size=8,
        use_temporal_pairs=True,
        temporal_pair_offset=5,
        wandb_enabled=False,
    )
    
    # 创建数据加载器
    loader = data_loader.create_data_loader(
        train_config,
        shuffle=True,
        skip_norm_stats=True,
    )
    
    # 获取一个batch
    observation, actions = next(iter(loader))
    
    # 示例：提取配对的样本
    batch_size = actions.shape[0]
    
    # 将batch分成两半：第一半是t时刻，第二半需要重新组织
    # 实际上配对是交错的：[t, t+5, t, t+5, ...]
    # 我们需要将它们分离
    
    # 提取所有偶数索引（t时刻）和奇数索引（t+5时刻）
    t_indices = jnp.arange(0, batch_size, 2)
    t_plus_5_indices = jnp.arange(1, batch_size, 2)
    
    actions_t = actions[t_indices]  # shape: (batch_size//2, ...)
    actions_t_plus_5 = actions[t_plus_5_indices]  # shape: (batch_size//2, ...)
    
    logger.info(f"\n提取配对样本：")
    logger.info(f"  - actions_t shape: {actions_t.shape} (时刻t的动作)")
    logger.info(f"  - actions_t_plus_5 shape: {actions_t_plus_5.shape} (时刻t+5的动作)")
    
    # 示例：计算时序一致性损失
    # 这只是一个示例，实际使用时需要根据具体任务设计
    temporal_consistency_loss = jnp.mean(jnp.abs(actions_t - actions_t_plus_5))
    logger.info(f"  - 示例时序一致性损失: {temporal_consistency_loss:.4f}")
    
    logger.info(f"\n说明：")
    logger.info(f"  这个损失可以鼓励模型在同一轨迹的不同时刻")
    logger.info(f"  产生连贯的动作序列。")


def example_3_different_offsets():
    """示例3：尝试不同的时间偏移量"""
    logger.info("\n" + "="*80)
    logger.info("示例3：比较不同的时间偏移量")
    logger.info("="*80)
    
    offsets = [3, 5, 10]
    
    for offset in offsets:
        logger.info(f"\n测试时间偏移量: {offset}")
        
        train_config = config.TrainConfig(
            name="example_temporal",
            exp_name=f"test_offset_{offset}",
            model=pi0_config.Pi0Config(),
            data=config.FakeDataConfig(),
            batch_size=6,
            use_temporal_pairs=True,
            temporal_pair_offset=offset,
            wandb_enabled=False,
        )
        
        try:
            loader = data_loader.create_data_loader(
                train_config,
                shuffle=True,
                skip_norm_stats=True,
            )
            
            observation, actions = next(iter(loader))
            logger.info(f"  ✓ 成功创建数据加载器，batch shape: {actions.shape}")
            logger.info(f"    配对：[t, t+{offset}], [t, t+{offset}], [t, t+{offset}]")
            
        except Exception as e:
            logger.error(f"  ✗ 失败: {e}")


def example_4_practical_training_loop():
    """示例4：完整的训练循环示例"""
    logger.info("\n" + "="*80)
    logger.info("示例4：实际训练循环中使用时间配对")
    logger.info("="*80)
    
    # 创建配置
    train_config = config.TrainConfig(
        name="example_temporal",
        exp_name="training_with_temporal_pairs",
        model=pi0_config.Pi0Config(),
        data=config.FakeDataConfig(),
        batch_size=8,
        use_temporal_pairs=True,
        temporal_pair_offset=5,
        wandb_enabled=False,
        num_train_steps=100,
    )
    
    # 创建数据加载器
    loader = data_loader.create_data_loader(
        train_config,
        shuffle=True,
        skip_norm_stats=True,
    )
    
    logger.info(f"开始训练（前5步）...")
    
    # 模拟训练循环
    for step, (observation, actions) in enumerate(loader):
        if step >= 5:  # 只演示前5步
            break
        
        # 提取配对样本
        batch_size = actions.shape[0]
        t_indices = jnp.arange(0, batch_size, 2)
        t_plus_5_indices = jnp.arange(1, batch_size, 2)
        
        actions_t = actions[t_indices]
        actions_t_plus_5 = actions[t_plus_5_indices]
        
        # 计算损失（这里只是示例）
        standard_loss = jnp.mean(actions_t ** 2)
        temporal_loss = jnp.mean((actions_t - actions_t_plus_5) ** 2)
        total_loss = standard_loss + 0.1 * temporal_loss
        
        logger.info(f"  Step {step}: "
                   f"standard_loss={standard_loss:.4f}, "
                   f"temporal_loss={temporal_loss:.4f}, "
                   f"total_loss={total_loss:.4f}")
    
    logger.info(f"\n说明：")
    logger.info(f"  在实际训练中，您可以：")
    logger.info(f"  1. 使用标准的策略损失")
    logger.info(f"  2. 添加额外的时序约束损失")
    logger.info(f"  3. 利用配对样本进行对比学习")
    logger.info(f"  4. 实现各种时序建模目标")


def main():
    """运行所有示例"""
    logger.info("\n" + "="*80)
    logger.info("时间配对采样 - 使用示例")
    logger.info("="*80)
    
    try:
        example_1_basic_usage()
        example_2_with_loss_function()
        example_3_different_offsets()
        example_4_practical_training_loop()
        
        logger.info("\n" + "="*80)
        logger.info("✓ 所有示例运行成功！")
        logger.info("="*80)
        
        logger.info("\n下一步：")
        logger.info("1. 在您的实际数据集上测试")
        logger.info("2. 根据任务调整 temporal_pair_offset")
        logger.info("3. 设计利用时间配对的损失函数")
        logger.info("4. 开始训练！")
        
    except Exception as e:
        logger.error(f"\n✗ 出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

