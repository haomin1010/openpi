"""测试时间配对采样功能的脚本。

使用方法：
    python test_temporal_pair_sampling.py

这个脚本将：
1. 创建一个假数据集
2. 使用时间配对采样加载数据
3. 验证配对是否正确
"""

import logging
from openpi.training import data_loader, config

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_temporal_pair_sampling():
    """测试时间配对采样功能。"""
    logger.info("开始测试时间配对采样...")
    
    # 使用debug配置（使用假数据）
    train_config = config.get_config("debug")
    
    # 修改配置以启用时间配对采样
    train_config = config.TrainConfig(
        name="debug_temporal",
        data=config.FakeDataConfig(),
        batch_size=4,  # 必须是偶数
        model=train_config.model,
        save_interval=100,
        overwrite=True,
        exp_name="debug_temporal",
        num_train_steps=10,
        wandb_enabled=False,
        use_temporal_pairs=True,  # 启用时间配对采样
        temporal_pair_offset=5,    # 时间偏移量为5
    )
    
    logger.info(f"配置：batch_size={train_config.batch_size}, "
                f"use_temporal_pairs={train_config.use_temporal_pairs}, "
                f"temporal_pair_offset={train_config.temporal_pair_offset}")
    
    # 创建数据加载器
    try:
        loader = data_loader.create_data_loader(
            train_config,
            shuffle=True,
            skip_norm_stats=True,
            use_temporal_pairs=train_config.use_temporal_pairs,
            temporal_pair_offset=train_config.temporal_pair_offset,
        )
        logger.info("数据加载器创建成功！")
    except Exception as e:
        logger.error(f"创建数据加载器失败：{e}")
        return False
    
    # 获取一个batch
    try:
        batch_iter = iter(loader)
        observation, actions = next(batch_iter)
        logger.info(f"成功获取一个batch")
        logger.info(f"Observation类型：{type(observation)}")
        logger.info(f"Actions shape：{actions.shape}")
        
        # 验证batch size
        batch_size = actions.shape[0]
        if batch_size != train_config.batch_size:
            logger.error(f"Batch size不匹配！期望：{train_config.batch_size}，实际：{batch_size}")
            return False
        
        logger.info(f"✓ Batch size正确：{batch_size}")
        
        # 验证batch size是偶数
        if batch_size % 2 != 0:
            logger.error(f"Batch size不是偶数：{batch_size}")
            return False
        
        logger.info(f"✓ Batch size是偶数")
        
        logger.info("\n测试成功！时间配对采样功能正常工作。")
        return True
        
    except Exception as e:
        logger.error(f"获取batch失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_without_temporal_pairs():
    """测试不使用时间配对采样的情况（对照组）。"""
    logger.info("\n开始测试标准采样（不使用时间配对）...")
    
    train_config = config.get_config("debug")
    
    # 创建数据加载器（不使用时间配对采样）
    try:
        loader = data_loader.create_data_loader(
            train_config,
            shuffle=True,
            skip_norm_stats=True,
            use_temporal_pairs=False,  # 不使用时间配对采样
        )
        logger.info("数据加载器创建成功！")
    except Exception as e:
        logger.error(f"创建数据加载器失败：{e}")
        return False
    
    # 获取一个batch
    try:
        batch_iter = iter(loader)
        observation, actions = next(batch_iter)
        logger.info(f"成功获取一个batch")
        logger.info(f"Actions shape：{actions.shape}")
        
        logger.info("\n标准采样测试成功！")
        return True
        
    except Exception as e:
        logger.error(f"获取batch失败：{e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("时间配对采样功能测试")
    logger.info("=" * 80)
    
    # 测试标准采样
    success1 = test_without_temporal_pairs()
    
    # 测试时间配对采样
    success2 = test_temporal_pair_sampling()
    
    logger.info("\n" + "=" * 80)
    if success1 and success2:
        logger.info("✓ 所有测试通过！")
    else:
        logger.error("✗ 某些测试失败")
    logger.info("=" * 80)

