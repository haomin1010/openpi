#!/usr/bin/env python3
"""下载 DROID 归一化统计信息文件"""

import pathlib
from openpi.shared import download
from openpi.shared import normalize

# DROID 归一化统计信息的两个来源
# 选项 1: 从 pi05_base 加载（用于完整 DROID 数据集训练）
assets_dir_1 = "gs://openpi-assets/checkpoints/pi05_base/assets"
asset_id_1 = "droid"

# 选项 2: 从 pi05_droid 加载（用于自定义 DROID 数据集微调）
assets_dir_2 = "gs://openpi-assets/checkpoints/pi05_droid/assets"
asset_id_2 = "droid"

# 选择要下载的版本
use_pi05_droid = True  # 改为 False 使用 pi05_base 版本

if use_pi05_droid:
    assets_dir = assets_dir_2
    asset_id = asset_id_2
    output_name = "droid_norm_stats_from_pi05_droid.json"
else:
    assets_dir = assets_dir_1
    asset_id = asset_id_1
    output_name = "droid_norm_stats_from_pi05_base.json"

# 下载并加载归一化统计信息
print(f"正在从 {assets_dir}/{asset_id} 下载归一化统计信息...")
norm_stats_dir = download.maybe_download(f"{assets_dir}/{asset_id}")
norm_stats = normalize.load(norm_stats_dir)

# 保存到当前目录
output_path = pathlib.Path(output_name)
normalize.save(output_path.parent, norm_stats)
print(f"归一化统计信息已保存到: {output_path}")
print(f"文件路径: {norm_stats_dir / 'norm_stats.json'}")