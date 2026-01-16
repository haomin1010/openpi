#!/usr/bin/env python3
"""
安全区域解析工具
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger("SafetyBox")


@dataclass(frozen=True)
class SafetyBox:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains(self, point_xyz: Sequence[float]) -> bool:
        x, y, z = point_xyz
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )


def load_safety_box(boundingbox_path: str | Path) -> SafetyBox:
    """
    从 boundingbox.txt 文件加载安全区域。
    
    不限定文件中条目的数量，遍历所有条目，自动记录 x、y、z 这三个坐标的最大值与最小值。
    
    Args:
        boundingbox_path: boundingbox.txt 文件路径
        
    Returns:
        SafetyBox: 安全区域对象
        
    Raises:
        FileNotFoundError: 如果文件不存在
        ValueError: 如果文件中没有有效的点
        
    文件格式:
        每行一个点，格式为 [x, y, z]，必须且仅包含3个坐标值（x, y, z）。
        例如:
            [0.523453, 0.122365, 0.221617]
            [0.4, 0.1, 0.2]
            [0.6, 0.15, 0.25]
        
    处理逻辑:
        1. 遍历文件中的所有行
        2. 解析每一行，验证必须包含且仅包含3个值（x, y, z）
        3. 收集所有有效的点
        4. 计算所有点的 x, y, z 坐标的最小值和最大值
        5. 生成安全区域长方体
    """
    path = Path(boundingbox_path)
    if not path.exists():
        raise FileNotFoundError(f"boundingbox 文件不存在: {path}")

    # 遍历所有行，收集所有有效的点坐标
    points = []
    for line_num, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        # 跳过空行
        if not line:
            continue
        
        try:
            # 解析行内容（支持列表或元组格式）
            values = ast.literal_eval(line)
        except Exception as e:
            logger.warning("无法解析 boundingbox 第 %d 行: %s (错误: %s)", line_num, line, e)
            continue
        
        # 验证格式：必须是列表或元组
        if not isinstance(values, (list, tuple)):
            logger.warning("boundingbox 第 %d 行格式无效（不是列表/元组）: %s", line_num, line)
            continue
        
        # 严格要求必须且仅包含3个值（x, y, z）
        if len(values) != 3:
            logger.warning(
                "boundingbox 第 %d 行参数数量错误（必须且仅包含3个参数 x, y, z）: %s (实际: %d 个参数)",
                line_num, line, len(values)
            )
            continue
        
        # 提取 x, y, z 坐标
        x, y, z = values[0], values[1], values[2]
        points.append([x, y, z])

    # 验证是否收集到有效的点
    if not points:
        raise ValueError(f"boundingbox 文件无有效点: {path}")

    # 转换为 numpy 数组以便计算
    points = np.asarray(points, dtype=np.float64)
    
    # 计算所有点的 x, y, z 坐标的最小值和最大值
    # axis=0 表示沿着列（坐标维度）计算，得到每个坐标的最小/最大值
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)
    
    logger.info(
        "从 %d 个点生成安全区域: x=[%.6f, %.6f], y=[%.6f, %.6f], z=[%.6f, %.6f]",
        len(points), x_min, x_max, y_min, y_max, z_min, z_max
    )
    
    return SafetyBox(x_min, x_max, y_min, y_max, z_min, z_max)
