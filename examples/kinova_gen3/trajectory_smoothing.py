"""
轨迹平滑处理模块

提供轨迹平滑功能,可用于轨迹回放和VLA策略输出动作的平滑处理。

主要功能:
- 滑动窗口平滑滤波
- 轨迹插值
- 末端位姿轨迹平滑
"""

import numpy as np
from typing import Optional
from collections import deque
from scipy.spatial.transform import Rotation


class TrajectorySmoother:
    """
    轨迹平滑器
    
    使用滑动窗口对末端位姿轨迹进行平滑处理。
    可用于轨迹回放和实时动作平滑。
    
    属性:
        window_size (int): 滑动窗口大小
        position_weights (Optional[np.ndarray]): 位置平滑权重（用于加权平均）
        quaternion_interpolation (bool): 是否使用四元数插值（默认True）
    """
    
    def __init__(
        self,
        window_size: int = 5,
        position_weights: Optional[np.ndarray] = None,
        quaternion_interpolation: bool = True,
    ):
        """
        初始化轨迹平滑器
        
        Args:
            window_size: 滑动窗口大小（用于平滑的点数）
            position_weights: 位置平滑权重数组，长度应为window_size。
                             如果为None，使用均匀权重
            quaternion_interpolation: 是否对四元数使用球面线性插值（SLERP）
        """
        self.window_size = window_size
        self.quaternion_interpolation = quaternion_interpolation
        
        # 初始化权重（均匀权重或指定权重）
        if position_weights is None:
            self.position_weights = np.ones(window_size) / window_size
        else:
            if len(position_weights) != window_size:
                raise ValueError(f"权重数组长度 ({len(position_weights)}) 必须等于窗口大小 ({window_size})")
            self.position_weights = np.array(position_weights) / np.sum(position_weights)
        
        # 滑动窗口（存储末端位姿历史）
        self._position_window = deque(maxlen=window_size)  # 位置窗口
        self._quaternion_window = deque(maxlen=window_size)  # 四元数窗口
    
    def add_pose(self, eef_pose: np.ndarray):
        """
        添加一个新的末端位姿到滑动窗口
        
        Args:
            eef_pose: 末端执行器位姿数组，格式为 [x, y, z, qx, qy, qz, qw]
        """
        if len(eef_pose) != 7:
            raise ValueError(f"末端位姿数组长度应为7，但收到 {len(eef_pose)}")
        
        position = eef_pose[:3]  # 位置 (x, y, z)
        quaternion = eef_pose[3:]  # 四元数 (qx, qy, qz, qw)
        
        # 归一化四元数（确保单位四元数）
        quaternion = quaternion / np.linalg.norm(quaternion)
        
        self._position_window.append(position.copy())
        self._quaternion_window.append(quaternion.copy())
    
    def get_smoothed_pose(self) -> Optional[np.ndarray]:
        """
        获取平滑后的末端位姿
        
        Returns:
            平滑后的末端位姿数组 [x, y, z, qx, qy, qz, qw]，如果窗口未满则返回None
        """
        if len(self._position_window) < self.window_size:
            return None
        
        # 平滑位置（加权平均）
        positions = np.array(list(self._position_window))
        smoothed_position = np.average(positions, axis=0, weights=self.position_weights)
        
        # 平滑四元数
        quaternions = np.array(list(self._quaternion_window))
        if self.quaternion_interpolation:
            # 使用球面线性插值（SLERP）对四元数进行平滑
            smoothed_quaternion = self._slerp_quaternions(quaternions, self.position_weights)
        else:
            # 简单平均（不推荐，但作为备选）
            smoothed_quaternion = np.average(quaternions, axis=0, weights=self.position_weights)
            smoothed_quaternion = smoothed_quaternion / np.linalg.norm(smoothed_quaternion)
        
        return np.concatenate([smoothed_position, smoothed_quaternion])
    
    def _slerp_quaternions(self, quaternions: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        使用加权球面线性插值（SLERP）平滑四元数
        
        Args:
            quaternions: 四元数数组 (N, 4)
            weights: 权重数组 (N,)
        
        Returns:
            平滑后的四元数 (4,)
        """
        # 简化实现：使用加权平均后归一化（对于小角度变化足够好）
        # 更精确的实现可以使用多个SLERP步骤
        weighted_avg = np.average(quaternions, axis=0, weights=weights)
        return weighted_avg / np.linalg.norm(weighted_avg)
    
    def reset(self):
        """重置滑动窗口"""
        self._position_window.clear()
        self._quaternion_window.clear()
    
    def is_ready(self) -> bool:
        """检查窗口是否已满（可以输出平滑后的位姿）"""
        return len(self._position_window) >= self.window_size


def interpolate_trajectory(
    eef_poses: np.ndarray,
    target_frequency: float,
    source_frequency: float,
) -> np.ndarray:
    """
    对末端位姿轨迹进行插值，将轨迹从源频率插值到目标频率
    
    Args:
        eef_poses: 原始末端位姿数组 (N, 7)，格式为 [x, y, z, qx, qy, qz, qw]
        target_frequency: 目标频率 (Hz)
        source_frequency: 源频率 (Hz)
    
    Returns:
        插值后的末端位姿数组 (M, 7)
    
    注意:
        - 如果 target_frequency <= source_frequency，返回原始轨迹（不进行上采样）
        - 使用线性插值对位置进行插值
        - 使用球面线性插值（SLERP）对四元数进行插值
    """
    if target_frequency <= source_frequency:
        return eef_poses.copy()
    
    num_source = len(eef_poses)
    num_target = int(num_source * target_frequency / source_frequency)
    
    if num_target == num_source:
        return eef_poses.copy()
    
    # 创建源时间点和目标时间点
    source_times = np.linspace(0, 1, num_source)
    target_times = np.linspace(0, 1, num_target)
    
    # 分离位置和四元数
    positions = eef_poses[:, :3]  # (N, 3)
    quaternions = eef_poses[:, 3:]  # (N, 4)
    
    # 归一化四元数
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    
    # 插值位置（线性插值）
    interpolated_positions = np.interp(target_times, source_times, positions.T).T
    
    # 插值四元数（球面线性插值）
    interpolated_quaternions = np.zeros((num_target, 4))
    for i, t in enumerate(target_times):
        # 找到插值区间
        idx = np.searchsorted(source_times, t, side='right') - 1
        idx = max(0, min(idx, num_source - 2))
        
        t0, t1 = source_times[idx], source_times[idx + 1]
        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        alpha = np.clip(alpha, 0.0, 1.0)
        
        q0 = quaternions[idx]
        q1 = quaternions[idx + 1]
        
        # 球面线性插值
        interpolated_quaternions[i] = _slerp_single(q0, q1, alpha)
    
    # 组合结果
    interpolated_poses = np.concatenate([interpolated_positions, interpolated_quaternions], axis=1)
    
    return interpolated_poses


def _slerp_single(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    两个四元数之间的球面线性插值（SLERP）
    
    Args:
        q0: 起始四元数 (4,)
        q1: 结束四元数 (4,)
        t: 插值参数 [0, 1]
    
    Returns:
        插值后的四元数 (4,)
    """
    # 计算点积（cosine of the angle between quaternions）
    dot = np.dot(q0, q1)
    
    # 如果点积为负，取反q1以确保走最短路径
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    # 如果角度很小，使用线性插值
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    
    # 计算角度和SLERP
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    
    w0 = np.sin((1.0 - t) * theta) / sin_theta
    w1 = np.sin(t * theta) / sin_theta
    
    result = w0 * q0 + w1 * q1
    return result / np.linalg.norm(result)
