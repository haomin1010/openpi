"""
笛卡尔空间速度控制模块

提供基于末端位姿的速度控制功能,可用于轨迹回放和VLA策略输出动作的执行。

主要功能:
- 基于位置误差计算速度命令
- TwistCommand 生成和发送
- 比例控制（P控制器）
"""

import numpy as np
from typing import Optional
import logging

# Kinova kortex API
try:
    from kortex_api.autogen.messages import Base_pb2
except ImportError:
    Base_pb2 = None
    logging.warning("kortex_api 未安装，速度控制功能将不可用")

logger = logging.getLogger(__name__)


class CartesianVelocityController:
    """
    笛卡尔空间速度控制器
    
    基于末端位姿误差计算速度命令,可用于平滑的轨迹跟踪。
    可用于轨迹回放和VLA策略输出动作的执行。
    
    属性:
        max_linear_velocity (float): 最大线速度 (m/s)
        max_angular_velocity (float): 最大角速度 (rad/s)
        position_gain (float): 位置控制增益
        orientation_gain (float): 姿态控制增益
        min_distance_threshold (float): 最小距离阈值（小于此值不发送速度命令）
    """
    
    def __init__(
        self,
        max_linear_velocity: float = 0.05,  # 5 cm/s
        max_angular_velocity: float = 0.5,  # rad/s
        position_gain: float = 2.0,
        orientation_gain: float = 1.0,
        min_distance_threshold: float = 0.001,  # 1 mm
    ):
        """
        初始化速度控制器
        
        Args:
            max_linear_velocity: 最大线速度 (m/s)
            max_angular_velocity: 最大角速度 (rad/s)
            position_gain: 位置控制增益（比例控制器增益）
            orientation_gain: 姿态控制增益
            min_distance_threshold: 最小距离阈值 (m)，小于此值不发送速度命令
        """
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.position_gain = position_gain
        self.orientation_gain = orientation_gain
        self.min_distance_threshold = min_distance_threshold
    
    def compute_velocity(
        self,
        current_pose: np.ndarray,
        target_pose: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        计算从当前位置到目标位置的速度命令
        
        Args:
            current_pose: 当前末端位姿 [x, y, z, qx, qy, qz, qw]
            target_pose: 目标末端位姿 [x, y, z, qx, qy, qz, qw]
        
        Returns:
            (linear_velocity, angular_velocity) 元组:
                - linear_velocity: 线速度 (3,) [vx, vy, vz] (m/s)
                - angular_velocity: 角速度 (3,) [wx, wy, wz] (rad/s)
        """
        if len(current_pose) != 7 or len(target_pose) != 7:
            raise ValueError("位姿数组长度必须为7 [x, y, z, qx, qy, qz, qw]")
        
        # 位置误差
        position_error = target_pose[:3] - current_pose[:3]
        position_distance = np.linalg.norm(position_error)
        
        # 计算线速度（比例控制）
        if position_distance < self.min_distance_threshold:
            linear_velocity = np.zeros(3)
        else:
            # 比例控制器：速度 = gain * 误差，但限制在最大速度内
            desired_velocity_magnitude = self.position_gain * position_distance
            velocity_magnitude = min(desired_velocity_magnitude, self.max_linear_velocity)
            velocity_direction = position_error / position_distance
            linear_velocity = velocity_direction * velocity_magnitude
        
        # 姿态误差（使用四元数）
        current_quat = current_pose[3:] / np.linalg.norm(current_pose[3:])
        target_quat = target_pose[3:] / np.linalg.norm(target_pose[3:])
        
        # 计算角速度（简化的四元数误差转角速度）
        angular_velocity = self._quaternion_error_to_angular_velocity(
            current_quat, target_quat
        )
        
        # 限制角速度大小
        angular_velocity_norm = np.linalg.norm(angular_velocity)
        if angular_velocity_norm > self.max_angular_velocity:
            angular_velocity = angular_velocity * (self.max_angular_velocity / angular_velocity_norm)
        
        return linear_velocity, angular_velocity
    
    def _quaternion_error_to_angular_velocity(
        self,
        current_quat: np.ndarray,
        target_quat: np.ndarray,
    ) -> np.ndarray:
        """
        将四元数误差转换为角速度
        
        Args:
            current_quat: 当前四元数 [qx, qy, qz, qw]
            target_quat: 目标四元数 [qx, qy, qz, qw]
        
        Returns:
            角速度 [wx, wy, wz] (rad/s)
        """
        # 计算相对旋转四元数: q_error = q_target * q_current^-1
        # 对于单位四元数，逆就是共轭 [qx, qy, qz, qw] -> [-qx, -qy, -qz, qw]
        current_quat_conj = np.array([
            -current_quat[0],
            -current_quat[1],
            -current_quat[2],
            current_quat[3]
        ])
        
        # 四元数乘法: q_error = q_target * q_current_conj
        q_error = self._quaternion_multiply(target_quat, current_quat_conj)
        
        # 如果标量部分为负，取反（确保走最短路径）
        if q_error[3] < 0:
            q_error = -q_error
        
        # 从四元数提取角速度（小角度近似）
        # 对于小角度，q_error ≈ [wx*dt/2, wy*dt/2, wz*dt/2, 1]
        # 这里使用简化的方法：angular_velocity = 2 * gain * q_error[0:3]
        angular_velocity = 2.0 * self.orientation_gain * q_error[:3]
        
        return angular_velocity
    
    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        四元数乘法
        
        Args:
            q1: 第一个四元数 [qx, qy, qz, qw]
            q2: 第二个四元数 [qx, qy, qz, qw]
        
        Returns:
            乘积四元数 [qx, qy, qz, qw]
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        
        return np.array([x, y, z, w])


def create_twist_command(
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    duration_ms: int = 20,  # 20ms (50Hz)
) -> Optional[Base_pb2.TwistCommand]:
    """
    创建 TwistCommand 消息
    
    Args:
        linear_velocity: 线速度 [vx, vy, vz] (m/s)
        angular_velocity: 角速度 [wx, wy, wz] (rad/s)
        duration_ms: 命令持续时间（毫秒）
    
    Returns:
        TwistCommand 消息对象，如果 kortex_api 未安装则返回 None
    """
    if Base_pb2 is None:
        logger.error("kortex_api 未安装，无法创建 TwistCommand")
        return None
    
    twist = Base_pb2.TwistCommand()
    twist.twist.linear_x = float(linear_velocity[0])
    twist.twist.linear_y = float(linear_velocity[1])
    twist.twist.linear_z = float(linear_velocity[2])
    twist.twist.angular_x = float(angular_velocity[0])
    twist.twist.angular_y = float(angular_velocity[1])
    twist.twist.angular_z = float(angular_velocity[2])
    twist.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    twist.duration = duration_ms
    
    return twist


def create_stop_command() -> Optional[Base_pb2.TwistCommand]:
    """
    创建停止命令（零速度的 TwistCommand，duration=0）
    
    Returns:
        TwistCommand 停止命令，如果 kortex_api 未安装则返回 None
    """
    if Base_pb2 is None:
        return None
    
    return create_twist_command(
        np.zeros(3),
        np.zeros(3),
        duration_ms=0
    )
