#!/usr/bin/env python3
"""
Kinova 机械臂轨迹回放脚本

从 data 目录读取轨迹数据并复现机器人动作。

使用示例：
    # 回放最新的轨迹
    python replay_routine.py
    
    # 回放指定的轨迹文件
    python replay_routine.py --data-path data/General_manipulation_task_20260114_093000/replay_data/episode_001_replay_20260114_093159.npz
    
    # 回放指定 session 的最新轨迹
    python replay_routine.py --session data/General_manipulation_task_20260114_093000
    
    # 指定机器人 IP 和夹爪 IP
    python replay_routine.py --robot-ip 192.168.1.10 --gripper-ip 192.168.1.43
"""

import sys
import os

# 设置 protobuf 环境变量以兼容 kortex_api（必须在导入 kortex_api 之前）
if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" not in os.environ:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

# 导入本地模块
from kinova_env import KinovaRobotEnv, ActionMode
from trajectory_smoothing import TrajectorySmoother
from velocity_controller import CartesianVelocityController, create_twist_command, create_stop_command

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("ReplayRoutine")


class TrajectoryReplayer:
    """
    轨迹回放器
    
    从保存的轨迹数据文件中读取机器人轨迹，并使用 KinovaRobotEnv
    精确复现机器人的动作序列。
    
    主要功能：
        - 加载轨迹数据（.npz 格式）
        - 使用绝对位置模式精确回放轨迹
        - 支持回放速度控制
        - 支持指定回放范围
    
    数据格式要求：
        - joint_positions: (N, 7) 关节位置数组（弧度）
        - gripper_pos: (N,) 夹爪状态数组，0.0=张开，1.0=闭合（二值动作）
        - timestamp: (N,) 时间戳数组（可选）
        - eef_pose: (N, 7) 末端执行器位姿（可选）
        - action: (N, 7) 动作数组（可选）
    
    属性：
        robot_ip (str): Kinova 机械臂 IP 地址
        gripper_ip (str): Arduino 夹爪控制器 IP 地址
        env (KinovaRobotEnv): 机器人环境实例（使用绝对位置模式）
    """
    
    def __init__(
        self,
        robot_ip: str = "192.168.1.10",
        gripper_ip: str = "192.168.1.43",
        external_camera_serial: Optional[str] = None,
        wrist_camera_serial: Optional[str] = None,
    ):
        """
        初始化轨迹回放器。
        
        Args:
            robot_ip: Kinova 机械臂 IP 地址
            gripper_ip: Arduino 夹爪控制器 IP 地址
            external_camera_serial: 外部相机序列号（可选）
            wrist_camera_serial: 腕部相机序列号（可选）
        """
        self.robot_ip = robot_ip
        self.gripper_ip = gripper_ip
        self.external_camera_serial = external_camera_serial
        self.wrist_camera_serial = wrist_camera_serial
        
        # 初始化机器人环境（使用绝对位置模式进行回放）
        logger.info(f"连接机器人: {robot_ip}")
        self.env = KinovaRobotEnv(
            robot_ip=robot_ip,
            gripper_ip=gripper_ip,
            external_camera_serial=external_camera_serial,
            wrist_camera_serial=wrist_camera_serial,
            action_mode=ActionMode.ABSOLUTE,  # 使用绝对位置模式回放
        )
        logger.info("机器人环境初始化完成\n")
        
    def load_trajectory(self, data_path: Path) -> dict:
        """
        加载轨迹数据文件
        
        Args:
            data_path: 轨迹数据文件路径（.npz 格式）
            
        Returns:
            dict: 包含轨迹数据的字典，包含以下键：
                - 'joint_positions': (N, 7) 关节位置数组（弧度）
                - 'gripper_positions': (N,) 夹爪状态数组，0.0=张开，1.0=闭合（二值动作）
                - 'timestamps': (N,) 时间戳数组（可选，如果文件中存在）
                - 'steps': (N,) 步数数组（可选）
                - 'eef_poses': (N, 7) 末端执行器位姿（可选）
                - 'actions': (N, 7) 动作数组（可选）
        
        Raises:
            FileNotFoundError: 如果轨迹文件不存在
            ValueError: 如果轨迹数据为空
        
        注意：
            - 必需字段：joint_positions, gripper_pos
            - 可选字段：timestamp, step, eef_pose, action
            - gripper_pos 是二值状态（0.0=张开，1.0=闭合），不是连续的归一化角度值
        """
        if not data_path.exists():
            raise FileNotFoundError(f"轨迹文件不存在: {data_path}")
        
        logger.info(f"加载轨迹数据: {data_path.name}")
        data = np.load(data_path, allow_pickle=True)
        
        # 提取轨迹数据（注意：np.load 返回的对象支持字典访问）
        trajectory = {
            'joint_positions': data['joint_positions'],  # (N, 7)
            'gripper_positions': data['gripper_pos'],    # (N,)
        }
        
        # 可选字段
        if 'timestamp' in data:
            trajectory['timestamps'] = data['timestamp']
        else:
            trajectory['timestamps'] = None
            
        if 'step' in data:
            trajectory['steps'] = data['step']
        else:
            trajectory['steps'] = None
            
        if 'eef_pose' in data:
            trajectory['eef_poses'] = data['eef_pose']
        else:
            trajectory['eef_poses'] = None
            
        if 'action' in data:
            trajectory['actions'] = data['action']
        else:
            trajectory['actions'] = None
        
        # 采集频率（用于平滑回放）
        if 'collection_frequency' in data:
            trajectory['collection_frequency'] = float(data['collection_frequency'])
        else:
            # 旧数据可能不包含该字段；默认回退到 30Hz（与当前采集默认一致）
            trajectory['collection_frequency'] = 30.0
        
        # 验证数据
        num_steps = len(trajectory['joint_positions'])
        if num_steps == 0:
            raise ValueError("轨迹数据为空")
        
        logger.info(f"轨迹数据加载完成: {num_steps} 步，采集频率: {trajectory['collection_frequency']} Hz\n")
        return trajectory
    
    def replay_trajectory(
        self,
        trajectory: dict,
        playback_speed: float = 1.0,
        start_step: int = 0,
        end_step: Optional[int] = None,
    ):
        """
        回放轨迹
        
        使用绝对位置模式精确复现机器人的动作序列。
        
        Args:
            trajectory: 轨迹数据字典（由 load_trajectory 返回）
            playback_speed: 回放速度倍数
                - 1.0: 原始速度
                - 2.0: 2倍速（更快）
                - 0.5: 0.5倍速（更慢）
            start_step: 起始步数（默认从第0步开始）
            end_step: 结束步数（默认到轨迹末尾，即 None）
        
        Raises:
            ValueError: 如果起始或结束步数无效
        
        执行流程：
            1. 验证步数范围
            2. 计算时间间隔（优先使用时间戳；否则使用数据中记录的 collection_frequency）
            3. 移动到起始位置
            4. 按时间间隔逐步执行轨迹
            5. 显示回放进度（每 50 步）
        
        注意：
            - 使用绝对位置模式（ActionMode.ABSOLUTE）确保精确复现
            - 如果轨迹包含时间戳，使用原始时间间隔；否则使用固定频率
            - 支持键盘中断（Ctrl+C）
        """
        joint_positions = trajectory['joint_positions']
        gripper_positions = trajectory['gripper_positions']
        timestamps = trajectory.get('timestamps')
        
        num_steps = len(joint_positions)
        end_step = end_step if end_step is not None else num_steps
        
        if start_step < 0 or start_step >= num_steps:
            raise ValueError(f"起始步数 {start_step} 超出范围 [0, {num_steps})")
        if end_step <= start_step or end_step > num_steps:
            raise ValueError(f"结束步数 {end_step} 无效")
        
        logger.info(f"开始回放轨迹: 步数 {start_step} 到 {end_step} (共 {end_step - start_step} 步)")
        logger.info(f"回放速度: {playback_speed}x\n")
        
        # 计算时间间隔
        # 时间间隔用于控制回放速度，确保按照原始采集频率或时间戳回放
        num_actions = end_step - start_step
        collection_frequency = trajectory.get('collection_frequency', 30.0)
        if timestamps is not None and len(timestamps) > 1:
            # 使用原始时间戳计算间隔（更精确）
            # 注意：时间戳数组长度应该与 joint_positions 相同
            if len(timestamps) != num_steps:
                logger.warning(f"时间戳数组长度 ({len(timestamps)}) 与轨迹长度 ({num_steps}) 不匹配，使用固定频率")
                # 时间戳不匹配，回退到固定频率
                time_diffs = np.ones(num_actions) / collection_frequency / playback_speed
            else:
                # 计算相邻步骤之间的时间差
                # np.diff 计算相邻元素差值，得到每步之间的时间间隔
                time_diffs = np.diff(timestamps[start_step:end_step+1])
                # 根据回放速度调整时间间隔
                time_diffs = time_diffs / playback_speed
        else:
            # 如果没有时间戳，使用采集频率（优先使用数据中记录的 collection_frequency）
            time_diffs = np.ones(num_actions) / collection_frequency / playback_speed
        
        try:
            # 移动到起始位置
            # 先移动到轨迹的起始位置，确保从正确的位置开始回放
            logger.info("移动到起始位置...")
            start_joint_pos = joint_positions[start_step]  # 起始关节位置（弧度）
            start_gripper_pos = gripper_positions[start_step]  # 起始夹爪状态：0.0=张开，1.0=闭合
            
            # 构造动作：7个关节位置 + 1个夹爪状态（0.0=张开，1.0=闭合）
            # KinovaRobotEnv.step() 需要 (8,) 数组
            start_action = np.concatenate([start_joint_pos, [start_gripper_pos]])
            self.env.step(start_action)  # 使用绝对位置模式移动到起始位置
            time.sleep(1.0)  # 等待机器人到达起始位置
            
            logger.info("开始回放...\n")
            
            # 回放轨迹
            # 从 start_step 到 end_step-1，每次执行下一步的目标位置
            for i in range(start_step, end_step):
                # 获取目标位置
                # 注意：我们在第 i 步时执行第 i+1 步的目标位置
                # 这样可以确保从当前位置平滑移动到下一个位置
                target_joint_pos = joint_positions[i + 1]  # 下一步的目标关节位置（弧度）
                target_gripper_pos = gripper_positions[i + 1]  # 下一步的目标夹爪状态：0.0=张开，1.0=闭合
                
                # 构造动作数组
                action = np.concatenate([target_joint_pos, [target_gripper_pos]])
                
                # 执行动作（使用绝对位置模式）
                self.env.step(action)
                
                # 等待到下一步
                # 根据计算的时间间隔等待，确保回放速度正确
                if i < end_step - 1:
                    wait_time = time_diffs[i - start_step]
                    if wait_time > 0:
                        time.sleep(wait_time)
                
                # 显示进度（每 50 步显示一次）
                if (i - start_step + 1) % 50 == 0:
                    progress = (i - start_step + 1) / (end_step - start_step) * 100
                    logger.info(f"回放进度: {progress:.1f}% ({i - start_step + 1}/{end_step - start_step})")
            
            logger.info("\n轨迹回放完成！\n")
            
        except KeyboardInterrupt:
            logger.warning("\n回放被用户中断\n")
        except Exception as e:
            logger.error(f"回放过程中出错: {e}\n")
            raise
    
    def _get_current_eef_pose(self) -> np.ndarray:
        """
        获取当前末端执行器位姿（从机器人反馈中获取）
        
        Returns:
            末端执行器位姿数组 [x, y, z, qx, qy, qz, qw]（四元数格式）
        """
        from scipy.spatial.transform import Rotation
        import math
        
        obs = self.env.get_observation()
        cartesian_position = obs['robot_state']['cartesian_position']  # [x, y, z, rx, ry, rz] (欧拉角)
        
        # 转换为四元数
        r = Rotation.from_euler('xyz', cartesian_position[3:], degrees=False)
        quat = r.as_quat()  # [qx, qy, qz, qw]
        
        # 组合位置和四元数
        eef_pose = np.concatenate([cartesian_position[:3], quat])
        return eef_pose
    
    def replay_trajectory_smooth(
        self,
        trajectory: dict,
        playback_speed: float = 1.0,
        start_step: int = 0,
        end_step: Optional[int] = None,
        smoothing_window_size: int = 5,
    ):
        """
        平滑轨迹回放（使用速度控制和平滑滤波）
        
        使用笛卡尔空间速度控制和平滑滤波实现更平滑的轨迹回放。
        可用于VLA策略输出动作的执行。
        
        Args:
            trajectory: 轨迹数据字典（由 load_trajectory 返回）
            playback_speed: 回放速度倍数（1.0 = 原始速度）
            start_step: 起始步数（默认从第0步开始）
            end_step: 结束步数（默认到轨迹末尾，即 None）
            smoothing_window_size: 平滑窗口大小（用于轨迹平滑的点数）
        
        Raises:
            ValueError: 如果轨迹数据无效或缺少末端位姿信息
        """
        joint_positions = trajectory['joint_positions']
        gripper_positions = trajectory['gripper_positions']
        eef_poses = trajectory.get('eef_poses')
        collection_frequency = trajectory.get('collection_frequency', 30.0)
        
        num_steps = len(joint_positions)
        end_step = end_step if end_step is not None else num_steps
        
        if start_step < 0 or start_step >= num_steps:
            raise ValueError(f"起始步数 {start_step} 超出范围 [0, {num_steps})")
        if end_step <= start_step or end_step > num_steps:
            raise ValueError(f"结束步数 {end_step} 无效")
        
        # 检查是否有末端位姿数据
        if eef_poses is None:
            raise ValueError("轨迹数据中缺少末端位姿信息（eef_pose），无法进行平滑回放")
        
        logger.info(f"开始平滑轨迹回放: 步数 {start_step} 到 {end_step} (共 {end_step - start_step} 步)")
        logger.info(f"回放速度: {playback_speed}x，采集频率: {collection_frequency} Hz\n")
        
        # 初始化平滑器和速度控制器
        smoother = TrajectorySmoother(window_size=smoothing_window_size)
        velocity_controller = CartesianVelocityController(
            max_linear_velocity=0.05,  # 5 cm/s
            max_angular_velocity=0.5,  # rad/s
            position_gain=2.0,
            orientation_gain=1.0,
        )
        
        # 计算控制周期（使用原始采集频率）
        control_dt = 1.0 / collection_frequency / playback_speed
        
        try:
            # 移动到起始位置（使用关节位置控制）
            logger.info("移动到起始位置...")
            start_joint_pos = joint_positions[start_step]
            start_gripper_pos = gripper_positions[start_step]
            start_action = np.concatenate([start_joint_pos, [start_gripper_pos]])
            self.env.step(start_action)
            time.sleep(1.0)  # 等待到达起始位置
            
            # 初始化平滑器（填充窗口）
            current_eef_pose = self._get_current_eef_pose()
            for _ in range(smoothing_window_size):
                smoother.add_pose(current_eef_pose)
            
            logger.info("开始平滑回放...\n")
            
            # 回放轨迹
            for i in range(start_step, end_step):
                # 获取目标末端位姿
                target_eef_pose = eef_poses[i]
                target_gripper_pos = gripper_positions[i]
                
                # 添加到平滑器
                smoother.add_pose(target_eef_pose)
                
                # 获取平滑后的目标位姿
                smoothed_target_pose = smoother.get_smoothed_pose()
                if smoothed_target_pose is None:
                    # 窗口未满，使用原始目标位姿
                    smoothed_target_pose = target_eef_pose
                
                # 获取当前位置
                current_eef_pose = self._get_current_eef_pose()
                
                # 计算速度命令
                linear_velocity, angular_velocity = velocity_controller.compute_velocity(
                    current_eef_pose,
                    smoothed_target_pose
                )
                
                # 创建并发送速度命令
                twist_cmd = create_twist_command(
                    linear_velocity,
                    angular_velocity,
                    duration_ms=int(control_dt * 1000)
                )
                if twist_cmd is not None:
                    self.env._base.SendTwistCommand(twist_cmd)
                
                # 控制夹爪（使用关节位置控制）
                if abs(target_gripper_pos - self.env._current_gripper_pos) > 0.1:
                    self.env._control_gripper(target_gripper_pos)
                
                # 等待控制周期
                time.sleep(control_dt)
                
                # 显示进度（每 50 步显示一次）
                if (i - start_step + 1) % 50 == 0:
                    progress = (i - start_step + 1) / (end_step - start_step) * 100
                    logger.info(f"回放进度: {progress:.1f}% ({i - start_step + 1}/{end_step - start_step})")
            
            # 停止机器人
            stop_cmd = create_stop_command()
            if stop_cmd is not None:
                self.env._base.SendTwistCommand(stop_cmd)
            
            logger.info("\n平滑轨迹回放完成！\n")
            
        except KeyboardInterrupt:
            logger.warning("\n回放被用户中断\n")
            # 停止机器人
            stop_cmd = create_stop_command()
            if stop_cmd is not None:
                self.env._base.SendTwistCommand(stop_cmd)
        except Exception as e:
            logger.error(f"回放过程中出错: {e}\n")
            # 停止机器人
            stop_cmd = create_stop_command()
            if stop_cmd is not None:
                self.env._base.SendTwistCommand(stop_cmd)
            raise
    
    def close(self):
        """
        关闭机器人环境
        
        断开机器人连接、关闭相机等资源。
        应在回放完成后调用此方法。
        """
        if hasattr(self, 'env'):
            self.env.close()
            logger.info("机器人环境已关闭\n")


def find_latest_trajectory(data_dir: Path) -> Optional[Path]:
    """
    查找最新的轨迹文件
    
    在所有 session 目录的 replay_data 子目录中查找最新的轨迹文件。
    
    Args:
        data_dir: 数据根目录路径（包含多个 session 目录）
        
    Returns:
        Optional[Path]: 最新的轨迹文件路径，如果未找到则返回 None
    
    查找逻辑：
        1. 遍历 data_dir 下的所有 session 目录
        2. 在每个 session/replay_data 目录中查找轨迹文件
        3. 按文件修改时间排序，返回最新的文件
    
    轨迹文件命名格式：
        episode_{count:03d}_replay_{timestamp}.npz
    """
    if not data_dir.exists():
        return None
    
    # 查找所有 replay_data 目录
    replay_dirs = []
    for session_dir in data_dir.iterdir():
        if session_dir.is_dir():
            replay_dir = session_dir / "replay_data"
            if replay_dir.exists():
                replay_dirs.append(replay_dir)
    
    if not replay_dirs:
        return None
    
    # 查找所有轨迹文件
    trajectory_files = []
    for replay_dir in replay_dirs:
        for file in replay_dir.glob("episode_*_replay_*.npz"):
            trajectory_files.append(file)
    
    if not trajectory_files:
        return None
    
    # 按修改时间排序，返回最新的
    trajectory_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return trajectory_files[0]


def find_trajectory_in_session(session_dir: Path) -> Optional[Path]:
    """
    在指定 session 中查找最新的轨迹文件
    
    Args:
        session_dir: session 目录路径（例如：data/General_manipulation_task_20260114_093000）
        
    Returns:
        Optional[Path]: 最新的轨迹文件路径，如果未找到则返回 None
    
    查找逻辑：
        1. 在 session_dir/replay_data 目录中查找轨迹文件
        2. 按文件修改时间排序，返回最新的文件
    
    注意：
        - 如果 replay_data 目录不存在，返回 None
        - 如果目录中没有轨迹文件，返回 None
    """
    replay_dir = session_dir / "replay_data"
    if not replay_dir.exists():
        return None
    
    trajectory_files = list(replay_dir.glob("episode_*_replay_*.npz"))
    if not trajectory_files:
        return None
    
    # 按修改时间排序，返回最新的
    trajectory_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return trajectory_files[0]


def main():
    """
    主函数：解析命令行参数并执行轨迹回放
    
    支持的命令行参数：
        - 数据选择（互斥）：
            --data-path: 指定轨迹文件路径
            --session: 指定 session 目录（使用最新的轨迹）
            （默认）：自动查找最新的轨迹文件
        
        - 机器人连接：
            --robot-ip: Kinova 机械臂 IP（默认: 192.168.1.10）
            --gripper-ip: 夹爪控制器 IP（默认: 192.168.1.43）
            --external-camera-serial: 外部相机序列号（可选）
            --wrist-camera-serial: 腕部相机序列号（可选）
        
        - 回放参数：
            --playback-speed: 回放速度倍数（默认: 1.0）
            --start-step: 起始步数（默认: 0）
            --end-step: 结束步数（默认: 到轨迹末尾）
        
        - 其他：
            --data-dir: 数据根目录（默认: 脚本目录/data）
    
    执行流程：
        1. 解析命令行参数
        2. 确定轨迹文件路径
        3. 创建轨迹回放器
        4. 加载轨迹数据
        5. 执行回放
        6. 清理资源
    """
    parser = argparse.ArgumentParser(
        description="Kinova 机械臂轨迹回放脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 回放最新的轨迹
  python replay_routine.py
  
  # 回放指定的轨迹文件
  python replay_routine.py --data-path data/.../replay_data/episode_001_replay_xxx.npz
  
  # 回放指定 session 的最新轨迹
  python replay_routine.py --session data/General_manipulation_task_20260114_093000
  
  # 指定回放速度和范围
  python replay_routine.py --playback-speed 0.5 --start-step 100 --end-step 500
        """
    )
    
    # 数据选择参数
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        '--data-path',
        type=str,
        help='轨迹数据文件路径（.npz 文件）'
    )
    data_group.add_argument(
        '--session',
        type=str,
        help='session 目录路径（将使用该 session 中最新的轨迹）'
    )
    
    # 机器人连接参数
    parser.add_argument(
        '--robot-ip',
        type=str,
        default='192.168.1.10',
        help='Kinova 机械臂 IP 地址（默认: 192.168.1.10）'
    )
    parser.add_argument(
        '--gripper-ip',
        type=str,
        default='192.168.1.43',
        help='Arduino 夹爪控制器 IP 地址（默认: 192.168.1.43）'
    )
    parser.add_argument(
        '--external-camera-serial',
        type=str,
        default=None,
        help='外部相机序列号（可选）'
    )
    parser.add_argument(
        '--wrist-camera-serial',
        type=str,
        default=None,
        help='腕部相机序列号（可选）'
    )
    
    # 回放参数
    parser.add_argument(
        '--playback-speed',
        type=float,
        default=1.0,
        help='回放速度倍数（1.0 = 原始速度，2.0 = 2倍速，0.5 = 0.5倍速，默认: 1.0）'
    )
    parser.add_argument(
        '--start-step',
        type=int,
        default=0,
        help='起始步数（默认: 0）'
    )
    parser.add_argument(
        '--end-step',
        type=int,
        default=None,
        help='结束步数（默认: 到轨迹末尾）'
    )
    parser.add_argument(
        '--smooth',
        action='store_true',
        default=True,
        help='使用平滑回放（速度控制和平滑滤波，默认启用）'
    )
    parser.add_argument(
        '--no-smooth',
        dest='smooth',
        action='store_false',
        help='禁用平滑回放（使用原始位置控制模式）'
    )
    parser.add_argument(
        '--smoothing-window-size',
        type=int,
        default=5,
        help='平滑窗口大小（用于轨迹平滑的点数，默认: 5）'
    )
    
    # 数据目录
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='数据根目录（默认: 脚本所在目录的 data 子目录）'
    )
    
    args = parser.parse_args()
    
    # 确定数据目录
    # 如果用户指定了数据目录，使用指定的；否则使用脚本目录下的 data 子目录
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        script_dir = Path(__file__).parent
        data_dir = script_dir / "data"
    
    # 确定轨迹文件路径
    # 优先级：--data-path > --session > 默认（最新轨迹）
    if args.data_path:
        # 直接使用指定的轨迹文件路径
        trajectory_path = Path(args.data_path)
    elif args.session:
        # 在指定的 session 目录中查找最新的轨迹文件
        session_dir = Path(args.session)
        trajectory_path = find_trajectory_in_session(session_dir)
        if trajectory_path is None:
            logger.error(f"在 session {session_dir} 中未找到轨迹文件\n")
            sys.exit(1)
    else:
        # 默认：在所有 session 中查找最新的轨迹文件
        trajectory_path = find_latest_trajectory(data_dir)
        if trajectory_path is None:
            logger.error(f"在 {data_dir} 中未找到轨迹文件\n")
            logger.info("请使用 --data-path 或 --session 参数指定轨迹文件\n")
            sys.exit(1)
    
    logger.info(f"使用轨迹文件: {trajectory_path}\n")
    
    # 创建轨迹回放器
    # 初始化机器人环境（使用绝对位置模式）
    replayer = TrajectoryReplayer(
        robot_ip=args.robot_ip,
        gripper_ip=args.gripper_ip,
        external_camera_serial=args.external_camera_serial,
        wrist_camera_serial=args.wrist_camera_serial,
    )
    
    try:
        # 加载轨迹数据
        trajectory = replayer.load_trajectory(trajectory_path)
        
        # 执行轨迹回放（根据参数选择平滑或原始模式）
        if args.smooth:
            replayer.replay_trajectory_smooth(
                trajectory,
                playback_speed=args.playback_speed,
                start_step=args.start_step,
                end_step=args.end_step,
                smoothing_window_size=args.smoothing_window_size,
            )
        else:
            replayer.replay_trajectory(
                trajectory,
                playback_speed=args.playback_speed,
                start_step=args.start_step,
                end_step=args.end_step,
            )
        
    except KeyboardInterrupt:
        # 用户中断（Ctrl+C）
        logger.warning("\n程序被用户中断\n")
    except Exception as e:
        # 其他异常
        logger.error(f"回放失败: {e}\n")
        sys.exit(1)
    finally:
        # 确保清理资源（断开机器人连接、关闭相机等）
        replayer.close()


if __name__ == "__main__":
    main()
