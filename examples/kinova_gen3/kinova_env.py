"""
Kinova Gen3 机器人环境

与 openpi 兼容的 RobotEnv 实现，支持：
- Kinova Gen3 7DOF 机械臂
- 多种动作模式：增量(delta)、绝对位置(absolute)、速度(velocity)
- 双 RealSense D435i 相机
- UDP 夹爪控制
- 键盘急停（ESC/q）

使用示例：
    from kinova_env import KinovaRobotEnv, ActionMode

    env = KinovaRobotEnv(
        robot_ip="192.168.1.10",
        gripper_ip="192.168.1.43",
        external_camera_serial="123456789",
        wrist_camera_serial="987654321",
        action_mode=ActionMode.DELTA,  # 与 openpi 输出兼容
    )

    obs = env.get_observation()
    env.step(action)  # action: (8,) = 7 关节动作 + 1 夹爪
    env.reset()
    env.close()

动作模式说明：
    - ActionMode.DELTA: 增量模式（默认）
      action[:7] 是相对当前关节位置的增量
      target_position = current_position + action
      
    - ActionMode.ABSOLUTE: 绝对位置模式
      action[:7] 是目标关节位置（弧度）
      
    - ActionMode.VELOCITY: 速度模式
      action[:7] 是关节速度（弧度/秒）
"""

import dataclasses
import logging
import math
import threading
import time
from typing import Callable, Optional

import numpy as np

# 设置 protobuf 环境变量以兼容 kortex_api
# kortex_api 需要 protobuf <= 3.20.x，但项目可能使用更新版本
# 这个设置使用纯 Python 实现，性能较慢但兼容性更好
import os
if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" not in os.environ:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


# Kinova kortex API
try:
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2, Session_pb2
    from kortex_api.RouterClient import RouterClient
    from kortex_api.SessionManager import SessionManager
    from kortex_api.TCPTransport import TCPTransport
except ImportError:
    raise ImportError(
        "kortex_api 未安装。请从 Kinova 官网下载并安装 kortex_api wheel 包。"
    )

# 键盘监听
try:
    from pynput import keyboard
except ImportError:
    keyboard = None
    logging.warning("pynput 未安装，键盘急停功能将不可用。请运行: pip install pynput")

# 本地模块
from realsense_camera import DualRealSenseCamera
from control_gripper import send_control, request_feedback_once

logger = logging.getLogger(__name__)

# Kinova Gen3 关节数量
NUM_JOINTS = 7

# 默认控制频率
DEFAULT_CONTROL_FREQUENCY = 40  # Hz

# 夹爪角度范围（完全闭合需要的角度）
GRIPPER_FULL_ANGLE = 2040.0  # 完全闭合所需角度（度）
GRIPPER_SPEED = 20.0  # rad/s
# 初始化时完全张开的角度（确保完全张开）
GRIPPER_INIT_OPEN_ANGLE = 2100.0  # 初始化张开角度（度）


class ActionMode:
    """动作模式"""
    ABSOLUTE = "absolute"  # 绝对位置：action 直接是目标关节位置
    DELTA = "delta"        # 增量位置：action 是相对当前位置的增量
    VELOCITY = "velocity"  # 速度模式：action 是关节速度


@dataclasses.dataclass
class KinovaConfig:
    """Kinova 机器人配置"""
    robot_ip: str = "192.168.1.10"
    robot_port: int = 10000
    username: str = "admin"
    password: str = "admin"
    
    # 夹爪配置
    gripper_ip: str = "192.168.1.43"
    gripper_port: int = 2390
    
    # 相机配置
    external_camera_serial: Optional[str] = None
    wrist_camera_serial: Optional[str] = None
    camera_width: int = 640
    camera_height: int = 480
    
    # 控制配置
    control_frequency: int = DEFAULT_CONTROL_FREQUENCY
    
    # 动作模式：absolute, delta, velocity
    # - absolute: action 是目标关节位置（弧度）- 与 pi05_kinova 配置兼容
    # - delta: action 是相对当前位置的增量，target = current + action
    # - velocity: action 是关节速度，需要低延迟控制
    action_mode: str = ActionMode.ABSOLUTE  # 默认使用绝对位置模式（与 pi05_kinova 输出兼容）
    
    # 初始位置（弧度）- 默认 home 位置
    home_position: tuple = (0.0, -0.26, 3.14, -2.27, 0.0, -0.96, 1.57)


class EmergencyStop:
    """键盘急停监听器"""

    def __init__(self, callback: Callable[[], None]):
        """
        初始化急停监听器。

        Args:
            callback: 急停触发时调用的回调函数
        """
        self._callback = callback
        self._triggered = False
        self._listener = None

        if keyboard is None:
            logger.warning("pynput 不可用，键盘急停功能已禁用")
            return

        def on_press(key):
            try:
                # ESC 键或 q 键触发急停
                if key == keyboard.Key.esc or (hasattr(key, 'char') and key.char == 'q'):
                    if not self._triggered:
                        self._triggered = True
                        logger.warning("急停触发！按下了 ESC/q 键")
                        self._callback()
            except Exception as e:
                logger.error(f"急停处理出错: {e}")

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()
        logger.info("键盘急停监听已启动 (按 ESC 或 q 触发)")

    @property
    def triggered(self) -> bool:
        """返回急停是否已触发"""
        return self._triggered

    def reset(self):
        """重置急停状态"""
        self._triggered = False

    def stop(self):
        """停止监听"""
        if self._listener is not None:
            self._listener.stop()
            logger.info("键盘急停监听已停止")


class KinovaRobotEnv:
    """Kinova Gen3 机器人环境"""

    def __init__(
        self,
        robot_ip: str = "192.168.1.10",
        gripper_ip: str = "192.168.1.43",
        external_camera_serial: Optional[str] = None,
        wrist_camera_serial: Optional[str] = None,
        action_mode: str = ActionMode.DELTA,
        config: Optional[KinovaConfig] = None,
    ):
        """
        初始化 Kinova 机器人环境。

        Args:
            robot_ip: Kinova 机械臂 IP 地址
            gripper_ip: Arduino 夹爪控制器 IP 地址
            external_camera_serial: 外部相机序列号
            wrist_camera_serial: 腕部相机序列号
            action_mode: 动作模式，可选值：
                - "delta": 增量模式，action 是相对当前位置的增量（默认，与 openpi 兼容）
                - "absolute": 绝对位置模式，action 是目标关节位置
                - "velocity": 速度模式，action 是关节速度
            config: 完整配置对象（如果提供，会覆盖其他参数）
        """
        # 配置
        if config is not None:
            self._config = config
        else:
            self._config = KinovaConfig(
                robot_ip=robot_ip,
                gripper_ip=gripper_ip,
                external_camera_serial=external_camera_serial,
                wrist_camera_serial=wrist_camera_serial,
                action_mode=action_mode,
            )

        self._is_connected = False
        self._estop_triggered = False
        
        # 当前夹爪状态：0.0=张开，1.0=闭合（二值动作，非连续角度值）
        self._current_gripper_pos = 0.0

        # 初始化组件
        self._init_robot()
        self._init_cameras()
        self._init_estop()
        
        # 初始化时完全张开夹爪（2100°），确保绝对角度值对应
        self._initialize_gripper()

        logger.info("KinovaRobotEnv 初始化完成")

    def _init_robot(self):
        """初始化 Kinova 机械臂连接"""
        logger.info(f"连接 Kinova 机械臂: {self._config.robot_ip}:{self._config.robot_port}")

        # 创建 TCP 连接
        self._transport = TCPTransport()
        self._transport.connect(self._config.robot_ip, self._config.robot_port)

        # 创建路由器
        self._router = RouterClient(self._transport, lambda e: logger.error(f"Router error: {e}"))

        # 创建会话信息对象
        self._session_manager = SessionManager(self._router)
        create_session_info = Session_pb2.CreateSessionInfo()
        create_session_info.username = self._config.username
        create_session_info.password = self._config.password
        create_session_info.session_inactivity_timeout = 60000  # 60秒（毫秒）
        create_session_info.connection_inactivity_timeout = 2000  # 2秒（毫秒）

        # 创建会话
        session_result = self._session_manager.CreateSession(create_session_info)
        logger.info(f"会话已创建: {session_result}")

        # 创建服务客户端
        self._base = BaseClient(self._router)
        self._base_cyclic = BaseCyclicClient(self._router)

        # 获取关节数量
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self._base.SetServoingMode(base_servo_mode)

        self._is_connected = True
        logger.info("Kinova 机械臂连接成功")

    def _init_cameras(self):
        """初始化相机"""
        logger.info("初始化双相机系统...")
        self._cameras = DualRealSenseCamera(
            external_serial=self._config.external_camera_serial,
            wrist_serial=self._config.wrist_camera_serial,
            width=self._config.camera_width,
            height=self._config.camera_height,
        )

    def _init_estop(self):
        """初始化急停功能"""
        self._estop = EmergencyStop(callback=self._on_estop)
    
    def _initialize_gripper(self):
        """
        初始化夹爪：完全张开到 2100°
        
        在初始化阶段将夹爪完全张开，确保绝对角度值能够对应上。
        使用直接角度控制，不通过归一化位置。
        """
        if not self._is_connected:
            return
        
        logger.info("初始化夹爪：完全张开到 2100°...")
        try:
            # 直接发送张开命令：角度 2100°，速度负值表示张开
            send_control(
                host=self._config.gripper_ip,
                port=self._config.gripper_port,
                speed=-GRIPPER_SPEED,  # 负值表示张开
                angle=GRIPPER_INIT_OPEN_ANGLE,  # 2100 度
                timeout=2.0,  # 初始化时使用更长的超时
            )
            # 更新当前夹爪位置为 0.0（完全张开）
            self._current_gripper_pos = 0.0
            time.sleep(3.0)  # 等待夹爪完全张开
            logger.info("夹爪已完全张开")
        except Exception as e:
            logger.warning(f"初始化夹爪失败: {e}")

    def _on_estop(self):
        """急停回调"""
        self._estop_triggered = True
        try:
            # 调用 Kinova 急停
            self._base.ApplyEmergencyStop()
            logger.warning("已发送急停命令到机械臂")
        except Exception as e:
            logger.error(f"急停命令发送失败: {e}")

    def get_observation(self) -> dict:
        """
        获取当前观察。

        Returns:
            dict: 包含图像和机器人状态的字典，格式与 DROID 兼容
        """
        # 获取相机图像
        external_img, wrist_img = self._cameras.get_frames()

        # 获取机器人状态
        feedback = self._base_cyclic.RefreshFeedback()

        # 提取关节位置（转换为弧度）
        # 将角度 wrap 到 [-180, 180] 度，然后转换为弧度
        joint_positions = np.array([
            math.radians((actuator.position + 180) % 360 - 180)
            for actuator in feedback.actuators
        ])

        # 提取末端执行器位姿
        cartesian_position = np.array([
            feedback.base.tool_pose_x,
            feedback.base.tool_pose_y,
            feedback.base.tool_pose_z,
            math.radians(feedback.base.tool_pose_theta_x),
            math.radians(feedback.base.tool_pose_theta_y),
            math.radians(feedback.base.tool_pose_theta_z),
        ])

        return {
            "image": {
                f"{self._config.external_camera_serial}_left": external_img,
                f"{self._config.wrist_camera_serial}_left": wrist_img,
            },
            "robot_state": {
                "joint_positions": joint_positions,
                "gripper_position": self._current_gripper_pos,
                "cartesian_position": cartesian_position,
            },
        }

    def step(self, action: np.ndarray):
        """
        执行动作。

        Args:
            action: (8,) 数组，前 7 个是关节动作，最后 1 个是夹爪状态（0.0=张开，1.0=闭合）
                   动作解释取决于 action_mode:
                   - absolute: 目标关节位置（弧度）
                   - delta: 相对当前位置的增量
                   - velocity: 关节速度
                   注意：夹爪状态是二值动作，实际使用中只有 0.0 和 1.0 两个值
        """
        if self._estop_triggered:
            logger.warning("急停已触发，忽略动作命令")
            return

        action = np.asarray(action)
        if action.shape != (8,):
            raise ValueError(f"动作维度应为 (8,)，但收到 {action.shape}")

        # 分离关节动作和夹爪动作
        joint_action = action[:7]
        gripper_pos = float(action[7])  # 夹爪状态：0.0=张开，1.0=闭合（二值动作）

        # 根据动作模式处理关节动作
        if self._config.action_mode == ActionMode.ABSOLUTE:
            #print("111111111")
            # 绝对位置模式：直接使用 action 作为目标位置
            target_positions = joint_action
        elif self._config.action_mode == ActionMode.DELTA:
            # 增量模式：target = current + delta
            current_positions = self._get_current_joint_positions()
            target_positions = current_positions + joint_action
        elif self._config.action_mode == ActionMode.VELOCITY:
            # 速度模式：使用速度控制（需要不同的控制接口）
            self._send_joint_velocities(joint_action)
            self._control_gripper(gripper_pos)
            return
        else:
            raise ValueError(f"未知的动作模式: {self._config.action_mode}")

        # 执行关节位置控制
        self._move_to_joint_positions(target_positions)

        # 执行夹爪控制
        self._control_gripper(gripper_pos)

    def _get_current_joint_positions(self) -> np.ndarray:
        """
        获取当前关节位置。

        Returns:
            np.ndarray: (7,) 当前关节位置（弧度）
        """
        feedback = self._base_cyclic.RefreshFeedback()
        positions = np.array([
            math.radians(actuator.position)
            for actuator in feedback.actuators
        ])
        return positions

    def _move_to_joint_positions(self, target_positions: np.ndarray):
        """
        移动到目标关节位置。

        Args:
            target_positions: (7,) 目标关节位置（弧度）
        """
        # 创建动作对象
        action = Base_pb2.Action()
        action.name = "joint_position_action"
        action.application_data = ""

        # 设置关节角度目标
        for i, pos in enumerate(target_positions):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = i
            joint_angle.value = math.degrees(pos) % 360  # kortex 使用角度

        # 执行动作（非阻塞）
        self._base.ExecuteAction(action)

    def _send_joint_velocities(self, velocities: np.ndarray):
        """
        发送关节速度命令（低延迟控制）。

        Args:
            velocities: (7,) 关节速度（弧度/秒）
        """
        # 使用 BaseCyclic 进行低延迟速度控制
        command = BaseCyclic_pb2.Command()
        
        # 获取当前状态作为基础
        feedback = self._base_cyclic.RefreshFeedback()
        
        for i, vel in enumerate(velocities):
            actuator_command = command.actuators.add()
            actuator_command.position = feedback.actuators[i].position
            actuator_command.velocity = math.degrees(vel)  # kortex 使用角度/秒
            actuator_command.torque_joint = 0.0
            actuator_command.command_id = feedback.actuators[i].command_id
        
        # 发送命令
        try:
            self._base_cyclic.Refresh(command)
        except Exception as e:
            logger.warning(f"速度控制命令发送失败: {e}")

    def _control_gripper(self, target_pos: float):
        """
        控制夹爪状态（张开/闭合）。

        Args:
            target_pos: 目标状态，0.0=闭合，1.0=张开（二值动作，非连续角度值）
        
        注意：
            - 虽然函数接受 [0, 1] 范围的浮点数，但在实际使用中只接受 0.0 和 1.0 两个值
            - 函数内部会将状态变化转换为角度变化量（delta * GRIPPER_FULL_ANGLE）来控制物理夹爪
        """
        target_pos = np.clip(target_pos, 0.0, 1.0)

        # 计算位置变化
        delta = target_pos - self._current_gripper_pos

        if abs(delta) < 0.01:
            # 位置变化太小，不需要移动
            return

        # 计算角度变化量
        angle = abs(delta) * GRIPPER_FULL_ANGLE

        # 速度符号决定方向：正=闭合，负=张开
        speed = GRIPPER_SPEED if delta > 0 else -GRIPPER_SPEED

        try:
            send_control(
                host=self._config.gripper_ip,
                port=self._config.gripper_port,
                speed=speed,
                angle=angle,
                timeout=0.5,
            )
            self._current_gripper_pos = target_pos
        except Exception as e:
            logger.warning(f"夹爪控制失败: {e}")

    def reset(self):
        """重置机器人到初始位置"""
        if self._estop_triggered:
            logger.warning("急停状态下无法重置，请先清除急停")
            return

        logger.info("重置机器人到初始位置...")

        # 移动到 home 位置
        home_positions = np.array(self._config.home_position)
        self._move_to_home(home_positions)

        # 完全张开夹爪到 2100°（确保绝对角度值对应）
        self._initialize_gripper()

        # 等待移动完成
        time.sleep(2.0)
        logger.info("机器人已重置")

    def _move_to_home(self, home_positions: np.ndarray):
        """移动到 home 位置（阻塞）"""
        action = Base_pb2.Action()
        action.name = "home_action"
        action.application_data = ""

        for i, pos in enumerate(home_positions):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = i
            joint_angle.value = math.degrees(pos)

        # 使用 ExecuteActionFromReference 进行阻塞移动
        self._base.ExecuteAction(action)

    def clear_estop(self):
        """清除急停状态"""
        try:
            self._base.ClearFaults()
            self._estop_triggered = False
            self._estop.reset()
            logger.info("急停状态已清除")
        except Exception as e:
            logger.error(f"清除急停失败: {e}")

    def close(self):
        """释放所有资源"""
        logger.info("关闭 KinovaRobotEnv...")

        # 停止急停监听
        if hasattr(self, '_estop'):
            self._estop.stop()

        # 关闭相机
        if hasattr(self, '_cameras'):
            self._cameras.close()

        # 断开机器人连接
        if self._is_connected:
            try:
                self._session_manager.CloseSession()
                self._transport.disconnect()
                self._is_connected = False
                logger.info("Kinova 机械臂连接已断开")
            except Exception as e:
                logger.error(f"断开连接时出错: {e}")

        logger.info("KinovaRobotEnv 已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ============================================================
# 便捷函数
# ============================================================

def create_kinova_env(
    robot_ip: str = "192.168.1.10",
    gripper_ip: str = "192.168.1.43",
    external_camera_serial: Optional[str] = None,
    wrist_camera_serial: Optional[str] = None,
    action_mode: str = ActionMode.DELTA,
) -> KinovaRobotEnv:
    """
    创建 Kinova 机器人环境的便捷函数。

    Args:
        robot_ip: Kinova 机械臂 IP 地址
        gripper_ip: Arduino 夹爪控制器 IP 地址
        external_camera_serial: 外部相机序列号
        wrist_camera_serial: 腕部相机序列号
        action_mode: 动作模式（delta/absolute/velocity）

    Returns:
        KinovaRobotEnv: 机器人环境实例
    """
    return KinovaRobotEnv(
        robot_ip=robot_ip,
        gripper_ip=gripper_ip,
        external_camera_serial=external_camera_serial,
        wrist_camera_serial=wrist_camera_serial,
        action_mode=action_mode,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kinova 机器人环境测试")
    parser.add_argument("--robot-ip", type=str, default="192.168.1.10", help="机械臂 IP")
    parser.add_argument("--gripper-ip", type=str, default="192.168.1.43", help="夹爪 IP")
    parser.add_argument("--external-serial", type=str, help="外部相机序列号")
    parser.add_argument("--wrist-serial", type=str, help="腕部相机序列号")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print("创建机器人环境...")
    env = KinovaRobotEnv(
        robot_ip=args.robot_ip,
        gripper_ip=args.gripper_ip,
        external_camera_serial=args.external_serial,
        wrist_camera_serial=args.wrist_serial,
    )

    try:
        print("\n获取观察...")
        obs = env.get_observation()
        print(f"关节位置: {obs['robot_state']['joint_positions']}")
        print(f"夹爪位置: {obs['robot_state']['gripper_position']}")
        print(f"末端位姿: {obs['robot_state']['cartesian_position']}")

        for key, img in obs['image'].items():
            print(f"图像 {key}: shape={img.shape}, dtype={img.dtype}")

        print("\n按 Ctrl+C 退出...")
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        env.close()

