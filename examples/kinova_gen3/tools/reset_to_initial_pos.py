#!/usr/bin/env python3
"""
Kinova Gen3 复位脚本

根据 initial_pos.txt 文件中的关节角度信息，将机械臂复位到初始位置。

使用示例:
    # 使用默认路径和机器人 IP（默认会等待复位完成）
    python reset_to_initial_pos.py

    # 指定初始位置文件路径
    python reset_to_initial_pos.py --initial-pos-file /path/to/initial_pos.txt

    # 指定机器人 IP
    python reset_to_initial_pos.py --robot-ip 192.168.1.11

    # 不等待复位完成
    python reset_to_initial_pos.py --no-wait-completion
"""

import sys
import os
import math
import argparse
import logging
import time
from pathlib import Path

# 设置 protobuf 环境变量以兼容 kortex_api（必须在导入 kortex_api 之前）
if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" not in os.environ:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("ResetKinovaToInitialPos")

# 导入 kortex_api
try:
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.messages import Base_pb2
    from kortex_api.RouterClient import RouterClient
    from kortex_api.SessionManager import SessionManager
    from kortex_api.TCPTransport import TCPTransport
    from kortex_api.autogen.messages import Session_pb2
except ImportError:
    logger.error("kortex_api 未安装。请从 Kinova 官网下载并安装 kortex_api wheel 包。")
    sys.exit(1)


def connect_to_robot(robot_ip: str, robot_port: int = 10000, username: str = "admin", password: str = "admin"):
    """
    连接到 Kinova 机械臂
    
    Returns:
        tuple: (base, base_cyclic, router, transport, session_manager)
    """
    # 创建 TCP 连接
    transport = TCPTransport()
    transport.connect(robot_ip, robot_port)
    
    # 创建路由器
    router = RouterClient(transport, lambda e: logger.error(f"Router error: {e}"))
    
    # 创建会话
    session_manager = SessionManager(router)
    create_session_info = Session_pb2.CreateSessionInfo()
    create_session_info.username = username
    create_session_info.password = password
    create_session_info.session_inactivity_timeout = 60000  # 60秒
    create_session_info.connection_inactivity_timeout = 2000  # 2秒
    
    session_result = session_manager.CreateSession(create_session_info)
    logger.info(f"会话已创建: {session_result}")
    
    # 创建 Base 客户端（用于位置控制）
    base = BaseClient(router)
    base_cyclic = BaseCyclicClient(router)
    
    return base, base_cyclic, router, transport, session_manager


def read_initial_positions(file_path: str) -> np.ndarray:
    """
    从文件中读取初始关节角度
    
    Args:
        file_path: initial_pos.txt 文件路径
    
    Returns:
        (7,) 关节角度数组（弧度）
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"初始位置文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # 解析数组格式 [value1, value2, ...]
    if content.startswith('[') and content.endswith(']'):
        content = content[1:-1]  # 移除方括号
    
    try:
        # 分割并转换为浮点数
        values = [float(x.strip()) for x in content.split(',')]
        if len(values) != 7:
            raise ValueError(f"期望 7 个关节角度，但文件中包含 {len(values)} 个值")
        return np.array(values)
    except Exception as e:
        raise ValueError(f"解析初始位置文件失败: {e}")


def get_current_joint_positions(base_cyclic: BaseCyclicClient) -> np.ndarray:
    """
    获取当前关节位置
    
    Args:
        base_cyclic: BaseCyclic 客户端
    
    Returns:
        (7,) 关节角度数组（弧度）
    """
    feedback = base_cyclic.RefreshFeedback()
    joint_positions = np.array([
        math.radians(actuator.position) 
        for actuator in feedback.actuators
    ])
    return joint_positions


def format_joint_positions(joint_positions: np.ndarray) -> str:
    """
    格式化关节角度数组为字符串
    """
    return "[" + ", ".join(f"{angle:.6f}" for angle in joint_positions) + "]"


def prepare_robot(base: BaseClient):
    """
    准备机器人：清除故障并设置 servoing mode
    
    Args:
        base: Base 客户端
    """
    # 清除故障（如果有）
    try:
        base.ClearFaults()
        logger.info("已清除故障")
        time.sleep(0.5)  # 等待清除故障完成
    except Exception as e:
        logger.warning(f"清除故障失败（可能没有故障）: {e}")
    
    # 获取当前 servoing mode
    try:
        current_mode = base.GetServoingMode()
        logger.info(f"当前 Servoing mode: {current_mode.servoing_mode}")
    except Exception as e:
        logger.warning(f"获取当前 servoing mode 失败: {e}")
    
    # 设置 servoing mode 为 SINGLE_LEVEL_SERVOING
    try:
        servo_mode = Base_pb2.ServoingModeInformation()
        servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(servo_mode)
        logger.info("正在设置 Servoing mode 为 SINGLE_LEVEL_SERVOING...")
        time.sleep(1.0)  # 增加等待时间，确保设置生效
        
        # 验证 servoing mode
        current_mode = base.GetServoingMode()
        if current_mode.servoing_mode == Base_pb2.SINGLE_LEVEL_SERVOING:
            logger.info(f"Servoing mode 已成功设置为: {current_mode.servoing_mode}")
        else:
            logger.warning(f"Servoing mode 设置后验证失败: 期望 {Base_pb2.SINGLE_LEVEL_SERVOING}, 实际 {current_mode.servoing_mode}")
            # 再次尝试设置
            base.SetServoingMode(servo_mode)
            time.sleep(0.5)
            current_mode = base.GetServoingMode()
            logger.info(f"重新设置后的 Servoing mode: {current_mode.servoing_mode}")
    except Exception as e:
        logger.error(f"设置 servoing mode 失败: {e}")
        raise
    
    # 额外等待，确保机器人准备好接受命令
    time.sleep(0.5)


def move_to_joint_positions(base: BaseClient, joint_positions: np.ndarray, max_retries: int = 3):
    """
    移动到目标关节位置
    
    Args:
        base: Base 客户端
        joint_positions: (7,) 目标关节位置（弧度）
        max_retries: 最大重试次数（默认 3）
    """
    # 创建动作对象
    action = Base_pb2.Action()
    action.name = "reset_to_initial_position"
    action.application_data = ""
    
    # 设置关节角度目标
    for i, pos in enumerate(joint_positions):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = i
        # 转换为角度并归一化到 [0, 360)
        joint_angle.value = math.degrees(pos) % 360
    
    # 执行动作（带重试逻辑）
    logger.info("发送复位命令...")
    last_exception = None
    
    for attempt in range(1, max_retries + 1):
        try:
            base.ExecuteAction(action)
            logger.info("复位命令已成功发送")
            return
        except Exception as e:
            last_exception = e
            error_str = str(e)
            
            # 检查是否是控制权问题
            if "SESSION_NOT_IN_CONTROL" in error_str or "not in control" in error_str.lower():
                logger.warning(f"尝试 {attempt}/{max_retries}: 会话未获得控制权")
                if attempt < max_retries:
                    logger.info("等待 1 秒后重试...")
                    time.sleep(1.0)
                    # 再次尝试准备机器人
                    try:
                        prepare_robot(base)
                    except Exception as prep_error:
                        logger.warning(f"重新准备机器人时出错: {prep_error}")
                else:
                    logger.error("多次尝试后仍无法获得控制权")
                    logger.error("可能的原因：")
                    logger.error("1. 机器人正在被其他程序控制（请关闭其他控制程序）")
                    logger.error("2. 机器人正在执行其他动作（请等待当前动作完成）")
                    logger.error("3. 机器人处于错误状态（可能需要手动复位）")
            else:
                # 其他错误，直接抛出
                raise
    
    # 如果所有重试都失败，抛出最后一个异常
    if last_exception:
        raise last_exception


def wait_for_completion(base: BaseClient, base_cyclic: BaseCyclicClient, target_positions: np.ndarray, 
                       timeout: float = 30.0, tolerance: float = 0.05):
    """
    等待机器人到达目标位置
    
    Args:
        base: Base 客户端（未使用，保留以兼容接口）
        base_cyclic: BaseCyclic 客户端
        target_positions: (7,) 目标关节位置（弧度）
        timeout: 超时时间（秒）
        tolerance: 位置容差（弧度）
    """
    logger.info("等待复位完成...")
    start_time = time.perf_counter()
    
    last_positions = None
    no_motion_count = 0
    no_motion_threshold = 0.001  # 位置变化阈值（弧度）
    no_motion_max_count = 5  # 连续几次无运动则认为完成
    
    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed > timeout:
            logger.warning(f"等待超时（{timeout} 秒），可能仍在运动中")
            return False
        
        try:
            # 获取当前位置
            current_positions = get_current_joint_positions(base_cyclic)
            
            # 检查是否到达目标位置
            errors = np.abs(current_positions - target_positions)
            max_error = np.max(errors)
            
            if max_error < tolerance:
                logger.info(f"复位完成！位置误差: {math.degrees(max_error):.3f} 度")
                return True
            
            # 检查是否停止运动（连续几次位置变化很小）
            if last_positions is not None:
                position_change = np.linalg.norm(current_positions - last_positions)
                if position_change < no_motion_threshold:
                    no_motion_count += 1
                    if no_motion_count >= no_motion_max_count:
                        logger.info(f"运动已停止，位置误差: {math.degrees(max_error):.3f} 度（容差: {math.degrees(tolerance):.3f} 度）")
                        return False
                else:
                    no_motion_count = 0
            
            last_positions = current_positions
            time.sleep(0.1)  # 等待 100ms 后再次检查
            
        except Exception as e:
            logger.warning(f"检查运动状态时出错: {e}，继续等待...")
            time.sleep(0.1)
    
    return False


def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    default_initial_pos_file = script_dir / "initial_pos.txt"
    
    parser = argparse.ArgumentParser(
        description="将 Kinova Gen3 机械臂复位到初始位置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认设置复位（默认会等待复位完成）
  python reset_to_initial_pos.py

  # 指定初始位置文件路径
  python reset_to_initial_pos.py --initial-pos-file /path/to/initial_pos.txt

  # 指定机器人 IP（默认会等待复位完成）
  python reset_to_initial_pos.py --robot-ip 192.168.1.11

  # 不等待复位完成
  python reset_to_initial_pos.py --no-wait-completion
        """
    )
    
    parser.add_argument(
        '--initial-pos-file',
        type=str,
        default=str(default_initial_pos_file),
        help=f'初始位置文件路径（默认: {default_initial_pos_file}）'
    )
    parser.add_argument(
        '--robot-ip',
        type=str,
        default='192.168.1.10',
        help='Kinova 机械臂 IP 地址（默认: 192.168.1.10）'
    )
    parser.add_argument(
        '--robot-port',
        type=int,
        default=10000,
        help='Kinova 机械臂端口（默认: 10000）'
    )
    parser.add_argument(
        '--username',
        type=str,
        default='admin',
        help='登录用户名（默认: admin）'
    )
    parser.add_argument(
        '--password',
        type=str,
        default='admin',
        help='登录密码（默认: admin）'
    )
    parser.add_argument(
        '--wait-completion',
        action='store_true',
        help='等待复位完成（默认启用，使用 --no-wait-completion 可禁用）'
    )
    parser.add_argument(
        '--no-wait-completion',
        dest='wait_completion',
        action='store_false',
        help='不等待复位完成（禁用等待）'
    )
    # 设置默认值为 True（默认启用等待）
    parser.set_defaults(wait_completion=True)
    parser.add_argument(
        '--timeout',
        type=float,
        default=30.0,
        help='等待完成的超时时间（秒，默认: 30.0）'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.05,
        help='位置容差（弧度，默认: 0.05，约 2.87 度）'
    )
    
    args = parser.parse_args()
    
    # 读取初始位置
    logger.info(f"读取初始位置文件: {args.initial_pos_file}")
    try:
        initial_positions = read_initial_positions(args.initial_pos_file)
        logger.info(f"初始关节角度（弧度）: {format_joint_positions(initial_positions)}")
        logger.info(f"初始关节角度（度）  : [{', '.join(f'{math.degrees(p):.3f}' for p in initial_positions)}]")
    except Exception as e:
        logger.error(f"读取初始位置失败: {e}")
        sys.exit(1)
    
    # 连接机器人
    logger.info(f"连接机器人: {args.robot_ip}:{args.robot_port}")
    try:
        base, base_cyclic, router, transport, session_manager = connect_to_robot(
            robot_ip=args.robot_ip,
            robot_port=args.robot_port,
            username=args.username,
            password=args.password,
        )
        logger.info("机器人连接成功\n")
    except Exception as e:
        logger.error(f"连接机器人失败: {e}")
        sys.exit(1)
    
    try:
        # 获取当前位置
        try:
            current_positions = get_current_joint_positions(base_cyclic)
            logger.info(f"当前关节角度（弧度）: {format_joint_positions(current_positions)}")
            logger.info(f"当前关节角度（度）  : [{', '.join(f'{math.degrees(p):.3f}' for p in current_positions)}]")
            
            # 计算位移
            displacement = np.abs(current_positions - initial_positions)
            max_displacement = np.max(displacement)
            logger.info(f"最大关节位移: {math.degrees(max_displacement):.3f} 度\n")
        except Exception as e:
            logger.warning(f"获取当前位置失败: {e}，继续执行复位...")
        
        # 准备机器人：清除故障并设置 servoing mode
        logger.info("准备机器人...")
        try:
            prepare_robot(base)
            logger.info("机器人准备完成\n")
        except Exception as e:
            logger.error(f"准备机器人失败: {e}")
            raise
        
        # 执行复位
        move_to_joint_positions(base, initial_positions)
        
        # 等待完成（如果请求）
        if args.wait_completion:
            wait_for_completion(
                base, 
                base_cyclic, 
                initial_positions, 
                timeout=args.timeout,
                tolerance=args.tolerance
            )
        else:
            logger.info("复位命令已发送，机器人正在移动...")
            logger.info("提示: 默认会等待复位完成，使用 --no-wait-completion 可禁用等待")
        
    except Exception as e:
        logger.error(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 关闭连接
        try:
            session_manager.CloseSession()
            transport.disconnect()
            logger.info("\n已断开连接")
        except Exception as e:
            logger.warning(f"断开连接时出错: {e}")


if __name__ == "__main__":
    main()
