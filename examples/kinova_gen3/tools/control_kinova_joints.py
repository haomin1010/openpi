#!/usr/bin/env python3
"""
Kinova Gen3 关节角度控制工具

根据输入的关节角度（弧度）直接控制机械臂移动到目标位置。

使用示例:
    # 交互式输入关节角度
    python control_kinova_joints.py

    # 通过命令行参数指定关节角度
    python control_kinova_joints.py --joints "0.1, -0.2, 1.5, 0.0, -0.5, 0.8, 0.3"

    # 指定机器人 IP
    python control_kinova_joints.py --robot-ip 192.168.1.10 --joints "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0"
"""

import sys
import os
import math
import argparse
import logging

# 设置 protobuf 环境变量以兼容 kortex_api（必须在导入 kortex_api 之前）
if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" not in os.environ:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("ControlKinovaJoints")

# 导入 kortex_api
try:
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
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
        tuple: (base, router, transport, session_manager)
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
    
    return base, router, transport, session_manager


def move_to_joint_positions(base: BaseClient, joint_positions: np.ndarray):
    """
    移动到目标关节位置
    
    Args:
        base: Base 客户端
        joint_positions: (7,) 目标关节位置（弧度）
    """
    # 创建动作对象
    action = Base_pb2.Action()
    action.name = "joint_position_action"
    action.application_data = ""
    
    # 设置关节角度目标
    for i, pos in enumerate(joint_positions):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = i
        # 转换为角度并归一化到 [0, 360)
        joint_angle.value = math.degrees(pos) % 360
    
    # 执行动作（非阻塞）
    logger.info("发送关节位置命令...")
    base.ExecuteAction(action)
    logger.info("命令已发送（非阻塞，机器人正在移动）")


def parse_joint_string(joint_str: str) -> np.ndarray:
    """
    解析关节角度字符串（逗号分隔）
    
    Args:
        joint_str: 关节角度字符串，例如 "0.1, -0.2, 1.5, 0.0, -0.5, 0.8, 0.3"
    
    Returns:
        (7,) 关节角度数组（弧度）
    """
    try:
        values = [float(x.strip()) for x in joint_str.split(',')]
        if len(values) != 7:
            raise ValueError(f"期望 7 个关节角度，但收到 {len(values)} 个")
        return np.array(values)
    except Exception as e:
        raise ValueError(f"解析关节角度失败: {e}")


def get_current_joint_positions(base_cyclic):
    """
    获取当前关节位置
    
    Args:
        base_cyclic: BaseCyclic 客户端
    
    Returns:
        (7,) 关节角度数组（弧度）
    """
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    if not isinstance(base_cyclic, BaseCyclicClient):
        raise ValueError("需要 BaseCyclic 客户端来获取当前状态")
    
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


def interactive_mode(base: BaseClient, router: RouterClient):
    """
    交互式模式：循环输入关节角度并执行
    
    Args:
        base: Base 客户端
        router: RouterClient 对象（用于创建 BaseCyclic 客户端）
    """
    logger.info("\n进入交互式模式")
    logger.info("输入关节角度（7个值，逗号分隔，单位：弧度）")
    logger.info("输入 'q' 或 'quit' 退出")
    logger.info("输入 'current' 显示当前关节角度\n")
    
    # 导入 BaseCyclic 用于获取当前状态
    try:
        from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
        base_cyclic = BaseCyclicClient(router)
    except Exception as e:
        logger.warning(f"无法创建 BaseCyclic 客户端: {e}，将无法显示当前状态")
        base_cyclic = None
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                logger.info("退出交互式模式")
                break
            
            if user_input.lower() == 'current':
                if base_cyclic is not None:
                    try:
                        current_joints = get_current_joint_positions(base_cyclic)
                        logger.info(f"当前关节角度（弧度）: {format_joint_positions(current_joints)}")
                    except Exception as e:
                        logger.error(f"获取当前状态失败: {e}")
                else:
                    logger.warning("无法获取当前状态（BaseCyclic 客户端不可用）")
                continue
            
            # 解析并执行关节角度命令
            joint_positions = parse_joint_string(user_input)
            logger.info(f"目标关节角度（弧度）: {format_joint_positions(joint_positions)}")
            move_to_joint_positions(base, joint_positions)
            
        except KeyboardInterrupt:
            logger.info("\n用户中断")
            break
        except ValueError as e:
            logger.error(f"错误: {e}")
        except Exception as e:
            logger.error(f"执行失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="根据关节角度控制 Kinova Gen3 机械臂",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式模式（手动输入关节角度）
  python control_kinova_joints.py

  # 通过命令行指定关节角度（弧度）
  python control_kinova_joints.py --joints "0.1, -0.2, 1.5, 0.0, -0.5, 0.8, 0.3"

  # 指定机器人 IP
  python control_kinova_joints.py --robot-ip 192.168.1.11 --joints "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0"
        """
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
        '--joints',
        type=str,
        default=None,
        help='关节角度（7个值，逗号分隔，单位：弧度），例如: "0.1, -0.2, 1.5, 0.0, -0.5, 0.8, 0.3"。如果不指定，进入交互式模式'
    )
    
    args = parser.parse_args()
    
    # 连接机器人
    logger.info(f"连接机器人: {args.robot_ip}:{args.robot_port}")
    try:
        base, router, transport, session_manager = connect_to_robot(
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
        if args.joints is None:
            # 交互式模式
            interactive_mode(base, router)
        else:
            # 命令行模式：解析关节角度并执行
            try:
                joint_positions = parse_joint_string(args.joints)
                logger.info(f"目标关节角度（弧度）: {format_joint_positions(joint_positions)}")
                move_to_joint_positions(base, joint_positions)
                
            except ValueError as e:
                logger.error(f"参数错误: {e}")
                sys.exit(1)
        
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
            logger.info("已断开连接")
        except Exception as e:
            logger.warning(f"断开连接时出错: {e}")


if __name__ == "__main__":
    main()
