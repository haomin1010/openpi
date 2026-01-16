#!/usr/bin/env python3
"""
Kinova Gen3 状态查询工具

打印当前机械臂的关节角度和末端位姿信息。

使用示例:
    python get_kinova_state.py
    python get_kinova_state.py --robot-ip 192.168.1.10
    python get_kinova_state.py --detail #额外打印一个[7,3]的矩阵，代表每个关节的坐标
"""

import sys
import os

# 设置 protobuf 环境变量以兼容 kortex_api（必须在导入 kortex_api 之前）
if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" not in os.environ:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import argparse
import logging
import math
from pathlib import Path

import numpy as np

# 设置日志（在导入 kortex_api 之前，以便错误处理可以使用 logger）
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("GetKinovaState")

# 导入 kortex_api
try:
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.RouterClient import RouterClient
    from kortex_api.SessionManager import SessionManager
    from kortex_api.TCPTransport import TCPTransport
    from kortex_api.autogen.messages import Session_pb2
except ImportError:
    logger.error("kortex_api 未安装。请从 Kinova 官网下载并安装 kortex_api wheel 包。")
    sys.exit(1)

# 添加父目录到路径，以便导入 urdf_kinematics（仅在需要时导入）
# 注意：这里不立即导入，而是在需要时（--detail 参数）才导入
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))


def format_joint_positions(joint_positions):
    """
    格式化关节角度数组为字符串
    
    Args:
        joint_positions: (7,) 关节角度数组（弧度）
    
    Returns:
        str: 格式化的字符串，例如 "[0.1, -0.2, 1.5, 0.0, -0.5, 0.8, 0.3]"
    """
    return "[" + ", ".join(f"{angle:.6f}" for angle in joint_positions) + "]"


def format_cartesian_position(cartesian_position):
    """
    格式化末端位姿数组为字符串（格式与 boundingbox.txt 一致）
    
    Args:
        cartesian_position: (6,) 末端位姿数组 [x, y, z, theta_x, theta_y, theta_z]（弧度）
    
    Returns:
        str: 格式化的字符串，例如 "[0.523453, 0.122365, 0.221617, 102.7658, -2.0315, 92.3576]"
             其中位置为米，角度为度
    """
    x, y, z = cartesian_position[:3]
    # 将弧度转换为度
    rx_deg = math.degrees(cartesian_position[3])
    ry_deg = math.degrees(cartesian_position[4])
    rz_deg = math.degrees(cartesian_position[5])
    return f"[{x:.6f}, {y:.6f}, {z:.6f}, {rx_deg:.4f}, {ry_deg:.4f}, {rz_deg:.4f}]"


def format_joint_matrix(joint_positions):
    """
    格式化关节坐标矩阵为字符串

    Args:
        joint_positions: (7, 3) 关节坐标矩阵

    Returns:
        str: 多行字符串，每行表示一个关节坐标
    """
    lines = []
    for row in joint_positions:
        lines.append("[" + ", ".join(f"{val:.6f}" for val in row) + "]")
    return "[\n  " + ",\n  ".join(lines) + "\n]"


def connect_to_robot(robot_ip: str, robot_port: int = 10000, username: str = "admin", password: str = "admin"):
    """
    连接到 Kinova 机械臂（不初始化相机）
    
    Returns:
        tuple: (base_cyclic, transport, session_manager)
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
    
    # 创建 BaseCyclic 客户端（用于获取反馈）
    base_cyclic = BaseCyclicClient(router)
    
    return base_cyclic, transport, session_manager


def get_robot_state(base_cyclic):
    """
    获取机器人当前状态（不依赖相机）
    
    Returns:
        tuple: (joint_positions, cartesian_position)
            - joint_positions: (7,) 关节角度数组（弧度）
            - cartesian_position: (6,) 末端位姿 [x, y, z, rx, ry, rz]（弧度）
    """
    feedback = base_cyclic.RefreshFeedback()
    
    # 提取关节位置（转换为弧度）
    joint_positions = [
        math.radians(actuator.position) 
        for actuator in feedback.actuators
    ]
    
    # 提取末端执行器位姿
    cartesian_position = [
        feedback.base.tool_pose_x,
        feedback.base.tool_pose_y,
        feedback.base.tool_pose_z,
        math.radians(feedback.base.tool_pose_theta_x),
        math.radians(feedback.base.tool_pose_theta_y),
        math.radians(feedback.base.tool_pose_theta_z),
    ]
    
    return np.array(joint_positions), np.array(cartesian_position)


def main():
    """主函数：连接机器人并打印当前状态"""
    parser = argparse.ArgumentParser(
        description="获取 Kinova Gen3 机械臂当前状态",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认 IP 地址
  python get_kinova_state.py
  
  # 指定机器人 IP
  python get_kinova_state.py --robot-ip 192.168.1.10
  
  # 输出关节位置矩阵（基于 URDF 正运动学）
  python get_kinova_state.py --detail
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
        '--detail',
        action='store_true',
        help='打印详细信息（关节位置矩阵）'
    )
    parser.add_argument(
        '--urdf-path',
        type=str,
        default=None,
        help='URDF 文件路径（默认使用 kinova_gen3 目录下的 GEN3_URDF_V12_with_dampint.urdf）'
    )
    
    args = parser.parse_args()
    
    # 连接机器人
    logger.info(f"连接机器人: {args.robot_ip}:{args.robot_port}")
    try:
        base_cyclic, transport, session_manager = connect_to_robot(
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
        # 获取当前状态
        joint_positions, cartesian_position = get_robot_state(base_cyclic)
        
        # 格式化并打印
        joint_str = format_joint_positions(joint_positions)
        cartesian_str = format_cartesian_position(cartesian_position)
        
        print("关节角度（弧度）:")
        print(joint_str)
        print("\n末端位姿 [x(m), y(m), z(m), rx(deg), ry(deg), rz(deg)]:")
        print(cartesian_str)
        print()

        if args.detail:
            # 仅在需要时导入 urdf_kinematics
            try:
                from urdf_kinematics import URDFKinematics
            except ImportError:
                logger.error("无法导入 urdf_kinematics，请确认 examples/kinova_gen3/urdf_kinematics.py 存在")
                sys.exit(1)
            
            script_dir = Path(__file__).parent.parent
            urdf_path = Path(args.urdf_path) if args.urdf_path else script_dir / "GEN3_URDF_V12_with_dampint.urdf"
            if not urdf_path.exists():
                logger.error(f"URDF 文件不存在: {urdf_path}")
                sys.exit(1)
            
            kinematics = URDFKinematics(urdf_path)
            joint_xyz, _ = kinematics.compute_joint_positions(joint_positions)
            print("关节坐标矩阵 [7, 3] (m):")
            print(format_joint_matrix(joint_xyz))
            print()
        
    except Exception as e:
        logger.error(f"获取状态失败: {e}")
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
