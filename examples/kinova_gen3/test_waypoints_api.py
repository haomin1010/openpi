import os
import math
import time

# 必须在导入 kortex_api 之前设置
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, Session_pb2



ROBOT_IP = "192.168.1.10"
USERNAME = "admin"
PASSWORD = "admin"

def main():
    # 连接
    transport = TCPTransport()
    transport.connect(ROBOT_IP, 10000)
    router = RouterClient(transport, lambda e: print("Router error:", e))

    session_manager = SessionManager(router)
    session_info = Session_pb2.CreateSessionInfo()
    session_info.username = USERNAME
    session_info.password = PASSWORD
    session_info.session_inactivity_timeout = 60000
    session_info.connection_inactivity_timeout = 2000
    session_manager.CreateSession(session_info)

    base = BaseClient(router)
    base_cyclic = BaseCyclicClient(router)

    # Clear faults (if any)
    try:
        base.ClearFaults()
        print("✅ ClearFaults sent")
    except Exception as exc:
        print(f"⚠️ ClearFaults failed: {exc}")

    # 确保 servoing mode
    servo_mode = Base_pb2.ServoingModeInformation()
    servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(servo_mode)
    time.sleep(0.2)
    try:
        current_mode = base.GetServoingMode()
        print(f"✅ Servoing mode: {current_mode.servoing_mode}")
    except Exception as exc:
        print(f"⚠️ GetServoingMode failed: {exc}")

    # 读取当前关节位置（角度）
    feedback = base_cyclic.RefreshFeedback()
    current_deg = [a.position for a in feedback.actuators]  # 0-360 deg
    print("Current joints (deg):", ["{:.2f}".format(v) for v in current_deg])

    # 设计一段小轨迹：多关节轻微摆动（小幅度）
    # 关节索引: 0-6，角度单位为度
    offsets = [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 5.0, -5.0, 8.0, 12.0),
        (0.0, 0.0, 0.0, 8.0, -8.0, 12.0, 18.0),
        (0.0, 0.0, 0.0, 5.0, -5.0, 8.0, 12.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    waypoints_deg = []
    for off in offsets:
        wp = current_deg[:]
        for j in range(7):
            wp[j] = (wp[j] + off[j]) % 360.0
        waypoints_deg.append(wp)
    print("Waypoints (deg):")
    for idx, wp in enumerate(waypoints_deg):
        print(f"  wp{idx}: " + ", ".join([f"{v:.2f}" for v in wp]))

    # 构造 waypoints
    action = Base_pb2.Action()
    action.name = "demo_waypoint"

    max_vel = 20.0  # deg/s
    min_dt = 0.2
    durations = [min_dt]
    for prev, nxt in zip(waypoints_deg[:-1], waypoints_deg[1:]):
        deltas = []
        for c, t in zip(prev, nxt):
            delta = (t - c + 540.0) % 360.0 - 180.0  # shortest signed delta
            deltas.append(abs(delta))
        segment_time = max(deltas) / max_vel
        durations.append(max(segment_time, min_dt))

    action.execute_waypoint_list.duration = float(sum(durations))
    for idx, (wp_deg, dt) in enumerate(zip(waypoints_deg, durations)):
        wp = action.execute_waypoint_list.waypoints.add()
        wp.name = f"wp{idx}"
        wp.angular_waypoint.angles.extend(wp_deg)
        wp.angular_waypoint.duration = float(dt)
        wp.angular_waypoint.maximum_velocities.extend([max_vel] * len(wp_deg))

    # Validate waypoint list before execution and auto-adjust duration if needed
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            result = base.ValidateWaypointList(action.execute_waypoint_list)
        except Exception as exc:
            print(f"❌ ValidateWaypointList failed: {exc}")
            break

        print(f"✅ ValidateWaypointList (attempt {attempt}):", result)
        errors = getattr(result, "trajectory_error_report", None)
        if errors is None or not errors.trajectory_error_elements:
            break

        scale = 1.0
        for elem in errors.trajectory_error_elements:
            if elem.error_type in (
                Base_pb2.TRAJECTORY_ERROR_TYPE_INVALID_JOINT_SPEED,
                Base_pb2.TRAJECTORY_ERROR_TYPE_INVALID_JOINT_ACCELERATION,
            ):
                if elem.max_value > 0:
                    scale = max(scale, (elem.error_value / elem.max_value) * 1.1)
        if scale <= 1.0:
            break

        durations = [d * scale for d in durations]
        action.execute_waypoint_list.duration = float(sum(durations))
        for idx, wp in enumerate(action.execute_waypoint_list.waypoints):
            wp.angular_waypoint.duration = float(durations[idx])
        print(
            f"⚠️  Adjust duration scale={scale:.2f}, total_time={sum(durations):.2f}s"
        )

    # 执行
    print("ExecuteAction: moving joint7 small trajectory")
    try:
        base.ExecuteAction(action)
    except Exception as exc:
        print(f"❌ ExecuteAction failed: {exc}")
        session_manager.CloseSession()
        transport.disconnect()
        return

    time.sleep(4)
    feedback_after = base_cyclic.RefreshFeedback()
    after_deg = [a.position for a in feedback_after.actuators]
    print("After joints (deg):", ["{:.2f}".format(v) for v in after_deg])

    # 清理
    session_manager.CloseSession()
    transport.disconnect()

if __name__ == "__main__":
    main()