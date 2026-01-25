# ruff: noqa
"""
Kinova Gen3 机器人 OpenPI 策略推理脚本

使用 openpi 服务器进行 VLA 策略推理，控制 Kinova Gen3 机械臂执行任务。

使用方法：
    # 1. 在有 GPU 的机器上启动策略服务器（使用 pi05_kinova 配置）
    uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_kinova --policy.dir=gs://openpi-assets/checkpoints/pi05_base

    # 2. 在机器人控制电脑上运行此脚本
    python main.py --robot-ip 192.168.1.10 --remote-host <server_ip> --external-serial <cam1> --wrist-serial <cam2>

配置说明：
    - pi05_kinova: 使用 DROID 格式（7 DOF + 夹爪），输出绝对关节位置
    - 也可以使用 pi05_droid checkpoint: --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
"""

import dataclasses
import datetime
import faulthandler
import os
import time
from typing import Optional

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import tqdm
import tyro
from scipy.spatial.transform import Rotation

from execution_utils import (
    _monitor_waypoints_execution,
    _sleep_to_rate,
    _wait_until_reached,
    prevent_keyboard_interrupt,
)
from io_utils import _extract_observation, _save_video
from log_utils import _append_control_log, _save_control_log
from trajectory_utils import (
    _build_waypoint_trajectory,
    _clamp_joint_targets_rad,
    _interpolate_joint_targets,
)
from kinova_env import KinovaRobotEnv, ActionMode
from safety_monitor import SafetyMonitor
from trajectory_smoothing import TrajectorySmoother
from urdf_kinematics import URDFKinematics
from velocity_controller import CartesianVelocityController, create_stop_command, create_twist_command

faulthandler.enable()

# 动作执行频率默认值（每个策略动作的执行周期）
DEFAULT_CONTROL_FREQUENCY = 1  # Hz


@dataclasses.dataclass
class Args:
    # =========================================================================
    # 硬件配置
    # =========================================================================
    # Kinova 机械臂
    robot_ip: str = "192.168.1.10"
    
    # 夹爪（Arduino UDP）
    gripper_ip: str = "192.168.1.43"
    
    # RealSense D435i 相机序列号
    external_camera_serial: Optional[str] = None  # 外部相机（左侧视角）
    wrist_camera_serial: Optional[str] = None     # 腕部相机

    # =========================================================================
    # 策略服务器配置
    # =========================================================================
    remote_host: str = "0.0.0.0"  # 策略服务器 IP
    remote_port: int = 8000       # 策略服务器端口

    # =========================================================================
    # 推理配置
    # =========================================================================
    max_timesteps: int = 600  # 最大时间步数

    # 控制频率（每秒动作数）
    control_freq: int = DEFAULT_CONTROL_FREQUENCY
    
    # 从预测的 action chunk 中执行多少个动作后再查询服务器
    # 8 通常是个好默认值（约 0.5 秒的动作执行）
    open_loop_horizon: int = 8
    
    # 动作模式：
    # - absolute: 绝对位置模式（默认，与 pi05_kinova 配置兼容）
    # - delta: 增量模式，action 是相对当前位置的增量
    # - velocity: 速度模式
    action_mode: str = "absolute"

    # =========================================================================
    # 平滑控制配置
    # =========================================================================
    # 三种互斥控制模式：
    # - smooth: 速度控制（默认）
    # - no_smooth: 关节角度控制（单点）
    # - waypoints: 关节角度控制（轨迹）
    control_mode: str = "smooth"
    smoothing_window_size: int = 5
    max_linear_velocity: float = 0.05
    max_angular_velocity: float = 0.5
    position_gain: float = 2.0
    orientation_gain: float = 1.0

    # =========================================================================
    # 无平滑模式下的插值控制配置
    # =========================================================================
    # 在每两个动作点之间插入的中间点数量（0 表示不插值）
    inter: int = 0
    no_smooth_inner_loop: bool = False
    no_smooth_inner_loop_pos_tol: float = 0.001
    no_smooth_inner_loop_timeout_s: float = 2.0
    no_smooth_inner_loop_poll_dt: float = 0.02

    # =========================================================================
    # Waypoints 轨迹执行配置
    # =========================================================================
    waypoints_inner_loop: bool = False
    waypoints_inner_loop_dt: float = 0.1
    waypoints_inner_loop_pos_tol: float = 0.01
    waypoints_no_motion_threshold: float = 1e-4
    waypoints_no_motion_max_count: int = 5
    waypoints_step_to_step: bool = False
    waypoints_speed_scale: float = 0.8
    waypoints_min_joint_speed: float = 5.0
    waypoints_max_joint_speed: float = 25.0
    waypoints_max_joint_accel: float = 50.0

    # =========================================================================
    # 安全检测配置
    # =========================================================================
    safety: bool = True
    safety_mode: str = "soft"
    safety_urdf: Optional[str] = None
    safety_bbox: Optional[str] = None
    safety_joints: Optional[str] = None


def main(args: Args):
    # 检查相机序列号
    if args.external_camera_serial is None or args.wrist_camera_serial is None:
        print("警告: 未指定相机序列号，将尝试自动检测")
        print("建议使用 --external-serial 和 --wrist-serial 明确指定")

    if args.control_freq <= 0:
        raise ValueError("control_freq 必须为正数")
    if args.inter < 0:
        raise ValueError("inter 必须为非负整数")
    control_frequency = float(args.control_freq)

    # 验证控制模式
    control_mode = args.control_mode
    if control_mode not in ("smooth", "no_smooth", "waypoints"):
        raise ValueError(
            f"未知的控制模式: {control_mode}，必须是 smooth/no_smooth/waypoints 之一"
        )

    # 初始化机器人环境
    print(f"连接 Kinova 机械臂: {args.robot_ip}")
    print(f"连接夹爪控制器: {args.gripper_ip}")
    print(f"动作模式: {args.action_mode}")
    print(f"控制模式: {control_mode}")
    env = KinovaRobotEnv(
        robot_ip=args.robot_ip,
        gripper_ip=args.gripper_ip,
        external_camera_serial=args.external_camera_serial,
        wrist_camera_serial=args.wrist_camera_serial,
        action_mode=args.action_mode,
    )
    print("机器人环境创建成功!")

    joint_limits_deg = None
    if control_mode == "waypoints":
        joint_limits_deg = env.get_joint_position_limits_deg()
        if joint_limits_deg is None:
            print("[轨迹] 未获取到关节硬限，将不进行关节限位 clamp")

    # 初始化安全检测器（可选）
    safety_monitor = None
    if args.safety:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = args.safety_urdf if args.safety_urdf else os.path.join(script_dir, "GEN3_URDF_V12_with_dampint.urdf")
        bbox_path = args.safety_bbox if args.safety_bbox else os.path.join(script_dir, "boundingbox.txt")

        monitored_joints = None
        if args.safety_joints:
            try:
                monitored_joints = [int(x) for x in args.safety_joints.split()]
                if not all(1 <= j <= 8 for j in monitored_joints):
                    raise ValueError("安全关节编号必须在 1-8 范围内（1-7=关节，8=末端）")
            except ValueError as e:
                print(f"安全关节参数解析失败: {e}")
                return
        else:
            monitored_joints = list(range(2, 9))  # 默认监督关节 2-8

        safety_monitor = SafetyMonitor(
            urdf_path=urdf_path,
            boundingbox_path=bbox_path,
            mode=args.safety_mode,
            monitored_joints=monitored_joints,
        )

    # 初始化平滑控制器（可选）
    smoother = None
    velocity_controller = None
    kinematics = None
    if control_mode == "smooth":
        smoother = TrajectorySmoother(window_size=args.smoothing_window_size)
        velocity_controller = CartesianVelocityController(
            max_linear_velocity=args.max_linear_velocity,
            max_angular_velocity=args.max_angular_velocity,
            position_gain=args.position_gain,
            orientation_gain=args.orientation_gain,
        )
        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = args.safety_urdf if args.safety_urdf else os.path.join(script_dir, "GEN3_URDF_V12_with_dampint.urdf")
        kinematics = URDFKinematics(urdf_path)

    # 连接策略服务器
    print(f"连接策略服务器: {args.remote_host}:{args.remote_port}")
    policy_client = websocket_client_policy.WebsocketClientPolicy(
        args.remote_host, args.remote_port
    )
    print("策略服务器连接成功!")

    # 用于保存视频的列表
    last_commanded_joint_pos = None
    try:
        while True:
            instruction = input("\n输入任务指令 (或 'q' 退出): ")
            if instruction.lower() == 'q':
                break

            # Rollout 参数
            actions_from_chunk_completed = 0
            pred_action_chunk = None
            chunk_index = -1
            control_log_entries = []

            # 准备保存视频
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
            video_frames = []

            bar = tqdm.tqdm(range(args.max_timesteps), desc="执行中")
            print("开始执行... 按 Ctrl+C 或 ESC/q 键停止")

            estop_triggered = False
            for t_step in bar:
                start_time = time.time()
                try:
                    # 获取当前观察
                    curr_obs = env.get_observation()

                    # 提取观察数据
                    obs_data = _extract_observation(
                        curr_obs,
                        external_camera_serial=args.external_camera_serial,
                        wrist_camera_serial=args.wrist_camera_serial,
                        save_to_disk=(t_step == 0),
                    )

                    # 保存外部相机图像用于视频
                    video_frames.append(obs_data["external_image"])
                    print(f"step={actions_from_chunk_completed}")
                    # 如果需要，查询策略服务器获取新的动作 chunk
                    if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:

                        actions_from_chunk_completed = 0
                        chunk_index += 1


                        # 准备请求数据（图像需要 resize 到 224x224）
                        base_image = image_tools.resize_with_pad(
                            obs_data["external_image"], 224, 224
                        )
                        wrist_image = image_tools.resize_with_pad(
                            obs_data["wrist_image"], 224, 224
                        )
                        # 同时兼容 DROID 与 Kinova policy 的输入键名
                        request_data = {
                            # DROID 风格输入
                            "observation/exterior_image_1_left": base_image,
                            "observation/wrist_image_left": wrist_image,
                            "observation/joint_position": obs_data["joint_position"],
                            "observation/gripper_position": obs_data["gripper_position"],
                            # Kinova 自采集风格输入
                            "observation/image": base_image,
                            "observation/wrist_image": wrist_image,
                            "observation/state": np.concatenate(
                                [obs_data["joint_position"], obs_data["gripper_position"]]
                            ),
                            "prompt": instruction,
                        }

                        # 使用上下文管理器防止 Ctrl+C 中断服务器调用
                        with prevent_keyboard_interrupt():
                            pred_action_chunk = policy_client.infer(request_data)["actions"]
                        
                        # 验证动作格式 [chunk_size, action_dim]
                        assert pred_action_chunk.shape[1] == 8, \
                            f"期望动作维度为 8，但收到 {pred_action_chunk.shape[1]}"

                    # 从 chunk 中选择当前要执行的动作
                    action_index_in_chunk = actions_from_chunk_completed
                    action = pred_action_chunk[action_index_in_chunk]
                    actions_from_chunk_completed += 1
                    # 二值化夹爪动作
                    if action[-1] > 0.5:
                        action = np.concatenate([action[:-1], np.ones((1,))])
                    else:
                        action = np.concatenate([action[:-1], np.zeros((1,))])

                    # 打印执行前的 action
                    joint_angles_str = ", ".join([f"{x:.6f}" for x in action[:7]])
                    gripper_str = f"{action[-1]:.6f}"
                    if t_step % 8 == 0 or t_step % 8 == 7:
                        print(f"\n[t={t_step}] 执行 Action:")
                        print(f"  关节角度: [{joint_angles_str}]")
                        print(f"  夹爪位置: {gripper_str}")

                    # 执行动作（平滑/速度控制 或 直接位置控制）
                    skip_rate_sleep = False
                    waypoints_total_duration = 0.0
                    target_joint_pos = action[:7]
                    target_gripper_pos = action[-1]

                    if control_mode == "smooth":
                        # 安全检测（若启用）
                        if safety_monitor is not None:
                            try:
                                if not safety_monitor.check_and_handle(env, target_joint_pos):
                                    print("[安全] 软急停: 速度置零，等待下一条指令")
                                    if env._base is not None:
                                        stop_cmd = create_stop_command()
                                        if stop_cmd is not None:
                                            env._base.SendTwistCommand(stop_cmd)
                                    elapsed_time = time.time() - start_time
                                    if elapsed_time < 1 / control_frequency:
                                        time.sleep(1 / control_frequency - elapsed_time)
                                    continue
                            except RuntimeError as exc:
                                print(f"[安全] 硬急停: {exc}")
                                estop_triggered = True
                                break

                        # 计算目标末端位置（由关节角度通过 URDF 正运动学推导）
                        joint_xyz, target_eef_pos, target_eef_rot = (
                            kinematics.compute_joint_positions_and_pose(target_joint_pos)
                        )

                        # 当前末端位姿（位置 + 四元数）
                        cart_pos = curr_obs["robot_state"]["cartesian_position"]  # [x,y,z,rx,ry,rz] (rad)
                        current_quat = Rotation.from_euler("xyz", cart_pos[3:], degrees=False).as_quat()
                        current_pose = np.concatenate([cart_pos[:3], current_quat])

                        # 目标末端位姿：位置与姿态都来自策略输出（经 URDF 正运动学推导）
                        target_quat = Rotation.from_matrix(target_eef_rot).as_quat()
                        target_pose = np.concatenate([target_eef_pos, target_quat])

                        # 平滑轨迹
                        smoother.add_pose(target_pose)
                        smoothed_pose = smoother.get_smoothed_pose()
                        if smoothed_pose is None:
                            smoothed_pose = target_pose

                        # 计算速度命令并发送
                        linear_vel, angular_vel = velocity_controller.compute_velocity(
                            current_pose, smoothed_pose
                        )
                        twist_cmd = create_twist_command(
                            linear_vel, angular_vel, duration_ms=int(1000 / control_frequency)
                        )
                        if twist_cmd is not None:
                            env._base.SendTwistCommand(twist_cmd)

                        # 控制夹爪（使用关节位置控制）
                        if abs(target_gripper_pos - env._current_gripper_pos) > 0.1:
                            env._control_gripper(target_gripper_pos)

                        _append_control_log(
                            control_log_entries,
                            t_step=t_step,
                            chunk_index=chunk_index,
                            action_index=action_index_in_chunk,
                            interp_step=1,
                            interp_steps=1,
                            state={
                                "joint_position": obs_data["joint_position"].tolist(),
                                "gripper_position": float(obs_data["gripper_position"][0]),
                                "cartesian_position": curr_obs["robot_state"]["cartesian_position"].tolist(),
                            },
                            action={
                                "joint_position": target_joint_pos.tolist(),
                                "gripper_position": float(target_gripper_pos),
                                "linear_velocity": linear_vel.tolist(),
                                "angular_velocity": angular_vel.tolist(),
                            },
                            mode="smooth",
                            timestamp=time.time(),
                        )
                    elif control_mode == "no_smooth":
                        # 直接关节位置控制（可选插值）
                        if args.inter <= 0 or action_index_in_chunk >= len(pred_action_chunk) - 1:
                            if safety_monitor is not None:
                                try:
                                    if not safety_monitor.check_and_handle(env, target_joint_pos):
                                        print(
                                            f"[安全] 软急停: 跳过不安全动作: "
                                            f"t={t_step}, action_index={action_index_in_chunk}"
                                        )
                                        elapsed_time = time.time() - start_time
                                        if elapsed_time < 1 / control_frequency:
                                            time.sleep(1 / control_frequency - elapsed_time)
                                        continue
                                except RuntimeError as exc:
                                    print(f"[安全] 硬急停: {exc}")
                                    estop_triggered = True
                                    break
                            env.step(action)
                            if args.no_smooth_inner_loop:
                                _wait_until_reached(
                                    env,
                                    target_joint_pos,
                                    pos_tol=args.no_smooth_inner_loop_pos_tol,
                                    timeout_s=args.no_smooth_inner_loop_timeout_s,
                                    poll_dt=args.no_smooth_inner_loop_poll_dt,
                                )
                            _append_control_log(
                                control_log_entries,
                                t_step=t_step,
                                chunk_index=chunk_index,
                                action_index=action_index_in_chunk,
                                interp_step=1,
                                interp_steps=1,
                                state={
                                    "joint_position": obs_data["joint_position"].tolist(),
                                    "gripper_position": float(obs_data["gripper_position"][0]),
                                    "cartesian_position": curr_obs["robot_state"]["cartesian_position"].tolist(),
                                },
                                action={
                                    "joint_position": target_joint_pos.tolist(),
                                    "gripper_position": float(target_gripper_pos),
                                },
                                mode="direct",
                                timestamp=time.time(),
                            )
                        else:
                            next_action = pred_action_chunk[action_index_in_chunk + 1]
                            if next_action[-1] > 0.5:
                                next_action = np.concatenate([next_action[:-1], np.ones((1,))])
                            else:
                                next_action = np.concatenate([next_action[:-1], np.zeros((1,))])
                            next_joint_pos = next_action[:7]

                            interp_steps = args.inter + 1
                            actual_interp_hz = control_frequency * interp_steps
                            interpolated_joints = _interpolate_joint_targets(
                                target_joint_pos, next_joint_pos, args.inter
                            )

                            sub_joint_targets = [target_joint_pos] + list(interpolated_joints)
                            for interp_step_index, joint_target in enumerate(
                                sub_joint_targets, start=1
                            ):
                                sub_step_start = time.time()
                                if safety_monitor is not None:
                                    try:
                                        if not safety_monitor.check_and_handle(env, joint_target):
                                            print(
                                                "[安全] 软急停: 跳过不安全插值动作: "
                                                f"t={t_step}, action_index={action_index_in_chunk}, "
                                                f"interp_step={interp_step_index}"
                                            )
                                            _sleep_to_rate(sub_step_start, actual_interp_hz)
                                            continue
                                    except RuntimeError as exc:
                                        print(f"[安全] 硬急停: {exc}")
                                        estop_triggered = True
                                        break
                                sub_action = np.concatenate(
                                    [joint_target, np.array([target_gripper_pos])]
                                )
                                env.step(sub_action)
                                _append_control_log(
                                    control_log_entries,
                                    t_step=t_step,
                                    chunk_index=chunk_index,
                                    action_index=action_index_in_chunk,
                                    interp_step=interp_step_index,
                                    interp_steps=interp_steps,
                                    state={
                                        "joint_position": obs_data["joint_position"].tolist(),
                                        "gripper_position": float(obs_data["gripper_position"][0]),
                                        "cartesian_position": curr_obs["robot_state"]["cartesian_position"].tolist(),
                                    },
                                    action={
                                        "joint_position": joint_target.tolist(),
                                        "gripper_position": float(target_gripper_pos),
                                    },
                                    mode="direct_interp",
                                    timestamp=time.time(),
                                )
                                if args.no_smooth_inner_loop:
                                    _wait_until_reached(
                                        env,
                                        joint_target,
                                        pos_tol=args.no_smooth_inner_loop_pos_tol,
                                        timeout_s=args.no_smooth_inner_loop_timeout_s,
                                        poll_dt=args.no_smooth_inner_loop_poll_dt,
                                    )
                                else:
                                    _sleep_to_rate(sub_step_start, actual_interp_hz)

                            if estop_triggered:
                                break

                        last_commanded_joint_pos = target_joint_pos
                    else:
                        # waypoints: use trajectory execution
                        chunk_actions = np.array(pred_action_chunk, copy=True)
                        # Binarize gripper for the whole chunk
                        chunk_actions[:, -1] = np.where(
                            chunk_actions[:, -1] > 0.5, 1.0, 0.0
                        )
                        if args.action_mode == ActionMode.DELTA:
                            # Convert delta actions to absolute joint targets
                            base_joint_pos = obs_data["joint_position"]
                            abs_joints = np.zeros_like(chunk_actions[:, :7])
                            running = np.array(base_joint_pos, dtype=float)
                            for idx in range(len(chunk_actions)):
                                running = running + chunk_actions[idx][:7]
                                abs_joints[idx] = running
                            chunk_actions[:, :7] = abs_joints
                        elif args.action_mode == ActionMode.VELOCITY:
                            raise ValueError("waypoints 模式不支持 velocity 动作模式")

                        speed_limit = args.waypoints_max_joint_speed
                        accel_limit = args.waypoints_max_joint_accel if args.waypoints_inner_loop else None
                        if args.waypoints_step_to_step:
                            total_duration = 0.0
                            segment_start_pos = np.array(
                                obs_data["joint_position"], dtype=float
                            )
                            for seg_idx, seg_action in enumerate(chunk_actions):
                                seg_chunk = np.asarray([seg_action], dtype=float)
                                joint_targets, durations, gripper_targets = _build_waypoint_trajectory(
                                    seg_chunk,
                                    control_frequency=control_frequency,
                                    inter=args.inter,
                                    max_joint_speed_deg_s=speed_limit,
                                    current_joint_pos=segment_start_pos,
                                    max_joint_accel_deg_s2=accel_limit,
                                )
                                if joint_limits_deg is not None and joint_targets.size > 0:
                                    clamped_targets = _clamp_joint_targets_rad(
                                        joint_targets, joint_limits_deg
                                    )
                                    if not np.allclose(clamped_targets, joint_targets):
                                        print("[轨迹] 关节目标超出硬限，已自动 clamp")
                                    joint_targets = clamped_targets
                                if joint_targets.size == 0:
                                    print("[安全] 轨迹为空，跳过执行")
                                    break
                                if safety_monitor is not None:
                                    last_safe_idx = -1
                                    try:
                                        for idx, joint_target in enumerate(joint_targets):
                                            if not safety_monitor.check_and_handle(env, joint_target):
                                                print(
                                                    "[安全] 软急停: 轨迹超界，截断到最后安全点 "
                                                    f"index={last_safe_idx}"
                                                )
                                                break
                                            last_safe_idx = idx
                                    except RuntimeError as exc:
                                        print(f"[安全] 硬急停: {exc}")
                                        estop_triggered = True

                                    if not estop_triggered:
                                        if last_safe_idx < 0:
                                            print("[安全] 软急停: 轨迹全部超界，停止执行")
                                            break
                                        if last_safe_idx < len(joint_targets) - 1:
                                            joint_targets = joint_targets[: last_safe_idx + 1]
                                            durations = durations[: last_safe_idx + 1]
                                            gripper_targets = gripper_targets[: last_safe_idx + 1]

                                if estop_triggered:
                                    break

                                print(
                                    f"[轨迹] step={seg_idx + 1}/{len(chunk_actions)}, "
                                    f"waypoints={len(joint_targets)}, "
                                    f"总时长≈{durations.sum():.2f}s, "
                                    f"speed_limit={speed_limit:.2f} deg/s"
                                )
                                env._execute_joint_waypoints(
                                    joint_targets,
                                    durations,
                                    blending_radius=0.0,
                                )
                                total_duration += float(np.sum(durations))
                                waypoints_total_duration = total_duration

                                target_gripper_pos = float(gripper_targets[-1])
                                if abs(target_gripper_pos - env._current_gripper_pos) > 0.1:
                                    env._control_gripper(target_gripper_pos)

                                if args.waypoints_inner_loop:
                                    _monitor_waypoints_execution(
                                        env,
                                        joint_targets[-1],
                                        total_duration=float(np.sum(durations)),
                                        poll_dt=args.waypoints_inner_loop_dt,
                                        pos_tol=args.waypoints_inner_loop_pos_tol,
                                        no_motion_threshold=args.waypoints_no_motion_threshold,
                                        no_motion_max_count=args.waypoints_no_motion_max_count,
                                    )

                                for idx, joint_target in enumerate(joint_targets):
                                    _append_control_log(
                                        control_log_entries,
                                        t_step=t_step,
                                        chunk_index=chunk_index,
                                        action_index=seg_idx,
                                        interp_step=idx + 1,
                                        interp_steps=len(joint_targets),
                                        state={
                                            "joint_position": obs_data["joint_position"].tolist(),
                                            "gripper_position": float(obs_data["gripper_position"][0]),
                                            "cartesian_position": curr_obs["robot_state"]["cartesian_position"].tolist(),
                                        },
                                        action={
                                            "joint_position": joint_target.tolist(),
                                            "gripper_position": float(gripper_targets[idx]),
                                        },
                                        mode="waypoints_step",
                                        timestamp=time.time(),
                                    )

                                segment_start_pos = joint_targets[-1]

                            if not estop_triggered:
                                skip_rate_sleep = True
                                actions_from_chunk_completed = args.open_loop_horizon
                        else:
                            joint_targets, durations, gripper_targets = _build_waypoint_trajectory(
                                chunk_actions,
                                control_frequency=control_frequency,
                                inter=args.inter,
                                max_joint_speed_deg_s=speed_limit,
                                current_joint_pos=obs_data["joint_position"],
                                max_joint_accel_deg_s2=accel_limit,
                            )
                            if joint_limits_deg is not None and joint_targets.size > 0:
                                clamped_targets = _clamp_joint_targets_rad(
                                    joint_targets, joint_limits_deg
                                )
                                if not np.allclose(clamped_targets, joint_targets):
                                    print("[轨迹] 关节目标超出硬限，已自动 clamp")
                                joint_targets = clamped_targets
                            if joint_targets.size == 0:
                                print("[安全] 轨迹为空，跳过执行")
                            else:
                                if safety_monitor is not None:
                                    last_safe_idx = -1
                                    try:
                                        for idx, joint_target in enumerate(joint_targets):
                                            if not safety_monitor.check_and_handle(env, joint_target):
                                                print(
                                                    "[安全] 软急停: 轨迹超界，截断到最后安全点 "
                                                    f"index={last_safe_idx}"
                                                )
                                                break
                                            last_safe_idx = idx
                                    except RuntimeError as exc:
                                        print(f"[安全] 硬急停: {exc}")
                                        estop_triggered = True

                                    if not estop_triggered:
                                        if last_safe_idx < 0:
                                            print("[安全] 软急停: 轨迹全部超界，跳过执行")
                                        else:
                                            if last_safe_idx < len(joint_targets) - 1:
                                                joint_targets = joint_targets[: last_safe_idx + 1]
                                                durations = durations[: last_safe_idx + 1]
                                                gripper_targets = gripper_targets[: last_safe_idx + 1]

                                if not estop_triggered:
                                    print(
                                        f"[轨迹] waypoints={len(joint_targets)}, "
                                        f"总时长≈{durations.sum():.2f}s, "
                                        f"speed_limit={speed_limit:.2f} deg/s"
                                    )

                                    env._execute_joint_waypoints(
                                        joint_targets,
                                        durations,
                                        blending_radius=0.0,
                                    )
                                    waypoints_total_duration = float(np.sum(durations))
                                    target_gripper_pos = float(gripper_targets[-1])
                                    if abs(target_gripper_pos - env._current_gripper_pos) > 0.1:
                                        env._control_gripper(target_gripper_pos)

                                    if args.waypoints_inner_loop:
                                        _monitor_waypoints_execution(
                                            env,
                                            joint_targets[-1],
                                            total_duration=float(np.sum(durations)),
                                            poll_dt=args.waypoints_inner_loop_dt,
                                            pos_tol=args.waypoints_inner_loop_pos_tol,
                                            no_motion_threshold=args.waypoints_no_motion_threshold,
                                            no_motion_max_count=args.waypoints_no_motion_max_count,
                                        )

                            for idx, joint_target in enumerate(joint_targets):
                                _append_control_log(
                                    control_log_entries,
                                    t_step=t_step,
                                    chunk_index=chunk_index,
                                    action_index=idx,
                                    interp_step=idx + 1,
                                    interp_steps=len(joint_targets),
                                    state={
                                        "joint_position": obs_data["joint_position"].tolist(),
                                        "gripper_position": float(obs_data["gripper_position"][0]),
                                        "cartesian_position": curr_obs["robot_state"]["cartesian_position"].tolist(),
                                    },
                                    action={
                                        "joint_position": joint_target.tolist(),
                                        "gripper_position": float(gripper_targets[idx]),
                                    },
                                    mode="waypoints",
                                    timestamp=time.time(),
                                )

                            skip_rate_sleep = True
                            actions_from_chunk_completed = args.open_loop_horizon

                    # 获取执行后的关节角度
                    post_obs = env.get_observation()
                    post_joint_positions = np.array(post_obs["robot_state"]["joint_positions"])
                    post_gripper_position = post_obs["robot_state"]["gripper_position"]
                    
                    # 打印执行后的关节角度
                    post_joint_angles_str = ", ".join([f"{x:.6f}" for x in post_joint_positions])
                    #if t_step % 8 == 0 or t_step % 8 == 7:
                    print(f"  执行后关节角度: [{post_joint_angles_str}]")
                    print(f"  执行后夹爪位置: {post_gripper_position:.6f}")

                    # 等待以匹配控制频率
                    elapsed_time = time.time() - start_time
                    if not skip_rate_sleep:
                        if elapsed_time < 1 / control_frequency:
                            time.sleep(1 / control_frequency - elapsed_time)
                    else:
                        remaining = waypoints_total_duration - (time.time() - start_time)
                        if remaining > 0:
                            time.sleep(remaining)

                except KeyboardInterrupt:
                    print("\n用户中断")
                    break

                if estop_triggered:
                    break

            if estop_triggered:
                print("[安全] 已触发硬急停，退出当前任务")
                break

            # 保存视频
            if video_frames:
                _save_video(video_frames, timestamp)

            # 保存控制日志
            _save_control_log(control_log_entries, timestamp)

            # 询问结果
            success = input("任务是否成功? (y/n): ")
            print(f"结果已记录: {'成功' if success.lower() == 'y' else '失败'}")

            # 询问是否继续
            if input("继续下一个任务? (y/n): ").lower() != 'y':
                break

            # 重置机器人
            print("重置机器人...")
            env.reset()

            if estop_triggered:
                break

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("程序结束")


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)

