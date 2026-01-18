#!/usr/bin/env python3
"""
Kinova Gen3 轨迹回放脚本（基于 libero_format 数据）

功能：
1) 自动读取 data 目录中最新的 libero_format/*.npz
2) 回放前回到初始位置（states[0]）
3) 按 action chunk 执行（类似 pi05 部署逻辑）
4) 关节角度控制 + 插值提升控制频率，实现更平滑轨迹

使用示例：
    # 回放最新的 libero_format 轨迹
    python replay_routine.py
    
    # 指定数据根目录
    python replay_routine.py --data-dir /path/to/data
    
    # 指定 action chunk 和插值频率
    python replay_routine.py --action-horizon 8 --interpolated-control-frequency-hz 120
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import json

# 设置 protobuf 环境变量以兼容 kortex_api（必须在导入 kortex_api 之前）
if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" not in os.environ:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# 导入本地模块
from kinova_env import KinovaRobotEnv, ActionMode

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("ReplayRoutine")


def _sleep_to_rate(start_time: float, hz: float) -> None:
    """Sleep to maintain a target loop rate."""
    if hz <= 0:
        return
    elapsed = time.time() - start_time
    period = 1.0 / hz
    if elapsed < period:
        time.sleep(period - elapsed)


def _interpolate_joint_targets(
    start_joint_pos: np.ndarray, end_joint_pos: np.ndarray, steps: int
) -> np.ndarray:
    """Linearly interpolate joint targets between two poses.

    Returns an array with shape (steps, 7). When steps == 1, it returns [end].
    """
    steps = max(1, int(steps))
    return np.linspace(start_joint_pos, end_joint_pos, steps + 1, endpoint=True)[1:]


def _wait_until_reached(
    env: KinovaRobotEnv,
    target_joint_pos: np.ndarray,
    *,
    pos_tol: float,
    timeout_s: float,
    poll_dt: float = 0.01,
) -> bool:
    """Wait until joint positions are within tolerance or timeout."""
    start_time = time.time()
    while True:
        obs = env.get_observation()
        actual_joint_pos = obs["robot_state"]["joint_positions"]
        err = np.linalg.norm(target_joint_pos - actual_joint_pos)
        if err <= pos_tol:
            return True
        if time.time() - start_time >= timeout_s:
            return False
        time.sleep(poll_dt)


def _append_replay_log(
    log_entries: list,
    *,
    chunk_index: int,
    action_index_in_chunk: int,
    global_action_index: int,
    interp_step: int,
    interp_steps: int,
    state: dict,
    action: dict,
    timestamp: float,
) -> None:
    log_entries.append(
        {
            "chunk_index": chunk_index,
            "action_index_in_chunk": action_index_in_chunk,
            "global_action_index": global_action_index,
            "interp_step": interp_step,
            "interp_steps": interp_steps,
            "state": state,
            "action": action,
            "timestamp": timestamp,
        }
    )


def _save_replay_log(log_entries: list, output_dir: Path, timestamp: str) -> None:
    if not log_entries:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"replay_log_{timestamp}.jsonl"
    with open(filename, "w", encoding="utf-8") as f:
        for entry in log_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("回放日志已保存: %s", filename)


def _find_latest_libero_npz(data_root: Path) -> Optional[Path]:
    libero_dirs = sorted([p for p in data_root.rglob("libero_format") if p.is_dir()])
    npz_files: list[Path] = []
    for libero_dir in libero_dirs:
        npz_files.extend(sorted(libero_dir.glob("*_libero_*.npz")))
    if not npz_files:
        return None
    npz_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return npz_files[0]


def _load_libero_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    if "states" not in data or "actions" not in data:
        raise ValueError("libero_format npz 缺少 states/actions 字段")

    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)

    if states.ndim != 2 or states.shape[1] != 8:
        raise ValueError(f"states 维度应为 (T, 8)，但收到 {states.shape}")
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError(f"actions 维度应为 (T, 8)，但收到 {actions.shape}")
    if len(actions) != len(states):
        raise ValueError(f"actions 长度与 states 不一致: {len(actions)} vs {len(states)}")

    collection_frequency = float(data["collection_frequency"]) if "collection_frequency" in data else 30.0
    task = str(data["task"]) if "task" in data else "unknown"

    return {
        "states": states,
        "actions": actions,
        "collection_frequency": collection_frequency,
        "task": task,
    }


def _data_gripper_to_env(gripper_data: float) -> float:
    """Convert dataset gripper value to env value.

    Dataset (libero_format) uses 0=闭合, 1=张开; env uses 0=张开, 1=闭合.
    """
    return 1.0 if gripper_data < 0.5 else 0.0


def _move_to_start(env: KinovaRobotEnv, start_state: np.ndarray) -> None:
    start_state = np.asarray(start_state, dtype=np.float32)
    if start_state.shape != (8,):
        raise ValueError(f"start_state 维度应为 (8,), 但收到 {start_state.shape}")
    start_action = np.concatenate(
        [start_state[:7], [ _data_gripper_to_env(float(start_state[7])) ]]
    )
    env.step(start_action)
        time.sleep(1.0)


def _execute_action_chunk(
    env: KinovaRobotEnv,
    chunk: np.ndarray,
    *,
    current_state: np.ndarray,
    base_control_hz: float,
    interpolated_control_frequency_hz: float,
    last_commanded_joint_pos: Optional[np.ndarray],
    log_entries: list,
    chunk_index: int,
    chunk_start_index: int,
    inner_loop: bool,
    inner_loop_pos_tol: float,
    inner_loop_timeout_s: float,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    steps_per_action = max(1, int(interpolated_control_frequency_hz / base_control_hz))
    actual_interp_hz = base_control_hz * steps_per_action

    for action_index_in_chunk, action_delta in enumerate(chunk):
        global_action_index = chunk_start_index + action_index_in_chunk
            next_state = current_state + action_delta
            next_state[7] = float(np.clip(next_state[7], 0.0, 1.0))
            target_joint_pos = next_state[:7]
        target_gripper_pos = _data_gripper_to_env(float(next_state[7]))

        if steps_per_action == 1:
            step_start = time.time()
            action = np.concatenate([target_joint_pos, [target_gripper_pos]])
            env.step(action)
            _append_replay_log(
                log_entries,
                chunk_index=chunk_index,
                action_index_in_chunk=action_index_in_chunk,
                global_action_index=global_action_index,
                interp_step=1,
                interp_steps=1,
                state={
                    "current_state": current_state.tolist(),
                    "next_state": next_state.tolist(),
                },
                action={
                    "action_delta": action_delta.tolist(),
                    "command_joint_pos": target_joint_pos.tolist(),
                    "command_gripper_pos": float(target_gripper_pos),
                },
                timestamp=time.time(),
            )
            logger.info(
                "chunk=%d action=%d global=%d interp=%d/%d",
                chunk_index,
                action_index_in_chunk,
                global_action_index,
                1,
                1,
            )
            if inner_loop:
                _wait_until_reached(
                    env,
                    target_joint_pos,
                    pos_tol=inner_loop_pos_tol,
                    timeout_s=inner_loop_timeout_s,
                )
            else:
                _sleep_to_rate(step_start, base_control_hz)
                else:
            if last_commanded_joint_pos is None:
                obs = env.get_observation()
                start_joint_pos = obs["robot_state"]["joint_positions"]
            else:
                start_joint_pos = last_commanded_joint_pos

            interpolated_joints = _interpolate_joint_targets(
                start_joint_pos, target_joint_pos, steps_per_action
            )
            for interp_step_index, joint_target in enumerate(interpolated_joints, start=1):
                sub_action = np.concatenate([joint_target, [target_gripper_pos]])
                sub_step_start = time.time()
                env.step(sub_action)
                _append_replay_log(
                    log_entries,
                    chunk_index=chunk_index,
                    action_index_in_chunk=action_index_in_chunk,
                    global_action_index=global_action_index,
                    interp_step=interp_step_index,
                    interp_steps=len(interpolated_joints),
                    state={
                        "current_state": current_state.tolist(),
                        "next_state": next_state.tolist(),
                    },
                    action={
                        "action_delta": action_delta.tolist(),
                        "command_joint_pos": joint_target.tolist(),
                        "command_gripper_pos": float(target_gripper_pos),
                    },
                    timestamp=time.time(),
                )
                logger.info(
                    "chunk=%d action=%d global=%d interp=%d/%d",
                    chunk_index,
                    action_index_in_chunk,
                    global_action_index,
                    interp_step_index,
                    len(interpolated_joints),
                )
                if inner_loop:
                    _wait_until_reached(
                        env,
                        joint_target,
                        pos_tol=inner_loop_pos_tol,
                        timeout_s=inner_loop_timeout_s,
                    )
                else:
                    _sleep_to_rate(sub_step_start, actual_interp_hz)

        last_commanded_joint_pos = target_joint_pos
        current_state = next_state

    return current_state, last_commanded_joint_pos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kinova Gen3 轨迹回放脚本（libero_format）"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="数据根目录（默认: examples/kinova_gen3/data）",
    )
    parser.add_argument(
        "--robot-ip",
        type=str,
        default="192.168.1.10",
        help="Kinova 机械臂 IP 地址",
    )
    parser.add_argument(
        "--gripper-ip",
        type=str,
        default="192.168.1.43",
        help="夹爪控制器 IP 地址",
    )
    parser.add_argument(
        "--external-camera-serial",
        type=str,
        default=None,
        help="外部相机序列号（可选）",
    )
    parser.add_argument(
        "--wrist-camera-serial",
        type=str,
        default=None,
        help="腕部相机序列号（可选）",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=8,
        help="每次执行的 action chunk 大小（默认: 8）",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="回放速度倍数（1.0 = 原始速度）",
    )
    parser.add_argument(
        "--interpolated-control-frequency-hz",
        type=float,
        default=120.0,
        help="插值控制频率（默认: 120Hz）",
    )
    parser.add_argument(
        "--inner-loop",
        action="store_true",
        help="每次指令后等待机械臂到达目标位置再进入下一步",
    )
    parser.add_argument(
        "--inner-loop-pos-tol",
        type=float,
        default=0.01,
        help="内环等待的关节误差阈值（弧度，默认: 0.01）",
    )
    parser.add_argument(
        "--inner-loop-timeout-s",
        type=float,
        default=2.0,
        help="内环等待超时（秒，默认: 2.0）",
    )
    
    args = parser.parse_args()
    
        script_dir = Path(__file__).parent
    data_root = Path(args.data_dir) if args.data_dir else script_dir / "data"

    npz_path = _find_latest_libero_npz(data_root)
    if npz_path is None:
        logger.error(f"在 {data_root} 下未找到 libero_format/*.npz")
            sys.exit(1)

    logger.info(f"使用数据文件: {npz_path}")
    data = _load_libero_npz(npz_path)

    states = data["states"]
    actions = data["actions"]
    collection_frequency = data["collection_frequency"]
    task = data["task"]

    base_control_hz = collection_frequency * args.playback_speed
    if base_control_hz <= 0:
        logger.error("无效的回放频率")
            sys.exit(1)
    
    logger.info(f"任务: {task}")
    logger.info(
        "采集频率: %.2f Hz, 回放速度: %.2fx, 基准控制频率: %.2f Hz",
        collection_frequency,
        args.playback_speed,
        base_control_hz,
    )
    logger.info("action_horizon: %d, 插值频率: %.2f Hz", args.action_horizon, args.interpolated_control_frequency_hz)

    env = KinovaRobotEnv(
        robot_ip=args.robot_ip,
        gripper_ip=args.gripper_ip,
        external_camera_serial=args.external_camera_serial,
        wrist_camera_serial=args.wrist_camera_serial,
        action_mode=ActionMode.ABSOLUTE,
    )

    try:
        # 清除可能存在的急停/故障状态，避免指令被忽略
        try:
            env.clear_estop()
            logger.info("已清除急停/故障状态")
        except Exception as e:
            logger.warning("清除急停/故障状态失败: %s", e)

        logger.info("移动到初始位置...")
        _move_to_start(env, states[0])
        logger.info("开始回放...\n")

        replay_log_entries = []
        log_timestamp = time.strftime("%Y_%m_%d_%H-%M-%S")
        last_commanded_joint_pos = None
        current_state = states[0].copy()
        num_actions = len(actions)
        for chunk_index, chunk_start in enumerate(range(0, num_actions, args.action_horizon)):
            chunk_end = min(chunk_start + args.action_horizon, num_actions)
            chunk = actions[chunk_start:chunk_end]
            current_state, last_commanded_joint_pos = _execute_action_chunk(
                env,
                chunk,
                current_state=current_state,
                base_control_hz=base_control_hz,
                interpolated_control_frequency_hz=args.interpolated_control_frequency_hz,
                last_commanded_joint_pos=last_commanded_joint_pos,
                log_entries=replay_log_entries,
                chunk_index=chunk_index,
                chunk_start_index=chunk_start,
                inner_loop=args.inner_loop,
                inner_loop_pos_tol=args.inner_loop_pos_tol,
                inner_loop_timeout_s=args.inner_loop_timeout_s,
                )

        logger.info("\n轨迹回放完成！")
        _save_replay_log(replay_log_entries, npz_path.parent, log_timestamp)
    except KeyboardInterrupt:
        logger.warning("\n回放被用户中断\n")
    except Exception as e:
        logger.error(f"回放失败: {e}")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
