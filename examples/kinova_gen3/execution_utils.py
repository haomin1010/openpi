# ruff: noqa
"""执行控制相关工具函数。"""

import contextlib
import signal
import time

import numpy as np

from kinova_env import KinovaRobotEnv


def _sleep_to_rate(start_time: float, hz: float) -> None:
    """Sleep to maintain a target loop rate."""
    if hz <= 0:
        return
    elapsed = time.time() - start_time
    period = 1.0 / hz
    if elapsed < period:
        time.sleep(period - elapsed)


def _wait_until_reached(
    env: KinovaRobotEnv,
    target_joint_pos: np.ndarray,
    *,
    pos_tol: float,
    timeout_s: float,
    poll_dt: float = 0.02,
) -> bool:
    """Wait until joint positions are within tolerance or timeout.
    
    优化：直接使用 RefreshFeedback() 而不是 get_observation()，
    避免创建图像和处理不必要的数据，减少轮询开销。
    """
    import math
    start_time = time.time()
    while True:
        # 直接使用 RefreshFeedback()，避免 get_observation() 的开销
        # get_observation() 会创建图像和处理不必要的数据
        feedback = env._base_cyclic.RefreshFeedback()
        actual_joint_pos = np.array([
            math.radians((actuator.position + 180) % 360 - 180)
            for actuator in feedback.actuators
        ])
        err = np.linalg.norm(target_joint_pos - actual_joint_pos)
        if err <= pos_tol:
            return True
        if time.time() - start_time >= timeout_s:
            return False
        time.sleep(poll_dt)


def _monitor_waypoints_execution(
    env: KinovaRobotEnv,
    target_joint_pos: np.ndarray,
    *,
    total_duration: float,
    poll_dt: float,
    pos_tol: float,
    no_motion_threshold: float,
    no_motion_max_count: int,
) -> tuple[bool, np.ndarray]:
    """Return (moved, max_joint_speed_rad_s)."""
    obs = env.get_observation()
    last_pos = np.array(obs["robot_state"]["joint_positions"], dtype=float)
    last_time = time.time()
    max_speed = np.zeros_like(last_pos)
    no_motion_count = 0
    start_time = last_time

    while True:
        time.sleep(poll_dt)
        obs = env.get_observation()
        cur_pos = np.array(obs["robot_state"]["joint_positions"], dtype=float)
        now = time.time()
        dt = max(now - last_time, 1e-6)
        delta = cur_pos - last_pos
        speed = np.abs(delta) / dt
        max_speed = np.maximum(max_speed, speed)

        if np.linalg.norm(delta) < no_motion_threshold:
            no_motion_count += 1
        else:
            no_motion_count = 0

        if np.linalg.norm(target_joint_pos - cur_pos) <= pos_tol:
            return True, max_speed

        if no_motion_count >= no_motion_max_count:
            return False, max_speed

        if now - start_time >= total_duration + 1.0:
            return True, max_speed

        last_pos = cur_pos
        last_time = now


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """
    临时阻止键盘中断，延迟处理直到受保护的代码执行完毕。
    用于防止在等待策略服务器响应时被 Ctrl+C 中断导致连接问题。
    """
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt
