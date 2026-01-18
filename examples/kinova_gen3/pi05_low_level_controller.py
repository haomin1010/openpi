#!/usr/bin/env python3
"""
Kinova Gen3 低层控制脚本（1kHz Low-Level Servoing）。

功能概述：
- 接收 pi05 模型输出的 action_chunk（N 步关节角度，弧度）
- Temporal Ensembling：新旧 chunk 平滑加权过渡
- Cubic Spline 上采样至 1kHz
- 每帧发送 position + velocity 前馈（BaseCyclic.Refresh）
- 软限速与限加速度
- 掉帧/延迟时使用上次有效速度线性外推
- 平滑下电（Servo Off）退出
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

# 设置 protobuf 环境变量以兼容 kortex_api
if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" not in os.environ:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Session_pb2
    from kortex_api.RouterClient import RouterClient
    from kortex_api.SessionManager import SessionManager
    from kortex_api.TCPTransport import TCPTransport
except ImportError as exc:
    raise ImportError(
        "kortex_api 未安装。请从 Kinova 官网下载并安装 kortex_api wheel 包。"
    ) from exc

logger = logging.getLogger("KinovaLowLevelController")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

NUM_JOINTS = 7


def smoothstep(x: np.ndarray) -> np.ndarray:
    return x * x * (3.0 - 2.0 * x)


def soft_clip(x: np.ndarray, limit: np.ndarray) -> np.ndarray:
    if np.all(limit <= 0):
        return x
    return limit * np.tanh(x / limit)


def wrap_to_near(target_deg: float, current_deg: float) -> float:
    delta = target_deg - current_deg
    return target_deg - 360.0 * round(delta / 360.0)


def to_array(value: float | np.ndarray, size: int) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    return np.full((size,), float(value), dtype=np.float64)


@dataclasses.dataclass
class ControllerConfig:
    robot_ip: str = "192.168.1.10"
    robot_port: int = 10000
    username: str = "admin"
    password: str = "admin"
    control_hz: int = 1000
    model_hz: int = 20
    vel_limit: float | np.ndarray = 2.0  # rad/s
    acc_limit: float | np.ndarray = 10.0  # rad/s^2
    spline_bc: str = "clamped"
    position_alpha: float = 1.0
    velocity_alpha: float = 1.0
    use_feedback_start: bool = True
    temporal_ensemble: bool = True
    chunk_starts_at_current: bool = False
    plan_mode: str = "replace"  # replace | append


class KinovaController:
    def __init__(self, config: ControllerConfig):
        self._config = config
        self._dt_control = 1.0 / float(config.control_hz)
        self._dt_model = 1.0 / float(config.model_hz)
        self._vel_limit = to_array(config.vel_limit, NUM_JOINTS)
        self._acc_limit = to_array(config.acc_limit, NUM_JOINTS)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._plan_queue: Deque[Tuple[np.ndarray, np.ndarray]] = deque()
        self._current_chunk: Optional[np.ndarray] = None
        self._chunk_start_time: Optional[float] = None

        self._last_command_pos = np.zeros((NUM_JOINTS,), dtype=np.float64)
        self._last_velocity = np.zeros((NUM_JOINTS,), dtype=np.float64)
        self._last_valid_velocity = np.zeros((NUM_JOINTS,), dtype=np.float64)
        self._last_feedback_pos = np.zeros((NUM_JOINTS,), dtype=np.float64)
        self._ll_initialized = False

        self._connect()

    def _connect(self) -> None:
        self._transport = TCPTransport()
        self._transport.connect(self._config.robot_ip, self._config.robot_port)
        self._router = RouterClient(self._transport, lambda e: logger.error(f"Router error: {e}"))
        self._session_manager = SessionManager(self._router)
        create_session_info = Session_pb2.CreateSessionInfo()
        create_session_info.username = self._config.username
        create_session_info.password = self._config.password
        create_session_info.session_inactivity_timeout = 60000
        create_session_info.connection_inactivity_timeout = 2000
        self._session_manager.CreateSession(create_session_info)

        self._base = BaseClient(self._router)
        self._base_cyclic = BaseCyclicClient(self._router)

        feedback = self._base_cyclic.RefreshFeedback()
        self._last_feedback_pos = np.array(
            [math.radians(actuator.position) for actuator in feedback.actuators],
            dtype=np.float64,
        )
        self._last_command_pos = self._last_feedback_pos.copy()

    def _send_hold_for(self, duration_s: float = 0.5) -> None:
        feedback = self._base_cyclic.RefreshFeedback()
        t_end = time.perf_counter() + duration_s
        while time.perf_counter() < t_end:
            command = BaseCyclic_pb2.Command()
            command.frame_id = feedback.frame_id
            for i in range(NUM_JOINTS):
                actuator_command = command.actuators.add()
                actuator_command.position = feedback.actuators[i].position
                actuator_command.velocity = 0.0
                actuator_command.torque_joint = 0.0
                actuator_command.command_id = feedback.actuators[i].command_id
            self._base_cyclic.Refresh(command)
            time.sleep(self._dt_control)

    def _ensure_low_level(self) -> None:
        servo_mode = Base_pb2.ServoingModeInformation()
        servo_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        self._base.SetServoingMode(servo_mode)
        self._send_hold_for(0.5)
        logger.info("已切换到 Low-Level Servoing 模式并完成保持")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        logger.info("控制循环已启动 (1kHz)")

    def update_chunk(self, new_chunk: np.ndarray) -> None:
        new_chunk = np.asarray(new_chunk, dtype=np.float64)
        if new_chunk.ndim != 2 or new_chunk.shape[1] != NUM_JOINTS:
            raise ValueError("new_chunk 形状应为 (N, 7)")

        now = time.perf_counter()
        with self._lock:
            old_remaining = None
            if self._current_chunk is not None and self._chunk_start_time is not None:
                elapsed_steps = int(max(0.0, (now - self._chunk_start_time) * self._config.model_hz))
                if elapsed_steps < len(self._current_chunk):
                    old_remaining = self._current_chunk[elapsed_steps:]

            if self._config.temporal_ensemble:
                blended_chunk = self._temporal_ensemble(new_chunk, old_remaining)
            else:
                blended_chunk = new_chunk
            self._current_chunk = blended_chunk
            self._chunk_start_time = now
            self._plan_from_chunk(blended_chunk)

    def _temporal_ensemble(
        self,
        new_chunk: np.ndarray,
        old_remaining: Optional[np.ndarray],
    ) -> np.ndarray:
        if old_remaining is None or len(old_remaining) == 0:
            return new_chunk

        m = min(len(new_chunk), len(old_remaining))
        if m == 0:
            return new_chunk

        weights = smoothstep(np.linspace(0.0, 1.0, m, dtype=np.float64))
        blended = (1.0 - weights[:, None]) * old_remaining[:m] + weights[:, None] * new_chunk[:m]

        if len(new_chunk) > m:
            tail = new_chunk[m:]
        else:
            tail = old_remaining[m:]

        if len(tail) > 0:
            blended = np.vstack([blended, tail])
        return blended

    def _plan_from_chunk(self, chunk: np.ndarray) -> None:
        if len(chunk) == 0:
            return

        start_pos = (
            self._last_feedback_pos.copy()
            if self._config.use_feedback_start
            else self._last_command_pos.copy()
        )
        start_vel = self._last_velocity.copy()

        if self._config.chunk_starts_at_current:
            positions = chunk
            t_points = np.arange(0.0, len(chunk) * self._dt_model, self._dt_model)
            if len(positions) > 1:
                v_start = (positions[1] - positions[0]) / self._dt_model
                v_end = (positions[-1] - positions[-2]) / self._dt_model
            else:
                v_start = np.zeros_like(start_pos)
                v_end = np.zeros_like(start_pos)
        else:
            t_points = np.arange(0.0, (len(chunk) + 1) * self._dt_model, self._dt_model)
            positions = np.vstack([start_pos, chunk])
            v_start = start_vel
            if len(positions) > 1:
                v_end = (positions[-1] - positions[-2]) / self._dt_model
            else:
                v_end = np.zeros_like(start_pos)

        t_hr = np.arange(0.0, t_points[-1] + self._dt_control * 0.5, self._dt_control)
        pos_hr = np.zeros((len(t_hr), NUM_JOINTS), dtype=np.float64)
        for j in range(NUM_JOINTS):
            if self._config.spline_bc == "clamped":
                cs = CubicSpline(
                    t_points,
                    positions[:, j],
                    bc_type=((1, v_start[j]), (1, v_end[j])),
                )
            else:
                cs = CubicSpline(t_points, positions[:, j], bc_type="natural")
            pos_hr[:, j] = cs(t_hr)

        vel_hr = np.zeros_like(pos_hr)
        if len(pos_hr) > 1:
            vel_hr[:-1] = np.diff(pos_hr, axis=0) / self._dt_control
            vel_hr[-1] = vel_hr[-2]

        limited_vel = np.zeros_like(vel_hr)
        prev_vel = self._last_velocity.copy()
        for i in range(len(vel_hr)):
            raw_vel = vel_hr[i]
            acc = (raw_vel - prev_vel) / self._dt_control
            acc = soft_clip(acc, self._acc_limit)
            vel = prev_vel + acc * self._dt_control
            vel = soft_clip(vel, self._vel_limit)
            limited_vel[i] = vel
            prev_vel = vel

        if self._config.plan_mode == "replace":
            self._plan_queue.clear()
        for i in range(len(t_hr)):
            self._plan_queue.append((pos_hr[i], limited_vel[i]))

    def _control_loop(self) -> None:
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            try:
                if not self._ll_initialized:
                    self._ensure_low_level()
                    self._ll_initialized = True

                feedback = self._base_cyclic.RefreshFeedback()
                self._last_feedback_pos = np.array(
                    [math.radians(actuator.position) for actuator in feedback.actuators],
                    dtype=np.float64,
                )

                with self._lock:
                    if self._plan_queue:
                        target_pos, target_vel = self._plan_queue.popleft()
                        self._last_valid_velocity = target_vel
                    else:
                        target_vel = self._last_valid_velocity
                        target_pos = self._last_command_pos + target_vel * self._dt_control

                pos_alpha = float(np.clip(self._config.position_alpha, 0.0, 1.0))
                vel_alpha = float(np.clip(self._config.velocity_alpha, 0.0, 1.0))
                if pos_alpha < 1.0:
                    target_pos = (1.0 - pos_alpha) * self._last_command_pos + pos_alpha * target_pos
                if vel_alpha < 1.0:
                    target_vel = (1.0 - vel_alpha) * self._last_velocity + vel_alpha * target_vel

                acc = (target_vel - self._last_velocity) / self._dt_control
                acc = soft_clip(acc, self._acc_limit)
                target_vel = self._last_velocity + acc * self._dt_control
                target_vel = soft_clip(target_vel, self._vel_limit)

                command = BaseCyclic_pb2.Command()
                command.frame_id = feedback.frame_id
                for i in range(NUM_JOINTS):
                    actuator_command = command.actuators.add()
                    pos_deg = math.degrees(target_pos[i])
                    pos_deg = wrap_to_near(pos_deg, feedback.actuators[i].position)
                    actuator_command.position = pos_deg
                    actuator_command.velocity = math.degrees(target_vel[i])
                    actuator_command.torque_joint = 0.0
                    actuator_command.command_id = feedback.actuators[i].command_id

                self._base_cyclic.Refresh(command)
                self._last_command_pos = target_pos
                self._last_velocity = target_vel
            except Exception as exc:
                logger.warning(f"控制循环异常: {exc}")

            elapsed = time.perf_counter() - loop_start
            sleep_time = self._dt_control - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        try:
            feedback = self._base_cyclic.RefreshFeedback()
            for _ in range(int(0.2 / self._dt_control)):
                command = BaseCyclic_pb2.Command()
                command.frame_id = feedback.frame_id
                for i in range(NUM_JOINTS):
                    actuator_command = command.actuators.add()
                    actuator_command.position = feedback.actuators[i].position
                    actuator_command.velocity = 0.0
                    actuator_command.torque_joint = 0.0
                    actuator_command.command_id = feedback.actuators[i].command_id
                self._base_cyclic.Refresh(command)
                time.sleep(self._dt_control)
        except Exception as exc:
            logger.warning(f"平滑下电发送失败: {exc}")

        try:
            servo_mode = Base_pb2.ServoingModeInformation()
            servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            self._base.SetServoingMode(servo_mode)
        except Exception as exc:
            logger.warning(f"退出 servoing 模式失败: {exc}")

        try:
            self._session_manager.CloseSession()
        except Exception as exc:
            logger.warning(f"关闭会话失败: {exc}")
        try:
            self._transport.disconnect()
        except Exception as exc:
            logger.warning(f"断开连接失败: {exc}")

    def __enter__(self) -> "KinovaController":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()


@dataclasses.dataclass
class Args:
    robot_ip: str = "192.168.1.10"
    control_hz: int = 1000
    model_hz: int = 20
    chunk_size: int = 8
    vel_limit: float = 2.0
    acc_limit: float = 10.0
    simulate: bool = True


def _simulate_chunks(controller: KinovaController, args: Args) -> None:
    t = 0.0
    dt = 1.0 / args.model_hz
    base_pos = controller._last_command_pos.copy()
    while True:
        chunk = []
        for _ in range(args.chunk_size):
            offset = 0.1 * math.sin(2.0 * math.pi * 0.2 * t)
            chunk.append(base_pos + np.array([offset, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            t += dt
        controller.update_chunk(np.asarray(chunk))
        time.sleep(args.chunk_size * dt)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Kinova Gen3 低层控制示例")
    parser.add_argument("--robot-ip", type=str, default="192.168.1.10")
    parser.add_argument("--control-hz", type=int, default=1000)
    parser.add_argument("--model-hz", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--vel-limit", type=float, default=2.0)
    parser.add_argument("--acc-limit", type=float, default=10.0)
    parser.add_argument("--simulate", action="store_true", default=False)
    args = parser.parse_args()

    config = ControllerConfig(
        robot_ip=args.robot_ip,
        control_hz=args.control_hz,
        model_hz=args.model_hz,
        vel_limit=args.vel_limit,
        acc_limit=args.acc_limit,
    )

    controller = KinovaController(config)
    controller.start()

    try:
        if args.simulate:
            logger.info("使用模拟 action_chunk 进行测试")
            _simulate_chunks(controller, args)
        else:
            logger.info("已启动控制器，等待外部调用 update_chunk()")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("收到中断信号，准备退出")
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
