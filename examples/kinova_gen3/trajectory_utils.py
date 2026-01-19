# ruff: noqa
"""轨迹与插值相关工具函数。"""

import math
from typing import Optional

import numpy as np


def _interpolate_joint_targets(
    start_joint_pos: np.ndarray, end_joint_pos: np.ndarray, steps: int
) -> np.ndarray:
    """Linearly interpolate joint targets between two poses.

    Returns an array with shape (steps, 7). When steps == 1, it returns [end].
    """
    steps = max(1, int(steps))
    return np.linspace(start_joint_pos, end_joint_pos, steps + 1, endpoint=True)[1:]


def _clamp_joint_targets_rad(
    joint_targets: np.ndarray, joint_limits_deg: list[tuple[float, float]]
) -> np.ndarray:
    """Clamp joint targets to hard limits in degrees, return radians."""
    joint_targets = np.asarray(joint_targets, dtype=float)
    if joint_targets.ndim != 2 or joint_targets.shape[1] != len(joint_limits_deg):
        return joint_targets
    targets_deg = np.degrees(joint_targets)
    # wrap to [-180, 180] for stable clamping
    targets_deg = (targets_deg + 180.0) % 360.0 - 180.0
    for j, (lo, hi) in enumerate(joint_limits_deg):
        targets_deg[:, j] = np.clip(targets_deg[:, j], lo, hi)
    return np.radians(targets_deg)


def _build_waypoint_trajectory(
    action_chunk: np.ndarray,
    *,
    control_frequency: float,
    inter: int,
    max_joint_speed_deg_s: float,
    current_joint_pos: Optional[np.ndarray] = None,
    max_joint_accel_deg_s2: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a joint waypoint trajectory from an action chunk.

    Args:
        action_chunk: (N, 8) action chunk with absolute joint targets
        control_frequency: Control frequency (Hz)
        inter: Interpolation points between actions
        max_joint_speed_deg_s: Maximum joint speed (deg/s)
        current_joint_pos: (7,) Current joint position in radians (optional).
                          If provided, used to compute first waypoint duration.

    Returns:
        joint_targets: (N, 7) joint positions
        durations: (N,) seconds, each point relative to the previous
        gripper_targets: (N,) gripper targets aligned with joint_targets
    """
    # action_chunk is expected to contain absolute joint targets.
    action_chunk = np.asarray(action_chunk)
    if action_chunk.size == 0:
        return np.zeros((0, 7)), np.zeros((0,)), np.zeros((0,))

    joint_targets = []
    gripper_targets = []

    # If current_joint_pos is provided, interpolate from current -> first action
    if current_joint_pos is not None:
        pre_interp = _interpolate_joint_targets(
            np.asarray(current_joint_pos, dtype=float),
            action_chunk[0][:7],
            inter + 1,
        )
        for joint_target in pre_interp:
            joint_targets.append(joint_target)
            gripper_targets.append(action_chunk[0][-1])
    else:
        # Always include the first action
        joint_targets.append(action_chunk[0][:7])
        gripper_targets.append(action_chunk[0][-1])

    for i in range(len(action_chunk) - 1):
        start = action_chunk[i][:7]
        end = action_chunk[i + 1][:7]
        interpolated = _interpolate_joint_targets(start, end, inter + 1)
        for k, joint_target in enumerate(interpolated):
            joint_targets.append(joint_target)
            if k == len(interpolated) - 1:
                gripper_targets.append(action_chunk[i + 1][-1])
            else:
                gripper_targets.append(action_chunk[i][-1])

    def _min_duration_deg(delta_deg: np.ndarray) -> float:
        max_speed = max(max_joint_speed_deg_s, 1e-6)
        safety_scale = 1.2
        if max_joint_accel_deg_s2 is None:
            max_delta_deg = float(np.max(delta_deg))
            duration = max_delta_deg / max_speed
            return max(duration * safety_scale, 0.01)
        max_accel = max(max_joint_accel_deg_s2, 1e-6)
        per_joint = []
        for d in np.abs(delta_deg):
            if d <= 0.0:
                per_joint.append(0.01)
                continue
            accel_dist = (max_speed ** 2) / max_accel
            if d <= accel_dist:
                t = 2.0 * math.sqrt(d / max_accel)
            else:
                t = 2.0 * (max_speed / max_accel) + (d - accel_dist) / max_speed
            per_joint.append(max(t, 0.01))
        base = float(np.max(per_joint)) if per_joint else 0.01
        return max(base * safety_scale, 0.01)

    # Derive per-step durations based on joint speed/accel limits (deg/s, deg/s^2).
    durations = []
    if len(joint_targets) <= 1:
        if current_joint_pos is not None:
            delta_rad = np.abs(joint_targets[0] - current_joint_pos)
            delta_deg = np.degrees(delta_rad)
            durations = [_min_duration_deg(delta_deg)]
        else:
            durations = [1.0 / max(control_frequency, 1e-6)]
    else:
        # First waypoint duration
        if current_joint_pos is not None:
            delta_rad = np.abs(joint_targets[0] - current_joint_pos)
            delta_deg = np.degrees(delta_rad)
            durations.append(_min_duration_deg(delta_deg))
        else:
            durations.append(0.01)

        # Remaining waypoint durations
        for idx in range(1, len(joint_targets)):
            delta_rad = np.abs(joint_targets[idx] - joint_targets[idx - 1])
            delta_deg = np.degrees(delta_rad)
            durations.append(_min_duration_deg(delta_deg))

    return (
        np.asarray(joint_targets, dtype=float),
        np.asarray(durations, dtype=float),
        np.asarray(gripper_targets, dtype=float),
    )
