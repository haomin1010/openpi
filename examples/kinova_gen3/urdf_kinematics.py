#!/usr/bin/env python3
"""
URDF 关节正运动学工具

提供:
- URDFKinematics: 计算关节与末端在基座坐标系的位置
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class JointInfo:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


class URDFKinematics:
    """基于 URDF 的简易正运动学（串联结构）。"""

    def __init__(
        self,
        urdf_path: str | Path,
        base_link: str = "base_link",
        tip_link: str = "end_effector_link",
    ) -> None:
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF 文件不存在: {self.urdf_path}")
        self.base_link = base_link
        self.tip_link = tip_link

        self._joints = self._parse_joints()
        self._chain = self._build_chain()
        self.actuated_joint_names = [
            joint.name for joint in self._chain if joint.joint_type in ("revolute", "continuous")
        ]

    def _parse_joints(self) -> List[JointInfo]:
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        joints: List[JointInfo] = []
        for joint in root.findall("joint"):
            name = joint.get("name")
            joint_type = joint.get("type")
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            origin = joint.find("origin")
            if origin is not None:
                xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()], dtype=np.float64)
                rpy = np.array([float(v) for v in origin.get("rpy", "0 0 0").split()], dtype=np.float64)
            else:
                xyz = np.zeros(3, dtype=np.float64)
                rpy = np.zeros(3, dtype=np.float64)
            axis_tag = joint.find("axis")
            if axis_tag is not None:
                axis = np.array([float(v) for v in axis_tag.get("xyz", "0 0 1").split()], dtype=np.float64)
            else:
                axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            joints.append(
                JointInfo(
                    name=name,
                    joint_type=joint_type,
                    parent=parent,
                    child=child,
                    origin_xyz=xyz,
                    origin_rpy=rpy,
                    axis=axis,
                )
            )
        return joints

    def _build_chain(self) -> List[JointInfo]:
        parent_to_joint = {}
        for joint in self._joints:
            parent_to_joint[joint.parent] = joint

        chain: List[JointInfo] = []
        current = self.base_link
        visited = set()
        while current in parent_to_joint:
            if current in visited:
                raise ValueError(f"URDF 关节链存在循环，无法构建: {current}")
            visited.add(current)
            joint = parent_to_joint[current]
            chain.append(joint)
            current = joint.child
            if current == self.tip_link:
                break
        if not chain:
            raise ValueError(f"无法从 base_link 构建关节链: {self.base_link}")
        return chain

    def compute_joint_positions(
        self, joint_angles: Sequence[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算各个关节与末端在基座坐标系中的位置。

        Returns:
            joint_positions: (N, 3) 按关节顺序（Actuator1..7）排列
            eef_position: (3,) 末端执行器位置
        """
        joint_angles = np.asarray(joint_angles, dtype=np.float64)
        if len(joint_angles) != len(self.actuated_joint_names):
            raise ValueError(
                f"关节角数量不匹配: got {len(joint_angles)}, expected {len(self.actuated_joint_names)}"
            )

        T = np.eye(4, dtype=np.float64)
        joint_positions = []
        angle_idx = 0

        for joint in self._chain:
            T_origin = _make_transform(joint.origin_xyz, joint.origin_rpy)
            T_joint = T @ T_origin

            if joint.joint_type in ("revolute", "continuous"):
                joint_positions.append(T_joint[:3, 3].copy())
                angle = float(joint_angles[angle_idx])
                angle_idx += 1
                T = T_joint @ _axis_angle_transform(joint.axis, angle)
            else:
                T = T_joint

        eef_position = T[:3, 3].copy()
        return np.asarray(joint_positions, dtype=np.float64), eef_position


def _make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    R = _rpy_to_rotation(rpy)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


def _rpy_to_rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    # URDF: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _axis_angle_transform(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        axis = axis / norm
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c

    R = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T
