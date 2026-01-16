#!/usr/bin/env python3
"""
Kinova Gen3 安全检测模块

功能:
1) 基于 URDF 的关节正运动学，计算各关节与末端在基座坐标系的位置
2) 从 boundingbox.txt 提取安全区域长方体范围
3) 提供软急停/硬急停安全检测逻辑
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger("SafetyMonitor")

from safety_box import SafetyBox, load_safety_box
from urdf_kinematics import URDFKinematics


class SafetyMonitor:
    """安全检测器：基于安全区域判断是否允许执行动作。"""

    def __init__(
        self,
        urdf_path: str | Path,
        boundingbox_path: str | Path,
        mode: str = "soft",
        ignore_joint_names: Optional[Iterable[str]] = None,
        monitored_joints: Optional[Iterable[int]] = None,
    ) -> None:
        """
        初始化安全检测器
        
        Args:
            urdf_path: URDF 文件路径
            boundingbox_path: boundingbox.txt 文件路径
            mode: 安全模式，"soft" 或 "hard"
            ignore_joint_names: 要忽略的关节名称列表（旧接口，已废弃，建议使用 monitored_joints）
            monitored_joints: 要监督的关节编号列表（1-7 表示关节1-7，8 表示末端位置）
                            默认 None 表示监督所有关节（除了第一个关节，如果 ignore_joint_names 包含 Actuator1）
        """
        self.kinematics = URDFKinematics(urdf_path)
        self.safety_box = load_safety_box(boundingbox_path)
        self.mode = mode.lower()
        self.ignore_joint_names = set(ignore_joint_names or [])
        
        # 处理 monitored_joints 参数
        if monitored_joints is not None:
            # 使用新的 monitored_joints 参数
            monitored_set = set(monitored_joints)
            if not all(1 <= j <= 8 for j in monitored_set):
                raise ValueError("monitored_joints 中的值必须在 1-8 范围内（1-7=关节，8=末端）")
            self.monitored_joints = monitored_set
        else:
            # 兼容旧接口：如果没有指定 monitored_joints，使用 ignore_joint_names 的逻辑
            # 默认监督所有关节（除了被 ignore_joint_names 忽略的）
            if "Actuator1" in self.ignore_joint_names:
                # 如果忽略第一个关节，默认监督 2-8
                self.monitored_joints = set(range(2, 9))  # 2, 3, 4, 5, 6, 7, 8
            else:
                # 否则监督所有 1-8
                self.monitored_joints = set(range(1, 9))  # 1, 2, 3, 4, 5, 6, 7, 8

        if self.mode not in ("soft", "hard"):
            raise ValueError(f"未知安全模式: {mode}, 仅支持 soft/hard")

        logger.info(
            "安全区边界: x=[%.4f, %.4f], y=[%.4f, %.4f], z=[%.4f, %.4f]",
            self.safety_box.x_min,
            self.safety_box.x_max,
            self.safety_box.y_min,
            self.safety_box.y_max,
            self.safety_box.z_min,
            self.safety_box.z_max,
        )
        logger.info("监督的关节: %s", sorted(self.monitored_joints))

    def check_and_handle(self, env, joint_angles: Sequence[float]) -> bool:
        """
        返回 True 表示安全；False 表示软急停下忽略指令。
        硬急停会直接触发急停并抛出异常。
        """
        joint_positions, eef_pos = self.kinematics.compute_joint_positions(joint_angles)
        violations = self._find_violations(joint_positions, eef_pos)
        if not violations:
            return True

        message = "; ".join(violations)
        if self.mode == "soft":
            logger.warning("安全区超界（软急停，忽略指令）: %s", message)
            return False

        logger.error("安全区超界（硬急停）: %s", message)
        self._apply_emergency_stop(env)
        raise RuntimeError("触发硬急停：机械臂已锁死")

    def _find_violations(self, joint_positions: np.ndarray, eef_pos: np.ndarray) -> List[str]:
        """
        查找违反安全区域的关节/末端位置
        
        Args:
            joint_positions: (N, 3) 关节位置数组
            eef_pos: (3,) 末端执行器位置
            
        Returns:
            违反安全区域的关节/末端列表
        """
        violations = []
        
        # 检查关节位置（关节编号 1-7）
        for joint_idx, (name, pos) in enumerate(zip(self.kinematics.actuated_joint_names, joint_positions), start=1):
            # 只检查在 monitored_joints 中的关节
            if joint_idx in self.monitored_joints:
                if not self.safety_box.contains(pos):
                    violations.append(f"{name}(关节{joint_idx})={pos.tolist()}")
        
        # 检查末端位置（编号 8）
        if 8 in self.monitored_joints:
            if not self.safety_box.contains(eef_pos):
                violations.append(f"end_effector(末端)={eef_pos.tolist()}")
        
        return violations

    @staticmethod
    def _apply_emergency_stop(env) -> None:
        try:
            if hasattr(env, "_base"):
                env._base.ApplyEmergencyStop()
            if hasattr(env, "_estop_triggered"):
                env._estop_triggered = True
        except Exception as exc:
            logger.error("急停命令发送失败: %s", exc)
