#!/usr/bin/env python3
"""
根据最新的 libero_format 数据，模拟 pi05 输出 action_chunk，
驱动 KinovaController 复现轨迹。
python3 examples/kinova_gen3/simulated_chunk_generator.py --robot-ip 192.168.1.10
轨迹正确：
python3 examples/kinova_gen3/simulated_chunk_generator.py \
  --robot-ip 192.168.1.10 \
  --append-plan \
  --position-alpha 1.0 \
  --velocity-alpha 1.0
平滑：
python3 examples/kinova_gen3/simulated_chunk_generator.py \
  --robot-ip 192.168.1.10 \
  --append-plan \
  --position-alpha 0.6 \
  --velocity-alpha 0.6
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from pi05_low_level_controller import ControllerConfig, KinovaController

logger = logging.getLogger("SimulatedChunkGenerator")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def find_latest_libero_npz(data_root: Path) -> Path:
    libero_dirs = [p for p in data_root.rglob("libero_format") if p.is_dir()]
    npz_files: list[Path] = []
    for libero_dir in libero_dirs:
        npz_files.extend(sorted(libero_dir.glob("*_libero_*.npz")))

    if not npz_files:
        raise FileNotFoundError(f"在 {data_root} 下未找到 libero_format 数据")

    return max(npz_files, key=lambda p: p.stat().st_mtime)


def load_latest_episode(data_dir: Optional[Path]) -> Tuple[np.ndarray, int, Path]:
    if data_dir is None:
        script_data_root = Path(__file__).resolve().parent / "data"
        cwd_data_root = Path.cwd() / "data"
        data_root = script_data_root if script_data_root.exists() else cwd_data_root
    else:
        data_root = data_dir

    npz_path = find_latest_libero_npz(data_root)
    data = np.load(npz_path, allow_pickle=True)

    states = np.asarray(data["states"], dtype=np.float64)
    if states.ndim != 2 or states.shape[1] < 7:
        raise ValueError(f"states 维度异常: {states.shape}")

    collection_frequency = 30
    if "collection_frequency" in data:
        collection_frequency = int(data["collection_frequency"])

    joint_positions = states[:, :7].copy()
    return joint_positions, collection_frequency, npz_path


class SimulatedChunkGenerator:
    def __init__(
        self,
        joint_positions: np.ndarray,
        model_hz: int,
        chunk_size: int,
        loop: bool = False,
    ):
        self.joint_positions = joint_positions
        self.model_hz = model_hz
        self.chunk_size = chunk_size
        self.loop = loop
        self._index = 0

    def next_chunk(self) -> Optional[np.ndarray]:
        if self._index >= len(self.joint_positions):
            if not self.loop:
                return None
            self._index = 0

        end = min(self._index + self.chunk_size, len(self.joint_positions))
        chunk = self.joint_positions[self._index:end]
        self._index = end

        if len(chunk) < self.chunk_size:
            pad_count = self.chunk_size - len(chunk)
            pad = np.repeat(chunk[-1][None, :], pad_count, axis=0)
            chunk = np.vstack([chunk, pad])

        return chunk

    def stream(self, callback) -> None:
        interval = self.chunk_size / float(self.model_hz)
        while True:
            chunk = self.next_chunk()
            if chunk is None:
                break
            callback(chunk)
            time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="基于最新 libero_format 数据模拟 pi05 输出")
    parser.add_argument("--data-dir", type=str, default=None, help="数据根目录")
    parser.add_argument("--robot-ip", type=str, default="192.168.1.10")
    parser.add_argument("--control-hz", type=int, default=1000)
    parser.add_argument("--model-hz", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--vel-limit", type=float, default=2.0)
    parser.add_argument("--acc-limit", type=float, default=10.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只打印 chunk，不发送到控制器")
    parser.add_argument("--position-alpha", type=float, default=1.0)
    parser.add_argument("--velocity-alpha", type=float, default=1.0)
    parser.add_argument("--append-plan", action="store_true", help="将 chunk 追加到轨迹末尾")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    joint_positions, inferred_hz, npz_path = load_latest_episode(data_dir)
    model_hz = args.model_hz if args.model_hz is not None else inferred_hz

    logger.info(f"使用数据: {npz_path}")
    logger.info(f"轨迹长度: {len(joint_positions)} 步, 采样频率: {model_hz} Hz")

    generator = SimulatedChunkGenerator(
        joint_positions=joint_positions,
        model_hz=model_hz,
        chunk_size=args.chunk_size,
        loop=args.loop,
    )

    if args.dry_run:
        logger.info("dry-run 模式：仅打印 chunk 尺寸")
        generator.stream(lambda chunk: logger.info(f"chunk shape: {chunk.shape}"))
        return

    config = ControllerConfig(
        robot_ip=args.robot_ip,
        control_hz=args.control_hz,
        model_hz=model_hz,
        vel_limit=args.vel_limit,
        acc_limit=args.acc_limit,
        temporal_ensemble=False,
        chunk_starts_at_current=True,
        plan_mode="append" if args.append_plan else "replace",
        position_alpha=args.position_alpha,
        velocity_alpha=args.velocity_alpha,
        use_feedback_start=False,
    )

    controller = KinovaController(config)
    controller.start()

    try:
        generator.stream(controller.update_chunk)
    except KeyboardInterrupt:
        logger.info("收到中断信号，准备退出")
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
