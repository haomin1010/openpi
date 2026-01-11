"""
RealSense D435i 双相机封装

支持通过序列号区分外部相机和腕部相机，提供统一的图像获取接口。

使用示例：
    from realsense_camera import DualRealSenseCamera

    cameras = DualRealSenseCamera(
        external_serial="123456789",  # 外部相机序列号
        wrist_serial="987654321",     # 腕部相机序列号
    )
    
    external_img, wrist_img = cameras.get_frames()
    # 返回 (H, W, 3) RGB uint8 格式的图像
    
    cameras.close()
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    raise ImportError(
        "pyrealsense2 未安装。请运行: pip install pyrealsense2"
    )

logger = logging.getLogger(__name__)


class RealSenseCamera:
    """单个 RealSense D435i 相机封装"""

    def __init__(
        self,
        serial_number: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        """
        初始化 RealSense 相机。

        Args:
            serial_number: 相机序列号。如果为 None，则连接第一个可用的相机。
            width: 图像宽度
            height: 图像高度
            fps: 帧率
        """
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps

        self._pipeline = rs.pipeline()
        self._config = rs.config()

        # 如果指定了序列号，则启用特定设备
        if serial_number:
            self._config.enable_device(serial_number)

        # 配置 RGB 流
        self._config.enable_stream(
            rs.stream.color, width, height, rs.format.rgb8, fps
        )

        # 启动相机
        try:
            self._profile = self._pipeline.start(self._config)
            device = self._profile.get_device()
            self._device_serial = device.get_info(rs.camera_info.serial_number)
            logger.info(f"RealSense 相机已启动: {self._device_serial}")
        except Exception as e:
            raise RuntimeError(f"无法启动 RealSense 相机 (serial={serial_number}): {e}")

        # 等待自动曝光稳定
        for _ in range(30):
            self._pipeline.wait_for_frames()

    @property
    def device_serial(self) -> str:
        """返回实际连接的设备序列号"""
        return self._device_serial

    def get_color_frame(self) -> np.ndarray:
        """
        获取一帧 RGB 图像。

        Returns:
            np.ndarray: (H, W, 3) uint8 格式的 RGB 图像
        """
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            raise RuntimeError("无法获取彩色图像帧")

        # 转换为 numpy 数组
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def close(self):
        """停止相机并释放资源"""
        try:
            self._pipeline.stop()
            logger.info(f"RealSense 相机已停止: {self._device_serial}")
        except Exception as e:
            logger.warning(f"停止相机时出错: {e}")


class DualRealSenseCamera:
    """双 RealSense D435i 相机管理器"""

    def __init__(
        self,
        external_serial: Optional[str] = None,
        wrist_serial: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        """
        初始化双相机系统。

        Args:
            external_serial: 外部相机（左侧视角）的序列号
            wrist_serial: 腕部相机的序列号
            width: 图像宽度
            height: 图像高度
            fps: 帧率

        注意：
            如果两个序列号都为 None，会尝试自动连接前两个可用的相机。
            建议明确指定序列号以确保相机分配正确。
        """
        self.width = width
        self.height = height

        # 获取所有连接的 RealSense 设备
        ctx = rs.context()
        devices = ctx.query_devices()
        available_serials = [d.get_info(rs.camera_info.serial_number) for d in devices]

        if len(available_serials) < 2:
            raise RuntimeError(
                f"需要 2 个 RealSense 相机，但只检测到 {len(available_serials)} 个。"
                f"可用设备: {available_serials}"
            )

        logger.info(f"检测到 RealSense 设备: {available_serials}")

        # 如果未指定序列号，自动分配
        if external_serial is None:
            external_serial = available_serials[0]
            logger.warning(f"未指定外部相机序列号，自动使用: {external_serial}")
        if wrist_serial is None:
            wrist_serial = available_serials[1] if len(available_serials) > 1 else available_serials[0]
            logger.warning(f"未指定腕部相机序列号，自动使用: {wrist_serial}")

        # 验证序列号
        if external_serial not in available_serials:
            raise ValueError(f"外部相机序列号 {external_serial} 不在可用设备列表中: {available_serials}")
        if wrist_serial not in available_serials:
            raise ValueError(f"腕部相机序列号 {wrist_serial} 不在可用设备列表中: {available_serials}")
        if external_serial == wrist_serial:
            raise ValueError("外部相机和腕部相机不能使用相同的序列号")

        # 初始化两个相机
        self._external_camera = RealSenseCamera(
            serial_number=external_serial, width=width, height=height, fps=fps
        )
        self._wrist_camera = RealSenseCamera(
            serial_number=wrist_serial, width=width, height=height, fps=fps
        )

        logger.info(
            f"双相机系统已初始化: 外部={external_serial}, 腕部={wrist_serial}"
        )

    def get_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """
        获取两个相机的 RGB 图像。

        Returns:
            tuple: (external_image, wrist_image)
                - external_image: 外部相机图像 (H, W, 3) uint8
                - wrist_image: 腕部相机图像 (H, W, 3) uint8
        """
        external_img = self._external_camera.get_color_frame()
        wrist_img = self._wrist_camera.get_color_frame()
        return external_img, wrist_img

    def get_external_frame(self) -> np.ndarray:
        """获取外部相机图像"""
        return self._external_camera.get_color_frame()

    def get_wrist_frame(self) -> np.ndarray:
        """获取腕部相机图像"""
        return self._wrist_camera.get_color_frame()

    def close(self):
        """停止所有相机并释放资源"""
        self._external_camera.close()
        self._wrist_camera.close()
        logger.info("双相机系统已关闭")


def list_connected_cameras() -> list[str]:
    """
    列出所有连接的 RealSense 相机序列号。

    Returns:
        list[str]: 相机序列号列表
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = []
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        serials.append(serial)
        print(f"  - {name}: {serial}")
    return serials


if __name__ == "__main__":
    # 测试代码：列出相机并尝试获取图像
    import argparse

    parser = argparse.ArgumentParser(description="RealSense 相机测试")
    parser.add_argument("--list", action="store_true", help="列出所有连接的相机")
    parser.add_argument("--external-serial", type=str, help="外部相机序列号")
    parser.add_argument("--wrist-serial", type=str, help="腕部相机序列号")
    parser.add_argument("--save", action="store_true", help="保存测试图像")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.list:
        print("检测到的 RealSense 相机:")
        list_connected_cameras()
    else:
        print("初始化双相机系统...")
        cameras = DualRealSenseCamera(
            external_serial=args.external_serial,
            wrist_serial=args.wrist_serial,
        )

        print("获取图像帧...")
        external_img, wrist_img = cameras.get_frames()
        print(f"外部相机图像: {external_img.shape}, dtype={external_img.dtype}")
        print(f"腕部相机图像: {wrist_img.shape}, dtype={wrist_img.dtype}")

        if args.save:
            from PIL import Image

            Image.fromarray(external_img).save("external_camera_test.png")
            Image.fromarray(wrist_img).save("wrist_camera_test.png")
            print("测试图像已保存")

        cameras.close()

