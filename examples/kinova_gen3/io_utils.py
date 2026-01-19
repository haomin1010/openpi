# ruff: noqa
"""输入输出相关工具函数。"""

import os
from typing import Optional

import numpy as np
from PIL import Image


def _extract_observation(
    obs_dict: dict,
    *,
    external_camera_serial: Optional[str],
    wrist_camera_serial: Optional[str],
    save_to_disk: bool = False,
) -> dict:
    """
    从机器人观察中提取所需数据。

    Args:
        obs_dict: KinovaRobotEnv.get_observation() 返回的字典
        external_camera_serial: 外部相机序列号
        wrist_camera_serial: 腕部相机序列号
        save_to_disk: 是否保存图像到磁盘

    Returns:
        dict: 提取后的观察数据
    """
    image_dict = obs_dict["image"]
    robot_state = obs_dict["robot_state"]

    # 获取图像（根据相机序列号）
    external_image = None
    wrist_image = None

    for key, img in image_dict.items():
        if external_camera_serial and external_camera_serial in key:
            external_image = img
        elif wrist_camera_serial and wrist_camera_serial in key:
            wrist_image = img

    # 如果未通过序列号匹配，尝试按顺序获取
    if external_image is None or wrist_image is None:
        images = list(image_dict.values())
        print(f"len(images): {len(images)}")
        if len(images) >= 2:
            external_image = images[0]
            wrist_image = images[1]
        elif len(images) == 1:
            external_image = images[0]
            wrist_image = images[0]  # 使用同一图像作为 fallback

    # 提取机器人状态
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # 保存组合图像到磁盘
    if save_to_disk and external_image is not None and wrist_image is not None:
        # 创建组合图像便于实时查看
        combined_image = np.concatenate([external_image, wrist_image], axis=1)
        combined_pil = Image.fromarray(combined_image)
        combined_pil.save("robot_camera_views.png")
        print("已保存相机图像到 robot_camera_views.png")

    return {
        "external_image": external_image,
        "wrist_image": wrist_image,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


def _save_video(frames: list, timestamp: str):
    """保存视频"""
    try:
        from moviepy.editor import ImageSequenceClip

        os.makedirs("videos", exist_ok=True)
        filename = f"videos/rollout_{timestamp}.mp4"

        video_array = np.stack(frames)
        clip = ImageSequenceClip(list(video_array), fps=10)
        clip.write_videofile(filename, codec="libx264", verbose=False, logger=None)
        print(f"视频已保存: {filename}")
    except ImportError:
        print("moviepy 未安装，跳过视频保存。安装: pip install moviepy")
    except Exception as e:
        print(f"保存视频失败: {e}")
