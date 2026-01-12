# ruff: noqa
"""
Kinova Gen3 机器人 OpenPI 策略推理脚本

使用 openpi 服务器进行 VLA 策略推理，控制 Kinova Gen3 机械臂执行任务。

使用方法：
    # 1. 在有 GPU 的机器上启动策略服务器
    uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_base --policy.dir=<checkpoint_path>

    # 2. 在机器人控制电脑上运行此脚本
    python main.py --robot-ip 192.168.1.10 --remote-host <server_ip> --external-serial <cam1> --wrist-serial <cam2>
"""

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image
import tqdm
import tyro

from kinova_env import KinovaRobotEnv, ActionMode

faulthandler.enable()

# 控制频率
CONTROL_FREQUENCY = 15  # Hz


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
    external_camera_serial: str | None = None  # 外部相机（左侧视角）
    wrist_camera_serial: str | None = None     # 腕部相机

    # =========================================================================
    # 策略服务器配置
    # =========================================================================
    remote_host: str = "0.0.0.0"  # 策略服务器 IP
    remote_port: int = 8000       # 策略服务器端口

    # =========================================================================
    # 推理配置
    # =========================================================================
    max_timesteps: int = 600  # 最大时间步数
    
    # 从预测的 action chunk 中执行多少个动作后再查询服务器
    # 8 通常是个好默认值（约 0.5 秒的动作执行）
    open_loop_horizon: int = 8
    
    # 动作模式：
    # - delta: 增量模式，action 是相对当前位置的增量（默认，与 openpi 输出兼容）
    # - absolute: 绝对位置模式
    # - velocity: 速度模式
    action_mode: str = "delta"


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


def main(args: Args):
    # 检查相机序列号
    if args.external_camera_serial is None or args.wrist_camera_serial is None:
        print("警告: 未指定相机序列号，将尝试自动检测")
        print("建议使用 --external-serial 和 --wrist-serial 明确指定")

    # 初始化机器人环境
    print(f"连接 Kinova 机械臂: {args.robot_ip}")
    print(f"连接夹爪控制器: {args.gripper_ip}")
    print(f"动作模式: {args.action_mode}")
    env = KinovaRobotEnv(
        robot_ip=args.robot_ip,
        gripper_ip=args.gripper_ip,
        external_camera_serial=args.external_camera_serial,
        wrist_camera_serial=args.wrist_camera_serial,
        action_mode=args.action_mode,
    )
    print("机器人环境创建成功!")

    # 连接策略服务器
    print(f"连接策略服务器: {args.remote_host}:{args.remote_port}")
    policy_client = websocket_client_policy.WebsocketClientPolicy(
        args.remote_host, args.remote_port
    )
    print("策略服务器连接成功!")

    # 用于保存视频的列表
    try:
        while True:
            instruction = input("\n输入任务指令 (或 'q' 退出): ")
            if instruction.lower() == 'q':
                break

            # Rollout 参数
            actions_from_chunk_completed = 0
            pred_action_chunk = None

            # 准备保存视频
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
            video_frames = []

            bar = tqdm.tqdm(range(args.max_timesteps), desc="执行中")
            print("开始执行... 按 Ctrl+C 或 ESC/q 键停止")

            for t_step in bar:
                start_time = time.time()
                try:
                    # 获取当前观察
                    curr_obs = env.get_observation()

                    # 提取观察数据
                    obs_data = _extract_observation(args, curr_obs, save_to_disk=(t_step == 0))

                    # 保存外部相机图像用于视频
                    video_frames.append(obs_data["external_image"])

                    # 如果需要，查询策略服务器获取新的动作 chunk
                    if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                        actions_from_chunk_completed = 0

                        # 准备请求数据（图像需要 resize 到 224x224）
                        request_data = {
                            "observation/exterior_image_1_left": image_tools.resize_with_pad(
                                obs_data["external_image"], 224, 224
                            ),
                            "observation/wrist_image_left": image_tools.resize_with_pad(
                                obs_data["wrist_image"], 224, 224
                            ),
                            "observation/joint_position": obs_data["joint_position"],
                            "observation/gripper_position": obs_data["gripper_position"],
                            "prompt": instruction,
                        }

                        # 使用上下文管理器防止 Ctrl+C 中断服务器调用
                        with prevent_keyboard_interrupt():
                            pred_action_chunk = policy_client.infer(request_data)["actions"]
                        
                        # 验证动作格式 [chunk_size, action_dim]
                        assert pred_action_chunk.shape[1] == 8, \
                            f"期望动作维度为 8，但收到 {pred_action_chunk.shape[1]}"

                    # 从 chunk 中选择当前要执行的动作
                    action = pred_action_chunk[actions_from_chunk_completed]
                    actions_from_chunk_completed += 1

                    # 二值化夹爪动作
                    if action[-1] > 0.5:
                        action = np.concatenate([action[:-1], np.ones((1,))])
                    else:
                        action = np.concatenate([action[:-1], np.zeros((1,))])

                    # 执行动作
                    env.step(action)

                    # 等待以匹配控制频率
                    elapsed_time = time.time() - start_time
                    if elapsed_time < 1 / CONTROL_FREQUENCY:
                        time.sleep(1 / CONTROL_FREQUENCY - elapsed_time)

                except KeyboardInterrupt:
                    print("\n用户中断")
                    break

            # 保存视频
            if video_frames:
                _save_video(video_frames, timestamp)

            # 询问结果
            success = input("任务是否成功? (y/n): ")
            print(f"结果已记录: {'成功' if success.lower() == 'y' else '失败'}")

            # 询问是否继续
            if input("继续下一个任务? (y/n): ").lower() != 'y':
                break

            # 重置机器人
            print("重置机器人...")
            env.reset()

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("程序结束")


def _extract_observation(args: Args, obs_dict: dict, *, save_to_disk: bool = False) -> dict:
    """
    从机器人观察中提取所需数据。

    Args:
        args: 命令行参数
        obs_dict: KinovaRobotEnv.get_observation() 返回的字典
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
        if args.external_camera_serial and args.external_camera_serial in key:
            external_image = img
        elif args.wrist_camera_serial and args.wrist_camera_serial in key:
            wrist_image = img

    # 如果未通过序列号匹配，尝试按顺序获取
    if external_image is None or wrist_image is None:
        images = list(image_dict.values())
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


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)

