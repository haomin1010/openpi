"""
Kinova Gen3 数据集转换脚本

将 collect_data.py 生成的 .npz 文件转换为 LeRobot 数据集格式，
以便在 OpenPi 中进行训练。

用法：
    uv run examples/kinova_gen3/convert_to_lerobot.py --data-dir /path/to/your/data/libero_format

依赖：
    需要安装 lerobot 和 tensorflow_datasets (如果使用 RLDS)
    但这里我们直接使用 LeRobotDataset API
"""

import argparse
import logging
from pathlib import Path
import shutil
import numpy as np
import tyro
from typing import Optional

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("Converter")

def main(
    data_dir: Path,
    repo_name: str = "kinova_gen3_dataset",
    force_override: bool = False,
    push_to_hub: bool = False,
    hub_username: Optional[str] = None,
):
    """
    将 Kinova 采集的 .npz 数据转换为 LeRobot 格式。

    Args:
        data_dir: 包含 .npz 文件的目录 (通常是 .../libero_format)
        repo_name: 输出数据集的名称 (HF_LEROBOT_HOME/repo_name)
        force_override: 是否覆盖已存在的输出数据集
        push_to_hub: 是否上传到 Hugging Face Hub
        hub_username: HF 用户名 (用于推送到 Hub)
    """
    
    # 1. 检查输入目录
    if not data_dir.exists():
        logger.error(f"输入目录不存在: {data_dir}")
        return
        
    npz_files = sorted(list(data_dir.glob("*_libero_*.npz")))
    if not npz_files:
        logger.error(f"在 {data_dir} 中未找到符合 *_libero_*.npz 模式的文件")
        return
        
    logger.info(f"找到 {len(npz_files)} 个数据文件")

    # 2. 准备输出路径
    if hub_username:
        full_repo_id = f"{hub_username}/{repo_name}"
    else:
        full_repo_id = repo_name
        
    output_path = HF_LEROBOT_HOME / full_repo_id
    
    if output_path.exists():
        if force_override:
            logger.warning(f"删除现有数据集: {output_path}")
            shutil.rmtree(output_path)
        else:
            logger.error(f"输出路径已存在: {output_path}. 使用 --force-override 覆盖。")
            return

    # 3. 创建 LeRobot 数据集
    # 定义特征结构，与 collect_data.py 中的保存格式对应
    # agent_images: (256, 256, 3)
    # wrist_images: (256, 256, 3)
    # states: (8,) -> [x, y, z, qx, qy, qz, qw, gripper]
    # actions: (7,) -> [vx, vy, vz, wx, wy, wz, gripper_cmd]
    
    dataset = LeRobotDataset.create(
        repo_id=full_repo_id,
        robot_type="kinova_gen3",
        fps=60,  # collect_data.py 默认 60Hz，请根据实际情况确认
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"], # 具体含义见上文注释
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 4. 处理每个 episode
    total_frames = 0
    
    for npz_file in npz_files:
        try:
            logger.info(f"正在处理: {npz_file.name}")
            data = np.load(npz_file, allow_pickle=True)
            
            # 读取数据
            # 注意：collect_data.py 保存的是 keys: agent_images, wrist_images, states, actions, task
            agent_images = data['agent_images']
            wrist_images = data['wrist_images']
            states = data['states']
            actions = data['actions']
            task = str(data['task'])
            
            num_frames = len(states)
            total_frames += num_frames
            
            for i in range(num_frames):
                dataset.add_frame({
                    "image": agent_images[i],
                    "wrist_image": wrist_images[i],
                    "state": states[i],
                    "actions": actions[i],
                    "task": task,
                })
            
            # 保存一个完整的 episode
            dataset.save_episode()
            
        except Exception as e:
            logger.error(f"处理文件 {npz_file} 时出错: {e}")
            # 可以选择 continue 跳过或 return 停止，这里选择继续
            continue

    logger.info(f"转换完成！共处理 {len(npz_files)} 个 episode, {total_frames} 帧。")
    logger.info(f"数据集保存在: {output_path}")

    # 5. (可选) 上传到 Hub
    if push_to_hub:
        logger.info("正在推送到 Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["kinova", "openpi", "lerobot"],
            private=True, # 默认私有
            push_videos=True,
        )
        logger.info("推送完成！")

if __name__ == "__main__":
    tyro.cli(main)
