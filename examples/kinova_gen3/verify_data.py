#!/usr/bin/env python3
"""
数据验证和可视化脚本

用于解包和验证采集的数据文件，生成可读的数据文件和视频文件。

功能：
    - 加载 LIBERO 格式的 npz 数据文件
    - 生成可读的数据摘要文件（JSON/文本格式）
    - 将双摄像头图像序列转换为视频文件
    - 显示数据统计信息

使用方式：
    python verify_data.py <npz_file_path>
    
示例：
    python verify_data.py data/General_manipulation_task_20260114_161043/libero_format/episode_001_libero_20260114_161101.npz
"""

import sys
import json
import numpy as np
from pathlib import Path
import cv2
from datetime import datetime

def load_data(npz_path):
    """加载 npz 数据文件"""
    print(f"正在加载数据文件: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return data

def verify_data_format(data):
    """验证数据格式并返回检查结果"""
    checks = {
        'required_fields': True,
        'states_format': False,
        'actions_format': False,
        'images_format': False,
        'data_consistency': False
    }
    
    required_fields = ['agent_images', 'wrist_images', 'states', 'actions', 'task']
    missing_fields = []
    
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
            checks['required_fields'] = False
    
    if missing_fields:
        print(f"❌ 缺失字段: {missing_fields}")
        return checks
    
    # 检查状态格式 (应该是 8D: 7个关节 + 1个夹爪)
    states = data['states']
    if len(states.shape) == 2 and states.shape[1] == 8:
        checks['states_format'] = True
        print(f"✅ 状态格式正确: {states.shape} (N, 8)")
    else:
        print(f"❌ 状态格式错误: {states.shape}, 期望 (N, 8)")
    
    # 检查动作格式 (应该是 7D)
    actions = data['actions']
    if len(actions.shape) == 2 and actions.shape[1] == 7:
        checks['actions_format'] = True
        print(f"✅ 动作格式正确: {actions.shape} (N, 7)")
    else:
        print(f"❌ 动作格式错误: {actions.shape}, 期望 (N, 7)")
    
    # 检查图像格式
    agent_images = data['agent_images']
    wrist_images = data['wrist_images']
    if (len(agent_images.shape) == 4 and agent_images.shape[1:3] == (256, 256) and
        len(wrist_images.shape) == 4 and wrist_images.shape[1:3] == (256, 256)):
        checks['images_format'] = True
        print(f"✅ 图像格式正确: agent_images {agent_images.shape}, wrist_images {wrist_images.shape}")
    else:
        print(f"❌ 图像格式错误: agent_images {agent_images.shape}, wrist_images {wrist_images.shape}")
    
    # 检查数据一致性
    lengths = {
        'states': len(states),
        'actions': len(actions),
        'agent_images': len(agent_images),
        'wrist_images': len(wrist_images)
    }
    if len(set(lengths.values())) == 1:
        checks['data_consistency'] = True
        print(f"✅ 数据长度一致: {list(lengths.values())[0]} 步")
    else:
        print(f"❌ 数据长度不一致: {lengths}")
    
    return checks

def save_readable_data(data, output_dir):
    """保存可读的数据文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n正在生成可读数据文件到: {output_dir}")
    
    # 1. 保存数据摘要 (JSON格式)
    summary = {
        'data_info': {
            'total_steps': len(data['states']),
            'states_shape': list(data['states'].shape),
            'actions_shape': list(data['actions'].shape),
            'agent_images_shape': list(data['agent_images'].shape),
            'wrist_images_shape': list(data['wrist_images'].shape),
        },
        'states_info': {
            'dtype': str(data['states'].dtype),
            'min': data['states'].min(axis=0).tolist(),
            'max': data['states'].max(axis=0).tolist(),
            'mean': data['states'].mean(axis=0).tolist(),
            'std': data['states'].std(axis=0).tolist(),
        },
        'actions_info': {
            'dtype': str(data['actions'].dtype),
            'min': data['actions'].min(axis=0).tolist(),
            'max': data['actions'].max(axis=0).tolist(),
            'mean': data['actions'].mean(axis=0).tolist(),
            'std': data['actions'].std(axis=0).tolist(),
        },
        'task': str(data['task']) if 'task' in data else 'N/A',
        'verification_timestamp': datetime.now().isoformat()
    }
    
    summary_path = output_dir / "data_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 数据摘要: {summary_path}")
    
    # 2. 保存状态数据 (CSV格式，便于查看)
    states_path = output_dir / "states.csv"
    np.savetxt(states_path, data['states'], delimiter=',', 
               header='joint_1,joint_2,joint_3,joint_4,joint_5,joint_6,joint_7,gripper',
               comments='', fmt='%.6f')
    print(f"  ✅ 状态数据: {states_path}")
    
    # 3. 保存动作数据 (CSV格式)
    actions_path = output_dir / "actions.csv"
    np.savetxt(actions_path, data['actions'], delimiter=',',
               header='action_1,action_2,action_3,action_4,action_5,action_6,action_7',
               comments='', fmt='%.6f')
    print(f"  ✅ 动作数据: {actions_path}")
    
    # 4. 保存前10帧的状态和动作样本 (文本格式，便于快速查看)
    sample_path = output_dir / "data_samples.txt"
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("数据样本 (前10帧)\n")
        f.write("=" * 80 + "\n\n")
        
        num_samples = min(10, len(data['states']))
        for i in range(num_samples):
            f.write(f"帧 {i}:\n")
            f.write(f"  状态 (8D): {data['states'][i]}\n")
            f.write(f"     - 关节角度 (7D): {data['states'][i][:7]}\n")
            f.write(f"     - 夹爪状态: {data['states'][i][7]}\n")
            f.write(f"  动作 (7D): {data['actions'][i]}\n")
            f.write("\n")
    
    print(f"  ✅ 数据样本: {sample_path}")
    
    # 5. 保存验证报告
    checks = verify_data_format(data)
    report_path = output_dir / "verification_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("数据格式验证报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"验证时间: {datetime.now().isoformat()}\n\n")
        
        f.write("检查项目:\n")
        f.write(f"  ✅ 必需字段: {'通过' if checks['required_fields'] else '失败'}\n")
        f.write(f"  ✅ 状态格式 (8D): {'通过' if checks['states_format'] else '失败'}\n")
        f.write(f"  ✅ 动作格式 (7D): {'通过' if checks['actions_format'] else '失败'}\n")
        f.write(f"  ✅ 图像格式: {'通过' if checks['images_format'] else '失败'}\n")
        f.write(f"  ✅ 数据一致性: {'通过' if checks['data_consistency'] else '失败'}\n\n")
        
        f.write("数据统计:\n")
        f.write(f"  总帧数: {len(data['states'])}\n")
        f.write(f"  状态维度: {data['states'].shape}\n")
        f.write(f"  动作维度: {data['actions'].shape}\n")
        f.write(f"  外部相机图像: {data['agent_images'].shape}\n")
        f.write(f"  腕部相机图像: {data['wrist_images'].shape}\n")
        
        f.write("\n状态统计 (8D: 7个关节 + 1个夹爪):\n")
        for i in range(8):
            if i < 7:
                f.write(f"  关节 {i+1}: min={data['states'][:, i].min():.4f}, "
                       f"max={data['states'][:, i].max():.4f}, "
                       f"mean={data['states'][:, i].mean():.4f}\n")
            else:
                f.write(f"  夹爪: min={data['states'][:, i].min():.4f}, "
                       f"max={data['states'][:, i].max():.4f}, "
                       f"mean={data['states'][:, i].mean():.4f}\n")
    
    print(f"  ✅ 验证报告: {report_path}")

def create_video_from_images(images, output_path, fps=30):
    """将图像序列转换为视频文件"""
    if len(images) == 0:
        print(f"  ⚠️  图像序列为空，跳过视频生成")
        return False
    
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for img in images:
        # 确保图像是 BGR 格式（OpenCV 使用 BGR）
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                # 假设输入是 RGB，转换为 BGR
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img
        else:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        out.write(img_bgr)
    
    out.release()
    return True

def save_videos(data, output_dir):
    """保存双摄像头视频文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n正在生成视频文件到: {output_dir}")
    
    agent_images = data['agent_images']
    wrist_images = data['wrist_images']
    
    # 计算帧率（假设采集频率是60Hz，但视频通常用30fps）
    fps = 60
    
    # 生成外部相机视频
    agent_video_path = output_dir / "agent_camera.mp4"
    print(f"  正在生成外部相机视频: {agent_video_path}")
    if create_video_from_images(agent_images, agent_video_path, fps):
        print(f"  ✅ 外部相机视频已保存: {agent_video_path}")
    else:
        print(f"  ❌ 外部相机视频生成失败")
    
    # 生成腕部相机视频
    wrist_video_path = output_dir / "wrist_camera.mp4"
    print(f"  正在生成腕部相机视频: {wrist_video_path}")
    if create_video_from_images(wrist_images, wrist_video_path, fps):
        print(f"  ✅ 腕部相机视频已保存: {wrist_video_path}")
    else:
        print(f"  ❌ 腕部相机视频生成失败")
    
    # 生成并排对比视频（可选）
    if len(agent_images) > 0 and len(wrist_images) > 0:
        side_by_side_path = output_dir / "side_by_side_cameras.mp4"
        print(f"  正在生成并排对比视频: {side_by_side_path}")
        
        height, width = agent_images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(side_by_side_path), fourcc, fps, (width * 2, height))
        
        for agent_img, wrist_img in zip(agent_images, wrist_images):
            # 转换为 BGR
            agent_bgr = cv2.cvtColor(agent_img, cv2.COLOR_RGB2BGR)
            wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
            
            # 并排拼接
            combined = np.hstack([agent_bgr, wrist_bgr])
            out.write(combined)
        
        out.release()
        print(f"  ✅ 并排对比视频已保存: {side_by_side_path}")

def main():
    if len(sys.argv) < 2:
        print("用法: python verify_data.py <npz_file_path>")
        print("\n示例:")
        print("  python verify_data.py data/General_manipulation_task_20260114_161043/libero_format/episode_001_libero_20260114_161101.npz")
        sys.exit(1)
    
    npz_path = Path(sys.argv[1])
    
    if not npz_path.exists():
        print(f"❌ 错误: 文件不存在: {npz_path}")
        sys.exit(1)
    
    # 加载数据
    data = load_data(npz_path)
    
    # 验证数据格式
    print("\n" + "=" * 80)
    print("数据格式验证")
    print("=" * 80)
    checks = verify_data_format(data)
    
    # 确定输出目录（与npz文件同一目录）
    output_dir = npz_path.parent / f"{npz_path.stem}_verification"
    
    print("\n" + "=" * 80)
    print("生成验证文件")
    print("=" * 80)
    
    # 保存可读数据文件
    save_readable_data(data, output_dir)
    
    # 保存视频文件
    save_videos(data, output_dir)
    
    print("\n" + "=" * 80)
    print("验证完成!")
    print("=" * 80)
    print(f"所有文件已保存到: {output_dir}")
    print("\n生成的文件:")
    print(f"  - data_summary.json: 数据摘要（JSON格式）")
    print(f"  - states.csv: 状态数据（CSV格式）")
    print(f"  - actions.csv: 动作数据（CSV格式）")
    print(f"  - data_samples.txt: 数据样本（文本格式）")
    print(f"  - verification_report.txt: 验证报告")
    print(f"  - agent_camera.mp4: 外部相机视频")
    print(f"  - wrist_camera.mp4: 腕部相机视频")
    print(f"  - side_by_side_cameras.mp4: 并排对比视频")

if __name__ == "__main__":
    main()
