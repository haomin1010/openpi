#!/usr/bin/env python3
"""
Kinova 机械臂数据收集脚本

基于 KinovaRobotEnv 实现的数据收集系统，用于收集真机演示数据，
保持与 LIBERO 训练格式一致。支持键盘交互控制、实时数据采集、
增量保存和完整 episode 保存。

主要功能：
    - 实时数据采集（60Hz）
    - LIBERO 格式数据保存
    - 轨迹回放数据保存
    - 键盘交互控制（开始/停止录制、夹爪控制、机器人复位等）
    - 增量数据保存（防止数据丢失）

数据格式：
    - LIBERO 格式：agent_images, wrist_images, states (8D), actions (7D)
    - 回放格式：joint_positions (7D), gripper_pos, eef_pose, timestamp, action
    - 注意：gripper_pos 是二值状态（0.0=张开，1.0=闭合），不是连续的归一化角度值

使用方式：
    运行脚本后，使用键盘控制：
    - Enter: 开始/停止录制
    - O: 张开夹爪
    - P: 闭合夹爪
    - R: 复位机器人
    - H: 显示状态
    - ESC: 退出程序

依赖：
    - kinova_env: KinovaRobotEnv 机器人环境
    - kortex_api: Kinova 机器人 API
    - pyrealsense2: RealSense 相机支持
    - opencv-python: 图像处理
    - scipy: 四元数转换
"""

import sys
import os

# 设置 protobuf 环境变量以兼容 kortex_api（必须在导入 kortex_api 之前）
# kortex_api 需要 protobuf <= 3.20.x，但项目可能使用更新版本
# 这个设置使用纯 Python 实现，性能较慢但兼容性更好
if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" not in os.environ:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import time
import json
import datetime
import numpy as np
from pathlib import Path
import threading
import queue
import termios
import tty
import logging

# 导入本地模块
from kinova_env import KinovaRobotEnv, ActionMode
from kortex_api.autogen.messages import Base_pb2

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("DataCollector")

class LiberoDataCollector:
    """
    LIBERO 格式兼容的数据收集器
    
    负责从 Kinova 机械臂收集演示数据，包括：
    - 双相机图像（外部相机和腕部相机）
    - 机器人状态（关节位置、末端位姿、夹爪状态）
    - 动作数据（用于训练）
    - 回放数据（用于轨迹复现）
    
    数据保存格式：
        - LIBERO 格式：用于训练的数据格式
        - 回放格式：包含完整关节位置和位姿信息，用于轨迹回放
    
    属性：
        robot_ip (str): Kinova 机械臂 IP 地址
        gripper_ip (str): Arduino 夹爪控制器 IP 地址
        task_description (str): 任务描述
        num_demonstrations (int): 需要收集的演示数量
        data_dir (str): 数据保存目录
        collection_frequency (int): 数据采集频率（Hz）
        env (KinovaRobotEnv): 机器人环境实例
        is_recording (bool): 是否正在录制
        episode_count (int): 当前已收集的 episode 数量
    """
    
    def __init__(self):
        """
        初始化数据收集器
        
        执行以下操作：
        1. 设置默认参数（机器人 IP、夹爪 IP、任务描述等）
        2. 初始化 Kinova 机器人环境
        3. 创建数据保存目录结构
        4. 启动键盘监听线程
        5. 打印使用说明
        """
        # 参数设置
        self.robot_ip = "192.168.1.10"
        self.gripper_ip = "192.168.1.43"
        self.task_description = "General manipulation task"
        self.num_demonstrations = 10
        # 数据保存目录：相对于脚本文件所在目录
        script_dir = Path(__file__).parent
        self.data_dir = str(script_dir / "data")
        self.save_replay_data = True
        
        # 采集频率 (Hz)
        self.collection_frequency = 60
        
        # 外部相机序列号（左侧）
        self.external_camera_serial = None
        # 腕部相机序列号
        self.wrist_camera_serial = None
        
        # 初始化 Kinova 环境
        try:
            logger.info(f"Connecting to robot at {self.robot_ip}...\n")
            self.env = KinovaRobotEnv(
                robot_ip=self.robot_ip,
                gripper_ip=self.gripper_ip,
                external_camera_serial=self.external_camera_serial,
                wrist_camera_serial=self.wrist_camera_serial,
                action_mode=ActionMode.DELTA # 默认模式，实际上我们会用底层指令覆盖
            )
            logger.info("✅ Robot environment initialized successfully\n")
        except Exception as e:
            logger.error(f"Failed to initialize robot environment: {e}\n")
            sys.exit(1)
            
        # 数据收集状态
        self.is_recording = False
        self.episode_count = 0
        self.episode_data = None
        
        # 脉冲移动状态
        self.pulse_movement_end_time = 0.0
        self.waiting_for_movement = False
        self.last_executed_action = np.zeros(7)
        
        # 目录设置
        self.setup_directories()
        
        # 键盘控制
        self.key_queue = queue.Queue()
        self.running = True
        # 保存原始终端状态，用于程序退出时恢复
        self.original_termios = None
        try:
            self.original_termios = termios.tcgetattr(sys.stdin.fileno())
        except:
            pass
        self.start_keyboard_listener()
        
        # 打印说明
        self.print_instructions()
        
    def get_logger(self):
        """
        获取日志记录器
        
        Returns:
            logging.Logger: 日志记录器实例
        """
        return logger

    def setup_directories(self):
        """
        设置数据目录结构
        
        创建以下目录结构：
            data/
            └── {task_name}_{timestamp}/
                ├── libero_format/          # LIBERO 格式数据
                ├── replay_data/           # 回放数据
                └── session_info.json      # 会话信息
        
        同时保存会话信息到 JSON 文件，包括任务描述、机器人配置等。
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = self.task_description.replace(" ", "_").strip('""')
        
        # 主目录
        self.session_dir = Path(self.data_dir) / f"{task_name}_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # LIBERO格式数据目录
        self.libero_dir = self.session_dir / "libero_format"
        self.libero_dir.mkdir(exist_ok=True)
        
        # 回放数据目录
        if self.save_replay_data:
            self.replay_dir = self.session_dir / "replay_data"
            self.replay_dir.mkdir(exist_ok=True)
            
        # 保存会话信息
        session_info = {
            'task_description': self.task_description,
            'robot_ip': self.robot_ip,
            'gripper_ip': self.gripper_ip,
            'num_demonstrations': self.num_demonstrations,
            'timestamp': timestamp,
            'format': 'LIBERO-compatible'
        }
        
        info_path = self.session_dir / "session_info.json"
        with open(info_path, 'w') as f:
            json.dump(session_info, f, indent=2)

    def start_recording(self):
        """
        开始录制新的 episode
        
        执行以下操作：
        1. 检查是否已达到最大演示数量
        2. 初始化数据结构（图像、状态、动作等）
        3. 启动数据采集线程（固定频率采集）
        
        注意：
            - 如果已达到最大演示数量，将不会开始新的录制
            - 采集线程以 collection_frequency (默认 60Hz) 的频率运行
        """
        if self.episode_count >= self.num_demonstrations:
            logger.warning(f"已收集完所有演示 ({self.num_demonstrations})\n")
            return

        logger.info("\n🎬 录制开始! (Recording started)\n")
        
        self.step_count = 0
        self.recording_start_time = time.time()
        
        # 初始化数据结构
        self.continuous_episode_data = {
            'agent_images': [],      # 外部相机
            'wrist_images': [],      # 腕部相机
            'states': [],            # 8D状态
            'actions': [],           # 7D动作
            'task': self.task_description,
            'replay_data': []
        }
        self.is_recording = True
        
        # 启动采集线程
        self.collect_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collect_thread.start()

    def _collection_loop(self):
        """
        固定频率数据采集循环（后台线程）
        
        以 collection_frequency (默认 60Hz) 的频率持续采集数据，
        直到 is_recording 标志为 False。
        
        采集内容：
            - 相机图像（外部相机和腕部相机）
            - 机器人状态（关节位置、末端位姿、夹爪状态）
            - 动作数据（last_executed_action）
        
        注意：
            - 使用精确的时间控制确保采集频率稳定
            - 如果采集速度过慢，会跳过一些帧以赶上进度
        """
        interval = 1.0 / self.collection_frequency
        logger.info(f"Starting data collection at {self.collection_frequency} Hz\n")
        
        next_time = time.time()
        
        while self.is_recording:
            # 记录当前动作（如果没有按键，则为全0）
            # 注意：在示教模式下，我们通常不记录键盘动作，而是记录机械臂的实际状态作为动作（如果是闭环）
            # 但在模仿学习数据采集中，如果是遥操作，action 是用户的输入。
            # 如果是手动拖动示教，action 通常是 下一时刻状态 - 当前状态 (delta) 或者 实际速度
            # 这里我们简单记录 last_executed_action，如果是手动拖动，这个值可能一直是0
            # TODO: 如果是手动拖动，这里的 action 可能需要改为记录实际关节速度/末端速度
            
            # 为了兼容性，我们暂且记录当前的 last_executed_action
            # 或者是记录实际的机械臂反馈速度？KinovaRobotEnv 的 get_observation 返回了状态
            # 我们在 collect_step_data 中处理
            
            self.collect_step_data(self.last_executed_action)
            
            # 频率控制
            next_time += interval
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 采集太慢，跳过一些帧以赶上进度
                # logger.warning(f"Collection lag: {-sleep_time*1000:.1f} ms")
                pass

    def stop_recording_and_save(self):
        """
        停止录制并保存数据
        
        执行以下操作：
        1. 停止录制标志，等待采集线程结束
        2. 停止机器人运动
        3. 验证数据长度（至少 5 步）
        4. 保存完整的 episode 数据
        5. 更新 episode 计数
        6. 如果达到最大演示数量，创建摘要文件
        
        注意：
            - 如果录制时间太短（< 5 步），将不会保存数据
            - 保存的数据包括 LIBERO 格式和回放格式
        """
        logger.info("\n⏹️ 录制停止 (Recording stopped)\n")
        self.is_recording = False
        
        # 等待采集线程结束
        if hasattr(self, 'collect_thread') and self.collect_thread.is_alive():
            self.collect_thread.join(timeout=1.0)
        
        # 停止机械臂
        self.stop_robot()
        
        # 检查数据长度
        if len(self.continuous_episode_data['states']) < 5:
            logger.warning("录制时间太短 (< 5 steps)，不保存\n")
            return
            
        success = self.save_complete_episode()
        
        if success:
            remaining = self.num_demonstrations - self.episode_count
            logger.info(f"\n📋 Episode saved! Remaining: {remaining}/{self.num_demonstrations}\n")
            
            if self.episode_count >= self.num_demonstrations:
                logger.info(f"\n🎉 所有 {self.num_demonstrations} 条演示已收集完毕!\n")
                self.create_summary()

    def collect_step_data(self, action_7d):
        """
        收集单步数据
        
        从机器人环境获取当前观测，处理并保存为 LIBERO 格式。
        
        Args:
            action_7d: 7 维动作数组，格式为 [vel(6), gripper(1)]
                      注意：在手动拖动示教模式下，这个值可能一直是 0
        
        处理流程：
            1. 获取观测（图像和机器人状态）
            2. 调整图像大小到 256x256 (LIBERO 标准)
            3. 提取机器人状态（关节位置、末端位姿、夹爪状态）
            4. 转换笛卡尔位姿为四元数
            5. 构造 8D 状态 [eef_pos(3), eef_quat(4), gripper(1)]
            6. 保存到连续 episode 数据中
            7. 如果启用回放数据保存，同时保存回放数据
            8. 每 50 步执行一次增量保存
        
        数据格式说明：
            - LIBERO 状态：8D [x, y, z, qx, qy, qz, qw, gripper]
            - LIBERO 动作：7D [vel(6), gripper(1)]
            - 夹爪状态：二值动作（张开/闭合），LIBERO 格式 +1=张开, -1=闭合；env 格式 0.0=张开, 1.0=闭合
            - 注意：夹爪状态不是连续的归一化角度值，而是离散的张开/闭合两种动作状态
        """
        try:
            # 使用 env 获取观测
            obs = self.env.get_observation()
            
            self.step_count += 1
            
            # 提取图像
            # 注意：KinovaRobotEnv 返回的 image 字典 key 格式为 "{serial_number}_left"
            # 我们需要适配一下，假设第一个是外部相机，第二个是腕部相机
            imgs = list(obs['image'].values())
            # 简单假设：如果指定了 serial，就按 serial 找，否则按顺序
            # 如果相机未连接，使用黑色图像占位
            ext_img = imgs[0] if len(imgs) > 0 else np.zeros((256, 256, 3), dtype=np.uint8)
            wrist_img = imgs[1] if len(imgs) > 1 else np.zeros((256, 256, 3), dtype=np.uint8)
            
            # 调整图像大小到 256x256 (LIBERO 标准)
            # 注意: KinovaRobotEnv 获取的图像大小取决于其配置（默认 640x480）
            # LIBERO 格式要求图像大小为 256x256
            import cv2
            if ext_img.shape[:2] != (256, 256):
                ext_img = cv2.resize(ext_img, (256, 256))
            if wrist_img.shape[:2] != (256, 256):
                wrist_img = cv2.resize(wrist_img, (256, 256))

            # 提取机器人状态
            robot_state = obs['robot_state']
            joint_pos = robot_state['joint_positions']  # (7,) 关节位置数组（弧度）
            cart_pos = robot_state['cartesian_position']  # (6,) [x, y, z, theta_x, theta_y, theta_z] (弧度)
            gripper_pos = robot_state['gripper_position']  # (1,) 夹爪状态：0.0=张开，1.0=闭合（二值动作，非连续角度值）
            
            # 转换笛卡尔位姿：从欧拉角 [x,y,z,rx,ry,rz] 转为四元数 [x,y,z,qx,qy,qz,qw]
            # KinovaRobotEnv 返回的是欧拉角 rx,ry,rz (弧度)
            # LIBERO 格式需要四元数表示旋转
            from scipy.spatial.transform import Rotation
            r = Rotation.from_euler('xyz', cart_pos[3:], degrees=False)  # 从欧拉角创建旋转对象
            quat = r.as_quat()  # (4,) 四元数 [x, y, z, w]
            
            # 转换夹爪状态格式（二值动作：张开/闭合）
            # LIBERO 格式: +1=张开, -1=闭合
            # env 格式: 0.0=张开, 1.0=闭合（二值状态，非连续角度值）
            libero_gripper = 1.0 if gripper_pos < 0.5 else -1.0
            
            # 构造 8D 状态数组 [eef_pos(3), eef_quat(4), gripper(1)]
            # LIBERO 格式要求的状态表示
            state_8d = np.concatenate([
                cart_pos[:3],      # 位置 (x, y, z)
                quat,              # 四元数 (qx, qy, qz, qw)
                [libero_gripper]   # 夹爪状态
            ]).astype(np.float32)
            
            # 构造 7D 动作数组
            # 传入的 action_7d 格式为 [vel(6), gripper(1)]
            # 注意：在手动拖动示教模式下，这个值可能一直是 0
            action_final = np.array(action_7d).astype(np.float32)
            
            # 保存数据
            self.continuous_episode_data['states'].append(state_8d)
            self.continuous_episode_data['actions'].append(action_final)
            self.continuous_episode_data['agent_images'].append(ext_img)
            self.continuous_episode_data['wrist_images'].append(wrist_img)
            
            # 保存回放数据（用于轨迹回放）
            # 回放数据包含完整的关节位置和位姿信息，用于精确复现轨迹
            if self.save_replay_data:
                replay_data = {
                    'timestamp': time.time() - self.recording_start_time,  # 相对录制开始的时间戳
                    'step': self.step_count,  # 当前步数
                    'joint_positions': joint_pos,  # (7,) 关节位置（弧度）
                    'eef_pose': np.concatenate([cart_pos[:3], quat]),  # (7,) 末端执行器位姿 [x,y,z,qx,qy,qz,qw]
                    'gripper_pos': gripper_pos,  # (1,) 夹爪状态：0.0=张开，1.0=闭合（二值动作）
                    'action': action_final  # (7,) 动作数组
                }
                self.continuous_episode_data['replay_data'].append(replay_data)
                
            # 增量保存
            if self.step_count % 50 == 0:
                self.save_incremental_data()
                
        except Exception as e:
            logger.error(f"Failed to collect step data: {e}\n")

    def save_incremental_data(self):
        """
        增量保存数据（防止数据丢失）
        
        将当前已收集的数据保存为增量备份文件，文件名包含步数和时间戳。
        每 50 步自动调用一次。
        
        保存的数据包括：
            - agent_images: 外部相机图像 (N, 256, 256, 3) uint8
            - wrist_images: 腕部相机图像 (N, 256, 256, 3) uint8
            - states: 8D 状态数组 (N, 8) float32
            - actions: 7D 动作数组 (N, 7) float32
            - task: 任务描述字符串
            - step_count: 当前步数
        """
        try:
            if not self.continuous_episode_data or len(self.continuous_episode_data['states']) == 0:
                return
                
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            incremental_path = self.session_dir / f"incremental_data_step_{self.step_count}_{timestamp}.npz"
            
            np.savez_compressed(
                incremental_path,
                agent_images=np.asarray(self.continuous_episode_data['agent_images'], dtype=np.uint8),
                wrist_images=np.asarray(self.continuous_episode_data['wrist_images'], dtype=np.uint8),
                states=np.asarray(self.continuous_episode_data['states'], dtype=np.float32),
                actions=np.asarray(self.continuous_episode_data['actions'], dtype=np.float32),
                task=np.array(self.task_description),
                step_count=np.array(self.step_count)
            )
            logger.info(f"💾 Incremental data saved: {incremental_path.name}\n")
            sys.stdout.flush()  # 立即刷新，确保格式正确
        except Exception as e:
            logger.error(f"Failed to save incremental data: {e}\n")

    def save_complete_episode(self):
        """
        保存完整的 episode 数据
        
        将整个 episode 的数据保存为两个文件：
        1. LIBERO 格式文件：用于训练的数据格式
        2. 回放格式文件：包含完整关节位置和位姿，用于轨迹回放
        
        Returns:
            bool: 保存是否成功
        
        保存的文件：
            - libero_format/episode_{count:03d}_libero_{timestamp}.npz
            - replay_data/episode_{count:03d}_replay_{timestamp}.npz
        
        注意：
            - 保存前会验证 LIBERO 格式
            - 回放数据包含：joint_positions, eef_pose, gripper_pos, timestamp, step, action
            - gripper_pos 是二值状态（0.0=张开，1.0=闭合），记录的是张开/闭合动作，非连续角度值
        """
        try:
            self.episode_count += 1
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            libero_path = self.libero_dir / f"episode_{self.episode_count:03d}_libero_{timestamp}.npz"
            
            np.savez_compressed(
                libero_path,
                agent_images=np.asarray(self.continuous_episode_data['agent_images'], dtype=np.uint8),
                wrist_images=np.asarray(self.continuous_episode_data['wrist_images'], dtype=np.uint8),
                states=np.asarray(self.continuous_episode_data['states'], dtype=np.float32),
                actions=np.asarray(self.continuous_episode_data['actions'], dtype=np.float32),
                task=np.array(self.task_description)
            )
            
            # 验证
            self.validate_libero_format(str(libero_path))
            
            logger.info(f"Episode {self.episode_count} saved to: {libero_path.name}\n")
            
            # 保存回放数据（用于轨迹回放）
            # 将回放数据从 list of dicts 转换为 dict of lists，然后保存为 numpy 数组
            if self.save_replay_data and self.continuous_episode_data['replay_data']:
                replay_path = self.replay_dir / f"episode_{self.episode_count:03d}_replay_{timestamp}.npz"
                
                # 转换数据结构：从 list of dicts 转为 dict of lists
                # 例如：[{'joint_positions': [1,2,3]}, {'joint_positions': [4,5,6]}]
                # 转为：{'joint_positions': [[1,2,3], [4,5,6]]}
                replay_dict = {}
                for k in self.continuous_episode_data['replay_data'][0].keys():
                    replay_dict[k] = []
                for d in self.continuous_episode_data['replay_data']:
                    for k, v in d.items():
                        replay_dict[k].append(v)
                        
                # 转换为 numpy 数组并保存
                # 每个键对应一个 numpy 数组，例如：
                # - 'joint_positions': (N, 7) 数组
                # - 'gripper_pos': (N,) 数组
                saved_data = {k: np.array(v) for k, v in replay_dict.items()}
                # 保存采集频率（用于回放时使用正确的控制频率）
                saved_data['collection_frequency'] = np.array(self.collection_frequency)
                np.savez_compressed(str(replay_path), **saved_data)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to save episode: {e}\n")
            return False

    def validate_libero_format(self, file_path):
        """
        验证 LIBERO 格式数据文件
        
        Args:
            file_path: 数据文件路径
        
        Returns:
            bool: 验证是否通过
        
        检查的必需字段：
            - agent_images: 外部相机图像
            - wrist_images: 腕部相机图像
            - states: 8D 状态数组
            - actions: 7D 动作数组
            - task: 任务描述
        """
        try:
            data = np.load(file_path)
            required = ['agent_images', 'wrist_images', 'states', 'actions', 'task']
            for f in required:
                if f not in data:
                    logger.warn(f"Missing field: {f}\n")
                    return False
            logger.info("✅ LIBERO format validation passed\n")
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}\n")
            return False

    def create_summary(self):
        """
        创建数据收集摘要文件
        
        当所有演示收集完毕后，创建摘要 JSON 文件，包含：
            - task_description: 任务描述
            - total_episodes: 总 episode 数量
            - session_dir: 会话目录路径
            - timestamp: 完成时间戳
        
        文件保存为：session_dir/collection_summary.json
        """
        summary = {
            'task_description': self.task_description,
            'total_episodes': self.episode_count,
            'session_dir': str(self.session_dir),
            'timestamp': datetime.datetime.now().isoformat()
        }
        with open(self.session_dir / "collection_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

    def send_pulse_movement(self, movement_direction):
        """
        发送脉冲移动命令到机器人
        
        Args:
            movement_direction: 6D 移动方向数组 [x, y, z, rx, ry, rz]
                               - 正值表示正向移动
                               - 负值表示反向移动
                               - 0 表示不移动
        
        执行流程：
            1. 将方向转换为速度命令（位置步长 0.02m，旋转步长 1.0 度）
            2. 创建 TwistCommand 消息
            3. 发送到机器人（非阻塞）
            4. 启动延迟停止线程，在指定时间后自动停止
        
        注意：
            - 旋转移动持续时间更长（400ms vs 200ms）
            - 动作会被记录到 last_executed_action 中
        """
        if not self.env or not self.env._is_connected:
            return
            
        # 构造 7D 动作数组 [vel(6), gripper(1)]
        # 将方向转换为速度值：正值 -> 5.625，负值 -> -5.625，0 -> 0.0
        action_7d_scaled = []
        for val in movement_direction:
            if val > 0:
                action_7d_scaled.append(5.625)  # 正向速度
            elif val < 0:
                action_7d_scaled.append(-5.625)  # 负向速度
            else:
                action_7d_scaled.append(0.0)  # 不移动
                
        # 添加夹爪状态 (LIBERO 格式)
        # 从 env 获取当前夹爪状态（二值：0.0=张开，1.0=闭合），转换为 LIBERO 格式 (+1/-1)
        curr_grip = self.env._current_gripper_pos
        libero_grip = 1.0 if curr_grip < 0.5 else -1.0
        action_7d_scaled.append(libero_grip)
        
        # 记录本次动作（用于数据采集）
        # 确保 last_executed_action 数组长度为 7
        if len(self.last_executed_action) != 7:
            self.last_executed_action = np.zeros(7)
        self.last_executed_action[:] = action_7d_scaled
        
        # 执行物理移动（使用 TwistCommand）
        # TwistCommand 是 Kinova API 提供的笛卡尔空间速度控制命令
        try:
            # 定义移动步长
            pos_step = 0.02  # 位置移动步长（米）
            rot_step = 1.0   # 旋转移动步长（度）
            
            # 计算实际移动量
            direction = np.array(movement_direction)
            pos_delta = direction[0:3] * pos_step   # 位置增量 (x, y, z)
            rot_delta = direction[3:6] * rot_step   # 旋转增量 (rx, ry, rz)
            
            # 创建 TwistCommand 消息
            twist = Base_pb2.TwistCommand()
            twist.twist.linear_x = float(pos_delta[0])   # X 方向线速度
            twist.twist.linear_y = float(pos_delta[1])   # Y 方向线速度
            twist.twist.linear_z = float(pos_delta[2])   # Z 方向线速度
            twist.twist.angular_x = float(rot_delta[0])  # X 轴角速度
            twist.twist.angular_y = float(rot_delta[1])  # Y 轴角速度
            twist.twist.angular_z = float(rot_delta[2])  # Z 轴角速度
            twist.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE  # 基坐标系
            
            # 设置持续时间（旋转移动需要更长时间）
            duration = 400 if np.any(direction[3:6] != 0) else 200  # 毫秒
            twist.duration = duration
            
            # 使用 env 内部的 _base 客户端发送指令（非阻塞）
            self.env._base.SendTwistCommand(twist)
            
            # 启动延迟停止线程
            # 在指定时间后自动发送停止命令（duration=0 的 TwistCommand）
            wait_time = (duration + 20) / 1000.0  # 转换为秒，额外等待 20ms
            threading.Thread(target=self._delayed_stop, args=(wait_time,), daemon=True).start()
            
            # 移除这里的 collect_step_data，统一由 _collection_loop 处理
            # if self.is_recording:
            #     self.collect_step_data(action_7d_scaled)
            #     logger.info(f"Step recorded. Action: {action_7d_scaled}")
                
        except Exception as e:
            logger.error(f"Movement failed: {e}\n")

    def _delayed_stop(self, delay):
        """
        延迟停止机器人运动（后台线程）
        
        Args:
            delay: 延迟时间（秒）
        
        在指定延迟后发送停止命令（duration=0 的 TwistCommand）到机器人。
        用于实现脉冲移动的自动停止。
        """
        time.sleep(delay)
        try:
            stop = Base_pb2.TwistCommand()
            stop.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
            stop.duration = 0  # duration=0 表示停止
            if self.env and self.env._is_connected:
                self.env._base.SendTwistCommand(stop)
        except:
            pass
            
    def stop_robot(self):
        """
        立即停止机器人运动
        
        通过发送 duration=0 的 TwistCommand 来停止机器人的所有运动。
        """
        self._delayed_stop(0)

    def set_gripper_position(self, pos):
        """
        控制夹爪状态（张开/闭合）
        
        Args:
            pos: 夹爪状态，0.0=张开，1.0=闭合（二值动作，非连续角度值）
        
        注意：
            - 实际使用中只接受 0.0（张开）或 1.0（闭合）两个值
            - 使用机器人环境的内部方法控制夹爪
        """
        # 使用 env 的内部方法
        self.env._control_gripper(pos)

    def reset_robot(self):
        """
        复位机器人到初始位置
        
        将机器人移动到 home 位置，并张开夹爪。
        如果急停已触发，需要先清除急停状态。
        """
        self.env.reset()

    def handle_key_press(self, key):
        """
        处理键盘按键事件
        
        Args:
            key: 按键字符
        
        支持的按键：
            - Enter ('\r'): 开始/停止录制
            - 'o': 张开夹爪
            - 'p': 闭合夹爪
            - 'r': 复位机器人
            - 'h': 打印当前状态
            - ESC ('\x1b'): 退出程序
        """
        if key == '\r': # Enter
            if not self.is_recording:
                self.start_recording()
            else:
                self.stop_recording_and_save()
        elif key == 'o':
            logger.info("Opening Gripper\n")
            self.set_gripper_position(0.0)
        elif key == 'p':
            logger.info("Closing Gripper\n")
            self.set_gripper_position(1.0)
        elif key == 'r':
            logger.info("Resetting robot...\n")
            self.reset_robot()
        elif key == 'h':
            self.print_state()
        elif key == '\x1b': # ESC
            logger.info("\nExiting...\n")
            self.running = False
            self.cleanup()
            sys.exit(0)

    def print_instructions(self):
        """
        打印使用说明
        
        显示所有可用的键盘控制命令。
        """
        print("\n" + "="*60)
        print("Controls:")
        print("  Enter: Start/Stop Recording")
        print("  O: Open Gripper")
        print("  P: Close Gripper")
        print("  R: Reset Robot")
        print("  H: Print status")
        print("  ESC: Exit")
        print("="*60 + "\n")

    def print_state(self):
        """
        打印当前数据收集状态
        
        显示的信息包括：
            - Episode 计数（当前/总数）
            - 录制状态
            - 已记录的步数（如果正在录制）
            - 录制时间（如果正在录制）
            - 数据目录路径
            - 会话目录名称
        """
        print("\n" + "="*60)
        print("Current Status:")
        print(f"  Episode Count: {self.episode_count}/{self.num_demonstrations}")
        print(f"  Recording: {'Yes' if self.is_recording else 'No'}")
        if self.is_recording:
            print(f"  Steps Recorded: {self.step_count}")
            elapsed = time.time() - self.recording_start_time
            print(f"  Recording Time: {elapsed:.1f}s")
        print(f"  Data Directory: {self.data_dir}")
        print(f"  Session Directory: {self.session_dir.name if hasattr(self, 'session_dir') else 'N/A'}")
        print("="*60 + "\n")

    def start_keyboard_listener(self):
        """
        启动键盘监听线程
        
        创建后台线程监听键盘输入，主线程处理按键事件。
        使用非阻塞方式读取按键，避免频繁切换终端模式导致输出格式混乱。
        
        注意：
            - 使用 termios 和 tty 设置终端为原始模式以读取单个字符
            - 每次读取后立即恢复终端状态，避免影响日志输出
            - 如果出错，会尝试恢复原始终端状态
        """
        def get_key():
            """
            非阻塞读取单个按键字符
            
            Returns:
                str: 按键字符，如果出错则返回 None
            
            注意：
                - 每次调用都会临时设置终端为原始模式
                - 读取后立即恢复终端状态，确保不影响其他输出
            """
            try:
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)  # 设置终端为原始模式
                    ch = sys.stdin.read(1)
                    return ch
                finally:
                    # 确保恢复终端状态
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    # 刷新输出缓冲区，确保终端状态正确
                    sys.stdout.flush()
            except Exception as e:
                # 如果出错，尝试恢复终端状态
                try:
                    if self.original_termios:
                        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.original_termios)
                        sys.stdout.flush()
                except:
                    pass
                return None
                
        def listener():
            """
            键盘监听线程函数
            
            持续读取按键并放入队列，直到 running 标志为 False。
            """
            while self.running:
                k = get_key()
                if k: 
                    self.key_queue.put(k)
                
        # 启动后台监听线程
        t = threading.Thread(target=listener, daemon=True)
        t.start()
        
        # 主线程处理按键事件
        while self.running:
            try:
                k = self.key_queue.get(timeout=0.1)
                self.handle_key_press(k)
            except queue.Empty:
                pass

    def cleanup(self):
        """
        清理资源并恢复终端状态
        
        执行以下操作：
        1. 恢复原始终端状态（重要！确保退出后终端正常）
        2. 关闭机器人环境（断开连接、关闭相机等）
        
        注意：
            - 必须在程序退出前调用此方法
            - 如果终端状态未恢复，退出后终端可能无法正常显示输入
        """
        # 恢复终端状态（重要！确保退出后终端正常）
        try:
            if self.original_termios:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.original_termios)
                sys.stdout.flush()
        except:
            pass
        
        if self.env:
            self.env.close()

if __name__ == "__main__":
    import signal
    collector = None
    
    def signal_handler(sig, frame):
        """
        处理系统信号（Ctrl+C 等），确保恢复终端状态
        
        Args:
            sig: 信号编号
            frame: 当前堆栈帧
        
        当收到 SIGINT (Ctrl+C) 或 SIGTERM 信号时，清理资源并退出。
        确保终端状态正确恢复。
        """
        if collector:
            collector.running = False
            collector.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        collector = LiberoDataCollector()
    except KeyboardInterrupt:
        if collector:
            collector.cleanup()
        sys.exit(0)
