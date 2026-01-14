#!/usr/bin/env python3
"""
Kinova机械臂数据收集脚本 - 基于KinovaRobotEnv
用于收集真机演示数据，保持与LIBERO训练格式一致。
复用 kinova_env.py 中的 KinovaRobotEnv 进行硬件交互。
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
    """LIBERO格式兼容的数据收集器"""
    
    def __init__(self):
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
        return logger

    def setup_directories(self):
        """设置数据目录结构"""
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
        """开始录制"""
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
        """固定频率采集循环"""
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
        """停止录制并保存"""
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
        """收集单步数据"""
        try:
            # 使用 env 获取观测
            obs = self.env.get_observation()
            
            self.step_count += 1
            
            # 提取图像
            # 注意：KinovaRobotEnv 返回的 image 字典 key 是 serial_number_left
            # 我们需要适配一下，假设第一个是外部，第二个是腕部，或者根据 key 判断
            imgs = list(obs['image'].values())
            # 简单假设：如果指定了 serial，就按 serial 找，否则按顺序
            ext_img = imgs[0] if len(imgs) > 0 else np.zeros((256, 256, 3), dtype=np.uint8)
            wrist_img = imgs[1] if len(imgs) > 1 else np.zeros((256, 256, 3), dtype=np.uint8)
            
            # 调整图像大小到 256x256 (LIBERO标准)
            # 注意: KinovaRobotEnv 获取的图像大小取决于其配置(默认640x480)
            # 这里我们简单 resize
            import cv2
            if ext_img.shape[:2] != (256, 256):
                ext_img = cv2.resize(ext_img, (256, 256))
            if wrist_img.shape[:2] != (256, 256):
                wrist_img = cv2.resize(wrist_img, (256, 256))

            # 提取状态
            robot_state = obs['robot_state']
            joint_pos = robot_state['joint_positions'] # 弧度
            cart_pos = robot_state['cartesian_position'] # [x, y, z, theta_x, theta_y, theta_z] (弧度)
            gripper_pos = robot_state['gripper_position'] # [0, 1]
            
            # 转换笛卡尔位姿：从 [x,y,z,rx,ry,rz] 转为 [x,y,z,qx,qy,qz,qw]
            # KinovaRobotEnv 返回的是 rx,ry,rz (弧度)
            # 我们需要转四元数
            from scipy.spatial.transform import Rotation
            r = Rotation.from_euler('xyz', cart_pos[3:], degrees=False)
            quat = r.as_quat() # [x, y, z, w]
            
            # LIBERO 夹爪状态: +1=张开, -1=闭合
            # env gripper_pos: 0=张开, 1=闭合
            libero_gripper = 1.0 if gripper_pos < 0.5 else -1.0
            
            # 构造 8D 状态 [eef_pos(3), eef_quat(4), gripper(1)]
            state_8d = np.concatenate([
                cart_pos[:3],
                quat,
                [libero_gripper]
            ]).astype(np.float32)
            
            # 构造 7D 动作 (传入的 action_7d 已经是 [vel(6), gripper(1)])
            # 确保类型正确
            action_final = np.array(action_7d).astype(np.float32)
            
            # 保存数据
            self.continuous_episode_data['states'].append(state_8d)
            self.continuous_episode_data['actions'].append(action_final)
            self.continuous_episode_data['agent_images'].append(ext_img)
            self.continuous_episode_data['wrist_images'].append(wrist_img)
            
            # 保存回放数据
            if self.save_replay_data:
                replay_data = {
                    'timestamp': time.time() - self.recording_start_time,
                    'step': self.step_count,
                    'joint_positions': joint_pos,
                    'eef_pose': np.concatenate([cart_pos[:3], quat]),
                    'gripper_pos': gripper_pos,
                    'action': action_final
                }
                self.continuous_episode_data['replay_data'].append(replay_data)
                
            # 增量保存
            if self.step_count % 50 == 0:
                self.save_incremental_data()
                
        except Exception as e:
            logger.error(f"Failed to collect step data: {e}\n")

    def save_incremental_data(self):
        """增量保存数据"""
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
        """保存完整episode"""
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
            
            # 保存回放数据
            if self.save_replay_data and self.continuous_episode_data['replay_data']:
                replay_path = self.replay_dir / f"episode_{self.episode_count:03d}_replay_{timestamp}.npz"
                
                # 转换 list of dicts to dict of lists
                replay_dict = {}
                for k in self.continuous_episode_data['replay_data'][0].keys():
                    replay_dict[k] = []
                for d in self.continuous_episode_data['replay_data']:
                    for k, v in d.items():
                        replay_dict[k].append(v)
                        
                # 存为 numpy
                saved_data = {k: np.array(v) for k, v in replay_dict.items()}
                np.savez_compressed(str(replay_path), **saved_data)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to save episode: {e}\n")
            return False

    def validate_libero_format(self, file_path):
        """验证数据格式"""
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
        """创建摘要"""
        summary = {
            'task_description': self.task_description,
            'total_episodes': self.episode_count,
            'session_dir': str(self.session_dir),
            'timestamp': datetime.datetime.now().isoformat()
        }
        with open(self.session_dir / "collection_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

    def send_pulse_movement(self, movement_direction):
        """发送脉冲移动"""
        if not self.env or not self.env._is_connected:
            return
            
        # 构造 7D 动作 [vel(6), gripper(1)]
        action_7d_scaled = []
        for val in movement_direction:
            if val > 0:
                action_7d_scaled.append(5.625)
            elif val < 0:
                action_7d_scaled.append(-5.625)
            else:
                action_7d_scaled.append(0.0)
                
        # 添加夹爪状态 (LIBERO格式)
        # 从 env 获取当前夹爪位置 (0-1)
        curr_grip = self.env._current_gripper_pos
        libero_grip = 1.0 if curr_grip < 0.5 else -1.0
        action_7d_scaled.append(libero_grip)
        
        # 记录本次动作
        if len(self.last_executed_action) != 7:
            self.last_executed_action = np.zeros(7)
        self.last_executed_action[:] = action_7d_scaled
        
        # 执行物理移动
        try:
            pos_step = 0.02
            rot_step = 1.0
            
            direction = np.array(movement_direction)
            pos_delta = direction[0:3] * pos_step
            rot_delta = direction[3:6] * rot_step
            
            twist = Base_pb2.TwistCommand()
            twist.twist.linear_x = float(pos_delta[0])
            twist.twist.linear_y = float(pos_delta[1])
            twist.twist.linear_z = float(pos_delta[2])
            twist.twist.angular_x = float(rot_delta[0])
            twist.twist.angular_y = float(rot_delta[1])
            twist.twist.angular_z = float(rot_delta[2])
            twist.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
            
            duration = 400 if np.any(direction[3:6] != 0) else 200
            twist.duration = duration
            
            # 使用 env 内部的 _base 客户端发送指令
            self.env._base.SendTwistCommand(twist)
            
            # 启动停止线程
            wait_time = (duration + 20) / 1000.0
            threading.Thread(target=self._delayed_stop, args=(wait_time,), daemon=True).start()
            
            # 移除这里的 collect_step_data，统一由 _collection_loop 处理
            # if self.is_recording:
            #     self.collect_step_data(action_7d_scaled)
            #     logger.info(f"Step recorded. Action: {action_7d_scaled}")
                
        except Exception as e:
            logger.error(f"Movement failed: {e}\n")

    def _delayed_stop(self, delay):
        time.sleep(delay)
        try:
            stop = Base_pb2.TwistCommand()
            stop.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
            stop.duration = 0
            if self.env and self.env._is_connected:
                self.env._base.SendTwistCommand(stop)
        except:
            pass
            
    def stop_robot(self):
        """停止机器人"""
        self._delayed_stop(0)

    def set_gripper_position(self, pos):
        """控制夹爪"""
        # 使用 env 的内部方法
        self.env._control_gripper(pos)

    def reset_robot(self):
        """复位"""
        self.env.reset()

    def handle_key_press(self, key):
        """处理按键"""
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
        """打印当前状态"""
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
        def get_key():
            try:
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
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
            while self.running:
                k = get_key()
                if k: 
                    self.key_queue.put(k)
                
        t = threading.Thread(target=listener, daemon=True)
        t.start()
        
        while self.running:
            try:
                k = self.key_queue.get(timeout=0.1)
                self.handle_key_press(k)
            except queue.Empty:
                pass

    def cleanup(self):
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
        """处理 Ctrl+C 等信号，确保恢复终端状态"""
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
