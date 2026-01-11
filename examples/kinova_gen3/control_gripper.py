"""
UDP 控制与反馈工具

用途
- 通过 UDP 与 Arduino 固件控制的局域网夹爪通信。
- 两类操作：
  1) 发送控制命令：文本 key=value，逗号分隔（如 "speed=10.0,angle=1872.0,dir=1"），等待设备返回 ACK（优先解析为 JSON）。
  2) 请求设备反馈：发送查询 payload，解析设备返回的 JSON 或文本。

当前Arduino IP为192.168.1.43

重要说明：
- 参数中的angle表示夹爪角度改变量绝对值，1872为完全打开/关闭需要的角度。打开（负）和关闭（正）由速度控制。
- 调整夹爪通讯接口为USART，波特率设置为115200，反馈方式设为固定频率。

快速开始
- Python 调用示例：
  from control_gripper import send_control, request_feedback_once, wait_for_fresh_feedback

  # 发送控制命令（正=闭合，负=张开；内部会转换为 dir 并取绝对值）
  ack = send_control(host="192.168.1.43", port=2390, speed=20.0, angle=1872.0,
                     microstep=32, id=1, mode=2, timeout=2.0)
  print(ack)  # 若为 JSON，返回 dict；否则返回 {"_raw": "...", "_from": (...)}

  # 请求一次反馈
  fb = request_feedback_once(host="192.168.1.43", port=2390, timeout=0.5)
  print(fb)

  # 轮询直至收到反馈
  fresh = wait_for_fresh_feedback(host="192.168.1.43", port=2390,
                                  timeout=2.0, poll_interval=0.2)
  print(fresh)

- 命令行用法示例：
  1) 发送控制命令并等待 ACK
     python control_gripper.py --host 192.168.1.43 --port 2390 --speed 20.0 --angle 1872 \
         --microstep 32 --id 1 --mode 2 --timeout 2.0

  2) 查询一次反馈（不发送控制命令）
     python control_gripper.py --host 192.168.1.43 --get-feedback --timeout 0.5

  3) 等待最新反馈（status==ok），直到超时
     python control_gripper.py --host 192.168.1.43 --get-feedback --wait \
         --timeout 2.0 --poll-interval 0.2

常用 Python 接口
- send_control(host, port, speed, angle, microstep=32, id=1, mode=2, timeout=2.0) -> dict
  上层控制接口；超时抛出 TimeoutError。
- build_kv_payload(speed, angle, microstep=None, id=None, mode=None) -> str
  构造 "key=value" 的控制 payload。约定：speed>=0 -> dir=1；speed<0 -> dir=0；发送时用 abs(speed)。
- send_and_wait_ack(host, port, payload: bytes, timeout=2.0) -> dict|None
  低层发送并等待一次应答；优先解析 JSON。超时返回 None。
- request_feedback_once(host, port=2390, timeout=0.5) -> dict
  发送一次反馈查询（默认 payload 为 "feedback"），返回 JSON 或 {"_raw": ...}。超时抛出 socket.timeout。
- wait_for_fresh_feedback(host, port=2390, timeout=2.0, poll_interval=0.2) -> dict
  轮询直到设备返回 status=="ok" 的反馈；否则超时抛出 TimeoutError。
- cli_main()
  命令行入口。

命令行参数
- --host/-H: 设备 IP（必填）
- --port/-p: UDP 端口，默认 2390
- 控制参数（未加 --get-feedback 时需同时提供 --speed 与 --angle）
  --speed/-s: 转速（rad/s），正=顺时针(闭合)，负=逆时针(张开)
  --angle/-a: 角度（deg）
  --microstep/-m: 细分，默认 32
  --id: 控制 ID，默认 1
  --mode: 控制模式，默认 2（位置控制）
  --timeout/-t: 等待响应超时（秒），默认 2.0
- 反馈查询
  --get-feedback: 只查询反馈，不发送控制
  --wait: 与 --get-feedback 一起使用，等待“最近”反馈（status==ok）
  --poll-interval: 轮询间隔（秒，--wait 有效），默认 0.2，请勿设置过小

行为与返回
- ACK/反馈优先解析为 JSON；解析失败时返回 {"_raw": 原始文本, "_from": (ip, port)}。
- send_and_wait_ack 超时返回 None；send_control/等待“最近”反馈接口超时会抛出 TimeoutError。
- 控制 payload 示例：
  speed=10.0,angle=1872.0,dir=1,microstep=32,id=1,mode=2

注意事项
- 方向约定：speed 的符号仅用于计算 dir（>=0 -> dir=1，<0 -> dir=0），实际发送 speed 字段为绝对值。
- 若存在防火墙/路由限制，请放行 UDP 端口（默认 2390）。
"""

from typing import Optional, Dict, Any
import socket
import json
import argparse
import time

DEFAULT_PORT = 2390
DEFAULT_TIMEOUT = 2.0  # seconds
DEFAULT_POLL_INTERVAL = 0.2  # seconds for feedback polling

def build_kv_payload(speed: float,
                     angle: float,
                     microstep: Optional[int] = None,
                     id: Optional[int] = None,
                     mode: Optional[int] = None) -> str:
    """
    构造 key=value 文本 payload（逗号分隔）。

    参数:
    - speed: 目标速度（数值）。符号表示方向：>=0 -> dir=1，<0 -> dir=0。发送时使用绝对值。
    - angle: 目标角度（数值）。
    - microstep: 可选，细分值（int）。
    - id: 可选，控制 ID（int）。
    - mode: 可选，控制模式（int）。

    返回:
    - payload 字符串，例如 "speed=10.0,angle=1872.0,dir=1,microstep=32,id=1,mode=2"
    """
    dir_val = 1 if speed >= 0 else 0
    speed_abs = abs(speed)
    parts = [
        f"speed={speed_abs}",
        f"angle={angle}",
        f"dir={dir_val}"
    ]
    if microstep is not None:
        parts.append(f"microstep={int(microstep)}")
    if id is not None:
        parts.append(f"id={int(id)}")
    if mode is not None:
        parts.append(f"mode={int(mode)}")
    return ",".join(parts)

def send_and_wait_ack(host: str,
                      port: int,
                      payload: bytes,
                      timeout: float = DEFAULT_TIMEOUT) -> Optional[Dict[str, Any]]:
    """
    通过 UDP 发送 payload（bytes），阻塞等待一次应答并尝试解析为 JSON。

    参数:
    - host: 目标主机 IP 或域名
    - port: 目标 UDP 端口
    - payload: 要发送的 bytes（例如 UTF-8 编码后的 payload 文本）
    - timeout: 等待应答的超时时间（秒）

    返回:
    - 如果收到可解析为 JSON 的文本，返回解析后的 dict。
    - 如果收到文本但无法解析为 JSON，返回 {"_raw": text, "_from": addr}。
    - 如果在超时时间内没有收到任何数据，返回 None。

    注意:
    - 本函数自己不会抛出 socket.timeout；在底层 recv 超时时会返回 None。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(timeout)
        sock.sendto(payload, (host, port))
        try:
            data, addr = sock.recvfrom(4096)
        except socket.timeout:
            return None
        text = data.decode('utf-8', errors='replace').strip()
        try:
            parsed = json.loads(text)
            return parsed
        except Exception:
            return {"_raw": text, "_from": addr}
    finally:
        sock.close()

def send_control(host: str,
                 port: int,
                 speed: float,
                 angle: float,
                 microstep: Optional[int] = 32,
                 id: Optional[int] = 1,
                 mode: Optional[int] = 2,
                 timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    上层接口：构造 payload 并发送控制命令，等待并返回解析后的 ACK（dict）。

    参数:
    - host: 目标主机 IP 或域名
    - port: 目标 UDP 端口
    - speed: 发送的速度（float），符号表示方向（函数内部会转换为 dir 字段并发送绝对值）
    - angle: 角度（float）
    - microstep: 可选，细分（int），默认 32
    - id: 可选，控制 id（int），默认 1
    - mode: 可选，控制模式（int），默认 2
    - timeout: 等待 ACK 的超时（秒），默认 DEFAULT_TIMEOUT

    返回:
    - 解析后的 ACK（dict）。如果收到的是不可解析的文本，也会作为 dict 返回（包含 "_raw"）。

    异常:
    - 如果在 timeout 时间内没有收到 ACK，会抛出 TimeoutError。
    """
    payload_str = build_kv_payload(speed, angle, microstep, id, mode)
    resp = send_and_wait_ack(host, port, payload_str.encode('utf-8'), timeout=timeout)
    if resp is None:
        raise TimeoutError(f"No ACK from {host}:{port} within {timeout} seconds")
    return resp

# ---------------------------------------------------------------------
# 新增：反馈查询（与控制 ACK 不同）
# Arduino 固件会对收到包含 "get_feedback" 的 UDP payload 作为反馈查询进行回复。
# 以下函数将专门用于请求 TTL 反馈并把结果返回给调用者。
# ---------------------------------------------------------------------

def request_feedback_once(host: str,
                          port: int = DEFAULT_PORT,
                          timeout: float = 0.5) -> Optional[Dict[str, Any]]:
    """
    向设备发送一次反馈查询 "get_feedback" 并等待回复（一次性）。

    参数:
    - host: 目标主机 IP 或域名
    - port: 目标 UDP 端口（默认 DEFAULT_PORT）
    - timeout: 等待回复的超时（秒）

    返回:
    - 如果收到 JSON 字符串且能解析，返回解析后的 dict。
    - 如果收到非 JSON 文本，返回 {"_raw": text, "_from": addr}。
    - 如果超时，抛出 socket.timeout（调用方可捕获），或在内部处理后返回 None（在本实现中会抛出 socket.timeout）。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(timeout)
        # 固件约定的查询关键字为 "get_feedback"
        sock.sendto(b"feedback", (host, port))
        data, addr = sock.recvfrom(4096)
        text = data.decode('utf-8', errors='replace').strip()
        try:
            parsed = json.loads(text)
            return parsed
        except Exception:
            return {"_raw": text, "_from": addr}
    finally:
        sock.close()

def wait_for_fresh_feedback(host: str,
                            port: int = DEFAULT_PORT,
                            timeout: float = DEFAULT_TIMEOUT,
                            poll_interval: float = DEFAULT_POLL_INTERVAL) -> Dict[str, Any]:
    """
    循环发送反馈查询直到收到设备返回 status=="ok"（表示最近帧 <= 200 ms），或直到 timeout 秒已过。

    参数:
    - host: 目标主机 IP 或域名
    - port: 目标 UDP 端口（默认 DEFAULT_PORT）
    - timeout: 最大等待时间（秒）
    - poll_interval: 每次轮询间隔（秒），同时作为单次 recv 的超时时间

    返回:
    - 设备返回的 JSON dict（当其包含 status=="ok" 时返回）。通常包含 id/arrived/speed/angle/age_ms 等字段。

    异常:
    - 若在 timeout 时间内未收到 status=="ok"，抛出 TimeoutError（并在消息中包含最后一次收到的回复或 None）。
    """
    end_time = time.time() + timeout
    last_resp = None
    while time.time() < end_time:
        try:
            # 将单次请求的超时设为 poll_interval，以便快速轮询
            resp = request_feedback_once(host, port, timeout=poll_interval)
            print(f"Feedback received: {resp}")
        except socket.timeout:
            resp = None
        except Exception:
            resp = None
        if resp is None:
            # 没有收到任何回应，继续重试直到超时
            last_resp = None
            time.sleep(poll_interval)
            continue
        last_resp = resp
        # 如果设备返回 JSON 并 status == "ok"，则为最新反馈（设备固件判定 age<=200ms）
        if isinstance(resp, dict) and resp.get("status") == "ok":
            return resp
        # 否则继续重试，等待设备提供最近帧
        time.sleep(poll_interval)
    # 超时
    raise TimeoutError(f"No recent feedback (status==ok) from {host}:{port} within {timeout} seconds. Last reply: {last_resp}")

# ---------------------------------------------------------------------
# 命令行入口：支持控制与反馈查询
# ---------------------------------------------------------------------

def cli_main():
    p = argparse.ArgumentParser(description="UDP 控制与反馈工具：发送 speed/angle 并/或 查询 Arduino TTL 反馈")
    p.add_argument('--host', '-H', required=True, help="设备 IP 地址")
    p.add_argument('--port', '-p', type=int, default=DEFAULT_PORT, help=f"设备 UDP 端口，默认 {DEFAULT_PORT}")

    # 控制参数（如果不指定 --get-feedback 将执行控制）
    p.add_argument('--speed', '-s', type=float, help="转速（rad/s）。符号表示方向：正数=顺时针(闭合)，负数=逆时针(张开)。")
    p.add_argument('--angle', '-a', type=float, help="角度（deg）")
    p.add_argument('--microstep', '-m', type=int, default=32, help="细分值，默认 32")
    p.add_argument('--id', type=int, default=1, help="控制 ID，默认 1")
    p.add_argument('--mode', type=int, default=2, help="控制模式，默认 2（位置控制）")
    p.add_argument('--timeout', '-t', type=float, default=DEFAULT_TIMEOUT, help=f"等待响应超时(s)，默认 {DEFAULT_TIMEOUT}")

    # 反馈查询参数
    p.add_argument('--get-feedback', action='store_true', help="请求设备返回最近的 TTL 反馈（不发送控制命令）")
    p.add_argument('--wait', action='store_true', help="与 --get-feedback 一起使用：循环等待直到收到最近反馈（status==ok）或超时")
    p.add_argument('--poll-interval', type=float, default=DEFAULT_POLL_INTERVAL, help="轮询间隔（秒），仅在 --wait 时有效")

    args = p.parse_args()

    if args.get_feedback:
        # 只做反馈查询（不发送控制命令）
        if args.wait:
            try:
                fb = wait_for_fresh_feedback(args.host, port=args.port, timeout=args.timeout, poll_interval=args.poll_interval)
                print("Fresh feedback received:")
                print(json.dumps(fb, ensure_ascii=False, indent=2))
            except TimeoutError as e:
                print("Timeout waiting for fresh feedback:", e)
        else:
            # 一次性请求设备的即时回复（可能返回 no_recent_feedback）
            try:
                resp = request_feedback_once(args.host, port=args.port, timeout=args.timeout)
                if resp is None:
                    print("No reply received.")
                else:
                    print("Feedback reply:")
                    print(json.dumps(resp, ensure_ascii=False, indent=2))
            except socket.timeout:
                print("No reply received (socket timeout).")
            except Exception as e:
                print("Error requesting feedback:", e)
        return

    # 否则为控制命令模式（speed & angle 必须提供）
    if args.speed is None or args.angle is None:
        print("Error: speed and angle are required when not using --get-feedback")
        return

    try:
        ack = send_control(args.host, args.port, speed=args.speed, angle=args.angle,
                           microstep=args.microstep, id=args.id, mode=args.mode, timeout=args.timeout)
        print("Received control ACK:")
        print(json.dumps(ack, ensure_ascii=False, indent=2))
    except TimeoutError as e:
        print("Timeout waiting for control ACK:", e)
    except Exception as e:
        print("Error sending control:", e)

if __name__ == "__main__":
    cli_main()