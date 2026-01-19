# ruff: noqa
"""控制日志工具函数。"""

import json
import os


def _append_control_log(
    log_entries: list,
    *,
    t_step: int,
    chunk_index: int,
    action_index: int,
    interp_step: int,
    interp_steps: int,
    state: dict,
    action: dict,
    mode: str,
    timestamp: float,
) -> None:
    log_entries.append(
        {
            "t_step": t_step,
            "chunk_index": chunk_index,
            "action_index": action_index,
            "interp_step": interp_step,
            "interp_steps": interp_steps,
            "state": state,
            "action": action,
            "mode": mode,
            "timestamp": timestamp,
        }
    )


def _save_control_log(log_entries: list, timestamp: str) -> None:
    if not log_entries:
        return
    os.makedirs("logs", exist_ok=True)
    filename = f"logs/control_log_{timestamp}.jsonl"
    with open(filename, "w", encoding="utf-8") as f:
        for entry in log_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"控制日志已保存: {filename}")
