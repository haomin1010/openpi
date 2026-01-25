#!/usr/bin/env python
import json
import pathlib
import sys

def main(assets_dir: str, action_key: str = "actions", keep_dims: int = 8):
    assets_path = pathlib.Path(assets_dir)
    norm_path = assets_path / "norm_stats.json"
    if not norm_path.exists():
        raise FileNotFoundError(f"norm_stats.json not found at: {norm_path}")

    data = json.loads(norm_path.read_text())
    norm_stats = data.get("norm_stats", data)  # 兼容两种结构

    if action_key not in norm_stats:
        raise KeyError(f"Key '{action_key}' not found in norm_stats. Available keys: {list(norm_stats.keys())}")

    s = norm_stats[action_key]

    def fix_array(name, default_tail):
        if name not in s:
            return
        arr = s[name]
        if not isinstance(arr, list):
            return
        dim = len(arr)
        if dim <= keep_dims:
            return
        # 前 keep_dims 维保持不变，后面维度改成 identity 参数
        for i in range(keep_dims, dim):
            arr[i] = default_tail

    # z-score 情况：mean=0, std=1 使后面维度不变
    fix_array("mean", 0.0)
    fix_array("std", 1.0)

    # 分位数归一化情况：q01=-1, q99=1 时 x=0 会保持 0，不再被压到 -1
    fix_array("q01", -1.0)
    fix_array("q99", 1.0)

    # 写回文件（备份一份原始文件）
    backup_path = norm_path.with_suffix(".json.bak")
    backup_path.write_text(json.dumps(data, indent=2))
    norm_path.write_text(json.dumps(data, indent=2))
    print(f"Updated {norm_path}, backup saved to {backup_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python edit_actions_norm_stats.py ASSETS_DIR [ACTION_KEY] [KEEP_DIMS]", file=sys.stderr)
        sys.exit(1)
    assets_dir = sys.argv[1]
    action_key = sys.argv[2] if len(sys.argv) > 2 else "actions"
    keep_dims = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    main(assets_dir, action_key, keep_dims)