import os
import shutil
import argparse


def has_libero_npz(run_dir: str) -> bool:
    """
    判断某个 run 目录下是否有 libero_format/*.npz 文件。
    """
    libero_dir = os.path.join(run_dir, "libero_format")
    if not os.path.isdir(libero_dir):
        return False

    for name in os.listdir(libero_dir):
        if name.endswith(".npz"):
            return True
    return False


def clean_empty_runs(data_root: str, dry_run: bool = True) -> None:
    """
    删除 data_root 下没有收集到 libero npz 数据的子目录。

    :param data_root: 例如 /home/kinova/qyh/openpi_kinova/openpi/examples/kinova_gen3/data
    :param dry_run: 为 True 时只打印将要删除的目录，不实际删除
    """
    if not os.path.isdir(data_root):
        print(f"[错误] data_root 不存在或不是目录: {data_root}")
        return

    print(f"扫描数据目录: {data_root}")
    to_delete = []

    for name in sorted(os.listdir(data_root)):
        run_dir = os.path.join(data_root, name)
        if not os.path.isdir(run_dir):
            continue

        if has_libero_npz(run_dir):
            print(f"[保留] {run_dir} (找到 libero_format/*.npz)")
        else:
            print(f"[删除候选] {run_dir} (未找到 libero_format/*.npz)")
            to_delete.append(run_dir)

    if not to_delete:
        print("没有需要删除的目录。")
        return

    if dry_run:
        print("\nDry-run 模式：以下目录将会被删除（目前尚未删除）：")
        for d in to_delete:
            print("  ", d)
        print("\n若要真正删除，请去掉 --dry-run 参数重新运行。")
        return

    print("\n开始删除目录...")
    for d in to_delete:
        try:
            shutil.rmtree(d)
            print(f"[已删除] {d}")
        except Exception as e:
            print(f"[失败] 删除 {d} 时出错: {e}")

    print("清理完成。")


def main():
    parser = argparse.ArgumentParser(
        description="清理没有 libero_format npz 数据的 General_manipulation_task_* 子目录"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/kinova/qyh/openpi_kinova/openpi/examples/kinova_gen3/data",
        help="数据根目录，默认为 examples/kinova_gen3/data 的绝对路径",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要删除的目录，不实际删除",
    )

    args = parser.parse_args()
    clean_empty_runs(args.data_root, dry_run=args.dry_run or False)


if __name__ == "__main__":
    main()
