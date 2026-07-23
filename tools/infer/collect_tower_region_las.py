"""把 test、val 的杆塔区域 LAS 汇总到同一个平铺目录。

本脚本只收集并重命名文件，不会把不同杆塔的点记录拼接到一个 LAS 中。
输出文件名由上一级场景目录名和原杆塔文件名组成，例如：
``110v12_merged_4cls_tower_001_杆塔1.las``。
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect cropped test/val tower LAS files into one directory."
    )
    parser.add_argument("--test-dir", required=True, help="裁剪后的 test 杆塔目录。")
    parser.add_argument("--val-dir", required=True, help="裁剪后的 val 杆塔目录。")
    parser.add_argument("--output-dir", required=True, help="平铺汇总输出目录。")
    parser.add_argument(
        "--method",
        choices=("copy", "hardlink"),
        default="copy",
        help="copy 复制文件；hardlink 创建硬链接以节省空间，默认 copy。",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="可选汇总报告 JSON；默认写到输出目录/collect_report.json。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖同名输出文件。")
    return parser.parse_args()


def find_las_files(root):
    """递归查找目录中的 LAS/LAZ 文件。"""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{root}")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in (".las", ".laz")
    )
    if not files:
        raise FileNotFoundError(f"目录中没有 LAS/LAZ 文件：{root}")
    return files


def output_name(source_path):
    """使用上一级场景目录名作为前缀，同时保留杆塔编号。"""
    return f"{source_path.parent.name}_{source_path.name}"


def transfer_file(source, target, method, overwrite):
    """复制或硬链接单个文件。"""
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"输出已存在，请添加 --overwrite：{target}")
        target.unlink()
    if method == "hardlink":
        os.link(source, target)
    else:
        shutil.copy2(source, target)


def main():
    args = parse_args()
    roots = {
        "test": Path(args.test_dir),
        "val": Path(args.val_dir),
    }
    sources = []
    for split, root in roots.items():
        sources.extend((split, path) for path in find_las_files(root))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        Path(args.report) if args.report else output_dir / "collect_report.json"
    )
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"报告已存在，请添加 --overwrite：{report_path}")

    target_names = {}
    for split, source in sources:
        name = output_name(source)
        if name in target_names:
            previous_split, previous_source = target_names[name]
            raise ValueError(
                f"输出文件名冲突：{name}\n"
                f"  {previous_split}: {previous_source}\n"
                f"  {split}: {source}\n"
                "请检查 test、val 是否包含了同名场景。"
            )
        target_names[name] = (split, source)

    records = []
    split_counts = {"test": 0, "val": 0}
    for index, (name, (split, source)) in enumerate(
        sorted(target_names.items()), start=1
    ):
        target = output_dir / name
        transfer_file(source, target, args.method, args.overwrite)
        split_counts[split] += 1
        records.append(
            {
                "split": split,
                "scene": source.parent.name,
                "source": str(source),
                "output": str(target),
                "size_bytes": int(target.stat().st_size),
            }
        )
        print(f"[{index}/{len(target_names)}] {split}: {target.name}", flush=True)

    report = {
        "test_dir": str(roots["test"]),
        "val_dir": str(roots["val"]),
        "output_dir": str(output_dir),
        "method": args.method,
        "summary": {
            "test_files": split_counts["test"],
            "val_files": split_counts["val"],
            "total_files": len(records),
        },
        "files": records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"完成：test={split_counts['test']}，val={split_counts['val']}，"
        f"总计={len(records)}，报告={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()


"""
使用方法：
python tools/infer/collect_tower_region_las.py \
  --test-dir /24085403037/24085403037/shixi/dataset/0617-4Name_original_test_tower_regions \
  --val-dir /24085403037/24085403037/shixi/dataset/0617-4Name_original_val_tower_regions \
  --output-dir /24085403037/24085403037/shixi/dataset/0617-4Name_original_test_val_towers \
  --overwrite
"""
