#!/usr/bin/env python
"""生成一阶段 4 类绝缘子过采样训练集。

用途：
    针对当前一阶段 4 类模型中 insulator 类样本少的问题，只复制 train 中含
    insulator 的 tile，提高训练时小目标出现频率。val/test 保持原始分布，
    便于指标真实可比。
输入：
    默认 data/transmission_line_stage1_random。
输出：
    默认 data/transmission_line_stage1_random_ins_oversample。
实现：
    默认使用 hardlink，表现为多份样本，但不重复占用大点云文件空间。
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


DEFAULT_CLASS_NAMES = ("ground", "tower", "line", "insulator")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Oversample train tiles containing insulator points."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("data/transmission_line_stage1_random"),
        help="Input dataset root containing train/val/test.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path("data/transmission_line_stage1_random_ins_oversample"),
        help="Output dataset root.",
    )
    parser.add_argument(
        "--insulator-label",
        type=int,
        default=3,
        help="Semantic label id for insulator.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="Number of semantic classes.",
    )
    parser.add_argument(
        "--min-insulator-points",
        type=int,
        default=1,
        help="Only oversample train tiles with at least this many insulator points.",
    )
    parser.add_argument(
        "--extra-copies",
        type=int,
        default=2,
        help="Extra copies for normal insulator-containing train tiles.",
    )
    parser.add_argument(
        "--rich-insulator-points",
        type=int,
        default=5000,
        help="Tiles with at least this many insulator points use --rich-extra-copies.",
    )
    parser.add_argument(
        "--rich-insulator-ratio",
        type=float,
        default=0.01,
        help="Tiles with at least this insulator ratio use --rich-extra-copies.",
    )
    parser.add_argument(
        "--rich-extra-copies",
        type=int,
        default=4,
        help="Extra copies for insulator-rich train tiles.",
    )
    parser.add_argument(
        "--method",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to create files in the output dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing dst-root before generating.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report what would be generated without writing files.",
    )
    parser.add_argument(
        "--stats-cache",
        type=Path,
        default=None,
        help=(
            "Optional JSON cache for per-tile class counts. Reuse this to avoid "
            "loading all .pth files again when changing copy factors."
        ),
    )
    return parser.parse_args()


def load_metadata(src_root):
    metadata_path = src_root / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def class_names_from_metadata(metadata, num_classes):
    names = metadata.get("class_names") or DEFAULT_CLASS_NAMES
    names = list(names)
    if len(names) < num_classes:
        names.extend([f"class_{i}" for i in range(len(names), num_classes)])
    return names[:num_classes]


def safe_remove(path):
    if path.exists():
        shutil.rmtree(path)


def link_or_copy(src, dst, method):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if method == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif method == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def to_numpy_label(label):
    import torch

    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    return np.asarray(label, dtype=np.int64).reshape(-1)


def count_file(path, label_key, num_classes):
    import torch

    data = torch.load(path, map_location="cpu")
    if label_key not in data:
        raise KeyError(f"{path} does not contain '{label_key}'")
    label = to_numpy_label(data[label_key])
    valid = (label >= 0) & (label < num_classes)
    counts = np.bincount(label[valid], minlength=num_classes).astype(np.int64)
    ignored = int(label.size - np.count_nonzero(valid))
    return counts, ignored


def load_stats_cache(cache_path):
    if cache_path is None or not cache_path.exists():
        return None
    with cache_path.open("r", encoding="utf-8") as file:
        cache = json.load(file)
    return {
        item["relative_path"]: (
            np.asarray(item["counts"], dtype=np.int64),
            int(item.get("ignored", 0)),
        )
        for item in cache.get("files", [])
    }


def save_stats_cache(cache_path, stats):
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "files": [
            {
                "relative_path": key,
                "counts": counts.tolist(),
                "ignored": ignored,
            }
            for key, (counts, ignored) in sorted(stats.items())
        ]
    }
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def collect_split_stats(src_root, splits, num_classes, label_key, stats_cache):
    cached = load_stats_cache(stats_cache)
    stats = {} if cached is None else dict(cached)
    changed = False

    for split in splits:
        files = sorted((src_root / split).glob("*.pth"))
        for index, path in enumerate(files, start=1):
            relative = str(path.relative_to(src_root))
            if relative in stats:
                continue
            counts, ignored = count_file(path, label_key, num_classes)
            stats[relative] = (counts, ignored)
            changed = True
            if index % 100 == 0:
                print(f"[scan] {split}: {index}/{len(files)} files")

    if changed:
        save_stats_cache(stats_cache, stats)
    return stats


def extra_copies_for_count(counts, args):
    total = int(counts.sum())
    ins_count = int(counts[args.insulator_label])
    if total <= 0 or ins_count < args.min_insulator_points:
        return 0
    ins_ratio = ins_count / total
    if ins_count >= args.rich_insulator_points or ins_ratio >= args.rich_insulator_ratio:
        return args.rich_extra_copies
    return args.extra_copies


def copy_split(src_root, dst_root, split, files, method, dry_run):
    if dry_run:
        return
    for path in files:
        dst = dst_root / split / path.name
        link_or_copy(path, dst, method)


def main():
    args = parse_args()
    if not args.src_root.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {args.src_root}")
    if args.insulator_label < 0 or args.insulator_label >= args.num_classes:
        raise ValueError("--insulator-label must be within [0, num_classes)")
    if args.dst_root.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output exists, use --overwrite: {args.dst_root}")

    metadata = load_metadata(args.src_root)
    class_names = class_names_from_metadata(metadata, args.num_classes)
    stats_cache = args.stats_cache or (
        None if args.dry_run else args.dst_root / "tile_class_stats_cache.json"
    )

    if not args.dry_run:
        safe_remove(args.dst_root)
        args.dst_root.mkdir(parents=True, exist_ok=True)

    stats = collect_split_stats(
        args.src_root,
        splits=("train", "val", "test"),
        num_classes=args.num_classes,
        label_key="semantic_gt",
        stats_cache=stats_cache,
    )

    split_summary = {}
    total_summary = {
        "source_files": 0,
        "output_files": 0,
        "extra_files": 0,
        "source_points": np.zeros(args.num_classes, dtype=np.int64),
        "output_points": np.zeros(args.num_classes, dtype=np.int64),
        "ignored": 0,
    }

    for split in ("val", "test"):
        files = sorted((args.src_root / split).glob("*.pth"))
        copy_split(args.src_root, args.dst_root, split, files, args.method, args.dry_run)
        counts = np.zeros(args.num_classes, dtype=np.int64)
        ignored = 0
        for path in files:
            relative = str(path.relative_to(args.src_root))
            file_counts, file_ignored = stats[relative]
            counts += file_counts
            ignored += file_ignored
        split_summary[split] = {
            "source_files": len(files),
            "output_files": len(files),
            "extra_files": 0,
            "source_points": counts,
            "output_points": counts.copy(),
            "ignored": ignored,
        }

    train_files = sorted((args.src_root / "train").glob("*.pth"))
    train_counts = np.zeros(args.num_classes, dtype=np.int64)
    train_output_counts = np.zeros(args.num_classes, dtype=np.int64)
    train_ignored = 0
    insulator_tiles = 0
    rich_tiles = 0
    extra_files = 0

    for path in train_files:
        relative = str(path.relative_to(args.src_root))
        counts, ignored = stats[relative]
        train_counts += counts
        train_output_counts += counts
        train_ignored += ignored

        copies = extra_copies_for_count(counts, args)
        if copies > 0:
            insulator_tiles += 1
        if copies == args.rich_extra_copies:
            rich_tiles += 1

        if not args.dry_run:
            link_or_copy(path, args.dst_root / "train" / path.name, args.method)
            for copy_id in range(copies):
                dst_name = f"{path.stem}_insaug{copy_id:02d}{path.suffix}"
                link_or_copy(path, args.dst_root / "train" / dst_name, args.method)

        train_output_counts += counts * copies
        extra_files += copies

    split_summary["train"] = {
        "source_files": len(train_files),
        "output_files": len(train_files) + extra_files,
        "extra_files": extra_files,
        "source_points": train_counts,
        "output_points": train_output_counts,
        "ignored": train_ignored,
        "insulator_tiles": insulator_tiles,
        "rich_insulator_tiles": rich_tiles,
    }

    for split, item in split_summary.items():
        total_summary["source_files"] += item["source_files"]
        total_summary["output_files"] += item["output_files"]
        total_summary["extra_files"] += item["extra_files"]
        total_summary["source_points"] += item["source_points"]
        total_summary["output_points"] += item["output_points"]
        total_summary["ignored"] += item["ignored"]

    print("\nOversampling summary")
    print(f"source: {args.src_root}")
    print(f"output: {args.dst_root}")
    print(f"method: {args.method}")
    print(f"dry_run: {args.dry_run}")

    for split in ("train", "val", "test"):
        item = split_summary[split]
        print(
            f"\n[{split}] source_files={item['source_files']:,} "
            f"output_files={item['output_files']:,} extra={item['extra_files']:,}"
        )
        if split == "train":
            print(
                f"insulator_tiles={item['insulator_tiles']:,} "
                f"rich_insulator_tiles={item['rich_insulator_tiles']:,}"
            )
        total = int(item["output_points"].sum())
        for label, name in enumerate(class_names):
            points = int(item["output_points"][label])
            ratio = points / total * 100.0 if total > 0 else 0.0
            print(f"  {label} {name:<12s}: {points:16,}  {ratio:8.4f}%")

    output_total = int(total_summary["output_points"].sum())
    print("\n[all]")
    print(
        f"source_files={total_summary['source_files']:,} "
        f"output_files={total_summary['output_files']:,} "
        f"extra={total_summary['extra_files']:,}"
    )
    for label, name in enumerate(class_names):
        points = int(total_summary["output_points"][label])
        ratio = points / output_total * 100.0 if output_total > 0 else 0.0
        print(f"  {label} {name:<12s}: {points:16,}  {ratio:8.4f}%")

    if not args.dry_run:
        metadata_out = {
            "source_dataset": str(args.src_root),
            "method": args.method,
            "class_names": class_names,
            "insulator_label": args.insulator_label,
            "min_insulator_points": args.min_insulator_points,
            "extra_copies": args.extra_copies,
            "rich_insulator_points": args.rich_insulator_points,
            "rich_insulator_ratio": args.rich_insulator_ratio,
            "rich_extra_copies": args.rich_extra_copies,
            "splits": {
                split: {
                    key: (
                        value.tolist()
                        if isinstance(value, np.ndarray)
                        else int(value)
                        if isinstance(value, (np.integer, int))
                        else value
                    )
                    for key, value in item.items()
                }
                for split, item in split_summary.items()
            },
        }
        with (args.dst_root / "metadata_insulator_oversample.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metadata_out, file, indent=2)


if __name__ == "__main__":
    main()
