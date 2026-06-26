#!/usr/bin/env python
"""统计输电线路预处理数据集中各语义类别的点数和占比。

用途：
    用来检查 train/val/test 的类别分布，判断 ground、tower、line、
    insulator 是否严重不均衡，以及过采样/重采样后的数据是否符合预期。
输入：
    Pointcept 预处理后的 .pth tile，默认读取 semantic_gt 标签。
输出：
    终端表格；可选保存 CSV。
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_CLASS_NAMES = ("ground", "tower", "line", "insulator")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Count class point totals and ratios from Pointcept transmission-line "
            ".pth files containing semantic_gt labels."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/transmission_line_stage1_random"),
        help="Dataset root containing train/val/test split directories.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val", "test"),
        help="Split directories to scan.",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help=(
            "Class names in label-id order. If omitted, metadata.json is used when "
            "available, otherwise ground tower line insulator."
        ),
    )
    parser.add_argument(
        "--label-key",
        default="semantic_gt",
        help="Label key inside each .pth file.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to save the summary as CSV.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "metadata", "pth"),
        default="auto",
        help=(
            "Where to read counts from. auto uses metadata.json when it contains "
            "per-scene saved_points, otherwise scans .pth files."
        ),
    )
    return parser.parse_args()


def load_class_names(data_root, cli_class_names):
    if cli_class_names:
        return tuple(cli_class_names)

    metadata_path = data_root / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        class_names = metadata.get("class_names")
        if class_names:
            return tuple(class_names)

    return DEFAULT_CLASS_NAMES


def label_array(data, label_key, path):
    if label_key not in data:
        available = ", ".join(sorted(data.keys()))
        raise KeyError(f"{path} does not contain '{label_key}'. Available keys: {available}")
    return np.asarray(data[label_key], dtype=np.int64).reshape(-1)


def count_split(split_path, label_key, num_classes):
    import torch

    counts = np.zeros(num_classes, dtype=np.int64)
    ignored = 0
    files = sorted(split_path.glob("*.pth"))
    for path in files:
        data = torch.load(path, map_location="cpu")
        labels = label_array(data, label_key, path)
        valid = (labels >= 0) & (labels < num_classes)
        if np.any(valid):
            counts += np.bincount(labels[valid], minlength=num_classes)
        ignored += int(labels.size - np.count_nonzero(valid))
    return files, counts, ignored


def load_metadata(data_root):
    metadata_path = data_root / "metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def metadata_has_counts(metadata):
    scenes = metadata.get("scenes") if metadata else None
    return bool(scenes) and all("split" in scene and "saved_points" in scene for scene in scenes)


def count_from_metadata(metadata, splits, num_classes):
    split_counts = {
        split: dict(
            counts=np.zeros(num_classes, dtype=np.int64),
            ignored=0,
            files=0,
            scenes=0,
        )
        for split in splits
    }

    for scene in metadata["scenes"]:
        split = scene["split"]
        if split not in split_counts:
            continue
        saved_points = np.asarray(scene["saved_points"], dtype=np.int64).reshape(-1)
        kept = min(saved_points.size, num_classes)
        split_counts[split]["counts"][:kept] += saved_points[:kept]
        if saved_points.size > num_classes:
            split_counts[split]["ignored"] += int(saved_points[num_classes:].sum())
        split_counts[split]["files"] += int(scene.get("tiles", 0))
        split_counts[split]["scenes"] += 1

    return split_counts


def format_int(value):
    return f"{int(value):,}"


def print_table(title, class_names, counts, ignored, files_count, scenes_count=None):
    total = int(counts.sum())
    print(f"\n{title}")
    scene_text = "" if scenes_count is None else f"scenes: {scenes_count:,}  "
    print(
        f"{scene_text}files/tiles: {files_count:,}  "
        f"valid points: {total:,}  ignored/out-of-range: {ignored:,}"
    )
    print("-" * 72)
    print(f"{'label':>5}  {'class':<16}  {'points':>16}  {'ratio':>10}")
    print("-" * 72)
    for label, name in enumerate(class_names):
        ratio = (counts[label] / total * 100.0) if total > 0 else 0.0
        print(f"{label:>5}  {name:<16}  {format_int(counts[label]):>16}  {ratio:>9.4f}%")
    print("-" * 72)


def write_csv(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("split", "label", "class", "points", "ratio", "files", "ignored"),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    metadata = load_metadata(args.data_root)
    class_names = load_class_names(args.data_root, args.class_names)
    num_classes = len(class_names)

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

    use_metadata = args.source == "metadata" or (
        args.source == "auto" and metadata_has_counts(metadata)
    )
    if args.source == "metadata" and not metadata_has_counts(metadata):
        raise ValueError(
            f"{args.data_root / 'metadata.json'} does not contain scenes[].saved_points counts."
        )

    all_counts = np.zeros(num_classes, dtype=np.int64)
    all_ignored = 0
    all_files = 0
    all_scenes = 0
    csv_rows = []

    if use_metadata:
        print(f"Reading counts from {args.data_root / 'metadata.json'}")
        split_stats = count_from_metadata(metadata, args.splits, num_classes)
    else:
        print("Reading counts by scanning .pth files. This can be slow for large datasets.")
        split_stats = {}
        for split in args.splits:
            split_path = args.data_root / split
            if not split_path.exists():
                print(f"\n{split}: skipped, directory does not exist: {split_path}")
                continue
            files, counts, ignored = count_split(split_path, args.label_key, num_classes)
            split_stats[split] = dict(
                counts=counts,
                ignored=ignored,
                files=len(files),
                scenes=None,
            )

    for split in args.splits:
        if split not in split_stats:
            continue
        counts = split_stats[split]["counts"]
        ignored = split_stats[split]["ignored"]
        files_count = split_stats[split]["files"]
        scenes_count = split_stats[split].get("scenes")
        print_table(split, class_names, counts, ignored, files_count, scenes_count)

        total = int(counts.sum())
        for label, name in enumerate(class_names):
            ratio = (counts[label] / total * 100.0) if total > 0 else 0.0
            csv_rows.append(
                dict(
                    split=split,
                    label=label,
                    **{"class": name},
                    points=int(counts[label]),
                    ratio=f"{ratio:.8f}",
                    files=files_count,
                    ignored=ignored,
                )
            )

        all_counts += counts
        all_ignored += ignored
        all_files += files_count
        if scenes_count is not None:
            all_scenes += scenes_count

    print_table("all", class_names, all_counts, all_ignored, all_files, all_scenes if use_metadata else None)
    all_total = int(all_counts.sum())
    for label, name in enumerate(class_names):
        ratio = (all_counts[label] / all_total * 100.0) if all_total > 0 else 0.0
        csv_rows.append(
            dict(
                split="all",
                label=label,
                **{"class": name},
                points=int(all_counts[label]),
                ratio=f"{ratio:.8f}",
                files=all_files,
                ignored=all_ignored,
            )
        )

    if args.csv_out:
        write_csv(args.csv_out, csv_rows)
        print(f"\nSaved CSV to {args.csv_out}")


if __name__ == "__main__":
    main()
