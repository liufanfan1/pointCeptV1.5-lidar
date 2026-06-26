"""从原始 6 类输电线路数据生成 Stage-2 杆塔 ROI 4 类数据集。

用途：
    两阶段流程的第二步数据准备。脚本根据原始标签中的 tower、insulator、
    hengdan 找到杆塔相关前景范围，外扩成局部 ROI，并把标签重映射成
    tower/insulator/hengdan/background 四类，供 Stage-2 精分训练。
输入：
    默认 data/transmission_line，标签为原始 6 类。
输出：
    默认 data/transmission_line_stage2_tower，标签为 Stage-2 4 类。

Stage-2 4 类：
    0 tower, 1 insulator, 2 hengdan, 3 background
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch


NEW_CLASS_NAMES = [
    "tower",
    "insulator",
    "hengdan",
    "background",
]

# old label -> new label
REMAP = np.array(
    [
        3,  # old 0 ground     -> new 3 background
        0,  # old 1 tower      -> new 0 tower
        3,  # old 2 line       -> new 3 background
        1,  # old 3 insulator  -> new 1 insulator
        2,  # old 4 hengdan    -> new 2 hengdan
        3,  # old 5 other      -> new 3 background
    ],
    dtype=np.int64,
)

# 用于确定杆塔 ROI 的前景类别：tower + insulator + hengdan
TARGET_OLD_LABELS = (1, 3, 4)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Stage-2 tower ROI dataset from original 6-class Pointcept .pth files."
    )
    parser.add_argument(
        "--src-root",
        default="data/transmission_line",
        help="Original 6-class Pointcept dataset root.",
    )
    parser.add_argument(
        "--dst-root",
        default="data/transmission_line_stage2_tower",
        help="Output Stage-2 tower ROI dataset root.",
    )
    parser.add_argument(
        "--xy-margin",
        type=float,
        default=8.0,
        help="ROI expansion margin in X/Y directions, in meters.",
    )
    parser.add_argument(
        "--z-margin",
        type=float,
        default=3.0,
        help="ROI expansion margin in Z direction, in meters.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=256,
        help="Skip ROI if total points after cropping are fewer than this.",
    )
    parser.add_argument(
        "--min-target-points",
        type=int,
        default=32,
        help="Skip ROI if tower/insulator/hengdan target points are fewer than this.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing dst-root before generating.",
    )
    return parser.parse_args()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x


def crop_one_file(p, args):
    data = torch.load(p, map_location="cpu")

    if "coord" not in data or "semantic_gt" not in data:
        raise KeyError(f"{p} must contain 'coord' and 'semantic_gt'.")

    coord = to_numpy(data["coord"]).astype(np.float32)
    label_old = to_numpy(data["semantic_gt"]).astype(np.int64)

    if coord.shape[0] != label_old.shape[0]:
        raise ValueError(
            f"Point count mismatch in {p}: coord={coord.shape}, label={label_old.shape}"
        )

    if label_old.min() < 0 or label_old.max() > 5:
        raise ValueError(
            f"Unexpected label range in {p}: min={label_old.min()}, max={label_old.max()}"
        )

    target_mask = np.isin(label_old, TARGET_OLD_LABELS)
    target_count = int(target_mask.sum())

    if target_count < args.min_target_points:
        return None

    target_coord = coord[target_mask]

    xyz_min = target_coord.min(axis=0)
    xyz_max = target_coord.max(axis=0)

    xyz_min[0] -= args.xy_margin
    xyz_min[1] -= args.xy_margin
    xyz_min[2] -= args.z_margin

    xyz_max[0] += args.xy_margin
    xyz_max[1] += args.xy_margin
    xyz_max[2] += args.z_margin

    roi_mask = (
        (coord[:, 0] >= xyz_min[0])
        & (coord[:, 0] <= xyz_max[0])
        & (coord[:, 1] >= xyz_min[1])
        & (coord[:, 1] <= xyz_max[1])
        & (coord[:, 2] >= xyz_min[2])
        & (coord[:, 2] <= xyz_max[2])
    )

    if int(roi_mask.sum()) < args.min_points:
        return None

    label_new = REMAP[label_old[roi_mask]].astype(np.int64)

    out = {}
    for k, v in data.items():
        if k in ["coord", "color", "semantic_gt"]:
            continue
        out[k] = v

    out["coord"] = coord[roi_mask].astype(np.float32)
    if "color" in data:
        color = to_numpy(data["color"])
        out["color"] = color[roi_mask]
    else:
        out["color"] = np.zeros((out["coord"].shape[0], 3), dtype=np.uint8)

    out["semantic_gt"] = label_new
    out["source_tile"] = p.name
    out["roi_min"] = xyz_min.astype(np.float32)
    out["roi_max"] = xyz_max.astype(np.float32)

    counts = np.bincount(label_new, minlength=len(NEW_CLASS_NAMES))
    return out, counts, target_count


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_root}")

    if dst_root.exists():
        if args.overwrite:
            print(f"[WARN] Removing existing output directory: {dst_root}")
            shutil.rmtree(dst_root)
        else:
            raise FileExistsError(
                f"Output directory already exists: {dst_root}. Use --overwrite to regenerate."
            )

    total_counts = {
        "train": np.zeros(len(NEW_CLASS_NAMES), dtype=np.int64),
        "val": np.zeros(len(NEW_CLASS_NAMES), dtype=np.int64),
        "test": np.zeros(len(NEW_CLASS_NAMES), dtype=np.int64),
    }

    total_files = {}
    kept_files = {}
    skipped_files = {}

    for split in ["train", "val", "test"]:
        src_dir = src_root / split
        dst_dir = dst_root / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(src_dir.glob("*.pth"))
        total_files[split] = len(files)
        kept_files[split] = 0
        skipped_files[split] = 0

        for p in files:
            result = crop_one_file(p, args)
            if result is None:
                skipped_files[split] += 1
                continue

            out, counts, target_count = result

            out_name = p.stem + "_tower_roi.pth"
            torch.save(out, dst_dir / out_name)

            total_counts[split] += counts
            kept_files[split] += 1

        print(
            f"{split}: source={total_files[split]}, "
            f"kept={kept_files[split]}, skipped={skipped_files[split]}"
        )

    metadata = {
        "source_dataset": str(src_root),
        "stage": "stage2_tower_roi",
        "class_names": NEW_CLASS_NAMES,
        "old_to_new_remap": {
            "0_ground": 3,
            "1_tower": 0,
            "2_line": 3,
            "3_insulator": 1,
            "4_hengdan": 2,
            "5_other": 3,
        },
        "target_old_labels_for_roi": {
            "1": "tower",
            "3": "insulator",
            "4": "hengdan",
        },
        "xy_margin": args.xy_margin,
        "z_margin": args.z_margin,
        "min_points": args.min_points,
        "min_target_points": args.min_target_points,
        "files": {
            split: {
                "source": total_files[split],
                "kept": kept_files[split],
                "skipped": skipped_files[split],
            }
            for split in ["train", "val", "test"]
        },
        "points": {
            split: {
                NEW_CLASS_NAMES[i]: int(total_counts[split][i])
                for i in range(len(NEW_CLASS_NAMES))
            }
            for split in ["train", "val", "test"]
        },
    }

    dst_root.mkdir(parents=True, exist_ok=True)
    with open(dst_root / "metadata_stage2_tower_roi.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nSaved Stage-2 tower ROI dataset to:", dst_root)
    print("\nClass distribution:")
    for split in ["train", "val", "test"]:
        counts = total_counts[split]
        total = counts.sum()
        print(f"\n[{split}] files={kept_files[split]}, points={int(total)}")
        if total == 0:
            continue
        for i, name in enumerate(NEW_CLASS_NAMES):
            print(
                f"  {i} {name:12s}: {int(counts[i]):12d}, ratio={counts[i] / total:.6f}"
            )


if __name__ == "__main__":
    main()
