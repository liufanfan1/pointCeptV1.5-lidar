import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch


"""
Generate insulator-centered Stage-2 crop dataset.

Input:
    Original 6-class dataset:
        data/transmission_line

    Existing Stage-2 base dataset:
        data/transmission_line_stage2_tower_balance

Output:
    New Stage-2 dataset:
        data/transmission_line_stage2_tower_ins_centered

Stage-2 labels:
    0 tower
    1 insulator
    2 hengdan
    3 background

Original 6-class labels:
    0 ground
    1 tower
    2 line
    3 insulator
    4 hengdan
    5 other
"""


CLASS_NAMES = ["tower", "insulator", "hengdan", "background"]

# old 6-class label -> stage2 4-class label
REMAP = np.array([
    3,  # old 0 ground     -> background
    0,  # old 1 tower      -> tower
    3,  # old 2 line       -> background
    1,  # old 3 insulator  -> insulator
    2,  # old 4 hengdan    -> hengdan
    3,  # old 5 other      -> background
], dtype=np.int64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate insulator-centered crops for Stage-2 tower fine segmentation."
    )
    parser.add_argument(
        "--src-root",
        default="data/transmission_line",
        help="Original 6-class Pointcept dataset root.",
    )
    parser.add_argument(
        "--base-stage2-root",
        default="data/transmission_line_stage2_tower_balance",
        help="Existing Stage-2 dataset root. Its train/val/test will be copied first.",
    )
    parser.add_argument(
        "--dst-root",
        default="data/transmission_line_stage2_tower_ins_centered",
        help="Output Stage-2 insulator-centered dataset root.",
    )
    parser.add_argument(
        "--ins-crops-per-tile",
        type=int,
        default=8,
        help="Number of insulator-centered crops generated per source tile containing insulator.",
    )
    parser.add_argument(
        "--hengdan-crops-per-tile",
        type=int,
        default=2,
        help="Optional hengdan-centered crops per source tile containing hengdan.",
    )
    parser.add_argument(
        "--ins-xy-radius",
        type=float,
        default=2.5,
        help="Half size in x/y for insulator-centered crop, in meters.",
    )
    parser.add_argument(
        "--ins-z-radius",
        type=float,
        default=1.8,
        help="Half size in z for insulator-centered crop, in meters.",
    )
    parser.add_argument(
        "--hengdan-xy-radius",
        type=float,
        default=4.0,
        help="Half size in x/y for hengdan-centered crop, in meters.",
    )
    parser.add_argument(
        "--hengdan-z-radius",
        type=float,
        default=2.0,
        help="Half size in z for hengdan-centered crop, in meters.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=128,
        help="Skip crop if total points are fewer than this.",
    )
    parser.add_argument(
        "--min-ins-points",
        type=int,
        default=5,
        help="Skip insulator-centered crop if insulator points are fewer than this.",
    )
    parser.add_argument(
        "--min-hengdan-points",
        type=int,
        default=20,
        help="Skip hengdan-centered crop if hengdan points are fewer than this.",
    )
    parser.add_argument(
        "--center-jitter",
        type=float,
        default=0.3,
        help="Random center jitter in meters for x/y/z, useful to create crop diversity.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory.",
    )
    return parser.parse_args()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x


def copy_split(src_root, dst_root, split):
    src_dir = src_root / split
    dst_dir = dst_root / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.pth"))
    for p in files:
        shutil.copy2(p, dst_dir / p.name)

    print(f"{split}: copied {len(files)} base files from {src_dir}")


def make_crop(data, center, xy_radius, z_radius):
    coord = to_numpy(data["coord"]).astype(np.float32)
    old_label = to_numpy(data["semantic_gt"]).astype(np.int64)

    x, y, z = center

    mask = (
        (coord[:, 0] >= x - xy_radius) & (coord[:, 0] <= x + xy_radius) &
        (coord[:, 1] >= y - xy_radius) & (coord[:, 1] <= y + xy_radius) &
        (coord[:, 2] >= z - z_radius) & (coord[:, 2] <= z + z_radius)
    )

    if int(mask.sum()) == 0:
        return None

    new_label = REMAP[old_label[mask]].astype(np.int64)

    out = {}
    for k, v in data.items():
        if k in ["coord", "color", "semantic_gt"]:
            continue
        out[k] = v

    out["coord"] = coord[mask].astype(np.float32)

    if "color" in data:
        color = to_numpy(data["color"])
        out["color"] = color[mask]
    else:
        out["color"] = np.zeros((out["coord"].shape[0], 3), dtype=np.uint8)

    out["semantic_gt"] = new_label

    return out


def save_crop_if_valid(out, out_path, min_points, min_target_points, target_new_label):
    if out is None:
        return False, None

    y = out["semantic_gt"]
    cnt = np.bincount(y, minlength=4)

    if int(cnt.sum()) < min_points:
        return False, cnt

    if int(cnt[target_new_label]) < min_target_points:
        return False, cnt

    torch.save(out, out_path)
    return True, cnt


def main():
    args = parse_args()

    src_root = Path(args.src_root)
    base_root = Path(args.base_stage2_root)
    dst_root = Path(args.dst_root)

    if not src_root.exists():
        raise FileNotFoundError(f"Original 6-class dataset not found: {src_root}")

    if not base_root.exists():
        raise FileNotFoundError(f"Base Stage-2 dataset not found: {base_root}")

    if dst_root.exists():
        if args.overwrite:
            print(f"[WARN] remove old output: {dst_root}")
            shutil.rmtree(dst_root)
        else:
            raise FileExistsError(f"{dst_root} exists. Use --overwrite.")

    rng = np.random.default_rng(args.seed)

    # 先复制已有的 stage2 balance 数据集
    for split in ["train", "val", "test"]:
        copy_split(base_root, dst_root, split)

    dst_train = dst_root / "train"

    total_added_counts = np.zeros(4, dtype=np.int64)
    added_ins_crops = 0
    added_hengdan_crops = 0
    skipped_ins_crops = 0
    skipped_hengdan_crops = 0
    source_tiles_with_ins = 0
    source_tiles_with_hengdan = 0

    src_train_files = sorted((src_root / "train").glob("*.pth"))

    for p in src_train_files:
        data = torch.load(p, map_location="cpu")

        coord = to_numpy(data["coord"]).astype(np.float32)
        old_label = to_numpy(data["semantic_gt"]).astype(np.int64)

        if coord.shape[0] != old_label.shape[0]:
            raise ValueError(f"Point count mismatch in {p}")

        if old_label.min() < 0 or old_label.max() > 5:
            raise ValueError(f"Bad label range in {p}: {old_label.min()} ~ {old_label.max()}")

        # ----------------------------
        # 1. insulator-centered crops
        # ----------------------------
        ins_idx = np.where(old_label == 3)[0]
        if len(ins_idx) > 0:
            source_tiles_with_ins += 1

            for k in range(args.ins_crops_per_tile):
                center_idx = rng.choice(ins_idx)
                center = coord[center_idx].copy()

                if args.center_jitter > 0:
                    center += rng.uniform(
                        -args.center_jitter,
                        args.center_jitter,
                        size=3,
                    ).astype(np.float32)

                out = make_crop(
                    data,
                    center=center,
                    xy_radius=args.ins_xy_radius,
                    z_radius=args.ins_z_radius,
                )

                if out is not None:
                    out["source_tile"] = p.name
                    out["roi_type"] = "insulator_centered"
                    out["center"] = center.astype(np.float32)
                    out["xy_radius"] = np.float32(args.ins_xy_radius)
                    out["z_radius"] = np.float32(args.ins_z_radius)

                out_name = f"{p.stem}_ins_center_{k:02d}.pth"
                ok, cnt = save_crop_if_valid(
                    out,
                    dst_train / out_name,
                    min_points=args.min_points,
                    min_target_points=args.min_ins_points,
                    target_new_label=1,
                )

                if ok:
                    added_ins_crops += 1
                    total_added_counts += cnt
                else:
                    skipped_ins_crops += 1

        # ----------------------------
        # 2. hengdan-centered crops
        # ----------------------------
        hengdan_idx = np.where(old_label == 4)[0]
        if len(hengdan_idx) > 0 and args.hengdan_crops_per_tile > 0:
            source_tiles_with_hengdan += 1

            for k in range(args.hengdan_crops_per_tile):
                center_idx = rng.choice(hengdan_idx)
                center = coord[center_idx].copy()

                if args.center_jitter > 0:
                    center += rng.uniform(
                        -args.center_jitter,
                        args.center_jitter,
                        size=3,
                    ).astype(np.float32)

                out = make_crop(
                    data,
                    center=center,
                    xy_radius=args.hengdan_xy_radius,
                    z_radius=args.hengdan_z_radius,
                )

                if out is not None:
                    out["source_tile"] = p.name
                    out["roi_type"] = "hengdan_centered"
                    out["center"] = center.astype(np.float32)
                    out["xy_radius"] = np.float32(args.hengdan_xy_radius)
                    out["z_radius"] = np.float32(args.hengdan_z_radius)

                out_name = f"{p.stem}_hengdan_center_{k:02d}.pth"
                ok, cnt = save_crop_if_valid(
                    out,
                    dst_train / out_name,
                    min_points=args.min_points,
                    min_target_points=args.min_hengdan_points,
                    target_new_label=2,
                )

                if ok:
                    added_hengdan_crops += 1
                    total_added_counts += cnt
                else:
                    skipped_hengdan_crops += 1

    metadata = {
        "source_6class_dataset": str(src_root),
        "base_stage2_dataset": str(base_root),
        "output_dataset": str(dst_root),
        "class_names": CLASS_NAMES,
        "stage": "stage2_insulator_centered",
        "old_to_new_remap": {
            "0_ground": 3,
            "1_tower": 0,
            "2_line": 3,
            "3_insulator": 1,
            "4_hengdan": 2,
            "5_other": 3,
        },
        "insulator_crop": {
            "crops_per_tile": args.ins_crops_per_tile,
            "xy_radius": args.ins_xy_radius,
            "z_radius": args.ins_z_radius,
            "min_points": args.min_points,
            "min_ins_points": args.min_ins_points,
        },
        "hengdan_crop": {
            "crops_per_tile": args.hengdan_crops_per_tile,
            "xy_radius": args.hengdan_xy_radius,
            "z_radius": args.hengdan_z_radius,
            "min_points": args.min_points,
            "min_hengdan_points": args.min_hengdan_points,
        },
        "center_jitter": args.center_jitter,
        "seed": args.seed,
        "source_tiles_with_insulator": source_tiles_with_ins,
        "source_tiles_with_hengdan": source_tiles_with_hengdan,
        "added_insulator_centered_crops": added_ins_crops,
        "skipped_insulator_centered_crops": skipped_ins_crops,
        "added_hengdan_centered_crops": added_hengdan_crops,
        "skipped_hengdan_centered_crops": skipped_hengdan_crops,
        "added_crop_points": {
            CLASS_NAMES[i]: int(total_added_counts[i])
            for i in range(4)
        },
    }

    with open(dst_root / "metadata_stage2_insulator_centered.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n================ Done ================")
    print("Output:", dst_root)
    print("source tiles with insulator:", source_tiles_with_ins)
    print("source tiles with hengdan:", source_tiles_with_hengdan)
    print("added insulator-centered crops:", added_ins_crops)
    print("skipped insulator-centered crops:", skipped_ins_crops)
    print("added hengdan-centered crops:", added_hengdan_crops)
    print("skipped hengdan-centered crops:", skipped_hengdan_crops)

    print("\nAdded crop point distribution:")
    total = total_added_counts.sum()
    for i, name in enumerate(CLASS_NAMES):
        ratio = total_added_counts[i] / total if total > 0 else 0
        print(f"  {i} {name:12s}: {int(total_added_counts[i]):12d}, ratio={ratio:.6f}")

    print("\nFinal file counts:")
    for split in ["train", "val", "test"]:
        print(split, len(list((dst_root / split).glob("*.pth"))))


if __name__ == "__main__":
    main()
