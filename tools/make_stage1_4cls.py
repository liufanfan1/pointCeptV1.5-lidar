import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch


"""
Generate Stage-1 4-class dataset from original 6-class transmission-line dataset.

Original 6 classes:
    0 ground
    1 tower
    2 line
    3 insulator
    4 hengdan
    5 other

Stage-1 4 classes:
    0 ground
    1 tower_structure = tower + insulator + hengdan
    2 line
    3 other
"""


NEW_CLASS_NAMES = [
    "ground",
    "tower_structure",
    "line",
    "other",
]

# old label -> new label
REMAP = np.array(
    [
        0,  # old 0 ground     -> new 0 ground
        1,  # old 1 tower      -> new 1 tower_structure
        2,  # old 2 line       -> new 2 line
        1,  # old 3 insulator  -> new 1 tower_structure
        1,  # old 4 hengdan    -> new 1 tower_structure
        3,  # old 5 other      -> new 3 other
    ],
    dtype=np.int64,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert 6-class transmission-line .pth dataset to Stage-1 4-class dataset."
    )
    parser.add_argument(
        "--src-root",
        default="data/transmission_line",
        help="Original 6-class Pointcept dataset root.",
    )
    parser.add_argument(
        "--dst-root",
        default="data/transmission_line_stage1_4cls",
        help="Output Stage-1 4-class dataset root.",
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
                f"Output directory already exists: {dst_root}. "
                f"Use --overwrite to regenerate."
            )

    total_counts = {
        "train": np.zeros(len(NEW_CLASS_NAMES), dtype=np.int64),
        "val": np.zeros(len(NEW_CLASS_NAMES), dtype=np.int64),
        "test": np.zeros(len(NEW_CLASS_NAMES), dtype=np.int64),
    }

    total_files = {}

    for split in ["train", "val", "test"]:
        src_dir = src_root / split
        dst_dir = dst_root / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {src_dir}")

        files = sorted(src_dir.glob("*.pth"))
        total_files[split] = len(files)

        for p in files:
            data = torch.load(p, map_location="cpu")

            if "semantic_gt" not in data:
                raise KeyError(f"{p} does not contain key 'semantic_gt'")

            old_label = to_numpy(data["semantic_gt"]).astype(np.int64)

            if old_label.size == 0:
                raise ValueError(f"Empty semantic_gt in {p}")

            if old_label.min() < 0 or old_label.max() > 5:
                raise ValueError(
                    f"Unexpected label range in {p}: "
                    f"min={old_label.min()}, max={old_label.max()}"
                )

            new_label = REMAP[old_label].astype(np.int64)

            # Keep original fields: coord, color, origin, scene, etc.
            data["semantic_gt"] = new_label

            torch.save(data, dst_dir / p.name)

            total_counts[split] += np.bincount(
                new_label, minlength=len(NEW_CLASS_NAMES)
            )

        print(f"{split}: converted {len(files)} files")

    metadata = {
        "source_dataset": str(src_root),
        "stage": "stage1_4cls",
        "class_names": NEW_CLASS_NAMES,
        "old_to_new_remap": {
            "0_ground": 0,
            "1_tower": 1,
            "2_line": 2,
            "3_insulator": 1,
            "4_hengdan": 1,
            "5_other": 3,
        },
        "files": total_files,
        "points": {
            split: {
                NEW_CLASS_NAMES[i]: int(total_counts[split][i])
                for i in range(len(NEW_CLASS_NAMES))
            }
            for split in ["train", "val", "test"]
        },
    }

    dst_root.mkdir(parents=True, exist_ok=True)
    with open(dst_root / "metadata_stage1_4cls.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nSaved Stage-1 dataset to:", dst_root)
    print("\nClass distribution:")
    for split in ["train", "val", "test"]:
        counts = total_counts[split]
        total = counts.sum()
        print(f"\n[{split}] files={total_files[split]}, points={int(total)}")
        for i, name in enumerate(NEW_CLASS_NAMES):
            ratio = counts[i] / total if total > 0 else 0
            print(f"  {i} {name:16s}: {int(counts[i]):12d}, ratio={ratio:.6f}")


if __name__ == "__main__":
    main()
