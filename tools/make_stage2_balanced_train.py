import shutil
from pathlib import Path

import numpy as np
import torch


SRC_ROOT = Path("data/transmission_line_stage2_tower")
DST_ROOT = Path("data/transmission_line_stage2_tower_balance")

# stage2 classes:
# 0 tower
# 1 insulator
# 2 hengdan
# 3 background


def load_label(p):
    data = torch.load(p, map_location="cpu")
    y = data["semantic_gt"]
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    return y.astype(np.int64)


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    if DST_ROOT.exists():
        print(f"[WARN] remove old output: {DST_ROOT}")
        shutil.rmtree(DST_ROOT)

    # val/test 不增强，保持真实分布
    for split in ["val", "test"]:
        src_dir = SRC_ROOT / split
        dst_dir = DST_ROOT / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        for p in sorted(src_dir.glob("*.pth")):
            copy_file(p, dst_dir / p.name)

        print(f"{split}: copied {len(list(src_dir.glob('*.pth')))} files")

    src_train = SRC_ROOT / "train"
    dst_train = DST_ROOT / "train"
    dst_train.mkdir(parents=True, exist_ok=True)

    stats = {
        "base": 0,
        "insulator_extra": 0,
        "hengdan_extra": 0,
        "tower_extra": 0,
        "mixed_extra": 0,
    }

    for p in sorted(src_train.glob("*.pth")):
        y = load_label(p)
        cnt = np.bincount(y, minlength=4)
        total = cnt.sum()

        tower_ratio = cnt[0] / total
        insulator_ratio = cnt[1] / total
        hengdan_ratio = cnt[2] / total

        stem = p.stem

        # 原始样本保留 1 份
        copy_file(p, dst_train / p.name)
        stats["base"] += 1

        # 绝缘子最少，重点增强
        if cnt[1] > 0 or insulator_ratio > 0.005:
            for k in range(4):
                copy_file(p, dst_train / f"{stem}_ins_aug{k}.pth")
                stats["insulator_extra"] += 1

        # 横担次重点增强
        if cnt[2] > 0 or hengdan_ratio > 0.02:
            for k in range(2):
                copy_file(p, dst_train / f"{stem}_hengdan_aug{k}.pth")
                stats["hengdan_extra"] += 1

        # tower 只轻微增强，避免模型全预测 tower
        if tower_ratio > 0.08 or cnt[0] > 30000:
            copy_file(p, dst_train / f"{stem}_tower_aug0.pth")
            stats["tower_extra"] += 1

        # 同时包含绝缘子和横担的 ROI 最有价值
        if cnt[1] > 0 and cnt[2] > 0:
            for k in range(3):
                copy_file(p, dst_train / f"{stem}_mixed_aug{k}.pth")
                stats["mixed_extra"] += 1

    print("\nStage2 train oversampling done:")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\nOutput:", DST_ROOT)
    print("train files:", len(list((DST_ROOT / "train").glob("*.pth"))))
    print("val files:", len(list((DST_ROOT / "val").glob("*.pth"))))
    print("test files:", len(list((DST_ROOT / "test").glob("*.pth"))))


if __name__ == "__main__":
    main()
