"""生成 Stage-1 4 类训练集的过采样/均衡版本。

用途：
    在 Stage-1 4 类数据中额外复制 tower_structure、line 或二者混合较多的
    train tile，缓解 ground 占比过高导致的类别不均衡。val/test 不增强，
    保持真实分布用于评估。
输入：
    默认 data/transmission_line_stage1_4cls_random。
输出：
    默认 data/transmission_line_stage1_4cls_random_balance。
"""

import shutil
from pathlib import Path

import numpy as np
import torch


SRC_ROOT = Path("data/transmission_line_stage1_4cls_random")
DST_ROOT = Path("data/transmission_line_stage1_4cls_random_balance")

# stage1 4 classes:
# 0 ground
# 1 tower_structure
# 2 line
# 3 other


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

    # val/test 不做过采样，保持真实分布
    for split in ["val", "test"]:
        src_dir = SRC_ROOT / split
        dst_dir = DST_ROOT / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(src_dir.glob("*.pth")):
            copy_file(p, dst_dir / p.name)
        print(f"{split}: copied {len(list(src_dir.glob('*.pth')))} files")

    # train 做过采样
    src_train = SRC_ROOT / "train"
    dst_train = DST_ROOT / "train"
    dst_train.mkdir(parents=True, exist_ok=True)

    stats = {
        "base": 0,
        "tower_extra": 0,
        "line_extra": 0,
        "mixed_extra": 0,
    }

    for p in sorted(src_train.glob("*.pth")):
        y = load_label(p)
        cnt = np.bincount(y, minlength=4)
        total = cnt.sum()

        ground_ratio = cnt[0] / total
        tower_ratio = cnt[1] / total
        line_ratio = cnt[2] / total
        other_ratio = cnt[3] / total

        # 先复制原始样本 1 份
        copy_file(p, dst_train / p.name)
        stats["base"] += 1

        stem = p.stem

        # tower_structure-rich：额外复制 2 份
        if tower_ratio > 0.05 or cnt[1] > 50000:
            for k in range(2):
                copy_file(p, dst_train / f"{stem}_tower_aug{k}.pth")
                stats["tower_extra"] += 1

        # tower + line mixed：额外复制 2 份
        if tower_ratio > 0.03 and line_ratio > 0.02:
            for k in range(2):
                copy_file(p, dst_train / f"{stem}_mixed_aug{k}.pth")
                stats["mixed_extra"] += 1

        # line-rich：额外复制 1 份
        # 注意 line 不要过采样太多，避免塔材被分成 line
        if line_ratio > 0.03 or cnt[2] > 30000:
            copy_file(p, dst_train / f"{stem}_line_aug0.pth")
            stats["line_extra"] += 1

    print("\nTrain oversampling done:")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\nOutput:", DST_ROOT)
    print("train files:", len(list((DST_ROOT / "train").glob("*.pth"))))
    print("val files:", len(list((DST_ROOT / "val").glob("*.pth"))))
    print("test files:", len(list((DST_ROOT / "test").glob("*.pth"))))


if __name__ == "__main__":
    main()
