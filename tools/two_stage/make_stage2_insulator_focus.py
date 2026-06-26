"""生成偏绝缘子/横担关注的 Stage-2 ROI 增强数据集。

用途：
    在 Stage-2 base 数据基础上，从原始 6 类数据里按目标类别外扩 ROI，
    生成更聚焦 insulator/hengdan 的训练样本。和 centered 版本相比，这个
    脚本更像按目标包围盒外扩裁剪，参数写在脚本常量里。
输入：
    原始 6 类数据 data/transmission_line，以及 Stage-2 base/balance 数据。
输出：
    默认 data/transmission_line_stage2_tower_ins_focus。
"""

import shutil
from pathlib import Path

import numpy as np
import torch


# 原始 6 类数据
SRC_6CLS = Path("data/transmission_line")

# 你已有的第二阶段增强数据
SRC_STAGE2 = Path("data/transmission_line_stage2_tower_balance")

# 新的绝缘子增强数据
DST_ROOT = Path("data/transmission_line_stage2_tower_ins_focus")

# old 6cls -> stage2 4cls
# 0 ground     -> 3 background
# 1 tower      -> 0 tower
# 2 line       -> 3 background
# 3 insulator  -> 1 insulator
# 4 hengdan    -> 2 hengdan
# 5 other      -> 3 background
REMAP = np.array([3, 0, 3, 1, 2, 3], dtype=np.int64)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x


def copy_tree_split(split):
    src = SRC_STAGE2 / split
    dst = DST_ROOT / split
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob("*.pth")):
        shutil.copy2(p, dst / p.name)
    print(f"{split}: copied {len(list(src.glob('*.pth')))} base files")


def make_roi_from_old6(
    p, target_old_label, xy_margin, z_margin, min_target_points, suffix
):
    data = torch.load(p, map_location="cpu")

    coord = to_numpy(data["coord"]).astype(np.float32)
    old_y = to_numpy(data["semantic_gt"]).astype(np.int64)

    target_mask = old_y == target_old_label
    if int(target_mask.sum()) < min_target_points:
        return None

    target_coord = coord[target_mask]
    xyz_min = target_coord.min(axis=0)
    xyz_max = target_coord.max(axis=0)

    xyz_min[0] -= xy_margin
    xyz_min[1] -= xy_margin
    xyz_min[2] -= z_margin

    xyz_max[0] += xy_margin
    xyz_max[1] += xy_margin
    xyz_max[2] += z_margin

    roi_mask = (
        (coord[:, 0] >= xyz_min[0])
        & (coord[:, 0] <= xyz_max[0])
        & (coord[:, 1] >= xyz_min[1])
        & (coord[:, 1] <= xyz_max[1])
        & (coord[:, 2] >= xyz_min[2])
        & (coord[:, 2] <= xyz_max[2])
    )

    if int(roi_mask.sum()) < 128:
        return None

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

    out["semantic_gt"] = REMAP[old_y[roi_mask]].astype(np.int64)
    out["source_tile"] = p.name
    out["roi_type"] = suffix
    out["roi_min"] = xyz_min.astype(np.float32)
    out["roi_max"] = xyz_max.astype(np.float32)

    return out


def main():
    if DST_ROOT.exists():
        print(f"[WARN] remove old output: {DST_ROOT}")
        shutil.rmtree(DST_ROOT)

    # val/test 保持不变
    copy_tree_split("val")
    copy_tree_split("test")

    # train 先复制已有 stage2 balance
    copy_tree_split("train")

    dst_train = DST_ROOT / "train"

    ins_count = 0
    hengdan_count = 0

    for p in sorted((SRC_6CLS / "train").glob("*.pth")):
        # 绝缘子最关键：小范围 ROI，多生成
        ins_roi = make_roi_from_old6(
            p,
            target_old_label=3,
            xy_margin=3.0,
            z_margin=2.0,
            min_target_points=10,
            suffix="insulator_focus",
        )
        if ins_roi is not None:
            for k in range(3):
                out_name = f"{p.stem}_ins_focus{k}.pth"
                torch.save(ins_roi, dst_train / out_name)
                ins_count += 1

        # 横担适量增强
        hengdan_roi = make_roi_from_old6(
            p,
            target_old_label=4,
            xy_margin=5.0,
            z_margin=2.0,
            min_target_points=30,
            suffix="hengdan_focus",
        )
        if hengdan_roi is not None:
            out_name = f"{p.stem}_hengdan_focus0.pth"
            torch.save(hengdan_roi, dst_train / out_name)
            hengdan_count += 1

    print("\nAdded focused ROI:")
    print("insulator-focused files:", ins_count)
    print("hengdan-focused files:", hengdan_count)

    print("\nFinal files:")
    for split in ["train", "val", "test"]:
        print(split, len(list((DST_ROOT / split).glob("*.pth"))))

    print("\nSaved to:", DST_ROOT)


if __name__ == "__main__":
    main()
