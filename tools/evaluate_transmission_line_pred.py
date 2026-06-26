#!/usr/bin/env python
"""评估输电线路预测结果 *_pred.npy 的 mIoU/mAcc/allAcc。

用途：
    不重新跑模型，只读取已有预测文件和对应 .pth 真值标签，快速复算整体
    指标和每类 IoU/Accuracy。适合比较不同推理目录或导出后的预测结果。
输入：
    result 目录中的 *_pred.npy，以及 data-root/split 下同名 .pth。
输出：
    终端指标；可选保存 JSON。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


CLASS_NAMES = ("ground", "tower", "line", "insulator", "hengdan", "other")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute mIoU/mAcc/allAcc for transmission-line prediction files."
    )
    parser.add_argument(
        "pred",
        type=Path,
        nargs="+",
        help="Prediction file(s) or directories containing *_pred.npy.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/transmission_line"),
        help="Pointcept transmission-line data root.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split containing matching .pth files.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=6,
        help="Number of semantic classes.",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=-1,
        help="Ignore label index.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path for metrics.",
    )
    return parser.parse_args()


def collect_pred_paths(paths):
    pred_paths = []
    for path in paths:
        if path.is_dir():
            pred_paths.extend(sorted(path.glob("*_pred.npy")))
        else:
            pred_paths.append(path)
    pred_paths = sorted(set(pred_paths))
    if not pred_paths:
        raise FileNotFoundError("No *_pred.npy files were found.")
    for path in pred_paths:
        if not path.name.endswith("_pred.npy"):
            raise ValueError(f"Expected *_pred.npy, got {path}")
    return pred_paths


def tile_name_from_pred(pred_path):
    return pred_path.name[: -len("_pred.npy")]


def load_label(tile_path):
    data = torch.load(tile_path, map_location="cpu")
    if "semantic_gt" not in data:
        raise KeyError(f"{tile_path} does not contain semantic_gt.")
    label = data["semantic_gt"]
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    return np.asarray(label, dtype=np.int64).reshape(-1)


def intersection_and_union(pred, label, num_classes, ignore_index):
    pred = pred.reshape(-1).copy()
    label = label.reshape(-1)
    if pred.shape != label.shape:
        raise ValueError(
            f"Prediction/label shape mismatch: {pred.shape} vs {label.shape}"
        )
    pred[label == ignore_index] = ignore_index
    intersection = pred[pred == label]
    area_intersection = np.histogram(intersection, bins=np.arange(num_classes + 1))[0]
    area_pred = np.histogram(pred, bins=np.arange(num_classes + 1))[0]
    area_label = np.histogram(label, bins=np.arange(num_classes + 1))[0]
    area_union = area_pred + area_label - area_intersection
    return area_intersection, area_union, area_label


def safe_divide(num, den):
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den != 0)


def main():
    args = parse_args()
    pred_paths = collect_pred_paths(args.pred)

    total_intersection = np.zeros(args.num_classes, dtype=np.int64)
    total_union = np.zeros(args.num_classes, dtype=np.int64)
    total_target = np.zeros(args.num_classes, dtype=np.int64)
    per_tile = []

    for pred_path in pred_paths:
        tile_name = tile_name_from_pred(pred_path)
        tile_path = args.data_root / args.split / f"{tile_name}.pth"
        if not tile_path.exists():
            raise FileNotFoundError(f"Missing matching tile: {tile_path}")

        pred = np.load(pred_path).astype(np.int64).reshape(-1)
        label = load_label(tile_path)
        intersection, union, target = intersection_and_union(
            pred, label, args.num_classes, args.ignore_index
        )
        total_intersection += intersection
        total_union += union
        total_target += target

        iou = safe_divide(intersection.astype(np.float64), union.astype(np.float64))
        valid = union != 0
        per_tile.append(
            dict(
                name=tile_name,
                points=int(label.shape[0]),
                miou=float(iou[valid].mean()) if valid.any() else 0.0,
            )
        )

    iou_class = safe_divide(
        total_intersection.astype(np.float64), total_union.astype(np.float64)
    )
    acc_class = safe_divide(
        total_intersection.astype(np.float64), total_target.astype(np.float64)
    )
    valid_iou = total_union != 0
    valid_acc = total_target != 0
    miou = float(iou_class[valid_iou].mean()) if valid_iou.any() else 0.0
    macc = float(acc_class[valid_acc].mean()) if valid_acc.any() else 0.0
    allacc = float(total_intersection.sum() / max(total_target.sum(), 1))

    result = dict(
        num_tiles=len(pred_paths),
        num_classes=args.num_classes,
        mIoU=miou,
        mAcc=macc,
        allAcc=allacc,
        classes=[],
        per_tile=per_tile,
    )

    print("Evaluation result")
    print(f"tiles: {len(pred_paths)}")
    print(f"mIoU/mAcc/allAcc {miou:.4f}/{macc:.4f}/{allacc:.4f}")
    for idx in range(args.num_classes):
        name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
        item = dict(
            id=idx,
            name=name,
            iou=float(iou_class[idx]),
            accuracy=float(acc_class[idx]),
            intersection=int(total_intersection[idx]),
            union=int(total_union[idx]),
            target=int(total_target[idx]),
        )
        result["classes"].append(item)
        print(
            "Class_{idx} - {name}: iou/accuracy {iou:.4f}/{acc:.4f} "
            "intersection/union/target {inter}/{union}/{target}".format(
                idx=idx,
                name=name,
                iou=item["iou"],
                acc=item["accuracy"],
                inter=item["intersection"],
                union=item["union"],
                target=item["target"],
            )
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved metrics: {args.out}")


if __name__ == "__main__":
    main()

"""
脚本评价标准
cd /24085403037/PointTransformerV3/Pointcept-v1.5.1

/opt/conda/envs/pointcept/bin/python tools/evaluate_transmission_line_pred.py \
  /24085403037/PointTransformerV3/Pointcept-v1.5.1/exp/transmission/two_stage_infer_scope_ts/result \
  --data-root data/transmission_line \
  --split test \
  --out exp/transmission/two_stage_infer_scope_ts/two_stage_metrics.json 
  """
