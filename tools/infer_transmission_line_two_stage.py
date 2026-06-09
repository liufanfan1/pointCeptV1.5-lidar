#!/usr/bin/env python
"""Two-stage inference for transmission-line semantic segmentation.

Stage 1 predicts coarse 4-class labels on original tiles:
    0 ground, 1 tower_structure, 2 line, 3 other

Stage 2 refines a predicted tower ROI:
    0 tower, 1 insulator, 2 hengdan, 3 background

The final output is a 6-class prediction compatible with
tools/export_transmission_line_pred_ply.py:
    0 ground, 1 tower, 2 line, 3 insulator, 4 hengdan, 5 other
"""

import argparse
import copy
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointcept.datasets import collate_fn
from pointcept.datasets.transform import Compose, TRANSFORMS
from pointcept.models import build_model
from pointcept.utils.config import Config


STAGE1_TO_FINAL = np.array([0, 1, 2, 5], dtype=np.int64)
STAGE2_TO_FINAL = np.array([1, 3, 4, -1], dtype=np.int64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Stage1 + Stage2 transmission-line inference and output final 6-class predictions."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/transmission_line"),
        help="Original 6-class Pointcept .pth dataset root.",
    )
    parser.add_argument("--split", default="test", help="Dataset split under data-root.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="*",
        default=None,
        help="Specific .pth tile(s). If omitted, all data-root/split/*.pth are used.",
    )
    parser.add_argument(
        "--stage1-config",
        type=Path,
        default=Path("exp/transmission/stage1_4cls_balance_w8_clean/config.py"),
    )
    parser.add_argument(
        "--stage1-weight",
        type=Path,
        default=Path("exp/transmission/stage1_4cls_balance_w8_clean/model/model_best.pth"),
    )
    parser.add_argument(
        "--stage2-config",
        type=Path,
        default=Path("exp/transmission/stage2_tower_ins_centered_w24/config.py"),
    )
    parser.add_argument(
        "--stage2-weight",
        type=Path,
        default=Path("exp/transmission/stage2_tower_ins_centered_w24/model/model_best.pth"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("exp/transmission/two_stage_infer/result"),
        help="Output directory for final *_pred.npy files.",
    )
    parser.add_argument("--xy-margin", type=float, default=8.0, help="Stage2 ROI XY margin in meters.")
    parser.add_argument("--z-margin", type=float, default=3.0, help="Stage2 ROI Z margin in meters.")
    parser.add_argument(
        "--min-stage1-target-points",
        type=int,
        default=32,
        help="Skip Stage2 if Stage1 predicts fewer tower_structure points.",
    )
    parser.add_argument(
        "--min-roi-points",
        type=int,
        default=256,
        help="Skip Stage2 if generated ROI has fewer points.",
    )
    parser.add_argument(
        "--stage2-foreground-threshold",
        type=float,
        default=0.0,
        help="Only apply Stage2 foreground labels with max probability >= threshold.",
    )
    parser.add_argument(
        "--stage2-background-policy",
        choices=("keep", "other"),
        default="keep",
        help=(
            "For Stage1 tower_structure points where Stage2 predicts background: "
            "'keep' leaves tower fallback, 'other' maps them to final other."
        ),
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Also save Stage1 and Stage2 ROI predictions for debugging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute predictions even if final output already exists.",
    )
    parser.add_argument("--device", default="cuda", help="Inference device, usually cuda.")
    return parser.parse_args()


def load_cfg(path):
    cfg = Config.fromfile(str(path))
    cfg.model.criteria = []
    return cfg


def load_model(cfg, weight_path, device):
    model = build_model(cfg.model)
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    weight = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        weight[key] = value
    model.load_state_dict(weight, strict=True)
    model.to(device)
    model.eval()
    return model


def build_test_pipeline(cfg):
    test_cfg = cfg.data.test.test_cfg
    return dict(
        transform=Compose(cfg.data.test.transform),
        voxelize=TRANSFORMS.build(test_cfg.voxelize),
        crop=TRANSFORMS.build(test_cfg.crop) if test_cfg.crop else None,
        post_transform=Compose(test_cfg.post_transform),
        aug_transform=[Compose(aug) for aug in test_cfg.aug_transform],
    )


def numpy_or_default(data, key, default):
    value = data.get(key, default)
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()
    return np.asarray(value)


def load_tile(path):
    data = torch.load(path, map_location="cpu")
    coord = numpy_or_default(data, "coord", None).astype(np.float32)
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError(f"{path} has invalid coord shape: {coord.shape}")
    color = numpy_or_default(
        data, "color", np.zeros((coord.shape[0], 3), dtype=np.uint8)
    ).astype(np.uint8)
    if color.shape[0] != coord.shape[0]:
        raise ValueError(f"{path} coord/color point count mismatch.")
    segment = data.get("semantic_gt")
    if segment is None:
        segment = np.full(coord.shape[0], -1, dtype=np.int64)
    else:
        segment = numpy_or_default(data, "semantic_gt", None).astype(np.int64).reshape(-1)
    if segment.shape[0] != coord.shape[0]:
        raise ValueError(f"{path} coord/semantic_gt point count mismatch.")
    return dict(coord=coord, color=color, segment=segment)


def prepare_fragments(data_dict, pipeline):
    base = pipeline["transform"](copy.deepcopy(data_dict))
    input_dict_list = []
    for aug in pipeline["aug_transform"]:
        aug_data = aug(copy.deepcopy(base))
        data_part_list = pipeline["voxelize"](aug_data)
        for data_part in data_part_list:
            if pipeline["crop"]:
                cropped_parts = pipeline["crop"](data_part)
            else:
                cropped_parts = [data_part]
            input_dict_list.extend(cropped_parts)
    return [pipeline["post_transform"](part) for part in input_dict_list]


@torch.no_grad()
def predict(model, cfg, pipeline, data_dict, device):
    fragments = prepare_fragments(data_dict, pipeline)
    num_points = data_dict["coord"].shape[0]
    score = torch.zeros((num_points, cfg.data.num_classes), device=device)

    for fragment in fragments:
        input_dict = collate_fn([fragment])
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(device, non_blocking=True)
        idx_part = input_dict["index"].long()
        logits = model(input_dict)["seg_logits"]
        prob = F.softmax(logits, dim=-1)
        start = 0
        for end in input_dict["offset"]:
            end = int(end.item())
            score[idx_part[start:end], :] += prob[start:end]
            start = end

    pred = score.argmax(dim=1).cpu().numpy().astype(np.int64)
    conf = score.max(dim=1).values.cpu().numpy().astype(np.float32)
    return pred, conf


def make_stage2_roi(tile_data, stage1_pred, args):
    coord = tile_data["coord"]
    target_mask = stage1_pred == 1
    target_count = int(target_mask.sum())
    if target_count < args.min_stage1_target_points:
        return None, None, target_count

    target_coord = coord[target_mask]
    xyz_min = target_coord.min(axis=0).astype(np.float32)
    xyz_max = target_coord.max(axis=0).astype(np.float32)
    xyz_min[:2] -= args.xy_margin
    xyz_max[:2] += args.xy_margin
    xyz_min[2] -= args.z_margin
    xyz_max[2] += args.z_margin

    roi_mask = (
        (coord[:, 0] >= xyz_min[0])
        & (coord[:, 0] <= xyz_max[0])
        & (coord[:, 1] >= xyz_min[1])
        & (coord[:, 1] <= xyz_max[1])
        & (coord[:, 2] >= xyz_min[2])
        & (coord[:, 2] <= xyz_max[2])
    )
    if int(roi_mask.sum()) < args.min_roi_points:
        return None, roi_mask, target_count

    roi_data = dict(
        coord=tile_data["coord"][roi_mask].astype(np.float32),
        color=tile_data["color"][roi_mask].astype(np.uint8),
        segment=np.full(int(roi_mask.sum()), -1, dtype=np.int64),
    )
    return roi_data, roi_mask, target_count


def fuse_predictions(stage1_pred, stage2_pred, stage2_conf, roi_mask, args):
    final = STAGE1_TO_FINAL[stage1_pred].copy()
    if stage2_pred is None or roi_mask is None:
        return final

    roi_indices = np.where(roi_mask)[0]
    mapped = STAGE2_TO_FINAL[stage2_pred]
    foreground = mapped >= 0
    if args.stage2_foreground_threshold > 0:
        foreground &= stage2_conf >= args.stage2_foreground_threshold
    final[roi_indices[foreground]] = mapped[foreground]

    if args.stage2_background_policy == "other":
        background = stage2_pred == 3
        stage1_target_in_roi = stage1_pred[roi_indices] == 1
        final[roi_indices[background & stage1_target_in_roi]] = 5

    return final


def collect_input_paths(args):
    if args.input:
        paths = args.input
    else:
        paths = sorted((args.data_root / args.split).glob("*.pth"))
    paths = [Path(p) for p in paths]
    if not paths:
        raise FileNotFoundError("No input .pth files found.")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return paths


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.save_intermediate:
        (args.out / "stage1").mkdir(parents=True, exist_ok=True)
        (args.out / "stage2_roi").mkdir(parents=True, exist_ok=True)

    stage1_cfg = load_cfg(args.stage1_config)
    stage2_cfg = load_cfg(args.stage2_config)
    stage1_model = load_model(stage1_cfg, args.stage1_weight, device)
    stage2_model = load_model(stage2_cfg, args.stage2_weight, device)
    stage1_pipeline = build_test_pipeline(stage1_cfg)
    stage2_pipeline = build_test_pipeline(stage2_cfg)

    summary = []
    for tile_path in collect_input_paths(args):
        out_path = args.out / f"{tile_path.stem}_pred.npy"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path}")
            continue

        tile_data = load_tile(tile_path)
        stage1_pred, _ = predict(stage1_model, stage1_cfg, stage1_pipeline, tile_data, device)
        roi_data, roi_mask, target_count = make_stage2_roi(tile_data, stage1_pred, args)

        stage2_pred = None
        stage2_conf = None
        roi_points = int(roi_mask.sum()) if roi_mask is not None else 0
        if roi_data is not None:
            stage2_pred, stage2_conf = predict(stage2_model, stage2_cfg, stage2_pipeline, roi_data, device)

        final_pred = fuse_predictions(stage1_pred, stage2_pred, stage2_conf, roi_mask, args)
        np.save(out_path, final_pred.astype(np.int64))

        if args.save_intermediate:
            np.save(args.out / "stage1" / f"{tile_path.stem}_stage1_pred.npy", stage1_pred)
            if stage2_pred is not None:
                np.save(args.out / "stage2_roi" / f"{tile_path.stem}_stage2_roi_pred.npy", stage2_pred)
                np.save(args.out / "stage2_roi" / f"{tile_path.stem}_stage2_roi_mask.npy", roi_mask)

        counts = np.bincount(final_pred, minlength=6)
        item = dict(
            tile=tile_path.name,
            points=int(tile_data["coord"].shape[0]),
            stage1_tower_structure_points=int(target_count),
            roi_points=roi_points,
            ran_stage2=stage2_pred is not None,
            final_counts={str(i): int(counts[i]) for i in range(6)},
        )
        summary.append(item)
        print(
            "[done] {} -> {} | points={} stage1_tower_structure={} roi={} stage2={}".format(
                tile_path.name,
                out_path,
                item["points"],
                item["stage1_tower_structure_points"],
                item["roi_points"],
                item["ran_stage2"],
            )
        )

    with (args.out / "two_stage_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            dict(
                stage1_config=str(args.stage1_config),
                stage1_weight=str(args.stage1_weight),
                stage2_config=str(args.stage2_config),
                stage2_weight=str(args.stage2_weight),
                data_root=str(args.data_root),
                split=args.split,
                xy_margin=args.xy_margin,
                z_margin=args.z_margin,
                stage2_background_policy=args.stage2_background_policy,
                tiles=summary,
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
