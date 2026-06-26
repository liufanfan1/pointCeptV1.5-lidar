#!/usr/bin/env python
"""输电线路两阶段推理脚本：Stage-1 粗分 + Stage-2 ROI 精分 + 6 类合成。

用途：
    用 Stage-1 在原始 tile 上先找 ground/tower_structure/line/other，再对
    tower_structure 附近的 ROI 调 Stage-2 精分 tower/insulator/hengdan，
    最后合成原始 6 类预测。适合绝缘子、横担这类小目标需要局部精分的版本。
输入：
    原始 6 类 .pth 数据、Stage-1 config/weight、Stage-2 config/weight。
输出：
    最终 6 类 *_pred.npy，可直接用 export_transmission_line_pred_ply.py 可视化。

Stage-1 4 类：
    0 ground, 1 tower_structure, 2 line, 3 other
Stage-2 4 类：
    0 tower, 1 insulator, 2 hengdan, 3 background
最终 6 类：
    0 ground, 1 tower, 2 line, 3 insulator, 4 hengdan, 5 other
"""

import argparse
import copy
import json
import sys
import time
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
    parser.add_argument(
        "--split", default="test", help="Dataset split under data-root."
    )
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
        default=Path(
            "exp/transmission/stage1_4cls_balance_w8_clean/model/model_best.pth"
        ),
    )
    parser.add_argument(
        "--stage1-pred-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing existing Stage1 *_pred.npy files. "
            "When provided, Stage1 model inference is skipped for matching tiles."
        ),
    )
    parser.add_argument(
        "--stage2-config",
        type=Path,
        default=Path("exp/transmission/stage2_tower_ins_centered_w24/config.py"),
    )
    parser.add_argument(
        "--stage2-weight",
        type=Path,
        default=Path(
            "exp/transmission/stage2_tower_ins_centered_w24/model/model_best.pth"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("exp/transmission/two_stage_infer/result"),
        help="Output directory for final *_pred.npy files.",
    )
    parser.add_argument(
        "--xy-margin", type=float, default=8.0, help="Stage2 ROI XY margin in meters."
    )
    parser.add_argument(
        "--z-margin", type=float, default=3.0, help="Stage2 ROI Z margin in meters."
    )
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
        "--stage2-overwrite-scope",
        choices=("tower_structure", "roi"),
        default="tower_structure",
        help=(
            "Controls where Stage2 can overwrite final labels. "
            "'tower_structure' only overwrites points predicted as Stage1 tower_structure; "
            "'roi' overwrites all points inside the generated ROI."
        ),
    )
    parser.add_argument(
        "--fast-crop",
        action="store_true",
        help=(
            "Use simple point chunks instead of SphereCrop(mode='all') during inference. "
            "This is much faster for million-point tiles."
        ),
    )
    parser.add_argument(
        "--fast-crop-point-max",
        type=int,
        default=65536,
        help="Maximum points per chunk when --fast-crop is enabled.",
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
    parser.add_argument(
        "--device", default="cuda", help="Inference device, usually cuda."
    )
    return parser.parse_args()


def load_cfg(path):
    cfg = Config.fromfile(str(path))
    cfg.model.criteria = []
    return cfg


def now_text():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def format_seconds(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{seconds:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes)}m{seconds:.1f}s"


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_cuda_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def cuda_memory_stats(device):
    if device.type != "cuda":
        return None
    mib = 1024**2
    return dict(
        allocated_mb=round(torch.cuda.memory_allocated(device) / mib, 2),
        reserved_mb=round(torch.cuda.memory_reserved(device) / mib, 2),
        max_allocated_mb=round(torch.cuda.max_memory_allocated(device) / mib, 2),
        max_reserved_mb=round(torch.cuda.max_memory_reserved(device) / mib, 2),
    )


def format_cuda_memory(stats):
    if stats is None:
        return "gpu_mem=n/a"
    return (
        "gpu_mem="
        f"alloc={stats['allocated_mb']:.2f}MiB "
        f"reserved={stats['reserved_mb']:.2f}MiB "
        f"peak_alloc={stats['max_allocated_mb']:.2f}MiB "
        f"peak_reserved={stats['max_reserved_mb']:.2f}MiB"
    )


def scene_name_from_tile(tile_name):
    stem = Path(tile_name).stem
    if "_tile" in stem:
        return stem.split("_tile", 1)[0]
    return stem


def summarize_runtime(tile_summary):
    scene_stats = OrderedDict()
    peak_memory = None
    total_tile_elapsed_sec = 0.0

    for item in tile_summary:
        elapsed_sec = float(item.get("elapsed_sec", 0.0))
        total_tile_elapsed_sec += elapsed_sec

        scene_name = item.get("scene") or scene_name_from_tile(item["tile"])
        if scene_name not in scene_stats:
            scene_stats[scene_name] = dict(
                scene=scene_name,
                tile_count=0,
                total_elapsed_sec=0.0,
                avg_tile_elapsed_sec=0.0,
                stage_time_totals={},
                stage_time_avg={},
            )
        scene_item = scene_stats[scene_name]
        scene_item["tile_count"] += 1
        scene_item["total_elapsed_sec"] += elapsed_sec
        for key, value in item.get("times", {}).items():
            scene_item["stage_time_totals"][key] = scene_item["stage_time_totals"].get(
                key, 0.0
            ) + float(value)

        memory = item.get("cuda_memory")
        if memory is not None:
            if peak_memory is None:
                peak_memory = memory.copy()
            else:
                for key, value in memory.items():
                    peak_memory[key] = max(
                        float(peak_memory.get(key, 0.0)), float(value)
                    )

    for scene_item in scene_stats.values():
        tile_count = max(scene_item["tile_count"], 1)
        scene_item["total_elapsed_sec"] = round(scene_item["total_elapsed_sec"], 4)
        scene_item["avg_tile_elapsed_sec"] = round(
            scene_item["total_elapsed_sec"] / tile_count, 4
        )
        scene_item["stage_time_totals"] = {
            key: round(value, 4)
            for key, value in scene_item["stage_time_totals"].items()
        }
        scene_item["stage_time_avg"] = {
            key: round(value / tile_count, 4)
            for key, value in scene_item["stage_time_totals"].items()
        }

    processed_tiles = len(tile_summary)
    return dict(
        processed_tiles=processed_tiles,
        total_tile_elapsed_sec=round(total_tile_elapsed_sec, 4),
        avg_tile_elapsed_sec=(
            round(total_tile_elapsed_sec / processed_tiles, 4)
            if processed_tiles
            else 0.0
        ),
        scenes=list(scene_stats.values()),
        cuda_memory_peak=peak_memory,
    )


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
        segment = (
            numpy_or_default(data, "semantic_gt", None).astype(np.int64).reshape(-1)
        )
    if segment.shape[0] != coord.shape[0]:
        raise ValueError(f"{path} coord/semantic_gt point count mismatch.")
    return dict(coord=coord, color=color, segment=segment)


def split_data_part(data_part, point_max):
    if "coord" not in data_part or data_part["coord"].shape[0] <= point_max:
        return [data_part]

    num_points = data_part["coord"].shape[0]
    chunks = []
    for start in range(0, num_points, point_max):
        end = min(start + point_max, num_points)
        chunk = {}
        for key, value in data_part.items():
            if (
                hasattr(value, "shape")
                and len(value.shape) > 0
                and value.shape[0] == num_points
            ):
                chunk[key] = value[start:end]
            else:
                chunk[key] = value
        chunks.append(chunk)
    return chunks


def prepare_fragments(data_dict, pipeline, fast_crop=False, fast_crop_point_max=65536):
    base = pipeline["transform"](copy.deepcopy(data_dict))
    if fast_crop:
        coord = base["coord"]
        grid_size = pipeline["voxelize"].grid_size
        grid_coord = np.floor(coord / np.array(grid_size)).astype(np.int64)
        grid_coord -= grid_coord.min(0)
        base["grid_coord"] = grid_coord
        base["index"] = np.arange(coord.shape[0])
        return [
            pipeline["post_transform"](part)
            for part in split_data_part(base, fast_crop_point_max)
        ]

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
def predict(model, cfg, pipeline, data_dict, device, args=None):
    fast_crop = bool(args.fast_crop) if args is not None else False
    fast_crop_point_max = args.fast_crop_point_max if args is not None else 65536
    fragments = prepare_fragments(
        data_dict,
        pipeline,
        fast_crop=fast_crop,
        fast_crop_point_max=fast_crop_point_max,
    )
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


def load_stage1_prediction(pred_dir, tile_path, num_points):
    pred_path = pred_dir / f"{tile_path.stem}_pred.npy"
    if not pred_path.exists():
        return None
    pred = np.load(pred_path).astype(np.int64).reshape(-1)
    if pred.shape[0] != num_points:
        raise ValueError(
            f"{pred_path} has {pred.shape[0]} predictions, "
            f"but {tile_path} has {num_points} points."
        )
    return pred


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
    if args.stage2_overwrite_scope == "tower_structure":
        foreground &= stage1_pred[roi_indices] == 1
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
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    run_start = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.save_intermediate:
        (args.out / "stage1").mkdir(parents=True, exist_ok=True)
        (args.out / "stage2_roi").mkdir(parents=True, exist_ok=True)

    print(f"[init] {now_text()} device={device}", flush=True)
    print(
        "[init] paths | stage1_config={} stage1_weight={} stage1_pred_dir={} "
        "stage2_config={} stage2_weight={} out={}".format(
            args.stage1_config,
            args.stage1_weight,
            args.stage1_pred_dir,
            args.stage2_config,
            args.stage2_weight,
            args.out,
        ),
        flush=True,
    )
    stage1_cfg = None
    if args.stage1_pred_dir is None:
        stage1_cfg = load_cfg(args.stage1_config)
    stage2_cfg = load_cfg(args.stage2_config)
    stage1_model = None
    if args.stage1_pred_dir is None:
        stage1_model = load_model(stage1_cfg, args.stage1_weight, device)
    stage2_model = load_model(stage2_cfg, args.stage2_weight, device)
    stage1_pipeline = build_test_pipeline(stage1_cfg) if stage1_model else None
    stage2_pipeline = build_test_pipeline(stage2_cfg)
    sync_if_cuda(device)
    print(
        f"[init] {now_text()} models_loaded | "
        f"{format_cuda_memory(cuda_memory_stats(device))}",
        flush=True,
    )

    summary = []
    input_paths = collect_input_paths(args)
    for tile_idx, tile_path in enumerate(input_paths, start=1):
        out_path = args.out / f"{tile_path.stem}_pred.npy"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {tile_idx}/{len(input_paths)} {out_path}", flush=True)
            continue

        reset_cuda_peak(device)
        sync_if_cuda(device)
        tile_start = time.perf_counter()
        stage_times = {}

        print(
            f"[start] {now_text()} {tile_idx}/{len(input_paths)} {tile_path.name}",
            flush=True,
        )

        step_start = time.perf_counter()
        tile_data = load_tile(tile_path)
        stage_times["load_sec"] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        stage1_pred = None
        if args.stage1_pred_dir is not None:
            stage1_pred = load_stage1_prediction(
                args.stage1_pred_dir, tile_path, tile_data["coord"].shape[0]
            )
            if stage1_pred is None:
                raise FileNotFoundError(
                    "Missing Stage1 prediction for {} in {}. "
                    "Either generate Stage1 *_pred.npy files first, or remove "
                    "--stage1-pred-dir and provide --stage1-config/--stage1-weight.".format(
                        tile_path.name, args.stage1_pred_dir
                    )
                )
        if stage1_pred is None:
            sync_if_cuda(device)
            step_start = time.perf_counter()
            stage1_pred, _ = predict(
                stage1_model, stage1_cfg, stage1_pipeline, tile_data, device, args
            )
            sync_if_cuda(device)
        stage_times["stage1_sec"] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        roi_data, roi_mask, target_count = make_stage2_roi(tile_data, stage1_pred, args)
        stage_times["roi_sec"] = time.perf_counter() - step_start

        stage2_pred = None
        stage2_conf = None
        roi_points = int(roi_mask.sum()) if roi_mask is not None else 0
        step_start = time.perf_counter()
        if roi_data is not None:
            sync_if_cuda(device)
            step_start = time.perf_counter()
            stage2_pred, stage2_conf = predict(
                stage2_model, stage2_cfg, stage2_pipeline, roi_data, device, args
            )
            sync_if_cuda(device)
        stage_times["stage2_sec"] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        final_pred = fuse_predictions(
            stage1_pred, stage2_pred, stage2_conf, roi_mask, args
        )
        np.save(out_path, final_pred.astype(np.int64))
        stage_times["fuse_save_sec"] = time.perf_counter() - step_start

        if args.save_intermediate:
            step_start = time.perf_counter()
            np.save(
                args.out / "stage1" / f"{tile_path.stem}_stage1_pred.npy", stage1_pred
            )
            if stage2_pred is not None:
                np.save(
                    args.out / "stage2_roi" / f"{tile_path.stem}_stage2_roi_pred.npy",
                    stage2_pred,
                )
                np.save(
                    args.out / "stage2_roi" / f"{tile_path.stem}_stage2_roi_mask.npy",
                    roi_mask,
                )
            stage_times["save_intermediate_sec"] = time.perf_counter() - step_start

        sync_if_cuda(device)
        elapsed_sec = time.perf_counter() - tile_start
        memory_stats = cuda_memory_stats(device)
        counts = np.bincount(final_pred, minlength=6)
        item = dict(
            tile=tile_path.name,
            scene=scene_name_from_tile(tile_path.name),
            points=int(tile_data["coord"].shape[0]),
            stage1_tower_structure_points=int(target_count),
            roi_points=roi_points,
            ran_stage2=stage2_pred is not None,
            elapsed_sec=round(elapsed_sec, 4),
            times={key: round(value, 4) for key, value in stage_times.items()},
            cuda_memory=memory_stats,
            final_counts={str(i): int(counts[i]) for i in range(6)},
        )
        summary.append(item)
        print(
            (
                "[done] {} -> {} | points={} stage1_tower_structure={} "
                "roi={} stage2={} elapsed={} load={} stage1={} "
                "roi_make={} stage2={} fuse_save={} | {}"
            ).format(
                tile_path.name,
                out_path,
                item["points"],
                item["stage1_tower_structure_points"],
                item["roi_points"],
                item["ran_stage2"],
                format_seconds(elapsed_sec),
                format_seconds(stage_times.get("load_sec", 0.0)),
                format_seconds(stage_times.get("stage1_sec", 0.0)),
                format_seconds(stage_times.get("roi_sec", 0.0)),
                format_seconds(stage_times.get("stage2_sec", 0.0)),
                format_seconds(stage_times.get("fuse_save_sec", 0.0)),
                format_cuda_memory(memory_stats),
            ),
            flush=True,
        )

    sync_if_cuda(device)
    total_elapsed_sec = time.perf_counter() - run_start
    runtime_summary = summarize_runtime(summary)
    final_memory_stats = cuda_memory_stats(device)
    with (args.out / "two_stage_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            dict(
                stage1_config=str(args.stage1_config),
                stage1_weight=str(args.stage1_weight),
                stage1_pred_dir=(
                    str(args.stage1_pred_dir)
                    if args.stage1_pred_dir is not None
                    else None
                ),
                stage2_config=str(args.stage2_config),
                stage2_weight=str(args.stage2_weight),
                data_root=str(args.data_root),
                split=args.split,
                xy_margin=args.xy_margin,
                z_margin=args.z_margin,
                stage2_background_policy=args.stage2_background_policy,
                total_elapsed_sec=round(total_elapsed_sec, 4),
                cuda_memory=final_memory_stats,
                runtime=runtime_summary,
                tiles=summary,
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )
    peak_memory_text = format_cuda_memory(runtime_summary["cuda_memory_peak"])
    print(
        f"[summary] total_elapsed={format_seconds(total_elapsed_sec)} "
        f"total_tile_elapsed={format_seconds(runtime_summary['total_tile_elapsed_sec'])} "
        f"avg_tile_elapsed={format_seconds(runtime_summary['avg_tile_elapsed_sec'])} "
        f"processed_tiles={len(summary)} skipped_tiles={len(input_paths) - len(summary)} | "
        f"peak_{peak_memory_text} | final_{format_cuda_memory(final_memory_stats)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
