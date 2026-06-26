"""Run Pointcept semantic segmentation on LAS files and save predicted LAS.

The script reads one LAS file or a directory of LAS files, runs the configured
Pointcept model, and writes predictions into the LAS classification field.
It prefers laspy when installed and falls back to a small LAS 1.2 reader/writer
for common point formats 0-3.
"""

import argparse
import copy
import shutil
import struct
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F

from pointcept.datasets.transform import Compose, TRANSFORMS
from pointcept.datasets.utils import collate_fn
from pointcept.models import build_model
from pointcept.utils.config import Config

# 配置文件
DEFAULT_CONFIG = (
    "exp/transmission_line/ptv3-4cls-ins-oversample_v2/config.py"
)
# 权重
DEFAULT_WEIGHT = (
    "exp/transmission_line/ptv3-4cls-ins-oversample_v2/model/model_best.pth"
)

# RGB colors used for direct visualization in CloudCompare and similar tools.
CLASS_COLOR_8BIT = np.array(
    [
        [145, 145, 145],  # 0 ground
        [230, 65, 55],    # 1 tower
        [45, 125, 255],   # 2 line
        [255, 210, 40],   # 3 insulator
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer transmission-line semantic labels for LAS files."
    )
    parser.add_argument("--input", required=True, help="Input .las file or directory.")
    parser.add_argument("--output", required=True, help="Output .las file or directory.")
    parser.add_argument("--config-file", default=DEFAULT_CONFIG, help="Pointcept config.")
    parser.add_argument("--weight", default=DEFAULT_WEIGHT, help="Checkpoint path.")
    parser.add_argument(
        "--las-backend",
        choices=("auto", "laspy", "fallback"),
        default="auto",
        help="LAS IO backend. auto uses the faster fallback for supported .las files.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--disable-flash",
        action="store_true",
        help="Set cfg.model.backbone.enable_flash=False. Useful on Windows without flash-attn.",
    )
    parser.add_argument(
        "--fragment-batch-size",
        type=int,
        default=8,
        help="Number of cropped fragments sent to the model at once.",
    )
    parser.add_argument(
        "--point-max",
        type=int,
        default=None,
        help="Override test SphereCrop point_max from config.",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        default=None,
        help="Override test GridSample grid_size from config.",
    )
    parser.add_argument(
        "--tile-size",
        type=float,
        default=40.0,
        help="XY tile size in meters. Use <=0 to infer the whole LAS at once.",
    )
    parser.add_argument(
        "--tile-stride",
        type=float,
        default=40.0,
        help="XY tile stride in meters. Default matches the training eval tiles.",
    )
    parser.add_argument(
        "--merge-mode",
        choices=("plain", "halo", "overlap"),
        default="plain",
        help=(
            "Tile merge mode. plain is current non-overlap behavior; "
            "halo uses context around each core tile and writes only the core; "
            "overlap averages probabilities from overlapping tiles."
        ),
    )
    parser.add_argument(
        "--context-margin",
        type=float,
        default=0.0,
        help="Halo context margin in meters for --merge-mode halo.",
    )
    parser.add_argument(
        "--min-tile-points",
        type=int,
        default=1,
        help="Skip tiles with fewer points than this value.",
    )
    parser.add_argument(
        "--pre-voxel-size",
        type=float,
        default=0.0,
        help=(
            "Voxel size before model inference inside each tile. "
            "Use 0.05 to greatly speed up dense LAS inference; "
            "predictions are mapped back to all original points."
        ),
    )
    parser.add_argument(
        "--default-color",
        type=int,
        nargs=3,
        default=(255, 255, 255),
        metavar=("R", "G", "B"),
        help="RGB value used when the input LAS has no color fields.",
    )
    parser.add_argument(
        "--no-colorize",
        action="store_true",
        help="Do not overwrite LAS RGB values with prediction colors.", 
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files."
    )
    return parser.parse_args()


def try_import_laspy():
    try:
        import laspy  # noqa: WPS433
    except ImportError:
        return None
    return laspy


def read_las_header(path):
    with path.open("rb") as file:
        header = file.read(375)
    if header[:4] != b"LASF":
        raise ValueError(f"Not a LAS file: {path}")
    version = (header[24], header[25])
    if version > (1, 4):
        raise ValueError(f"Unsupported LAS version {version[0]}.{version[1]}: {path}")
    header_size = struct.unpack_from("<H", header, 94)[0]
    point_offset = struct.unpack_from("<I", header, 96)[0]
    point_format = header[104] & 0x3F
    record_length = struct.unpack_from("<H", header, 105)[0]
    point_count = struct.unpack_from("<I", header, 107)[0]
    if point_count == 0 and header_size >= 375:
        point_count = struct.unpack_from("<Q", header, 247)[0]
    scale = np.array(struct.unpack_from("<3d", header, 131), dtype=np.float64)
    offset = np.array(struct.unpack_from("<3d", header, 155), dtype=np.float64)
    return dict(
        point_offset=point_offset,
        point_format=point_format,
        record_length=record_length,
        point_count=point_count,
        scale=scale,
        offset=offset,
    )


def color_from_16bit(color_16):
    if color_16.size == 0:
        return color_16.astype(np.uint8)
    if color_16.max(initial=0) > 255:
        return (color_16 >> 8).astype(np.uint8)
    return color_16.astype(np.uint8)


def read_las_fallback(path, default_color):
    header = read_las_header(path)
    point_format = header["point_format"]
    if point_format not in (0, 1, 2, 3):
        raise ValueError(
            "laspy is not installed and the fallback reader only supports "
            f"LAS point formats 0-3, got format {point_format}: {path}"
        )

    count = header["point_count"]
    record_length = header["record_length"]
    raw = np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=header["point_offset"],
        shape=(count * record_length,),
    )
    xyz_integer = np.stack(
        [
            np.ndarray(
                (count,),
                dtype="<i4",
                buffer=raw,
                offset=i * 4,
                strides=(record_length,),
            )
            for i in range(3)
        ],
        axis=1,
    )
    coord = xyz_integer.astype(np.float64) * header["scale"] + header["offset"]
    coord -= coord.min(axis=0)

    color_offset = 20 if point_format == 2 else 28 if point_format == 3 else None
    if color_offset is None:
        color = np.tile(np.asarray(default_color, dtype=np.uint8), (count, 1))
    else:
        color_16 = np.stack(
            [
                np.ndarray(
                    (count,),
                    dtype="<u2",
                    buffer=raw,
                    offset=color_offset + i * 2,
                    strides=(record_length,),
                )
                for i in range(3)
            ],
            axis=1,
        )
        color = color_from_16bit(color_16)
    return coord.astype(np.float32), color.astype(np.float32), None


def read_las(path, default_color, backend="auto"):
    if backend == "fallback" or (backend == "auto" and path.suffix.lower() == ".las"):
        try:
            return read_las_fallback(path, default_color)
        except ValueError:
            if backend == "fallback":
                raise

    laspy = try_import_laspy()
    if laspy is None:
        return read_las_fallback(path, default_color)

    las = laspy.read(path)
    coord = np.stack([las.x, las.y, las.z], axis=1)
    coord -= coord.min(axis=0)
    coord = coord.astype(np.float32)
    dimension_names = set(las.point_format.dimension_names)
    if {"red", "green", "blue"}.issubset(dimension_names):
        color_16 = np.stack([las.red, las.green, las.blue], axis=1)
        color = color_from_16bit(color_16).astype(np.float32)
    else:
        color = np.tile(np.asarray(default_color, dtype=np.float32), (len(las.x), 1))
    return coord, color, las


def pred_to_color16(pred):
    colors = CLASS_COLOR_8BIT[np.clip(pred, 0, len(CLASS_COLOR_8BIT) - 1)]
    return (colors.astype(np.uint16) * 257).astype(np.uint16)


def write_las_fallback(input_path, output_path, pred, colorize=True):
    header = read_las_header(input_path)
    point_format = header["point_format"]
    if point_format not in (0, 1, 2, 3):
        raise ValueError(
            "laspy is not installed and the fallback writer only supports "
            f"LAS point formats 0-3, got format {point_format}: {input_path}"
        )
    if pred.shape[0] != header["point_count"]:
        raise ValueError(
            f"Prediction count {pred.shape[0]} does not match LAS point count "
            f"{header['point_count']}: {input_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, output_path)
    raw = np.memmap(
        output_path,
        dtype=np.uint8,
        mode="r+",
        offset=header["point_offset"],
        shape=(header["point_count"] * header["record_length"],),
    )
    classification = np.ndarray(
        (header["point_count"],),
        dtype=np.uint8,
        buffer=raw,
        offset=15,
        strides=(header["record_length"],),
    )
    classification[:] = pred.astype(np.uint8, copy=False)
    if colorize and point_format in (2, 3):
        color_offset = 20 if point_format == 2 else 28
        colors = pred_to_color16(pred)
        for channel in range(3):
            channel_view = np.ndarray(
                (header["point_count"],),
                dtype="<u2",
                buffer=raw,
                offset=color_offset + channel * 2,
                strides=(header["record_length"],),
            )
            channel_view[:] = colors[:, channel]
    raw.flush()


def write_las(input_path, output_path, las, pred, colorize=True):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if las is None:
        write_las_fallback(input_path, output_path, pred, colorize=colorize)
        return
    las.classification = pred.astype(np.uint8)
    if colorize:
        dimension_names = set(las.point_format.dimension_names)
        if {"red", "green", "blue"}.issubset(dimension_names):
            colors = pred_to_color16(pred)
            las.red = colors[:, 0]
            las.green = colors[:, 1]
            las.blue = colors[:, 2]
        else:
            print(
                "Warning: LAS point format has no RGB fields; only classification was written.",
                flush=True,
            )
    las.write(output_path)


def set_enable_flash(cfg, enabled):
    try:
        cfg.model.backbone.enable_flash = bool(enabled)
    except Exception:
        pass


def load_model(cfg, weight_path, device):
    model = build_model(cfg.model)
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned_state_dict[key] = value
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def build_test_pipeline(cfg, point_max=None, grid_size=None):
    test_cfg = copy.deepcopy(cfg.data.test.test_cfg)
    if grid_size is not None:
        test_cfg.voxelize.grid_size = grid_size
    if point_max is not None:
        test_cfg.crop.point_max = point_max
    return dict(
        transform=Compose(cfg.data.test.transform),
        voxelize=TRANSFORMS.build(test_cfg.voxelize),
        crop=TRANSFORMS.build(test_cfg.crop),
        post_transform=Compose(test_cfg.post_transform),
        aug_transform=[Compose(aug) for aug in test_cfg.aug_transform],
    )


def make_fragments(coord, color, pipeline):
    data_dict = dict(
        coord=coord.copy(),
        color=color.copy(),
        segment=np.full(coord.shape[0], -1, dtype=np.int64),
    )
    data_dict = pipeline["transform"](data_dict)
    data_dict.pop("segment", None)

    fragment_list = []
    for aug in pipeline["aug_transform"]:
        aug_data = aug(copy.deepcopy(data_dict))
        data_part_list = pipeline["voxelize"](aug_data)
        for data_part in data_part_list:
            fragment_list.extend(pipeline["crop"](data_part))

    for idx, fragment in enumerate(fragment_list):
        fragment_list[idx] = pipeline["post_transform"](fragment)
    return fragment_list


def move_tensors_to_device(input_dict, device):
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            input_dict[key] = value.to(device, non_blocking=True)
    return input_dict


@torch.inference_mode()
def infer_las_probs(model, cfg, pipeline, coord, color, device, fragment_batch_size):
    if coord.shape[0] == 0:
        return np.zeros((0, cfg.data.num_classes), dtype=np.float32)

    fragment_list = make_fragments(coord, color, pipeline)
    print(f"    fragments: {len(fragment_list)}", flush=True)
    pred = torch.zeros((coord.shape[0], cfg.data.num_classes), device=device)
    for start in range(0, len(fragment_list), fragment_batch_size):
        batch = fragment_list[start : start + fragment_batch_size]
        input_dict = move_tensors_to_device(collate_fn(batch), device)
        index = input_dict["index"]
        logits = model(input_dict)["seg_logits"]
        prob = F.softmax(logits, dim=-1)
        begin = 0
        for end in input_dict["offset"]:
            pred[index[begin:end], :] += prob[begin:end]
            begin = end
    return pred.cpu().numpy().astype(np.float32, copy=False)


def infer_las(model, cfg, pipeline, coord, color, device, fragment_batch_size):
    prob = infer_las_probs(model, cfg, pipeline, coord, color, device, fragment_batch_size)
    return prob.argmax(axis=1).astype(np.uint8, copy=False)


def tile_starts(min_value, max_value, tile_size, tile_stride):
    if tile_size <= 0:
        return np.array([min_value], dtype=np.float64)
    starts = np.arange(min_value, max_value + 1e-6, tile_stride, dtype=np.float64)
    if starts.size == 0:
        starts = np.array([min_value], dtype=np.float64)
    return starts


def non_overlapping_tile_groups(coord, tile_size):
    min_xy = coord[:, :2].min(axis=0)
    tile_xy = np.floor((coord[:, :2] - min_xy) / tile_size).astype(np.int64)
    x_count = int(tile_xy[:, 0].max()) + 1
    y_count = int(tile_xy[:, 1].max()) + 1
    tile_id = tile_xy[:, 0] * y_count + tile_xy[:, 1]
    order = np.argsort(tile_id, kind="mergesort")
    sorted_id = tile_id[order]
    unique_id, starts, counts = np.unique(
        sorted_id, return_index=True, return_counts=True
    )
    return min_xy, x_count, y_count, order, unique_id, starts, counts


def voxel_reduce_points(coord, color, voxel_size):
    if voxel_size <= 0 or coord.shape[0] == 0:
        return coord, color, None
    grid_coord = np.floor((coord - coord.min(axis=0)) / voxel_size).astype(np.int64)
    _, unique_index, inverse = np.unique(
        grid_coord, axis=0, return_index=True, return_inverse=True
    )
    return coord[unique_index], color[unique_index], inverse


def infer_tile_probs(
    model, cfg, pipeline, coord, color, device, fragment_batch_size, pre_voxel_size
):
    if pre_voxel_size <= 0:
        return infer_las_probs(model, cfg, pipeline, coord, color, device, fragment_batch_size)
    reduce_start = time.perf_counter()
    reduced_coord, reduced_color, inverse = voxel_reduce_points(coord, color, pre_voxel_size)
    print(
        f"    pre-voxel {coord.shape[0]} -> {reduced_coord.shape[0]} "
        f"points in {time.perf_counter() - reduce_start:.2f}s",
        flush=True,
    )
    reduced_prob = infer_las_probs(
        model, cfg, pipeline, reduced_coord, reduced_color, device, fragment_batch_size
    )
    return reduced_prob[inverse].astype(np.float32, copy=False)


def infer_tile_points(
    model, cfg, pipeline, coord, color, device, fragment_batch_size, pre_voxel_size
):
    prob = infer_tile_probs(
        model, cfg, pipeline, coord, color, device, fragment_batch_size, pre_voxel_size
    )
    return prob.argmax(axis=1).astype(np.uint8, copy=False)


def tile_grid_bounds(coord, tile_size, tile_stride):
    x_starts = tile_starts(float(coord[:, 0].min()), float(coord[:, 0].max()), tile_size, tile_stride)
    y_starts = tile_starts(float(coord[:, 1].min()), float(coord[:, 1].max()), tile_size, tile_stride)
    return x_starts, y_starts


def points_in_xy_box(coord, x0, x1, y0, y1):
    return (coord[:, 0] >= x0) & (coord[:, 0] <= x1) & (coord[:, 1] >= y0) & (coord[:, 1] <= y1)


def infer_las_tiled(
    model,
    cfg,
    pipeline,
    coord,
    color,
    device,
    fragment_batch_size,
    tile_size,
    tile_stride,
    min_tile_points,
    pre_voxel_size,
    merge_mode="plain",
    context_margin=0.0,
):
    if tile_size <= 0 or coord.shape[0] <= pipeline["crop"].point_max:
        print(f"  infer whole cloud: {coord.shape[0]} points", flush=True)
        return infer_tile_points(
            model, cfg, pipeline, coord, color, device, fragment_batch_size, pre_voxel_size
        )

    if tile_stride <= 0:
        raise ValueError("--tile-stride must be > 0 when --tile-size is enabled")

    if merge_mode not in ("plain", "halo", "overlap"):
        raise ValueError(f"Unsupported merge mode: {merge_mode}")
    if merge_mode == "halo" and context_margin <= 0:
        raise ValueError("--context-margin must be > 0 when --merge-mode halo")

    if merge_mode in ("halo", "overlap"):
        x_starts, y_starts = tile_grid_bounds(coord, tile_size, tile_stride)
        total_tiles = len(x_starts) * len(y_starts)
        used_tiles = 0
        done = np.zeros(coord.shape[0], dtype=bool)
        if merge_mode == "overlap":
            prob_sum = np.zeros((coord.shape[0], cfg.data.num_classes), dtype=np.float32)
            vote_count = np.zeros(coord.shape[0], dtype=np.uint16)
        else:
            pred = np.full(coord.shape[0], 0, dtype=np.uint8)
        print(
            f"  tile inference: {coord.shape[0]} points, "
            f"{len(x_starts)} x {len(y_starts)} = {total_tiles} tiles, "
            f"tile_size={tile_size}, stride={tile_stride}, mode={merge_mode}, "
            f"context_margin={context_margin}",
            flush=True,
        )
        tile_idx = 0
        for x0 in x_starts:
            x1 = x0 + tile_size
            for y0 in y_starts:
                tile_idx += 1
                y1 = y0 + tile_size
                if merge_mode == "halo":
                    infer_x0, infer_x1 = x0 - context_margin, x1 + context_margin
                    infer_y0, infer_y1 = y0 - context_margin, y1 + context_margin
                    infer_mask = points_in_xy_box(coord, infer_x0, infer_x1, infer_y0, infer_y1)
                    write_mask = points_in_xy_box(coord, x0, x1, y0, y1) & (~done)
                else:
                    infer_mask = points_in_xy_box(coord, x0, x1, y0, y1)
                    write_mask = infer_mask
                infer_indices = np.flatnonzero(infer_mask)
                if infer_indices.size < min_tile_points:
                    continue
                if merge_mode == "halo":
                    local_write = np.flatnonzero(write_mask[infer_indices])
                    if local_write.size == 0:
                        continue
                used_tiles += 1
                tile_start = time.perf_counter()
                print(
                    f"  tile {tile_idx}/{total_tiles}: infer={infer_indices.size} "
                    f"write={int(write_mask.sum())} "
                    f"(x={x0:.2f}..{x1:.2f}, y={y0:.2f}..{y1:.2f})",
                    flush=True,
                )
                tile_prob = infer_tile_probs(
                    model,
                    cfg,
                    pipeline,
                    coord[infer_indices],
                    color[infer_indices],
                    device,
                    fragment_batch_size,
                    pre_voxel_size,
                )
                if merge_mode == "halo":
                    write_indices = infer_indices[local_write]
                    pred[write_indices] = tile_prob[local_write].argmax(axis=1).astype(np.uint8)
                    done[write_indices] = True
                else:
                    prob_sum[infer_indices] += tile_prob
                    vote_count[infer_indices] += 1
                    done[infer_indices] = True
                print(f"    done in {time.perf_counter() - tile_start:.2f}s", flush=True)
        if merge_mode == "overlap":
            pred = np.full(coord.shape[0], 0, dtype=np.uint8)
            covered = vote_count > 0
            pred[covered] = prob_sum[covered].argmax(axis=1).astype(np.uint8)
            done = covered
        missing = int((~done).sum())
        if missing:
            print(f"  warning: {missing} points were not covered by tiles; kept label 0", flush=True)
        print(f"  finished {used_tiles} non-empty tiles", flush=True)
        return pred

    pred = np.full(coord.shape[0], 0, dtype=np.uint8)
    done = np.zeros(coord.shape[0], dtype=bool)
    used_tiles = 0

    if np.isclose(tile_stride, tile_size):
        group_start = time.perf_counter()
        min_xy, x_count, y_count, order, unique_id, starts, counts = non_overlapping_tile_groups(
            coord, tile_size
        )
        total_tiles = x_count * y_count
        print(
            f"  tile inference: {coord.shape[0]} points, "
            f"{x_count} x {y_count} = {total_tiles} tiles, "
            f"non-empty={len(unique_id)}, tile_size={tile_size}, "
            f"grouping={time.perf_counter() - group_start:.2f}s",
            flush=True,
        )
        for group_idx, (tile_id, start, count) in enumerate(
            zip(unique_id, starts, counts), start=1
        ):
            if count < min_tile_points:
                continue
            used_tiles += 1
            indices = order[start : start + count]
            ix = int(tile_id // y_count)
            iy = int(tile_id % y_count)
            x0 = float(min_xy[0] + ix * tile_size)
            y0 = float(min_xy[1] + iy * tile_size)
            tile_start = time.perf_counter()
            print(
                f"  tile {group_idx}/{len(unique_id)}: {count} points "
                f"(x={x0:.2f}..{x0 + tile_size:.2f}, "
                f"y={y0:.2f}..{y0 + tile_size:.2f})",
                flush=True,
            )
            tile_pred = infer_tile_points(
                model,
                cfg,
                pipeline,
                coord[indices],
                color[indices],
                device,
                fragment_batch_size,
                pre_voxel_size,
            )
            pred[indices] = tile_pred
            done[indices] = True
            print(f"    done in {time.perf_counter() - tile_start:.2f}s", flush=True)
    else:
        x_starts = tile_starts(float(coord[:, 0].min()), float(coord[:, 0].max()), tile_size, tile_stride)
        y_starts = tile_starts(float(coord[:, 1].min()), float(coord[:, 1].max()), tile_size, tile_stride)
        total_tiles = len(x_starts) * len(y_starts)
        tile_idx = 0
        print(
            f"  tile inference: {coord.shape[0]} points, "
            f"{len(x_starts)} x {len(y_starts)} = {total_tiles} tiles, "
            f"tile_size={tile_size}, stride={tile_stride}",
            flush=True,
        )

        for x0 in x_starts:
            x1 = x0 + tile_size
            x_mask = (coord[:, 0] >= x0) & (coord[:, 0] <= x1)
            for y0 in y_starts:
                tile_idx += 1
                y1 = y0 + tile_size
                mask = x_mask & (coord[:, 1] >= y0) & (coord[:, 1] <= y1)
                indices = np.flatnonzero(mask)
                if indices.size < min_tile_points:
                    continue
                used_tiles += 1
                tile_start = time.perf_counter()
                print(
                    f"  tile {tile_idx}/{total_tiles}: {indices.size} points "
                    f"(x={x0:.2f}..{x1:.2f}, y={y0:.2f}..{y1:.2f})",
                    flush=True,
                )
                tile_pred = infer_tile_points(
                    model,
                    cfg,
                    pipeline,
                    coord[indices],
                    color[indices],
                    device,
                    fragment_batch_size,
                    pre_voxel_size,
                )
                pred[indices] = tile_pred
                done[indices] = True
                print(f"    done in {time.perf_counter() - tile_start:.2f}s", flush=True)

    missing = int((~done).sum())
    if missing:
        print(f"  warning: {missing} points were not covered by tiles; kept label 0", flush=True)
    print(f"  finished {used_tiles} non-empty tiles", flush=True)
    return pred


def iter_jobs(input_path, output_path):
    if input_path.is_file():
        if output_path.exists() and output_path.is_dir():
            return [(input_path, output_path / input_path.name)]
        return [(input_path, output_path)]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    return [
        (path, output_path / path.relative_to(input_path))
        for path in sorted(input_path.rglob("*.las"))
    ]


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    cfg = Config.fromfile(args.config_file)
    if args.disable_flash:
        set_enable_flash(cfg, False)
    device = torch.device(args.device)

    if args.fragment_batch_size < 1:
        raise ValueError("--fragment-batch-size must be >= 1")

    model = load_model(cfg, args.weight, device)
    pipeline = build_test_pipeline(cfg, args.point_max, args.grid_size)
    jobs = iter_jobs(input_path, output_path)
    if not jobs:
        raise FileNotFoundError(f"No .las files found under: {input_path}")

    for source_path, target_path in jobs:
        if target_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists, use --overwrite: {target_path}")
        print(f"Reading {source_path}", flush=True)
        read_start = time.perf_counter()
        coord, color, las = read_las(source_path, args.default_color, args.las_backend)
        print(
            f"Loaded {coord.shape[0]} points in {time.perf_counter() - read_start:.2f}s",
            flush=True,
        )
        pred = infer_las_tiled(
            model=model,
            cfg=cfg,
            pipeline=pipeline,
            coord=coord,
            color=color,
            device=device,
            fragment_batch_size=args.fragment_batch_size,
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
            min_tile_points=args.min_tile_points,
            pre_voxel_size=args.pre_voxel_size,
            merge_mode=args.merge_mode,
            context_margin=args.context_margin,
        )
        write_start = time.perf_counter()
        write_las(source_path, target_path, las, pred, colorize=not args.no_colorize)
        print(f"Wrote LAS in {time.perf_counter() - write_start:.2f}s", flush=True)
        counts = np.bincount(pred, minlength=cfg.data.num_classes)
        summary = ", ".join(
            f"{idx}:{cfg.data.names[idx]}={int(counts[idx])}"
            for idx in range(cfg.data.num_classes)
        )
        print(f"Saved {target_path} ({coord.shape[0]} points; {summary})")


if __name__ == "__main__":
    main()

# 使用教程：

"""   
python tools/infer_las_semseg.py \
    --input /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb009b5736892392a.las \
    --output /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb_outPut.las \
    --overwrite

  也支持目录批量推理：

  python tools/infer_las_semseg.py \
    --input /path/to/las_dir \
    --output /path/to/output_dir \
    --overwrite
    
    python tools/infer_las_semseg.py \
    --input /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb009b5736892392a.las \
    --output /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb_outPut.las \
    --fragment-batch-size 6 \
    --min-tile-points 1024 \
    --pre-voxel-size 0.05
    --overwrite
    
      1. 当前模式 plain

  python tools/infer_las_semseg.py \
    --input /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb009b5736892392a_v2.las \
    --output /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb_plain_v1.las \
    --merge-mode plain \
    --tile-size 40 \
    --tile-stride 40 \
    --pre-voxel-size 0.05 \
    --overwrite

  2. Halo 上下文模式，推荐优先试

  python tools/infer_las_semseg.py \
    --input /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb009b5736892392a.las \
    --output /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb_plain_v1.las \
    --merge-mode halo \
    --tile-size 40 \
    --tile-stride 40 \
    --context-margin 10 \
    --pre-voxel-size 0.05 \
    --overwrite

  含义是：每个核心块仍是 40m，但实际推理用周围额外 10m 上下文，只写回中间 40m 核心区。这个通常比 plain 稳，比 overlap 快。

  3. Overlap 概率平均模式

  python tools/infer_las_semseg.py \
    --input .../cloudb009b5736892392a.las \
    --output .../cloudb_overlap.las \
    --merge-mode overlap \
    --tile-size 40 \
    --tile-stride 20 \
    --pre-voxel-size 0.05 \
    --overwrite

  含义是重叠推理，同一个点可能被多个 tile 预测，脚本会累加 softmax 概率后取平均结果。这个通常最稳，但最慢，也更吃内存。

  建议对比顺序：

  plain -> halo context 10m -> overlap stride 20m

  如果 halo 效果已经够好，就不建议用 overlap，因为 overlap 会重复推理很多点。对于你这种 4800 万点 LAS，halo 更可能是速度和精度的折中最优。

 """