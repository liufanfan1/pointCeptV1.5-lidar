#!/usr/bin/env python
"""Export Stage-1 transmission-line 4-class predictions to PLY."""

# stage1 专用PLY导出脚本


""" 
0 ground            灰色
1 tower_structure   红色
2 line              蓝色
3 other             紫色
"""

import argparse
from pathlib import Path

import numpy as np
import torch


CLASS_NAMES = ("ground", "tower_structure", "line", "other")
STAGE1_PALETTE = np.array(
    [
        [142, 142, 142],  # ground
        [214, 39, 40],  # tower_structure
        [31, 119, 180],  # line
        [148, 103, 189],  # other
    ],
    dtype=np.uint8,
)
CORRECT_PALETTE = np.array(
    [
        [220, 53, 69],  # wrong
        [40, 167, 69],  # correct
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert Stage-1 transmission-line *_pred.npy results and matching "
            ".pth point tiles to PLY point clouds."
        )
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
        default=Path("data/transmission_line_stage1_4cls_balance"),
        help="Stage-1 Pointcept data root.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split containing the matching .pth files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PLY path for one prediction, or output directory for many.",
    )
    parser.add_argument(
        "--merge-out",
        type=Path,
        default=None,
        help="Also save all exported predictions into one merged PLY file.",
    )
    parser.add_argument(
        "--color",
        choices=("pred", "label", "rgb", "correct"),
        default="pred",
        help="PLY RGB source: predicted class, ground-truth label, original RGB, or correctness.",
    )
    parser.add_argument(
        "--world-coord",
        action="store_true",
        help="Add the saved scene origin back to tile-local coordinates when available.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Write ASCII PLY instead of binary little-endian PLY.",
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


def load_tile(tile_path):
    data = torch.load(tile_path, map_location="cpu")
    coord = np.asarray(data["coord"], dtype=np.float32)
    color = np.asarray(data.get("color", np.zeros((coord.shape[0], 3))), dtype=np.uint8)
    label = data.get("semantic_gt")
    if label is None:
        label = np.full(coord.shape[0], -1, dtype=np.int32)
    else:
        label = np.asarray(label, dtype=np.int32).reshape(-1)
    origin = np.asarray(data.get("origin", np.zeros(3)), dtype=np.float32)
    return coord, color, label, origin


def color_by_label(label):
    color = np.zeros((label.shape[0], 3), dtype=np.uint8)
    valid = (label >= 0) & (label < len(STAGE1_PALETTE))
    color[valid] = STAGE1_PALETTE[label[valid]]
    color[~valid] = [0, 0, 0]
    return color


def choose_color(mode, pred, label, original_color):
    if mode == "pred":
        return color_by_label(pred)
    if mode == "label":
        return color_by_label(label)
    if mode == "correct":
        valid = label >= 0
        correct = (pred == label) & valid
        color = CORRECT_PALETTE[correct.astype(np.int64)]
        color[~valid] = [80, 80, 80]
        return color.astype(np.uint8)
    return original_color.astype(np.uint8, copy=False)


def vertex_array(coord, color, pred, label):
    arr = np.empty(
        coord.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("pred", "<i4"),
            ("label", "<i4"),
            ("correct", "u1"),
        ],
    )
    arr["x"] = coord[:, 0]
    arr["y"] = coord[:, 1]
    arr["z"] = coord[:, 2]
    arr["red"] = color[:, 0]
    arr["green"] = color[:, 1]
    arr["blue"] = color[:, 2]
    arr["pred"] = pred.astype(np.int32, copy=False)
    arr["label"] = label.astype(np.int32, copy=False)
    arr["correct"] = ((pred == label) & (label >= 0)).astype(np.uint8)
    return arr


def ply_header(point_count, ascii_format):
    fmt = "ascii 1.0" if ascii_format else "binary_little_endian 1.0"
    classes = ", ".join(f"{i}:{name}" for i, name in enumerate(CLASS_NAMES))
    return (
        "ply\n"
        f"format {fmt}\n"
        f"comment classes {classes}\n"
        f"element vertex {point_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property int pred\n"
        "property int label\n"
        "property uchar correct\n"
        "end_header\n"
    )


def write_vertices(file, coord, color, pred, label, ascii_format):
    vertices = vertex_array(coord, color, pred, label)
    if ascii_format:
        for row in vertices:
            file.write(
                "{:.6f} {:.6f} {:.6f} {} {} {} {} {} {}\n".format(
                    row["x"],
                    row["y"],
                    row["z"],
                    int(row["red"]),
                    int(row["green"]),
                    int(row["blue"]),
                    int(row["pred"]),
                    int(row["label"]),
                    int(row["correct"]),
                )
            )
    else:
        vertices.tofile(file)


def write_ply(path, coord, color, pred, label, ascii_format):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if ascii_format else "wb"
    encoding = "utf-8" if ascii_format else None
    with path.open(mode, encoding=encoding) as file:
        header = ply_header(coord.shape[0], ascii_format)
        if ascii_format:
            file.write(header)
        else:
            file.write(header.encode("ascii"))
        write_vertices(file, coord, color, pred, label, ascii_format)


def output_path_for(pred_path, out, multiple):
    if out is None:
        return pred_path.with_suffix(".ply")
    if not multiple and out.suffix:
        return out
    return out / pred_path.with_suffix(".ply").name


def load_prediction(pred_path, args):
    tile_name = tile_name_from_pred(pred_path)
    tile_path = args.data_root / args.split / f"{tile_name}.pth"
    if not tile_path.exists():
        raise FileNotFoundError(f"Missing matching point tile: {tile_path}")

    pred = np.load(pred_path).astype(np.int32).reshape(-1)
    coord, original_color, label, origin = load_tile(tile_path)
    if pred.shape[0] != coord.shape[0]:
        raise ValueError(
            f"{pred_path} has {pred.shape[0]} predictions, but {tile_path} "
            f"has {coord.shape[0]} points."
        )
    if args.world_coord:
        coord = coord + origin.reshape(1, 3)

    color = choose_color(args.color, pred, label, original_color)
    return coord, color, pred, label


def prediction_point_count(pred_path):
    pred = np.load(pred_path, mmap_mode="r")
    return int(np.prod(pred.shape))


def write_merged_ply(path, pred_paths, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    point_count = sum(prediction_point_count(pred_path) for pred_path in pred_paths)
    header = ply_header(point_count, args.ascii)
    mode = "w" if args.ascii else "wb"
    encoding = "utf-8" if args.ascii else None

    with path.open(mode, encoding=encoding) as file:
        if args.ascii:
            file.write(header)
        else:
            file.write(header.encode("ascii"))
        written = 0
        for pred_path in pred_paths:
            coord, color, pred, label = load_prediction(pred_path, args)
            write_vertices(file, coord, color, pred, label, args.ascii)
            written += coord.shape[0]

    print(f"Saved merged {path} ({written:,} points, color={args.color})")


def main():
    args = parse_args()
    pred_paths = collect_pred_paths(args.pred)
    multiple = len(pred_paths) > 1
    if multiple and args.out is not None and args.out.suffix:
        raise ValueError(
            "--out must be a directory when exporting multiple predictions."
        )

    for pred_path in pred_paths:
        coord, color, pred, label = load_prediction(pred_path, args)
        out_path = output_path_for(pred_path, args.out, multiple)
        write_ply(out_path, coord, color, pred, label, args.ascii)
        print(f"Saved {out_path} ({coord.shape[0]:,} points, color={args.color})")

    if args.merge_out is not None:
        write_merged_ply(args.merge_out, pred_paths, args)


if __name__ == "__main__":
    main()

# 使用说明：
""" 
导出全部stage1测试结果
cd /24085403037/PointTransformerV3/Pointcept-v1.5.1

/opt/conda/envs/pointcept/bin/python tools/export_transmission_line_stage1_pred_ply.py \
  exp/transmission/stage1_4cls_balance_w8_clean/result \
  --data-root data/transmission_line_stage1_4cls_balance \
  --split test \
  --out exp/transmission/stage1_4cls_balance_w8_clean/stage1_result_ply \
  --merge-out exp/transmission/stage1_4cls_balance_w8_clean/stage1_test_pred.ply \
  --color pred \
  --world-coord 

  
 看预测对错，其中 绿色是预测正确，红色是预测错误
 /opt/conda/envs/pointcept/bin/python tools/export_transmission_line_stage1_pred_ply.py \
  exp/transmission/stage1_4cls_balance_w8_clean/result \
  --data-root data/transmission_line_stage1_4cls_balance \
  --split test \
  --out exp/transmission/stage1_4cls_balance_w8_clean/stage1_correct_ply \
  --merge-out exp/transmission/stage1_4cls_balance_w8_clean/stage1_test_correct.ply \
  --color correct \
  --world-coord 
  
  
"""
