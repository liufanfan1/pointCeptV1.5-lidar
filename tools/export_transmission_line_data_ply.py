#!/usr/bin/env python
"""Export preprocessed transmission-line .pth tiles to visualizable PLY files."""

import argparse
from pathlib import Path

import numpy as np
import torch


CLASS_NAMES = ("ground", "tower", "line", "insulator", "hengdan", "other")
TRANSMISSION_LINE_PALETTE = np.array(
    [
        [142, 142, 142],  # ground 灰色
        [214, 39, 40],  # tower 红色
        [31, 119, 180],  # line 蓝色
        [255, 127, 14],  # insulator 橙色
        [44, 160, 44],  # hengdan 绿色
        [148, 103, 189],  # other 紫色
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert preprocessed transmission-line .pth tiles to PLY point clouds."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help="Input .pth file(s) or directories containing .pth tiles.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PLY path for one input, or output directory for many inputs.",
    )
    parser.add_argument(
        "--merge-out",
        type=Path,
        default=None,
        help="Also save all input tiles into one merged PLY file.",
    )
    parser.add_argument(
        "--color",
        choices=("label", "rgb"),
        default="label",
        help="PLY RGB source: semantic label colors or original RGB colors.",
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


def collect_pth_paths(paths):
    pth_paths = []
    for path in paths:
        if path.is_dir():
            pth_paths.extend(sorted(path.glob("*.pth")))
        else:
            pth_paths.append(path)
    pth_paths = sorted(set(pth_paths))
    if not pth_paths:
        raise FileNotFoundError("No .pth files were found.")
    for path in pth_paths:
        if path.suffix != ".pth":
            raise ValueError(f"Expected a .pth file, got {path}")
    return pth_paths


def load_tile(path, world_coord=False):
    data = torch.load(path, map_location="cpu")
    coord = np.asarray(data["coord"], dtype=np.float32)
    color = np.asarray(data.get("color", np.zeros((coord.shape[0], 3))), dtype=np.uint8)
    label = data.get("semantic_gt")
    if label is None:
        label = np.full(coord.shape[0], -1, dtype=np.int32)
    else:
        label = np.asarray(label, dtype=np.int32).reshape(-1)
    if world_coord:
        origin = np.asarray(data.get("origin", np.zeros(3)), dtype=np.float32)
        coord = coord + origin.reshape(1, 3)
    return coord, color, label


def color_by_label(label):
    color = np.zeros((label.shape[0], 3), dtype=np.uint8)
    valid = (label >= 0) & (label < len(TRANSMISSION_LINE_PALETTE))
    color[valid] = TRANSMISSION_LINE_PALETTE[label[valid]]
    color[~valid] = [0, 0, 0]
    return color


def choose_color(mode, label, original_color):
    if mode == "label":
        return color_by_label(label)
    return original_color.astype(np.uint8, copy=False)


def vertex_array(coord, color, label):
    arr = np.empty(
        coord.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("label", "<i4"),
        ],
    )
    arr["x"] = coord[:, 0]
    arr["y"] = coord[:, 1]
    arr["z"] = coord[:, 2]
    arr["red"] = color[:, 0]
    arr["green"] = color[:, 1]
    arr["blue"] = color[:, 2]
    arr["label"] = label.astype(np.int32, copy=False)
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
        "property int label\n"
        "end_header\n"
    )


def write_vertices(file, coord, color, label, ascii_format):
    vertices = vertex_array(coord, color, label)
    if ascii_format:
        for row in vertices:
            file.write(
                "{:.6f} {:.6f} {:.6f} {} {} {} {}\n".format(
                    row["x"],
                    row["y"],
                    row["z"],
                    int(row["red"]),
                    int(row["green"]),
                    int(row["blue"]),
                    int(row["label"]),
                )
            )
    else:
        vertices.tofile(file)


def write_ply(path, coord, color, label, ascii_format):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ply_header(coord.shape[0], ascii_format)
    if ascii_format:
        with path.open("w", encoding="utf-8") as file:
            file.write(header)
            write_vertices(file, coord, color, label, ascii_format=True)
    else:
        with path.open("wb") as file:
            file.write(header.encode("ascii"))
            write_vertices(file, coord, color, label, ascii_format=False)


def output_path_for(input_path, out, multiple):
    if out is None:
        return input_path.with_suffix(".ply")
    if not multiple and out.suffix:
        return out
    return out / input_path.with_suffix(".ply").name


def point_count(path):
    data = torch.load(path, map_location="cpu")
    return int(np.asarray(data["coord"]).shape[0])


def write_merged_ply(path, pth_paths, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    total_points = sum(point_count(path) for path in pth_paths)
    header = ply_header(total_points, args.ascii)
    mode = "w" if args.ascii else "wb"
    encoding = "utf-8" if args.ascii else None

    with path.open(mode, encoding=encoding) as file:
        if args.ascii:
            file.write(header)
        else:
            file.write(header.encode("ascii"))
        written = 0
        for pth_path in pth_paths:
            coord, original_color, label = load_tile(
                pth_path, world_coord=args.world_coord
            )
            color = choose_color(args.color, label, original_color)
            write_vertices(file, coord, color, label, args.ascii)
            written += coord.shape[0]
    print(f"Saved merged {path} ({written:,} points, color={args.color})")


def main():
    args = parse_args()
    pth_paths = collect_pth_paths(args.input)
    multiple = len(pth_paths) > 1
    if multiple and args.out is not None and args.out.suffix:
        raise ValueError("--out must be a directory when exporting multiple inputs.")

    for pth_path in pth_paths:
        coord, original_color, label = load_tile(pth_path, world_coord=args.world_coord)
        color = choose_color(args.color, label, original_color)
        out_path = output_path_for(pth_path, args.out, multiple)
        write_ply(out_path, coord, color, label, args.ascii)
        print(f"Saved {out_path} ({coord.shape[0]:,} points, color={args.color})")

    if args.merge_out is not None:
        write_merged_ply(args.merge_out, pth_paths, args)


if __name__ == "__main__":
    main()
