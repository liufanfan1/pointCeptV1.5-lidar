#!/usr/bin/env python
import argparse
import re
from pathlib import Path

import numpy as np
import torch


S3DIS_PALETTE = np.array(
    [
        [0, 255, 0],      # ceiling
        [0, 0, 255],      # floor
        [0, 255, 255],    # wall
        [255, 255, 0],    # beam
        [255, 0, 255],    # column
        [100, 100, 255],  # window
        [200, 200, 100],  # door
        [255, 0, 0],      # table
        [170, 120, 200],  # chair
        [255, 150, 120],  # sofa
        [200, 100, 100],  # bookcase
        [10, 200, 100],   # board
        [200, 200, 200],  # clutter
    ],
    dtype=np.uint8,
)


def natural_key(path):
    parts = re.split(r"(\d+)", str(path))
    return [int(part) if part.isdigit() else part for part in parts]


def parse_pred_name(pred_path):
    name = pred_path.name
    if not name.endswith("_pred.npy"):
        return None, None
    scene = name[: -len("_pred.npy")]
    if "-" in scene:
        area, room = scene.split("-", 1)
        return area, room
    return None, scene


def collect_area_predictions(result_dir, data_root, area):
    room_to_pred = {}
    room_to_explicit = {}

    for pred_path in sorted(result_dir.glob("*_pred.npy"), key=natural_key):
        pred_area, room = parse_pred_name(pred_path)
        if room is None:
            continue
        if pred_area is not None and pred_area != area:
            continue

        room_path = data_root / area / "{}.pth".format(room)
        if not room_path.exists():
            continue

        is_explicit = pred_area == area
        if room not in room_to_pred or (is_explicit and not room_to_explicit[room]):
            room_to_pred[room] = pred_path
            room_to_explicit[room] = is_explicit

    return [(room, room_to_pred[room]) for room in sorted(room_to_pred, key=natural_key)]


def load_room(room_path):
    data = torch.load(room_path, map_location="cpu")
    return data["coord"], data["color"]


def ply_header(vertex_count, binary):
    fmt = "binary_little_endian 1.0" if binary else "ascii 1.0"
    return (
        "ply\n"
        "format {}\n"
        "element vertex {}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property int pred\n"
        "end_header\n"
    ).format(fmt, vertex_count)


def vertex_array(coord, color, pred):
    coord = coord.astype(np.float32, copy=False)
    color = color.astype(np.uint8, copy=False)
    pred = pred.astype(np.int32, copy=False).reshape(-1)

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
        ],
    )
    arr["x"] = coord[:, 0]
    arr["y"] = coord[:, 1]
    arr["z"] = coord[:, 2]
    arr["red"] = color[:, 0]
    arr["green"] = color[:, 1]
    arr["blue"] = color[:, 2]
    arr["pred"] = pred
    return arr


def write_vertices_ascii(file, coord, color, pred):
    coord = coord.astype(np.float32, copy=False)
    color = color.astype(np.uint8, copy=False)
    pred = pred.astype(np.int32, copy=False).reshape(-1)
    for xyz, rgb, label in zip(coord, color, pred):
        file.write(
            "{:.6f} {:.6f} {:.6f} {} {} {} {}\n".format(
                xyz[0],
                xyz[1],
                xyz[2],
                int(rgb[0]),
                int(rgb[1]),
                int(rgb[2]),
                int(label),
            )
        )


def write_room_ply(out_path, coord, color, pred, binary):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if binary else "w"
    with out_path.open(mode) as f:
        header = ply_header(coord.shape[0], binary)
        if binary:
            f.write(header.encode("ascii"))
            vertex_array(coord, color, pred).tofile(f)
        else:
            f.write(header)
            write_vertices_ascii(f, coord, color, pred)


def main():
    parser = argparse.ArgumentParser(
        description="Export all S3DIS room predictions in one area to a merged PLY."
    )
    parser.add_argument("--result-dir", type=Path, required=True, help="Directory with *_pred.npy files.")
    parser.add_argument("--data-root", type=Path, default=Path("data/s3dis"))
    parser.add_argument("--area", default="Area_5")
    parser.add_argument("--out", type=Path, default=None, help="Merged area PLY path.")
    parser.add_argument("--room-out", type=Path, default=None, help="Optional directory for per-room PLY files.")
    parser.add_argument("--color", choices=("pred", "rgb"), default="pred")
    parser.add_argument("--format", choices=("binary", "ascii"), default="binary")
    args = parser.parse_args()

    binary = args.format == "binary"
    out_path = args.out or args.result_dir / "{}_pred.ply".format(args.area)

    items = collect_area_predictions(args.result_dir, args.data_root, args.area)
    if not items:
        raise FileNotFoundError(
            "No prediction files in {} matched {}/<room>.pth".format(
                args.result_dir, args.data_root / args.area
            )
        )

    vertex_count = 0
    room_counts = []
    for room, pred_path in items:
        pred = np.load(pred_path, mmap_mode="r")
        room_path = args.data_root / args.area / "{}.pth".format(room)
        coord, _ = load_room(room_path)
        if pred.shape[0] != coord.shape[0]:
            raise ValueError(
                "{} length {} does not match {} points {}".format(
                    pred_path, pred.shape[0], room_path, coord.shape[0]
                )
            )
        room_counts.append((room, coord.shape[0]))
        vertex_count += coord.shape[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if binary else "w"
    with out_path.open(mode) as area_file:
        header = ply_header(vertex_count, binary)
        if binary:
            area_file.write(header.encode("ascii"))
        else:
            area_file.write(header)

        for index, (room, pred_path) in enumerate(items, start=1):
            room_path = args.data_root / args.area / "{}.pth".format(room)
            pred = np.load(pred_path)
            coord, original_color = load_room(room_path)
            color = (
                S3DIS_PALETTE[pred.reshape(-1) % len(S3DIS_PALETTE)]
                if args.color == "pred"
                else original_color
            )

            if binary:
                vertex_array(coord, color, pred).tofile(area_file)
            else:
                write_vertices_ascii(area_file, coord, color, pred)

            if args.room_out is not None:
                room_out = args.room_out / "{}_pred.ply".format(room)
                write_room_ply(room_out, coord, color, pred, binary)

            print(
                "[{}/{}] exported {} ({:,} points)".format(
                    index, len(items), room, coord.shape[0]
                )
            )

    print("Saved merged area PLY: {}".format(out_path))
    print("Rooms: {}, points: {:,}".format(len(room_counts), vertex_count))


if __name__ == "__main__":
    main()
