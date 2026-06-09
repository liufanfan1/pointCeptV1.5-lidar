#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import torch


S3DIS_PALETTE = np.array(
    [
        [0, 255, 0],  # ceiling
        [0, 0, 255],  # floor
        [0, 255, 255],  # wall
        [255, 255, 0],  # beam
        [255, 0, 255],  # column
        [100, 100, 255],  # window
        [200, 200, 100],  # door
        [255, 0, 0],  # table
        [170, 120, 200],  # chair
        [255, 150, 120],  # sofa
        [200, 100, 100],  # bookcase
        [10, 200, 100],  # board
        [200, 200, 200],  # clutter
    ],
    dtype=np.uint8,
)


def scene_from_pred(pred_path: Path):
    stem = pred_path.name
    if not stem.endswith("_pred.npy"):
        raise ValueError(f"Expected a file ending with '_pred.npy', got {pred_path}")
    scene = stem[: -len("_pred.npy")]
    if "-" in scene:
        area, room = scene.split("-", 1)
        return area, room
    return None, scene


def room_path_from_pred(pred_path: Path, data_root: Path) -> Path:
    area, room = scene_from_pred(pred_path)

    if area is not None:
        pth_path = data_root / area / f"{room}.pth"
        if pth_path.exists():
            return pth_path

        room_dir = data_root / area / room
        if room_dir.exists():
            return room_dir

    matches = sorted(data_root.glob(f"Area_*/{room}.pth"))
    matches.extend(sorted(p for p in data_root.glob(f"Area_*/{room}") if p.is_dir()))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Room name {room!r} matched multiple files. Use Area_x-room_pred.npy naming."
        )

    expected = data_root / (area or "Area_x") / f"{room}.pth"
    raise FileNotFoundError(
        f"Could not find S3DIS room data for {pred_path}; expected {expected}"
    )


def load_room_arrays(room_path: Path):
    if room_path.is_file() and room_path.suffix == ".pth":
        data = torch.load(room_path, map_location="cpu")
        return data["coord"], data["color"]

    return np.load(room_path / "coord.npy"), np.load(room_path / "color.npy")


def write_ply(
    path: Path, coord: np.ndarray, color: np.ndarray, pred: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coord = coord.astype(np.float32)
    color = color.astype(np.uint8)
    pred = pred.astype(np.int32).reshape(-1)

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {coord.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property int pred\n"
        "end_header\n"
    )

    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for xyz, rgb, label in zip(coord, color, pred):
            f.write(
                f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} {int(label)}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Pointcept S3DIS *_pred.npy predictions to colored PLY."
    )
    parser.add_argument("pred", type=Path, nargs="+", help="Path(s) to *_pred.npy")
    parser.add_argument("--data-root", type=Path, default=Path("data/s3dis"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--color",
        choices=("pred", "rgb"),
        default="pred",
        help="Use predicted label colors or original RGB colors.",
    )
    args = parser.parse_args()

    if args.out is not None and len(args.pred) > 1 and args.out.suffix:
        raise ValueError(
            "--out must be a directory when exporting multiple prediction files"
        )

    for pred_path in args.pred:
        pred = np.load(pred_path)
        room_path = room_path_from_pred(pred_path, args.data_root)
        coord, original_color = load_room_arrays(room_path)

        if pred.shape[0] != coord.shape[0]:
            raise ValueError(
                f"Prediction length {pred.shape[0]} does not match coord length {coord.shape[0]}"
            )

        if args.color == "pred":
            color = S3DIS_PALETTE[pred.reshape(-1) % len(S3DIS_PALETTE)]
        else:
            color = original_color

        if args.out is None:
            out = pred_path.with_suffix(".ply")
        elif len(args.pred) == 1 and args.out.suffix:
            out = args.out
        else:
            out = args.out / pred_path.with_suffix(".ply").name

        write_ply(out, coord, color, pred)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
