# 点云数据预处理的脚本

"""
Preprocess classified transmission-line LAS scenes for Pointcept.

Each source scene contains one LAS file per semantic class, e.g.
``0_ground.las`` and ``2_line.las``. The script merges class files into
spatial tiles while preserving labels in ``semantic_gt``.
"""

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np
import torch


CLASS_NAMES = ("ground", "tower", "line", "insulator")
CLASS_REMAP = {0: 0, 1: 1, 2: 2, 3: 3}
FILE_PATTERN = re.compile(r"^(\d+)[_-](.+)\.las$", re.IGNORECASE)
SCENE_PATTERN = re.compile(r"^scene(\d+)$", re.IGNORECASE)
NATURAL_TOKEN_PATTERN = re.compile(r"(\d+)")
SAFE_NAME_PATTERN = re.compile(r"[^0-9A-Za-z_.-]+")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert classified transmission-line LAS scenes to Pointcept .pth tiles."
    )
    parser.add_argument(
        "--dataset-root", required=True, help="Directory containing Scene*/ folders."
    )
    parser.add_argument(
        "--output-root", required=True, help="Output Pointcept data directory."
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.02,
        help="Class-preserving voxel sampling size in metres. Set 0 to disable.",
    )
    parser.add_argument(
        "--tile-size",
        type=float,
        default=20.0,
        help="XY tile width and height in metres. Set 0 to save one tile per scene.",
    )
    parser.add_argument(
        "--train-tile-stride",
        type=float,
        default=20.0,
        help="Training tile stride in metres; set below tile size to add overlap.",
    )
    parser.add_argument(
        "--eval-tile-stride",
        type=float,
        default=20.0,
        help="Validation/test tile stride in metres.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=1024,
        help=(
            "Skip output tiles with fewer points after voxel sampling. "
            "Sparse edge tiles contain little geometric context and can destabilize training."
        ),
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=None,
        help="Only convert a named scene; repeat for several scenes. Useful for checking output.",
    )
    parser.add_argument(
        "--train-scenes",
        type=int,
        default=40,
        help="Number of ordered source directories assigned to train when not using scene numbers.",
    )
    parser.add_argument(
        "--val-scenes",
        type=int,
        default=8,
        help="Number of ordered source directories assigned to val when not using scene numbers.",
    )
    parser.add_argument(
        "--split-by",
        choices=("auto", "scene-number", "order", "random"),
        default="auto",
        help=(
            "Split strategy. auto uses scene<number> ids only when every source directory "
            "matches that pattern; otherwise it splits by natural directory order. "
            "random shuffles source directories deterministically and uses train/val ratios."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training source-directory ratio used by --split-by random.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation source-directory ratio used by --split-by random.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=2026,
        help="Random seed used by --split-by random.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output tiles."
    )
    return parser.parse_args()


def read_header(path):
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
    maximum = np.array(
        [struct.unpack_from("<d", header, pos)[0] for pos in (179, 195, 211)],
        dtype=np.float64,
    )
    minimum = np.array(
        [struct.unpack_from("<d", header, pos)[0] for pos in (187, 203, 219)],
        dtype=np.float64,
    )
    return dict(
        version=version,
        point_offset=point_offset,
        point_format=point_format,
        record_length=record_length,
        point_count=point_count,
        scale=scale,
        offset=offset,
        minimum=minimum,
        maximum=maximum,
    )


def read_points(path, origin):
    header = read_header(path)
    point_format = header["point_format"]
    if point_format not in (0, 1, 2, 3):
        raise ValueError(f"Unsupported LAS point format {point_format}: {path}")
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
    coord = xyz_integer.astype(np.float64) * header["scale"] + header["offset"] - origin
    if point_format == 2:
        color_offset = 20
    elif point_format == 3:
        color_offset = 28
    else:
        color_offset = None
    if color_offset is None:
        color = np.zeros((count, 3), dtype=np.uint8)
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
        if color_16.max(initial=0) > 255:
            color = (color_16 >> 8).astype(np.uint8)
        else:
            color = color_16.astype(np.uint8)
    return coord, color


def natural_scene_key(path):
    tokens = NATURAL_TOKEN_PATTERN.split(path.name.lower())
    return tuple(
        (0, int(token)) if token.isdigit() else (1, token)
        for token in tokens
        if token
    )


def scene_number(scene_path):
    match = SCENE_PATTERN.match(scene_path.name)
    if match is None:
        raise ValueError(f"Scene directory must be named scene<number>: {scene_path}")
    return int(match.group(1))


def scene_identifier(scene_path, fallback_index):
    match = SCENE_PATTERN.match(scene_path.name)
    if match is not None:
        return f"scene{int(match.group(1)):02d}"
    scene_id = SAFE_NAME_PATTERN.sub("_", scene_path.name).strip("_")
    return scene_id or f"scene{fallback_index:02d}"


def split_for_number(number, train_scenes, val_scenes):
    if number <= train_scenes:
        return "train"
    if number <= train_scenes + val_scenes:
        return "val"
    return "test"


def random_split_map(scenes, train_ratio, val_ratio, seed):
    if not 0 < train_ratio < 1:
        raise ValueError(f"--train-ratio must be in (0, 1), got {train_ratio}")
    if not 0 <= val_ratio < 1:
        raise ValueError(f"--val-ratio must be in [0, 1), got {val_ratio}")
    if train_ratio + val_ratio >= 1:
        raise ValueError(
            "--train-ratio + --val-ratio must be less than 1 so test is non-empty"
        )
    indices = np.arange(len(scenes))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_count = int(round(len(scenes) * train_ratio))
    val_count = int(round(len(scenes) * val_ratio))
    train_count = min(max(train_count, 1), len(scenes) - 2)
    val_count = min(max(val_count, 1), len(scenes) - train_count - 1)

    split_by_name = {}
    for position, scene_index in enumerate(indices):
        if position < train_count:
            split = "train"
        elif position < train_count + val_count:
            split = "val"
        else:
            split = "test"
        split_by_name[scenes[scene_index].name] = split
    return split_by_name


def choose_split(scene_path, ordered_index, args, all_scene_numbered):
    if args.split_by == "scene-number" or (
        args.split_by == "auto" and all_scene_numbered
    ):
        number = scene_number(scene_path)
    else:
        number = ordered_index
    return split_for_number(number, args.train_scenes, args.val_scenes)


def tile_starts(extent, tile_size, stride):
    if tile_size <= 0:
        return [0.0]
    # Edge tiles may be smaller; this prevents an almost duplicate final tile.
    return list(np.arange(0.0, max(extent, 1e-9), stride))


def voxel_select(coord, voxel_size):
    if voxel_size <= 0 or coord.shape[0] == 0:
        return np.arange(coord.shape[0])
    grid_coord = np.floor(coord / voxel_size).astype(np.int64)
    _, index = np.unique(grid_coord, axis=0, return_index=True)
    return np.sort(index)


def scene_class_files(scene_path):
    files = []
    for path in sorted(scene_path.glob("*.las")):
        match = FILE_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected LAS filename: {path}")
        source_label = int(match.group(1))
        if source_label not in CLASS_REMAP:
            raise ValueError(f"Unsupported semantic label {source_label}: {path}")
        files.append((path, CLASS_REMAP[source_label]))
    if not files:
        raise ValueError(f"No LAS class files found in {scene_path}")
    return files


def convert_scene(scene_path, output_root, args, split, scene_id):
    class_files = scene_class_files(scene_path)
    headers = [read_header(path) for path, _ in class_files]
    origin = np.min(np.stack([item["minimum"] for item in headers]), axis=0)
    maximum = np.max(np.stack([item["maximum"] for item in headers]), axis=0)
    extent = maximum - origin
    stride = args.train_tile_stride if split == "train" else args.eval_tile_stride
    x_starts = tile_starts(extent[0], args.tile_size, stride)
    y_starts = tile_starts(extent[1], args.tile_size, stride)
    tiles = [
        dict(coord=[], color=[], semantic_gt=[], bounds=(x, y))
        for x in x_starts
        for y in y_starts
    ]

    source_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for path, label in class_files:
        coord, color = read_points(path, origin)
        source_counts[label] += coord.shape[0]
        for tile in tiles:
            x, y = tile["bounds"]
            if args.tile_size > 0:
                mask = (
                    (coord[:, 0] >= x)
                    & (coord[:, 0] <= x + args.tile_size)
                    & (coord[:, 1] >= y)
                    & (coord[:, 1] <= y + args.tile_size)
                )
                tile_coord = coord[mask]
                tile_color = color[mask]
            else:
                tile_coord = coord
                tile_color = color
            index = voxel_select(tile_coord, args.voxel_size)
            if index.size == 0:
                continue
            tile["coord"].append(tile_coord[index].astype(np.float32))
            tile["color"].append(tile_color[index])
            tile["semantic_gt"].append(np.full(index.size, label, dtype=np.int64))

    split_path = output_root / split
    split_path.mkdir(parents=True, exist_ok=True)
    tile_count = 0
    skipped_tile_count = 0
    saved_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for tile_id, tile in enumerate(tiles):
        if not tile["coord"]:
            continue
        coord = np.concatenate(tile["coord"], axis=0)
        color = np.concatenate(tile["color"], axis=0)
        semantic_gt = np.concatenate(tile["semantic_gt"], axis=0)
        if coord.shape[0] < args.min_points:
            skipped_tile_count += 1
            continue
        output_path = split_path / f"{scene_id}_tile{tile_id:03d}.pth"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {output_path}")
        data = dict(
            coord=coord,
            color=color,
            semantic_gt=semantic_gt,
            origin=origin.astype(np.float64),
            scene=scene_id,
        )
        torch.save(data, output_path)
        saved_counts += np.bincount(semantic_gt, minlength=len(CLASS_NAMES))
        tile_count += 1
    return dict(
        source_scene=scene_path.name,
        scene=scene_id,
        split=split,
        tiles=tile_count,
        skipped_tiles=skipped_tile_count,
        source_points=source_counts.tolist(),
        saved_points=saved_counts.tolist(),
        origin=origin.tolist(),
    )


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    scenes = sorted(
        [path for path in dataset_root.iterdir() if path.is_dir() and any(path.glob("*.las"))],
        key=natural_scene_key,
    )
    if args.scene:
        wanted = {name.lower() for name in args.scene}
        scenes = [scene for scene in scenes if scene.name.lower() in wanted]
    if not scenes:
        raise ValueError("No matching scene directories found.")
    all_scene_numbered = all(SCENE_PATTERN.match(scene.name) for scene in scenes)
    if args.split_by == "scene-number" and not all_scene_numbered:
        bad_scene = next(scene.name for scene in scenes if not SCENE_PATTERN.match(scene.name))
        raise ValueError(
            "--split-by scene-number requires all source directories to be named "
            f"scene<number>; first non-matching directory: {bad_scene}"
        )
    random_splits = (
        random_split_map(scenes, args.train_ratio, args.val_ratio, args.split_seed)
        if args.split_by == "random"
        else None
    )

    output_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for ordered_index, scene in enumerate(scenes, start=1):
        split = (
            random_splits[scene.name]
            if random_splits is not None
            else choose_split(scene, ordered_index, args, all_scene_numbered)
        )
        scene_id = scene_identifier(scene, ordered_index)
        result = convert_scene(scene, output_root, args, split, scene_id)
        summary.append(result)
        print(
            "{scene} -> {split}: {tiles} tiles ({skipped} sparse skipped), "
            "{source:,} source points, {saved:,} saved points".format(
                scene=result["scene"],
                split=result["split"],
                tiles=result["tiles"],
                skipped=result["skipped_tiles"],
                source=sum(result["source_points"]),
                saved=sum(result["saved_points"]),
            )
        )
    metadata = dict(
        class_names=CLASS_NAMES,
        source_label_remap=CLASS_REMAP,
        voxel_size=args.voxel_size,
        tile_size=args.tile_size,
        train_tile_stride=args.train_tile_stride,
        eval_tile_stride=args.eval_tile_stride,
        min_points=args.min_points,
        train_scenes=args.train_scenes,
        val_scenes=args.val_scenes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        split_by=(
            "scene-number"
            if args.split_by == "auto" and all_scene_numbered
            else "order"
            if args.split_by == "auto"
            else args.split_by
        ),
        scenes=summary,
    )
    with (output_root / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Saved metadata to {output_root / 'metadata.json'}")


if __name__ == "__main__":
    main()
