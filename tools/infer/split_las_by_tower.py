"""把四分类后的 LAS 按物理杆塔拆成多个 LAS 文件。

输入 LAS 需要已经把语义分割结果写入 classification 字段：
0=背景，1=杆塔，2=导线，3=绝缘子。

默认输出模式是 local-scene：每个杆塔输出一个局部场景 LAS，
包含该杆塔包围盒外扩范围内的所有类别点。也可以用 tower-only
只保存该杆塔自身的杆塔点。
"""

import argparse
import copy
import json
import time
from itertools import product
from pathlib import Path

import laspy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a four-class transmission-line LAS into one LAS per tower."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入四分类后的 LAS 文件或目录；目录模式会递归处理 LAS/LAZ。",
    )
    parser.add_argument(
        "--render-base-las",
        default=None,
        help=(
            "可选：用于输出的原始测试 LAS 文件或目录。杆塔识别仍使用 --input，"
            "输出点记录、颜色和原始分类来自这里。目录模式会按相对路径或文件名匹配。"
        ),
    )
    parser.add_argument("--output-dir", required=True, help="每座杆塔 LAS 输出目录。")
    parser.add_argument(
        "--report",
        default=None,
        help="可选：输出汇总报告 JSON。默认写到 output-dir/split_by_tower_report.json。",
    )
    parser.add_argument("--tower-class", type=int, default=1, help="杆塔类别，默认 1。")
    parser.add_argument(
        "--mode",
        choices=("local-scene", "tower-only"),
        default="local-scene",
        help=(
            "local-scene：输出杆塔附近所有类别点；"
            "tower-only：只输出该物理杆塔的杆塔点。"
        ),
    )
    parser.add_argument(
        "--tower-voxel-size",
        type=float,
        default=0.75,
        help="杆塔点体素连通域聚类体素大小，单位米。",
    )
    parser.add_argument(
        "--tower-connectivity",
        type=int,
        choices=(6, 26),
        default=26,
        help="体素连通域邻接方式。",
    )
    parser.add_argument(
        "--min-tower-points",
        type=int,
        default=200,
        help="一个物理杆塔最少杆塔点数。",
    )
    parser.add_argument(
        "--min-tower-height",
        type=float,
        default=4.0,
        help="一个物理杆塔最小高度，单位米。",
    )
    parser.add_argument(
        "--crop-xy-margin",
        type=float,
        default=15.0,
        help="local-scene 模式下，杆塔包围盒 XY 外扩范围，单位米。",
    )
    parser.add_argument(
        "--crop-z-margin",
        type=float,
        default=8.0,
        help="local-scene 模式下，杆塔包围盒 Z 外扩范围，单位米。",
    )
    parser.add_argument(
        "--min-output-points",
        type=int,
        default=1,
        help="输出 LAS 至少需要包含的点数，低于该值跳过。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出文件。")
    return parser.parse_args()


def to_float_list(values, ndigits=6):
    return [round(float(v), ndigits) for v in values]


def ensure_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_path(args, output_dir):
    if args.report:
        path = Path(args.report)
    else:
        path = output_dir / "split_by_tower_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"Report exists, use --overwrite: {path}")
    return path


def voxel_component_labels(points, voxel_size, connectivity):
    """用体素连通域给杆塔点编号。"""
    if points.shape[0] == 0:
        return np.empty(0, dtype=np.int32)
    if voxel_size <= 0:
        raise ValueError("--tower-voxel-size must be > 0")

    origin = points.min(axis=0)
    voxels = np.floor((points - origin[None, :]) / float(voxel_size)).astype(np.int64)
    unique_voxels, inverse = np.unique(voxels, axis=0, return_inverse=True)
    voxel_to_id = {tuple(v): i for i, v in enumerate(unique_voxels)}

    offsets = []
    for offset in product((-1, 0, 1), repeat=3):
        if offset == (0, 0, 0):
            continue
        if connectivity == 6 and sum(abs(v) for v in offset) != 1:
            continue
        # 只保留半边邻域，避免重复边。
        if offset <= (0, 0, 0):
            continue
        offsets.append(np.asarray(offset, dtype=np.int64))

    rows = []
    cols = []
    for voxel_id, voxel in enumerate(unique_voxels):
        for offset in offsets:
            neighbor_id = voxel_to_id.get(tuple(voxel + offset))
            if neighbor_id is None:
                continue
            rows.extend((voxel_id, neighbor_id))
            cols.extend((neighbor_id, voxel_id))

    if rows:
        graph = csr_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
            shape=(unique_voxels.shape[0], unique_voxels.shape[0]),
        )
        _, voxel_labels = connected_components(graph, directed=False, return_labels=True)
    else:
        voxel_labels = np.arange(unique_voxels.shape[0], dtype=np.int32)
    return voxel_labels[inverse].astype(np.int32, copy=False)


def build_towers(tower_coord, tower_indices, args):
    """从所有杆塔语义点中提取物理杆塔实例。"""
    labels = voxel_component_labels(
        tower_coord, args.tower_voxel_size, args.tower_connectivity
    )
    towers = []
    for label in sorted(set(labels.tolist())):
        if label < 0:
            continue
        mask = labels == label
        points = tower_coord[mask]
        indices = tower_indices[mask]
        if points.shape[0] < args.min_tower_points:
            continue
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        extent = bbox_max - bbox_min
        height = float(extent[2])
        if height < args.min_tower_height:
            continue
        towers.append(
            {
                "indices": indices,
                "point_count": int(points.shape[0]),
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "extent": extent,
                "height": height,
                "center": points.mean(axis=0),
            }
        )

    towers.sort(key=lambda item: (item["center"][0], item["center"][1], -item["center"][2]))
    for idx, tower in enumerate(towers, start=1):
        tower["id"] = idx
        tower["name"] = f"杆塔{idx}"
    return towers


def make_empty_like_input_las(input_las):
    """创建空 LAS，并完整继承原始 LAS 头、比例、偏移和 CRS。"""
    return laspy.LasData(copy.deepcopy(input_las.header))


def crop_indices_for_tower(coord, tower, args):
    if args.mode == "tower-only":
        return np.asarray(tower["indices"], dtype=np.int64)

    bbox_min = tower["bbox_min"]
    bbox_max = tower["bbox_max"]
    mask = (
        (coord[:, 0] >= bbox_min[0] - args.crop_xy_margin)
        & (coord[:, 0] <= bbox_max[0] + args.crop_xy_margin)
        & (coord[:, 1] >= bbox_min[1] - args.crop_xy_margin)
        & (coord[:, 1] <= bbox_max[1] + args.crop_xy_margin)
        & (coord[:, 2] >= bbox_min[2] - args.crop_z_margin)
        & (coord[:, 2] <= bbox_max[2] + args.crop_z_margin)
    )
    return np.where(mask)[0]


def write_tower_las(render_las, crop_indices, output_path):
    output_las = make_empty_like_input_las(render_las)
    output_las.points = laspy.ScaleAwarePointRecord(
        render_las.points.array[crop_indices],
        output_las.header.point_format,
        output_las.header.scales,
        output_las.header.offsets,
    )
    output_las.write(output_path)


def class_counts(cls_values):
    if cls_values.size == 0:
        return {}
    values, counts = np.unique(cls_values.astype(np.int64), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def las_files(path):
    """返回单个 LAS/LAZ，或递归返回目录中的全部 LAS/LAZ。"""
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in (".las", ".laz"):
            raise ValueError(f"只支持 LAS/LAZ 文件：{path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"输入路径不存在：{path}")
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in (".las", ".laz")
    )
    if not files:
        raise FileNotFoundError(f"目录中没有 LAS/LAZ 文件：{path}")
    return files


def normalized_stem(path):
    """去除常见推理结果后缀，便于匹配对应的原始 LAS。"""
    stem = Path(path).stem.lower()
    suffixes = (
        "_output_fp16",
        "_output",
        "_prediction",
        "_pred",
        "_infer",
        "_segmented",
        "_seg",
    )
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
                break
    return stem


def match_render_files(segmented_files, input_root, render_root):
    """为每个分割结果匹配对应的原始测试 LAS。"""
    if render_root is None:
        return {path: path for path in segmented_files}

    render_root = Path(render_root)
    render_files = las_files(render_root)
    if len(segmented_files) == 1 and render_root.is_file():
        return {segmented_files[0]: render_files[0]}
    if render_root.is_file():
        raise ValueError("--input 为目录时，--render-base-las 也必须是目录。")

    by_stem = {}
    for path in render_files:
        by_stem.setdefault(normalized_stem(path), []).append(path)

    matches = {}
    for segmented_path in segmented_files:
        candidates = []
        if input_root.is_dir():
            relative = segmented_path.relative_to(input_root)
            relative_candidate = render_root / relative
            if relative_candidate.exists():
                candidates.append(relative_candidate)
            for suffix in (".las", ".laz"):
                normalized_candidate = (
                    render_root / relative.parent / normalized_stem(relative)
                ).with_suffix(suffix)
                if normalized_candidate.exists():
                    candidates.append(normalized_candidate)
        # 相对路径匹配优先，只有目录结构不一致时才退回到全目录文件名匹配。
        if not candidates:
            candidates.extend(by_stem.get(normalized_stem(segmented_path), []))
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            raise FileNotFoundError(
                f"找不到分割结果对应的原始 LAS：{segmented_path}"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"原始 LAS 匹配不唯一：{segmented_path} -> "
                + ", ".join(str(path) for path in candidates)
            )
        matches[segmented_path] = candidates[0]
    return matches


def scene_output_dir(output_root, input_root, segmented_path, batch_mode):
    """批量模式下为每个输入场景创建独立输出目录。"""
    if not batch_mode:
        return ensure_output_dir(output_root)
    relative = segmented_path.relative_to(input_root)
    return ensure_output_dir(
        output_root / relative.parent / normalized_stem(relative)
    )


def process_scene(segmented_path, render_path, output_dir, args):
    """识别单个分割结果中的杆塔，并从原始 LAS 裁剪杆塔区域。"""
    scene_start = time.perf_counter()
    print(f"Reading segmented LAS {segmented_path}", flush=True)
    las = laspy.read(segmented_path)
    if "classification" not in set(las.point_format.dimension_names):
        raise ValueError(f"分割 LAS 没有 classification 字段：{segmented_path}")

    coord = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    cls = np.asarray(las.classification, dtype=np.int32)

    if render_path == segmented_path:
        render_las = las
        render_coord = coord
    else:
        print(f"Reading original LAS {render_path}", flush=True)
        render_las = laspy.read(render_path)
        render_coord = np.column_stack(
            [render_las.x, render_las.y, render_las.z]
        ).astype(np.float64)

    if args.mode == "tower-only" and render_path != segmented_path:
        same_points = render_coord.shape == coord.shape and np.allclose(
            render_coord, coord, rtol=0.0, atol=float(np.max(render_las.header.scales))
        )
        if not same_points:
            raise ValueError(
                "tower-only 输出原始 LAS 时要求两份点云数量、顺序和坐标一致；"
                "当前实验建议使用 --mode local-scene。"
            )

    render_dimensions = set(render_las.point_format.dimension_names)
    render_cls = (
        np.asarray(render_las.classification, dtype=np.int32)
        if "classification" in render_dimensions
        else None
    )
    tower_mask = cls == int(args.tower_class)
    tower_coord = coord[tower_mask]
    tower_indices = np.where(tower_mask)[0]
    print(
        f"Loaded {coord.shape[0]} points; tower_points={tower_coord.shape[0]}",
        flush=True,
    )

    towers = build_towers(tower_coord, tower_indices, args)
    print(f"Found {len(towers)} tower instances", flush=True)

    outputs = []
    for tower in towers:
        crop_indices = crop_indices_for_tower(render_coord, tower, args)
        if crop_indices.shape[0] < int(args.min_output_points):
            continue

        filename = f"tower_{int(tower['id']):03d}_{tower['name']}.las"
        output_path = output_dir / filename
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists, use --overwrite: {output_path}")

        write_tower_las(render_las, crop_indices, output_path)
        outputs.append(
            {
                "tower_id": int(tower["id"]),
                "tower_name": tower["name"],
                "path": str(output_path),
                "mode": args.mode,
                "tower_point_count": int(tower["point_count"]),
                "output_point_count": int(crop_indices.shape[0]),
                "bbox_min_xyz": to_float_list(tower["bbox_min"]),
                "bbox_max_xyz": to_float_list(tower["bbox_max"]),
                "center_xyz": to_float_list(tower["center"]),
                "extent_xyz": to_float_list(tower["extent"]),
                "height": round(float(tower["height"]), 6),
                "original_class_counts": class_counts(render_cls[crop_indices])
                if render_cls is not None
                else {},
            }
        )
        print(f"  wrote {filename}: {crop_indices.shape[0]} points", flush=True)

    return {
        "segmented_las": str(segmented_path),
        "original_las": str(render_path),
        "output_dir": str(output_dir),
        "total_points": int(coord.shape[0]),
        "original_points": int(render_coord.shape[0]),
        "tower_points": int(tower_coord.shape[0]),
        "tower_instances": int(len(towers)),
        "written_files": int(len(outputs)),
        "elapsed_sec": round(time.perf_counter() - scene_start, 3),
        "towers": outputs,
    }


def main():
    args = parse_args()
    start_time = time.perf_counter()
    input_path = Path(args.input)
    output_root = ensure_output_dir(args.output_dir)
    report = report_path(args, output_root)
    segmented_files = las_files(input_path)
    batch_mode = input_path.is_dir()
    render_root = Path(args.render_base_las) if args.render_base_las else None
    render_matches = match_render_files(segmented_files, input_path, render_root)

    print(
        f"Scenes to process: {len(segmented_files)}, batch_mode={batch_mode}",
        flush=True,
    )
    scenes = []
    for scene_index, segmented_path in enumerate(segmented_files, start=1):
        print(f"[{scene_index}/{len(segmented_files)}] {segmented_path}", flush=True)
        output_dir = scene_output_dir(
            output_root, input_path, segmented_path, batch_mode
        )
        scenes.append(
            process_scene(
                segmented_path,
                render_matches[segmented_path],
                output_dir,
                args,
            )
        )

    elapsed = time.perf_counter() - start_time
    data = {
        "input": str(input_path),
        "render_base_las": args.render_base_las,
        "output_dir": str(output_root),
        "tower_class": int(args.tower_class),
        "mode": args.mode,
        "parameters": {
            "tower_voxel_size": float(args.tower_voxel_size),
            "tower_connectivity": int(args.tower_connectivity),
            "min_tower_points": int(args.min_tower_points),
            "min_tower_height": float(args.min_tower_height),
            "crop_xy_margin": float(args.crop_xy_margin),
            "crop_z_margin": float(args.crop_z_margin),
        },
        "summary": {
            "scenes": int(len(scenes)),
            "total_points": int(sum(scene["total_points"] for scene in scenes)),
            "original_points": int(
                sum(scene["original_points"] for scene in scenes)
            ),
            "tower_points": int(sum(scene["tower_points"] for scene in scenes)),
            "tower_instances": int(
                sum(scene["tower_instances"] for scene in scenes)
            ),
            "written_files": int(sum(scene["written_files"] for scene in scenes)),
            "elapsed_sec": round(elapsed, 3),
        },
        "scenes": scenes,
    }
    with report.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(
        f"Wrote report {report}; scenes={len(scenes)}, "
        f"files={data['summary']['written_files']}, elapsed={elapsed:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
    
    
""" 
使用方法：
使用分割后的颜色：
python tools/infer/split_las_by_tower.py \
  --input /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb_two_stage_stage1.las \
  --output-dir /24085403037/24085403037/shixi/dataset/6_23_demo/test/test_insulator_hengdan/source_tower/Stage1_tower \
  --mode local-scene \
  --crop-xy-margin 15 \
  --crop-z-margin 8 \
  --overwrite
  
使用原始的颜色：
python tools/infer/split_las_by_tower.py \
  --input /24085403037/24085403037/shixi/dataset/6_23_demo/lidar/cloudb009b5736892392a_infer.las \
  --render-base-las /24085403037/24085403037/shixi/dataset/6_23_demo/lidar/cloudb009b5736892392a.las \
  --output-dir /24085403037/24085403037/shixi/dataset/6_23_demo/test/original \
  --mode local-scene \
  --crop-xy-margin 15 \
  --crop-z-margin 8 \
  --overwrite

批量裁剪整个测试集中的原始杆塔区域：
python tools/infer/split_las_by_tower.py \
  --input /24085403037/24085403037/shixi/dataset/0617-4Name_Output_val_merged \
  --render-base-las /24085403037/24085403037/shixi/dataset/0617-4Name_original_val_merged \
  --output-dir /24085403037/24085403037/shixi/dataset/0617-4Name_original_val_tower_regions \
  --mode local-scene \
  --tower-voxel-size 0.75 \
  --min-tower-points 200 \
  --min-tower-height 4.0 \
  --crop-xy-margin 15 \
  --crop-z-margin 8 \
  --overwrite
  
"""
