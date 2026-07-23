"""恢复输电线路语义分割中被误分为背景的杆塔点和导线点。

输入 LAS/LAZ 的 classification 默认约定：
    0：背景，1：杆塔，2：导线，3：绝缘子。

恢复原则：
1. 先从 class=1 中提取点数、高度合格的杆塔连通组件，作为可靠杆塔种子；
2. 候选点只有同时满足最近距离和邻域支持点数要求，才恢复为杆塔；
3. 使用原始 class=2 点作为可靠导线种子，以同样方式恢复导线漏分点；
4. 默认只恢复 class=0，避免覆盖绝缘子或把杆塔、导线互相改类；
5. 不增加或删除点，只更新 classification 和 RGB。

单文件示例：
python tools/infer/postprocess_tower_line_clean.py \
  --input /24085403037/24085403037/shixi/dataset/lidar_test/cloud0_normal.las \
  --output /24085403037/24085403037/shixi/dataset/lidar_tower_line_boxes/cloud0_clean.las \
  --tower-nearby-component-max-gap 1.5 \
  --tower-clutter-lower-height-ratio 0.50 \
  --tower-clutter-base-radius 4.0 \
  --overwrite

目录批处理示例：
python tools/box_lineAndTower/recover_tower_line_labels.py \
  --input segmented_las_dir \
  --output recovered_las_dir \
  --overwrite
"""

import argparse
import json
import time
from itertools import product
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree


CLASS_COLORS_16 = {
    0: (37008, 37008, 37008),
    1: (58880, 16640, 14080),
    2: (11520, 32000, 65280),
    3: (65280, 53760, 10240),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="恢复语义分割中紧邻可靠杆塔和导线结构的漏分点。"
    )
    parser.add_argument("--input", required=True, help="输入 LAS/LAZ 文件或目录。")
    parser.add_argument("--output", required=True, help="输出 LAS/LAZ 文件或目录。")
    parser.add_argument(
        "--report",
        default=None,
        help=(
            "恢复报告 JSON。单文件模式可指定文件；目录模式必须指定目录。"
            "默认在每个输出 LAS 旁生成同名 _recovery_report.json。"
        ),
    )
    parser.add_argument("--background-class", type=int, default=0)
    parser.add_argument("--tower-class", type=int, default=1)
    parser.add_argument("--line-class", type=int, default=2)
    parser.add_argument("--insulator-class", type=int, default=3)

    tower_group = parser.add_argument_group("杆塔漏分恢复")
    tower_group.add_argument(
        "--tower-recover-from-classes",
        type=int,
        nargs="+",
        default=[0],
        help="允许恢复成杆塔的原类别，默认只处理背景类 0。",
    )
    tower_group.add_argument(
        "--tower-component-voxel-size",
        type=float,
        default=0.50,
        help="筛选可靠杆塔组件时的体素边长，单位米。",
    )
    tower_group.add_argument(
        "--tower-connectivity",
        choices=("6", "18", "26"),
        default="26",
        help="可靠杆塔体素组件的邻接方式。",
    )
    tower_group.add_argument(
        "--min-tower-points",
        type=int,
        default=200,
        help="可靠杆塔组件至少包含的原始杆塔点数。",
    )
    tower_group.add_argument(
        "--min-tower-height",
        type=float,
        default=4.0,
        help="可靠杆塔组件的最小高度，单位米。",
    )
    tower_group.add_argument(
        "--tower-recover-radius",
        type=float,
        default=0.25,
        help="候选点到最近可靠杆塔点的最大距离，单位米。",
    )
    tower_group.add_argument(
        "--tower-support-radius",
        type=float,
        default=0.75,
        help="统计杆塔邻域支持点的半径，单位米。",
    )
    tower_group.add_argument(
        "--min-tower-support-points",
        type=int,
        default=5,
        help="候选点附近至少需要的可靠杆塔种子点数。",
    )
    tower_group.add_argument(
        "--disable-tower-recovery",
        action="store_true",
        help="关闭杆塔漏分恢复。",
    )

    line_group = parser.add_argument_group("导线漏分恢复")
    line_group.add_argument(
        "--line-recover-from-classes",
        type=int,
        nargs="+",
        default=[0],
        help="允许恢复成导线的原类别，默认只处理背景类 0。",
    )
    line_group.add_argument(
        "--line-recover-radius",
        type=float,
        default=0.15,
        help="候选点到最近导线种子点的最大距离，单位米。",
    )
    line_group.add_argument(
        "--line-support-radius",
        type=float,
        default=0.60,
        help="统计导线邻域支持点的半径，单位米。",
    )
    line_group.add_argument(
        "--min-line-support-points",
        type=int,
        default=5,
        help="候选点附近至少需要的导线种子点数。",
    )
    line_group.add_argument(
        "--disable-line-recovery",
        action="store_true",
        help="关闭导线漏分恢复。",
    )

    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=500000,
        help="KDTree 分块查询的候选点数，减小大场景内存峰值。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="KDTree 查询线程数，-1 表示使用全部 CPU 核心。",
    )
    parser.add_argument(
        "--no-recolor",
        action="store_true",
        help="只更新 classification，不更新 RGB。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出。")
    return parser.parse_args()


def supported_las(path):
    return path.is_file() and path.suffix.lower() in (".las", ".laz")


def iter_jobs(input_path, output_path):
    """生成单文件或递归目录处理任务。"""
    if input_path.is_file():
        if not supported_las(input_path):
            raise ValueError(f"输入只支持 LAS/LAZ：{input_path}")
        if output_path.suffix.lower() not in (".las", ".laz"):
            raise ValueError(f"单文件模式输出必须是 LAS/LAZ：{output_path}")
        return [(input_path, output_path)]
    if not input_path.is_dir():
        raise FileNotFoundError(f"输入不存在：{input_path}")
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("目录模式下 --output 必须是目录。")
    sources = sorted(path for path in input_path.rglob("*") if supported_las(path))
    if not sources:
        raise FileNotFoundError(f"目录中没有 LAS/LAZ：{input_path}")
    return [(path, output_path / path.relative_to(input_path)) for path in sources]


def resolve_report_path(args, input_path, source_path, target_path):
    """为单文件或目录任务确定报告路径。"""
    if args.report is None:
        return target_path.with_name(f"{target_path.stem}_recovery_report.json")
    report_root = Path(args.report)
    if input_path.is_file():
        if report_root.suffix.lower() != ".json":
            raise ValueError("单文件模式下 --report 必须是 .json 文件。")
        return report_root
    if report_root.suffix:
        raise ValueError("目录模式下 --report 必须是目录。")
    relative = source_path.relative_to(input_path)
    return (report_root / relative).with_suffix(".json")


def neighbor_offsets(connectivity):
    """生成指定体素连通方式的一半对称邻域。"""
    offsets = []
    for offset in product((-1, 0, 1), repeat=3):
        if offset == (0, 0, 0) or offset <= (0, 0, 0):
            continue
        nonzero = sum(value != 0 for value in offset)
        if connectivity == 6 and nonzero > 1:
            continue
        if connectivity == 18 and nonzero > 2:
            continue
        offsets.append(np.asarray(offset, dtype=np.int64))
    return offsets


def voxel_component_labels(points, voxel_size, connectivity):
    """通过体素邻接和并查集生成点级连通组件标签。"""
    if points.shape[0] == 0:
        return np.empty((0,), dtype=np.int32)
    if voxel_size <= 0:
        raise ValueError("--tower-component-voxel-size 必须大于 0。")

    origin = points.min(axis=0)
    grid = np.floor((points - origin) / float(voxel_size)).astype(np.int64)
    voxels, point_to_voxel = np.unique(grid, axis=0, return_inverse=True)
    lookup = {tuple(voxel): index for index, voxel in enumerate(voxels)}
    parent = np.arange(voxels.shape[0], dtype=np.int64)
    rank = np.zeros(voxels.shape[0], dtype=np.uint8)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    offsets = neighbor_offsets(int(connectivity))
    for voxel_index, voxel in enumerate(voxels):
        for offset in offsets:
            neighbor_index = lookup.get(tuple(voxel + offset))
            if neighbor_index is not None:
                union(voxel_index, neighbor_index)

    roots = np.asarray([find(index) for index in range(voxels.shape[0])])
    _, voxel_labels = np.unique(roots, return_inverse=True)
    return voxel_labels[point_to_voxel].astype(np.int32, copy=False)


def reliable_tower_seed_indices(coord, classification, args):
    """从杆塔语义点中保留点数和高度合格的可靠组件。"""
    tower_indices = np.flatnonzero(classification == int(args.tower_class))
    if tower_indices.size == 0:
        return np.empty((0,), dtype=np.int64), 0, 0
    tower_points = coord[tower_indices]
    labels = voxel_component_labels(
        tower_points,
        args.tower_component_voxel_size,
        args.tower_connectivity,
    )
    keep = np.zeros(tower_indices.shape[0], dtype=bool)
    total_components = 0
    kept_components = 0
    for label in np.unique(labels):
        local_indices = np.flatnonzero(labels == label)
        total_components += 1
        if local_indices.size < int(args.min_tower_points):
            continue
        points = tower_points[local_indices]
        if float(np.ptp(points[:, 2])) < float(args.min_tower_height):
            continue
        keep[local_indices] = True
        kept_components += 1
    return tower_indices[keep], total_components, kept_components


def recover_supported_candidates(
    coord,
    classification,
    seed_indices,
    from_classes,
    recover_radius,
    support_radius,
    min_support_points,
    chunk_size,
    workers,
):
    """使用最近距离和第 K 个邻点距离共同判断可恢复候选点。"""
    if seed_indices.size == 0:
        return np.empty((0,), dtype=np.int64)
    if recover_radius <= 0 or support_radius <= 0:
        raise ValueError("恢复半径和支持半径必须大于 0。")
    if support_radius < recover_radius:
        raise ValueError("支持半径不能小于恢复半径。")
    if min_support_points < 1:
        raise ValueError("邻域支持点数必须至少为 1。")
    if chunk_size < 1:
        raise ValueError("--query-chunk-size 必须至少为 1。")
    if seed_indices.size < int(min_support_points):
        return np.empty((0,), dtype=np.int64)

    candidate_mask = np.isin(
        classification,
        np.asarray(sorted(set(from_classes)), dtype=classification.dtype),
    )
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        return np.empty((0,), dtype=np.int64)

    seed_points = coord[seed_indices]
    tree = cKDTree(seed_points)
    k = int(min_support_points)
    recovered_parts = []
    for start in range(0, candidate_indices.size, int(chunk_size)):
        indices = candidate_indices[start : start + int(chunk_size)]
        distances, _ = tree.query(
            coord[indices],
            k=k,
            distance_upper_bound=float(support_radius),
            workers=int(workers),
        )
        if k == 1:
            distances = distances[:, None]
        nearest_ok = distances[:, 0] <= float(recover_radius)
        support_ok = distances[:, -1] <= float(support_radius)
        selected = indices[nearest_ok & support_ok]
        if selected.size:
            recovered_parts.append(selected)
    if not recovered_parts:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(recovered_parts).astype(np.int64, copy=False)


def recolor(las, classification, args):
    """按照恢复后的四分类更新 RGB。"""
    if args.no_recolor:
        return
    dimensions = set(las.point_format.dimension_names)
    if not {"red", "green", "blue"}.issubset(dimensions):
        print("警告：输入 LAS 没有 RGB 字段，只更新 classification。", flush=True)
        return
    for class_id, color in CLASS_COLORS_16.items():
        mask = classification == int(class_id)
        if not np.any(mask):
            continue
        las.red[mask] = color[0]
        las.green[mask] = color[1]
        las.blue[mask] = color[2]


def process_one(source_path, target_path, report_path, args):
    """恢复一个 LAS/LAZ 中的杆塔和导线漏分点。"""
    if target_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在，请添加 --overwrite：{target_path}")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"报告已存在，请添加 --overwrite：{report_path}")

    start = time.perf_counter()
    las = laspy.read(source_path)
    if "classification" not in set(las.point_format.dimension_names):
        raise ValueError(f"输入没有 classification 字段：{source_path}")
    coord = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    classification = np.asarray(las.classification, dtype=np.uint8).copy()
    before_counts = np.bincount(classification, minlength=32)

    reliable_tower_indices = np.empty((0,), dtype=np.int64)
    tower_components = 0
    kept_tower_components = 0
    recovered_tower_indices = np.empty((0,), dtype=np.int64)
    if not args.disable_tower_recovery:
        (
            reliable_tower_indices,
            tower_components,
            kept_tower_components,
        ) = reliable_tower_seed_indices(coord, classification, args)
        recovered_tower_indices = recover_supported_candidates(
            coord,
            classification,
            reliable_tower_indices,
            args.tower_recover_from_classes,
            args.tower_recover_radius,
            args.tower_support_radius,
            args.min_tower_support_points,
            args.query_chunk_size,
            args.workers,
        )
        classification[recovered_tower_indices] = int(args.tower_class)

    recovered_line_indices = np.empty((0,), dtype=np.int64)
    line_seed_indices = np.flatnonzero(classification == int(args.line_class))
    if not args.disable_line_recovery:
        recovered_line_indices = recover_supported_candidates(
            coord,
            classification,
            line_seed_indices,
            args.line_recover_from_classes,
            args.line_recover_radius,
            args.line_support_radius,
            args.min_line_support_points,
            args.query_chunk_size,
            args.workers,
        )
        classification[recovered_line_indices] = int(args.line_class)

    las.classification = classification
    recolor(las, classification, args)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(target_path)

    elapsed = time.perf_counter() - start
    after_counts = np.bincount(classification, minlength=32)
    report = {
        "input": str(source_path),
        "output": str(target_path),
        "point_count": int(coord.shape[0]),
        "class_counts_before": {
            str(index): int(count)
            for index, count in enumerate(before_counts)
            if count
        },
        "class_counts_after": {
            str(index): int(count)
            for index, count in enumerate(after_counts)
            if count
        },
        "tower_components_total": int(tower_components),
        "tower_components_used_as_seed": int(kept_tower_components),
        "reliable_tower_seed_points": int(reliable_tower_indices.size),
        "line_seed_points": int(line_seed_indices.size),
        "recovered_tower_points": int(recovered_tower_indices.size),
        "recovered_line_points": int(recovered_line_indices.size),
        "elapsed_sec": round(float(elapsed), 3),
        "parameters": vars(args),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return report


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    jobs = iter_jobs(input_path, output_path)

    total_start = time.perf_counter()
    total_tower = 0
    total_line = 0
    for index, (source_path, target_path) in enumerate(jobs, start=1):
        report_path = resolve_report_path(
            args,
            input_path,
            source_path,
            target_path,
        )
        print(f"[{index}/{len(jobs)}] 处理：{source_path}", flush=True)
        report = process_one(source_path, target_path, report_path, args)
        total_tower += int(report["recovered_tower_points"])
        total_line += int(report["recovered_line_points"])
        print(
            "  完成：杆塔恢复={}，导线恢复={}，耗时={:.2f}s".format(
                report["recovered_tower_points"],
                report["recovered_line_points"],
                report["elapsed_sec"],
            ),
            flush=True,
        )
        print(f"  输出：{target_path}", flush=True)
        print(f"  报告：{report_path}", flush=True)

    total_time = time.perf_counter() - total_start
    print(
        "全部完成：文件数={}，杆塔恢复总数={}，导线恢复总数={}，"
        "总耗时={:.2f}s，平均每个LAS={:.2f}s".format(
            len(jobs),
            total_tower,
            total_line,
            total_time,
            total_time / len(jobs),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
