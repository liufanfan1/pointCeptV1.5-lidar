"""Clean tower false positives and add tower/line-span boxes to segmented LAS.

Input is a LAS produced by tools/infer_las_semseg.py:

    classification 0=background, 1=tower, 2=line, 3=insulator

The script:
1. clusters predicted tower points into 3D connected components;
2. removes components that are too small/flat, optionally without nearby line;
3. appends sampled box-edge points around each retained tower;
4. sorts retained towers along the main XY direction and appends one box around
   line points between each adjacent tower pair.

The output LAS keeps original points and appends synthetic edge points with
classification=31. RGB is recolored when the LAS point format supports RGB.
"""
# 后处理的脚本：
# 清理杆塔误分：主要是对class=1的杆塔点做聚类，然后根据点数、高度、附近是否有导线等条件，把明显不像杆塔的小块过滤掉。
# 恢复杆塔底部：会在真实杆塔底部附近找一些小组件，把它们恢复为杆塔的一部分，避免杆塔框底部缺失太严重。
# 给每个杆塔生成框：从左到右依次生成OBB旋转框，杆塔1、2......
# 给两个杆塔之间的整体线路段生成框。
# 拟合每个档距里的每根导线
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


laspy = None


DEFAULT_CLASS_COLORS_16 = {
    0: (37008, 37008, 37008),  # background / ground, gray
    1: (58880, 16640, 14080),  # tower, red
    2: (11520, 32000, 65280),  # line, blue
    3: (65280, 53760, 10240),  # insulator, yellow
}
TOWER_BOX_COLOR_16 = np.array([65535, 0, 65535], dtype=np.uint16)
LINE_BOX_COLOR_16 = np.array([0, 65535, 65535], dtype=np.uint16)
REMOVED_TOWER_COLOR_16 = np.array([18000, 18000, 18000], dtype=np.uint16)
BOX_CLASS = 31
FITTED_CONDUCTOR_CLASS = 30
BOX_CORNER_SIGNS = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float64,
)
BOX_EDGE_PAIRS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]
CONDUCTOR_COLORS_16 = np.array(
    [
        [0, 65535, 65535],
        [65535, 32768, 0],
        [0, 65535, 0],
        [65535, 0, 65535],
        [65535, 65535, 0],
        [0, 32768, 65535],
        [65535, 0, 0],
        [32768, 65535, 0],
        [32768, 0, 65535],
        [0, 65535, 32768],
        [65535, 32768, 32768],
        [32768, 32768, 65535],
    ],
    dtype=np.uint16,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess segmented transmission-line LAS: remove obvious tower "
            "false positives, box towers, and box line spans between towers."
        )
    )
    parser.add_argument("--input", required=True, help="Segmented input LAS/LAZ.")
    parser.add_argument("--output", required=True, help="Output LAS/LAZ.")
    parser.add_argument(
        "--report",
        default=None,
        help="Output JSON report. Default: <output stem>_tower_line_report.json",
    ) # 调试报告JSON
    parser.add_argument(
        "--tower-box-report",
        default=None,
        help=(
            "Optional extra tower-only OBB JSON. By default only the combined "
            "box JSON is written."
        ),
    ) # 只保存杆塔框JSON
    parser.add_argument(
        "--line-box-report",
        default=None,
        help=(
            "Output drawable OBB JSON containing both tower boxes and line boxes. "
            "Default: <report stem>_boxes.json"
        ),
    )# 旧参数名，保存框JSON文件
    parser.add_argument(
        "--combined-box-report",
        default=None,
        help=(
            "Output combined OBB JSON with tower and line boxes. If omitted, "
            "the --line-box-report path is used."
        ),
    )# 保存杆塔框 + 整体线路框JSON
    parser.add_argument(
        "--conductor-report",
        default=None,
        help=(
            "Output render JSON for fitted conductors grouped by span. Default: "
            "<report stem>_conductors.json"
        ),
    )# 保存每个档距下的每根导线的JSON
    parser.add_argument(
        "--obb-origin",
        choices=("las-offset", "las-min", "zero", "custom"),
        default="las-offset",
        help=(
            "Origin used for obb[0:3] relative coordinates. obb_global.lat_lng_alt "
            "stores this origin. Default: LAS header offsets."
        ),
    )# 控制JSON中的OBB旋转框使用什么相对坐标原点
    # las-offset  使用 LAS header offset，推荐
    # las-min     使用点云最小 xyz
    # zero        不用相对坐标，直接接近全局坐标
    # custom      自定义原点
    parser.add_argument(
        "--obb-origin-xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Custom origin for --obb-origin custom.",
    )# 自定义原点
  
    parser.add_argument("--background-class", type=int, default=0)
    parser.add_argument("--tower-class", type=int, default=1)
    parser.add_argument("--line-class", type=int, default=2)
    parser.add_argument("--insulator-class", type=int, default=3)
      # 类别定义
   
    parser.add_argument(
        "--tower-voxel-size",
        type=float,
        default=0.50,
        help="Voxel size in meters for tower connected-component clustering.",
    )
     # 杆塔点聚类用的体素大小
   
    parser.add_argument(
        "--tower-connectivity",
        choices=("6", "18", "26"),
        default="26",
        help="Voxel neighborhood used to connect tower components.",
    ) # 体素连通方式6/8/26
    
    parser.add_argument(
        "--min-tower-points",
        type=int,
        default=500,
        help="Remove tower components with fewer original points than this.",
    )# 杆塔组件最少点数。低于这个点数的杆塔候选框会被当做误检删掉
    parser.add_argument(
        "--min-tower-height",
        type=float,
        default=4.0,
        help="Remove tower components whose z height is lower than this many meters.",
    )# 杆塔组件的最小高度
    parser.add_argument(
        "--min-tower-xy-size",
        type=float,
        default=0.30,
        help="Remove tower components too tiny in both x and y.",
    )# 杆塔组件在XY平面的最小尺寸
    parser.add_argument(
        "--require-line-near-tower",
        action="store_true",
        help="Keep only tower components that have line points nearby.",
    )# 要求杆塔附近必须有导线点
    parser.add_argument(
        "--tower-line-radius",
        type=float,
        default=8.0,
        help="Radius in meters for --require-line-near-tower.",
    )# 判断“杆塔附近是否有导线”的搜索半径，单位为米.
    parser.add_argument(
        "--min-line-points-near-tower",
        type=int,
        default=0,
        help=(
            "Remove tower components with fewer line points than this inside "
            "--tower-line-radius. 0 disables this filter."
        ),
    )# 杆塔附近至少要有多少导线点。0 表示不启用这个过滤。
    parser.add_argument(
        "--min-tower-height-above-ground",
        type=float,
        default=0.0,
        help=(
            "Remove tower components whose top is less than this many meters "
            "above nearby background/ground points. 0 disables this filter."
        ),
    )# 杆塔顶部相对附近地面的最小高度。用于过滤贴地的假杆塔。
    parser.add_argument(
        "--local-ground-radius",
        type=float,
        default=8.0,
        help="XY radius in meters used to estimate local ground height.",
    )# 估计局部地面高度时，在杆塔附近搜索背景点的 XY 半径。
    parser.add_argument(
        "--merge-tower-xy-radius",
        type=float,
        default=12.0,
        help=(
            "Merge kept tower components whose XY centers are within this radius, "
            "so one physical tower gets one box."
        ),
    )
    parser.add_argument(
        "--no-recover-tower-base",
        action="store_true",
        help=(
            "Disable recovery of small low tower components near retained tower "
            "footprints. By default this protects tower bases from strict filtering."
        ),
    )
    parser.add_argument(
        "--recover-base-xy-margin",
        type=float,
        default=3.0,
        help="XY margin in meters around retained tower boxes for base recovery.",
    )
    parser.add_argument(
        "--recover-base-z-margin",
        type=float,
        default=8.0,
        help=(
            "Recover removed components whose max z is within this many meters "
            "above the retained tower bottom."
        ),
    )
    parser.add_argument(
        "--recover-base-min-points",
        type=int,
        default=20,
        help="Minimum component point count eligible for tower-base recovery.",
    )
    parser.add_argument(
        "--tower-box-margin",
        type=float,
        default=1.0,
        help="Margin in meters added to each tower box.",
    )
    parser.add_argument(
        "--line-box-margin",
        type=float,
        default=1.0,
        help="Margin in meters added to each between-tower line box.",
    )
    parser.add_argument(
        "--line-tower-gap",
        type=float,
        default=0.05,
        help=(
            "Gap in meters between tower boxes and line-span boxes along the "
            "span direction to avoid intersections."
        ),
    )
    parser.add_argument(
        "--line-box-mode",
        choices=("oriented", "axis"),
        default="oriented",
        help="oriented fits a rotated box to the line direction; axis uses XYZ boxes.",
    )
    parser.add_argument(
        "--line-fit-percentile",
        type=float,
        default=1.0,
        help=(
            "Percentile trimming for oriented line boxes. 1 means use 1..99%% "
            "to reduce outlier points before adding margin."
        ),
    )
    parser.add_argument(
        "--line-min-box-width",
        type=float,
        default=1.0,
        help="Minimum oriented line-box width in meters.",
    )
    parser.add_argument(
        "--line-min-box-height",
        type=float,
        default=1.0,
        help="Minimum oriented line-box height in meters.",
    )
    parser.add_argument(
        "--line-corridor-width",
        type=float,
        default=12.0,
        help="Max XY distance from the tower-to-tower segment when selecting line points.",
    )
    parser.add_argument(
        "--span-end-margin",
        type=float,
        default=5.0,
        help="Extra meters before/after adjacent tower centers when selecting line spans.",
    )
    parser.add_argument(
        "--min-span-line-points",
        type=int,
        default=50,
        help="Skip a between-tower line box if fewer line points are selected.",
    )
    parser.add_argument(
        "--edge-step",
        type=float,
        default=0.25,
        help="Spacing in meters for synthetic box-edge points.",
    )
    parser.add_argument(
        "--max-towers",
        type=int,
        default=0,
        help="Optional cap on retained towers after filtering. 0 keeps all.",
    )
    parser.add_argument(
        "--reverse-tower-order",
        action="store_true",
        help="Reverse left-to-right tower numbering if the automatic order is opposite.",
    )
    parser.add_argument(
        "--no-fit-conductors",
        action="store_true",
        help="Disable per-conductor fitting/coloring.",
    )
    parser.add_argument(
        "--conductor-bin-size",
        type=float,
        default=8.0,
        help="Along-span bin length in meters used to find conductor cross-section centers.",
    )
    parser.add_argument(
        "--conductor-cluster-radius",
        type=float,
        default=1.2,
        help="Radius in side/z meters for clustering conductor cross-section centers.",
    )
    parser.add_argument(
        "--conductor-track-radius",
        type=float,
        default=2.5,
        help="Max side/z distance for linking conductor centers across adjacent bins.",
    )
    parser.add_argument(
        "--min-conductor-points-per-bin",
        type=int,
        default=30,
        help="Minimum line points required for a cross-section cluster.",
    )
    parser.add_argument(
        "--min-conductor-track-bins",
        type=int,
        default=4,
        help="Minimum number of along-span bins needed to keep a fitted conductor.",
    )
    parser.add_argument(
        "--conductor-fit-step",
        type=float,
        default=2.0,
        help="Point spacing in meters for appended fitted conductor polylines.",
    )
    parser.add_argument(
        "--conductor-json-step",
        type=float,
        default=8.0,
        help=(
            "Point spacing in meters for saved conductor render JSON. Larger "
            "means fewer points."
        ),
    )
    parser.add_argument(
        "--conductor-poly-degree",
        type=int,
        default=2,
        help="Polynomial degree for fitting side/z against along distance.",
    )
    parser.add_argument(
        "--conductor-assign-radius",
        type=float,
        default=2.0,
        help="Max side/z distance for coloring original line points by fitted conductor.",
    )
    parser.add_argument(
        "--no-append-conductor-fit",
        action="store_true",
        help="Color original line points only; do not append fitted conductor points.",
    )
    parser.add_argument(
        "--no-recolor",
        action="store_true",
        help="Do not recolor original semantic classes.",
    )
    parser.add_argument(
        "--no-append-box-points",
        action="store_true",
        help="Only clean labels/recolor points; do not append box edge points.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def coords_from_las(las):
    return np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)


def has_rgb(las):
    dims = set(las.point_format.dimension_names)
    return {"red", "green", "blue"}.issubset(dims)


def recolor_by_class(las, cls, args):
    if args.no_recolor or not has_rgb(las):
        return
    for label, color in DEFAULT_CLASS_COLORS_16.items():
        mask = cls == label
        if not np.any(mask):
            continue
        las.red[mask] = color[0]
        las.green[mask] = color[1]
        las.blue[mask] = color[2]


def neighbor_offsets(connectivity):
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if connectivity == "6" and manhattan != 1:
                    continue
                if connectivity == "18" and manhattan > 2:
                    continue
                # Use only half of the symmetric neighborhood.
                if (dx, dy, dz) > (0, 0, 0):
                    offsets.append((dx, dy, dz))
    return offsets


def linear_keys(grid):
    shifted = grid - grid.min(axis=0)
    dims = shifted.max(axis=0).astype(np.int64) + 1
    return (shifted[:, 0] * dims[1] + shifted[:, 1]) * dims[2] + shifted[:, 2], dims


def connected_components_from_voxels(voxels, connectivity):
    keys, dims = linear_keys(voxels)
    order = np.argsort(keys)
    keys_sorted = keys[order]
    voxels_sorted = voxels[order]
    edge_left = []
    edge_right = []

    for offset in neighbor_offsets(connectivity):
        neigh = voxels_sorted + np.asarray(offset, dtype=np.int64)
        neigh_shifted = neigh - voxels.min(axis=0)
        valid = np.all((neigh_shifted >= 0) & (neigh_shifted < dims), axis=1)
        if not np.any(valid):
            continue
        neigh_keys = (
            (neigh_shifted[valid, 0] * dims[1] + neigh_shifted[valid, 1])
            * dims[2]
            + neigh_shifted[valid, 2]
        )
        pos = np.searchsorted(keys_sorted, neigh_keys)
        found = pos < keys_sorted.size
        if np.any(found):
            found[found] = keys_sorted[pos[found]] == neigh_keys[found]
        if np.any(found):
            left = np.nonzero(valid)[0][found]
            right = pos[found]
            edge_left.append(order[left])
            edge_right.append(order[right])

    if not edge_left:
        return np.arange(voxels.shape[0], dtype=np.int64)

    rows = np.concatenate(edge_left)
    cols = np.concatenate(edge_right)
    graph = coo_matrix(
        (np.ones(rows.shape[0], dtype=np.uint8), (rows, cols)),
        shape=(voxels.shape[0], voxels.shape[0]),
    )
    _, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.astype(np.int64, copy=False)


def cluster_towers(coord, cls, args):
    tower_indices = np.flatnonzero(cls == args.tower_class)
    if tower_indices.size == 0:
        return [], tower_indices, np.empty(0, dtype=np.int64)

    tower_coord = coord[tower_indices]
    origin = tower_coord.min(axis=0)
    grid = np.floor((tower_coord - origin) / args.tower_voxel_size).astype(np.int64)
    voxels, inverse = np.unique(grid, axis=0, return_inverse=True)
    print(
        f"Tower clustering: {tower_indices.size} points -> {voxels.shape[0]} voxels",
        flush=True,
    )

    voxel_comp = connected_components_from_voxels(voxels, args.tower_connectivity)
    point_comp = voxel_comp[inverse]
    comp_count = int(point_comp.max()) + 1 if point_comp.size else 0

    counts = np.bincount(point_comp, minlength=comp_count)
    lo = np.full((comp_count, 3), np.inf, dtype=np.float64)
    hi = np.full((comp_count, 3), -np.inf, dtype=np.float64)
    np.minimum.at(lo, point_comp, tower_coord)
    np.maximum.at(hi, point_comp, tower_coord)

    components = []
    for comp_id in range(comp_count):
        bbox_min = lo[comp_id]
        bbox_max = hi[comp_id]
        size = bbox_max - bbox_min
        components.append(
            {
                "id": int(comp_id),
                "point_count": int(counts[comp_id]),
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "size": size,
                "center": (bbox_min + bbox_max) / 2.0,
                "keep": True,
                "remove_reason": "",
            }
        )
    return components, tower_indices, point_comp


def mark_components_to_keep(components, coord, cls, args):
    line_coord = coord[cls == args.line_class]
    line_tree = cKDTree(line_coord[:, :2]) if line_coord.shape[0] else None
    bg_coord = coord[cls == args.background_class]
    bg_tree = cKDTree(bg_coord[:, :2]) if bg_coord.shape[0] else None
    global_ground_z = float(np.percentile(bg_coord[:, 2], 2)) if bg_coord.shape[0] else 0.0

    for comp in components:
        size = comp["size"]
        comp["nearby_line_points"] = 0
        comp["height_above_ground"] = None
        if line_tree is not None and (
            args.require_line_near_tower or args.min_line_points_near_tower > 0
        ):
            hits = line_tree.query_ball_point(comp["center"][:2], args.tower_line_radius)
            comp["nearby_line_points"] = int(len(hits))
        if args.min_tower_height_above_ground > 0:
            ground_z = global_ground_z
            if bg_tree is not None:
                hits = bg_tree.query_ball_point(
                    comp["center"][:2], args.local_ground_radius
                )
                if hits:
                    ground_z = float(np.percentile(bg_coord[hits, 2], 5))
            comp["height_above_ground"] = float(comp["bbox_max"][2] - ground_z)

        if comp["point_count"] < args.min_tower_points:
            comp["keep"] = False
            comp["remove_reason"] = "too_few_points"
        elif float(size[2]) < args.min_tower_height:
            comp["keep"] = False
            comp["remove_reason"] = "too_low"
        elif max(float(size[0]), float(size[1])) < args.min_tower_xy_size:
            comp["keep"] = False
            comp["remove_reason"] = "xy_too_small"
        elif (
            args.require_line_near_tower
            and line_tree is not None
            and comp["nearby_line_points"] == 0
        ):
            comp["keep"] = False
            comp["remove_reason"] = "no_nearby_line"
        elif (
            args.min_line_points_near_tower > 0
            and comp["nearby_line_points"] < args.min_line_points_near_tower
        ):
            comp["keep"] = False
            comp["remove_reason"] = "too_few_nearby_line_points"
        elif (
            args.min_tower_height_above_ground > 0
            and comp["height_above_ground"] < args.min_tower_height_above_ground
        ):
            comp["keep"] = False
            comp["remove_reason"] = "too_low_above_ground"


def apply_tower_cleanup(las, cls, tower_indices, point_comp, components, background_class):
    if tower_indices.size == 0:
        return np.empty(0, dtype=np.int64)
    keep_by_comp = np.asarray([comp["keep"] for comp in components], dtype=bool)
    keep_point = keep_by_comp[point_comp]
    remove_indices = tower_indices[~keep_point]
    cls[remove_indices] = background_class
    las.classification[remove_indices] = background_class
    if has_rgb(las) and remove_indices.size:
        las.red[remove_indices] = REMOVED_TOWER_COLOR_16[0]
        las.green[remove_indices] = REMOVED_TOWER_COLOR_16[1]
        las.blue[remove_indices] = REMOVED_TOWER_COLOR_16[2]
    return remove_indices


def merge_physical_towers(components, args):
    kept = [comp for comp in components if comp["keep"]]
    if not kept:
        return []
    if len(kept) == 1:
        comp = kept[0]
        return [
            {
                "id": 0,
                "component_ids": [int(comp["id"])],
                "point_count": int(comp["point_count"]),
                "bbox_min": comp["bbox_min"],
                "bbox_max": comp["bbox_max"],
                "center": comp["center"],
                "size": comp["size"],
            }
        ]

    centers = np.vstack([comp["center"] for comp in kept])
    if args.merge_tower_xy_radius <= 0:
        labels = np.arange(len(kept), dtype=np.int64)
        group_count = len(kept)
    else:
        pairs = np.asarray(
            list(cKDTree(centers[:, :2]).query_pairs(args.merge_tower_xy_radius)),
            dtype=np.int64,
        )
        if pairs.size == 0:
            labels = np.arange(len(kept), dtype=np.int64)
            group_count = len(kept)
        else:
            graph = coo_matrix(
                (
                    np.ones(pairs.shape[0], dtype=np.uint8),
                    (pairs[:, 0], pairs[:, 1]),
                ),
                shape=(len(kept), len(kept)),
            )
            group_count, labels = connected_components(
                graph, directed=False, return_labels=True
            )

    towers = []
    for group_id in range(group_count):
        group = [kept[i] for i in np.flatnonzero(labels == group_id)]
        bbox_min = np.min(np.vstack([comp["bbox_min"] for comp in group]), axis=0)
        bbox_max = np.max(np.vstack([comp["bbox_max"] for comp in group]), axis=0)
        point_count = int(sum(comp["point_count"] for comp in group))
        center = (bbox_min + bbox_max) / 2.0
        towers.append(
            {
                "id": int(group_id),
                "component_ids": [int(comp["id"]) for comp in group],
                "point_count": point_count,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "center": center,
                "size": bbox_max - bbox_min,
            }
        )

    if args.max_towers > 0 and len(towers) > args.max_towers:
        towers = sorted(towers, key=lambda item: item["point_count"], reverse=True)[
            : args.max_towers
        ]
        for new_id, tower in enumerate(towers):
            tower["id"] = int(new_id)
    return sorted(towers, key=lambda item: item["center"][0])


def recover_tower_base_components(components, seed_towers, args):
    if args.no_recover_tower_base or not seed_towers:
        return 0
    recovered = 0
    for comp in components:
        if comp["keep"] or comp["point_count"] < args.recover_base_min_points:
            continue
        comp_center_xy = comp["center"][:2]
        for tower in seed_towers:
            lo_xy = tower["bbox_min"][:2] - args.recover_base_xy_margin
            hi_xy = tower["bbox_max"][:2] + args.recover_base_xy_margin
            if not np.all((comp_center_xy >= lo_xy) & (comp_center_xy <= hi_xy)):
                continue
            z_limit = tower["bbox_min"][2] + args.recover_base_z_margin
            if comp["bbox_max"][2] > z_limit:
                continue
            comp["keep"] = True
            comp["remove_reason"] = "recovered_tower_base"
            recovered += 1
            break
    return recovered


def box_bounds(points, margin):
    lo = points.min(axis=0) - margin
    hi = points.max(axis=0) + margin
    return lo, hi


def axis_box_corners(lo, hi):
    return np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ],
        dtype=np.float64,
    )


def oriented_box_corners(center, axes, half_size):
    return center[None, :] + (BOX_CORNER_SIGNS * half_size[None, :]) @ axes


def corner_edges(corners):
    return [
        [corners[a].tolist(), corners[b].tolist()]
        for a, b in BOX_EDGE_PAIRS
    ]


def box_geometry_from_box(box):
    if box.get("box_mode") == "oriented":
        corners = oriented_box_corners(box["center"], box["axes"], box["half_size"])
    else:
        corners = axis_box_corners(box["bbox_min"], box["bbox_max"])
    return corners, corner_edges(corners)


def tower_projection_interval(tower_box, p0_xy, direction_xy):
    corners, _ = box_geometry_from_box(tower_box)
    rel = corners[:, :2] - p0_xy[None, :]
    proj = rel @ direction_xy
    return float(proj.min()), float(proj.max())


def sample_box_edges(lo, hi, step):
    corners = axis_box_corners(lo, hi)
    sampled = []
    for a, b in BOX_EDGE_PAIRS:
        p0, p1 = corners[a], corners[b]
        length = float(np.linalg.norm(p1 - p0))
        count = max(int(np.ceil(length / max(step, 1e-6))) + 1, 2)
        t = np.linspace(0.0, 1.0, count)
        sampled.append(p0[None, :] * (1.0 - t[:, None]) + p1[None, :] * t[:, None])
    return np.vstack(sampled)


def sample_oriented_box_edges(center, axes, half_size, step):
    corners = oriented_box_corners(center, axes, half_size)
    sampled = []
    for a, b in BOX_EDGE_PAIRS:
        p0, p1 = corners[a], corners[b]
        length = float(np.linalg.norm(p1 - p0))
        count = max(int(np.ceil(length / max(step, 1e-6))) + 1, 2)
        t = np.linspace(0.0, 1.0, count)
        sampled.append(p0[None, :] * (1.0 - t[:, None]) + p1[None, :] * t[:, None])
    return np.vstack(sampled)


def robust_bounds(values, trim_percentile):
    if trim_percentile <= 0:
        return float(values.min()), float(values.max())
    lo = float(np.percentile(values, trim_percentile))
    hi = float(np.percentile(values, 100.0 - trim_percentile))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def make_oriented_line_box(points, p0_xy, direction_xy, args, along_bounds=None):
    long_axis = np.array([direction_xy[0], direction_xy[1], 0.0], dtype=np.float64)
    side_axis = np.array([-direction_xy[1], direction_xy[0], 0.0], dtype=np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axes = np.vstack([long_axis, side_axis, z_axis])

    origin = np.array([p0_xy[0], p0_xy[1], 0.0], dtype=np.float64)
    rel = points - origin[None, :]
    along = rel @ long_axis
    side = rel @ side_axis
    height = points[:, 2]

    if along_bounds is None:
        along_lo, along_hi = robust_bounds(along, args.line_fit_percentile)
        along_lo -= args.line_box_margin
        along_hi += args.line_box_margin
    else:
        along_lo, along_hi = along_bounds
    side_lo, side_hi = robust_bounds(side, args.line_fit_percentile)
    z_lo, z_hi = robust_bounds(height, args.line_fit_percentile)

    side_lo -= args.line_box_margin
    side_hi += args.line_box_margin
    z_lo -= args.line_box_margin
    z_hi += args.line_box_margin

    if side_hi - side_lo < args.line_min_box_width:
        mid = (side_hi + side_lo) / 2.0
        side_lo = mid - args.line_min_box_width / 2.0
        side_hi = mid + args.line_min_box_width / 2.0
    if z_hi - z_lo < args.line_min_box_height:
        mid = (z_hi + z_lo) / 2.0
        z_lo = mid - args.line_min_box_height / 2.0
        z_hi = mid + args.line_min_box_height / 2.0

    center_local = np.array(
        [(along_lo + along_hi) / 2.0, (side_lo + side_hi) / 2.0, (z_lo + z_hi) / 2.0],
        dtype=np.float64,
    )
    half_size = np.array(
        [(along_hi - along_lo) / 2.0, (side_hi - side_lo) / 2.0, (z_hi - z_lo) / 2.0],
        dtype=np.float64,
    )
    center = origin + center_local @ axes
    return center, axes, half_size


def make_oriented_tower_box(points, margin):
    xy = points[:, :2]
    centered_xy = xy - xy.mean(axis=0, keepdims=True)
    if points.shape[0] >= 3 and np.any(np.abs(centered_xy) > 1e-9):
        cov = centered_xy.T @ centered_xy / max(points.shape[0] - 1, 1)
        values, vectors = np.linalg.eigh(cov)
        long_xy = vectors[:, int(np.argmax(values))]
    else:
        long_xy = np.array([1.0, 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(long_xy))
    if norm <= 1e-9:
        long_xy = np.array([1.0, 0.0], dtype=np.float64)
    else:
        long_xy = long_xy / norm
    if long_xy[0] < 0 or (abs(long_xy[0]) < 1e-9 and long_xy[1] < 0):
        long_xy = -long_xy
    axes = local_axes_from_direction(long_xy)
    local = points @ axes.T
    lo = local.min(axis=0) - margin
    hi = local.max(axis=0) + margin
    center_local = (lo + hi) / 2.0
    half_size = (hi - lo) / 2.0
    center = center_local @ axes
    return center, axes, half_size


def points_for_tower(coord, tower_indices, point_comp, component_ids):
    mask = np.isin(point_comp, np.asarray(component_ids, dtype=np.int64))
    return coord[tower_indices[mask]]


def tower_boxes(towers, coord, tower_indices, point_comp, margin):
    boxes = []
    for tower in towers:
        points = points_for_tower(coord, tower_indices, point_comp, tower["component_ids"])
        if points.size == 0:
            center = tower["center"]
            axes = np.eye(3, dtype=np.float64)
            half_size = (tower["bbox_max"] - tower["bbox_min"]) / 2.0 + margin
        else:
            center, axes, half_size = make_oriented_tower_box(points, margin)
        boxes.append(
            {
                "kind": "tower",
                "box_mode": "oriented",
                "tower_id": tower["id"],
                "tower_no": tower.get("tower_no", 0),
                "tower_name": tower.get("display_name", tower.get("name", "")),
                "component_ids": tower["component_ids"],
                "center": center,
                "axes": axes,
                "half_size": half_size,
                "point_count": tower["point_count"],
            }
        )
    return boxes


def main_axis_xy(centers):
    xy = centers[:, :2]
    if xy.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=np.float64)
    centered = xy - xy.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    norm = np.linalg.norm(axis)
    return axis / norm if norm > 0 else np.array([1.0, 0.0], dtype=np.float64)


def assign_tower_names(towers, args):
    if not towers:
        return []
    centers = np.vstack([tower["center"] for tower in towers])
    axis = main_axis_xy(centers)
    order = list(np.argsort(centers[:, :2] @ axis))
    if args.reverse_tower_order:
        order.reverse()
    ordered = [towers[int(i)] for i in order]
    for index, tower in enumerate(ordered, start=1):
        tower["tower_no"] = int(index)
        tower["name"] = f"tower_{index}"
        tower["display_name"] = f"杆塔{index}"
    return ordered


def local_axes_from_direction(direction_xy):
    long_axis = np.array([direction_xy[0], direction_xy[1], 0.0], dtype=np.float64)
    side_axis = np.array([-direction_xy[1], direction_xy[0], 0.0], dtype=np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return np.vstack([long_axis, side_axis, z_axis])


def world_to_line_local(points, p0_xy, direction_xy):
    axes = local_axes_from_direction(direction_xy)
    origin = np.array([p0_xy[0], p0_xy[1], 0.0], dtype=np.float64)
    local = (points - origin[None, :]) @ axes.T
    return local, axes, origin


def line_local_to_world(local, axes, origin):
    return origin[None, :] + local @ axes


def line_span_boxes(coord, cls, towers, tower_box_by_id, args):
    if len(towers) < 2:
        return []
    line_indices = np.flatnonzero(cls == args.line_class)
    if line_indices.size == 0:
        return []

    sorted_towers = sorted(towers, key=lambda item: item.get("tower_no", item["id"]))

    line_xy = coord[line_indices, :2]
    boxes = []
    for span_id, (left, right) in enumerate(
        zip(sorted_towers[:-1], sorted_towers[1:]), start=1
    ):
        p0 = left["center"][:2]
        p1 = right["center"][:2]
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length <= 1e-6:
            continue
        direction = vec / length
        left_box = tower_box_by_id.get(int(left["id"]))
        right_box = tower_box_by_id.get(int(right["id"]))
        if left_box is None or right_box is None:
            continue
        _, left_exit = tower_projection_interval(left_box, p0, direction)
        right_entry, _ = tower_projection_interval(right_box, p0, direction)
        span_start = left_exit + args.line_tower_gap
        span_end = right_entry - args.line_tower_gap
        if span_end <= span_start:
            continue
        rel = line_xy - p0[None, :]
        proj = rel @ direction
        perp = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])
        mask = (
            (proj >= span_start)
            & (proj <= span_end)
            & (perp <= args.line_corridor_width)
        )
        selected = line_indices[mask]
        if selected.size < args.min_span_line_points:
            continue
        box = {
            "kind": "line_span",
            "span_id": int(span_id),
            "left_tower_id": int(left["id"]),
            "right_tower_id": int(right["id"]),
            "left_tower_name": left.get("display_name", left.get("name", "")),
            "right_tower_name": right.get("display_name", right.get("name", "")),
            "line_point_count": int(selected.size),
            "_line_indices": selected,
            "_p0_xy": p0,
            "_direction_xy": direction,
        }
        if args.line_box_mode == "oriented":
            center, axes, half_size = make_oriented_line_box(
                coord[selected],
                p0,
                direction,
                args,
                along_bounds=(span_start, span_end),
            )
            box.update(
                {
                    "box_mode": "oriented",
                    "center": center,
                    "axes": axes,
                    "half_size": half_size,
                    "span_start_m": float(span_start),
                    "span_end_m": float(span_end),
                }
            )
        else:
            lo, hi = box_bounds(coord[selected], args.line_box_margin)
            box.update({"box_mode": "axis", "bbox_min": lo, "bbox_max": hi})
        boxes.append(box)
    return boxes


def cross_section_clusters(cross_coord, radius, min_points):
    if cross_coord.shape[0] < min_points:
        return []
    voxel_size = max(radius / 2.0, 1e-3)
    origin = cross_coord.min(axis=0)
    grid = np.floor((cross_coord - origin) / voxel_size).astype(np.int64)
    voxels, inverse, counts = np.unique(
        grid, axis=0, return_inverse=True, return_counts=True
    )
    side_sum = np.bincount(inverse, weights=cross_coord[:, 0])
    z_sum = np.bincount(inverse, weights=cross_coord[:, 1])
    centers = np.column_stack((side_sum / counts, z_sum / counts))

    if centers.shape[0] == 1:
        if int(counts[0]) < min_points:
            return []
        return [{"center": centers[0], "count": int(counts[0])}]

    pairs = np.asarray(list(cKDTree(centers).query_pairs(radius)), dtype=np.int64)
    if pairs.size == 0:
        labels = np.arange(centers.shape[0], dtype=np.int64)
        component_count = centers.shape[0]
    else:
        graph = coo_matrix(
            (
                np.ones(pairs.shape[0], dtype=np.uint8),
                (pairs[:, 0], pairs[:, 1]),
            ),
            shape=(centers.shape[0], centers.shape[0]),
        )
        component_count, labels = connected_components(
            graph, directed=False, return_labels=True
        )

    clusters = []
    for component_id in range(component_count):
        mask = labels == component_id
        point_count = int(counts[mask].sum())
        if point_count < min_points:
            continue
        center = np.average(centers[mask], axis=0, weights=counts[mask])
        clusters.append({"center": center, "count": point_count})
    return clusters


def track_cross_section_centers(bin_clusters, args):
    tracks = []
    active = []
    for bin_item in bin_clusters:
        centers = bin_item["clusters"]
        if not centers:
            continue
        used_tracks = set()
        for cluster in sorted(centers, key=lambda item: item["count"], reverse=True):
            best_track = None
            best_dist = np.inf
            center = cluster["center"]
            for track_id in active:
                if track_id in used_tracks:
                    continue
                track = tracks[track_id]
                if bin_item["bin"] - track["last_bin"] > 2:
                    continue
                dist = float(np.linalg.norm(center - track["last_center"]))
                if dist < best_dist:
                    best_dist = dist
                    best_track = track_id
            sample = {
                "along": float(bin_item["along"]),
                "side": float(center[0]),
                "z": float(center[1]),
                "count": int(cluster["count"]),
                "bin": int(bin_item["bin"]),
            }
            if best_track is not None and best_dist <= args.conductor_track_radius:
                track = tracks[best_track]
                track["samples"].append(sample)
                track["last_center"] = center
                track["last_bin"] = int(bin_item["bin"])
                used_tracks.add(best_track)
            else:
                tracks.append(
                    {
                        "samples": [sample],
                        "last_center": center,
                        "last_bin": int(bin_item["bin"]),
                    }
                )
                used_tracks.add(len(tracks) - 1)
        active = [
            i
            for i, track in enumerate(tracks)
            if bin_item["bin"] - track["last_bin"] <= 2
        ]
    return [
        track
        for track in tracks
        if len(track["samples"]) >= args.min_conductor_track_bins
    ]


def fit_track(track, args):
    samples = track["samples"]
    along = np.asarray([item["along"] for item in samples], dtype=np.float64)
    side = np.asarray([item["side"] for item in samples], dtype=np.float64)
    z = np.asarray([item["z"] for item in samples], dtype=np.float64)
    weights = np.sqrt(np.asarray([item["count"] for item in samples], dtype=np.float64))
    order = np.argsort(along)
    along, side, z, weights = along[order], side[order], z[order], weights[order]
    degree = int(min(max(args.conductor_poly_degree, 1), along.size - 1))
    side_coef = np.polyfit(along, side, degree, w=weights)
    z_coef = np.polyfit(along, z, degree, w=weights)
    return {
        "along_min": float(along.min()),
        "along_max": float(along.max()),
        "side_coef": side_coef,
        "z_coef": z_coef,
        "sample_count": int(along.size),
        "point_count": int(sum(item["count"] for item in samples)),
    }


def evaluate_fit(fit, along):
    side = np.polyval(fit["side_coef"], along)
    z = np.polyval(fit["z_coef"], along)
    return side, z


def sample_fit_world(fit, axes, origin, step, along_min=None, along_max=None):
    start = fit["along_min"] if along_min is None else max(float(along_min), fit["along_min"])
    end = fit["along_max"] if along_max is None else min(float(along_max), fit["along_max"])
    if end < start:
        return np.empty((0, 3), dtype=np.float64)
    count = max(int(np.ceil((end - start) / max(step, 1e-3))) + 1, 2)
    sample_along = np.linspace(start, end, count, dtype=np.float64)
    sample_side, sample_z = evaluate_fit(fit, sample_along)
    sample_local = np.column_stack((sample_along, sample_side, sample_z))
    return line_local_to_world(sample_local, axes, origin)


def append_colored_points(las, points, colors, point_class):
    if points.size == 0:
        return las, 0
    records = laspy.ScaleAwarePointRecord.zeros(points.shape[0], header=las.header)
    records.x = points[:, 0]
    records.y = points[:, 1]
    records.z = points[:, 2]
    if "classification" in set(las.point_format.dimension_names):
        records.classification = np.full(points.shape[0], point_class, dtype=np.uint8)
    if has_rgb(las):
        records.red = colors[:, 0]
        records.green = colors[:, 1]
        records.blue = colors[:, 2]
    combined = np.concatenate([las.points.array, records.array])
    las.points = laspy.ScaleAwarePointRecord(
        combined, las.header.point_format, las.header.scales, las.header.offsets
    )
    return las, int(points.shape[0])


def fit_conductors_for_spans(las, coord, span_boxes, args):
    if args.no_fit_conductors:
        return [], np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint16), 0

    conductor_reports = []
    fitted_points = []
    fitted_colors = []
    colored_original = 0
    global_fit_no = 1

    for span in [box for box in span_boxes if box["kind"] == "line_span"]:
        selected = span.get("_line_indices")
        if selected is None or selected.size == 0:
            continue
        local, axes, origin = world_to_line_local(
            coord[selected], span["_p0_xy"], span["_direction_xy"]
        )
        along = local[:, 0]
        side_z = local[:, 1:3]
        along_min = float(along.min())
        along_max = float(along.max())
        if along_max - along_min < args.conductor_bin_size:
            continue

        bin_id = np.floor((along - along_min) / args.conductor_bin_size).astype(np.int64)
        bin_clusters = []
        for item in np.unique(bin_id):
            mask = bin_id == item
            clusters = cross_section_clusters(
                side_z[mask],
                args.conductor_cluster_radius,
                args.min_conductor_points_per_bin,
            )
            if clusters:
                bin_clusters.append(
                    {
                        "bin": int(item),
                        "along": float(np.median(along[mask])),
                        "clusters": clusters,
                    }
                )
        tracks = track_cross_section_centers(bin_clusters, args)
        fits = [fit_track(track, args) for track in tracks]
        for fit in fits:
            mid = (fit["along_min"] + fit["along_max"]) / 2.0
            fit["sort_side"] = float(np.polyval(fit["side_coef"], mid))
            fit["sort_z"] = float(np.polyval(fit["z_coef"], mid))
        fits = sorted(
            fits,
            key=lambda fit: (fit["sort_side"], -fit["sort_z"]),
        )

        assigned = np.full(selected.shape[0], -1, dtype=np.int64)
        best_dist = np.full(selected.shape[0], np.inf, dtype=np.float64)
        for local_fit_id, fit in enumerate(fits):
            valid = (along >= fit["along_min"] - args.conductor_bin_size) & (
                along <= fit["along_max"] + args.conductor_bin_size
            )
            if not np.any(valid):
                continue
            pred_side, pred_z = evaluate_fit(fit, along[valid])
            dist = np.sqrt((side_z[valid, 0] - pred_side) ** 2 + (side_z[valid, 1] - pred_z) ** 2)
            valid_indices = np.flatnonzero(valid)
            improve = dist < best_dist[valid_indices]
            best_dist[valid_indices[improve]] = dist[improve]
            assigned[valid_indices[improve]] = local_fit_id

        for local_fit_id, fit in enumerate(fits):
            conductor_no = local_fit_id + 1
            color = CONDUCTOR_COLORS_16[(conductor_no - 1) % len(CONDUCTOR_COLORS_16)]
            assigned_mask = (assigned == local_fit_id) & (
                best_dist <= args.conductor_assign_radius
            )
            original_count = int(np.count_nonzero(assigned_mask))
            if original_count > 0 and has_rgb(las):
                original_indices = selected[assigned_mask]
                las.red[original_indices] = color[0]
                las.green[original_indices] = color[1]
                las.blue[original_indices] = color[2]
                colored_original += original_count

            sample_world = sample_fit_world(fit, axes, origin, args.conductor_fit_step)
            json_sample_world = sample_fit_world(
                fit,
                axes,
                origin,
                args.conductor_json_step,
                along_min=span.get("span_start_m"),
                along_max=span.get("span_end_m"),
            )
            if not args.no_append_conductor_fit:
                fitted_points.append(sample_world)
                fitted_colors.append(np.tile(color[None, :], (sample_world.shape[0], 1)))

            conductor_reports.append(
                {
                    "fit_no": int(global_fit_no),
                    "conductor_no": int(conductor_no),
                    "name": f"span_{span['span_id']}_conductor_{conductor_no}",
                    "display_name": f"导线{conductor_no}",
                    "span_id": int(span["span_id"]),
                    "left_tower_name": span.get("left_tower_name", ""),
                    "right_tower_name": span.get("right_tower_name", ""),
                    "track_bins": int(fit["sample_count"]),
                    "cluster_point_count": int(fit["point_count"]),
                    "colored_original_line_points": original_count,
                    "fitted_point_count": int(sample_world.shape[0]),
                    "color_rgb_16": color.astype(int).tolist(),
                    "along_min": float(fit["along_min"]),
                    "along_max": float(fit["along_max"]),
                    "json_along_min": (
                        None if json_sample_world.size == 0 else float(
                            max(span.get("span_start_m", fit["along_min"]), fit["along_min"])
                        )
                    ),
                    "json_along_max": (
                        None if json_sample_world.size == 0 else float(
                            min(span.get("span_end_m", fit["along_max"]), fit["along_max"])
                        )
                    ),
                    "span_start_m": float(span.get("span_start_m", fit["along_min"])),
                    "span_end_m": float(span.get("span_end_m", fit["along_max"])),
                    "sort_side": float(fit["sort_side"]),
                    "sort_z": float(fit["sort_z"]),
                    "side_poly_coef": fit["side_coef"].tolist(),
                    "z_poly_coef": fit["z_coef"].tolist(),
                    "polyline_xyz": json_sample_world.tolist(),
                }
            )
            global_fit_no += 1

    if fitted_points:
        return (
            conductor_reports,
            np.vstack(fitted_points),
            np.vstack(fitted_colors).astype(np.uint16, copy=False),
            colored_original,
        )
    return (
        conductor_reports,
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.uint16),
        colored_original,
    )


def append_boxes(las, boxes, edge_step):
    all_points = []
    all_colors = []
    for box in boxes:
        if box.get("box_mode") == "oriented":
            edges = sample_oriented_box_edges(
                box["center"], box["axes"], box["half_size"], edge_step
            )
        else:
            edges = sample_box_edges(box["bbox_min"], box["bbox_max"], edge_step)
        all_points.append(edges)
        color = TOWER_BOX_COLOR_16 if box["kind"] == "tower" else LINE_BOX_COLOR_16
        all_colors.append(np.tile(color[None, :], (edges.shape[0], 1)))
        box["edge_points"] = int(edges.shape[0])

    if not all_points:
        return las, 0

    edge_points = np.vstack(all_points)
    colors = np.vstack(all_colors).astype(np.uint16, copy=False)
    records = laspy.ScaleAwarePointRecord.zeros(edge_points.shape[0], header=las.header)
    records.x = edge_points[:, 0]
    records.y = edge_points[:, 1]
    records.z = edge_points[:, 2]
    if "classification" in set(las.point_format.dimension_names):
        records.classification = np.full(edge_points.shape[0], BOX_CLASS, dtype=np.uint8)
    if has_rgb(las):
        records.red = colors[:, 0]
        records.green = colors[:, 1]
        records.blue = colors[:, 2]

    combined = np.concatenate([las.points.array, records.array])
    las.points = laspy.ScaleAwarePointRecord(
        combined, las.header.point_format, las.header.scales, las.header.offsets
    )
    return las, int(edge_points.shape[0])


def json_ready_box(box):
    out = {}
    for key, value in box.items():
        if key.startswith("_"):
            continue
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def json_ready_component(comp):
    return {
        "id": int(comp["id"]),
        "point_count": int(comp["point_count"]),
        "bbox_min": comp["bbox_min"].tolist(),
        "bbox_max": comp["bbox_max"].tolist(),
        "size": comp["size"].tolist(),
        "center": comp["center"].tolist(),
        "nearby_line_points": int(comp.get("nearby_line_points", 0)),
        "height_above_ground": (
            None
            if comp.get("height_above_ground") is None
            else float(comp["height_above_ground"])
        ),
        "keep": bool(comp["keep"]),
        "remove_reason": comp["remove_reason"],
    }


def json_ready_tower(tower):
    return {
        "id": int(tower["id"]),
        "tower_no": int(tower.get("tower_no", 0)),
        "name": tower.get("name", ""),
        "display_name": tower.get("display_name", ""),
        "component_ids": [int(item) for item in tower["component_ids"]],
        "point_count": int(tower["point_count"]),
        "bbox_min": tower["bbox_min"].tolist(),
        "bbox_max": tower["bbox_max"].tolist(),
        "size": tower["size"].tolist(),
        "center": tower["center"].tolist(),
    }


def default_standard_json_paths(report_path):
    stem = report_path.stem
    if stem.endswith("_report"):
        stem = stem[: -len("_report")]
    return (
        report_path.with_name(stem + "_tower_boxes.json"),
        report_path.with_name(stem + "_boxes.json"),
    )


def drawable_box_item(box, id_fields):
    corners, edges = box_geometry_from_box(box)
    item = dict(id_fields)
    item["corners_xyz"] = corners.tolist()
    item["edges"] = edges
    return item


def make_standard_tower_boxes(tower_box_list, origin):
    return [
        box_to_obb_record(
            box,
            class_name="tower",
            instance_name=f"杆塔{int(box.get('tower_no', 0))}",
            origin=origin,
        )
        for box in sorted(tower_box_list, key=lambda item: int(item.get("tower_no", 0)))
    ]


def make_standard_line_boxes(span_boxes, origin):
    return [
        box_to_obb_record(
            span,
            class_name="line_span",
            instance_name="档距{}_{}-{}".format(
                index,
                span.get("left_tower_name", ""),
                span.get("right_tower_name", ""),
            ),
            origin=origin,
        )
        for index, span in enumerate(
            sorted(span_boxes, key=lambda item: int(item["span_id"])), start=1
        )
    ]


def make_conductor_render_report(conductor_reports, origin):
    origin = np.asarray(origin, dtype=np.float64)
    span_map = {}
    for conductor in conductor_reports:
        span_id = int(conductor["span_id"])
        span_map.setdefault(
            span_id,
            {
                "span_id": span_id,
                "span_name": "{}-{}".format(
                    conductor.get("left_tower_name", ""),
                    conductor.get("right_tower_name", ""),
                ),
                "lines": [],
            },
        )
        points = np.asarray(conductor.get("polyline_xyz", []), dtype=np.float64)
        if points.size == 0:
            relative_points = []
        else:
            relative_points = (points - origin[None, :]).tolist()
        span_map[span_id]["lines"].append(
            {
                "line_no": int(conductor["conductor_no"]),
                "line_name": f"线路{int(conductor['conductor_no'])}",
                "color_rgb_16": conductor.get("color_rgb_16", []),
                "points": relative_points,
            }
        )

    spans = []
    for span_id in sorted(span_map):
        span = span_map[span_id]
        span["lines"] = sorted(span["lines"], key=lambda item: item["line_no"])
        spans.append(span)
    return {
        "lat_lng_alt": origin.tolist(),
        "spans": spans,
    }


def rotation_matrix_to_quaternion(matrix):
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return (quat / norm).tolist()


def box_to_obb_record(box, class_name, instance_name, origin):
    if box.get("box_mode") == "oriented":
        center = box["center"].astype(np.float64, copy=False)
        extent = (box["half_size"] * 2.0).astype(np.float64, copy=False)
        rotation = rotation_matrix_to_quaternion(box["axes"].T)
    else:
        center = (box["bbox_min"] + box["bbox_max"]) / 2.0
        extent = box["bbox_max"] - box["bbox_min"]
        rotation = [0.0, 0.0, 0.0, 1.0]
    origin = np.asarray(origin, dtype=np.float64)
    relative_center = center - origin
    relative_center_list = relative_center.tolist()
    origin_list = origin.tolist()
    extent_list = extent.tolist()
    return {
        "class_name": class_name,
        "instance_name": instance_name,
        "obb": relative_center_list + extent_list + rotation,
        "obb_global": {
            "extent": extent_list,
            "lat_lng_alt": origin_list,
            "rotation": rotation,
        },
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def resolve_obb_origin(las, args):
    if args.obb_origin == "custom":
        if args.obb_origin_xyz is None:
            raise ValueError("--obb-origin custom requires --obb-origin-xyz X Y Z")
        return np.asarray(args.obb_origin_xyz, dtype=np.float64)
    if args.obb_origin == "las-min":
        return np.asarray(las.header.mins, dtype=np.float64)
    if args.obb_origin == "zero":
        return np.zeros(3, dtype=np.float64)
    return np.asarray(las.header.offsets, dtype=np.float64)


def main():
    args = parse_args()
    global laspy
    import laspy as laspy_module

    laspy = laspy_module
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = (
        Path(args.report)
        if args.report
        else output_path.with_name(output_path.stem + "_tower_line_report.json")
    )
    default_tower_box_report, default_line_box_report = default_standard_json_paths(
        report_path
    )
    tower_box_report_path = (
        Path(args.tower_box_report) if args.tower_box_report else None
    )
    combined_box_report_path = (
        Path(args.combined_box_report)
        if args.combined_box_report
        else Path(args.line_box_report)
        if args.line_box_report
        else default_line_box_report
    )
    conductor_report_path = (
        Path(args.conductor_report)
        if args.conductor_report
        else report_path.with_name(
            report_path.stem[: -len("_report")] + "_conductors.json"
            if report_path.stem.endswith("_report")
            else report_path.stem + "_conductors.json"
        )
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists, use --overwrite: {output_path}")

    start = time.perf_counter()
    print(f"Reading {input_path}", flush=True)
    las = laspy.read(input_path)
    if "classification" not in set(las.point_format.dimension_names):
        raise ValueError("Input LAS has no classification dimension.")

    coord = coords_from_las(las)
    obb_origin = resolve_obb_origin(las, args)
    cls = np.asarray(las.classification, dtype=np.uint8).copy()
    original_counts = {
        str(i): int(v) for i, v in enumerate(np.bincount(cls, minlength=32)) if v
    }
    print(f"Loaded {coord.shape[0]} points; class counts: {original_counts}", flush=True)

    recolor_by_class(las, cls, args)

    t0 = time.perf_counter()
    components, tower_indices, point_comp = cluster_towers(coord, cls, args)
    mark_components_to_keep(components, coord, cls, args)
    seed_towers = merge_physical_towers(components, args)
    recovered_base_components = recover_tower_base_components(
        components, seed_towers, args
    )
    physical_towers = merge_physical_towers(components, args)
    physical_towers = assign_tower_names(physical_towers, args)
    removed_indices = apply_tower_cleanup(
        las, cls, tower_indices, point_comp, components, args.background_class
    )
    print(
        "Tower cleanup: components={} kept={} removed_points={} in {:.2f}s".format(
            len(components),
            sum(1 for item in components if item["keep"]),
            removed_indices.size,
            time.perf_counter() - t0,
        ),
        flush=True,
    )
    if recovered_base_components:
        print(
            f"Recovered {recovered_base_components} low tower-base components",
            flush=True,
        )

    kept_components = [comp for comp in components if comp["keep"]]
    boxes = tower_boxes(
        physical_towers, coord, tower_indices, point_comp, args.tower_box_margin
    )
    tower_box_by_id = {int(box["tower_id"]): box for box in boxes if box["kind"] == "tower"}
    span_boxes = line_span_boxes(coord, cls, physical_towers, tower_box_by_id, args)
    boxes.extend(span_boxes)
    print(
        "Boxes: kept_components={}, physical_towers={}, line_span={}".format(
            len(kept_components), len(physical_towers), len(span_boxes)
        ),
        flush=True,
    )

    conductor_reports, conductor_points, conductor_colors, colored_line_points = (
        fit_conductors_for_spans(las, coord, span_boxes, args)
    )
    appended_conductor_points = 0
    if conductor_points.size:
        las, appended_conductor_points = append_colored_points(
            las, conductor_points, conductor_colors, FITTED_CONDUCTOR_CLASS
        )
    if conductor_reports:
        print(
            "Conductors: fitted={} colored_original_points={} appended_points={}".format(
                len(conductor_reports),
                colored_line_points,
                appended_conductor_points,
            ),
            flush=True,
        )

    appended_points = 0
    if not args.no_append_box_points:
        las, appended_points = append_boxes(las, boxes, args.edge_step)
        print(f"Appended {appended_points} box-edge points", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(output_path)
    elapsed = time.perf_counter() - start

    final_counts = {
        str(i): int(v) for i, v in enumerate(np.bincount(cls, minlength=32)) if v
    }
    debug_conductor_reports = []
    for conductor in conductor_reports:
        item = dict(conductor)
        item.pop("polyline_xyz", None)
        debug_conductor_reports.append(item)
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "original_class_counts": original_counts,
        "final_original_point_class_counts": final_counts,
        "tower_components_total": len(components),
        "tower_components_kept": len(kept_components),
        "tower_components_removed": len(components) - len(kept_components),
        "physical_towers": [json_ready_tower(tower) for tower in physical_towers],
        "physical_tower_count": len(physical_towers),
        "recovered_tower_base_components": int(recovered_base_components),
        "removed_tower_points": int(removed_indices.size),
        "conductors": debug_conductor_reports,
        "conductor_count": len(conductor_reports),
        "colored_original_line_points": int(colored_line_points),
        "appended_conductor_fit_points": int(appended_conductor_points),
        "tower_components": [json_ready_component(comp) for comp in components],
        "boxes": [json_ready_box(box) for box in boxes],
        "appended_box_edge_points": appended_points,
        "obb_origin_mode": args.obb_origin,
        "obb_origin_xyz": obb_origin.tolist(),
        "elapsed_sec": round(elapsed, 3),
        "parameters": vars(args),
    }
    tower_box_report = make_standard_tower_boxes(
        [box for box in boxes if box["kind"] == "tower"], obb_origin
    )
    line_box_report = make_standard_line_boxes(span_boxes, obb_origin)
    combined_box_report = tower_box_report + line_box_report
    conductor_render_report = make_conductor_render_report(
        conductor_reports, obb_origin
    )

    write_json(report_path, report)
    write_json(combined_box_report_path, combined_box_report)
    write_json(conductor_report_path, conductor_render_report)
    if tower_box_report_path is not None:
        write_json(tower_box_report_path, tower_box_report)

    print(f"Wrote LAS: {output_path}", flush=True)
    print(f"Wrote report: {report_path}", flush=True)
    if tower_box_report_path is not None:
        print(f"Wrote tower boxes: {tower_box_report_path}", flush=True)
    print(f"Wrote combined boxes: {combined_box_report_path}", flush=True)
    print(f"Wrote conductors: {conductor_report_path}", flush=True)
    print(f"Finished in {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()

""" 
/opt/conda/envs/pointcept/bin/python tools/postprocess_tower_line_boxes.py \
  --input /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb_plain_v1.las \
  --output /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb_tower_line_boxes_v3.las \
  --report /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb_tower_line_boxes_v3_report.json \
  --tower-voxel-size 0.50 \
  --min-tower-points 50000 \
  --min-tower-height 30.0 \
  --min-tower-height-above-ground 8.0 \
  --require-line-near-tower \
  --tower-line-radius 18.0 \
  --min-line-points-near-tower 100 \
  --merge-tower-xy-radius 18.0 \
  --line-corridor-width 20.0 \
  --span-end-margin 12.0 \
  --min-span-line-points 500 \
  --tower-box-margin 1.0 \
  --line-box-margin 1.0 \
  --edge-step 0.30 \
  --overwrite
"""
