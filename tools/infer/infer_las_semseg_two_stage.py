"""LAS 输电线路二阶段推理脚本。

流程：
1. Stage-1：对整幅 LAS 做粗结构分割，只保留背景、杆塔、导线三类结构；
2. Stage-1 中预测出的绝缘子先并入杆塔，用于生成整座杆塔 ROI；
3. Stage-2：只在杆塔 ROI 内使用同一个模型做精细重推，重点恢复绝缘子；
4. 融合：默认只把 Stage-2 高置信度绝缘子点覆盖回最终结果。

这个脚本不需要重新训练模型。模型仍然输出四类，但第一阶段只承担粗结构定位，
真正的绝缘子结果由第二阶段局部精推决定。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

SCRIPT_PATH = Path(__file__).resolve()
TOOL_DIR = SCRIPT_PATH.parent
ROOT_DIR = next(
    (path for path in SCRIPT_PATH.parents if (path / "pointcept").is_dir()),
    SCRIPT_PATH.parents[2],
)
for path in (TOOL_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import infer_las_semseg as base  # noqa: E402
import postprocess_tower_line_clean as tower_clean  # noqa: E402
from pointcept.utils.config import Config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run two-stage LAS semantic segmentation for transmission lines."
    )
    parser.add_argument("--input", required=True, help="输入 .las 文件或目录。")
    parser.add_argument("--output", required=True, help="输出 .las 文件或目录。")
    parser.add_argument(
        "--report",
        default=None,
        help="可选：单文件输入时的 JSON 报告路径；目录输入时会忽略，自动按输出文件名生成。",
    )
    parser.add_argument("--config-file", default=base.DEFAULT_CONFIG, help="Pointcept 配置。")
    parser.add_argument("--weight", default=base.DEFAULT_WEIGHT, help="模型权重。")
    parser.add_argument(
        "--las-backend",
        choices=("auto", "laspy", "fallback"),
        default="auto",
        help="LAS 读写后端。",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备，例如 cuda、cuda:0 或 cpu。",
    )
    parser.add_argument(
        "--disable-flash",
        action="store_true",
        help="关闭 flash attention，Windows 或 flash-attn 不可用时使用。",
    )
    parser.add_argument(
        "--default-color",
        type=int,
        nargs=3,
        default=(255, 255, 255),
        metavar=("R", "G", "B"),
        help="输入 LAS 没有 RGB 字段时使用的默认颜色。",
    )
    parser.add_argument("--no-colorize", action="store_true", help="不重写输出 LAS RGB。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出。")

    # Stage-1 全图粗分割参数。第一阶段只负责找背景、杆塔、导线等大结构。
    parser.add_argument("--stage1-tile-size", type=float, default=40.0)
    parser.add_argument("--stage1-tile-stride", type=float, default=40.0)
    parser.add_argument(
        "--stage1-merge-mode",
        choices=("plain", "halo", "overlap"),
        default="plain",
    )
    parser.add_argument("--stage1-context-margin", type=float, default=10.0)
    parser.add_argument("--stage1-pre-voxel-size", type=float, default=0.05)
    parser.add_argument("--stage1-point-max", type=int, default=100000)
    parser.add_argument("--stage1-grid-size", type=float, default=None)
    parser.add_argument("--stage1-fragment-batch-size", type=int, default=4)
    parser.add_argument("--stage1-min-tile-points", type=int, default=1024)
    parser.add_argument(
        "--save-stage1-las",
        default=None,
        help=(
            "可选：保存 Stage-1 粗结构 LAS，便于对比调试。默认会保存三类粗结构结果，"
            "也就是绝缘子已并入杆塔。"
        ),
    )
    parser.add_argument(
        "--stage1-keep-insulator-class",
        action="store_true",
        help=(
            "默认 Stage-1 会把绝缘子类并入杆塔类，只保留背景/杆塔/导线三类粗结构。"
            "加上该参数后保留 Stage-1 的绝缘子预测。"
        ),
    )

    # 第一阶段杆塔清理：在生成二阶段 ROI 前剔除明显的背景/导线杆塔误分。
    parser.add_argument(
        "--disable-stage1-tower-cleanup",
        action="store_true",
        help="关闭第一阶段杆塔误分清理。默认开启。",
    )
    parser.add_argument(
        "--stage1-tower-cleanup-voxel-size",
        type=float,
        default=0.5,
        help="第一阶段杆塔连通域聚类体素大小，单位米。",
    )
    parser.add_argument(
        "--stage1-tower-cleanup-connectivity",
        type=int,
        choices=(6, 18, 26),
        default=26,
        help="第一阶段杆塔体素连通性。",
    )
    parser.add_argument(
        "--stage1-tower-cleanup-min-points",
        type=int,
        default=512,
        help="保留一个第一阶段杆塔组件所需的最少预测杆塔点数。",
    )
    parser.add_argument(
        "--stage1-tower-cleanup-min-height",
        type=float,
        default=6.0,
        help="保留一个第一阶段杆塔组件所需的最小高度，单位米。",
    )
    parser.add_argument(
        "--stage1-tower-cleanup-min-xy-size",
        type=float,
        default=1.0,
        help="保留组件所需的最小 XY 尺寸，过小的孤立噪声会被剔除。",
    )
    parser.add_argument(
        "--stage1-tower-cleanup-min-vertical-ratio",
        type=float,
        default=1.0,
        help="杆塔高度与最大 XY 尺寸的最小比值，用于剔除横向延伸的导线误分。",
    )
    parser.add_argument(
        "--stage1-tower-cleanup-contact-radius",
        type=float,
        default=2.5,
        help="杆塔上部点到导线点的最大三维接触距离，单位米。",
    )
    parser.add_argument(
        "--stage1-tower-cleanup-upper-height-ratio",
        type=float,
        default=0.4,
        help="仅在杆塔顶部该高度比例以上检查导线接触。",
    )
    parser.add_argument(
        "--stage1-tower-cleanup-min-contact-points",
        type=int,
        default=20,
        help="保留杆塔所需的最少上部杆塔-导线接触点数。",
    )
    parser.add_argument(
        "--stage1-tower-cleanup-merge-xy-radius",
        type=float,
        default=12.0,
        help="第一阶段清理时，把同一座物理塔的碎组件合并到塔身 seed 的 XY 半径，单位米。",
    )

    # 候选区域生成参数。候选区来自“导线点靠近杆塔点”的位置。
    parser.add_argument("--background-class", type=int, default=0)
    parser.add_argument("--tower-class", type=int, default=1)
    parser.add_argument("--line-class", type=int, default=2)
    parser.add_argument("--insulator-class", type=int, default=3)
    parser.add_argument(
        "--roi-mode",
        choices=("connection", "tower"),
        default="tower",
        help=(
            "二阶段 ROI 生成方式。connection 根据导线靠近杆塔的位置生成小 ROI；"
            "tower 根据 Stage-1 粗结构杆塔点聚类生成整座杆塔 ROI。默认使用 tower，"
            "对应“第二阶段在杆塔里面找绝缘子”。"
        ),
    )
    parser.add_argument(
        "--line-tower-radius",
        type=float,
        default=4.0,
        help="导线点到杆塔点的最近距离小于该值时，认为是连接候选。",
    )
    parser.add_argument(
        "--include-stage1-insulator-candidates",
        action="store_true",
        help="把 Stage-1 已预测为绝缘子的点也作为候选中心。",
    )
    parser.add_argument(
        "--candidate-voxel-size",
        type=float,
        default=5.0,
        help="候选中心体素合并大小，越大 ROI 越少。",
    )
    parser.add_argument("--roi-xy-radius", type=float, default=8.0, help="ROI 的 XY 半径。")
    parser.add_argument("--roi-z-radius", type=float, default=5.0, help="ROI 的 Z 半径。")
    parser.add_argument(
        "--tower-roi-voxel-size",
        type=float,
        default=1.0,
        help="roi-mode=tower 时，杆塔点体素连通聚类大小，单位米。",
    )
    parser.add_argument(
        "--tower-roi-min-points",
        type=int,
        default=512,
        help="roi-mode=tower 时，保留一个杆塔 ROI 所需的最少杆塔点数。",
    )
    parser.add_argument(
        "--tower-roi-min-height",
        type=float,
        default=4.0,
        help="roi-mode=tower 时，保留一个杆塔 ROI 所需的最小高度，单位米。",
    )
    parser.add_argument(
        "--tower-roi-xy-margin",
        type=float,
        default=8.0,
        help="roi-mode=tower 时，杆塔包围盒 XY 方向向外扩张距离，单位米。",
    )
    parser.add_argument(
        "--tower-roi-z-margin",
        type=float,
        default=6.0,
        help="roi-mode=tower 时，杆塔包围盒 Z 方向上下扩张距离，单位米。",
    )
    parser.add_argument("--min-roi-points", type=int, default=256)
    parser.add_argument(
        "--max-rois",
        type=int,
        default=0,
        help="最多处理多少个 ROI，0 表示不限制。调试时可设小一些。",
    )

    # Stage-2 局部高精度重推参数。默认不做额外 pre-voxel，尽量保护绝缘子。
    parser.add_argument("--stage2-point-max", type=int, default=200000)
    parser.add_argument("--stage2-grid-size", type=float, default=None)
    parser.add_argument("--stage2-pre-voxel-size", type=float, default=0.0)
    parser.add_argument("--stage2-fragment-batch-size", type=int, default=1)

    # 融合策略。默认保守：只把 Stage-2 确认的绝缘子覆盖回来。
    parser.add_argument(
        "--fusion-mode",
        choices=("insulator-only", "replace-roi"),
        default="insulator-only",
        help="insulator-only 只覆盖绝缘子；replace-roi 用 Stage-2 覆盖整个 ROI。",
    )
    parser.add_argument(
        "--insulator-score-threshold",
        type=float,
        default=0.5,
        help="Stage-2 绝缘子概率超过该阈值才融合为绝缘子。",
    )
    parser.add_argument(
        "--require-insulator-argmax",
        action="store_true",
        help="要求 Stage-2 argmax 也是绝缘子才融合。默认只看绝缘子概率。",
    )
    parser.add_argument(
        "--replace-score-threshold",
        type=float,
        default=0.0,
        help="fusion-mode=replace-roi 时，Stage-2 最大概率超过该阈值才覆盖。",
    )
    parser.add_argument(
        "--recover-insulator-by-structure",
        action="store_true",
        help="开启结构先验恢复：把杆塔上部、贴近导线的小型杆塔类连通块恢复为绝缘子。",
    )
    parser.add_argument(
        "--recover-insulator-line-radius",
        type=float,
        default=1.2,
        help="结构恢复时，候选点到导线的最大距离，单位米。",
    )
    parser.add_argument(
        "--recover-insulator-upper-height-ratio",
        type=float,
        default=0.35,
        help="结构恢复时，仅检查 ROI 高度比例以上的点，避免把杆塔底部误恢复为绝缘子。",
    )
    parser.add_argument(
        "--recover-insulator-voxel-size",
        type=float,
        default=0.25,
        help="结构恢复候选点连通域体素大小，单位米。",
    )
    parser.add_argument(
        "--recover-insulator-min-points",
        type=int,
        default=10,
        help="结构恢复时，保留一个绝缘子候选连通块所需的最少点数。",
    )
    parser.add_argument(
        "--recover-insulator-max-extent",
        type=float,
        default=3.0,
        help="结构恢复时，候选连通块最大包围盒尺寸；过长通常是横担或塔身，不作为绝缘子。",
    )
    parser.add_argument(
        "--recover-insulator-min-z-extent",
        type=float,
        default=0.15,
        help="结构恢复时，候选连通块最小 Z 向厚度；太薄的水平板/横担会被剔除。",
    )
    parser.add_argument(
        "--recover-insulator-min-core-distance",
        type=float,
        default=0.8,
        help="结构恢复时，候选连通块中心到杆塔主体 XY 中心的最小距离，避免塔帽/塔身被改成绝缘子。",
    )
    parser.add_argument(
        "--recover-insulator-tower-neighbor-radius",
        type=float,
        default=0.4,
        help="结构恢复时，检查候选点是否过度贴近非候选杆塔点的半径，单位米。",
    )
    parser.add_argument(
        "--recover-insulator-max-tower-neighbor-ratio",
        type=float,
        default=0.75,
        help="结构恢复时，候选连通块中过度贴近杆塔主体的点占比上限。",
    )
    parser.add_argument(
        "--recover-insulator-min-stage2-score",
        type=float,
        default=0.03,
        help="结构恢复时，候选连通块内 Stage-2 最大绝缘子概率下限；0 表示完全不看模型弱证据。",
    )
    parser.add_argument(
        "--recover-insulator-min-line-contact-points",
        type=int,
        default=3,
        help="结构恢复时，候选连通块内至少需要多少个贴近导线的点。",
    )
    parser.add_argument(
        "--recover-insulator-min-linearity",
        type=float,
        default=0.65,
        help="结构恢复时，候选连通块的最小直线性；越大越要求像直线，弯曲跳线会被剔除。",
    )
    parser.add_argument(
        "--recover-insulator-max-bend-ratio",
        type=float,
        default=0.28,
        help="结构恢复时，候选点到主轴的均方根距离/主轴长度上限，用于剔除弯曲结构。",
    )
    parser.add_argument(
        "--recover-insulator-end-percentile",
        type=float,
        default=20.0,
        help="结构恢复时，沿主轴两端各取多少百分比点检查端点接触。",
    )
    parser.add_argument(
        "--recover-insulator-line-end-radius",
        type=float,
        default=1.2,
        help="结构恢复时，绝缘子导线端到导线点的最大距离，单位米。",
    )
    parser.add_argument(
        "--recover-insulator-tower-end-radius",
        type=float,
        default=0.8,
        help="结构恢复时，绝缘子杆塔端到非候选杆塔点的最大距离，单位米。",
    )
    return parser.parse_args()


def class_summary(pred, cfg):
    counts = np.bincount(pred, minlength=cfg.data.num_classes)
    return {
        str(idx): {
            "name": cfg.data.names[idx] if idx < len(cfg.data.names) else str(idx),
            "count": int(counts[idx]),
        }
        for idx in range(cfg.data.num_classes)
    }


def make_stage1_structure_prediction(stage1_pred, args):
    """把 Stage-1 结果转换成粗结构结果。

    当前权重仍然是四分类模型，但第一阶段的任务只需要背景、杆塔、导线三类大结构。
    因此默认把 Stage-1 预测出的绝缘子点并入杆塔类：

    - 背景 class=0 保持不变；
    - 杆塔 class=1 保持不变；
    - 导线 class=2 保持不变；
    - 绝缘子 class=3 先当作杆塔的一部分，交给 Stage-2 在杆塔 ROI 内重新精分割。
    """
    structure_pred = stage1_pred.copy()
    insulator_mask = structure_pred == int(args.insulator_class)
    merged_points = int(np.count_nonzero(insulator_mask))
    if (not args.stage1_keep_insulator_class) and merged_points:
        structure_pred[insulator_mask] = int(args.tower_class)
    return structure_pred, {
        "enabled": not bool(args.stage1_keep_insulator_class),
        "merged_insulator_to_tower_points": 0
        if args.stage1_keep_insulator_class
        else merged_points,
        "stage1_keep_insulator_class": bool(args.stage1_keep_insulator_class),
    }


def compact_candidate_centers(points, voxel_size):
    """把大量候选点合并成少量 ROI 中心。"""
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    if voxel_size <= 0:
        return points.astype(np.float64, copy=False)

    origin = points.min(axis=0)
    voxel = np.floor((points - origin[None, :]) / float(voxel_size)).astype(np.int64)
    unique_voxel, inverse = np.unique(voxel, axis=0, return_inverse=True)
    centers = np.empty((unique_voxel.shape[0], 3), dtype=np.float64)
    for idx in range(unique_voxel.shape[0]):
        centers[idx] = np.median(points[inverse == idx], axis=0)
    # 从左到右、从低到高排序，保证日志稳定。
    order = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))
    return centers[order]


def build_connection_roi_centers(coord, stage1_pred, args):
    """根据 Stage-1 的杆塔点和导线点自动生成二阶段候选中心。

    逻辑很朴素但有效：导线点如果离杆塔点很近，就说明附近可能存在绝缘子/金具/横担连接结构。
    """
    tower_mask = stage1_pred == int(args.tower_class)
    line_mask = stage1_pred == int(args.line_class)
    tower_coord = coord[tower_mask]
    line_coord = coord[line_mask]

    candidate_parts = []
    near_line_count = 0
    if tower_coord.shape[0] and line_coord.shape[0]:
        tower_tree = cKDTree(tower_coord[:, :3])
        distance, _ = tower_tree.query(line_coord[:, :3], k=1, workers=-1)
        near_line = line_coord[distance <= float(args.line_tower_radius)]
        near_line_count = int(near_line.shape[0])
        if near_line.shape[0]:
            candidate_parts.append(near_line)

    stage1_insulator_count = 0
    if args.include_stage1_insulator_candidates:
        insulator_mask = stage1_pred == int(args.insulator_class)
        insulator_coord = coord[insulator_mask]
        stage1_insulator_count = int(insulator_coord.shape[0])
        if insulator_coord.shape[0]:
            candidate_parts.append(insulator_coord)

    if candidate_parts:
        candidate_points = np.concatenate(candidate_parts, axis=0)
    else:
        candidate_points = np.empty((0, 3), dtype=np.float64)

    centers = compact_candidate_centers(candidate_points, args.candidate_voxel_size)
    if args.max_rois and centers.shape[0] > int(args.max_rois):
        centers = centers[: int(args.max_rois)]
    return centers, {
        "tower_points": int(tower_coord.shape[0]),
        "line_points": int(line_coord.shape[0]),
        "near_line_points": int(near_line_count),
        "stage1_insulator_candidate_points": int(stage1_insulator_count),
        "roi_centers": int(centers.shape[0]),
    }


def connection_centers_to_rois(centers, args):
    """把连接点中心转换成统一 ROI 记录。"""
    rois = []
    xy_radius = float(args.roi_xy_radius)
    z_radius = float(args.roi_z_radius)
    for roi_id, center in enumerate(centers, start=1):
        center = np.asarray(center, dtype=np.float64)
        min_xyz = center - np.array([xy_radius, xy_radius, z_radius], dtype=np.float64)
        max_xyz = center + np.array([xy_radius, xy_radius, z_radius], dtype=np.float64)
        rois.append(
            {
                "roi_id": int(roi_id),
                "roi_type": "connection",
                "center": center,
                "min_xyz": min_xyz,
                "max_xyz": max_xyz,
            }
        )
    return rois


def voxel_component_labels(points, voxel_size):
    """对点云做简单体素连通域标记，用于把 Stage-1 杆塔点分成多个杆塔候选。"""
    if points.shape[0] == 0:
        return np.empty((0,), dtype=np.int64), 0
    if voxel_size <= 0:
        return np.zeros(points.shape[0], dtype=np.int64), 1

    origin = points.min(axis=0)
    voxel = np.floor((points - origin[None, :]) / float(voxel_size)).astype(np.int64)
    unique_voxel, inverse = np.unique(voxel, axis=0, return_inverse=True)
    voxel_to_index = {tuple(v.tolist()): idx for idx, v in enumerate(unique_voxel)}
    offsets = np.array(
        [
            [dx, dy, dz]
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ],
        dtype=np.int64,
    )
    voxel_labels = np.full(unique_voxel.shape[0], -1, dtype=np.int64)
    component_id = 0
    for start in range(unique_voxel.shape[0]):
        if voxel_labels[start] >= 0:
            continue
        stack = [start]
        voxel_labels[start] = component_id
        while stack:
            current = stack.pop()
            base_voxel = unique_voxel[current]
            for offset in offsets:
                neighbor = voxel_to_index.get(tuple((base_voxel + offset).tolist()))
                if neighbor is not None and voxel_labels[neighbor] < 0:
                    voxel_labels[neighbor] = component_id
                    stack.append(neighbor)
        component_id += 1
    return voxel_labels[inverse], component_id


def build_tower_rois(coord, stage1_pred, args):
    """根据 Stage-1 杆塔点生成整座杆塔 ROI。

    这个模式对应方案 A：第一阶段负责找到杆塔大结构，第二阶段在整座杆塔
    包围区域内重新四分类，以提高绝缘子、小金具等小目标的召回。
    """
    tower_indices = np.flatnonzero(stage1_pred == int(args.tower_class))
    tower_coord = coord[tower_indices]
    if tower_coord.shape[0] == 0:
        return [], {
            "roi_mode": "tower",
            "tower_points": 0,
            "tower_components": 0,
            "kept_tower_rois": 0,
            "removed_small_components": 0,
            "removed_low_components": 0,
            "roi_centers": 0,
        }

    labels, component_count = voxel_component_labels(
        tower_coord, float(args.tower_roi_voxel_size)
    )
    rois = []
    removed_small = 0
    removed_low = 0
    for component_id in range(component_count):
        mask = labels == component_id
        points = tower_coord[mask]
        point_count = int(points.shape[0])
        if point_count < int(args.tower_roi_min_points):
            removed_small += 1
            continue
        min_xyz = points.min(axis=0).astype(np.float64)
        max_xyz = points.max(axis=0).astype(np.float64)
        height = float(max_xyz[2] - min_xyz[2])
        if height < float(args.tower_roi_min_height):
            removed_low += 1
            continue

        roi_min = min_xyz.copy()
        roi_max = max_xyz.copy()
        roi_min[:2] -= float(args.tower_roi_xy_margin)
        roi_max[:2] += float(args.tower_roi_xy_margin)
        roi_min[2] -= float(args.tower_roi_z_margin)
        roi_max[2] += float(args.tower_roi_z_margin)
        center = (roi_min + roi_max) / 2.0
        rois.append(
            {
                "roi_id": len(rois) + 1,
                "roi_type": "tower",
                "component_id": int(component_id),
                "center": center,
                "min_xyz": roi_min,
                "max_xyz": roi_max,
                "tower_point_count": point_count,
                "tower_height": height,
            }
        )

    rois.sort(key=lambda item: (float(item["center"][0]), float(item["center"][1]), float(item["center"][2])))
    for roi_id, roi in enumerate(rois, start=1):
        roi["roi_id"] = int(roi_id)
    if args.max_rois and len(rois) > int(args.max_rois):
        rois = rois[: int(args.max_rois)]

    return rois, {
        "roi_mode": "tower",
        "tower_points": int(tower_coord.shape[0]),
        "tower_components": int(component_count),
        "kept_tower_rois": int(len(rois)),
        "removed_small_components": int(removed_small),
        "removed_low_components": int(removed_low),
        "roi_centers": int(len(rois)),
    }


def build_stage2_rois(coord, stage1_pred, args):
    """根据参数选择二阶段 ROI 生成方式。"""
    if args.roi_mode == "tower":
        return build_tower_rois(coord, stage1_pred, args)
    centers, summary = build_connection_roi_centers(coord, stage1_pred, args)
    summary["roi_mode"] = "connection"
    return connection_centers_to_rois(centers, args), summary


def points_in_roi(coord, roi):
    min_xyz = roi["min_xyz"]
    max_xyz = roi["max_xyz"]
    return (
        (coord[:, 0] >= min_xyz[0])
        & (coord[:, 0] <= max_xyz[0])
        & (coord[:, 1] >= min_xyz[1])
        & (coord[:, 1] <= max_xyz[1])
        & (coord[:, 2] >= min_xyz[2])
        & (coord[:, 2] <= max_xyz[2])
    )


def recover_insulator_mask_by_structure(
    roi_coord, stage1_label, stage2_label, stage2_ins_score, args
):
    """根据输电线路结构关系恢复被误分成杆塔的绝缘子点。

    这个逻辑只在 ROI 内工作，目标是补救“绝缘子点被模型判成杆塔”的情况：
    1）候选点必须是杆塔类，避免把背景、导线大面积改成绝缘子；
    2）候选点必须位于 ROI 上部，避开塔身底部和地物；
    3）候选点必须贴近导线；
    4）候选点需要形成小型连通块，过长、过薄、太贴塔身主体的连通块会被剔除；
    5）候选连通块必须近似直线，弯曲跳线不会被恢复；
    6）候选连通块两端必须分别接触杆塔和导线；
    7）候选连通块需要具备一点 Stage-2 绝缘子弱证据，降低横担/塔帽误恢复。
    """
    recover_mask = np.zeros(roi_coord.shape[0], dtype=bool)
    if not args.recover_insulator_by_structure or roi_coord.shape[0] == 0:
        return recover_mask, {
            "enabled": bool(args.recover_insulator_by_structure),
            "candidate_points": 0,
            "components": 0,
            "kept_components": 0,
            "recovered_points": 0,
        }

    line_mask = (
        (stage1_label == int(args.line_class))
        | (stage2_label == int(args.line_class))
    )
    tower_mask = (
        (stage1_label == int(args.tower_class))
        | (stage2_label == int(args.tower_class))
    )
    if not np.any(line_mask) or not np.any(tower_mask):
        return recover_mask, {
            "enabled": True,
            "candidate_points": 0,
            "components": 0,
            "kept_components": 0,
            "recovered_points": 0,
        }

    roi_min = roi_coord.min(axis=0)
    roi_max = roi_coord.max(axis=0)
    roi_height = float(roi_max[2] - roi_min[2])
    upper_ratio = min(max(float(args.recover_insulator_upper_height_ratio), 0.0), 1.0)
    min_z = float(roi_min[2] + roi_height * upper_ratio)

    line_tree = cKDTree(roi_coord[line_mask])
    tower_xy = roi_coord[tower_mask, :2]
    tower_core_xy = np.median(tower_xy, axis=0)
    candidate_base = tower_mask & (roi_coord[:, 2] >= min_z)
    candidate_indices = np.flatnonzero(candidate_base)
    if candidate_indices.size == 0:
        return recover_mask, {
            "enabled": True,
            "candidate_points": 0,
            "components": 0,
            "kept_components": 0,
            "recovered_points": 0,
        }

    candidate_points = roi_coord[candidate_indices]
    labels, component_count = voxel_component_labels(
        candidate_points, float(args.recover_insulator_voxel_size)
    )
    non_candidate_tower_mask = tower_mask.copy()
    non_candidate_tower_mask[candidate_indices] = False
    tower_body_tree = (
        cKDTree(roi_coord[non_candidate_tower_mask])
        if np.any(non_candidate_tower_mask)
        else None
    )
    kept_components = 0
    for component_id in range(component_count):
        local_member = np.flatnonzero(labels == component_id)
        if local_member.size < int(args.recover_insulator_min_points):
            continue

        member_indices = candidate_indices[local_member]
        member_points = roi_coord[member_indices]
        extent = member_points.max(axis=0) - member_points.min(axis=0)
        if float(np.max(extent)) > float(args.recover_insulator_max_extent):
            continue
        if float(extent[2]) < float(args.recover_insulator_min_z_extent):
            continue

        member_center_xy = np.median(member_points[:, :2], axis=0)
        core_distance = float(np.linalg.norm(member_center_xy - tower_core_xy))
        if core_distance < float(args.recover_insulator_min_core_distance):
            continue

        if tower_body_tree is not None:
            tower_dist, _ = tower_body_tree.query(member_points, k=1, workers=-1)
            tower_neighbor_ratio = float(
                np.mean(
                    tower_dist <= float(args.recover_insulator_tower_neighbor_radius)
                )
            )
            if tower_neighbor_ratio > float(
                args.recover_insulator_max_tower_neighbor_ratio
            ):
                continue

        min_stage2_score = float(args.recover_insulator_min_stage2_score)
        if min_stage2_score > 0 and float(np.max(stage2_ins_score[member_indices])) < min_stage2_score:
            continue

        centered = member_points - member_points.mean(axis=0, keepdims=True)
        if member_points.shape[0] >= 3:
            _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
            first = float(singular_values[0])
            second = float(singular_values[1]) if singular_values.shape[0] > 1 else 0.0
            linearity = 1.0 - second / max(first, 1e-6)
            main_axis = vh[0]
        else:
            linearity = 1.0
            main_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if linearity < float(args.recover_insulator_min_linearity):
            continue

        projection = centered @ main_axis
        span = float(projection.max() - projection.min())
        if span <= 1e-6:
            continue
        closest = projection[:, None] * main_axis[None, :]
        perpendicular = centered - closest
        bend_ratio = float(np.sqrt(np.mean(np.sum(perpendicular * perpendicular, axis=1))) / span)
        if bend_ratio > float(args.recover_insulator_max_bend_ratio):
            continue

        member_line_dist, _ = line_tree.query(member_points, k=1, workers=-1)
        line_contact = int(
            np.count_nonzero(
                member_line_dist <= float(args.recover_insulator_line_radius)
            )
        )
        if line_contact < int(args.recover_insulator_min_line_contact_points):
            continue

        component_tower_mask = tower_mask.copy()
        component_tower_mask[member_indices] = False
        if not np.any(component_tower_mask):
            continue
        tower_endpoint_tree = cKDTree(roi_coord[component_tower_mask])
        endpoint_percent = min(
            max(float(args.recover_insulator_end_percentile), 1.0), 49.0
        )
        low_cut = np.percentile(projection, endpoint_percent)
        high_cut = np.percentile(projection, 100.0 - endpoint_percent)
        low_end = member_points[projection <= low_cut]
        high_end = member_points[projection >= high_cut]
        if low_end.shape[0] == 0 or high_end.shape[0] == 0:
            continue

        low_line_dist, _ = line_tree.query(low_end, k=1, workers=-1)
        high_line_dist, _ = line_tree.query(high_end, k=1, workers=-1)
        low_tower_dist, _ = tower_endpoint_tree.query(low_end, k=1, workers=-1)
        high_tower_dist, _ = tower_endpoint_tree.query(high_end, k=1, workers=-1)
        low_touches_line = float(np.min(low_line_dist)) <= float(
            args.recover_insulator_line_end_radius
        )
        high_touches_line = float(np.min(high_line_dist)) <= float(
            args.recover_insulator_line_end_radius
        )
        low_touches_tower = float(np.min(low_tower_dist)) <= float(
            args.recover_insulator_tower_end_radius
        )
        high_touches_tower = float(np.min(high_tower_dist)) <= float(
            args.recover_insulator_tower_end_radius
        )
        has_opposite_endpoint_contact = (
            low_touches_line and high_touches_tower
        ) or (
            high_touches_line and low_touches_tower
        )
        if not has_opposite_endpoint_contact:
            continue

        recover_mask[member_indices] = True
        kept_components += 1

    return recover_mask, {
        "enabled": True,
        "candidate_points": int(candidate_indices.size),
        "components": int(component_count),
        "kept_components": int(kept_components),
        "recovered_points": int(np.count_nonzero(recover_mask)),
    }


def run_stage1(model, cfg, pipeline, coord, color, device, args, stats):
    return base.infer_las_tiled(
        model=model,
        cfg=cfg,
        pipeline=pipeline,
        coord=coord,
        color=color,
        device=device,
        fragment_batch_size=args.stage1_fragment_batch_size,
        tile_size=args.stage1_tile_size,
        tile_stride=args.stage1_tile_stride,
        min_tile_points=args.stage1_min_tile_points,
        pre_voxel_size=args.stage1_pre_voxel_size,
        merge_mode=args.stage1_merge_mode,
        context_margin=args.stage1_context_margin,
        stats=stats,
    )


def run_stage2_and_fuse(model, cfg, pipeline, coord, color, stage1_pred, rois, device, args):
    """对 ROI 重推并融合结果。"""
    final_pred = stage1_pred.copy()
    best_insulator_score = np.zeros(coord.shape[0], dtype=np.float32)
    best_replace_score = np.zeros(coord.shape[0], dtype=np.float32)
    roi_reports = []
    processed = 0
    skipped = 0
    updated_insulator_points = 0
    structure_recovered_insulator_points = 0
    replaced_points = 0

    for roi_id, roi in enumerate(rois, start=1):
        roi_start = time.perf_counter()
        center = roi["center"]
        roi_mask = points_in_roi(coord, roi)
        roi_indices = np.flatnonzero(roi_mask)
        if roi_indices.size < int(args.min_roi_points):
            skipped += 1
            roi_reports.append(
                {
                    "roi_id": int(roi_id),
                    "roi_type": roi.get("roi_type", "unknown"),
                    "center_xyz": [round(float(v), 6) for v in center],
                    "min_xyz": [round(float(v), 6) for v in roi["min_xyz"]],
                    "max_xyz": [round(float(v), 6) for v in roi["max_xyz"]],
                    "points": int(roi_indices.size),
                    "skipped": True,
                    "reason": "too_few_points",
                }
            )
            continue

        processed += 1
        print(
            f"  stage2 roi {roi_id}/{len(rois)} ({roi.get('roi_type', 'unknown')}): "
            f"{roi_indices.size} points "
            f"center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})",
            flush=True,
        )
        prob = base.infer_tile_probs(
            model=model,
            cfg=cfg,
            pipeline=pipeline,
            coord=coord[roi_indices],
            color=color[roi_indices],
            device=device,
            fragment_batch_size=args.stage2_fragment_batch_size,
            pre_voxel_size=args.stage2_pre_voxel_size,
        )
        label = prob.argmax(axis=1).astype(np.uint8, copy=False)
        confidence = prob.max(axis=1)
        ins_score = prob[:, int(args.insulator_class)]

        roi_updated_ins = 0
        roi_replaced = 0
        roi_structure_recovered = 0
        structure_recovery_summary = {
            "enabled": bool(args.recover_insulator_by_structure),
            "candidate_points": 0,
            "components": 0,
            "kept_components": 0,
            "recovered_points": 0,
        }
        if args.fusion_mode == "replace-roi":
            replace_mask = confidence >= float(args.replace_score_threshold)
            global_replace = roi_indices[replace_mask]
            better = confidence[replace_mask] > best_replace_score[global_replace]
            if np.any(better):
                target_indices = global_replace[better]
                final_pred[target_indices] = label[replace_mask][better]
                best_replace_score[target_indices] = confidence[replace_mask][better]
                roi_replaced = int(target_indices.size)
                replaced_points += roi_replaced

        ins_mask = ins_score >= float(args.insulator_score_threshold)
        if args.require_insulator_argmax:
            ins_mask &= label == int(args.insulator_class)
        global_ins = roi_indices[ins_mask]
        if global_ins.size:
            better = ins_score[ins_mask] > best_insulator_score[global_ins]
            if np.any(better):
                target_indices = global_ins[better]
                final_pred[target_indices] = int(args.insulator_class)
                best_insulator_score[target_indices] = ins_score[ins_mask][better]
                roi_updated_ins = int(target_indices.size)
                updated_insulator_points += roi_updated_ins

        structure_mask, structure_recovery_summary = recover_insulator_mask_by_structure(
            roi_coord=coord[roi_indices],
            stage1_label=stage1_pred[roi_indices],
            stage2_label=label,
            stage2_ins_score=ins_score,
            args=args,
        )
        if np.any(structure_mask):
            global_structure = roi_indices[structure_mask]
            # 结构恢复只补救非导线点，避免把真正的导线改成绝缘子。
            global_structure = global_structure[
                final_pred[global_structure] != int(args.line_class)
            ]
            if global_structure.size:
                before = final_pred[global_structure] != int(args.insulator_class)
                target_indices = global_structure[before]
                final_pred[target_indices] = int(args.insulator_class)
                roi_structure_recovered = int(target_indices.size)
                structure_recovered_insulator_points += roi_structure_recovered

        roi_counts = np.bincount(label, minlength=cfg.data.num_classes)
        roi_reports.append(
            {
                "roi_id": int(roi_id),
                "roi_type": roi.get("roi_type", "unknown"),
                "center_xyz": [round(float(v), 6) for v in center],
                "min_xyz": [round(float(v), 6) for v in roi["min_xyz"]],
                "max_xyz": [round(float(v), 6) for v in roi["max_xyz"]],
                "points": int(roi_indices.size),
                "skipped": False,
                "stage2_pred_counts": {
                    str(idx): int(roi_counts[idx]) for idx in range(cfg.data.num_classes)
                },
                "updated_insulator_points": int(roi_updated_ins),
                "structure_recovered_insulator_points": int(roi_structure_recovered),
                "structure_recovery": structure_recovery_summary,
                "replaced_points": int(roi_replaced),
                "elapsed_sec": round(time.perf_counter() - roi_start, 3),
            }
        )
        print(
            f"    roi done in {time.perf_counter() - roi_start:.2f}s, "
            f"updated_insulator={roi_updated_ins}, "
            f"structure_recovered={roi_structure_recovered}, "
            f"replaced={roi_replaced}",
            flush=True,
        )

    return final_pred, {
        "roi_total": int(len(rois)),
        "roi_processed": int(processed),
        "roi_skipped": int(skipped),
        "updated_insulator_points": int(updated_insulator_points),
        "structure_recovered_insulator_points": int(
            structure_recovered_insulator_points
        ),
        "replaced_points": int(replaced_points),
        "rois": roi_reports,
    }


def report_path_for(source_path, target_path, args, multi_file):
    if args.report and not multi_file:
        return Path(args.report)
    return Path(target_path).with_name(Path(target_path).stem + "_two_stage_report.json")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    device = torch.device(args.device)

    if args.stage1_fragment_batch_size < 1 or args.stage2_fragment_batch_size < 1:
        raise ValueError("fragment batch size must be >= 1")

    cfg = Config.fromfile(args.config_file)
    if args.disable_flash:
        base.set_enable_flash(cfg, False)
    if int(args.insulator_class) >= int(cfg.data.num_classes):
        raise ValueError("--insulator-class 超过模型类别数")

    print(f"Loading model: {args.weight}", flush=True)
    model = base.load_model(cfg, args.weight, device)
    stage1_pipeline = base.build_test_pipeline(
        cfg, point_max=args.stage1_point_max, grid_size=args.stage1_grid_size
    )
    stage2_pipeline = base.build_test_pipeline(
        cfg, point_max=args.stage2_point_max, grid_size=args.stage2_grid_size
    )

    jobs = base.iter_jobs(input_path, output_path)
    if not jobs:
        raise FileNotFoundError(f"No .las files found under: {input_path}")
    multi_file = len(jobs) > 1

    all_start = time.perf_counter()
    for source_path, target_path in jobs:
        if target_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists, use --overwrite: {target_path}")

        job_start = time.perf_counter()
        base.reset_cuda_peak_memory(device)
        print(f"Reading {source_path}", flush=True)
        read_start = time.perf_counter()
        coord, color, las = base.read_las(source_path, args.default_color, args.las_backend)
        read_time = time.perf_counter() - read_start
        print(f"Loaded {coord.shape[0]} points in {read_time:.2f}s", flush=True)

        print("Stage-1 full-cloud inference", flush=True)
        stage1_stats = base.new_infer_stats()
        base.sync_cuda(device)
        stage1_start = time.perf_counter()
        stage1_pred = run_stage1(
            model, cfg, stage1_pipeline, coord, color, device, args, stage1_stats
        )
        base.sync_cuda(device)
        stage1_time = time.perf_counter() - stage1_start

        stage1_raw_class_counts = class_summary(stage1_pred, cfg)
        stage1_pred, stage1_structure_summary = make_stage1_structure_prediction(
            stage1_pred, args
        )
        print(
            "Stage-1 coarse structure: merged_insulator_to_tower_points={}".format(
                stage1_structure_summary["merged_insulator_to_tower_points"]
            ),
            flush=True,
        )

        print("Stage-1 tower false-positive cleanup", flush=True)
        stage1_cleanup_start = time.perf_counter()
        stage1_pred, stage1_cleanup_summary = cleanup_stage1_tower_predictions(
            coord, stage1_pred, args
        )
        stage1_cleanup_time = time.perf_counter() - stage1_cleanup_start
        print(
            "  tower cleanup: components={}/{} kept, removed_points={}, time={:.2f}s".format(
                stage1_cleanup_summary["components_kept"],
                stage1_cleanup_summary["components_total"],
                stage1_cleanup_summary["removed_tower_points"],
                stage1_cleanup_time,
            ),
            flush=True,
        )

        if args.save_stage1_las:
            stage1_path = Path(args.save_stage1_las)
            if multi_file:
                stage1_path = stage1_path / Path(source_path).name
            if stage1_path.exists() and not args.overwrite:
                raise FileExistsError(f"Stage1 output exists, use --overwrite: {stage1_path}")
            base.write_las(source_path, stage1_path, las, stage1_pred, colorize=not args.no_colorize)
            print(f"Saved Stage-1 LAS: {stage1_path}", flush=True)

        print(f"Build Stage-2 ROIs (mode={args.roi_mode})", flush=True)
        roi_start = time.perf_counter()
        rois, candidate_summary = build_stage2_rois(coord, stage1_pred, args)
        roi_build_time = time.perf_counter() - roi_start
        print(
            "  candidates: mode={}, roi_centers={}".format(
                candidate_summary.get("roi_mode", args.roi_mode),
                candidate_summary["roi_centers"],
            ),
            flush=True,
        )

        print("Stage-2 ROI inference and fusion", flush=True)
        base.sync_cuda(device)
        stage2_start = time.perf_counter()
        final_pred, stage2_summary = run_stage2_and_fuse(
            model, cfg, stage2_pipeline, coord, color, stage1_pred, rois, device, args
        )
        base.sync_cuda(device)
        stage2_time = time.perf_counter() - stage2_start

        write_start = time.perf_counter()
        base.write_las(source_path, target_path, las, final_pred, colorize=not args.no_colorize)
        write_time = time.perf_counter() - write_start

        report_path = report_path_for(source_path, target_path, args, multi_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "input": str(source_path),
            "output": str(target_path),
            "config_file": str(args.config_file),
            "weight": str(args.weight),
            "stage1_raw_class_counts": stage1_raw_class_counts,
            "stage1_structure_summary": stage1_structure_summary,
            "stage1_class_counts": class_summary(stage1_pred, cfg),
            "stage1_tower_cleanup": stage1_cleanup_summary,
            "final_class_counts": class_summary(final_pred, cfg),
            "candidate_summary": candidate_summary,
            "stage2_summary": stage2_summary,
            "parameters": vars(args),
            "timing_sec": {
                "read": round(read_time, 3),
                "stage1": round(stage1_time, 3),
                "stage1_tower_cleanup": round(stage1_cleanup_time, 3),
                "roi_build": round(roi_build_time, 3),
                "stage2": round(stage2_time, 3),
                "write": round(write_time, 3),
                "total": round(time.perf_counter() - job_start, 3),
            },
            "cuda_memory": base.cuda_peak_memory_summary(device),
        }
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)

        print(f"Saved final LAS: {target_path}", flush=True)
        print(f"Saved report: {report_path}", flush=True)
        print(
            "Timing: total={}, stage1={}, roi_build={}, stage2={}, write={}, {}".format(
                base.format_seconds(time.perf_counter() - job_start),
                base.format_seconds(stage1_time),
                base.format_seconds(roi_build_time),
                base.format_seconds(stage2_time),
                base.format_seconds(write_time),
                base.cuda_peak_memory_summary(device),
            ),
            flush=True,
        )

    print(f"All done in {base.format_seconds(time.perf_counter() - all_start)}", flush=True)


"""
python tools/infer/infer_las_semseg_two_stage.py \
  --input /24085403037/24085403037/shixi/dataset/6_23_demo/test/test_insulator_hengdan/source_tower/Stage1_tower/tower_004_杆塔4.las \
  --output /24085403037/24085403037/shixi/dataset/6_23_demo/test/test_insulator_hengdan/source_tower/tower_004_杆塔4_test.las \
  --save-stage1-las /24085403037/24085403037/shixi/dataset/6_23_demo/test/test_insulator_hengdan/source_towertower_004_杆塔4_stage1_test.las \
  --disable-stage1-tower-cleanup \
  --roi-mode tower \
  --stage1-merge-mode halo \
  --stage1-tile-size 40 \
  --stage1-tile-stride 40 \
  --stage1-context-margin 10 \
  --stage1-pre-voxel-size 0.08 \
  --stage2-pre-voxel-size 0 \
  --stage2-point-max 200000 \
  --tower-roi-xy-margin 10.0 \
  --tower-roi-z-margin 8.0 \
  --insulator-score-threshold 0.25 \
  --overwrite
"""
def cleanup_stage1_tower_predictions(coord, stage1_pred, args):
    """清理第一阶段中被误分为杆塔的背景点和导线点。

    清理只会将 class=1 改回背景 class=0，不修改导线、绝缘子等其他类别。
    保留杆塔需要同时满足：连通组件足够大、具有足够高度、整体以竖直方向为主，
    并且其上部存在一定数量与预测导线接触的杆塔点。最后一项用于剔除位于
    导线下方或旁边、但没有真实连接关系的背景/杆状误检。
    """
    raw_tower_count = int(np.count_nonzero(stage1_pred == int(args.tower_class)))
    if args.disable_stage1_tower_cleanup or raw_tower_count == 0:
        return stage1_pred, {
            "enabled": not args.disable_stage1_tower_cleanup,
            "raw_tower_points": raw_tower_count,
            "cleaned_tower_points": raw_tower_count,
            "removed_tower_points": 0,
            "components_total": 0,
            "components_kept": 0,
            "components_removed_by_reason": {},
            "line_contact_checked": False,
        }

    # 复用独立清理脚本中经过验证的快速体素连通域聚类实现。
    cluster_args = argparse.Namespace(
        tower_class=int(args.tower_class),
        tower_voxel_size=float(args.stage1_tower_cleanup_voxel_size),
        tower_connectivity=int(args.stage1_tower_cleanup_connectivity),
    )
    components, tower_indices, point_comp = tower_clean.cluster_towers(
        coord, stage1_pred, cluster_args
    )
    if not components:
        return stage1_pred, {
            "enabled": True,
            "raw_tower_points": raw_tower_count,
            "cleaned_tower_points": raw_tower_count,
            "removed_tower_points": 0,
            "components_total": 0,
            "components_kept": 0,
            "components_removed_by_reason": {},
            "line_contact_checked": False,
        }

    line_coord = coord[stage1_pred == int(args.line_class)]
    line_tree = cKDTree(line_coord) if line_coord.shape[0] else None
    component_order = np.argsort(point_comp, kind="stable")
    component_counts = np.bincount(point_comp, minlength=len(components))
    component_offsets = np.concatenate(([0], np.cumsum(component_counts)))
    removed_by_reason = {}
    keep_by_component = np.zeros(len(components), dtype=bool)
    component_members = []
    component_centers_xy = np.empty((len(components), 2), dtype=np.float64)
    seed_component_ids = []

    for component in components:
        component_id = int(component["id"])
        point_count = int(component["point_count"])
        start = int(component_offsets[component_id])
        end = int(component_offsets[component_id + 1])
        member_indices = tower_indices[component_order[start:end]]
        component_members.append(member_indices)
        component_centers_xy[component_id] = np.asarray(
            component["center"], dtype=np.float64
        )[:2]

        size = np.asarray(component["size"], dtype=np.float64)
        height = float(size[2])
        max_xy_size = float(max(size[0], size[1]))
        vertical_ratio = height / max(max_xy_size, 1e-6)

        # 先只找“塔身主干 seed”。横担、塔帽、塔身碎片可能很扁或很低，
        # 不能在这里直接删除，后面会按 XY 归并到同一座物理塔。
        is_seed = (
            point_count >= int(args.stage1_tower_cleanup_min_points)
            and height >= float(args.stage1_tower_cleanup_min_height)
            and max_xy_size >= float(args.stage1_tower_cleanup_min_xy_size)
            and vertical_ratio >= float(args.stage1_tower_cleanup_min_vertical_ratio)
        )
        component["stage1_cleanup_is_seed"] = bool(is_seed)
        if is_seed:
            seed_component_ids.append(component_id)

    if not seed_component_ids:
        # 找不到可靠塔身主干时，不做破坏性清理。这个分支比误删整座塔更安全。
        return stage1_pred, {
            "enabled": True,
            "raw_tower_points": raw_tower_count,
            "cleaned_tower_points": raw_tower_count,
            "removed_tower_points": 0,
            "components_total": int(len(components)),
            "components_kept": int(len(components)),
            "components_removed_by_reason": {"no_tower_seed_preserved_all": 0},
            "physical_tower_groups_total": 0,
            "physical_tower_groups_kept": 0,
            "seed_components": 0,
            "line_contact_checked": False,
        }

    merge_radius = float(args.stage1_tower_cleanup_merge_xy_radius)
    seed_component_ids = np.asarray(seed_component_ids, dtype=np.int64)
    seed_centers = component_centers_xy[seed_component_ids]
    assigned_group = np.full(len(components), -1, dtype=np.int64)

    # 把每个组件分配到最近的塔身 seed。只要它在 XY 上靠近塔身，就认为属于同一座物理塔。
    for component_id in range(len(components)):
        distances = np.linalg.norm(
            seed_centers - component_centers_xy[component_id][None, :], axis=1
        )
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) <= merge_radius:
            assigned_group[component_id] = nearest

    upper_ratio = min(
        max(float(args.stage1_tower_cleanup_upper_height_ratio), 0.0), 1.0
    )
    min_contact_points = int(args.stage1_tower_cleanup_min_contact_points)
    groups_kept = 0
    for group_id in range(seed_component_ids.size):
        group_component_ids = np.flatnonzero(assigned_group == group_id)
        if group_component_ids.size == 0:
            continue

        group_indices = np.concatenate(
            [component_members[int(component_id)] for component_id in group_component_ids]
        )
        group_points = coord[group_indices]
        group_min = group_points.min(axis=0)
        group_max = group_points.max(axis=0)
        group_height = float(group_max[2] - group_min[2])
        contact_count = 0

        if line_tree is not None and min_contact_points > 0:
            upper_z = float(group_min[2] + group_height * upper_ratio)
            upper_indices = group_indices[coord[group_indices, 2] >= upper_z]
            if upper_indices.size:
                distances, _ = line_tree.query(coord[upper_indices], k=1, workers=-1)
                contact_count = int(
                    np.count_nonzero(
                        distances <= float(args.stage1_tower_cleanup_contact_radius)
                    )
                )
            keep_group = contact_count >= min_contact_points
        else:
            keep_group = True

        for component_id in group_component_ids:
            component = components[int(component_id)]
            component["stage1_physical_group_id"] = int(group_id)
            component["stage1_group_line_contact_points"] = int(contact_count)
            component["stage1_cleanup_keep"] = bool(keep_group)
            if keep_group:
                keep_by_component[int(component_id)] = True
            else:
                component["stage1_cleanup_remove_reason"] = (
                    "too_few_group_upper_line_contacts"
                )

        if keep_group:
            groups_kept += 1

    for component_id, component in enumerate(components):
        if assigned_group[component_id] >= 0:
            if not keep_by_component[component_id]:
                reason = component.get(
                    "stage1_cleanup_remove_reason",
                    "too_few_group_upper_line_contacts",
                )
                removed_by_reason[reason] = removed_by_reason.get(reason, 0) + 1
            continue

        component["stage1_cleanup_keep"] = False
        component["stage1_cleanup_remove_reason"] = "not_near_tower_seed"
        removed_by_reason["not_near_tower_seed"] = (
            removed_by_reason.get("not_near_tower_seed", 0) + 1
        )

    keep_by_point = keep_by_component[point_comp]
    removed_indices = tower_indices[~keep_by_point]
    cleaned_pred = stage1_pred.copy()
    cleaned_pred[removed_indices] = int(args.background_class)
    return cleaned_pred, {
        "enabled": True,
        "raw_tower_points": raw_tower_count,
        "cleaned_tower_points": int(
            np.count_nonzero(cleaned_pred == int(args.tower_class))
        ),
        "removed_tower_points": int(removed_indices.size),
        "components_total": int(len(components)),
        "components_kept": int(np.count_nonzero(keep_by_component)),
        "components_removed_by_reason": removed_by_reason,
        "physical_tower_groups_total": int(seed_component_ids.size),
        "physical_tower_groups_kept": int(groups_kept),
        "seed_components": int(seed_component_ids.size),
        "merge_xy_radius": float(args.stage1_tower_cleanup_merge_xy_radius),
        "line_contact_checked": line_tree is not None,
    }


if __name__ == "__main__":
    main()
