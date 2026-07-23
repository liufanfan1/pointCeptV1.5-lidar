"""从分割完成的输电点云中提取横担左右端点和中心点。

输入 LAS/LAZ 的 classification 默认约定：
    1：杆塔
    2：导线
    3：绝缘子

横担提取规则：
    1. 对杆塔点做体素连通域聚类，得到独立杆塔；
    2. 在每个杆塔附近提取导线，并按 Z 高度从上往下分层；
    3. 在“当前导线层最高点到上一层横担最低点”之间扫描杆塔宽度；
    4. 取线路侧向宽度最大的杆塔切片作为横担；
    5. 第一层横担允许没有绝缘子，第二层及以下必须存在绝缘子点；
    6. 沿横担方向输出左端点、中心点和右端点的全局 XYZ 坐标。

示例：
python tools/infer/extract_segmented_crossarm_keypoints.py \
  --input input_pred.las \
  --output crossarm_keypoints.json \
  --tower-las-output-dir crossarm_visual \
  --overwrite
"""

import argparse
import json
from itertools import product
from pathlib import Path

import laspy
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="从分割后的 LAS/LAZ 中提取每层横担的左右端点和中心点。"
    )
    parser.add_argument("--input", required=True, help="分割后的 LAS/LAZ 文件。")
    parser.add_argument("--output", required=True, help="输出横担关键点 JSON。")
    parser.add_argument(
        "--tower-las-output-dir",
        default=None,
        help="可选：按杆塔输出包含横担标记的局部 LAS 文件。",
    )
    parser.add_argument("--tower-class", type=int, default=1, help="杆塔类别。")
    parser.add_argument("--line-class", type=int, default=2, help="导线类别。")
    parser.add_argument("--insulator-class", type=int, default=3, help="绝缘子类别。")
    parser.add_argument(
        "--tower-voxel-size",
        type=float,
        default=0.75,
        help="杆塔体素连通域边长，单位米。",
    )
    parser.add_argument(
        "--min-tower-points",
        type=int,
        default=200,
        help="一个杆塔实例至少包含的原始杆塔点数。",
    )
    parser.add_argument(
        "--min-tower-height",
        type=float,
        default=4.0,
        help="杆塔实例最小高度，单位米。",
    )
    parser.add_argument(
        "--line-search-radius",
        type=float,
        default=35.0,
        help="以杆塔中心为圆心搜索导线的 XY 半径，单位米。",
    )
    parser.add_argument(
        "--line-along-window",
        type=float,
        default=10.0,
        help="导线分层时，杆塔沿线路方向两侧的搜索半宽，单位米。",
    )
    parser.add_argument(
        "--line-layer-z-gap",
        type=float,
        default=2.0,
        help="导线原始高度簇允许的最大 Z 跨度，单位米。",
    )
    parser.add_argument(
        "--line-layer-merge-count",
        type=int,
        default=1,
        help=(
            "从上往下每几个原始导线高度簇合并成一个物理层。"
            "本脚本已经按 Z 跨度形成物理层，默认 1，不再二次合并。"
        ),
    )
    parser.add_argument(
        "--min-line-layer-points",
        type=int,
        default=500,
        help="一个原始导线高度簇至少包含的点数，默认 500，用于过滤层间少量噪点。",
    )
    parser.add_argument(
        "--scan-z-step",
        type=float,
        default=0.5,
        help="横担高度扫描步长，单位米。",
    )
    parser.add_argument(
        "--scan-z-window",
        type=float,
        default=1.2,
        help="每个扫描高度使用的杆塔点 Z 窗口厚度，单位米。",
    )
    parser.add_argument(
        "--crossarm-along-margin",
        type=float,
        default=5.0,
        help="横担切片在线路方向上的搜索半宽，单位米。",
    )
    parser.add_argument(
        "--min-crossarm-points",
        type=int,
        default=50,
        help="一个横担切片至少包含的杆塔点数。",
    )
    parser.add_argument(
        "--min-crossarm-width",
        type=float,
        default=2.0,
        help="横担在线路侧向上的最小宽度，单位米。",
    )
    parser.add_argument(
        "--endpoint-percentile",
        type=float,
        default=2.0,
        help="横担左右端使用的 side 方向分位数，默认取 2%% 和 98%%。",
    )
    parser.add_argument(
        "--insulator-z-margin",
        type=float,
        default=2.0,
        help="判断绝缘子是否接触横担时允许的 Z 距离，单位米。",
    )
    parser.add_argument(
        "--min-insulator-points-near-layer",
        type=int,
        default=3,
        help="第二层及以下横担附近至少需要的绝缘子点数。",
    )
    parser.add_argument(
        "--no-require-insulator-after-first-layer",
        dest="require_insulator_after_first_layer",
        action="store_false",
        help="关闭第二层及以下必须存在绝缘子的规则。",
    )
    parser.set_defaults(require_insulator_after_first_layer=True)
    parser.add_argument(
        "--tower-bind-xy-margin",
        type=float,
        default=8.0,
        help="把绝缘子绑定到杆塔时，杆塔 XY 包围盒的外扩距离，单位米。",
    )
    parser.add_argument(
        "--tower-bind-z-margin",
        type=float,
        default=8.0,
        help="把绝缘子绑定到杆塔时，杆塔 Z 包围盒的外扩距离，单位米。",
    )
    parser.add_argument(
        "--tower-crop-xy-margin",
        type=float,
        default=20.0,
        help="可视化 LAS 在杆塔 XY 包围盒外扩的距离，单位米。",
    )
    parser.add_argument(
        "--tower-crop-z-margin",
        type=float,
        default=5.0,
        help="可视化 LAS 在杆塔 Z 包围盒外扩的距离，单位米。",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.5,
        help="端点和中心点标记边长，单位米。",
    )
    parser.add_argument(
        "--marker-step",
        type=float,
        default=0.1,
        help="可视化标记采样间隔，单位米。",
    )
    parser.add_argument(
        "--crossarm-line-step",
        type=float,
        default=0.1,
        help="横担端点连线采样间隔，单位米。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有文件。")
    return parser.parse_args()


def read_las(path):
    """读取 LAS/LAZ，并返回点云对象、全局坐标和语义类别。"""
    if path.suffix.lower() not in (".las", ".laz"):
        raise ValueError(f"只支持 .las 或 .laz 输入：{path}")
    las = laspy.read(path)
    dimensions = set(las.point_format.dimension_names)
    if "classification" not in dimensions:
        raise ValueError(f"输入点云没有 classification 字段：{path}")
    coord = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    classification = np.asarray(las.classification, dtype=np.int32)
    return las, coord, classification


def neighbor_offsets():
    """生成 26 邻域的一半偏移，避免重复检查同一对体素。"""
    offsets = []
    for offset in product((-1, 0, 1), repeat=3):
        if offset == (0, 0, 0) or offset <= (0, 0, 0):
            continue
        offsets.append(np.asarray(offset, dtype=np.int64))
    return offsets


def voxel_component_labels(points, voxel_size):
    """用体素连通域为杆塔点生成实例标签。"""
    if points.shape[0] == 0:
        return np.empty((0,), dtype=np.int32)
    if voxel_size <= 0:
        raise ValueError("--tower-voxel-size 必须大于 0")

    origin = points.min(axis=0)
    voxel_coord = np.floor((points - origin) / float(voxel_size)).astype(np.int64)
    unique_voxels, point_to_voxel = np.unique(
        voxel_coord, axis=0, return_inverse=True
    )
    lookup = {tuple(voxel): index for index, voxel in enumerate(unique_voxels)}
    parent = np.arange(unique_voxels.shape[0], dtype=np.int64)
    rank = np.zeros(unique_voxels.shape[0], dtype=np.uint8)

    def find_root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find_root(left)
        right_root = find_root(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    offsets = neighbor_offsets()
    for voxel_id, voxel in enumerate(unique_voxels):
        for offset in offsets:
            neighbor_id = lookup.get(tuple(voxel + offset))
            if neighbor_id is not None:
                union(voxel_id, neighbor_id)

    roots = np.asarray(
        [find_root(index) for index in range(unique_voxels.shape[0])],
        dtype=np.int64,
    )
    _, voxel_labels = np.unique(roots, return_inverse=True)
    return voxel_labels[point_to_voxel].astype(np.int32, copy=False)


def extract_towers(tower_points, args):
    """提取杆塔实例，并按照全局 X、Y 坐标稳定编号。"""
    labels = voxel_component_labels(tower_points, args.tower_voxel_size)
    towers = []
    for label in sorted(set(labels.tolist())):
        points = tower_points[labels == label]
        if points.shape[0] < int(args.min_tower_points):
            continue
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        height = float(bbox_max[2] - bbox_min[2])
        if height < float(args.min_tower_height):
            continue
        towers.append(
            {
                "points": points,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "center": np.median(points, axis=0),
                "height": height,
                "insulator_points": np.empty((0, 3), dtype=np.float64),
                "crossarms": [],
            }
        )

    towers.sort(key=lambda tower: (tower["center"][0], tower["center"][1]))
    for tower_id, tower in enumerate(towers, start=1):
        tower["id"] = tower_id
        tower["name"] = f"杆塔{tower_id}"
    return towers


def bind_insulator_points(towers, insulator_points, args):
    """把每个绝缘子点绑定到包围盒外扩后最近的杆塔。"""
    if not towers or insulator_points.shape[0] == 0:
        return
    assigned = [[] for _ in towers]
    for point in insulator_points:
        candidates = []
        for tower_index, tower in enumerate(towers):
            bbox_min = tower["bbox_min"]
            bbox_max = tower["bbox_max"]
            inside = (
                bbox_min[0] - args.tower_bind_xy_margin
                <= point[0]
                <= bbox_max[0] + args.tower_bind_xy_margin
                and bbox_min[1] - args.tower_bind_xy_margin
                <= point[1]
                <= bbox_max[1] + args.tower_bind_xy_margin
                and bbox_min[2] - args.tower_bind_z_margin
                <= point[2]
                <= bbox_max[2] + args.tower_bind_z_margin
            )
            if inside:
                distance = float(np.linalg.norm(point[:2] - tower["center"][:2]))
                candidates.append((distance, tower_index))
        if candidates:
            _, tower_index = min(candidates, key=lambda item: item[0])
            assigned[tower_index].append(point)

    for tower, points in zip(towers, assigned):
        if points:
            tower["insulator_points"] = np.asarray(points, dtype=np.float64)


def pca_xy_axis(points_xy):
    """使用二维 PCA 估计线路走向，并固定轴方向符号。"""
    if points_xy.shape[0] < 2:
        return np.array([0.0, 1.0], dtype=np.float64)
    centered = points_xy - points_xy.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.array([0.0, 1.0], dtype=np.float64)
    axis = axis / norm
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0:
        axis = -axis
    return axis


def split_line_layers(points, z_gap, min_points):
    """将杆塔附近导线按高度从上往下划分为原始高度簇。"""
    if points.shape[0] == 0:
        return []
    order = np.argsort(points[:, 2])[::-1]
    sorted_points = points[order]
    layers = []
    start = 0
    layer_top = float(sorted_points[0, 2])
    for position in range(1, sorted_points.shape[0]):
        z = float(sorted_points[position, 2])
        if layer_top - z <= float(z_gap):
            continue
        layer = sorted_points[start:position]
        if layer.shape[0] >= int(min_points):
            layers.append(layer)
        start = position
        layer_top = z
    layer = sorted_points[start:]
    if layer.shape[0] >= int(min_points):
        layers.append(layer)
    return layers


def merge_line_layers(layers, merge_count):
    """把相邻原始导线高度簇合并为物理导线层。"""
    merge_count = max(int(merge_count), 1)
    merged = []
    for start in range(0, len(layers), merge_count):
        group = layers[start : start + merge_count]
        merged.append(np.concatenate(group, axis=0))
    return merged


def nearest_actual_point(points, target):
    """将几何端点吸附到最近的真实杆塔点。"""
    index = int(np.argmin(np.sum((points - target[None, :]) ** 2, axis=1)))
    return points[index]


def scan_crossarm(tower, line_layer, along_axis, side_axis, search_max_z, args):
    """在指定 Z 区间扫描 side 宽度，返回最宽的杆塔横担切片。"""
    tower_points = tower["points"]
    origin_xy = tower["center"][:2]
    local_xy = tower_points[:, :2] - origin_xy[None, :]
    tower_along = local_xy @ along_axis
    tower_side = local_xy @ side_axis
    tower_z = tower_points[:, 2]

    line_high_z = float(np.percentile(line_layer[:, 2], 98.0))
    search_min_z = max(float(tower["bbox_min"][2]), line_high_z)
    search_max_z = min(float(tower["bbox_max"][2]), float(search_max_z))
    if search_max_z < search_min_z:
        search_max_z = search_min_z

    step = max(float(args.scan_z_step), 1e-3)
    half_window = max(float(args.scan_z_window) / 2.0, 1e-3)
    centers = np.arange(search_min_z, search_max_z + step * 0.5, step)
    if centers.size == 0:
        centers = np.asarray([search_min_z], dtype=np.float64)
    if centers[-1] < search_max_z - 1e-9:
        centers = np.concatenate((centers, [search_max_z]))

    percentile = float(np.clip(args.endpoint_percentile, 0.0, 49.0))
    best = None
    for center_z in centers:
        mask = (
            (np.abs(tower_z - center_z) <= half_window)
            & (np.abs(tower_along) <= float(args.crossarm_along_margin))
        )
        indices = np.flatnonzero(mask)
        if indices.size < int(args.min_crossarm_points):
            continue
        side_values = tower_side[indices]
        side_min = float(np.percentile(side_values, percentile))
        side_max = float(np.percentile(side_values, 100.0 - percentile))
        width = side_max - side_min
        if best is None or width > best["width"]:
            best = {
                "center_z": float(center_z),
                "indices": indices,
                "side_min": side_min,
                "side_max": side_max,
                "width": float(width),
                "search_min_z": search_min_z,
                "search_max_z": search_max_z,
            }

    if best is None or best["width"] < float(args.min_crossarm_width):
        return None

    selected = tower_points[best["indices"]]
    selected_local = selected[:, :2] - origin_xy[None, :]
    selected_along = selected_local @ along_axis
    selected_side = selected_local @ side_axis
    along_center = float(np.median(selected_along))
    z_center = float(np.median(selected[:, 2]))

    left_target_xy = origin_xy + along_center * along_axis + best["side_min"] * side_axis
    right_target_xy = origin_xy + along_center * along_axis + best["side_max"] * side_axis
    left_target = np.array([left_target_xy[0], left_target_xy[1], z_center])
    right_target = np.array([right_target_xy[0], right_target_xy[1], z_center])
    left_point = nearest_actual_point(selected, left_target)
    right_point = nearest_actual_point(selected, right_target)
    center_point = (left_point + right_point) / 2.0

    return {
        "left_point": left_point,
        "right_point": right_point,
        "center_point": center_point,
        "width": float(np.linalg.norm(right_point - left_point)),
        "z_min": float(selected[:, 2].min()),
        "z_max": float(selected[:, 2].max()),
        "line_layer_z": float(np.median(line_layer[:, 2])),
        "line_layer_high_z": line_high_z,
        "search_min_z": best["search_min_z"],
        "search_max_z": best["search_max_z"],
        "tower_points": int(selected.shape[0]),
    }


def count_insulator_points_near_crossarm(tower, crossarm, args):
    """统计横担高度附近的绝缘子点，悬垂绝缘子只要上端接触即可。"""
    points = tower["insulator_points"]
    if points.shape[0] == 0:
        return 0
    center = crossarm["center_point"]
    xy_distance = np.linalg.norm(points[:, :2] - center[:2][None, :], axis=1)
    z_distance = np.abs(points[:, 2] - center[2])
    mask = (
        (xy_distance <= float(args.line_search_radius))
        & (z_distance <= float(args.insulator_z_margin))
    )
    return int(np.count_nonzero(mask))


def extract_crossarms_for_tower(tower, line_points, args):
    """按从上到下顺序，为一座杆塔提取横担。"""
    center_xy = tower["center"][:2]
    distance_xy = np.linalg.norm(line_points[:, :2] - center_xy[None, :], axis=1)
    radial_line = line_points[distance_xy <= float(args.line_search_radius)]
    if radial_line.shape[0] < int(args.min_line_layer_points):
        return np.array([0.0, 1.0]), np.array([-1.0, 0.0])

    along_axis = pca_xy_axis(radial_line[:, :2])
    side_axis = np.array([-along_axis[1], along_axis[0]], dtype=np.float64)
    radial_local = radial_line[:, :2] - center_xy[None, :]
    radial_along = radial_local @ along_axis
    near_line = radial_line[
        np.abs(radial_along) <= float(args.line_along_window)
    ]
    raw_layers = split_line_layers(
        near_line,
        args.line_layer_z_gap,
        args.min_line_layer_points,
    )
    line_layers = merge_line_layers(raw_layers, args.line_layer_merge_count)

    previous_crossarm_min_z = float(tower["bbox_max"][2])
    kept = []
    for source_layer_index, line_layer in enumerate(line_layers, start=1):
        crossarm = scan_crossarm(
            tower,
            line_layer,
            along_axis,
            side_axis,
            previous_crossarm_min_z,
            args,
        )
        if crossarm is None:
            continue

        insulator_count = count_insulator_points_near_crossarm(tower, crossarm, args)
        is_first_layer = source_layer_index == 1
        if (
            not is_first_layer
            and args.require_insulator_after_first_layer
            and insulator_count < int(args.min_insulator_points_near_layer)
        ):
            continue

        crossarm["source_layer_index"] = source_layer_index
        crossarm["insulator_points_near_layer"] = insulator_count
        crossarm["top_layer_without_insulator_allowed"] = bool(is_first_layer)
        kept.append(crossarm)
        previous_crossarm_min_z = crossarm["z_min"]

    tower["crossarms"] = kept
    return along_axis, side_axis


def xyz_list(point):
    return [round(float(value), 6) for value in point]


def build_json(towers):
    """构建每层横担左点、中点和右点组成的简洁 JSON。"""
    records = []
    for tower in towers:
        for layer_index, crossarm in enumerate(tower["crossarms"], start=1):
            records.append(
                {
                    "tower_id": int(tower["id"]),
                    "crossarm_id": int(layer_index),
                    "left_point_xyz": xyz_list(crossarm["left_point"]),
                    "middle_point_xyz": xyz_list(crossarm["center_point"]),
                    "right_point_xyz": xyz_list(crossarm["right_point"]),
                }
            )
    return {
        "coordinate_system": "input_global_xyz",
        "crossarms": records,
    }


def sample_values(size, step):
    """生成立方体表面采样使用的一维坐标。"""
    if size <= 0 or step <= 0:
        raise ValueError("--marker-size 和 --marker-step 必须大于 0")
    half = float(size) / 2.0
    values = np.arange(-half, half + float(step) * 0.5, float(step))
    return np.unique(np.concatenate((values, [-half, half])))


def cube_marker(center, size, step):
    """生成用于 CloudCompare 检查的立方体表面点。"""
    values = sample_values(size, step)
    points = []
    half = float(size) / 2.0
    for x in values:
        for y in values:
            points.append((x, y, -half))
            points.append((x, y, half))
    for x in values:
        for z in values[1:-1]:
            points.append((x, -half, z))
            points.append((x, half, z))
    for y in values[1:-1]:
        for z in values[1:-1]:
            points.append((-half, y, z))
            points.append((half, y, z))
    return np.asarray(points, dtype=np.float64) + np.asarray(center)[None, :]


def sample_segment(start, end, step):
    """在两个端点之间均匀采样连线。"""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    length = float(np.linalg.norm(end - start))
    count = max(int(np.ceil(length / max(float(step), 1e-3))) + 1, 2)
    ratio = np.linspace(0.0, 1.0, count)[:, None]
    return start[None, :] * (1.0 - ratio) + end[None, :] * ratio


def marker_arrays(crossarms, args):
    """把横担关键点转换为带类别和颜色的 LAS 标记点。"""
    point_groups = []
    class_groups = []
    color_groups = []
    styles = (
        ("left_point", 23, (65535, 0, 65535)),
        ("center_point", 25, (65535, 65535, 0)),
        ("right_point", 24, (0, 65535, 65535)),
    )
    for crossarm in crossarms:
        for key, class_id, color in styles:
            points = cube_marker(crossarm[key], args.marker_size, args.marker_step)
            point_groups.append(points)
            class_groups.append(np.full(points.shape[0], class_id, dtype=np.uint8))
            color_groups.append(
                np.tile(np.asarray(color, dtype=np.uint16), (points.shape[0], 1))
            )
        line = sample_segment(
            crossarm["left_point"], crossarm["right_point"], args.crossarm_line_step
        )
        point_groups.append(line)
        class_groups.append(np.full(line.shape[0], 26, dtype=np.uint8))
        color_groups.append(
            np.tile(np.asarray((0, 65535, 0), dtype=np.uint16), (line.shape[0], 1))
        )
    if not point_groups:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.uint8),
            np.empty((0, 3), dtype=np.uint16),
        )
    return (
        np.concatenate(point_groups),
        np.concatenate(class_groups),
        np.concatenate(color_groups),
    )


def make_empty_las(input_las):
    """创建继承输入点格式、比例、偏移和 CRS 的空 LAS。"""
    header = laspy.LasHeader(
        point_format=input_las.header.point_format,
        version=input_las.header.version,
    )
    header.scales = input_las.header.scales
    header.offsets = input_las.header.offsets
    crs = input_las.header.parse_crs()
    if crs is not None:
        header.add_crs(crs)
    return laspy.LasData(header)


def make_marker_records(input_las, points, classes, colors):
    """把标记坐标转换成与输入 LAS 点格式一致的记录。"""
    records = laspy.ScaleAwarePointRecord.zeros(points.shape[0], header=input_las.header)
    if points.shape[0] == 0:
        return records
    records.x = points[:, 0]
    records.y = points[:, 1]
    records.z = points[:, 2]
    dimensions = set(input_las.point_format.dimension_names)
    if "classification" in dimensions:
        records.classification = classes
    if {"red", "green", "blue"}.issubset(dimensions):
        records.red = colors[:, 0]
        records.green = colors[:, 1]
        records.blue = colors[:, 2]
    return records


def write_tower_visual_las(input_las, coord, towers, output_dir, args):
    """按杆塔裁剪原点云，并追加横担端点和连线标记。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for tower in towers:
        if not tower["crossarms"]:
            continue
        bbox_min = tower["bbox_min"]
        bbox_max = tower["bbox_max"]
        mask = (
            (coord[:, 0] >= bbox_min[0] - args.tower_crop_xy_margin)
            & (coord[:, 0] <= bbox_max[0] + args.tower_crop_xy_margin)
            & (coord[:, 1] >= bbox_min[1] - args.tower_crop_xy_margin)
            & (coord[:, 1] <= bbox_max[1] + args.tower_crop_xy_margin)
            & (coord[:, 2] >= bbox_min[2] - args.tower_crop_z_margin)
            & (coord[:, 2] <= bbox_max[2] + args.tower_crop_z_margin)
        )
        indices = np.flatnonzero(mask)
        points, classes, colors = marker_arrays(tower["crossarms"], args)
        marker_records = make_marker_records(input_las, points, classes, colors)
        arrays = [input_las.points.array[indices], marker_records.array]
        output_las = make_empty_las(input_las)
        output_las.points = laspy.ScaleAwarePointRecord(
            np.concatenate(arrays),
            output_las.header.point_format,
            output_las.header.scales,
            output_las.header.offsets,
        )
        path = output_dir / f"tower_{tower['id']:03d}_{tower['name']}_crossarms.las"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"输出已存在，请添加 --overwrite：{path}")
        output_las.write(path)
        written.append(path)
    return written


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    visual_dir = (
        None if args.tower_las_output_dir is None else Path(args.tower_las_output_dir)
    )

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")
    if output_path.suffix.lower() != ".json":
        raise ValueError(f"--output 必须是 .json 文件：{output_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在，请添加 --overwrite：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"读取点云：{input_path}", flush=True)
    input_las, coord, classification = read_las(input_path)
    tower_points = coord[classification == int(args.tower_class)]
    line_points = coord[classification == int(args.line_class)]
    insulator_points = coord[classification == int(args.insulator_class)]
    print(
        f"总点数={coord.shape[0]}，杆塔={tower_points.shape[0]}，"
        f"导线={line_points.shape[0]}，绝缘子={insulator_points.shape[0]}",
        flush=True,
    )

    towers = extract_towers(tower_points, args)
    bind_insulator_points(towers, insulator_points, args)
    for tower in towers:
        extract_crossarms_for_tower(tower, line_points, args)
        print(
            f"  {tower['name']}：绝缘子点={tower['insulator_points'].shape[0]}，"
            f"横担={len(tower['crossarms'])}",
            flush=True,
        )

    report = build_json(towers)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    written = []
    if visual_dir is not None:
        written = write_tower_visual_las(
            input_las, coord, towers, visual_dir, args
        )

    print(
        f"保存完成：{output_path}，杆塔={len(towers)}，"
        f"横担={len(report['crossarms'])}，可视化LAS={len(written)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
