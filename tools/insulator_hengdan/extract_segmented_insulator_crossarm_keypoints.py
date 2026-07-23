"""从分割完成的输电点云中同时提取绝缘子端点和横担关键点。

输入 LAS/LAZ 的 classification 默认约定：
    1：杆塔
    2：导线
    3：绝缘子

绝缘子提取规则：
    1. 对 classification=3 的原始点构建近邻连通图，不做体素化；
    2. 使用三维 PCA 估计每串绝缘子的主方向；
    3. 沿主方向提取两个真实点云端点。

横担提取规则：
    1. 对杆塔点做体素连通域聚类，得到独立杆塔；
    2. 只在杆塔上部按 Z 扫描水平切片，不使用导线和绝缘子辅助检测；
    3. 对每个切片执行二维 PCA，计算杆塔点的最大水平宽度；
    4. 从宽度曲线中寻找相互分离的局部峰值，每个峰值作为一层横担；
    5. 每层横担都允许挂载零个或多个绝缘子，不因缺少绝缘子而删除；
    6. 沿该层横担自己的 PCA 主轴输出左端点、中心点和右端点。

最终只生成一个嵌套 JSON，按照“杆塔 -> 横担层 -> 左右端点 -> 绝缘子”组织。
横担从上往下排列，同一端点的绝缘子沿局部线路方向排列。

示例：
python tools/insulator_hengdan/extract_segmented_insulator_crossarm_keypoints.py \
  --input /24085403037/24085403037/shixi/dataset/6_23_demo/test/infer/110v12_merged_4cls_Output.las \
  --output /24085403037/24085403037/shixi/dataset/6_23_demo/test/hengdan_insulator/tower_004_keypoints.json \
  --overwrite
"""

import argparse
import json
from itertools import product
from pathlib import Path

import laspy
import numpy as np
from scipy.signal import find_peaks
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser(
        description="一次提取绝缘子端点和横担左、中、右三个关键点。"
    )
    parser.add_argument("--input", required=True, help="分割后的 LAS/LAZ 文件。")
    parser.add_argument("--output", required=True, help="输出统一关键点 JSON。")
    parser.add_argument("--tower-class", type=int, default=1, help="杆塔类别。")
    parser.add_argument("--line-class", type=int, default=2, help="导线类别。")
    parser.add_argument("--insulator-class", type=int, default=3, help="绝缘子类别。")
    parser.add_argument(
        "--insulator-connect-radius",
        "--insulator-voxel-size",
        dest="insulator_connect_radius",
        type=float,
        default=0.20,
        help=(
            "绝缘子原始点近邻连通半径，单位米；旧参数名 "
            "--insulator-voxel-size 仍可兼容，但不再执行体素化。"
        ),
    )
    parser.add_argument(
        "--insulator-neighbors",
        type=int,
        default=16,
        help="构建绝缘子原始点连通图时，每个点最多查询的近邻数量。",
    )
    parser.add_argument(
        "--min-insulator-points",
        type=int,
        default=30,
        help="一个绝缘子实例至少包含的原始点数。",
    )
    parser.add_argument(
        "--min-insulator-height",
        type=float,
        default=0.0,
        help="绝缘子实例最小 Z 高度，单位米；默认不按高度过滤。",
    )
    parser.add_argument(
        "--insulator-endpoint-percentile",
        type=float,
        default=2.0,
        help="沿绝缘子 PCA 主轴提取端点时使用的两端分位数。",
    )
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
        "--crossarm-min-height-ratio",
        type=float,
        default=0.45,
        help="只在杆塔相对高度不低于该比例的位置搜索横担。",
    )
    parser.add_argument(
        "--crossarm-min-z-separation",
        type=float,
        default=2.5,
        help="相邻两层横担峰值的最小高度间隔，单位米。",
    )
    parser.add_argument(
        "--crossarm-min-prominence",
        type=float,
        default=0.5,
        help="横担宽度峰值相对周围切片至少突出的宽度，单位米。",
    )
    parser.add_argument(
        "--crossarm-min-relative-prominence",
        type=float,
        default=0.25,
        help="横担宽度峰突出值与自身宽度的最小比例，用于排除塔身缓慢变宽。",
    )
    parser.add_argument(
        "--crossarm-min-width-ratio",
        type=float,
        default=0.5,
        help="横担宽度至少达到杆塔上部最宽切片的比例，用于排除较窄塔身峰。",
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
        "--insulator-attach-radius",
        type=float,
        default=3.0,
        help="绝缘子端点与横担线段建立挂载关系的最大三维距离，单位米。",
    )
    parser.add_argument(
        "--downward-vertical-ratio",
        type=float,
        default=0.7,
        help=(
            "同一侧挂载不少于两串绝缘子时，逐串判定朝下绝缘子的最小垂直"
            "分量比例；满足条件的绝缘子全部删除，取值范围为 0 到 1。"
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有文件。")
    return parser.parse_args()


def read_las(path):
    """读取 LAS/LAZ，并返回全局坐标和语义类别。"""
    if path.suffix.lower() not in (".las", ".laz"):
        raise ValueError(f"只支持 .las 或 .laz 输入：{path}")
    las = laspy.read(path)
    dimensions = set(las.point_format.dimension_names)
    if "classification" not in dimensions:
        raise ValueError(f"输入点云没有 classification 字段：{path}")
    coord = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    classification = np.asarray(las.classification, dtype=np.int32)
    return coord, classification


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


def raw_point_component_labels(points, radius, max_neighbors):
    """直接使用原始点近邻图聚类，不进行体素化或坐标量化。"""
    point_count = int(points.shape[0])
    if point_count == 0:
        return np.empty((0,), dtype=np.int32)
    if radius <= 0:
        raise ValueError("--insulator-connect-radius 必须大于 0")
    if max_neighbors < 2:
        raise ValueError("--insulator-neighbors 必须不小于 2")

    query_count = min(int(max_neighbors), point_count)
    tree = cKDTree(points)
    try:
        distances, neighbors = tree.query(
            points,
            k=query_count,
            distance_upper_bound=float(radius),
            workers=-1,
        )
    except TypeError:
        # 兼容不支持 workers 参数的旧版 SciPy。
        distances, neighbors = tree.query(
            points,
            k=query_count,
            distance_upper_bound=float(radius),
        )

    if query_count == 1:
        distances = distances[:, None]
        neighbors = neighbors[:, None]
    rows = np.repeat(np.arange(point_count, dtype=np.int64), query_count)
    columns = np.asarray(neighbors, dtype=np.int64).reshape(-1)
    finite = np.isfinite(np.asarray(distances).reshape(-1))
    valid = finite & (columns < point_count) & (columns != rows)
    rows = rows[valid]
    columns = columns[valid]
    graph = coo_matrix(
        (np.ones(rows.shape[0], dtype=np.uint8), (rows, columns)),
        shape=(point_count, point_count),
    )
    _, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.astype(np.int32, copy=False)


def extract_insulator_instance_endpoints(points, endpoint_percentile):
    """沿绝缘子三维 PCA 主轴提取两个真实点云端点。"""
    center = points.mean(axis=0)
    centered = points - center[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]

    # 固定主轴符号，保证相同输入多次运行时端点编号稳定。
    dominant_axis = int(np.argmax(np.abs(axis)))
    if axis[dominant_axis] < 0:
        axis = -axis

    projection = centered @ axis
    percentile = float(np.clip(endpoint_percentile, 0.0, 49.0))
    projection_1 = float(np.percentile(projection, percentile))
    projection_2 = float(np.percentile(projection, 100.0 - percentile))
    target_1 = center + projection_1 * axis
    target_2 = center + projection_2 * axis
    endpoint_1 = nearest_actual_point(points, target_1)
    endpoint_2 = nearest_actual_point(points, target_2)
    return endpoint_1, endpoint_2


def extract_insulator_instances(insulator_points, args):
    """对绝缘子语义点聚类，并保留后续挂载所需的内部几何信息。"""
    labels = raw_point_component_labels(
        insulator_points,
        args.insulator_connect_radius,
        args.insulator_neighbors,
    )
    instances = []
    for label in sorted(set(labels.tolist())):
        points = insulator_points[labels == label]
        if points.shape[0] < int(args.min_insulator_points):
            continue
        height = float(points[:, 2].max() - points[:, 2].min())
        if height < float(args.min_insulator_height):
            continue
        endpoint_1, endpoint_2 = extract_insulator_instance_endpoints(
            points,
            args.insulator_endpoint_percentile,
        )
        middle_point = (endpoint_1 + endpoint_2) / 2.0
        instances.append(
            {
                "points": points,
                "endpoint_1": endpoint_1,
                "middle_point": middle_point,
                "endpoint_2": endpoint_2,
            }
        )

    # 这里只保证内部遍历稳定，最终编号会在每个横担端点内重新生成。
    instances.sort(
        key=lambda item: (
            float(item["middle_point"][0]),
            float(item["middle_point"][1]),
            -float(item["middle_point"][2]),
        )
    )
    return instances


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
                "crossarms": [],
                "along_axis": np.array([0.0, 1.0], dtype=np.float64),
                "side_axis": np.array([-1.0, 0.0], dtype=np.float64),
            }
        )

    towers.sort(key=lambda tower: (tower["center"][0], tower["center"][1]))
    for tower_id, tower in enumerate(towers, start=1):
        tower["id"] = tower_id
        tower["name"] = f"杆塔{tower_id}"
    return towers


def nearest_actual_point(points, target):
    """将几何端点吸附到最近的真实杆塔点。"""
    index = int(np.argmin(np.sum((points - target[None, :]) ** 2, axis=1)))
    return points[index]


def stable_xy_axis(axis):
    """归一化二维主轴并固定符号，保证左右端点编号稳定。"""
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    axis = axis / norm
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0:
        axis = -axis
    return axis


def measure_tower_slice(tower, center_z, args):
    """测量一个水平杆塔切片，并使用二维 PCA 拟合最宽方向。"""
    tower_points = tower["points"]
    half_window = max(float(args.scan_z_window) / 2.0, 1e-3)
    indices = np.flatnonzero(np.abs(tower_points[:, 2] - float(center_z)) <= half_window)
    if indices.size < int(args.min_crossarm_points):
        return None

    selected = tower_points[indices]
    origin_xy = np.median(selected[:, :2], axis=0)
    centered_xy = selected[:, :2] - origin_xy[None, :]
    covariance = centered_xy.T @ centered_xy
    values, vectors = np.linalg.eigh(covariance)
    side_axis = stable_xy_axis(vectors[:, int(np.argmax(values))])
    along_axis = np.array([-side_axis[1], side_axis[0]], dtype=np.float64)
    side_values = centered_xy @ side_axis
    along_values = centered_xy @ along_axis
    percentile = float(np.clip(args.endpoint_percentile, 0.0, 49.0))
    side_min = float(np.percentile(side_values, percentile))
    side_max = float(np.percentile(side_values, 100.0 - percentile))
    width = side_max - side_min
    if width < float(args.min_crossarm_width):
        return None

    along_center = float(np.median(along_values))
    z_center = float(np.median(selected[:, 2]))
    left_target_xy = origin_xy + along_center * along_axis + side_min * side_axis
    right_target_xy = origin_xy + along_center * along_axis + side_max * side_axis
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
        "scan_center_z": float(center_z),
        "tower_points": int(selected.shape[0]),
        "along_axis": along_axis,
        "side_axis": side_axis,
    }


def extract_crossarms_for_tower(tower, args):
    """仅根据杆塔点的切片宽度峰值，提取上部全部横担。"""
    height_ratio = float(args.crossarm_min_height_ratio)
    if not 0.0 <= height_ratio < 1.0:
        raise ValueError("--crossarm-min-height-ratio 必须位于 0 到 1 之间")
    step = max(float(args.scan_z_step), 1e-3)
    min_separation = max(float(args.crossarm_min_z_separation), step)
    min_prominence = max(float(args.crossarm_min_prominence), 0.0)
    min_relative_prominence = float(args.crossarm_min_relative_prominence)
    if not 0.0 <= min_relative_prominence <= 1.0:
        raise ValueError("--crossarm-min-relative-prominence 必须位于 0 到 1 之间")
    min_width_ratio = float(args.crossarm_min_width_ratio)
    if not 0.0 <= min_width_ratio <= 1.0:
        raise ValueError("--crossarm-min-width-ratio 必须位于 0 到 1 之间")
    tower_min_z = float(tower["bbox_min"][2])
    tower_max_z = float(tower["bbox_max"][2])
    search_min_z = tower_min_z + height_ratio * float(tower["height"])
    centers = np.arange(search_min_z, tower_max_z + step * 0.5, step)
    if centers.size == 0:
        centers = np.asarray([tower_max_z], dtype=np.float64)
    if centers[-1] < tower_max_z - 1e-9:
        centers = np.concatenate((centers, [tower_max_z]))

    measurements = [measure_tower_slice(tower, center_z, args) for center_z in centers]
    widths = np.asarray(
        [0.0 if item is None else float(item["width"]) for item in measurements],
        dtype=np.float64,
    )
    widest_slice = float(widths.max(initial=0.0))
    effective_min_width = max(
        float(args.min_crossarm_width), widest_slice * min_width_ratio
    )
    # 两端补零，使位于塔顶边界的地线横担也能形成宽度峰值。
    padded_widths = np.concatenate(([0.0], widths, [0.0]))
    minimum_peak_distance = max(int(np.ceil(min_separation / step)), 1)
    padded_peaks, peak_properties = find_peaks(
        padded_widths,
        distance=minimum_peak_distance,
        prominence=min_prominence,
    )
    peak_candidates = [
        (int(index - 1), float(prominence))
        for index, prominence in zip(
            padded_peaks, peak_properties.get("prominences", [])
        )
        if 0 < index <= widths.shape[0]
    ]
    # 搜索下边界是人为截取的位置，不将该处的塔身宽度当成横担。
    peak_candidates = [item for item in peak_candidates if item[0] != 0]
    kept = [
        measurements[index]
        for index, prominence in peak_candidates
        if measurements[index] is not None
        and widths[index] >= effective_min_width
        and prominence / max(widths[index], 1e-12) >= min_relative_prominence
    ]

    # 只有完全没有峰值候选时才使用最宽切片兜底。若候选峰因突出度不足被
    # 拒绝，则说明它更像缓慢变宽的塔身，不能在此处重新加入。
    if not kept and not peak_candidates and np.any(widths >= effective_min_width):
        fallback_index = int(np.argmax(widths))
        if measurements[fallback_index] is not None:
            kept = [measurements[fallback_index]]

    kept.sort(key=lambda item: -float(item["center_point"][2]))
    tower["crossarms"] = kept
    tower["crossarm_scan_stats"] = {
        "slices": int(centers.size),
        "peaks": int(len(peak_candidates)),
        "kept": int(len(kept)),
        "search_min_z": float(search_min_z),
        "search_max_z": float(tower_max_z),
        "effective_min_width": float(effective_min_width),
    }
    if kept:
        tower["along_axis"] = kept[0]["along_axis"]
        tower["side_axis"] = kept[0]["side_axis"]
    return tower["along_axis"], tower["side_axis"]


def xyz_list(point):
    return [round(float(value), 6) for value in point]


def initialize_crossarm_endpoints(towers):
    """为每个横担创建左右端点的绝缘子容器。"""
    for tower in towers:
        for crossarm in tower["crossarms"]:
            crossarm["left_insulators"] = []
            crossarm["right_insulators"] = []


def point_to_segment_distance(point, start, end):
    """计算三维点到线段的距离、最近点和线段位置比例。"""
    segment = end - start
    squared_length = float(segment @ segment)
    if squared_length <= 1e-12:
        ratio = 0.0
        nearest = start
    else:
        ratio = float(
            np.clip(((point - start) @ segment) / squared_length, 0.0, 1.0)
        )
        nearest = start + ratio * segment
    return float(np.linalg.norm(point - nearest)), nearest, ratio


def orient_insulator_for_crossarm(instance, crossarm):
    """按绝缘子两个端点到整条横担线段的距离确定挂载端。"""
    endpoint_1 = np.asarray(instance["endpoint_1"], dtype=np.float64)
    endpoint_2 = np.asarray(instance["endpoint_2"], dtype=np.float64)
    left_point = np.asarray(crossarm["left_point"], dtype=np.float64)
    right_point = np.asarray(crossarm["right_point"], dtype=np.float64)
    distance_1, nearest_1, ratio_1 = point_to_segment_distance(
        endpoint_1, left_point, right_point
    )
    distance_2, nearest_2, ratio_2 = point_to_segment_distance(
        endpoint_2, left_point, right_point
    )
    if distance_1 <= distance_2:
        attached_point, free_point = endpoint_1, endpoint_2
        attach_distance = distance_1
        nearest_point = nearest_1
        segment_ratio = ratio_1
    else:
        attached_point, free_point = endpoint_2, endpoint_1
        attach_distance = distance_2
        nearest_point = nearest_2
        segment_ratio = ratio_2

    direction = free_point - attached_point
    length = float(np.linalg.norm(direction))
    downward_ratio = (
        0.0 if length <= 1e-12 else max(0.0, -float(direction[2]) / length)
    )
    return {
        "attached_point": attached_point,
        "middle_point": np.asarray(instance["middle_point"], dtype=np.float64),
        "free_point": free_point,
        "attach_distance": attach_distance,
        "nearest_crossarm_point": nearest_point,
        "segment_ratio": segment_ratio,
        "downward_ratio": downward_ratio,
    }


def associate_insulators_to_crossarm_endpoints(towers, insulator_instances, args):
    """将每串绝缘子唯一分配给三维距离最近的横担线段和左右侧。"""
    attach_radius = float(args.insulator_attach_radius)
    downward_threshold = float(args.downward_vertical_ratio)
    if attach_radius <= 0:
        raise ValueError("--insulator-attach-radius 必须大于 0")
    if not 0.0 <= downward_threshold <= 1.0:
        raise ValueError("--downward-vertical-ratio 必须位于 0 到 1 之间")

    initialize_crossarm_endpoints(towers)
    assigned_count = 0
    for instance in insulator_instances:
        best = None
        for tower in towers:
            for crossarm in tower["crossarms"]:
                along_axis = np.asarray(crossarm["along_axis"], dtype=np.float64)
                side_axis = np.asarray(crossarm["side_axis"], dtype=np.float64)
                center_point = np.asarray(crossarm["center_point"], dtype=np.float64)
                oriented = orient_insulator_for_crossarm(instance, crossarm)
                if oriented["attach_distance"] > attach_radius:
                    continue

                # 绝缘子挂在横担中部或塔身连接处时，用其中点方向消除左右零值歧义。
                side_position = float(
                    (oriented["attached_point"][:2] - center_point[:2]) @ side_axis
                )
                if abs(side_position) <= 1e-6:
                    side_position = float(
                        (oriented["middle_point"][:2] - center_point[:2]) @ side_axis
                    )
                side_name = "left" if side_position <= 0.0 else "right"
                container_key = f"{side_name}_insulators"
                order_key = (
                    oriented["attach_distance"],
                    int(tower["id"]),
                    -float(crossarm["center_point"][2]),
                    0 if side_name == "left" else 1,
                )
                if best is None or order_key < best["order_key"]:
                    sort_along = float(
                        (oriented["middle_point"][:2] - center_point[:2]) @ along_axis
                    )
                    best = {
                        "order_key": order_key,
                        "container": crossarm[container_key],
                        "sort_along": sort_along,
                        **oriented,
                    }

        if best is None:
            continue
        container = best.pop("container")
        best.pop("order_key")
        container.append(best)
        assigned_count += 1

    removed_downward = 0
    for tower in towers:
        for crossarm in tower["crossarms"]:
            for container_key in ("left_insulators", "right_insulators"):
                attached = crossarm[container_key]
                if len(attached) >= 2:
                    kept_attached = [
                        item
                        for item in attached
                        if item["downward_ratio"] < downward_threshold
                    ]
                    removed_downward += len(attached) - len(kept_attached)
                    attached[:] = kept_attached
                attached.sort(
                    key=lambda item: (
                        item["sort_along"],
                        -float(item["middle_point"][2]),
                        float(item["middle_point"][0]),
                        float(item["middle_point"][1]),
                    )
                )

    return {
        "total": len(insulator_instances),
        "assigned": assigned_count,
        "unassigned": len(insulator_instances) - assigned_count,
        "removed_downward": removed_downward,
    }


def serialize_endpoint(point, attached_insulators):
    """将一个横担端点及其有序绝缘子转换为可写入 JSON 的结构。"""
    records = []
    for insulator_id, item in enumerate(attached_insulators, start=1):
        records.append(
            {
                "insulator_id": int(insulator_id),
                "endpoint_1_xyz": xyz_list(item["attached_point"]),
                "middle_point_xyz": xyz_list(item["middle_point"]),
                "endpoint_2_xyz": xyz_list(item["free_point"]),
            }
        )
    return {"point_xyz": xyz_list(point), "insulators": records}


def build_json(towers):
    """构建“杆塔、横担、端点、绝缘子”的嵌套 JSON。"""
    tower_records = []
    for tower in towers:
        crossarm_records = []
        for layer_index, crossarm in enumerate(tower["crossarms"], start=1):
            crossarm_records.append(
                {
                    "crossarm_id": int(layer_index),
                    "left_endpoint": serialize_endpoint(
                        crossarm["left_point"], crossarm["left_insulators"]
                    ),
                    "middle_point_xyz": xyz_list(crossarm["center_point"]),
                    "right_endpoint": serialize_endpoint(
                        crossarm["right_point"], crossarm["right_insulators"]
                    ),
                }
            )
        tower_records.append(
            {
                "tower_id": int(tower["id"]),
                "tower_name": tower["name"],
                "crossarms": crossarm_records,
            }
        )
    return {
        "coordinate_system": "input_global_xyz",
        "towers": tower_records,
    }


def report_counts(report):
    """统计嵌套报告中的杆塔、横担和绝缘子数量。"""
    towers = report.get("towers", [])
    crossarm_count = 0
    insulator_count = 0
    for tower in towers:
        crossarms = tower.get("crossarms", [])
        crossarm_count += len(crossarms)
        for crossarm in crossarms:
            insulator_count += len(crossarm["left_endpoint"].get("insulators", []))
            insulator_count += len(crossarm["right_endpoint"].get("insulators", []))
    return len(towers), crossarm_count, insulator_count


def extract_keypoints_from_arrays(coord, classification, args, verbose=True):
    """直接从内存中的 XYZ 和语义类别提取绝缘子、横担关键点。"""
    coord = np.asarray(coord, dtype=np.float64)
    classification = np.asarray(classification, dtype=np.int32).reshape(-1)
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError("coord 必须是形状为 (N, 3) 的坐标数组。")
    if classification.shape[0] != coord.shape[0]:
        raise ValueError("classification 数量必须与 coord 点数一致。")

    tower_points = coord[classification == int(args.tower_class)]
    line_points = coord[classification == int(args.line_class)]
    insulator_points = coord[classification == int(args.insulator_class)]
    if verbose:
        print(
            f"关键点输入：总点数={coord.shape[0]}，杆塔={tower_points.shape[0]}，"
            f"导线={line_points.shape[0]}，绝缘子={insulator_points.shape[0]}",
            flush=True,
        )

    towers = extract_towers(tower_points, args)
    insulator_instances = extract_insulator_instances(insulator_points, args)
    for tower in towers:
        extract_crossarms_for_tower(tower, args)
        if verbose:
            scan_stats = tower.get("crossarm_scan_stats", {})
            print(
                f"  {tower['name']}：横担扫描切片={scan_stats.get('slices', 0)}，"
                f"宽度峰={scan_stats.get('peaks', 0)}，"
                f"横担={len(tower['crossarms'])}",
                flush=True,
            )
    association_stats = associate_insulators_to_crossarm_endpoints(
        towers, insulator_instances, args
    )
    if verbose:
        print(
            "绝缘子挂载：总实例={total}，已关联={assigned}，未关联={unassigned}，"
            "删除朝下={removed_downward}".format(**association_stats),
            flush=True,
        )
    return build_json(towers)


def write_keypoints_json(report, output_path, overwrite=False):
    """保存统一关键点 JSON。"""
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".json":
        raise ValueError(f"关键点输出必须是 .json 文件：{output_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"输出已存在，请添加 --overwrite：{output_path}")
    if output_path.parent.exists() and not output_path.parent.is_dir():
        raise NotADirectoryError(
            f"关键点输出的父路径是文件而不是目录：{output_path.parent}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")
    if output_path.suffix.lower() != ".json":
        raise ValueError(f"--output 必须是 .json 文件：{output_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在，请添加 --overwrite：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"读取点云：{input_path}", flush=True)
    coord, classification = read_las(input_path)
    report = extract_keypoints_from_arrays(coord, classification, args)
    write_keypoints_json(report, output_path, overwrite=args.overwrite)
    tower_count, crossarm_count, insulator_count = report_counts(report)

    print(
        f"保存完成：{output_path}，杆塔={tower_count}，"
        f"横担={crossarm_count}，绝缘子={insulator_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
