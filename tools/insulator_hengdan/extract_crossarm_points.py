"""基于杆塔、导线、绝缘子分割结果推断横担左右端点。

当前模型没有显式横担类别，本脚本通过几何关系推断：
杆塔实例 + 绝缘子高度层 + 附近导线方向 -> 横担 left/right/center。

输入支持 LAS/LAZ/PLY。LAS 需要 classification 字段，PLY 需要包含 x/y/z
和 classification/class/label 等类别字段。
"""

import argparse
import json
import time
from itertools import product
from pathlib import Path

import laspy
import numpy as np
try:
    from plyfile import PlyData
except ImportError:
    PlyData = None
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer crossarm left/right points from segmented tower/line/insulator LAS/PLY."
    )
    parser.add_argument("--input", required=True, help="输入分割后的 LAS/LAZ/PLY 文件。")
    parser.add_argument(
        "--render-base-las",
        default=None,
        help=(
            "可选：用于绘制每杆塔输出 LAS 的原始底图 LAS/PLY。"
            "横担推断仍然使用 --input 的分割类别，输出底图使用原始点云颜色。"
        ),
    )
    parser.add_argument("--output", required=True, help="输出横担 JSON。")
    parser.add_argument(
        "--tower-las-output-dir",
        default=None,
        help=(
            "可选：按杆塔分别输出 LAS。每个 LAS 包含该杆塔附近的原始点云，"
            "并追加该杆塔横担左端、右端、中心点和连线标记。"
        ),
    )
    parser.add_argument("--tower-class", type=int, default=1, help="杆塔类别。")
    parser.add_argument("--line-class", type=int, default=2, help="导线类别。")
    parser.add_argument("--insulator-class", type=int, default=3, help="绝缘子类别。")
    parser.add_argument(
        "--tower-voxel-size",
        type=float,
        default=0.75,
        help="杆塔体素连通域聚类体素大小，单位米。",
    )
    parser.add_argument(
        "--insulator-voxel-size",
        type=float,
        default=0.25,
        help="绝缘子体素连通域聚类体素大小，单位米。",
    )
    parser.add_argument("--min-tower-points", type=int, default=200)
    parser.add_argument("--min-tower-height", type=float, default=4.0)
    parser.add_argument(
        "--min-insulator-points",
        type=int,
        default=10,
        help="一个绝缘子实例最少点数。分割结果较稀疏时建议 10~20。",
    )
    parser.add_argument("--min-insulator-height", type=float, default=0.2)
    parser.add_argument("--max-insulator-height", type=float, default=20.0)
    parser.add_argument(
        "--tower-bind-xy-margin",
        type=float,
        default=8.0,
        help="绝缘子绑定杆塔时，杆塔包围盒 XY 外扩距离。",
    )
    parser.add_argument(
        "--tower-bind-z-margin",
        type=float,
        default=8.0,
        help="绝缘子绑定杆塔时，杆塔包围盒 Z 外扩距离。",
    )
    parser.add_argument(
        "--line-search-radius",
        type=float,
        default=35.0,
        help="估计线路方向和横担层导线点时，杆塔周围搜索半径。",
    )
    parser.add_argument(
        "--line-z-margin",
        type=float,
        default=3.0,
        help="横担层附近导线点的 Z 搜索范围。",
    )
    parser.add_argument(
        "--crossarm-layer-z-gap",
        type=float,
        default=2.0,
        help="绝缘子按高度分层的最大层内高度差。",
    )
    parser.add_argument(
        "--crossarm-layer-source",
        choices=("tower_width", "line_layer", "insulator"),
        default="tower_width",
        help=(
            "横担层来源。tower_width 沿杆塔高度扫描 local_y/side 宽度峰值；"
            "line_layer 先按导线高度分层，每层导线在同高或上方寻找最大杆塔宽度作为横担；"
            "insulator 使用旧的绝缘子高度分层。"
        ),
    )
    parser.add_argument(
        "--crossarm-scan-z-step",
        type=float,
        default=0.5,
        help="tower_width 模式下，从塔顶到塔底扫描横担层的 Z 步长，单位米。",
    )
    parser.add_argument(
        "--crossarm-scan-z-window",
        type=float,
        default=1.2,
        help="tower_width 模式下，每个扫描高度统计杆塔宽度的 Z 窗口厚度，单位米。",
    )
    parser.add_argument(
        "--crossarm-min-y-span",
        type=float,
        default=2.0,
        help="tower_width 模式下，认为某层是横担候选所需的最小 local_y/side 宽度。",
    )
    parser.add_argument(
        "--crossarm-max-layers",
        type=int,
        default=0,
        help="tower_width 模式下最多保留多少个横担层，0 表示不限制。",
    )
    parser.add_argument(
        "--line-layer-search-above",
        type=float,
        default=4.0,
        help="line_layer 模式下，从导线层高度向上搜索横担最大宽度的距离，单位米。",
    )
    parser.add_argument(
        "--line-layer-search-below",
        type=float,
        default=0.5,
        help="line_layer 模式下，允许从导线层高度向下少量搜索横担的距离，单位米。",
    )
    parser.add_argument(
        "--line-layer-min-points",
        type=int,
        default=100,
        help="line_layer 模式下，一个导线高度层最少需要多少导线点。",
    )
    parser.add_argument(
        "--line-layer-merge-adjacent-count",
        type=int,
        default=2,
        help=(
            "line_layer 模式下，将从上到下相邻的几个原始导线高度簇合并成一个物理横担层。"
            "输电塔常见左右/双束导线会被原始 Z 分层拆成两簇，此时取 2 可得到真实横担层数；"
            "如果你的数据原始高度簇已经等于真实层数，设为 1。"
        ),
    )
    parser.add_argument(
        "--min-line-points-near-crossarm",
        type=int,
        default=30,
        help="每层横担附近最少导线点数。",
    )
    parser.add_argument(
        "--min-line-points-per-crossarm-side",
        type=int,
        default=5,
        help="tower_width 模式下，横担左右两端各自最少需要多少个导线点。",
    )
    parser.add_argument(
        "--crossarm-line-side-margin",
        type=float,
        default=1.5,
        help="tower_width 模式下，判断导线是否连接到横担左右端的 local_y/side 容差，单位米。",
    )
    parser.add_argument(
        "--crossarm-end-line-radius",
        type=float,
        default=1.5,
        help="tower_width 模式下，横担真实左右端点到导线点的最大三维连接距离，单位米。",
    )
    parser.add_argument(
        "--crossarm-end-line-along-margin",
        type=float,
        default=3.0,
        help="tower_width 模式下，端点连接导线在 local_x/along 方向的最大偏差，单位米。",
    )
    parser.add_argument(
        "--crossarm-end-insulator-radius",
        type=float,
        default=2.0,
        help="tower_width 模式下，横担端点到绝缘子点的最大桥接距离，单位米。",
    )
    parser.add_argument(
        "--crossarm-insulator-line-radius",
        type=float,
        default=2.0,
        help="tower_width 模式下，绝缘子点到导线点的最大桥接距离，单位米。",
    )
    parser.add_argument(
        "--min-insulator-bridge-points-per-crossarm-side",
        type=int,
        default=3,
        help="tower_width 模式下，横担每侧通过绝缘子桥接导线所需的最少绝缘子点数。",
    )
    parser.add_argument(
        "--require-insulator-after-first-layer",
        action="store_true",
        default=True,
        help=(
            "杆塔从上往下提取横担时，第一层不要求存在绝缘子；"
            "第二层及以下只有在该高度附近存在绝缘子点时才保留，默认开启。"
        ),
    )
    parser.add_argument(
        "--no-require-insulator-after-first-layer",
        dest="require_insulator_after_first_layer",
        action="store_false",
        help="关闭“第二层及以下必须存在绝缘子”的过滤，恢复所有候选层。",
    )
    parser.add_argument("--left-percentile", type=float, default=2.0)
    parser.add_argument("--right-percentile", type=float, default=98.0)
    parser.add_argument(
        "--crossarm-end-source",
        choices=("connection_box", "insulator_sides", "insulator_line_percentile"),
        default="connection_box",
        help=(
            "横担左右端推断方式。connection_box 按连接区域拟合横担立方体底面端点；"
            "insulator_sides 按杆塔两侧绝缘子组确定端点；"
            "insulator_line_percentile 使用旧逻辑，对绝缘子+导线整体取百分位。"
        ),
    )
    parser.add_argument(
        "--min-insulator-side-distance",
        type=float,
        default=1.0,
        help="按左右绝缘子组推断时，绝缘子中心到杆塔 side=0 的最小距离。",
    )
    parser.add_argument(
        "--crossarm-z-percentile",
        type=float,
        default=90.0,
        help="用绝缘子推断横担高度时使用的 Z 分位数，悬垂绝缘子建议取 90~95。",
    )
    parser.add_argument(
        "--crossarm-box-z-margin",
        type=float,
        default=0.9,
        help="connection_box 模式下，围绕横担高度搜索杆塔点的 Z 半径，单位米。",
    )
    parser.add_argument(
        "--crossarm-box-along-margin",
        type=float,
        default=4.0,
        help="connection_box 模式下，围绕连接区域 along 方向搜索杆塔点的半宽，单位米。",
    )
    parser.add_argument(
        "--crossarm-box-side-margin",
        type=float,
        default=2.0,
        help="connection_box 模式下，围绕连接区域 side 方向外扩搜索杆塔点的距离，单位米。",
    )
    parser.add_argument(
        "--crossarm-box-bottom-percentile",
        type=float,
        default=0.0,
        help="connection_box 模式下，横担底面 Z 使用的低分位数；0 表示直接取该层杆塔点最低点。",
    )
    parser.add_argument(
        "--crossarm-box-use-side-hint",
        action="store_true",
        help="connection_box 模式下，使用连接点 side 范围裁剪横担盒。默认关闭，以便盒子包含该层大部分杆塔点。",
    )
    parser.add_argument(
        "--crossarm-connection-radius",
        type=float,
        default=1.2,
        help="connection_box 模式下，杆塔点到导线小于该距离时视为杆塔-导线连接点。",
    )
    parser.add_argument(
        "--snap-crossarm-to-tower-points",
        action="store_true",
        default=True,
        help="根据绝缘子粗定位后，将横担中心线吸附到附近 class=1 杆塔横担点上，默认开启。",
    )
    parser.add_argument(
        "--no-snap-crossarm-to-tower-points",
        dest="snap_crossarm_to_tower_points",
        action="store_false",
        help="不把横担中心线吸附到杆塔点，保留仅由绝缘子/导线推断的位置。",
    )
    parser.add_argument(
        "--crossarm-tower-z-margin",
        type=float,
        default=0.8,
        help="吸附横担时，在初始横担高度上下搜索 class=1 杆塔点的范围，单位米。",
    )
    parser.add_argument(
        "--crossarm-tower-side-margin",
        type=float,
        default=1.0,
        help="吸附横担时，在初始左右端范围外扩搜索 class=1 杆塔点的宽度，单位米。",
    )
    parser.add_argument(
        "--min-crossarm-tower-points",
        type=int,
        default=50,
        help="吸附横担时，候选 class=1 杆塔点最少数量，不足则保留原推断结果。",
    )
    parser.add_argument(
        "--allow-one-sided-crossarm",
        action="store_true",
        default=True,
        help="如果某层只有单侧绝缘子，允许回退到旧的百分位逻辑。",
    )
    parser.add_argument(
        "--no-allow-one-sided-crossarm",
        dest="allow_one_sided_crossarm",
        action="store_false",
        help="某层必须同时存在左右两侧绝缘子才输出横担。",
    )
    parser.add_argument("--min-crossarm-length", type=float, default=1.0)
    parser.add_argument("--max-crossarm-length", type=float, default=30.0)
    parser.add_argument(
        "--min-height-ratio-on-tower",
        type=float,
        default=0.35,
        help="横担层高度至少位于杆塔高度的这个比例以上。",
    )
    parser.add_argument(
        "--tower-crop-xy-margin",
        type=float,
        default=15.0,
        help="每个杆塔 LAS 裁剪原始点云时，杆塔包围盒 XY 外扩距离。",
    )
    parser.add_argument(
        "--tower-crop-z-margin",
        type=float,
        default=8.0,
        help="每个杆塔 LAS 裁剪原始点云时，杆塔包围盒 Z 外扩距离。",
    )
    parser.add_argument(
        "--save-towers-without-crossarm",
        action="store_true",
        help="按杆塔输出 LAS 时，也保存没有横担结果的杆塔。",
    )
    parser.add_argument(
        "--marker-shape",
        choices=("cross", "cube"),
        default="cube",
        help="横担端点/中心点标记形状。cube 更容易在 CloudCompare 中看见。",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=1.5,
        help="横担端点/中心点标记大小，单位米。",
    )
    parser.add_argument(
        "--marker-step",
        type=float,
        default=0.15,
        help="横担端点/中心点标记采样间隔，单位米。",
    )
    parser.add_argument(
        "--crossarm-line-step",
        type=float,
        default=0.2,
        help="横担左右端连线的采样间隔，单位米。",
    )
    parser.add_argument(
        "--crossarm-box-edge-step",
        type=float,
        default=0.2,
        help="横担立方体边线的采样间隔，单位米。",
    )
    parser.add_argument(
        "--no-draw-crossarm-box",
        dest="draw_crossarm_box",
        action="store_false",
        help="不在可视化 LAS 中追加横担立方体边线。",
    )
    parser.set_defaults(draw_crossarm_box=True)
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出文件。")
    return parser.parse_args()


def to_float_list(values, ndigits=6):
    return [round(float(v), ndigits) for v in values]


def ensure_output_path(path, overwrite):
    path = Path(path)
    if path.suffix.lower() in (".las", ".laz"):
        raise ValueError(f"--output 是 JSON 报告路径，请使用 .json 后缀：{path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists, use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dir(path):
    if path is None:
        return None
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def color_to_uint16(color):
    """把 PLY/LAS 颜色统一成 LAS 使用的 uint16 RGB。"""
    if color is None:
        return None
    color = np.asarray(color)
    if color.size == 0:
        return color.reshape((-1, 3)).astype(np.uint16)
    if color.max(initial=0) <= 255:
        return (color.astype(np.uint16) * 257).astype(np.uint16)
    return color.astype(np.uint16)


def make_las_from_arrays(coord, cls, color=None):
    """把 PLY 读出的数组临时包装成 LasData，复用后续 LAS 可视化写出逻辑。"""
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001], dtype=np.float64)
    if coord.shape[0]:
        header.offsets = np.floor(coord.min(axis=0)).astype(np.float64)
    las = laspy.LasData(header)
    las.x = coord[:, 0]
    las.y = coord[:, 1]
    las.z = coord[:, 2]
    las.classification = cls.astype(np.uint8, copy=False)
    color16 = color_to_uint16(color)
    if color16 is None:
        color16 = np.zeros((coord.shape[0], 3), dtype=np.uint16)
    las.red = color16[:, 0]
    las.green = color16[:, 1]
    las.blue = color16[:, 2]
    return las


def find_ply_class_field(names):
    """自动识别 PLY 中常见的语义类别字段名。"""
    candidates = (
        "classification",
        "class",
        "label",
        "pred",
        "prediction",
        "semantic",
        "semantic_label",
        "scalar_class",
    )
    lower_to_name = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate in lower_to_name:
            return lower_to_name[candidate]
    raise ValueError(
        "PLY 中没有找到类别字段。支持字段名："
        + ", ".join(candidates)
        + f"；当前字段：{list(names)}"
    )


def read_ply_cloud(path):
    """读取带类别字段的 PLY 点云。"""
    if PlyData is None:
        raise ImportError("读取 PLY 需要安装 plyfile：pip install plyfile")
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"PLY 缺少 vertex 元素：{path}")
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    for axis in ("x", "y", "z"):
        if axis not in names:
            raise ValueError(f"PLY vertex 缺少坐标字段 {axis}: {path}")
    coord = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
    class_field = find_ply_class_field(names)
    cls = np.asarray(vertex[class_field], dtype=np.int32).reshape(-1)
    if {"red", "green", "blue"}.issubset(set(names)):
        color = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]])
    elif {"r", "g", "b"}.issubset(set(names)):
        color = np.column_stack([vertex["r"], vertex["g"], vertex["b"]])
    else:
        color = np.zeros((coord.shape[0], 3), dtype=np.uint16)
    las = make_las_from_arrays(coord, cls, color)
    return coord, cls, las


def read_segmented_cloud(path):
    """读取 LAS/LAZ/PLY 分割点云，返回坐标、类别和可用于写标记的 LasData。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".las", ".laz"):
        las = laspy.read(path)
        if "classification" not in set(las.point_format.dimension_names):
            raise ValueError("Input LAS has no classification dimension.")
        coord = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
        cls = np.asarray(las.classification, dtype=np.int32)
        return coord, cls, las
    if suffix == ".ply":
        return read_ply_cloud(path)
    raise ValueError(f"仅支持 .las/.laz/.ply 输入：{path}")


def read_render_cloud(path):
    """读取用于可视化底图的 LAS/PLY，不要求存在 classification 字段。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".las", ".laz"):
        las = laspy.read(path)
        coord = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
        return coord, las
    if suffix == ".ply":
        if PlyData is None:
            raise ImportError("读取 PLY 需要安装 plyfile：pip install plyfile")
        ply = PlyData.read(str(path))
        vertex = ply["vertex"].data
        names = vertex.dtype.names or ()
        coord = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
        cls = (
            np.asarray(vertex[find_ply_class_field(names)], dtype=np.int32).reshape(-1)
            if any(name.lower() in {"classification", "class", "label", "pred", "prediction", "semantic", "semantic_label", "scalar_class"} for name in names)
            else np.zeros(coord.shape[0], dtype=np.int32)
        )
        if {"red", "green", "blue"}.issubset(set(names)):
            color = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]])
        elif {"r", "g", "b"}.issubset(set(names)):
            color = np.column_stack([vertex["r"], vertex["g"], vertex["b"]])
        else:
            color = np.zeros((coord.shape[0], 3), dtype=np.uint16)
        return coord, make_las_from_arrays(coord, cls, color)
    raise ValueError(f"仅支持 .las/.laz/.ply 底图：{path}")


def normalize_xy(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return None
    return vector / norm


def orient_axis(axis, reference=None):
    axis = normalize_xy(axis)
    if axis is None:
        return None
    if reference is not None and np.dot(axis, reference) < 0:
        axis = -axis
    elif reference is None:
        if axis[0] < 0 or (abs(axis[0]) < 1e-8 and axis[1] < 0):
            axis = -axis
    return axis


def pca_xy_direction(points_xy):
    if points_xy.shape[0] < 2:
        return None
    centered = points_xy - points_xy.mean(axis=0, keepdims=True)
    if float(np.linalg.norm(centered)) < 1e-8:
        return None
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return normalize_xy(vh[0])


def voxel_component_labels(points, voxel_size, connectivity=26):
    """用体素连通域给点云实例编号。"""
    if points.shape[0] == 0:
        return np.empty(0, dtype=np.int32)
    if voxel_size <= 0:
        raise ValueError("voxel size must be > 0")

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


def build_instances(points, labels, min_points, min_height, max_height=None):
    instances = []
    for label in sorted(set(labels.tolist())):
        if label < 0:
            continue
        instance_points = points[labels == label]
        if instance_points.shape[0] < min_points:
            continue
        bbox_min = instance_points.min(axis=0)
        bbox_max = instance_points.max(axis=0)
        extent = bbox_max - bbox_min
        height = float(extent[2])
        if height < min_height:
            continue
        if max_height is not None and height > max_height:
            continue
        center = instance_points.mean(axis=0)
        instances.append(
            {
                "points": instance_points,
                "center": center,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "extent": extent,
                "height": height,
                "point_count": int(instance_points.shape[0]),
            }
        )
    return instances


def assign_stable_ids(instances, prefix):
    instances.sort(key=lambda item: (item["center"][0], item["center"][1], -item["center"][2]))
    for idx, item in enumerate(instances, start=1):
        item["id"] = idx
        item["name"] = f"{prefix}{idx}"


def expanded_bbox_contains(point, bbox_min, bbox_max, xy_margin, z_margin):
    return (
        bbox_min[0] - xy_margin <= point[0] <= bbox_max[0] + xy_margin
        and bbox_min[1] - xy_margin <= point[1] <= bbox_max[1] + xy_margin
        and bbox_min[2] - z_margin <= point[2] <= bbox_max[2] + z_margin
    )


def bind_insulators_to_towers(towers, insulators, xy_margin, z_margin):
    for tower in towers:
        tower["insulators"] = []
    if not towers:
        return

    for insulator in insulators:
        center = insulator["center"]
        candidates = []
        for tower in towers:
            if expanded_bbox_contains(
                center,
                tower["bbox_min"],
                tower["bbox_max"],
                xy_margin,
                z_margin,
            ):
                distance = float(np.linalg.norm(center[:2] - tower["center"][:2]))
                candidates.append((distance, tower))
        if not candidates:
            continue
        _, tower = min(candidates, key=lambda item: item[0])
        tower["insulators"].append(insulator)


def global_along_axis(towers, line_coord):
    if len(towers) >= 2:
        centers_xy = np.asarray([tower["center"][:2] for tower in towers])
        axis = pca_xy_direction(centers_xy)
        if axis is not None:
            return orient_axis(axis)
    if line_coord.shape[0] >= 2:
        axis = pca_xy_direction(line_coord[:, :2])
        if axis is not None:
            return orient_axis(axis)
    return np.array([0.0, 1.0], dtype=np.float64)


def estimate_tower_along_axis(tower, towers, line_coord, global_axis, search_radius):
    center_xy = tower["center"][:2]
    if line_coord.shape[0]:
        distance_xy = np.linalg.norm(line_coord[:, :2] - center_xy[None, :], axis=1)
        near_line = line_coord[distance_xy <= search_radius]
        if near_line.shape[0] >= 10:
            axis = pca_xy_direction(near_line[:, :2])
            axis = orient_axis(axis, global_axis)
            if axis is not None:
                return axis, "near_line_pca", int(near_line.shape[0])

    other_centers = [
        other["center"][:2] for other in towers if int(other["id"]) != int(tower["id"])
    ]
    if other_centers:
        vectors = np.asarray(other_centers) - center_xy[None, :]
        nearest = vectors[int(np.argmin(np.linalg.norm(vectors, axis=1)))]
        axis = orient_axis(nearest, global_axis)
        if axis is not None:
            return axis, "nearest_tower", 0

    return global_axis, "global_axis", 0


def split_insulator_layers(insulators, z_gap):
    if not insulators:
        return []
    ordered = sorted(insulators, key=lambda item: item["center"][2], reverse=True)
    layers = []
    current = [ordered[0]]
    current_z = float(ordered[0]["center"][2])
    for insulator in ordered[1:]:
        z = float(insulator["center"][2])
        if abs(current_z - z) <= z_gap:
            current.append(insulator)
            current_z = float(np.median([item["center"][2] for item in current]))
        else:
            layers.append(current)
            current = [insulator]
            current_z = z
    layers.append(current)
    return layers


def insulators_touching_crossarm_layer(tower, layer_z, z_margin):
    """返回在当前横担高度附近实际出现点云的绝缘子实例。

    这里检查绝缘子全部点的 Z 范围，而不是只比较实例中心。悬垂绝缘子的中心
    往往低于横担，但其上端仍然与横担接触，只比较中心会把它错误删除。
    """
    matched = []
    for insulator in tower.get("insulators", []):
        points = np.asarray(insulator.get("points", np.empty((0, 3))), dtype=np.float64)
        if points.shape[0] == 0:
            continue
        if float(np.min(np.abs(points[:, 2] - float(layer_z)))) <= float(z_margin):
            matched.append(insulator)
    return matched


def points_near_layer(line_coord, tower_center, layer_z, radius, z_margin):
    if line_coord.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    distance_xy = np.linalg.norm(line_coord[:, :2] - tower_center[:2][None, :], axis=1)
    mask = (distance_xy <= radius) & (np.abs(line_coord[:, 2] - layer_z) <= z_margin)
    return line_coord[mask]


def split_line_height_layers(line_points, z_gap, min_points):
    """把杆塔附近导线点按高度分层。

    line_layer 模式下，横担层由导线层直接驱动：导线有几层，就尝试生成几层横担。
    """
    if line_points.shape[0] == 0:
        return []
    order = np.argsort(line_points[:, 2])[::-1]
    sorted_points = line_points[order]
    layers = []
    current = [sorted_points[0]]
    current_z_values = [float(sorted_points[0, 2])]
    center_z = current_z_values[0]
    for point in sorted_points[1:]:
        z = float(point[2])
        if abs(center_z - z) <= float(z_gap):
            current.append(point)
            current_z_values.append(z)
            center_z = float(np.median(current_z_values))
        else:
            layer = np.asarray(current, dtype=np.float64)
            if layer.shape[0] >= int(min_points):
                layers.append(layer)
            current = [point]
            current_z_values = [z]
            center_z = z
    layer = np.asarray(current, dtype=np.float64)
    if layer.shape[0] >= int(min_points):
        layers.append(layer)
    return layers


def merge_adjacent_line_layers(line_layers, merge_count):
    """把相邻原始导线高度簇合并为物理导线层。

    原始导线点只按 Z 分层时，左右两侧导线或双束导线经常会被拆成相邻两层。
    横担数量应该跟物理导线层一致，因此这里支持按从上到下的顺序每 N 个高度簇合并一次。
    """
    merge_count = max(int(merge_count), 1)
    if merge_count == 1 or len(line_layers) <= 1:
        return [
            {
                "points": np.asarray(points, dtype=np.float64),
                "raw_layer_count": 1,
                "raw_layer_z": [float(np.median(points[:, 2]))],
            }
            for points in line_layers
        ]

    merged = []
    for start in range(0, len(line_layers), merge_count):
        group = line_layers[start : start + merge_count]
        points = np.concatenate(group, axis=0)
        merged.append(
            {
                "points": points,
                "raw_layer_count": int(len(group)),
                "raw_layer_z": [float(np.median(item[:, 2])) for item in group],
            }
        )
    return merged


def snap_crossarm_to_tower_points(
    tower,
    origin_xy,
    along_axis,
    side_axis,
    left_side,
    right_side,
    crossarm_along,
    crossarm_z,
    args,
):
    """把横担线从绝缘子粗定位位置吸附到真实杆塔点云的横担层中心。

    class=3 绝缘子通常位于横担端部或导线挂点附近，用它直接生成横担线会沿线路方向偏移。
    因此这里在该高度附近搜索 class=1 杆塔点，用这些点重新估计横担的 along、side 和 z。
    """
    tower_points = np.asarray(tower.get("points", np.empty((0, 3))), dtype=np.float64)
    if tower_points.shape[0] == 0:
        return left_side, right_side, crossarm_along, crossarm_z, 0, False

    local_xy = tower_points[:, :2] - origin_xy[None, :]
    local_along = local_xy @ along_axis
    local_side = local_xy @ side_axis
    z = tower_points[:, 2]
    side_min = min(float(left_side), float(right_side)) - float(args.crossarm_tower_side_margin)
    side_max = max(float(left_side), float(right_side)) + float(args.crossarm_tower_side_margin)
    mask = (
        (np.abs(z - float(crossarm_z)) <= float(args.crossarm_tower_z_margin))
        & (local_side >= side_min)
        & (local_side <= side_max)
    )
    count = int(np.sum(mask))
    if count < int(args.min_crossarm_tower_points):
        return left_side, right_side, crossarm_along, crossarm_z, count, False

    selected_along = local_along[mask]
    selected_side = local_side[mask]
    selected_z = z[mask]
    snapped_left = float(np.percentile(selected_side, args.left_percentile))
    snapped_right = float(np.percentile(selected_side, args.right_percentile))
    if snapped_right < snapped_left:
        snapped_left, snapped_right = snapped_right, snapped_left
    if snapped_right - snapped_left < float(args.min_crossarm_length):
        snapped_left = float(left_side)
        snapped_right = float(right_side)

    snapped_along = float(np.median(selected_along))
    snapped_z = float(np.median(selected_z))
    return snapped_left, snapped_right, snapped_along, snapped_z, count, True


def infer_crossarm_box_from_connections(
    tower,
    layer_insulator_points,
    layer_line_points,
    origin_xy,
    along_axis,
    side_axis,
    args,
):
    """根据连接区域拟合横担立方体，并返回底面左右边中点。

    逻辑对应人工观察方式：
    1）先在当前高度层找“杆塔通过绝缘子或直接与导线相连”的连接区域；
    2）再在连接区域附近搜索 class=1 杆塔点，这些点近似构成横担本体；
    3）用这些杆塔点拟合一个轴对齐于 along/side/z 的局部立方体；
    4）取立方体底面左右侧边的中点，作为横担最左和最右点。
    """
    tower_points = np.asarray(tower.get("points", np.empty((0, 3))), dtype=np.float64)
    if tower_points.shape[0] == 0 or layer_insulator_points.shape[0] == 0:
        return None

    insulator_z = float(
        np.percentile(
            layer_insulator_points[:, 2],
            np.clip(float(args.crossarm_z_percentile), 0.0, 100.0),
        )
    )

    tower_local_xy = tower_points[:, :2] - origin_xy[None, :]
    tower_along = tower_local_xy @ along_axis
    tower_side = tower_local_xy @ side_axis
    tower_z = tower_points[:, 2]

    connection_parts = [layer_insulator_points]
    if layer_line_points.shape[0] > 0:
        line_tree = cKDTree(layer_line_points)
        z_mask = np.abs(tower_z - insulator_z) <= float(args.crossarm_box_z_margin)
        z_tower_indices = np.flatnonzero(z_mask)
        if z_tower_indices.size:
            distance, _ = line_tree.query(tower_points[z_tower_indices], k=1, workers=-1)
            contact_indices = z_tower_indices[
                distance <= float(args.crossarm_connection_radius)
            ]
            if contact_indices.size:
                connection_parts.append(tower_points[contact_indices])

    connection_points = np.concatenate(connection_parts, axis=0)
    connection_local_xy = connection_points[:, :2] - origin_xy[None, :]
    connection_along = connection_local_xy @ along_axis
    connection_side = connection_local_xy @ side_axis

    # 连接点只用于确认横担层和左右大致覆盖范围，横担中心必须由杆塔点本体决定。
    z_mask = np.abs(tower_z - insulator_z) <= float(args.crossarm_box_z_margin)
    z_selected = np.flatnonzero(z_mask)
    if z_selected.size < int(args.min_crossarm_tower_points):
        return None

    rough_along_center = float(np.median(tower_along[z_selected]))
    side_min_hint = float(np.percentile(connection_side, args.left_percentile))
    side_max_hint = float(np.percentile(connection_side, args.right_percentile))
    if side_max_hint < side_min_hint:
        side_min_hint, side_max_hint = side_max_hint, side_min_hint

    layer_mask = (
        z_mask
        & (np.abs(tower_along - rough_along_center) <= float(args.crossarm_box_along_margin))
    )
    if args.crossarm_box_use_side_hint:
        # 可选：只在连接点附近拟合。默认关闭，否则容易把横担盒裁得过窄。
        layer_mask &= (
            (tower_side >= side_min_hint - float(args.crossarm_box_side_margin))
            & (tower_side <= side_max_hint + float(args.crossarm_box_side_margin))
        )
    selected = np.flatnonzero(layer_mask)

    # 如果使用了 side 提示且候选点不足，回退到只使用杆塔层自身。
    if selected.size < int(args.min_crossarm_tower_points):
        layer_mask = (
            z_mask
            & (np.abs(tower_along - rough_along_center) <= float(args.crossarm_box_along_margin))
        )
        selected = np.flatnonzero(layer_mask)

    if selected.size < int(args.min_crossarm_tower_points):
        return None

    selected_along = tower_along[selected]
    selected_side = tower_side[selected]
    selected_z = tower_z[selected]

    along_min = float(np.percentile(selected_along, args.left_percentile))
    along_max = float(np.percentile(selected_along, args.right_percentile))
    side_min = float(np.percentile(selected_side, args.left_percentile))
    side_max = float(np.percentile(selected_side, args.right_percentile))
    bottom_z = float(
        np.percentile(
            selected_z,
            np.clip(float(args.crossarm_box_bottom_percentile), 0.0, 50.0),
        )
    )

    if side_max < side_min:
        side_min, side_max = side_max, side_min
    # 端点位于横担底面左右侧边的中点：side 取左右边界，along 取杆塔点本体中心。
    along_mid = float(np.median(selected_along))

    return {
        "left_side": side_min,
        "right_side": side_max,
        "crossarm_along": along_mid,
        "crossarm_z": bottom_z,
        "tower_points": int(selected.size),
        "box": {
            "along_min": round(along_min, 6),
            "along_max": round(along_max, 6),
            "along_center": round(along_mid, 6),
            "rough_along_center": round(rough_along_center, 6),
            "side_min": round(side_min, 6),
            "side_max": round(side_max, 6),
            "z_min": round(float(np.min(selected_z)), 6),
            "z_max": round(float(np.max(selected_z)), 6),
            "bottom_z": round(bottom_z, 6),
        },
    }


def detect_tower_width_layers(tower, line_coord, along_axis, side_axis, args):
    """从杆塔点自身的 local_y/side 宽度峰值中检测横担层。

    local_x 使用线路走向 along_axis，local_y 使用垂直线路方向 side_axis。
    在每个 Z 窗口内统计杆塔点的 side 宽度，宽度峰值就是横担候选层。
    导线点只用于过滤：候选层附近必须有足够导线点，避免把塔身平台误当横担。
    """
    tower_points = np.asarray(tower.get("points", np.empty((0, 3))), dtype=np.float64)
    if tower_points.shape[0] == 0:
        return []

    origin_xy = tower["center"][:2]
    local_xy = tower_points[:, :2] - origin_xy[None, :]
    local_side = local_xy @ side_axis
    z = tower_points[:, 2]

    tower_bottom = float(tower["bbox_min"][2])
    tower_height = max(float(tower["height"]), 1e-6)
    scan_min_z = tower_bottom + tower_height * float(args.min_height_ratio_on_tower)
    scan_max_z = float(tower["bbox_max"][2])
    step = max(float(args.crossarm_scan_z_step), 1e-3)
    half_window = max(float(args.crossarm_scan_z_window) / 2.0, 1e-3)

    candidates = []
    centers = np.arange(scan_max_z, scan_min_z - step * 0.5, -step, dtype=np.float64)
    for center_z in centers:
        mask = np.abs(z - center_z) <= half_window
        point_count = int(np.count_nonzero(mask))
        if point_count < int(args.min_crossarm_tower_points):
            continue
        side_values = local_side[mask]
        side_min = float(np.percentile(side_values, args.left_percentile))
        side_max = float(np.percentile(side_values, args.right_percentile))
        if side_max < side_min:
            side_min, side_max = side_max, side_min
        width = side_max - side_min
        if width < max(float(args.crossarm_min_y_span), float(args.min_crossarm_length)):
            continue

        line_points = points_near_layer(
            line_coord,
            tower["center"],
            float(center_z),
            args.line_search_radius,
            args.line_z_margin,
        )
        if line_points.shape[0] < int(args.min_line_points_near_crossarm):
            continue
        line_local_xy = line_points[:, :2] - origin_xy[None, :]
        line_side = line_local_xy @ side_axis
        side_margin = float(args.crossarm_line_side_margin)
        left_line_count = int(np.count_nonzero(line_side <= side_min + side_margin))
        right_line_count = int(np.count_nonzero(line_side >= side_max - side_margin))
        if (
            left_line_count < int(args.min_line_points_per_crossarm_side)
            or right_line_count < int(args.min_line_points_per_crossarm_side)
        ):
            continue

        candidates.append(
            {
                "layer_z": float(center_z),
                "side_width": float(width),
                "tower_points": point_count,
                "line_points": int(line_points.shape[0]),
                "left_line_points": int(left_line_count),
                "right_line_points": int(right_line_count),
            }
        )

    if not candidates:
        return []

    # 同一根横担会在相邻多个 Z 窗口里命中，只保留宽度最大的那个窗口。
    selected = []
    for candidate in sorted(candidates, key=lambda item: item["side_width"], reverse=True):
        if any(
            abs(candidate["layer_z"] - item["layer_z"])
            <= float(args.crossarm_layer_z_gap)
            for item in selected
        ):
            continue
        selected.append(candidate)
        if int(args.crossarm_max_layers) > 0 and len(selected) >= int(args.crossarm_max_layers):
            break

    selected.sort(key=lambda item: item["layer_z"], reverse=True)
    return selected


def detect_line_driven_crossarm_layers(tower, line_coord, along_axis, side_axis, args):
    """以导线高度层为入口，为每层导线寻找对应横担。

    逻辑：
    1. 先取杆塔附近的导线点，按 Z 高度分层；
    2. 导线层按从上到下排序，最终横担数量与导线层数量一致；
    3. 第 N 层只在“当前导线最高点到上一层横担框最低 Z”之间向上搜索；
    4. 完整扫描该区间，选择 local_y/side 宽度最大的高度作为横担层；
    5. 如果该区间没有足够杆塔点，则用当前导线最高点兜底，保证不丢层。
    """
    tower_points = np.asarray(tower.get("points", np.empty((0, 3))), dtype=np.float64)
    if tower_points.shape[0] == 0 or line_coord.shape[0] == 0:
        return []

    origin_xy = tower["center"][:2]
    tower_local_xy = tower_points[:, :2] - origin_xy[None, :]
    tower_along = tower_local_xy @ along_axis
    tower_side = tower_local_xy @ side_axis
    tower_z = tower_points[:, 2]

    distance_xy = np.linalg.norm(line_coord[:, :2] - origin_xy[None, :], axis=1)
    tower_bottom = float(tower["bbox_min"][2])
    tower_height = max(float(tower["height"]), 1e-6)
    min_z = tower_bottom + tower_height * float(args.min_height_ratio_on_tower)
    max_z = float(tower["bbox_max"][2]) + float(args.tower_bind_z_margin)
    near_line_mask = (
        (distance_xy <= float(args.line_search_radius))
        & (line_coord[:, 2] >= min_z)
        & (line_coord[:, 2] <= max_z)
    )
    near_line = line_coord[near_line_mask]
    raw_line_layers = split_line_height_layers(
        near_line,
        args.crossarm_layer_z_gap,
        args.line_layer_min_points,
    )
    if not raw_line_layers:
        return []
    line_layers = merge_adjacent_line_layers(
        raw_line_layers,
        args.line_layer_merge_adjacent_count,
    )

    step = max(float(args.crossarm_scan_z_step), 1e-3)
    half_window = max(float(args.crossarm_scan_z_window) / 2.0, 1e-3)
    tower_top = float(tower["bbox_max"][2])
    selected = []
    line_layer_infos = [
        {
            "rank": idx,
            "points": item["points"],
            "z": float(np.median(item["points"][:, 2])),
            "raw_layer_count": int(item["raw_layer_count"]),
            "raw_layer_z": item["raw_layer_z"],
        }
        for idx, item in enumerate(line_layers, start=1)
    ]
    for layer_idx, layer_info in enumerate(line_layer_infos, start=1):
        layer_points = layer_info["points"]
        line_z = float(np.median(layer_points[:, 2]))
        line_high_z = float(np.max(layer_points[:, 2]))
        line_local_xy = layer_points[:, :2] - origin_xy[None, :]
        line_along_values = line_local_xy @ along_axis
        line_side_values = line_local_xy @ side_axis
        line_side_min = float(np.percentile(line_side_values, args.left_percentile))
        line_side_max = float(np.percentile(line_side_values, args.right_percentile))
        if line_side_max < line_side_min:
            line_side_min, line_side_max = line_side_max, line_side_min
        line_along_mid = float(np.median(line_along_values))
        previous_crossarm_min_z = None
        if selected:
            previous_crossarm_min_z = selected[-1].get("estimated_crossarm_min_z")
        search_min_z = max(tower_bottom, line_high_z)
        search_max_z = (
            tower_top
            if previous_crossarm_min_z is None
            else min(tower_top, float(previous_crossarm_min_z))
        )
        if search_max_z < search_min_z:
            search_max_z = search_min_z

        best = None
        first_hit = None
        scan_centers = np.arange(
            search_min_z,
            search_max_z + step * 0.5,
            step,
            dtype=np.float64,
        )
        scan_centers = scan_centers[scan_centers <= search_max_z + 1e-9]
        if scan_centers.size == 0:
            scan_centers = np.asarray([search_min_z], dtype=np.float64)
        elif scan_centers[-1] < search_max_z - 1e-9:
            scan_centers = np.concatenate(
                [scan_centers, np.asarray([search_max_z], dtype=np.float64)]
            )
        for center_z in scan_centers:
            z_mask = np.abs(tower_z - center_z) <= half_window
            z_selected = np.flatnonzero(z_mask)
            if z_selected.size < int(args.min_crossarm_tower_points):
                continue

            along_center = float(np.median(tower_along[z_selected]))
            local_mask = (
                z_mask
                & (
                    np.abs(tower_along - along_center)
                    <= float(args.crossarm_box_along_margin)
                )
            )
            selected_ids = np.flatnonzero(local_mask)
            if selected_ids.size < int(args.min_crossarm_tower_points):
                selected_ids = z_selected

            side_values = tower_side[selected_ids]
            side_min = float(np.percentile(side_values, args.left_percentile))
            side_max = float(np.percentile(side_values, args.right_percentile))
            if side_max < side_min:
                side_min, side_max = side_max, side_min
            width = side_max - side_min

            item = {
                "layer_z": float(center_z),
                "line_layer_z": line_z,
                "line_layer_rank": int(layer_idx),
                "side_width": float(width),
                "tower_points": int(selected_ids.size),
                "line_points": int(layer_points.shape[0]),
                "left_line_points": 0,
                "right_line_points": 0,
                "line_layer_driven": True,
                "search_min_z": float(search_min_z),
                "search_max_z": float(search_max_z),
                "line_layer_high_z": float(line_high_z),
                "raw_line_layer_count": int(layer_info["raw_layer_count"]),
                "raw_line_layer_z": [float(v) for v in layer_info["raw_layer_z"]],
                "previous_line_layer_z": None
                if layer_idx == 1
                else float(line_layer_infos[layer_idx - 2]["z"]),
                "previous_crossarm_min_z": None
                if previous_crossarm_min_z is None
                else float(previous_crossarm_min_z),
                "line_side_min": float(line_side_min),
                "line_side_max": float(line_side_max),
                "line_along_center": float(line_along_mid),
                "first_hit": False,
                "fallback_max_width": False,
            }
            if best is None or item["side_width"] > best["side_width"]:
                best = item
            if width >= float(args.crossarm_min_y_span):
                if first_hit is None:
                    first_hit = item

        if best is not None:
            # 在当前导线最高点到上一层横担底面之间完整扫描，取最宽的地方作为横担。
            best["first_hit"] = bool(first_hit is not None)
            best["first_hit_layer_z"] = (
                None if first_hit is None else float(first_hit["layer_z"])
            )
            best["fallback_max_width"] = bool(
                best["side_width"] < float(args.crossarm_min_y_span)
            )
            chosen = best
        else:
            # 极端情况下该区间没有足够杆塔点，用当前导线高度作为兜底层位。
            chosen = {
                "layer_z": float(search_min_z),
                "line_layer_z": line_z,
                "line_layer_rank": int(layer_idx),
                "side_width": 0.0,
                "tower_points": 0,
                "line_points": int(layer_points.shape[0]),
                "left_line_points": 0,
                "right_line_points": 0,
                "line_layer_driven": True,
                "search_min_z": float(search_min_z),
                "search_max_z": float(search_max_z),
                "line_layer_high_z": float(line_high_z),
                "raw_line_layer_count": int(layer_info["raw_layer_count"]),
                "raw_line_layer_z": [float(v) for v in layer_info["raw_layer_z"]],
                "previous_line_layer_z": None
                if layer_idx == 1
                else float(line_layer_infos[layer_idx - 2]["z"]),
                "previous_crossarm_min_z": None
                if previous_crossarm_min_z is None
                else float(previous_crossarm_min_z),
                "line_side_min": float(line_side_min),
                "line_side_max": float(line_side_max),
                "line_along_center": float(line_along_mid),
                "first_hit": False,
                "fallback_max_width": True,
                "fallback_empty_interval": True,
            }

        # 下一层的搜索上界使用“上一层横担框最低 Z”，而不是上一层导线高度。
        box_result = infer_crossarm_box_from_tower_width_layer(
            tower,
            chosen,
            origin_xy,
            along_axis,
            side_axis,
            args,
        )
        if box_result is not None and box_result.get("box") is not None:
            chosen["estimated_crossarm_min_z"] = float(
                box_result["box"].get("bottom_z", box_result["box"]["z_min"])
            )
        else:
            chosen["estimated_crossarm_min_z"] = float(
                chosen["layer_z"] - float(args.crossarm_box_z_margin)
            )
        selected.append(chosen)

    selected.sort(key=lambda item: item["line_layer_z"], reverse=True)
    if int(args.crossarm_max_layers) > 0:
        selected = selected[: int(args.crossarm_max_layers)]
    return selected


def infer_crossarm_box_from_tower_width_layer(
    tower,
    layer_info,
    origin_xy,
    along_axis,
    side_axis,
    args,
):
    """用 tower_width 检测出的横担层拟合横担立方体。"""
    tower_points = np.asarray(tower.get("points", np.empty((0, 3))), dtype=np.float64)
    if tower_points.shape[0] == 0:
        return None

    layer_z = float(layer_info["layer_z"])
    tower_local_xy = tower_points[:, :2] - origin_xy[None, :]
    tower_along = tower_local_xy @ along_axis
    tower_side = tower_local_xy @ side_axis
    tower_z = tower_points[:, 2]

    z_mask = np.abs(tower_z - layer_z) <= float(args.crossarm_box_z_margin)
    z_selected = np.flatnonzero(z_mask)
    if z_selected.size < int(args.min_crossarm_tower_points):
        return None

    along_center = float(np.median(tower_along[z_selected]))
    layer_mask = (
        z_mask
        & (np.abs(tower_along - along_center) <= float(args.crossarm_box_along_margin))
    )
    selected = np.flatnonzero(layer_mask)
    if selected.size < int(args.min_crossarm_tower_points):
        selected = z_selected

    selected_along = tower_along[selected]
    selected_side = tower_side[selected]
    selected_z = tower_z[selected]

    along_min = float(np.percentile(selected_along, args.left_percentile))
    along_max = float(np.percentile(selected_along, args.right_percentile))
    side_min = float(np.percentile(selected_side, args.left_percentile))
    side_max = float(np.percentile(selected_side, args.right_percentile))
    if side_max < side_min:
        side_min, side_max = side_max, side_min
    bottom_z = float(
        np.percentile(
            selected_z,
            np.clip(float(args.crossarm_box_bottom_percentile), 0.0, 50.0),
        )
    )
    along_mid = float(np.median(selected_along))
    target_left = np.array([along_mid, side_min, bottom_z], dtype=np.float64)
    target_right = np.array([along_mid, side_max, bottom_z], dtype=np.float64)
    local_selected = np.column_stack([selected_along, selected_side, selected_z])
    left_local_id = int(np.argmin(np.linalg.norm(local_selected - target_left[None, :], axis=1)))
    right_local_id = int(np.argmin(np.linalg.norm(local_selected - target_right[None, :], axis=1)))
    left_point_xyz = tower_points[selected[left_local_id]]
    right_point_xyz = tower_points[selected[right_local_id]]

    return {
        "left_side": side_min,
        "right_side": side_max,
        "crossarm_along": along_mid,
        "crossarm_z": bottom_z,
        "left_point_xyz": left_point_xyz,
        "right_point_xyz": right_point_xyz,
        "tower_points": int(selected.size),
        "box": {
            "along_min": round(along_min, 6),
            "along_max": round(along_max, 6),
            "along_center": round(along_mid, 6),
            "side_min": round(side_min, 6),
            "side_max": round(side_max, 6),
            "z_min": round(float(np.min(selected_z)), 6),
            "z_max": round(float(np.max(selected_z)), 6),
            "bottom_z": round(bottom_z, 6),
            "detected_layer_z": round(layer_z, 6),
            "detected_side_width": round(float(layer_info["side_width"]), 6),
        },
    }


def count_line_points_near_crossarm_endpoints(
    line_coord,
    insulator_coord,
    left_point,
    right_point,
    origin_xy,
    along_axis,
    side_axis,
    layer_z,
    args,
):
    """检查横担左右真实端点是否分别与导线连接。

    连接可以是两种：
    1）横担端点附近直接有导线点；
    2）横担端点附近有绝缘子点，并且这些绝缘子点附近有导线点。
    """
    if line_coord.shape[0] == 0:
        return {
            "left": {"direct_line_points": 0, "bridge_insulator_points": 0, "connected_points": 0},
            "right": {"direct_line_points": 0, "bridge_insulator_points": 0, "connected_points": 0},
        }

    line_points = points_near_layer(
        line_coord,
        np.array([origin_xy[0], origin_xy[1], layer_z], dtype=np.float64),
        float(layer_z),
        args.line_search_radius,
        args.line_z_margin,
    )
    if line_points.shape[0] == 0:
        return {
            "left": {"direct_line_points": 0, "bridge_insulator_points": 0, "connected_points": 0},
            "right": {"direct_line_points": 0, "bridge_insulator_points": 0, "connected_points": 0},
        }

    line_local_xy = line_points[:, :2] - origin_xy[None, :]
    line_along = line_local_xy @ along_axis
    line_side = line_local_xy @ side_axis
    line_tree = cKDTree(line_points)
    insulator_points = np.asarray(insulator_coord, dtype=np.float64)

    def endpoint_evidence(endpoint):
        endpoint = np.asarray(endpoint, dtype=np.float64)
        endpoint_local_xy = endpoint[:2] - origin_xy
        endpoint_along = float(endpoint_local_xy @ along_axis)
        endpoint_side = float(endpoint_local_xy @ side_axis)
        distance = np.linalg.norm(line_points - endpoint[None, :], axis=1)
        mask = (
            (distance <= float(args.crossarm_end_line_radius))
            & (np.abs(line_along - endpoint_along) <= float(args.crossarm_end_line_along_margin))
            & (np.abs(line_side - endpoint_side) <= float(args.crossarm_line_side_margin))
        )
        direct_line_count = int(np.count_nonzero(mask))

        bridge_count = 0
        if insulator_points.shape[0] > 0:
            ins_local_xy = insulator_points[:, :2] - origin_xy[None, :]
            ins_along = ins_local_xy @ along_axis
            ins_side = ins_local_xy @ side_axis
            endpoint_to_ins = np.linalg.norm(insulator_points - endpoint[None, :], axis=1)
            near_endpoint = (
                (endpoint_to_ins <= float(args.crossarm_end_insulator_radius))
                & (np.abs(ins_along - endpoint_along) <= float(args.crossarm_end_line_along_margin))
                & (np.abs(ins_side - endpoint_side) <= float(args.crossarm_line_side_margin))
            )
            near_insulator = insulator_points[near_endpoint]
            if near_insulator.shape[0] > 0:
                ins_line_dist, _ = line_tree.query(near_insulator, k=1, workers=-1)
                bridge_count = int(
                    np.count_nonzero(
                        ins_line_dist <= float(args.crossarm_insulator_line_radius)
                    )
                )

        return {
            "direct_line_points": int(direct_line_count),
            "bridge_insulator_points": int(bridge_count),
            "connected_points": int(max(direct_line_count, bridge_count)),
        }

    return {
        "left": endpoint_evidence(left_point),
        "right": endpoint_evidence(right_point),
    }


def marker_cross_points(center, size, step):
    """围绕关键点生成 3D 十字标记。"""
    half = max(float(size) / 2.0, 0.0)
    step = max(float(step), 1e-3)
    values = np.arange(-half, half + step * 0.5, step, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    points = []
    for axis in range(3):
        offsets = np.zeros((values.shape[0], 3), dtype=np.float64)
        offsets[:, axis] = values
        points.append(center[None, :] + offsets)
    return np.concatenate(points, axis=0)


def marker_cube_points(center, size, step):
    """围绕关键点生成小立方体表面点簇，便于 CloudCompare 中观察。"""
    half = max(float(size) / 2.0, 0.0)
    step = max(float(step), 1e-3)
    values = np.arange(-half, half + step * 0.5, step, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    points = []
    for fixed_axis in range(3):
        axes = [axis for axis in range(3) if axis != fixed_axis]
        grid_a, grid_b = np.meshgrid(values, values, indexing="ij")
        for fixed_value in (-half, half):
            offsets = np.zeros((grid_a.size, 3), dtype=np.float64)
            offsets[:, fixed_axis] = fixed_value
            offsets[:, axes[0]] = grid_a.ravel()
            offsets[:, axes[1]] = grid_b.ravel()
            points.append(center[None, :] + offsets)
    return np.concatenate(points, axis=0)


def marker_points(center, size, step, shape):
    if shape == "cube":
        return marker_cube_points(center, size, step)
    return marker_cross_points(center, size, step)


def sample_segment(p0, p1, step):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    length = float(np.linalg.norm(p1 - p0))
    count = max(int(np.ceil(length / max(float(step), 1e-3))) + 1, 2)
    t = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return p0[None, :] * (1.0 - t[:, None]) + p1[None, :] * t[:, None]


def local_crossarm_box_to_global(origin_xy, along_axis, side_axis, box):
    """把局部 along/side/z 横担盒子转换成全局 8 个角点。"""
    if box is None:
        return None
    along_min = float(box["along_min"])
    along_max = float(box["along_max"])
    side_min = float(box["side_min"])
    side_max = float(box["side_max"])
    z_min = float(box.get("bottom_z", box["z_min"]))
    z_max = float(box["z_max"])
    if z_max < z_min:
        z_min, z_max = z_max, z_min

    corners = []
    for z in (z_min, z_max):
        for along in (along_min, along_max):
            for side in (side_min, side_max):
                xy = origin_xy + along * along_axis + side * side_axis
                corners.append([float(xy[0]), float(xy[1]), float(z)])
    return np.asarray(corners, dtype=np.float64)


def box_edge_points(corners, step):
    """根据 8 个角点采样横担盒子的 12 条边。"""
    if corners is None or len(corners) != 8:
        return np.empty((0, 3), dtype=np.float64)
    edges = (
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    return np.concatenate(
        [sample_segment(corners[start], corners[end], step) for start, end in edges],
        axis=0,
    )


def make_empty_like_input_las(input_las):
    """创建空 LAS，但继承原始 LAS 的点格式、scale、offset 和 CRS。"""
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


def build_crossarm_marker_arrays(crossarms, args):
    """将横担左右端、中心点和连线转换成可追加到 LAS 的点。"""
    point_list = []
    class_list = []
    color_list = []
    point_specs = (
        ("left_point_xyz", 23, (65535, 0, 65535)),    # 左端：品红
        ("right_point_xyz", 24, (0, 65535, 65535)),   # 右端：青色
        ("center_xyz", 25, (65535, 65535, 0)),        # 中心：黄色
    )

    for crossarm in crossarms:
        for key, point_class, color in point_specs:
            points = marker_points(
                crossarm[key], args.marker_size, args.marker_step, args.marker_shape
            )
            point_list.append(points)
            class_list.append(np.full(points.shape[0], point_class, dtype=np.uint8))
            color_list.append(
                np.tile(np.asarray(color, dtype=np.uint16), (points.shape[0], 1))
            )

        line_points = sample_segment(
            crossarm["left_point_xyz"],
            crossarm["right_point_xyz"],
            args.crossarm_line_step,
        )
        point_list.append(line_points)
        class_list.append(np.full(line_points.shape[0], 26, dtype=np.uint8))
        color_list.append(
            np.tile(np.asarray((65535, 65535, 0), dtype=np.uint16), (line_points.shape[0], 1))
        )

        if args.draw_crossarm_box:
            box = crossarm.get("crossarm_box")
            if box and box.get("corners_xyz"):
                corners = np.asarray(box["corners_xyz"], dtype=np.float64)
                edge_points = box_edge_points(corners, args.crossarm_box_edge_step)
                if edge_points.shape[0]:
                    point_list.append(edge_points)
                    class_list.append(np.full(edge_points.shape[0], 27, dtype=np.uint8))
                    color_list.append(
                        np.tile(
                            np.asarray((0, 65535, 0), dtype=np.uint16),
                            (edge_points.shape[0], 1),
                        )
                    )

    if not point_list:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.uint8),
            np.empty((0, 3), dtype=np.uint16),
        )
    return (
        np.concatenate(point_list, axis=0),
        np.concatenate(class_list, axis=0),
        np.concatenate(color_list, axis=0),
    )


def marker_records_like(input_las, points, classes, colors):
    records = laspy.ScaleAwarePointRecord.zeros(points.shape[0], header=input_las.header)
    if points.shape[0] == 0:
        return records
    records.x = points[:, 0]
    records.y = points[:, 1]
    records.z = points[:, 2]
    if "classification" in set(input_las.point_format.dimension_names):
        records.classification = classes
    if {"red", "green", "blue"}.issubset(set(input_las.point_format.dimension_names)):
        records.red = colors[:, 0]
        records.green = colors[:, 1]
        records.blue = colors[:, 2]
    return records


def write_tower_las_files(render_las, render_coord, towers, output_dir, args):
    """按杆塔分别输出局部 LAS，并追加该杆塔横担标记。"""
    if output_dir is None:
        return []

    written = []
    for tower in towers:
        tower_crossarms = tower.get("crossarms", [])
        if not tower_crossarms and not args.save_towers_without_crossarm:
            continue

        bbox_min = tower["bbox_min"]
        bbox_max = tower["bbox_max"]
        crop_mask = (
            (render_coord[:, 0] >= bbox_min[0] - args.tower_crop_xy_margin)
            & (render_coord[:, 0] <= bbox_max[0] + args.tower_crop_xy_margin)
            & (render_coord[:, 1] >= bbox_min[1] - args.tower_crop_xy_margin)
            & (render_coord[:, 1] <= bbox_max[1] + args.tower_crop_xy_margin)
            & (render_coord[:, 2] >= bbox_min[2] - args.tower_crop_z_margin)
            & (render_coord[:, 2] <= bbox_max[2] + args.tower_crop_z_margin)
        )
        crop_indices = np.where(crop_mask)[0]

        marker_points_arr, marker_classes, marker_colors = build_crossarm_marker_arrays(
            tower_crossarms, args
        )
        marker_records = marker_records_like(
            render_las, marker_points_arr, marker_classes, marker_colors
        )

        point_arrays = [render_las.points.array[crop_indices]]
        if marker_records.array.shape[0] > 0:
            point_arrays.append(marker_records.array)

        output_las = make_empty_like_input_las(render_las)
        combined = np.concatenate(point_arrays)
        output_las.points = laspy.ScaleAwarePointRecord(
            combined,
            output_las.header.point_format,
            output_las.header.scales,
            output_las.header.offsets,
        )

        output_path = output_dir / f"tower_{int(tower['id']):03d}_{tower['name']}_crossarm.las"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Tower LAS exists, use --overwrite: {output_path}")
        output_las.write(output_path)
        written.append(
            {
                "tower_id": int(tower["id"]),
                "tower_name": tower["name"],
                "path": str(output_path),
                "crop_points": int(crop_indices.shape[0]),
                "marker_points": int(marker_points_arr.shape[0]),
                "crossarms": int(len(tower_crossarms)),
            }
        )
    return written


def infer_crossarm_from_layer(
    tower,
    layer,
    line_coord,
    along_axis,
    side_axis,
    args,
    crossarm_id,
):
    layer_centers = np.asarray([item["center"] for item in layer])
    layer_z = float(np.median(layer_centers[:, 2]))

    tower_bottom = float(tower["bbox_min"][2])
    tower_height = max(float(tower["height"]), 1e-6)
    height_ratio = (layer_z - tower_bottom) / tower_height
    if height_ratio < float(args.min_height_ratio_on_tower):
        return None

    layer_insulator_points = np.concatenate([item["points"] for item in layer], axis=0)
    layer_line_points = points_near_layer(
        line_coord,
        tower["center"],
        layer_z,
        args.line_search_radius,
        args.line_z_margin,
    )
    if layer_line_points.shape[0] < args.min_line_points_near_crossarm:
        return None

    origin_xy = tower["center"][:2]
    source = str(args.crossarm_end_source)
    left_layer = []
    right_layer = []
    left_side_count = 0
    right_side_count = 0
    box_debug = None
    snap_count = 0
    snapped_to_tower = False

    if args.crossarm_end_source == "insulator_sides":
        center_local_xy = layer_centers[:, :2] - origin_xy[None, :]
        center_side = center_local_xy @ side_axis
        side_distance = float(args.min_insulator_side_distance)
        left_layer = [item for item, side in zip(layer, center_side) if side <= -side_distance]
        right_layer = [item for item, side in zip(layer, center_side) if side >= side_distance]
        left_side_count = len(left_layer)
        right_side_count = len(right_layer)

    box_result = None
    if args.crossarm_end_source == "connection_box":
        center_local_xy = layer_centers[:, :2] - origin_xy[None, :]
        center_side = center_local_xy @ side_axis
        side_distance = float(args.min_insulator_side_distance)
        left_layer = [item for item, side in zip(layer, center_side) if side <= -side_distance]
        right_layer = [item for item, side in zip(layer, center_side) if side >= side_distance]
        left_side_count = len(left_layer)
        right_side_count = len(right_layer)
        box_result = infer_crossarm_box_from_connections(
            tower,
            layer_insulator_points,
            layer_line_points,
            origin_xy,
            along_axis,
            side_axis,
            args,
        )

    use_side_groups = bool(left_layer and right_layer)
    if box_result is not None:
        left_side = float(box_result["left_side"])
        right_side = float(box_result["right_side"])
        crossarm_along = float(box_result["crossarm_along"])
        crossarm_z = float(box_result["crossarm_z"])
        source = "connection_box_bottom_midpoint"
        box_debug = box_result["box"]
        snap_count = int(box_result["tower_points"])
        snapped_to_tower = True
    elif use_side_groups:
        left_points_group = np.concatenate([item["points"] for item in left_layer], axis=0)
        right_points_group = np.concatenate([item["points"] for item in right_layer], axis=0)
        side_source_points = np.concatenate([left_points_group, right_points_group], axis=0)
        left_local_xy = left_points_group[:, :2] - origin_xy[None, :]
        right_local_xy = right_points_group[:, :2] - origin_xy[None, :]
        source_local_xy = side_source_points[:, :2] - origin_xy[None, :]
        left_side_values = left_local_xy @ side_axis
        right_side_values = right_local_xy @ side_axis
        source_along = source_local_xy @ along_axis

        left_side = float(np.percentile(left_side_values, args.left_percentile))
        right_side = float(np.percentile(right_side_values, args.right_percentile))
        crossarm_along = float(np.median(source_along))
        crossarm_z = float(
            np.percentile(
                side_source_points[:, 2],
                np.clip(float(args.crossarm_z_percentile), 0.0, 100.0),
            )
        )
        source = "insulator_side_groups"
    elif args.crossarm_end_source in ("connection_box", "insulator_sides") and not args.allow_one_sided_crossarm:
        return None
    else:
        # 旧逻辑作为回退：对同层绝缘子点和导线点整体取 side 百分位。
        # 这种方式对导线比较敏感，可能把横担拉长，因此默认不优先使用。
        candidate_points = np.concatenate([layer_insulator_points, layer_line_points], axis=0)
        local_xy = candidate_points[:, :2] - origin_xy[None, :]
        local_along = local_xy @ along_axis
        local_side = local_xy @ side_axis
        left_side = float(np.percentile(local_side, args.left_percentile))
        right_side = float(np.percentile(local_side, args.right_percentile))
        crossarm_along = float(np.median(local_along))
        crossarm_z = float(
            np.percentile(
                layer_insulator_points[:, 2],
                np.clip(float(args.crossarm_z_percentile), 0.0, 100.0),
            )
        )
        source = "insulator_line_percentile"

    if right_side < left_side:
        left_side, right_side = right_side, left_side

    if args.snap_crossarm_to_tower_points and box_result is None:
        (
            left_side,
            right_side,
            crossarm_along,
            crossarm_z,
            snap_count,
            snapped_to_tower,
        ) = snap_crossarm_to_tower_points(
            tower,
            origin_xy,
            along_axis,
            side_axis,
            left_side,
            right_side,
            crossarm_along,
            crossarm_z,
            args,
        )

    length = right_side - left_side
    if length < args.min_crossarm_length or length > args.max_crossarm_length:
        return None

    left_xy = origin_xy + crossarm_along * along_axis + left_side * side_axis
    right_xy = origin_xy + crossarm_along * along_axis + right_side * side_axis
    left_point = np.array([left_xy[0], left_xy[1], crossarm_z], dtype=np.float64)
    right_point = np.array([right_xy[0], right_xy[1], crossarm_z], dtype=np.float64)
    center = (left_point + right_point) / 2.0
    crossarm_box = None
    if box_debug is not None:
        box_corners = local_crossarm_box_to_global(
            origin_xy, along_axis, side_axis, box_debug
        )
        crossarm_box = {
            "corners_xyz": [to_float_list(point) for point in box_corners],
            "local": box_debug,
        }

    return {
        "tower_id": int(tower["id"]),
        "tower_name": tower["name"],
        "crossarm_id": int(crossarm_id),
        "crossarm_name": f'{tower["name"]}_横担{crossarm_id}',
        "left_point_xyz": to_float_list(left_point),
        "right_point_xyz": to_float_list(right_point),
        "center_xyz": to_float_list(center),
        "height_z": round(crossarm_z, 6),
        "length": round(float(length), 6),
        "crossarm_box": crossarm_box,
        "direction": {
            "along_axis_xy": to_float_list(along_axis),
            "side_axis_xy": to_float_list(side_axis),
        },
        "support": {
            "insulator_instances": int(len(layer)),
            "left_insulator_instances": int(left_side_count),
            "right_insulator_instances": int(right_side_count),
            "insulator_points": int(layer_insulator_points.shape[0]),
            "line_points": int(layer_line_points.shape[0]),
            "tower_points": int(tower["point_count"]),
            "height_ratio_on_tower": round(float(height_ratio), 6),
        },
        "debug": {
            "local_left_side": round(float(left_side), 6),
            "local_right_side": round(float(right_side), 6),
            "local_along": round(float(crossarm_along), 6),
            "source": source,
            "snapped_to_tower_points": bool(snapped_to_tower),
            "snap_tower_points": int(snap_count),
            "crossarm_box": box_debug,
        },
    }


def infer_crossarm_from_tower_width_layer(
    tower,
    layer_info,
    line_coord,
    along_axis,
    side_axis,
    args,
    crossarm_id,
):
    """根据杆塔 local_y/side 宽度峰值层生成横担记录。"""
    layer_z = float(layer_info["layer_z"])
    tower_bottom = float(tower["bbox_min"][2])
    tower_height = max(float(tower["height"]), 1e-6)
    height_ratio = (layer_z - tower_bottom) / tower_height
    if height_ratio < float(args.min_height_ratio_on_tower):
        return None

    origin_xy = tower["center"][:2]
    box_result = infer_crossarm_box_from_tower_width_layer(
        tower,
        layer_info,
        origin_xy,
        along_axis,
        side_axis,
        args,
    )
    if box_result is None and bool(layer_info.get("line_layer_driven", False)):
        # line_layer 模式要求“导线层数 = 横担层数”。若当前区间内杆塔点不足，
        # 用导线层左右范围兜底生成横担，方便后续人工检查和迭代参数。
        side_min = float(layer_info.get("line_side_min", -float(args.min_crossarm_length) / 2.0))
        side_max = float(layer_info.get("line_side_max", float(args.min_crossarm_length) / 2.0))
        if side_max < side_min:
            side_min, side_max = side_max, side_min
        along_mid = float(layer_info.get("line_along_center", 0.0))
        layer_z = float(layer_info["layer_z"])
        left_xy = origin_xy + along_mid * along_axis + side_min * side_axis
        right_xy = origin_xy + along_mid * along_axis + side_max * side_axis
        box_result = {
            "left_side": side_min,
            "right_side": side_max,
            "crossarm_along": along_mid,
            "crossarm_z": layer_z,
            "left_point_xyz": np.array([left_xy[0], left_xy[1], layer_z], dtype=np.float64),
            "right_point_xyz": np.array([right_xy[0], right_xy[1], layer_z], dtype=np.float64),
            "tower_points": int(layer_info.get("tower_points", 0)),
            "box": {
                "along_min": round(along_mid - float(args.crossarm_box_along_margin), 6),
                "along_max": round(along_mid + float(args.crossarm_box_along_margin), 6),
                "along_center": round(along_mid, 6),
                "side_min": round(side_min, 6),
                "side_max": round(side_max, 6),
                "z_min": round(layer_z - float(args.crossarm_box_z_margin), 6),
                "z_max": round(layer_z + float(args.crossarm_box_z_margin), 6),
                "bottom_z": round(layer_z, 6),
                "detected_layer_z": round(layer_z, 6),
                "detected_side_width": round(float(layer_info.get("side_width", 0.0)), 6),
                "fallback_from_line_layer": True,
            },
        }
    if box_result is None:
        return None

    left_side = float(box_result["left_side"])
    right_side = float(box_result["right_side"])
    if right_side < left_side:
        left_side, right_side = right_side, left_side
    length = right_side - left_side
    if bool(layer_info.get("line_layer_driven", False)) and length < args.min_crossarm_length:
        center_side = (left_side + right_side) / 2.0
        half_length = float(args.min_crossarm_length) / 2.0
        left_side = center_side - half_length
        right_side = center_side + half_length
        length = right_side - left_side
    if length < args.min_crossarm_length or length > args.max_crossarm_length:
        return None

    crossarm_along = float(box_result["crossarm_along"])
    crossarm_z = float(box_result["crossarm_z"])
    left_point = np.asarray(box_result["left_point_xyz"], dtype=np.float64)
    right_point = np.asarray(box_result["right_point_xyz"], dtype=np.float64)
    nearby_insulators = [
        item
        for item in tower.get("insulators", [])
        if abs(float(item["center"][2]) - layer_z) <= float(args.crossarm_layer_z_gap)
    ]
    nearby_insulator_points = (
        np.concatenate([item["points"] for item in nearby_insulators], axis=0)
        if nearby_insulators
        else np.empty((0, 3), dtype=np.float64)
    )
    connection_evidence = count_line_points_near_crossarm_endpoints(
        line_coord,
        nearby_insulator_points,
        left_point,
        right_point,
        origin_xy,
        along_axis,
        side_axis,
        layer_z,
        args,
    )
    left_direct_ok = (
        connection_evidence["left"]["direct_line_points"]
        >= int(args.min_line_points_per_crossarm_side)
    )
    right_direct_ok = (
        connection_evidence["right"]["direct_line_points"]
        >= int(args.min_line_points_per_crossarm_side)
    )
    left_bridge_ok = (
        connection_evidence["left"]["bridge_insulator_points"]
        >= int(args.min_insulator_bridge_points_per_crossarm_side)
    )
    right_bridge_ok = (
        connection_evidence["right"]["bridge_insulator_points"]
        >= int(args.min_insulator_bridge_points_per_crossarm_side)
    )
    if (
        not bool(layer_info.get("line_layer_driven", False))
        and not ((left_direct_ok or left_bridge_ok) and (right_direct_ok or right_bridge_ok))
    ):
        return None
    center = (left_point + right_point) / 2.0

    box_debug = box_result["box"]
    box_corners = local_crossarm_box_to_global(
        origin_xy, along_axis, side_axis, box_debug
    )
    crossarm_box = {
        "corners_xyz": [to_float_list(point) for point in box_corners],
        "local": box_debug,
    }

    return {
        "tower_id": int(tower["id"]),
        "tower_name": tower["name"],
        "crossarm_id": int(crossarm_id),
        "crossarm_name": f'{tower["name"]}_横担{crossarm_id}',
        "left_point_xyz": to_float_list(left_point),
        "right_point_xyz": to_float_list(right_point),
        "center_xyz": to_float_list(center),
        "height_z": round(crossarm_z, 6),
        "length": round(float(length), 6),
        "crossarm_box": crossarm_box,
        "direction": {
            "along_axis_xy": to_float_list(along_axis),
            "side_axis_xy": to_float_list(side_axis),
        },
        "support": {
            "insulator_instances": int(len(nearby_insulators)),
            "left_insulator_instances": 0,
            "right_insulator_instances": 0,
            "insulator_points": int(
                sum(item["point_count"] for item in nearby_insulators)
            ),
            "line_points": int(layer_info["line_points"]),
            "left_line_points": int(connection_evidence["left"]["direct_line_points"]),
            "right_line_points": int(connection_evidence["right"]["direct_line_points"]),
            "left_bridge_insulator_points": int(
                connection_evidence["left"]["bridge_insulator_points"]
            ),
            "right_bridge_insulator_points": int(
                connection_evidence["right"]["bridge_insulator_points"]
            ),
            "tower_points": int(tower["point_count"]),
            "height_ratio_on_tower": round(float(height_ratio), 6),
        },
        "debug": {
            "local_left_side": round(float(left_side), 6),
            "local_right_side": round(float(right_side), 6),
            "local_along": round(float(crossarm_along), 6),
            "source": "tower_width_peak",
            "line_layer_driven": bool(layer_info.get("line_layer_driven", False)),
            "line_layer_z": None
            if "line_layer_z" not in layer_info
            else round(float(layer_info["line_layer_z"]), 6),
            "line_layer_rank": None
            if "line_layer_rank" not in layer_info
            else int(layer_info["line_layer_rank"]),
            "raw_line_layer_count": None
            if "raw_line_layer_count" not in layer_info
            else int(layer_info["raw_line_layer_count"]),
            "raw_line_layer_z": [
                round(float(value), 6)
                for value in layer_info.get("raw_line_layer_z", [])
            ],
            "line_layer_high_z": None
            if "line_layer_high_z" not in layer_info
            else round(float(layer_info["line_layer_high_z"]), 6),
            "previous_line_layer_z": None
            if layer_info.get("previous_line_layer_z") is None
            else round(float(layer_info["previous_line_layer_z"]), 6),
            "previous_crossarm_min_z": None
            if layer_info.get("previous_crossarm_min_z") is None
            else round(float(layer_info["previous_crossarm_min_z"]), 6),
            "estimated_crossarm_min_z": None
            if layer_info.get("estimated_crossarm_min_z") is None
            else round(float(layer_info["estimated_crossarm_min_z"]), 6),
            "search_min_z": None
            if "search_min_z" not in layer_info
            else round(float(layer_info["search_min_z"]), 6),
            "search_max_z": None
            if "search_max_z" not in layer_info
            else round(float(layer_info["search_max_z"]), 6),
            "first_hit": bool(layer_info.get("first_hit", False)),
            "first_hit_layer_z": None
            if layer_info.get("first_hit_layer_z") is None
            else round(float(layer_info["first_hit_layer_z"]), 6),
            "fallback_max_width": bool(layer_info.get("fallback_max_width", False)),
            "fallback_empty_interval": bool(
                layer_info.get("fallback_empty_interval", False)
            ),
            "snapped_to_tower_points": True,
            "snap_tower_points": int(box_result["tower_points"]),
            "crossarm_box": box_debug,
            "detected_layer_z": round(layer_z, 6),
            "detected_side_width": round(float(layer_info["side_width"]), 6),
            "connection_evidence": connection_evidence,
        },
    }


def main():
    args = parse_args()
    start_time = time.perf_counter()
    input_path = Path(args.input)
    output_path = ensure_output_path(args.output, args.overwrite)
    tower_las_output_dir = ensure_output_dir(args.tower_las_output_dir)

    print(f"Reading {input_path}", flush=True)
    coord, cls, las = read_segmented_cloud(input_path)
    render_las = las
    render_coord = coord
    if args.render_base_las is not None:
        render_path = Path(args.render_base_las)
        print(f"Reading render base LAS {render_path}", flush=True)
        render_coord, render_las = read_render_cloud(render_path)

    tower_coord = coord[cls == int(args.tower_class)]
    line_coord = coord[cls == int(args.line_class)]
    insulator_coord = coord[cls == int(args.insulator_class)]

    print(
        f"Loaded {coord.shape[0]} points; "
        f"tower={tower_coord.shape[0]}, line={line_coord.shape[0]}, "
        f"insulator={insulator_coord.shape[0]}",
        flush=True,
    )

    tower_labels = voxel_component_labels(tower_coord, args.tower_voxel_size)
    towers = build_instances(
        tower_coord,
        tower_labels,
        args.min_tower_points,
        args.min_tower_height,
        max_height=None,
    )
    assign_stable_ids(towers, "杆塔")
    for tower in towers:
        tower["crossarms"] = []

    insulator_labels = voxel_component_labels(
        insulator_coord, args.insulator_voxel_size
    )
    insulators = build_instances(
        insulator_coord,
        insulator_labels,
        args.min_insulator_points,
        args.min_insulator_height,
        max_height=args.max_insulator_height,
    )
    assign_stable_ids(insulators, "绝缘子")

    bind_insulators_to_towers(
        towers,
        insulators,
        args.tower_bind_xy_margin,
        args.tower_bind_z_margin,
    )

    global_axis = global_along_axis(towers, line_coord)
    crossarms = []
    tower_reports = []

    for tower in towers:
        along_axis, along_source, near_line_count = estimate_tower_along_axis(
            tower, towers, line_coord, global_axis, args.line_search_radius
        )
        side_axis = np.array([-along_axis[1], along_axis[0]], dtype=np.float64)
        side_axis = normalize_xy(side_axis)

        if args.crossarm_layer_source == "tower_width":
            layers = detect_tower_width_layers(
                tower, line_coord, along_axis, side_axis, args
            )
        elif args.crossarm_layer_source == "line_layer":
            layers = detect_line_driven_crossarm_layers(
                tower, line_coord, along_axis, side_axis, args
            )
        else:
            layers = split_insulator_layers(
                tower.get("insulators", []), args.crossarm_layer_z_gap
            )
        kept_for_tower = 0
        for layer_position, layer in enumerate(layers):
            layer_insulators = []
            if args.crossarm_layer_source in ("tower_width", "line_layer"):
                layer_z = float(layer["layer_z"])
                layer_insulators = insulators_touching_crossarm_layer(
                    tower,
                    layer_z,
                    args.crossarm_layer_z_gap,
                )
                # 最顶层可能是地线横担，本身没有绝缘子，因此始终允许提取。
                # 从第二层开始，只有当前高度确实存在绝缘子点才认为是有效横担。
                if (
                    layer_position > 0
                    and args.require_insulator_after_first_layer
                    and not layer_insulators
                ):
                    continue
            if args.crossarm_layer_source in ("tower_width", "line_layer"):
                record = infer_crossarm_from_tower_width_layer(
                    tower,
                    layer,
                    line_coord,
                    along_axis,
                    side_axis,
                    args,
                    kept_for_tower + 1,
                )
            else:
                record = infer_crossarm_from_layer(
                    tower,
                    layer,
                    line_coord,
                    along_axis,
                    side_axis,
                    args,
                    kept_for_tower + 1,
                )
            if record is None:
                continue
            record["support"]["top_layer_without_insulator_allowed"] = bool(
                layer_position == 0
                and args.crossarm_layer_source in ("tower_width", "line_layer")
            )
            if args.crossarm_layer_source in ("tower_width", "line_layer"):
                record["support"]["insulator_instances_touching_layer"] = int(
                    len(layer_insulators)
                )
            crossarms.append(record)
            tower["crossarms"].append(record)
            kept_for_tower += 1

        tower_reports.append(
            {
                "tower_id": int(tower["id"]),
                "tower_name": tower["name"],
                "point_count": int(tower["point_count"]),
                "center_xyz": to_float_list(tower["center"]),
                "bbox_min_xyz": to_float_list(tower["bbox_min"]),
                "bbox_max_xyz": to_float_list(tower["bbox_max"]),
                "bound_insulators": int(len(tower.get("insulators", []))),
                "layer_source": args.crossarm_layer_source,
                "crossarm_layer_candidates": int(len(layers)),
                "line_layers": int(len(layers))
                if args.crossarm_layer_source == "line_layer"
                else None,
                "raw_line_layers": int(
                    sum(int(item.get("raw_line_layer_count", 1)) for item in layers)
                )
                if args.crossarm_layer_source == "line_layer"
                else None,
                "insulator_layers": int(
                    len(split_insulator_layers(tower.get("insulators", []), args.crossarm_layer_z_gap))
                ),
                "kept_crossarms": int(kept_for_tower),
                "along_source": along_source,
                "near_line_points_for_axis": int(near_line_count),
            }
        )

    tower_las_files = []
    if tower_las_output_dir is not None:
        write_start = time.perf_counter()
        tower_las_files = write_tower_las_files(
            render_las,
            render_coord,
            towers,
            tower_las_output_dir,
            args,
        )
        print(
            f"Wrote {len(tower_las_files)} tower LAS files in "
            f"{time.perf_counter() - write_start:.2f}s",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time
    data = {
        "input": str(input_path),
        "coordinate_system": "las_global_xyz",
        "classes": {
            "tower": int(args.tower_class),
            "line": int(args.line_class),
            "insulator": int(args.insulator_class),
        },
        "parameters": {
            "tower_voxel_size": float(args.tower_voxel_size),
            "insulator_voxel_size": float(args.insulator_voxel_size),
            "tower_bind_xy_margin": float(args.tower_bind_xy_margin),
            "tower_bind_z_margin": float(args.tower_bind_z_margin),
            "line_search_radius": float(args.line_search_radius),
            "line_z_margin": float(args.line_z_margin),
            "crossarm_layer_z_gap": float(args.crossarm_layer_z_gap),
            "crossarm_layer_source": args.crossarm_layer_source,
            "crossarm_scan_z_step": float(args.crossarm_scan_z_step),
            "crossarm_scan_z_window": float(args.crossarm_scan_z_window),
            "crossarm_min_y_span": float(args.crossarm_min_y_span),
            "crossarm_max_layers": int(args.crossarm_max_layers),
            "line_layer_search_above": float(args.line_layer_search_above),
            "line_layer_search_below": float(args.line_layer_search_below),
            "line_layer_min_points": int(args.line_layer_min_points),
            "line_layer_merge_adjacent_count": int(args.line_layer_merge_adjacent_count),
            "min_line_points_near_crossarm": int(args.min_line_points_near_crossarm),
            "min_line_points_per_crossarm_side": int(args.min_line_points_per_crossarm_side),
            "crossarm_line_side_margin": float(args.crossarm_line_side_margin),
            "crossarm_end_line_radius": float(args.crossarm_end_line_radius),
            "crossarm_end_line_along_margin": float(args.crossarm_end_line_along_margin),
            "crossarm_end_insulator_radius": float(args.crossarm_end_insulator_radius),
            "crossarm_insulator_line_radius": float(args.crossarm_insulator_line_radius),
            "min_insulator_bridge_points_per_crossarm_side": int(
                args.min_insulator_bridge_points_per_crossarm_side
            ),
            "require_insulator_after_first_layer": bool(
                args.require_insulator_after_first_layer
            ),
            "left_percentile": float(args.left_percentile),
            "right_percentile": float(args.right_percentile),
            "crossarm_end_source": args.crossarm_end_source,
            "min_insulator_side_distance": float(args.min_insulator_side_distance),
            "allow_one_sided_crossarm": bool(args.allow_one_sided_crossarm),
            "crossarm_z_percentile": float(args.crossarm_z_percentile),
            "crossarm_box_z_margin": float(args.crossarm_box_z_margin),
            "crossarm_box_along_margin": float(args.crossarm_box_along_margin),
            "crossarm_box_side_margin": float(args.crossarm_box_side_margin),
            "crossarm_box_bottom_percentile": float(args.crossarm_box_bottom_percentile),
            "crossarm_box_use_side_hint": bool(args.crossarm_box_use_side_hint),
            "crossarm_connection_radius": float(args.crossarm_connection_radius),
            "snap_crossarm_to_tower_points": bool(args.snap_crossarm_to_tower_points),
            "crossarm_tower_z_margin": float(args.crossarm_tower_z_margin),
            "crossarm_tower_side_margin": float(args.crossarm_tower_side_margin),
            "min_crossarm_tower_points": int(args.min_crossarm_tower_points),
            "min_crossarm_length": float(args.min_crossarm_length),
            "max_crossarm_length": float(args.max_crossarm_length),
            "tower_las_output_dir": None
            if tower_las_output_dir is None
            else str(tower_las_output_dir),
            "render_base_las": args.render_base_las,
            "tower_crop_xy_margin": float(args.tower_crop_xy_margin),
            "tower_crop_z_margin": float(args.tower_crop_z_margin),
            "marker_shape": args.marker_shape,
            "marker_size": float(args.marker_size),
            "marker_step": float(args.marker_step),
            "crossarm_line_step": float(args.crossarm_line_step),
            "draw_crossarm_box": bool(args.draw_crossarm_box),
            "crossarm_box_edge_step": float(args.crossarm_box_edge_step),
        },
        "summary": {
            "total_points": int(coord.shape[0]),
            "tower_points": int(tower_coord.shape[0]),
            "line_points": int(line_coord.shape[0]),
            "insulator_points": int(insulator_coord.shape[0]),
            "towers": int(len(towers)),
            "insulator_instances": int(len(insulators)),
            "crossarms": int(len(crossarms)),
            "tower_las_files": int(len(tower_las_files)),
            "elapsed_sec": round(elapsed, 3),
        },
        "tower_las_files": tower_las_files,
        "towers": tower_reports,
        "crossarms": crossarms,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(
        f"Wrote {output_path} "
        f"(towers={len(towers)}, insulators={len(insulators)}, "
        f"crossarms={len(crossarms)}, elapsed={elapsed:.2f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()

""" 
python tools/infer/extract_crossarm_points.py \
  --input /24085403037/24085403037/shixi/dataset/6_23_demo/test/test_insulator_hengdan/110v12_merged_4cls_Output.las \
  --output/24085403037/24085403037/shixi/dataset/6_23_demo/test/hengdan/tower_004_keypoints.json \
  --tower-las-output-dir /24085403037/24085403037/shixi/dataset/6_23_demo/test/hengdan \
  --overwrite
"""
