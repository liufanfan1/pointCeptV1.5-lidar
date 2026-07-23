"""从分割后的 LAS/PLY 中提取每个绝缘子的最高点、最低点和中心点。

输入 LAS 需要已经把语义分割结果写入 classification 字段；
输入 PLY 需要包含 x/y/z 和 classification/class/label 等类别字段：
0=背景，1=杆塔，2=导线，3=绝缘子。
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
        description="Extract top/bottom/center points for each insulator instance."
    )
    parser.add_argument("--input", required=True, help="输入分割后的 LAS/LAZ/PLY 文件。")
    parser.add_argument(
        "--render-base-las",
        default=None,
        help=(
            "可选：用于绘制输出 LAS 的原始底图 LAS/PLY。"
            "类别提取仍然使用 --input，按杆塔输出/可视化输出使用这个原始点云的颜色。"
        ),
    )
    parser.add_argument("--output", required=True, help="输出绝缘子关键点 JSON。")
    parser.add_argument(
        "--visual-las-output",
        default=None,
        help=(
            "可选：输出用于 CloudCompare 叠加查看的 LAS 标记点。"
            "只保存最高点、最低点、中心点标记，不保存原始点云。"
        ),
    )
    parser.add_argument(
        "--segmented-las-output",
        default=None,
        help=(
            "可选：把输入分割点云另存为 LAS/LAZ。"
            "当 --input 是 PLY 时，这个参数用于生成完整的分割后 LAS。"
        ),
    )
    parser.add_argument(
        "--visual-las-mode",
        choices=("markers-only", "append-to-input"),
        default="markers-only",
        help=(
            "可视化 LAS 输出模式。markers-only 只写关键点标记，文件很小；"
            "append-to-input 会把标记点追加到原始点云后面，文件接近原 LAS 大小。"
        ),
    )
    parser.add_argument(
        "--tower-las-output-dir",
        default=None,
        help=(
            "可选：按杆塔分别输出 LAS。每个 LAS 包含该杆塔附近的原始点云，"
            "并追加该杆塔绝缘子最高点/最低点/中心点标记。"
        ),
    )
    parser.add_argument("--insulator-class", type=int, default=3, help="绝缘子类别。")
    parser.add_argument("--tower-class", type=int, default=1, help="杆塔类别。")
    parser.add_argument("--line-class", type=int, default=2, help="导线类别。")
    parser.add_argument(
        "--use-insulator-class",
        action="store_true",
        default=True,
        help=(
            "使用 class=3 绝缘子点。当前默认开启，适合已经分割出黄色/橙色绝缘子点的结果。"
        ),
    )
    parser.add_argument(
        "--no-use-insulator-class",
        dest="use_insulator_class",
        action="store_false",
        help=(
            "不使用 class=3 绝缘子点，改用导线接入位置推断绝缘子。"
            "仅在绝缘子类别严重漏分或误分时使用。"
        ),
    )
    parser.add_argument(
        "--tower-voxel-size",
        type=float,
        default=0.75,
        help="按杆塔输出 LAS 时，杆塔点体素连通域聚类体素大小。",
    )
    parser.add_argument(
        "--min-tower-points",
        type=int,
        default=200,
        help="按杆塔输出 LAS 时，一个杆塔实例最少杆塔点数。",
    )
    parser.add_argument(
        "--min-tower-height",
        type=float,
        default=4.0,
        help="按杆塔输出 LAS 时，一个杆塔实例最小高度。",
    )
    parser.add_argument(
        "--tower-bind-xy-margin",
        type=float,
        default=8.0,
        help="绝缘子绑定到杆塔时，杆塔包围盒 XY 外扩距离。",
    )
    parser.add_argument(
        "--tower-bind-z-margin",
        type=float,
        default=8.0,
        help="绝缘子绑定到杆塔时，杆塔包围盒 Z 外扩距离。",
    )
    parser.add_argument(
        "--tower-crop-xy-margin",
        type=float,
        default=22.0,
        help="每个杆塔 LAS 裁剪原始点云时，杆塔包围盒 XY 外扩距离。",
    )
    parser.add_argument(
        "--tower-crop-z-margin",
        type=float,
        default=8.0,
        help="每个杆塔 LAS 裁剪原始点云时，杆塔包围盒 Z 外扩距离。",
    )
    parser.add_argument(
        "--save-towers-without-insulator",
        action="store_true",
        help="按杆塔输出 LAS 时，也保存没有绝缘子实例的杆塔。",
    )
    parser.add_argument(
        "--cluster-method",
        choices=(
            "tower_shape",
            "attachment",
            "seed_grow",
            "tower_layer_side",
            "voxel_cc",
            "dbscan",
        ),
        default="tower_layer_side",
        help=(
            "绝缘子实例提取方式。tower_layer_side 直接使用 class=3 绝缘子点，"
            "按杆塔、高度层和左右侧分组，适合当前分割图；attachment 根据杆塔两侧导线接入位置推断绝缘子，"
            "适合绝缘子类别严重漏分的情况；tower_shape 自底向上建立杆塔主体包络，"
            "再提取沿线路 X 轴突出的线性候选；seed_grow 建立线路 X/Y/Z 局部坐标系，"
            "跳过最高导地线层，再从其余层的蓝红接触点沿 X 轴恢复绝缘子；"
            "voxel_cc/dbscan 是普通点云聚类方式。"
        ),
    )
    parser.add_argument(
        "--insulator-layer-z-gap",
        type=float,
        default=2.0,
        help="tower_layer_side 模式下，同一杆塔同一侧绝缘子按高度分层的阈值。",
    )
    parser.add_argument(
        "--min-insulator-side-distance",
        type=float,
        default=1.0,
        help="tower_layer_side 模式下，绝缘子点到杆塔中心 side=0 的最小距离。",
    )
    parser.add_argument(
        "--merge-same-side-layer-insulators",
        action="store_true",
        help=(
            "tower_layer_side 模式下，把同一杆塔、同一侧、同一高度层的多个连通片合并。"
            "默认不合并，优先保留每个 class=3 连通片，避免把多串绝缘子揉成一坨。"
        ),
    )
    parser.add_argument(
        "--axis-line-search-radius",
        type=float,
        default=35.0,
        help="tower_layer_side 模式下，估计线路方向时杆塔周围导线点搜索半径。",
    )
    parser.add_argument("--tower-shape-z-bin", type=float, default=0.50)
    parser.add_argument("--tower-shape-core-quantile", type=float, default=0.65)
    parser.add_argument("--tower-shape-core-margin", type=float, default=0.20)
    parser.add_argument("--tower-shape-center-search-margin", type=float, default=0.80)
    parser.add_argument("--tower-shape-min-z-ratio", type=float, default=0.20)
    parser.add_argument("--tower-shape-cluster-radius", type=float, default=0.25)
    parser.add_argument("--tower-shape-min-points", type=int, default=15)
    parser.add_argument("--tower-shape-min-linearity", type=float, default=0.60)
    parser.add_argument("--tower-shape-min-x-alignment", type=float, default=0.70)
    parser.add_argument("--tower-shape-min-length", type=float, default=0.30)
    parser.add_argument("--tower-shape-max-length", type=float, default=5.00)
    parser.add_argument("--tower-shape-max-width", type=float, default=0.90)
    parser.add_argument("--tower-shape-max-line-distance", type=float, default=0.50)
    parser.add_argument("--tower-shape-layer-z-gap", type=float, default=1.20)
    parser.add_argument("--tower-shape-ground-z-margin", type=float, default=0.60)
    parser.add_argument(
        "--expected-insulated-layers",
        type=int,
        default=0,
        help="预期需要绝缘子的相线层数；0 表示自动检测。",
    )
    parser.add_argument(
        "--insulators-per-line-group",
        type=int,
        default=2,
        help="每个单侧导线组对应的平行绝缘子串数量。",
    )
    parser.add_argument("--line-group-sample-distance", type=float, default=6.0)
    parser.add_argument("--line-group-sample-window", type=float, default=1.5)
    parser.add_argument("--line-group-yz-radius", type=float, default=0.75)
    parser.add_argument("--line-group-min-points", type=int, default=20)
    parser.add_argument("--line-group-corridor-radius", type=float, default=1.20)
    parser.add_argument("--tower-shape-symmetry-x-tolerance", type=float, default=1.00)
    parser.add_argument("--tower-shape-symmetry-y-tolerance", type=float, default=0.80)
    parser.add_argument("--tower-shape-symmetry-z-tolerance", type=float, default=0.80)
    parser.add_argument(
        "--no-tower-shape-require-symmetry",
        action="store_true",
        help="允许只出现杆塔单侧绝缘子候选；默认要求关于局部 Y 轴成对。",
    )
    parser.add_argument(
        "--attachment-min-along-distance",
        type=float,
        default=1.5,
        help="attachment 模式下，导线接入点到杆塔中心沿线路方向的最小距离，避免把塔身中间误当绝缘子。",
    )
    parser.add_argument(
        "--attachment-max-along-distance",
        type=float,
        default=18.0,
        help="attachment 模式下，导线接入点到杆塔中心沿线路方向的最大距离。",
    )
    parser.add_argument(
        "--attachment-side-width",
        type=float,
        default=8.0,
        help="attachment 模式下，沿线路方向两侧候选导线允许的横向宽度。",
    )
    parser.add_argument(
        "--attachment-z-gap",
        type=float,
        default=1.8,
        help="attachment 模式下，导线按高度分层的最大层间间隔。",
    )
    parser.add_argument(
        "--attachment-min-line-points",
        type=int,
        default=80,
        help="attachment 模式下，一个导线接入层最少需要的导线点数。",
    )
    parser.add_argument(
        "--attachment-end-window",
        type=float,
        default=2.5,
        help="attachment 模式下，靠近杆塔端部的导线点窗口大小，用于估计绝缘子位置。",
    )
    parser.add_argument(
        "--attachment-insulator-search-radius",
        type=float,
        default=2.5,
        help="attachment 模式下，在估计接入位置附近吸收黄色绝缘子点的搜索半径。",
    )
    parser.add_argument(
        "--attachment-tower-search-radius",
        type=float,
        default=3.0,
        help="attachment 模式下，在估计接入位置附近吸收被误分为杆塔的绝缘子点的搜索半径。",
    )
    parser.add_argument(
        "--attachment-estimated-height",
        type=float,
        default=2.0,
        help="attachment 模式下，附近黄色点不足时用于生成最高/最低点的估计绝缘子高度。",
    )
    parser.add_argument(
        "--attachment-min-tower-z-ratio",
        type=float,
        default=0.35,
        help="attachment 模式下，只在杆塔高度比例以上寻找绝缘子，过滤地面附近误检。",
    )
    parser.add_argument(
        "--attachment-skip-top-layers",
        type=int,
        default=1,
        help="attachment 模式下，每个杆塔每侧跳过最高的导线层数。你的场景中最顶层通常没有绝缘子，默认跳过 1 层。",
    )
    parser.add_argument(
        "--seed-grow-candidate-radius",
        type=float,
        default=3.0,
        help="seed_grow 模式下，导线端点附近参与生长的候选点半径。",
    )
    parser.add_argument(
        "--seed-grow-endpoint-seed-radius",
        type=float,
        default=1.5,
        help="兼容旧命令保留；当前拓扑模式不依赖黄色绝缘子种子。",
    )
    parser.add_argument(
        "--seed-grow-line-contact-radius",
        type=float,
        default=0.30,
        help=(
            "seed_grow 模式下，非最高导线层与杆塔类点的接触距离。"
            "距离内的红色点视为被误分的绝缘子伪种子。"
        ),
    )
    parser.add_argument(
        "--seed-grow-core-z-window",
        type=float,
        default=1.0,
        help="估计当前层杆塔 X 向结构边界时，接触点上下采用的 Z 窗口。",
    )
    parser.add_argument(
        "--seed-grow-core-x-quantile",
        type=float,
        default=0.65,
        help="用杆塔点到稳健中心的 X 向距离分位数估计杆塔主体半宽。",
    )
    parser.add_argument(
        "--seed-grow-core-x-margin",
        type=float,
        default=0.20,
        help="杆塔 X 向主体半宽的安全外扩距离。",
    )
    parser.add_argument(
        "--seed-grow-core-entry-margin",
        type=float,
        default=0.10,
        help="绝缘子在进入杆塔 X 向边界后允许继续保留的少量连接长度。",
    )
    parser.add_argument(
        "--no-seed-grow-line-contact-seeds",
        action="store_true",
        help="关闭非最高导线层的蓝线-红点直接接触伪种子。",
    )
    parser.add_argument(
        "--seed-grow-neighbor-radius",
        type=float,
        default=0.18,
        help="seed_grow 模式下，区域生长的邻域连接半径。",
    )
    parser.add_argument(
        "--seed-grow-min-seed-points",
        type=int,
        default=3,
        help="seed_grow 模式下，一处蓝线-红点接触至少需要的红色接触点数。",
    )
    parser.add_argument(
        "--seed-grow-min-points",
        type=int,
        default=15,
        help="seed_grow 模式下，恢复出的绝缘子实例最少点数。",
    )
    parser.add_argument(
        "--seed-grow-min-length",
        type=float,
        default=0.3,
        help="seed_grow 模式下，恢复结构沿导线端点到杆塔方向的最小长度。",
    )
    parser.add_argument(
        "--seed-grow-max-length",
        type=float,
        default=3.0,
        help="seed_grow 模式下，恢复结构沿导线端点到杆塔方向的最大长度，超过则认为进入横担/塔身。",
    )
    parser.add_argument(
        "--seed-grow-max-width",
        type=float,
        default=0.65,
        help="seed_grow 模式下，恢复结构相对主方向的最大横向宽度。",
    )
    parser.add_argument(
        "--seed-grow-junction-bin-size",
        type=float,
        default=0.20,
        help="兼容旧命令保留；当前使用杆塔 X 向结构包络面停止。",
    )
    parser.add_argument(
        "--seed-grow-junction-width-ratio",
        type=float,
        default=2.0,
        help="兼容旧命令保留；当前不再用绝缘子局部宽度判断停止。",
    )
    parser.add_argument(
        "--seed-grow-junction-count-ratio",
        type=float,
        default=2.5,
        help="兼容旧命令保留；当前不再用绝缘子局部点数判断停止。",
    )
    parser.add_argument(
        "--seed-grow-backward-margin",
        type=float,
        default=0.25,
        help="seed_grow 模式下，允许候选点位于导线端点外侧的少量距离。",
    )
    parser.add_argument(
        "--seed-grow-min-tower-core-distance",
        type=float,
        default=0.15,
        help=(
            "seed_grow 模式下，杆塔类候选点距离杆塔 XY 中心太近会被视为塔身中心。"
            "该限制不作用于原始绝缘子种子。"
        ),
    )
    parser.add_argument(
        "--corrected-las-output",
        default=None,
        help=(
            "可选：输出把 tower_shape/seed_grow 恢复出的误分杆塔点改成绝缘子类别后的 LAS/LAZ。"
            "不设置时只输出 JSON 和可视化标记。"
        ),
    )
    parser.add_argument(
        "--insulator-visual-radius",
        type=float,
        default=0.35,
        help="黄色点不足时，生成绝缘子单色可视化点簇的半径。",
    )
    parser.add_argument(
        "--insulator-visual-step",
        type=float,
        default=0.12,
        help="黄色点不足时，生成绝缘子单色可视化点簇的采样间隔。",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.25,
        help="体素连通域聚类的体素大小，单位米。",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.8,
        help="DBSCAN 聚类半径，单位米。",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=15,
        help="DBSCAN 的最少邻居点数。",
    )
    parser.add_argument(
        "--min-instance-points",
        type=int,
        default=30,
        help="一个绝缘子实例最少点数。",
    )
    parser.add_argument("--min-height", type=float, default=0.2, help="最小高度。")
    parser.add_argument("--max-height", type=float, default=20.0, help="最大高度。")
    parser.add_argument(
        "--max-extent-xy",
        type=float,
        default=10.0,
        help="XY 方向最大尺寸，超过则认为不是单个绝缘子。",
    )
    parser.add_argument(
        "--require-near-tower",
        action="store_true",
        help="只保留靠近杆塔的绝缘子实例。",
    )
    parser.add_argument(
        "--require-near-line",
        action="store_true",
        help="只保留靠近导线的绝缘子实例。",
    )
    parser.add_argument(
        "--max-distance-to-tower",
        type=float,
        default=8.0,
        help="绝缘子到杆塔的最大允许距离，单位米。",
    )
    parser.add_argument(
        "--max-distance-to-line",
        type=float,
        default=2.0,
        help="绝缘子到导线的最大允许距离，单位米。",
    )
    parser.add_argument(
        "--max-distance-sample-points",
        type=int,
        default=5000,
        help="计算到杆塔/导线最近距离时，每个实例最多采样点数。",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.5,
        help="可视化 LAS 中每个关键点十字标记的边长，单位米。",
    )
    parser.add_argument(
        "--marker-step",
        type=float,
        default=0.1,
        help="可视化 LAS 中十字标记的采样间隔，单位米。",
    )
    parser.add_argument(
        "--marker-shape",
        choices=("cross", "cube"),
        default="cross",
        help="可视化标记形状。cube 更容易在 CloudCompare 中看见。",
    )
    parser.add_argument(
        "--draw-keypoint-markers",
        action="store_true",
        default=True,
        help="绘制最高点/最低点/中心点三类关键点标记，默认开启。",
    )
    parser.add_argument(
        "--no-draw-keypoint-markers",
        action="store_false",
        dest="draw_keypoint_markers",
        help="不绘制最高点/最低点/中心点三类关键点标记。",
    )
    parser.add_argument(
        "--no-draw-insulator-support-points",
        action="store_true",
        help="不绘制单色绝缘子候选点簇，只输出 JSON。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出文件。")
    return parser.parse_args()


def to_float_list(values, ndigits=6):
    return [round(float(v), ndigits) for v in values]


def ensure_output_path(path, overwrite):
    path = Path(path)
    if path.suffix.lower() in (".las", ".laz"):
        raise ValueError(
            f"--output 是 JSON 报告路径，不能使用 LAS 后缀：{path}\n"
            f"请改成 .json；如果需要可视化 LAS，请额外使用 --visual-las-output xxx.las"
        )
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists, use --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_visual_output_path(path, overwrite):
    if path is None:
        return None
    path = Path(path)
    if path.suffix.lower() not in (".las", ".laz"):
        raise ValueError(f"--visual-las-output 应该使用 .las 或 .laz 后缀：{path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Visual output exists, use --overwrite: {path}")
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


def voxel_component_labels(points, voxel_size, connectivity=26):
    """用体素连通域给点云实例编号。"""
    if points.shape[0] == 0:
        return np.empty(0, dtype=np.int32)
    if voxel_size <= 0:
        raise ValueError("--voxel-size must be > 0")

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
        # 只保留一半邻域，避免重复建边。
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
        data = np.ones(len(rows), dtype=np.uint8)
        graph = csr_matrix(
            (data, (rows, cols)),
            shape=(unique_voxels.shape[0], unique_voxels.shape[0]),
        )
        _, voxel_labels = connected_components(graph, directed=False, return_labels=True)
    else:
        voxel_labels = np.arange(unique_voxels.shape[0], dtype=np.int32)

    return voxel_labels[inverse].astype(np.int32, copy=False)


def dbscan_labels(points, eps, min_samples):
    """可选 DBSCAN 聚类；没有 sklearn 时给出明确错误。"""
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as exc:
        raise ImportError(
            "cluster-method=dbscan 需要安装 scikit-learn；"
            "也可以改用默认的 --cluster-method voxel_cc。"
        ) from exc
    cluster = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(points)
    return cluster.labels_.astype(np.int32, copy=False)


def radius_component_labels(points, radius):
    """用固定邻接半径生成连通域，不依赖 sklearn。"""
    if points.shape[0] == 0:
        return np.empty((0,), dtype=np.int32)
    tree = cKDTree(points)
    pairs = tree.query_pairs(max(float(radius), 1e-3), output_type="ndarray")
    if pairs.size == 0:
        return np.arange(points.shape[0], dtype=np.int32)
    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    cols = np.concatenate((pairs[:, 1], pairs[:, 0]))
    graph = csr_matrix(
        (np.ones(rows.size, dtype=np.uint8), (rows, cols)),
        shape=(points.shape[0], points.shape[0]),
    )
    _, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.astype(np.int32, copy=False)


def sampled_points(points, max_points):
    if points.shape[0] <= max_points:
        return points
    sample_index = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
    return points[sample_index]


def nearest_distance(points, tree, max_points):
    if tree is None or points.shape[0] == 0:
        return None
    query_points = sampled_points(points, max_points)
    distance, _ = tree.query(query_points, k=1, workers=-1)
    return float(np.min(distance))


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
    elif axis[0] < 0 or (abs(axis[0]) < 1e-8 and axis[1] < 0):
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


def global_along_axis(towers, line_coord):
    """估计全局线路方向，优先用杆塔中心，其次用导线点。"""
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
    """估计单座杆塔附近的线路方向。"""
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


def build_tower_instances(tower_coord, args):
    """从杆塔语义点中提取物理杆塔实例。"""
    if tower_coord.shape[0] == 0:
        return []
    labels = voxel_component_labels(tower_coord, args.tower_voxel_size)
    towers = []
    for label in sorted(set(labels.tolist())):
        if label < 0:
            continue
        point_ids = np.flatnonzero(labels == label)
        points = tower_coord[point_ids]
        if points.shape[0] < args.min_tower_points:
            continue
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        extent = bbox_max - bbox_min
        if float(extent[2]) < args.min_tower_height:
            continue
        towers.append(
            {
                "points": points,
                "point_ids": point_ids,
                "point_count": int(points.shape[0]),
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "extent": extent,
                "center": points.mean(axis=0),
                "insulators": [],
            }
        )

    towers.sort(key=lambda item: (item["center"][0], item["center"][1], -item["center"][2]))
    for idx, tower in enumerate(towers, start=1):
        tower["id"] = idx
        tower["name"] = f"杆塔{idx}"
    return towers


def expanded_bbox_contains(point, bbox_min, bbox_max, xy_margin, z_margin):
    return (
        bbox_min[0] - xy_margin <= point[0] <= bbox_max[0] + xy_margin
        and bbox_min[1] - xy_margin <= point[1] <= bbox_max[1] + xy_margin
        and bbox_min[2] - z_margin <= point[2] <= bbox_max[2] + z_margin
    )


def assign_instances_to_towers(instances, towers, xy_margin, z_margin):
    """把绝缘子实例绑定到最近的杆塔外扩包围盒。"""
    for tower in towers:
        tower["insulators"] = []
    if not towers:
        for item in instances:
            item["tower_id"] = None
            item["tower_name"] = None
        return

    for item in instances:
        center = np.asarray(item["center_xyz"], dtype=np.float64)
        candidates = []
        for tower in towers:
            if not expanded_bbox_contains(
                center, tower["bbox_min"], tower["bbox_max"], xy_margin, z_margin
            ):
                continue
            distance_xy = float(np.linalg.norm(center[:2] - tower["center"][:2]))
            candidates.append((distance_xy, tower))

        if not candidates:
            item["tower_id"] = None
            item["tower_name"] = None
            continue

        _, tower = min(candidates, key=lambda value: value[0])
        item["tower_id"] = int(tower["id"])
        item["tower_name"] = tower["name"]
        tower["insulators"].append(item)


def split_height_layers(point_indices, points, z_gap):
    """把同一杆塔同一侧的绝缘子点按高度分层。"""
    if point_indices.size == 0:
        return []
    order = np.argsort(points[:, 2])[::-1]
    sorted_indices = point_indices[order]
    sorted_points = points[order]
    layers = []
    current_indices = [sorted_indices[0]]
    current_z_values = [float(sorted_points[0, 2])]
    center_z = current_z_values[0]

    for idx, point in zip(sorted_indices[1:], sorted_points[1:]):
        z = float(point[2])
        if abs(center_z - z) <= float(z_gap):
            current_indices.append(idx)
            current_z_values.append(z)
            center_z = float(np.median(current_z_values))
        else:
            layers.append(np.asarray(current_indices, dtype=np.int64))
            current_indices = [idx]
            current_z_values = [z]
            center_z = z
    layers.append(np.asarray(current_indices, dtype=np.int64))
    return layers


def split_side_component_layers(point_indices, points, args):
    """同一杆塔同一侧内，先按 class=3 连通域切出候选绝缘子。

    默认直接返回每个连通片，适合从分割结果中精细提取每串绝缘子；
    如果开启 --merge-same-side-layer-insulators，再把同高度层连通片合并成一层。
    """
    if point_indices.size == 0:
        return []
    comp_labels = voxel_component_labels(points, args.voxel_size)
    components = []
    for comp_label in sorted(set(comp_labels.tolist())):
        if comp_label < 0:
            continue
        local_mask = comp_labels == comp_label
        local_indices = point_indices[local_mask]
        if local_indices.size < int(args.min_instance_points):
            continue
        comp_points = points[local_mask]
        components.append(
            {
                "indices": local_indices,
                "center_z": float(np.median(comp_points[:, 2])),
            }
        )
    if not components:
        return []

    components.sort(key=lambda item: item["center_z"], reverse=True)
    if not args.merge_same_side_layer_insulators:
        return [item["indices"] for item in components]

    layers = []
    current = [components[0]]
    current_z = float(components[0]["center_z"])
    for comp in components[1:]:
        z = float(comp["center_z"])
        if abs(current_z - z) <= float(args.insulator_layer_z_gap):
            current.append(comp)
            current_z = float(np.median([item["center_z"] for item in current]))
        else:
            layers.append(np.concatenate([item["indices"] for item in current]))
            current = [comp]
            current_z = z
    layers.append(np.concatenate([item["indices"] for item in current]))
    return layers


def tower_layer_side_labels(insulator_coord, tower_coord, line_coord, args):
    """按真实杆塔结构分组绝缘子：杆塔 -> 左右侧 -> 高度层。

    这对应真实杆塔图里的结构：绝缘子挂在横担两边，中间塔身附近通常不是绝缘子。
    """
    labels = np.full(insulator_coord.shape[0], -1, dtype=np.int32)
    towers = build_tower_instances(tower_coord, args)
    if not towers or insulator_coord.shape[0] == 0:
        return labels, towers, []

    best_tower = np.full(insulator_coord.shape[0], -1, dtype=np.int32)
    best_dist = np.full(insulator_coord.shape[0], np.inf, dtype=np.float64)
    for tower_idx, tower in enumerate(towers):
        bbox_min = tower["bbox_min"]
        bbox_max = tower["bbox_max"]
        inside = (
            (insulator_coord[:, 0] >= bbox_min[0] - args.tower_bind_xy_margin)
            & (insulator_coord[:, 0] <= bbox_max[0] + args.tower_bind_xy_margin)
            & (insulator_coord[:, 1] >= bbox_min[1] - args.tower_bind_xy_margin)
            & (insulator_coord[:, 1] <= bbox_max[1] + args.tower_bind_xy_margin)
            & (insulator_coord[:, 2] >= bbox_min[2] - args.tower_bind_z_margin)
            & (insulator_coord[:, 2] <= bbox_max[2] + args.tower_bind_z_margin)
        )
        if not np.any(inside):
            continue
        dist = np.linalg.norm(insulator_coord[:, :2] - tower["center"][:2][None, :], axis=1)
        update = inside & (dist < best_dist)
        best_dist[update] = dist[update]
        best_tower[update] = tower_idx

    global_axis = global_along_axis(towers, line_coord)
    groups = []
    next_label = 0
    for tower_idx, tower in enumerate(towers):
        tower_point_indices = np.where(best_tower == tower_idx)[0]
        if tower_point_indices.size == 0:
            continue

        along_axis, axis_source, near_line_count = estimate_tower_along_axis(
            tower, towers, line_coord, global_axis, args.axis_line_search_radius
        )
        side_axis = np.array([-along_axis[1], along_axis[0]], dtype=np.float64)
        side_axis = normalize_xy(side_axis)
        if side_axis is None:
            continue

        local_xy = insulator_coord[tower_point_indices, :2] - tower["center"][:2][None, :]
        local_side = local_xy @ side_axis
        side_distance = float(args.min_insulator_side_distance)
        side_sets = (
            ("left", tower_point_indices[local_side <= -side_distance]),
            ("right", tower_point_indices[local_side >= side_distance]),
        )

        for side_name, side_indices in side_sets:
            if side_indices.size == 0:
                continue
            layers = split_side_component_layers(
                side_indices,
                insulator_coord[side_indices],
                args,
            )
            for layer_indices in layers:
                if layer_indices.size < int(args.min_instance_points):
                    continue
                labels[layer_indices] = next_label
                groups.append(
                    {
                        "label": int(next_label),
                        "tower_id": int(tower["id"]),
                        "tower_name": tower["name"],
                        "side": side_name,
                        "point_count": int(layer_indices.size),
                        "axis_source": axis_source,
                        "near_line_points_for_axis": int(near_line_count),
                    }
                )
                next_label += 1
    return labels, towers, groups


def split_z_layers_by_indices(local_indices, z_values, z_gap):
    """把候选导线点按高度分层。

    这里不使用连通域，因为输电线路是连续曲线；真正区分绝缘子层位的是高度层。
    """
    if local_indices.size == 0:
        return []
    order = np.argsort(z_values)[::-1]
    sorted_indices = local_indices[order]
    sorted_z = z_values[order]

    layers = []
    current = [int(sorted_indices[0])]
    current_z_values = [float(sorted_z[0])]
    center_z = float(sorted_z[0])
    for idx, z in zip(sorted_indices[1:], sorted_z[1:]):
        z = float(z)
        if abs(center_z - z) <= float(z_gap):
            current.append(int(idx))
            current_z_values.append(z)
            center_z = float(np.median(current_z_values))
        else:
            layers.append(np.asarray(current, dtype=np.int64))
            current = [int(idx)]
            current_z_values = [z]
            center_z = z
    layers.append(np.asarray(current, dtype=np.int64))
    return layers


def nearest_support_index(points, point_indices, target):
    """在支撑点中找离目标坐标最近的原始点索引。"""
    if points.shape[0] == 0 or point_indices.size == 0:
        return None
    distance2 = np.sum((points - target[None, :]) ** 2, axis=1)
    return int(point_indices[int(np.argmin(distance2))])


def generated_insulator_stack_points(center, height, radius, step):
    """生成一个小型竖向点簇，用于表示推断出的绝缘子串位置。

    这不是重新分割点云，只是为了在 CloudCompare 中把候选绝缘子位置用同一种颜色画出来。
    """
    center = np.asarray(center, dtype=np.float64)
    height = max(float(height), 0.1)
    radius = max(float(radius), 0.02)
    step = max(float(step), 0.02)
    z_values = np.arange(-height / 2.0, height / 2.0 + step * 0.5, step)
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    points = []
    for z in z_values:
        points.append(center + np.array([0.0, 0.0, z], dtype=np.float64))
        for angle in angles:
            points.append(
                center
                + np.array(
                    [np.cos(angle) * radius, np.sin(angle) * radius, z],
                    dtype=np.float64,
                )
            )
    return np.asarray(points, dtype=np.float64)


def build_attachment_record(
    instance_id,
    tower,
    side_name,
    layer_rank,
    candidate_center,
    support_points,
    support_indices,
    support_sources,
    line_point_count,
    insulator_point_count,
    tower_point_count,
    args,
    axis_source,
    near_line_count,
):
    """根据导线接入位置生成一个绝缘子记录。

    默认不依赖 class=3 黄色点，而是用导线接入位置附近的 class=1 杆塔点
    作为被误分的绝缘子支撑点，计算最高点、最低点和中心点；
    如果附近 class=1 支撑点也不足，就围绕导线接入中心生成一个估计的上下端点。
    """
    candidate_center = np.asarray(candidate_center, dtype=np.float64)
    support_points = np.asarray(support_points, dtype=np.float64)
    support_indices = np.asarray(support_indices, dtype=np.int64)
    support_sources = np.asarray(support_sources)

    estimated = False
    if support_points.shape[0] > 0:
        bbox_min = support_points.min(axis=0)
        bbox_max = support_points.max(axis=0)
        extent = bbox_max - bbox_min
        top_point = support_points[int(np.argmax(support_points[:, 2]))]
        bottom_point = support_points[int(np.argmin(support_points[:, 2]))]
        center = support_points.mean(axis=0)
        center_point = support_points[
            int(np.argmin(np.sum((support_points - center[None, :]) ** 2, axis=1)))
        ]
    else:
        bbox_min = candidate_center.copy()
        bbox_max = candidate_center.copy()
        extent = np.zeros(3, dtype=np.float64)
        top_point = candidate_center.copy()
        bottom_point = candidate_center.copy()
        center = candidate_center.copy()
        center_point = candidate_center.copy()

    semantic_point_count = int(insulator_point_count) + int(tower_point_count)
    # 当前模型常把绝缘子分成杆塔，所以这里不再只看黄色点数量；
    # class=1 中位于导线接入位置附近的局部点，也作为绝缘子支撑点使用。
    # 如果 class=1/class=3 支撑点仍然太少，才退化成几何估计点簇。
    if (
        semantic_point_count < int(args.min_instance_points)
        or float(top_point[2] - bottom_point[2]) < float(args.min_height)
    ):
        estimated = True
        half_h = max(float(args.attachment_estimated_height) / 2.0, 0.05)
        top_point = candidate_center + np.array([0.0, 0.0, half_h], dtype=np.float64)
        bottom_point = candidate_center - np.array([0.0, 0.0, half_h], dtype=np.float64)
        center = candidate_center.copy()
        center_point = candidate_center.copy()
        bbox_min = np.minimum(bbox_min, bottom_point)
        bbox_max = np.maximum(bbox_max, top_point)
        extent = bbox_max - bbox_min

    semantic_source_mask = (support_sources == "insulator") | (support_sources == "tower")
    if np.any(semantic_source_mask):
        visual_points = support_points[semantic_source_mask]
    else:
        visual_points = generated_insulator_stack_points(
            center,
            args.attachment_estimated_height,
            args.insulator_visual_radius,
            args.insulator_visual_step,
        )

    top_index = nearest_support_index(support_points, support_indices, top_point)
    bottom_index = nearest_support_index(support_points, support_indices, bottom_point)
    center_index = nearest_support_index(support_points, support_indices, center_point)

    return {
        "id": int(instance_id),
        "name": f"绝缘子{instance_id}",
        "tower_id": int(tower["id"]),
        "tower_name": tower["name"],
        "side": side_name,
        "layer_rank": int(layer_rank),
        "method": "line_tower_attachment",
        "estimated_from_line": bool(estimated),
        "point_count": int(support_points.shape[0]),
        "support_line_points": int(line_point_count),
        "support_insulator_points": int(insulator_point_count),
        "support_tower_points": int(tower_point_count),
        "support_source_counts": {
            "line": int(np.sum(support_sources == "line")),
            "tower": int(np.sum(support_sources == "tower")),
            "insulator": int(np.sum(support_sources == "insulator")),
        },
        "top_point_xyz": to_float_list(top_point),
        "bottom_point_xyz": to_float_list(bottom_point),
        "center_xyz": to_float_list(center),
        "center_point_xyz": to_float_list(center_point),
        "bbox_min_xyz": to_float_list(bbox_min),
        "bbox_max_xyz": to_float_list(bbox_max),
        "extent_xyz": to_float_list(extent),
        "height": round(float(top_point[2] - bottom_point[2]), 6),
        "distance_to_tower": None,
        "distance_to_line": None,
        "axis_source": axis_source,
        "near_line_points_for_axis": int(near_line_count),
        "original_point_index_top": top_index,
        "original_point_index_bottom": bottom_index,
        "original_point_index_center": center_index,
        "_visual_points_xyz": visual_points.astype(np.float64, copy=False),
    }


def attachment_based_instances(
    insulator_coord,
    insulator_indices,
    tower_coord,
    tower_indices,
    line_coord,
    line_indices,
    args,
):
    """基于“杆塔两侧导线接入位置”推断绝缘子实例。

    这个逻辑专门处理当前模型的典型问题：
    1. 绝缘子可能被误分成 class=1 杆塔；
    2. class=3 绝缘子点很碎甚至不可用；
    3. 真实绝缘子应在横担/导线接入两侧，而不是塔身中心；
    4. 每层导线在杆塔左/右侧各对应一个绝缘子候选区域。
    """
    towers = build_tower_instances(tower_coord, args)
    for tower in towers:
        tower["insulators"] = []
    if not towers or line_coord.shape[0] == 0:
        return [], towers, []

    global_axis = global_along_axis(towers, line_coord)
    line_xy_tree = cKDTree(line_coord[:, :2])
    insulator_tree = cKDTree(insulator_coord) if insulator_coord.shape[0] else None
    tower_tree = cKDTree(tower_coord) if tower_coord.shape[0] else None

    records = []
    groups = []
    search_radius_xy = float(
        np.hypot(args.attachment_max_along_distance, args.attachment_side_width)
    )

    for tower in towers:
        center = np.asarray(tower["center"], dtype=np.float64)
        height = float(tower["bbox_max"][2] - tower["bbox_min"][2])
        min_z = float(tower["bbox_min"][2] + height * args.attachment_min_tower_z_ratio)
        max_z = float(tower["bbox_max"][2] + args.tower_bind_z_margin)

        along_axis, axis_source, near_line_count = estimate_tower_along_axis(
            tower, towers, line_coord, global_axis, args.axis_line_search_radius
        )
        side_axis = np.array([-along_axis[1], along_axis[0]], dtype=np.float64)
        side_axis = normalize_xy(side_axis)
        if side_axis is None:
            continue

        nearby_line_ids = np.asarray(
            line_xy_tree.query_ball_point(center[:2], r=search_radius_xy),
            dtype=np.int64,
        )
        if nearby_line_ids.size == 0:
            continue
        near_line = line_coord[nearby_line_ids]
        local_xy = near_line[:, :2] - center[:2][None, :]
        local_along = local_xy @ along_axis
        local_side = local_xy @ side_axis

        common_mask = (
            (np.abs(local_along) >= float(args.attachment_min_along_distance))
            & (np.abs(local_along) <= float(args.attachment_max_along_distance))
            & (np.abs(local_side) <= float(args.attachment_side_width))
            & (near_line[:, 2] >= min_z)
            & (near_line[:, 2] <= max_z)
        )
        if not np.any(common_mask):
            continue

        for side_name, sign in (("left", -1.0), ("right", 1.0)):
            side_mask = common_mask & (local_along * sign > 0)
            side_line_ids = nearby_line_ids[side_mask]
            if side_line_ids.size < int(args.attachment_min_line_points):
                continue

            side_z = line_coord[side_line_ids, 2]
            layers = split_z_layers_by_indices(
                side_line_ids, side_z, args.attachment_z_gap
            )
            # 同一侧内从高到低编号，和真实横担层位一致。
            layers = layers[int(max(args.attachment_skip_top_layers, 0)) :]
            for layer_rank, layer_line_ids in enumerate(layers, start=1):
                if layer_line_ids.size < int(args.attachment_min_line_points):
                    continue

                layer_points = line_coord[layer_line_ids]
                layer_local_xy = layer_points[:, :2] - center[:2][None, :]
                layer_along = layer_local_xy @ along_axis
                abs_along = np.abs(layer_along)
                end_limit = float(np.min(abs_along) + args.attachment_end_window)
                end_mask = abs_along <= end_limit
                end_points = layer_points[end_mask]
                end_ids = layer_line_ids[end_mask]
                if end_points.shape[0] == 0:
                    end_points = layer_points
                    end_ids = layer_line_ids

                candidate_center = np.median(end_points, axis=0)

                support_points = []
                support_indices = []
                support_sources = []

                insulator_point_count = 0
                if insulator_tree is not None:
                    ins_ids = np.asarray(
                        insulator_tree.query_ball_point(
                            candidate_center,
                            r=float(args.attachment_insulator_search_radius),
                        ),
                        dtype=np.int64,
                    )
                    if ins_ids.size:
                        ins_points = insulator_coord[ins_ids]
                        support_points.append(ins_points)
                        support_indices.append(insulator_indices[ins_ids])
                        support_sources.append(
                            np.full(ins_points.shape[0], "insulator", dtype=object)
                        )
                        insulator_point_count = int(ins_points.shape[0])

                tower_point_count = 0
                if tower_tree is not None:
                    tower_ids = np.asarray(
                        tower_tree.query_ball_point(
                            candidate_center,
                            r=float(args.attachment_tower_search_radius),
                        ),
                        dtype=np.int64,
                    )
                    if tower_ids.size:
                        tower_points = tower_coord[tower_ids]
                        tower_local_xy = tower_points[:, :2] - center[:2][None, :]
                        tower_local_along = tower_local_xy @ along_axis
                        tower_local_side = tower_local_xy @ side_axis
                        # 只吸收当前接线侧附近的杆塔类点，避免把塔身中心和另一侧横担混进来。
                        tower_keep = (
                            (tower_local_along * sign > 0)
                            & (
                                np.abs(tower_local_along)
                                >= float(args.attachment_min_along_distance) * 0.5
                            )
                            & (
                                np.abs(tower_local_along)
                                <= float(args.attachment_max_along_distance)
                                + float(args.attachment_tower_search_radius)
                            )
                            & (
                                np.abs(tower_local_side)
                                <= float(args.attachment_side_width)
                            )
                        )
                        tower_ids = tower_ids[tower_keep]
                        tower_points = tower_points[tower_keep]
                    if tower_ids.size:
                        support_points.append(tower_points)
                        support_indices.append(tower_indices[tower_ids])
                        support_sources.append(
                            np.full(tower_points.shape[0], "tower", dtype=object)
                        )
                        tower_point_count = int(tower_points.shape[0])

                if support_points:
                    support_points = np.concatenate(support_points, axis=0)
                    support_indices = np.concatenate(support_indices, axis=0)
                    support_sources = np.concatenate(support_sources, axis=0)
                else:
                    support_points = np.empty((0, 3), dtype=np.float64)
                    support_indices = np.empty((0,), dtype=np.int64)
                    support_sources = np.empty((0,), dtype=object)

                record = build_attachment_record(
                    len(records) + 1,
                    tower,
                    side_name,
                    layer_rank,
                    candidate_center,
                    support_points,
                    support_indices,
                    support_sources,
                    int(layer_line_ids.size),
                    insulator_point_count,
                    tower_point_count,
                    args,
                    axis_source,
                    near_line_count,
                )
                records.append(record)
                tower["insulators"].append(record)
                groups.append(
                    {
                        "label": int(len(records) - 1),
                        "tower_id": int(tower["id"]),
                        "tower_name": tower["name"],
                        "side": side_name,
                        "layer_rank": int(layer_rank),
                        "line_point_count": int(layer_line_ids.size),
                        "insulator_point_count": int(insulator_point_count),
                        "axis_source": axis_source,
                    }
                )

    return records, towers, groups


def normalize_xyz(vector):
    """归一化三维向量。"""
    if vector is None:
        return None
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return None
    return vector / norm


def projection_width(points, origin, axis):
    """计算点到一条三维轴线的投影距离和横向距离。"""
    rel = points - np.asarray(origin, dtype=np.float64)[None, :]
    projection = rel @ axis
    closest = projection[:, None] * axis[None, :]
    perpendicular = rel - closest
    width = np.sqrt(np.sum(perpendicular * perpendicular, axis=1))
    return projection, width


def estimate_tower_x_core(tower_points, tower_center, along_axis, target_z, args):
    """估计当前高度附近杆塔主体在局部 X 轴上的中心和半宽。

    中位数和分位数对少量误分绝缘子分支不敏感；这里定义的是杆塔结构
    包络面，不使用绝缘子自身的点数、粗细或分叉作为停止条件。
    """
    tower_points = np.asarray(tower_points, dtype=np.float64)
    tower_center = np.asarray(tower_center, dtype=np.float64)
    local_x = (tower_points[:, :2] - tower_center[:2][None, :]) @ along_axis
    z_window = max(float(args.seed_grow_core_z_window), 0.10)
    z_mask = np.abs(tower_points[:, 2] - float(target_z)) <= z_window
    if np.count_nonzero(z_mask) < 30:
        z_mask = np.abs(tower_points[:, 2] - float(target_z)) <= 2.0 * z_window
    if np.count_nonzero(z_mask) < 30:
        z_mask = np.ones(tower_points.shape[0], dtype=bool)

    layer_x = local_x[z_mask]
    core_center_x = float(np.median(layer_x))
    quantile = float(np.clip(args.seed_grow_core_x_quantile, 0.50, 0.95))
    core_half_width = float(
        np.quantile(np.abs(layer_x - core_center_x), quantile)
    )
    core_half_width = max(
        core_half_width + float(args.seed_grow_core_x_margin),
        0.15,
    )
    return core_center_x, core_half_width, int(layer_x.size)


def grow_seeded_axis_candidate(
    candidate_points,
    candidate_indices,
    candidate_sources,
    endpoint,
    tower_center,
    axis,
    axis_name,
    args,
    stop_projection=None,
):
    """沿一个候选轴生长，并返回通过几何检查的结果。"""
    projection, width = projection_width(candidate_points, endpoint, axis)
    xy_dist_to_core = np.linalg.norm(
        candidate_points[:, :2] - tower_center[:2][None, :], axis=1
    )
    core_ok = (candidate_sources != "tower") | (
        xy_dist_to_core >= float(args.seed_grow_min_tower_core_distance)
    )
    max_projection = float(args.seed_grow_max_length)
    if stop_projection is not None:
        max_projection = min(max_projection, max(float(stop_projection), 0.0))
    valid = (
        (projection >= -float(args.seed_grow_backward_margin))
        & (projection <= max_projection)
        & (width <= float(args.seed_grow_max_width))
        & core_ok
    )
    if not np.any(valid):
        return None

    valid_points = candidate_points[valid]
    valid_indices = candidate_indices[valid]
    valid_sources = candidate_sources[valid]
    # 拓扑纠错只允许由“非最高导线层直接接触红点”启动。
    # 原始橙色预测点可以被合并进结果，但不再决定是否生成绝缘子。
    valid_seed_mask = valid_sources == "tower_seed"
    seed_count = int(np.count_nonzero(valid_seed_mask))
    if seed_count < int(args.seed_grow_min_seed_points):
        return None

    grown_local = seeded_component_indices(
        valid_points,
        valid_seed_mask,
        args.seed_grow_neighbor_radius,
    )
    if grown_local.size < int(args.seed_grow_min_points):
        return None

    grown_points = valid_points[grown_local]
    grown_indices = valid_indices[grown_local]
    grown_sources = valid_sources[grown_local]
    grown_projection, grown_width = projection_width(grown_points, endpoint, axis)

    grown_length = float(grown_projection.max() - grown_projection.min())
    max_width = float(grown_width.max(initial=0.0))
    tower_source_mask = np.isin(grown_sources, ("tower", "tower_seed"))
    recovered_tower_points = int(np.count_nonzero(tower_source_mask))
    if grown_length < float(args.seed_grow_min_length):
        return None
    if grown_length > float(args.seed_grow_max_length):
        return None
    if max_width > float(args.seed_grow_max_width):
        return None
    if recovered_tower_points == 0:
        return None

    # 优先选择包含更多黄色种子、形状更细长的方向；杆塔点数只做有限加分，
    # 避免横担因为点多而压过真实绝缘子方向。
    source_bonus = {
        "line_x_topology": 20.0,
    }.get(axis_name, 0.0)
    slenderness = grown_length / max(2.0 * max_width, 0.05)
    score = (
        30.0 * seed_count
        + min(float(recovered_tower_points), 120.0)
        + 20.0 * grown_length
        + 10.0 * slenderness
        - 15.0 * max_width
        + source_bonus
    )
    return {
        "points": grown_points,
        "indices": grown_indices,
        "sources": grown_sources,
        "axis": axis,
        "axis_name": axis_name,
        "seed_count": seed_count,
        "insulator_seed_count": int(np.count_nonzero(grown_sources == "insulator")),
        "contact_seed_count": int(np.count_nonzero(grown_sources == "tower_seed")),
        "recovered_tower_points": recovered_tower_points,
        "score": float(score),
    }


def seeded_component_indices(points, seed_mask, neighbor_radius):
    """在候选点中找与黄色种子连通的区域。

    这里使用 KDTree 邻域生长。只从黄色绝缘子种子开始扩展，所以不会
    对所有杆塔点盲目聚类。
    """
    if points.shape[0] == 0 or not np.any(seed_mask):
        return np.empty((0,), dtype=np.int64)
    tree = cKDTree(points)
    visited = np.zeros(points.shape[0], dtype=bool)
    queue = list(np.flatnonzero(seed_mask))
    visited[queue] = True
    head = 0
    radius = float(neighbor_radius)
    while head < len(queue):
        current = queue[head]
        head += 1
        neighbors = tree.query_ball_point(points[current], r=radius)
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(int(neighbor))
    return np.flatnonzero(visited)


def build_seed_grow_record(
    instance_id,
    tower,
    side_name,
    layer_rank,
    endpoint,
    grown_points,
    grown_indices,
    grown_sources,
    seed_count,
    line_point_count,
    recovered_tower_points,
    axis,
    axis_source,
    near_line_count,
    args,
):
    """根据种子生长结果生成绝缘子实例记录。"""
    bbox_min = grown_points.min(axis=0)
    bbox_max = grown_points.max(axis=0)
    top_point = grown_points[int(np.argmax(grown_points[:, 2]))]
    bottom_point = grown_points[int(np.argmin(grown_points[:, 2]))]
    center = grown_points.mean(axis=0)
    center_point = grown_points[
        int(np.argmin(np.sum((grown_points - center[None, :]) ** 2, axis=1)))
    ]
    projection, width = projection_width(grown_points, endpoint, axis)
    length = float(projection.max() - projection.min())
    max_width = float(width.max(initial=0.0))
    return {
        "id": int(instance_id),
        "name": f"绝缘子{instance_id}",
        "tower_id": int(tower["id"]),
        "tower_name": tower["name"],
        "side": side_name,
        "layer_rank": int(layer_rank),
        "method": "line_endpoint_seed_grow",
        "point_count": int(grown_points.shape[0]),
        "seed_insulator_points": int(seed_count),
        "line_contact_seed_points": int(
            np.count_nonzero(grown_sources == "tower_seed")
        ),
        "support_line_points": int(line_point_count),
        "support_insulator_points": int(np.sum(grown_sources == "insulator")),
        "support_tower_points": int(
            np.count_nonzero(np.isin(grown_sources, ("tower", "tower_seed")))
        ),
        "recovered_tower_points": int(recovered_tower_points),
        "endpoint_xyz": to_float_list(endpoint),
        "top_point_xyz": to_float_list(top_point),
        "bottom_point_xyz": to_float_list(bottom_point),
        "center_xyz": to_float_list(center),
        "center_point_xyz": to_float_list(center_point),
        "bbox_min_xyz": to_float_list(bbox_min),
        "bbox_max_xyz": to_float_list(bbox_max),
        "extent_xyz": to_float_list(bbox_max - bbox_min),
        "height": round(float(top_point[2] - bottom_point[2]), 6),
        "axis_xyz": to_float_list(axis),
        "axis_length": round(length, 6),
        "axis_max_width": round(max_width, 6),
        "axis_source": axis_source,
        "near_line_points_for_axis": int(near_line_count),
        "original_point_index_top": nearest_support_index(
            grown_points, grown_indices, top_point
        ),
        "original_point_index_bottom": nearest_support_index(
            grown_points, grown_indices, bottom_point
        ),
        "original_point_index_center": nearest_support_index(
            grown_points, grown_indices, center_point
        ),
        "_visual_points_xyz": grown_points.astype(np.float64, copy=False),
        "_grown_original_indices": grown_indices.astype(np.int64, copy=False),
    }


def bottom_up_tower_x_envelope(local_points, args):
    """从塔底向上逐层跟踪杆塔主体的 X 向中心和半宽。"""
    z = local_points[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())
    bin_size = max(float(args.tower_shape_z_bin), 0.10)
    bin_ids = np.floor((z - z_min) / bin_size).astype(np.int64)
    center_x = np.zeros(local_points.shape[0], dtype=np.float64)
    half_x = np.zeros(local_points.shape[0], dtype=np.float64)
    previous_center = None
    previous_half = None

    for bin_id in range(int(bin_ids.max(initial=0)) + 1):
        point_ids = np.flatnonzero(bin_ids == bin_id)
        if point_ids.size == 0:
            continue
        layer_x = local_points[point_ids, 0]
        if previous_center is not None:
            search_half = previous_half + float(args.tower_shape_center_search_margin)
            core_mask = np.abs(layer_x - previous_center) <= search_half
            core_x = layer_x[core_mask]
            if core_x.size < max(20, int(point_ids.size * 0.20)):
                core_x = layer_x
        else:
            core_x = layer_x

        current_center = float(np.median(core_x))
        quantile = float(np.clip(args.tower_shape_core_quantile, 0.50, 0.90))
        current_half = float(
            np.quantile(np.abs(core_x - current_center), quantile)
            + float(args.tower_shape_core_margin)
        )
        current_half = max(current_half, 0.15)
        center_x[point_ids] = current_center
        half_x[point_ids] = current_half
        previous_center = current_center
        previous_half = current_half

    return center_x, half_x, z_min, z_max


def nearest_z_indices(reference_z, query_z):
    """为每个查询高度找到参考点中 Z 最接近的索引。"""
    reference_z = np.asarray(reference_z, dtype=np.float64)
    query_z = np.asarray(query_z, dtype=np.float64)
    order = np.argsort(reference_z)
    sorted_z = reference_z[order]
    positions = np.searchsorted(sorted_z, query_z)
    right = np.clip(positions, 0, sorted_z.size - 1)
    left = np.clip(positions - 1, 0, sorted_z.size - 1)
    choose_left = np.abs(query_z - sorted_z[left]) <= np.abs(
        query_z - sorted_z[right]
    )
    return order[np.where(choose_left, left, right)]


def deterministic_kmeans(features, cluster_count, max_iterations=50):
    """小规模确定性 K-means，用于拆分平行绝缘子串。"""
    features = np.asarray(features, dtype=np.float64)
    cluster_count = max(int(cluster_count), 1)
    if features.shape[0] < cluster_count:
        return None
    if cluster_count == 1:
        return np.zeros(features.shape[0], dtype=np.int32)

    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[0]
    quantiles = np.linspace(0.05, 0.95, cluster_count)
    centers = np.asarray(
        [features[int(np.argmin(np.abs(projection - np.quantile(projection, q))))] for q in quantiles]
    )
    labels = np.zeros(features.shape[0], dtype=np.int32)
    for _ in range(max_iterations):
        distance2 = np.sum(
            (features[:, None, :] - centers[None, :, :]) ** 2, axis=2
        )
        new_labels = np.argmin(distance2, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(cluster_count):
            mask = labels == cluster_id
            if np.any(mask):
                centers[cluster_id] = features[mask].mean(axis=0)
            else:
                nearest_distance = np.min(distance2, axis=1)
                centers[cluster_id] = features[int(np.argmax(nearest_distance))]
    return labels


def detect_line_groups(local_line, args):
    """在塔两侧固定 X 截面上按 Y-Z 聚类，得到每一层的左右导线组。"""
    groups = []
    target_distance = max(float(args.line_group_sample_distance), 0.5)
    window = max(float(args.line_group_sample_window), 0.2)
    for side_name, sign in (("left", -1.0), ("right", 1.0)):
        side_mask = local_line[:, 0] * sign > 0.0
        sample_mask = side_mask & (
            np.abs(np.abs(local_line[:, 0]) - target_distance) <= window
        )
        if np.count_nonzero(sample_mask) < int(args.line_group_min_points):
            side_abs_x = np.abs(local_line[side_mask, 0])
            if side_abs_x.size == 0:
                continue
            fallback_distance = float(np.quantile(side_abs_x, 0.35))
            sample_mask = side_mask & (
                np.abs(np.abs(local_line[:, 0]) - fallback_distance) <= 2.0 * window
            )
        sample_ids = np.flatnonzero(sample_mask)
        if sample_ids.size == 0:
            continue
        yz_points = local_line[sample_ids, 1:3]
        yz_graph_points = np.column_stack(
            (yz_points, np.zeros(yz_points.shape[0], dtype=np.float64))
        )
        labels = radius_component_labels(yz_graph_points, args.line_group_yz_radius)
        for label in sorted(set(labels.tolist())):
            ids = sample_ids[labels == label]
            if ids.size < int(args.line_group_min_points):
                continue
            center_yz = np.median(local_line[ids, 1:3], axis=0)
            groups.append(
                {
                    "side": side_name,
                    "sign": sign,
                    "center_y": float(center_yz[0]),
                    "center_z": float(center_yz[1]),
                    "sample_point_count": int(ids.size),
                }
            )
    return groups


def assign_line_group_layers(groups, z_gap, skip_top_layers, expected_layers):
    """把左右导线组按 Z 从上到下归层，并排除最高导地线层。"""
    if not groups:
        return []
    ordered = sorted(range(len(groups)), key=lambda idx: groups[idx]["center_z"], reverse=True)
    layers = []
    for group_id in ordered:
        z = groups[group_id]["center_z"]
        if not layers:
            layers.append([group_id])
            continue
        layer_z = float(np.median([groups[idx]["center_z"] for idx in layers[-1]]))
        if abs(z - layer_z) <= float(z_gap):
            layers[-1].append(group_id)
        else:
            layers.append([group_id])
    layers = layers[max(int(skip_top_layers), 0) :]
    if int(expected_layers) > 0:
        layers = layers[: int(expected_layers)]
    for layer_rank, layer in enumerate(layers, start=1):
        for group_id in layer:
            groups[group_id]["layer_rank"] = layer_rank
    return layers


def hierarchical_line_group_instances(
    insulator_coord,
    insulator_indices,
    tower_coord,
    tower_indices,
    line_coord,
    args,
):
    """按“Z层 -> 左右导线组 -> 平行绝缘子串”生成实例。"""
    towers = build_tower_instances(tower_coord, args)
    for tower in towers:
        tower["insulators"] = []
    if not towers or line_coord.shape[0] == 0:
        return [], towers, [], np.empty((0,), dtype=np.int64)

    global_axis = global_along_axis(towers, line_coord)
    center_tree = cKDTree(
        np.asarray([tower["center"][:2] for tower in towers], dtype=np.float64)
    )
    _, tower_assignment = center_tree.query(tower_coord[:, :2], k=1)
    records = []
    groups_meta = []
    recovered_parts = []
    global_pair_id = 0

    for tower_position, tower in enumerate(towers):
        center = np.asarray(tower["center"], dtype=np.float64)
        along_axis, axis_source, near_line_count = estimate_tower_along_axis(
            tower, towers, line_coord, global_axis, args.axis_line_search_radius
        )
        side_axis = normalize_xy(
            np.array([-along_axis[1], along_axis[0]], dtype=np.float64)
        )
        if side_axis is None:
            continue

        assigned_tower_ids = np.flatnonzero(tower_assignment == tower_position)
        red_points = tower_coord[assigned_tower_ids]
        red_original_ids = tower_indices[assigned_tower_ids]
        red_rel_xy = red_points[:, :2] - center[:2][None, :]
        red_local = np.column_stack(
            (red_rel_xy @ along_axis, red_rel_xy @ side_axis, red_points[:, 2])
        )
        core_center_x, core_half_x, z_min, z_max = bottom_up_tower_x_envelope(
            red_local, args
        )
        height = max(z_max - z_min, 1e-6)
        red_excess = np.abs(red_local[:, 0] - core_center_x) - core_half_x
        red_candidate_mask = (
            (red_excess > 0.0)
            & (
                red_local[:, 2]
                >= z_min + height * float(args.tower_shape_min_z_ratio)
            )
        )

        pool_points = [red_points[red_candidate_mask]]
        pool_local = [red_local[red_candidate_mask]]
        pool_original = [red_original_ids[red_candidate_mask]]
        pool_sources = [
            np.full(np.count_nonzero(red_candidate_mask), "tower", dtype=object)
        ]
        pool_excess = [red_excess[red_candidate_mask]]

        if insulator_coord.shape[0]:
            ins_distance_xy = np.linalg.norm(
                insulator_coord[:, :2] - center[:2][None, :], axis=1
            )
            ins_mask = (
                (ins_distance_xy <= float(args.axis_line_search_radius))
                & (insulator_coord[:, 2] >= z_min)
                & (insulator_coord[:, 2] <= z_max + float(args.tower_bind_z_margin))
            )
            ins_ids = np.flatnonzero(ins_mask)
            if ins_ids.size:
                points = insulator_coord[ins_ids]
                rel_xy = points[:, :2] - center[:2][None, :]
                local = np.column_stack(
                    (rel_xy @ along_axis, rel_xy @ side_axis, points[:, 2])
                )
                nearest_red_z = nearest_z_indices(red_points[:, 2], points[:, 2])
                excess = (
                    np.abs(local[:, 0] - core_center_x[nearest_red_z])
                    - core_half_x[nearest_red_z]
                )
                keep = excess >= -float(args.tower_shape_cluster_radius)
                if np.any(keep):
                    pool_points.append(points[keep])
                    pool_local.append(local[keep])
                    pool_original.append(insulator_indices[ins_ids[keep]])
                    pool_sources.append(
                        np.full(np.count_nonzero(keep), "insulator", dtype=object)
                    )
                    pool_excess.append(excess[keep])

        candidate_points = np.concatenate(pool_points, axis=0)
        candidate_local = np.concatenate(pool_local, axis=0)
        candidate_original = np.concatenate(pool_original)
        candidate_sources = np.concatenate(pool_sources)
        candidate_excess = np.concatenate(pool_excess)
        if candidate_points.shape[0] == 0:
            continue

        line_distance_xy = np.linalg.norm(
            line_coord[:, :2] - center[:2][None, :], axis=1
        )
        near_line_ids = np.flatnonzero(
            line_distance_xy <= float(args.axis_line_search_radius)
        )
        if near_line_ids.size == 0:
            continue
        near_line_points = line_coord[near_line_ids]
        line_rel_xy = near_line_points[:, :2] - center[:2][None, :]
        local_line = np.column_stack(
            (line_rel_xy @ along_axis, line_rel_xy @ side_axis, near_line_points[:, 2])
        )
        line_groups = detect_line_groups(local_line, args)
        layers = assign_line_group_layers(
            line_groups,
            args.tower_shape_layer_z_gap,
            max(int(args.attachment_skip_top_layers), 1),
            args.expected_insulated_layers,
        )
        if not layers:
            print(f"{tower['name']}: no insulated line layers detected", flush=True)
            continue

        tower_candidates = []
        per_group_count = max(int(args.insulators_per_line_group), 1)
        corridor_radius = float(args.line_group_corridor_radius)

        for layer_rank, layer_group_ids in enumerate(layers, start=1):
            layer_candidates = []
            for group_order, group_id in enumerate(layer_group_ids, start=1):
                line_group = line_groups[group_id]
                sign = float(line_group["sign"])
                yz_center = np.array(
                    [line_group["center_y"], line_group["center_z"]],
                    dtype=np.float64,
                )
                yz_distance = np.linalg.norm(
                    candidate_local[:, 1:3] - yz_center[None, :], axis=1
                )
                corridor_mask = (
                    (candidate_local[:, 0] * sign > 0.0)
                    & (yz_distance <= corridor_radius)
                )
                corridor_ids = np.flatnonzero(corridor_mask)
                if corridor_ids.size < max(
                    int(args.tower_shape_min_points), per_group_count * 3
                ):
                    continue

                relevant_line_mask = (
                    (local_line[:, 0] * sign > 0.0)
                    & (
                        np.linalg.norm(
                            local_line[:, 1:3] - yz_center[None, :], axis=1
                        )
                        <= 1.5 * corridor_radius
                    )
                )
                relevant_line_ids = np.flatnonzero(relevant_line_mask)
                if relevant_line_ids.size == 0:
                    continue
                relevant_line_points = near_line_points[relevant_line_ids]
                relevant_line_tree = cKDTree(relevant_line_points)

                split_labels = deterministic_kmeans(
                    candidate_local[corridor_ids, 1:3], per_group_count
                )
                if split_labels is None:
                    continue
                for string_id in range(per_group_count):
                    string_ids = corridor_ids[split_labels == string_id]
                    if string_ids.size < int(args.tower_shape_min_points):
                        continue
                    points = candidate_points[string_ids]
                    local = candidate_local[string_ids]
                    sources = candidate_sources[string_ids]
                    original_ids = candidate_original[string_ids]
                    red_mask = sources == "tower"
                    if not np.any(red_mask):
                        continue
                    if float(candidate_excess[string_ids].min()) > 2.0 * float(
                        args.tower_shape_cluster_radius
                    ):
                        continue

                    centered = local - local.mean(axis=0, keepdims=True)
                    covariance = centered.T @ centered / max(points.shape[0] - 1, 1)
                    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                    order = np.argsort(eigenvalues)[::-1]
                    eigenvalues = eigenvalues[order]
                    principal_axis = eigenvectors[:, order[0]]
                    if eigenvalues[0] <= 1e-10:
                        continue
                    linearity = float(
                        (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
                    )
                    x_alignment = abs(float(principal_axis[0]))
                    projection = centered @ principal_axis
                    length = float(projection.max() - projection.min())
                    perpendicular = (
                        centered - projection[:, None] * principal_axis[None, :]
                    )
                    width = float(
                        2.0
                        * np.quantile(np.linalg.norm(perpendicular, axis=1), 0.90)
                    )
                    if linearity < float(args.tower_shape_min_linearity):
                        continue
                    if x_alignment < float(args.tower_shape_min_x_alignment):
                        continue
                    if not (
                        float(args.tower_shape_min_length)
                        <= length
                        <= float(args.tower_shape_max_length)
                    ):
                        continue
                    if width > float(args.tower_shape_max_width):
                        continue

                    line_distances, line_nn = relevant_line_tree.query(points, k=1)
                    nearest_id = int(np.argmin(line_distances))
                    min_line_distance = float(line_distances[nearest_id])
                    if min_line_distance > float(args.tower_shape_max_line_distance):
                        continue
                    endpoint = relevant_line_points[int(line_nn[nearest_id])]
                    mean_x = float(np.mean(local[:, 0]))
                    axis_xyz = normalize_xyz(
                        np.array([along_axis[0], along_axis[1], 0.0])
                    )
                    if mean_x < 0:
                        axis_xyz = -axis_xyz
                    layer_candidates.append(
                        {
                            "points": points,
                            "local": local,
                            "sources": sources,
                            "original_ids": original_ids,
                            "red_original_ids": original_ids[red_mask],
                            "side": line_group["side"],
                            "mean_x": mean_x,
                            "mean_y": float(np.mean(local[:, 1])),
                            "mean_z": float(np.mean(local[:, 2])),
                            "endpoint": endpoint,
                            "axis_xyz": axis_xyz,
                            "layer_rank": layer_rank,
                            "line_group_order": group_order,
                            "string_order": string_id + 1,
                            "linearity": linearity,
                            "x_alignment": x_alignment,
                            "width": width,
                            "line_distance": min_line_distance,
                        }
                    )

            left_ids = [
                idx for idx, item in enumerate(layer_candidates) if item["side"] == "left"
            ]
            unused_right = {
                idx for idx, item in enumerate(layer_candidates) if item["side"] == "right"
            }
            accepted = set()
            pair_ids = {}
            if args.no_tower_shape_require_symmetry:
                accepted.update(range(len(layer_candidates)))
            else:
                for left_id in left_ids:
                    left = layer_candidates[left_id]
                    possible = []
                    for right_id in unused_right:
                        right = layer_candidates[right_id]
                        dx = abs(abs(left["mean_x"]) - abs(right["mean_x"]))
                        dy = abs(left["mean_y"] - right["mean_y"])
                        dz = abs(left["mean_z"] - right["mean_z"])
                        if dx > float(args.tower_shape_symmetry_x_tolerance):
                            continue
                        if dy > float(args.tower_shape_symmetry_y_tolerance):
                            continue
                        if dz > float(args.tower_shape_symmetry_z_tolerance):
                            continue
                        possible.append((dx + dy + dz, right_id))
                    if not possible:
                        continue
                    _, right_id = min(possible)
                    unused_right.remove(right_id)
                    global_pair_id += 1
                    accepted.update((left_id, right_id))
                    pair_ids[left_id] = global_pair_id
                    pair_ids[right_id] = global_pair_id

            left_count = sum(
                1 for idx in accepted if layer_candidates[idx]["side"] == "left"
            )
            right_count = sum(
                1 for idx in accepted if layer_candidates[idx]["side"] == "right"
            )
            print(
                f"{tower['name']} layer {layer_rank}: "
                f"line_groups={len(layer_group_ids)}, "
                f"left_insulators={left_count}, right_insulators={right_count}",
                flush=True,
            )
            for candidate_id in sorted(accepted):
                item = layer_candidates[candidate_id]
                item["symmetry_pair_id"] = pair_ids.get(candidate_id)
                tower_candidates.append(item)

        for item in tower_candidates:
            record = build_seed_grow_record(
                len(records) + 1,
                tower,
                item["side"],
                int(item["layer_rank"]),
                item["endpoint"],
                item["points"],
                item["original_ids"],
                item["sources"],
                int(np.count_nonzero(item["sources"] == "insulator")),
                int(near_line_points.shape[0]),
                int(item["red_original_ids"].size),
                item["axis_xyz"],
                f"{axis_source}+hierarchical_line_group",
                near_line_count,
                args,
            )
            record["method"] = "hierarchical_line_group"
            record["line_group_order"] = int(item["line_group_order"])
            record["string_order"] = int(item["string_order"])
            record["symmetry_axis"] = "local_y"
            record["symmetry_pair_id"] = item["symmetry_pair_id"]
            record["linearity"] = round(item["linearity"], 6)
            record["x_alignment"] = round(item["x_alignment"], 6)
            record["pca_width"] = round(item["width"], 6)
            record["distance_to_line"] = round(item["line_distance"], 6)
            records.append(record)
            tower["insulators"].append(record)
            recovered_parts.append(item["red_original_ids"])
            groups_meta.append(
                {
                    "label": len(records) - 1,
                    "tower_id": int(tower["id"]),
                    "tower_name": tower["name"],
                    "layer_rank": int(item["layer_rank"]),
                    "side": item["side"],
                    "line_group_order": int(item["line_group_order"]),
                    "string_order": int(item["string_order"]),
                    "symmetry_pair_id": item["symmetry_pair_id"],
                }
            )

    recovered = (
        np.unique(np.concatenate(recovered_parts))
        if recovered_parts
        else np.empty((0,), dtype=np.int64)
    )
    return records, towers, groups_meta, recovered


def tower_shape_insulator_instances(
    insulator_coord,
    insulator_indices,
    tower_coord,
    tower_indices,
    line_coord,
    args,
):
    """由杆塔主体包络外的 X 向线性分支恢复绝缘子。"""
    towers = build_tower_instances(tower_coord, args)
    for tower in towers:
        tower["insulators"] = []
    if not towers or line_coord.shape[0] == 0:
        return [], towers, [], np.empty((0,), dtype=np.int64)

    global_axis = global_along_axis(towers, line_coord)
    records = []
    groups = []
    recovered_parts = []
    tower_center_tree = cKDTree(
        np.asarray([tower["center"][:2] for tower in towers], dtype=np.float64)
    )
    _, tower_assignment = tower_center_tree.query(tower_coord[:, :2], k=1)

    for tower_position, tower in enumerate(towers):
        center = np.asarray(tower["center"], dtype=np.float64)
        along_axis, axis_source, near_line_count = estimate_tower_along_axis(
            tower, towers, line_coord, global_axis, args.axis_line_search_radius
        )
        side_axis = normalize_xy(
            np.array([-along_axis[1], along_axis[0]], dtype=np.float64)
        )
        if side_axis is None:
            continue

        assigned_tower_ids = np.flatnonzero(tower_assignment == tower_position)
        tower_points = np.asarray(tower_coord[assigned_tower_ids], dtype=np.float64)
        rel_xy = tower_points[:, :2] - center[:2][None, :]
        local_points = np.column_stack(
            (rel_xy @ along_axis, rel_xy @ side_axis, tower_points[:, 2])
        )
        core_center_x, core_half_x, z_min, z_max = bottom_up_tower_x_envelope(
            local_points, args
        )
        height = max(z_max - z_min, 1e-6)
        x_excess = np.abs(local_points[:, 0] - core_center_x) - core_half_x
        candidate_mask = (
            (x_excess > 0.0)
            & (
                local_points[:, 2]
                >= z_min + height * float(args.tower_shape_min_z_ratio)
            )
        )
        candidate_local_ids = np.flatnonzero(candidate_mask)
        if candidate_local_ids.size == 0:
            continue

        # 蓝线不再参与普通相线的 Z 分层，避免悬垂曲线把多层连成一层。
        # 它只用于提供最高导地线参考高度，以及验证候选外端是否接线。
        line_distance_xy = np.linalg.norm(
            line_coord[:, :2] - center[:2][None, :], axis=1
        )
        near_line_ids = np.flatnonzero(
            line_distance_xy <= float(args.axis_line_search_radius)
        )
        if near_line_ids.size == 0:
            continue
        near_line_points = line_coord[near_line_ids]
        near_line_tree = cKDTree(near_line_points)
        ground_reference_z = float(np.quantile(near_line_points[:, 2], 0.995))

        tower_original_ids = tower_indices[assigned_tower_ids]
        candidate_points_parts = [tower_points[candidate_local_ids]]
        candidate_local_parts = [local_points[candidate_local_ids]]
        candidate_original_parts = [tower_original_ids[candidate_local_ids]]
        candidate_source_parts = [
            np.full(candidate_local_ids.size, "tower", dtype=object)
        ]
        candidate_excess_parts = [x_excess[candidate_local_ids]]

        # 已有橙色点只是连通路径的一部分，不能作为聚类终点。把杆塔附近的
        # 橙色点与红色突出点放进同一个图，聚类会穿过它们继续走到蓝线。
        if insulator_coord.shape[0]:
            ins_xy_distance = np.linalg.norm(
                insulator_coord[:, :2] - center[:2][None, :], axis=1
            )
            ins_mask = (
                (ins_xy_distance <= float(args.axis_line_search_radius))
                & (insulator_coord[:, 2] >= z_min)
                & (insulator_coord[:, 2] <= z_max + float(args.tower_bind_z_margin))
            )
            ins_ids = np.flatnonzero(ins_mask)
            if ins_ids.size:
                ins_points = insulator_coord[ins_ids]
                ins_rel_xy = ins_points[:, :2] - center[:2][None, :]
                ins_local = np.column_stack(
                    (ins_rel_xy @ along_axis, ins_rel_xy @ side_axis, ins_points[:, 2])
                )
                nearest_z_ids = nearest_z_indices(
                    tower_points[:, 2], ins_points[:, 2]
                )
                ins_excess = (
                    np.abs(ins_local[:, 0] - core_center_x[nearest_z_ids])
                    - core_half_x[nearest_z_ids]
                )
                keep_ins = ins_excess >= -float(args.tower_shape_cluster_radius)
                if np.any(keep_ins):
                    candidate_points_parts.append(ins_points[keep_ins])
                    candidate_local_parts.append(ins_local[keep_ins])
                    candidate_original_parts.append(insulator_indices[ins_ids[keep_ins]])
                    candidate_source_parts.append(
                        np.full(np.count_nonzero(keep_ins), "insulator", dtype=object)
                    )
                    candidate_excess_parts.append(ins_excess[keep_ins])

        candidate_points = np.concatenate(candidate_points_parts, axis=0)
        candidate_local = np.concatenate(candidate_local_parts, axis=0)
        candidate_original_ids = np.concatenate(candidate_original_parts)
        candidate_sources = np.concatenate(candidate_source_parts)
        candidate_excess = np.concatenate(candidate_excess_parts)
        component_labels = radius_component_labels(
            candidate_points, args.tower_shape_cluster_radius
        )
        valid_candidates = []

        for component_label in sorted(set(component_labels.tolist())):
            component_mask = component_labels == component_label
            points = candidate_points[component_mask]
            if points.shape[0] < int(args.tower_shape_min_points):
                continue

            local = candidate_local[component_mask]
            sources = candidate_sources[component_mask]
            original_ids = candidate_original_ids[component_mask]
            red_mask = sources == "tower"
            if not np.any(red_mask):
                continue
            centered = local - local.mean(axis=0, keepdims=True)
            covariance = centered.T @ centered / max(points.shape[0] - 1, 1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            principal_axis = eigenvectors[:, order[0]]
            if eigenvalues[0] <= 1e-10:
                continue
            linearity = float((eigenvalues[0] - eigenvalues[1]) / eigenvalues[0])
            x_alignment = abs(float(principal_axis[0]))
            projection = centered @ principal_axis
            length = float(projection.max() - projection.min())
            perpendicular = centered - projection[:, None] * principal_axis[None, :]
            width = float(
                2.0 * np.quantile(np.linalg.norm(perpendicular, axis=1), 0.90)
            )
            if linearity < float(args.tower_shape_min_linearity):
                continue
            if x_alignment < float(args.tower_shape_min_x_alignment):
                continue
            if not (
                float(args.tower_shape_min_length)
                <= length
                <= float(args.tower_shape_max_length)
            ):
                continue
            if width > float(args.tower_shape_max_width):
                continue

            # 内端必须回到主体包络附近，外端必须接触非最高层电线。
            component_x_excess = candidate_excess[component_mask]
            if float(component_x_excess.min()) > 2.0 * float(
                args.tower_shape_cluster_radius
            ):
                continue
            line_distances, nearest_line_local_ids = near_line_tree.query(points, k=1)
            nearest_point_id = int(np.argmin(line_distances))
            min_line_distance = float(line_distances[nearest_point_id])
            if min_line_distance > float(args.tower_shape_max_line_distance):
                continue

            nearest_core_ids = nearest_z_indices(
                tower_points[:, 2], points[:, 2]
            )
            mean_x = float(
                np.mean(local[:, 0] - core_center_x[nearest_core_ids])
            )
            side_name = "right" if mean_x >= 0 else "left"
            endpoint = near_line_points[int(nearest_line_local_ids[nearest_point_id])]
            axis_xyz = normalize_xyz(
                np.array([along_axis[0], along_axis[1], 0.0], dtype=np.float64)
            )
            if mean_x < 0:
                axis_xyz = -axis_xyz
            recovered_red_ids = original_ids[red_mask]
            valid_candidates.append(
                {
                    "points": points,
                    "local": local,
                    "sources": sources,
                    "original_ids": original_ids,
                    "recovered_red_ids": recovered_red_ids,
                    "side_name": side_name,
                    "mean_x": mean_x,
                    "mean_y": float(np.mean(local[:, 1])),
                    "mean_z": float(np.mean(local[:, 2])),
                    "endpoint": endpoint,
                    "axis_xyz": axis_xyz,
                    "linearity": linearity,
                    "x_alignment": x_alignment,
                    "width": width,
                    "length": length,
                    "min_line_distance": min_line_distance,
                    "line_endpoint_z": float(endpoint[2]),
                }
            )

        # 先排除最高导地线，再由候选自身的 Z 中心从上到下分层。
        # 悬垂蓝线的连续 Z 变化不会再把相邻电气层合并。
        phase_candidate_ids = [
            idx
            for idx, item in enumerate(valid_candidates)
            if item["line_endpoint_z"]
            < ground_reference_z - float(args.tower_shape_ground_z_margin)
        ]
        ordered_ids = sorted(
            phase_candidate_ids,
            key=lambda idx: valid_candidates[idx]["mean_z"],
            reverse=True,
        )
        candidate_layers = []
        for candidate_id in ordered_ids:
            candidate_z = valid_candidates[candidate_id]["mean_z"]
            if not candidate_layers:
                candidate_layers.append([candidate_id])
                continue
            current_z = float(
                np.median(
                    [valid_candidates[idx]["mean_z"] for idx in candidate_layers[-1]]
                )
            )
            if abs(candidate_z - current_z) <= float(args.tower_shape_layer_z_gap):
                candidate_layers[-1].append(candidate_id)
            else:
                candidate_layers.append([candidate_id])
        for layer_rank, layer_ids in enumerate(candidate_layers, start=1):
            for candidate_id in layer_ids:
                valid_candidates[candidate_id]["layer_rank"] = layer_rank

        accepted_ids = set()
        symmetry_pair_ids = {}
        if args.no_tower_shape_require_symmetry:
            accepted_ids.update(phase_candidate_ids)
        else:
            pair_counter = 0
            for layer_ids in candidate_layers:
                left_ids = [
                    idx for idx in layer_ids if valid_candidates[idx]["mean_x"] < 0
                ]
                unused_right = {
                    idx for idx in layer_ids if valid_candidates[idx]["mean_x"] >= 0
                }
                for left_id in left_ids:
                    left = valid_candidates[left_id]
                    possible = []
                    for right_id in unused_right:
                        right = valid_candidates[right_id]
                        dx = abs(abs(left["mean_x"]) - abs(right["mean_x"]))
                        dy = abs(left["mean_y"] - right["mean_y"])
                        dz = abs(left["mean_z"] - right["mean_z"])
                        if dx > float(args.tower_shape_symmetry_x_tolerance):
                            continue
                        if dy > float(args.tower_shape_symmetry_y_tolerance):
                            continue
                        if dz > float(args.tower_shape_symmetry_z_tolerance):
                            continue
                        possible.append((dx + dy + dz, right_id))
                    if not possible:
                        continue
                    _, right_id = min(possible)
                    unused_right.remove(right_id)
                    pair_counter += 1
                    accepted_ids.update((left_id, right_id))
                    symmetry_pair_ids[left_id] = pair_counter
                    symmetry_pair_ids[right_id] = pair_counter

        for candidate_id in sorted(accepted_ids):
            item = valid_candidates[candidate_id]
            record = build_seed_grow_record(
                len(records) + 1,
                tower,
                item["side_name"],
                int(item["layer_rank"]),
                item["endpoint"],
                item["points"],
                item["original_ids"],
                item["sources"],
                0,
                int(near_line_points.shape[0]),
                int(item["recovered_red_ids"].size),
                item["axis_xyz"],
                f"{axis_source}+bottom_up_tower_shape",
                near_line_count,
                args,
            )
            record["method"] = "bottom_up_tower_shape"
            record["linearity"] = round(item["linearity"], 6)
            record["x_alignment"] = round(item["x_alignment"], 6)
            record["pca_width"] = round(item["width"], 6)
            record["distance_to_lower_line"] = round(
                item["min_line_distance"], 6
            )
            record["symmetry_axis"] = "local_y"
            record["symmetry_pair_required"] = bool(
                not args.no_tower_shape_require_symmetry
            )
            record["symmetry_pair_id"] = symmetry_pair_ids.get(candidate_id)
            records.append(record)
            tower["insulators"].append(record)
            recovered_parts.append(item["recovered_red_ids"])
            groups.append(
                {
                    "label": len(records) - 1,
                    "tower_id": int(tower["id"]),
                    "tower_name": tower["name"],
                    "side": item["side_name"],
                    "symmetry_pair_id": symmetry_pair_ids.get(candidate_id),
                    "linearity": round(item["linearity"], 6),
                    "x_alignment": round(item["x_alignment"], 6),
                }
            )

    recovered = (
        np.unique(np.concatenate(recovered_parts))
        if recovered_parts
        else np.empty((0,), dtype=np.int64)
    )
    return records, towers, groups, recovered


def seed_grow_insulator_instances(
    insulator_coord,
    insulator_indices,
    tower_coord,
    tower_indices,
    line_coord,
    line_indices,
    args,
):
    """按“杆塔 -> 绝缘子 -> 导线”的线路拓扑恢复绝缘子。

    局部 X 轴沿线路方向，Y 轴为水平横向，Z 轴向上。最高导地线层允许
    直接连接杆塔；其余层若出现蓝线与红色杆塔点直接接触，就从接触红点
    沿 +/-X 向杆塔回溯，直到横担/塔体出现结构增粗或分叉。
    """
    towers = build_tower_instances(tower_coord, args)
    for tower in towers:
        tower["insulators"] = []
    if not towers or line_coord.shape[0] == 0:
        return [], towers, [], np.empty((0,), dtype=np.int64)

    global_axis = global_along_axis(towers, line_coord)
    line_xy_tree = cKDTree(line_coord[:, :2])
    insulator_tree = cKDTree(insulator_coord) if insulator_coord.shape[0] else None
    tower_tree = cKDTree(tower_coord) if tower_coord.shape[0] else None

    records = []
    groups = []
    grown_global_indices = []
    claimed_original_indices = set()
    search_radius_xy = float(
        np.hypot(args.attachment_max_along_distance, args.attachment_side_width)
    )

    for tower in towers:
        center = np.asarray(tower["center"], dtype=np.float64)
        height = float(tower["bbox_max"][2] - tower["bbox_min"][2])
        min_z = float(tower["bbox_min"][2] + height * args.attachment_min_tower_z_ratio)
        max_z = float(tower["bbox_max"][2] + args.tower_bind_z_margin)

        along_axis, axis_source, near_line_count = estimate_tower_along_axis(
            tower, towers, line_coord, global_axis, args.axis_line_search_radius
        )
        side_axis = np.array([-along_axis[1], along_axis[0]], dtype=np.float64)
        side_axis = normalize_xy(side_axis)
        if side_axis is None:
            continue

        nearby_line_ids = np.asarray(
            line_xy_tree.query_ball_point(center[:2], r=search_radius_xy),
            dtype=np.int64,
        )
        if nearby_line_ids.size == 0:
            continue
        near_line = line_coord[nearby_line_ids]
        local_xy = near_line[:, :2] - center[:2][None, :]
        local_along = local_xy @ along_axis
        local_side = local_xy @ side_axis
        common_mask = (
            (np.abs(local_along) >= float(args.attachment_min_along_distance))
            & (np.abs(local_along) <= float(args.attachment_max_along_distance))
            & (np.abs(local_side) <= float(args.attachment_side_width))
            & (near_line[:, 2] >= min_z)
            & (near_line[:, 2] <= max_z)
        )
        if not np.any(common_mask):
            continue

        for side_name, sign in (("left", -1.0), ("right", 1.0)):
            side_mask = common_mask & (local_along * sign > 0)
            side_line_ids = nearby_line_ids[side_mask]
            if side_line_ids.size < int(args.attachment_min_line_points):
                continue

            side_z = line_coord[side_line_ids, 2]
            layers = split_z_layers_by_indices(
                side_line_ids, side_z, args.attachment_z_gap
            )
            layers = layers[int(max(args.attachment_skip_top_layers, 0)) :]
            for layer_rank, layer_line_ids in enumerate(layers, start=1):
                if layer_line_ids.size < int(args.attachment_min_line_points):
                    continue

                layer_points = line_coord[layer_line_ids]
                layer_local_xy = layer_points[:, :2] - center[:2][None, :]
                layer_along = layer_local_xy @ along_axis
                abs_along = np.abs(layer_along)
                end_limit = float(np.min(abs_along) + args.attachment_end_window)
                end_points = layer_points[abs_along <= end_limit]
                if end_points.shape[0] == 0:
                    end_points = layer_points

                # 最高导地线层已经在上面跳过。其余层中，蓝线若直接接触红点，
                # 这些红点按线路拓扑应属于绝缘子，而不是杆塔。
                contact_tower_ids = np.empty((0,), dtype=np.int64)
                if (
                    tower_tree is not None
                    and not args.no_seed_grow_line_contact_seeds
                    and end_points.shape[0] > 0
                ):
                    contact_lists = tower_tree.query_ball_point(
                        end_points,
                        r=float(args.seed_grow_line_contact_radius),
                    )
                    nonempty_contacts = [
                        np.asarray(item, dtype=np.int64)
                        for item in contact_lists
                        if len(item) > 0
                    ]
                    if nonempty_contacts:
                        contact_tower_ids = np.unique(
                            np.concatenate(nonempty_contacts)
                        )

                if contact_tower_ids.size < int(args.seed_grow_min_seed_points):
                    continue

                # 接触红点比一整段导线的中位点更能代表真实连接位置。
                endpoint = np.median(tower_coord[contact_tower_ids], axis=0)

                candidate_points = []
                candidate_indices = []
                candidate_sources = []

                if insulator_tree is not None:
                    ins_ids = np.asarray(
                        insulator_tree.query_ball_point(
                            endpoint, r=float(args.seed_grow_candidate_radius)
                        ),
                        dtype=np.int64,
                    )
                else:
                    ins_ids = np.empty((0,), dtype=np.int64)
                if ins_ids.size:
                    candidate_points.append(insulator_coord[ins_ids])
                    candidate_indices.append(insulator_indices[ins_ids])
                    candidate_sources.append(
                        np.full(ins_ids.size, "insulator", dtype=object)
                    )

                if tower_tree is not None:
                    tower_ids = np.asarray(
                        tower_tree.query_ball_point(
                            endpoint, r=float(args.seed_grow_candidate_radius)
                        ),
                        dtype=np.int64,
                    )
                    if tower_ids.size:
                        candidate_points.append(tower_coord[tower_ids])
                        candidate_indices.append(tower_indices[tower_ids])
                        tower_sources = np.full(tower_ids.size, "tower", dtype=object)
                        if contact_tower_ids.size:
                            contact_original_indices = tower_indices[contact_tower_ids]
                            tower_sources[
                                np.isin(
                                    tower_indices[tower_ids],
                                    contact_original_indices,
                                )
                            ] = "tower_seed"
                        candidate_sources.append(tower_sources)

                if not candidate_points:
                    continue
                candidate_points = np.concatenate(candidate_points, axis=0)
                candidate_indices = np.concatenate(candidate_indices, axis=0)
                candidate_sources = np.concatenate(candidate_sources, axis=0)

                core_center_x, core_half_width, core_sample_points = (
                    estimate_tower_x_core(
                        tower["points"],
                        center,
                        along_axis,
                        endpoint[2],
                        args,
                    )
                )
                endpoint_local_x = float(
                    (endpoint[:2] - center[:2]) @ along_axis
                )
                contact_sign = 1.0 if endpoint_local_x >= core_center_x else -1.0
                core_boundary_x = core_center_x + contact_sign * core_half_width
                stop_projection = (
                    abs(endpoint_local_x - core_boundary_x)
                    + float(args.seed_grow_core_entry_margin)
                )
                if stop_projection < float(args.seed_grow_min_length):
                    continue

                # X 轴沿线路方向；从接触点沿 -contact_sign * X 回到杆塔。
                # 到达当前高度的杆塔 X 向包络面后停止，不看绝缘子自身粗细。
                topology_axis = normalize_xyz(
                    np.array(
                        [
                            -contact_sign * along_axis[0],
                            -contact_sign * along_axis[1],
                            0.0,
                        ],
                        dtype=np.float64,
                    )
                )
                if topology_axis is None:
                    continue
                best_growth = grow_seeded_axis_candidate(
                    candidate_points,
                    candidate_indices,
                    candidate_sources,
                    endpoint,
                    center,
                    topology_axis,
                    "line_x_topology",
                    args,
                    stop_projection=stop_projection,
                )
                if best_growth is None:
                    continue
                grown_points = best_growth["points"]
                grown_indices = best_growth["indices"]
                grown_sources = best_growth["sources"]
                axis = best_growth["axis"]
                recovered_tower_points = best_growth["recovered_tower_points"]

                # 同一串绝缘子可能被同高度的多根分裂导线重复命中。
                overlap_count = sum(
                    int(index) in claimed_original_indices for index in grown_indices
                )
                if overlap_count / max(int(grown_indices.size), 1) >= 0.5:
                    continue
                claimed_original_indices.update(int(index) for index in grown_indices)

                growth_axis_source = (
                    f"{axis_source}+{best_growth['axis_name']}"
                )

                record = build_seed_grow_record(
                    len(records) + 1,
                    tower,
                    side_name,
                    layer_rank,
                    endpoint,
                    grown_points,
                    grown_indices,
                    grown_sources,
                    int(best_growth["insulator_seed_count"]),
                    int(layer_line_ids.size),
                    recovered_tower_points,
                    axis,
                    growth_axis_source,
                    near_line_count,
                    args,
                )
                record["tower_core_x_center"] = round(core_center_x, 6)
                record["tower_core_x_half_width"] = round(core_half_width, 6)
                record["tower_core_sample_points"] = int(core_sample_points)
                record["tower_core_stop_projection"] = round(stop_projection, 6)
                records.append(record)
                tower["insulators"].append(record)
                grown_global_indices.append(
                    grown_indices[
                        np.isin(grown_sources, ("tower", "tower_seed"))
                    ]
                )
                groups.append(
                    {
                        "label": int(len(records) - 1),
                        "tower_id": int(tower["id"]),
                        "tower_name": tower["name"],
                        "side": side_name,
                        "layer_rank": int(layer_rank),
                        "line_point_count": int(layer_line_ids.size),
                        "seed_insulator_points": int(
                            best_growth["insulator_seed_count"]
                        ),
                        "line_contact_seed_points": int(
                            best_growth["contact_seed_count"]
                        ),
                        "recovered_tower_points": int(recovered_tower_points),
                        "axis_source": growth_axis_source,
                    }
                )

    if grown_global_indices:
        grown_global_indices = np.unique(np.concatenate(grown_global_indices))
    else:
        grown_global_indices = np.empty((0,), dtype=np.int64)
    return records, towers, groups, grown_global_indices


def make_empty_like_input_las(input_las):
    """创建一个空 LAS，但继承原始 LAS 的点格式、坐标缩放、offset 和 CRS。"""
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


def build_marker_arrays(instances, marker_size, marker_step, marker_shape):
    marker_point_list = []
    marker_class_list = []
    marker_color_list = []
    color_map = {
        "top_point_xyz": (65535, 0, 0),        # 最高点：红色
        "bottom_point_xyz": (0, 12000, 65535), # 最低点：蓝色
        "center_point_xyz": (0, 65535, 0),     # 中心真实点：绿色
    }
    class_map = {
        "top_point_xyz": 20,
        "bottom_point_xyz": 21,
        "center_point_xyz": 22,
    }

    for item in instances:
        for key in ("top_point_xyz", "bottom_point_xyz", "center_point_xyz"):
            points = marker_points(item[key], marker_size, marker_step, marker_shape)
            marker_point_list.append(points)
            marker_class_list.append(np.full(points.shape[0], class_map[key], dtype=np.uint8))
            marker_color_list.append(
                np.tile(np.asarray(color_map[key], dtype=np.uint16), (points.shape[0], 1))
            )

    if not marker_point_list:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.uint8),
            np.empty((0, 3), dtype=np.uint16),
        )
    return (
        np.concatenate(marker_point_list, axis=0),
        np.concatenate(marker_class_list, axis=0),
        np.concatenate(marker_color_list, axis=0),
    )


def build_insulator_support_arrays(instances):
    """把每个绝缘子候选点簇统一画成一种颜色。"""
    point_list = []
    for item in instances:
        points = item.get("_visual_points_xyz")
        if points is None:
            continue
        points = np.asarray(points, dtype=np.float64)
        if points.shape[0] == 0:
            continue
        point_list.append(points)

    if not point_list:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.uint8),
            np.empty((0, 3), dtype=np.uint16),
        )

    points = np.concatenate(point_list, axis=0)
    classes = np.full(points.shape[0], 23, dtype=np.uint8)
    # 绝缘子候选统一用亮品红色，区别于蓝色导线、红色杆塔和灰色背景。
    colors = np.tile(
        np.asarray((65535, 0, 65535), dtype=np.uint16), (points.shape[0], 1)
    )
    return points, classes, colors


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
    """按杆塔分别输出局部 LAS，并把该杆塔绝缘子关键点标记追加进去。"""
    if output_dir is None:
        return []

    written = []
    for tower in towers:
        instances = tower.get("insulators", [])
        if not instances and not args.save_towers_without_insulator:
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

        point_arrays = [render_las.points.array[crop_indices]]
        visual_point_count = 0

        if not args.no_draw_insulator_support_points:
            support_points, support_classes, support_colors = build_insulator_support_arrays(
                instances
            )
            support_records = marker_records_like(
                render_las, support_points, support_classes, support_colors
            )
            if support_records.array.shape[0] > 0:
                point_arrays.append(support_records.array)
                visual_point_count += int(support_points.shape[0])

        marker_points_arr = np.empty((0, 3), dtype=np.float64)
        if args.draw_keypoint_markers:
            marker_points_arr, marker_classes, marker_colors = build_marker_arrays(
                instances,
                args.marker_size,
                args.marker_step,
                args.marker_shape,
            )
            marker_records = marker_records_like(
                render_las, marker_points_arr, marker_classes, marker_colors
            )
            if marker_records.array.shape[0] > 0:
                point_arrays.append(marker_records.array)
                visual_point_count += int(marker_points_arr.shape[0])

        output_las = make_empty_like_input_las(render_las)
        combined = np.concatenate(point_arrays)
        output_las.points = laspy.ScaleAwarePointRecord(
            combined,
            output_las.header.point_format,
            output_las.header.scales,
            output_las.header.offsets,
        )

        filename = f"tower_{int(tower['id']):03d}_{tower['name']}_insulator.las"
        output_path = output_dir / filename
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Tower LAS exists, use --overwrite: {output_path}")
        output_las.write(output_path)
        written.append(
            {
                "tower_id": int(tower["id"]),
                "tower_name": tower["name"],
                "path": str(output_path),
                "crop_points": int(crop_indices.shape[0]),
                "visual_insulator_points": int(visual_point_count),
                "marker_points": int(marker_points_arr.shape[0]),
                "insulator_instances": int(len(instances)),
            }
        )
    return written


def build_instance_record(instance_id, points, point_indices, args, tower_tree, line_tree):
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extent = bbox_max - bbox_min
    height = float(extent[2])

    if points.shape[0] < args.min_instance_points:
        return None
    if height < args.min_height or height > args.max_height:
        return None
    if max(float(extent[0]), float(extent[1])) > args.max_extent_xy:
        return None

    distance_to_tower = nearest_distance(
        points, tower_tree, args.max_distance_sample_points
    )
    distance_to_line = nearest_distance(points, line_tree, args.max_distance_sample_points)

    if args.require_near_tower:
        if distance_to_tower is None or distance_to_tower > args.max_distance_to_tower:
            return None
    if args.require_near_line:
        if distance_to_line is None or distance_to_line > args.max_distance_to_line:
            return None

    top_point = points[int(np.argmax(points[:, 2]))]
    bottom_point = points[int(np.argmin(points[:, 2]))]
    center = points.mean(axis=0)
    center_point = points[int(np.argmin(np.sum((points - center[None, :]) ** 2, axis=1)))]

    return {
        "id": int(instance_id),
        "name": f"绝缘子{instance_id}",
        "point_count": int(points.shape[0]),
        "top_point_xyz": to_float_list(top_point),
        "bottom_point_xyz": to_float_list(bottom_point),
        "center_xyz": to_float_list(center),
        "center_point_xyz": to_float_list(center_point),
        "bbox_min_xyz": to_float_list(bbox_min),
        "bbox_max_xyz": to_float_list(bbox_max),
        "extent_xyz": to_float_list(extent),
        "height": round(float(top_point[2] - bottom_point[2]), 6),
        "distance_to_tower": None
        if distance_to_tower is None
        else round(distance_to_tower, 6),
        "distance_to_line": None if distance_to_line is None else round(distance_to_line, 6),
        "original_point_index_top": int(point_indices[int(np.argmax(points[:, 2]))]),
        "original_point_index_bottom": int(point_indices[int(np.argmin(points[:, 2]))]),
        "original_point_index_center": int(
            point_indices[int(np.argmin(np.sum((points - center[None, :]) ** 2, axis=1)))]
        ),
        # 仅用于生成可视化 LAS，写 JSON 时会过滤掉下划线开头字段。
        "_visual_points_xyz": points.astype(np.float64, copy=False),
    }


def marker_cross_points(center, size, step):
    """围绕关键点生成一个 3D 十字，避免 CloudCompare 中单点太小看不见。"""
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
    """围绕关键点生成一个小立方体表面点簇，适合做明显的可视化证明。"""
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


def write_visual_las(
    input_las,
    output_path,
    instances,
    marker_size,
    marker_step,
    marker_shape,
    mode,
    draw_keypoint_markers,
    draw_insulator_support_points,
):
    """输出关键点标记 LAS。

    markers-only：只输出标记点，需要和原始 LAS 叠加查看；
    append-to-input：把标记点追加到原始点云后面，单独打开即可查看。
    """
    output_path = Path(output_path)
    point_parts = []
    class_parts = []
    color_parts = []
    if draw_insulator_support_points:
        points, classes, colors = build_insulator_support_arrays(instances)
        if points.shape[0] > 0:
            point_parts.append(points)
            class_parts.append(classes)
            color_parts.append(colors)
    if draw_keypoint_markers:
        points, classes, colors = build_marker_arrays(
            instances, marker_size, marker_step, marker_shape
        )
        if points.shape[0] > 0:
            point_parts.append(points)
            class_parts.append(classes)
            color_parts.append(colors)

    if point_parts:
        points = np.concatenate(point_parts, axis=0)
        classes = np.concatenate(class_parts, axis=0)
        colors = np.concatenate(color_parts, axis=0)
    else:
        points = np.empty((0, 3), dtype=np.float64)
        classes = np.empty((0,), dtype=np.uint8)
        colors = np.empty((0, 3), dtype=np.uint16)

    if mode == "append-to-input":
        records = laspy.ScaleAwarePointRecord.zeros(points.shape[0], header=input_las.header)
        if points.shape[0]:
            records.x = points[:, 0]
            records.y = points[:, 1]
            records.z = points[:, 2]
            if "classification" in set(input_las.point_format.dimension_names):
                records.classification = classes
            if {"red", "green", "blue"}.issubset(set(input_las.point_format.dimension_names)):
                records.red = colors[:, 0]
                records.green = colors[:, 1]
                records.blue = colors[:, 2]
        combined = np.concatenate([input_las.points.array, records.array])
        input_las.points = laspy.ScaleAwarePointRecord(
            combined,
            input_las.header.point_format,
            input_las.header.scales,
            input_las.header.offsets,
        )
        input_las.write(output_path)
        return int(points.shape[0])

    # 新建空 LAS，不能直接复用原始 header 的 point record，
    # 否则 laspy 会保留原始点数，导致 marker 点数量无法写入。
    header = laspy.LasHeader(
        point_format=input_las.header.point_format,
        version=input_las.header.version,
    )
    header.scales = input_las.header.scales
    header.offsets = input_las.header.offsets
    crs = input_las.header.parse_crs()
    if crs is not None:
        header.add_crs(crs)
    visual_las = laspy.LasData(header)
    if points.shape[0]:
        visual_las.x = points[:, 0]
        visual_las.y = points[:, 1]
        visual_las.z = points[:, 2]
        visual_las.classification = classes
        if {"red", "green", "blue"}.issubset(set(visual_las.point_format.dimension_names)):
            visual_las.red = colors[:, 0]
            visual_las.green = colors[:, 1]
            visual_las.blue = colors[:, 2]
    else:
        visual_las.x = np.asarray([], dtype=np.float64)
        visual_las.y = np.asarray([], dtype=np.float64)
        visual_las.z = np.asarray([], dtype=np.float64)
    visual_las.write(output_path)
    return int(points.shape[0])


def main():
    args = parse_args()
    start_time = time.perf_counter()
    input_path = Path(args.input)
    output_path = ensure_output_path(args.output, args.overwrite)
    visual_output_path = ensure_visual_output_path(args.visual_las_output, args.overwrite)
    segmented_las_output_path = ensure_visual_output_path(
        args.segmented_las_output, args.overwrite
    )
    corrected_las_output_path = ensure_visual_output_path(
        args.corrected_las_output, args.overwrite
    )
    tower_las_output_dir = ensure_output_dir(args.tower_las_output_dir)

    print(f"Reading {input_path}", flush=True)
    coord, cls, las = read_segmented_cloud(input_path)
    render_las = las
    render_coord = coord
    if args.render_base_las is not None:
        render_path = Path(args.render_base_las)
        print(f"Reading render base LAS {render_path}", flush=True)
        render_coord, render_las = read_render_cloud(render_path)

    insulator_mask = cls == int(args.insulator_class)
    tower_mask = cls == int(args.tower_class)
    line_mask = cls == int(args.line_class)
    raw_insulator_coord = coord[insulator_mask]
    raw_insulator_indices = np.where(insulator_mask)[0]
    if args.use_insulator_class:
        insulator_coord = raw_insulator_coord
        insulator_indices = raw_insulator_indices
    else:
        insulator_coord = np.empty((0, 3), dtype=np.float64)
        insulator_indices = np.empty((0,), dtype=np.int64)
    tower_coord = coord[tower_mask]
    tower_indices = np.where(tower_mask)[0]
    line_coord = coord[line_mask]
    line_indices = np.where(line_mask)[0]

    print(
        f"Loaded {coord.shape[0]} points; "
        f"raw_insulator={raw_insulator_coord.shape[0]}, "
        f"used_insulator={insulator_coord.shape[0]}, "
        f"tower={tower_coord.shape[0]}, line={line_coord.shape[0]}",
        flush=True,
    )

    if segmented_las_output_path is not None:
        las.write(segmented_las_output_path)
        print(f"Wrote segmented LAS: {segmented_las_output_path}", flush=True)

    tower_tree = cKDTree(tower_coord) if tower_coord.shape[0] else None
    line_tree = cKDTree(line_coord) if line_coord.shape[0] else None

    towers = []
    structured_groups = []
    raw_component_count = 0
    instances = []
    labels = np.empty(0, dtype=np.int32)
    grown_reclassified_indices = np.empty((0,), dtype=np.int64)

    if args.cluster_method == "tower_shape":
        instances, towers, structured_groups, grown_reclassified_indices = (
            hierarchical_line_group_instances(
                insulator_coord,
                insulator_indices,
                tower_coord,
                tower_indices,
                line_coord,
                args,
            )
        )
        raw_component_count = len(instances)
    elif args.cluster_method == "attachment":
        instances, towers, structured_groups = attachment_based_instances(
            insulator_coord,
            insulator_indices,
            tower_coord,
            tower_indices,
            line_coord,
            line_indices,
            args,
        )
        raw_component_count = len(instances)
    elif args.cluster_method == "seed_grow":
        instances, towers, structured_groups, grown_reclassified_indices = (
            seed_grow_insulator_instances(
                insulator_coord,
                insulator_indices,
                tower_coord,
                tower_indices,
                line_coord,
                line_indices,
                args,
            )
        )
        raw_component_count = len(instances)
    elif insulator_coord.shape[0] == 0:
        labels = np.empty(0, dtype=np.int32)
    elif args.cluster_method == "tower_layer_side":
        labels, towers, structured_groups = tower_layer_side_labels(
            insulator_coord, tower_coord, line_coord, args
        )
    elif args.cluster_method == "dbscan":
        labels = dbscan_labels(insulator_coord, args.eps, args.min_samples)
    else:
        labels = voxel_component_labels(insulator_coord, args.voxel_size)

    if args.cluster_method not in ("tower_shape", "attachment", "seed_grow"):
        raw_component_count = int(len(set(labels.tolist())) - (1 if -1 in labels else 0))
        group_meta = {int(item["label"]): item for item in structured_groups}
        for label in sorted(set(labels.tolist())):
            if label < 0:
                continue
            mask = labels == label
            record = build_instance_record(
                len(instances) + 1,
                insulator_coord[mask],
                insulator_indices[mask],
                args,
                tower_tree,
                line_tree,
            )
            if record is not None:
                meta = group_meta.get(int(label))
                if meta is not None:
                    record["tower_id"] = meta["tower_id"]
                    record["tower_name"] = meta["tower_name"]
                    record["side"] = meta["side"]
                    record["axis_source"] = meta["axis_source"]
                    record["near_line_points_for_axis"] = meta["near_line_points_for_axis"]
                instances.append(record)

    # 按全局空间顺序重新编号，保证多次运行输出稳定。
    if args.cluster_method == "tower_shape":
        instances.sort(
            key=lambda item: (
                int(item.get("tower_id", 0)),
                int(item.get("layer_rank", 0)),
                0 if item.get("side") == "left" else 1,
            )
        )
    else:
        instances.sort(
            key=lambda item: (
                item["center_xyz"][0],
                item["center_xyz"][1],
                -item["center_xyz"][2],
            )
        )
    for idx, item in enumerate(instances, start=1):
        item["id"] = idx
        item["name"] = f"绝缘子{idx}"

    tower_las_files = []
    if tower_las_output_dir is not None:
        tower_start = time.perf_counter()
        print("Clustering tower instances for per-tower LAS output", flush=True)
        if not towers:
            towers = build_tower_instances(tower_coord, args)
        if args.cluster_method in ("tower_shape", "attachment", "seed_grow"):
            # attachment/seed_grow 模式已经在推断阶段把绝缘子绑定到了具体杆塔。
            # 这里按最终排序后的实例重新填充，避免再次用包围盒外扩距离误删远端横担绝缘子。
            tower_by_id = {int(tower["id"]): tower for tower in towers}
            for tower in towers:
                tower["insulators"] = []
            for item in instances:
                tower = tower_by_id.get(int(item.get("tower_id", -1)))
                if tower is not None:
                    tower["insulators"].append(item)
        else:
            assign_instances_to_towers(
                instances,
                towers,
                args.tower_bind_xy_margin,
                args.tower_bind_z_margin,
            )
        tower_las_files = write_tower_las_files(
            render_las,
            render_coord,
            towers,
            tower_las_output_dir,
            args,
        )
        print(
            f"Wrote {len(tower_las_files)} tower LAS files in "
            f"{time.perf_counter() - tower_start:.2f}s",
            flush=True,
        )

    if corrected_las_output_path is not None:
        corrected_cls = np.asarray(las.classification, dtype=np.uint8).copy()
        if grown_reclassified_indices.size:
            corrected_cls[grown_reclassified_indices] = int(args.insulator_class)
        las.classification = corrected_cls
        las.write(corrected_las_output_path)
        print(
            f"Wrote corrected LAS: {corrected_las_output_path} "
            f"(reclassified_tower_to_insulator={grown_reclassified_indices.size})",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time
    data = {
        "input": str(input_path),
        "insulator_class": int(args.insulator_class),
        "coordinate_system": "las_global_xyz",
        "cluster_method": args.cluster_method,
        "parameters": {
            "voxel_size": float(args.voxel_size),
            "eps": float(args.eps),
            "min_samples": int(args.min_samples),
            "insulator_layer_z_gap": float(args.insulator_layer_z_gap),
            "min_insulator_side_distance": float(args.min_insulator_side_distance),
            "merge_same_side_layer_insulators": bool(
                args.merge_same_side_layer_insulators
            ),
            "axis_line_search_radius": float(args.axis_line_search_radius),
            "tower_shape_z_bin": float(args.tower_shape_z_bin),
            "tower_shape_core_quantile": float(args.tower_shape_core_quantile),
            "tower_shape_core_margin": float(args.tower_shape_core_margin),
            "tower_shape_center_search_margin": float(
                args.tower_shape_center_search_margin
            ),
            "tower_shape_min_z_ratio": float(args.tower_shape_min_z_ratio),
            "tower_shape_cluster_radius": float(
                args.tower_shape_cluster_radius
            ),
            "tower_shape_min_points": int(args.tower_shape_min_points),
            "tower_shape_min_linearity": float(args.tower_shape_min_linearity),
            "tower_shape_min_x_alignment": float(
                args.tower_shape_min_x_alignment
            ),
            "tower_shape_min_length": float(args.tower_shape_min_length),
            "tower_shape_max_length": float(args.tower_shape_max_length),
            "tower_shape_max_width": float(args.tower_shape_max_width),
            "tower_shape_max_line_distance": float(
                args.tower_shape_max_line_distance
            ),
            "tower_shape_layer_z_gap": float(args.tower_shape_layer_z_gap),
            "tower_shape_ground_z_margin": float(
                args.tower_shape_ground_z_margin
            ),
            "expected_insulated_layers": int(args.expected_insulated_layers),
            "insulators_per_line_group": int(args.insulators_per_line_group),
            "line_group_sample_distance": float(args.line_group_sample_distance),
            "line_group_sample_window": float(args.line_group_sample_window),
            "line_group_yz_radius": float(args.line_group_yz_radius),
            "line_group_min_points": int(args.line_group_min_points),
            "line_group_corridor_radius": float(args.line_group_corridor_radius),
            "tower_shape_require_symmetry": bool(
                not args.no_tower_shape_require_symmetry
            ),
            "tower_shape_symmetry_x_tolerance": float(
                args.tower_shape_symmetry_x_tolerance
            ),
            "tower_shape_symmetry_y_tolerance": float(
                args.tower_shape_symmetry_y_tolerance
            ),
            "tower_shape_symmetry_z_tolerance": float(
                args.tower_shape_symmetry_z_tolerance
            ),
            "min_instance_points": int(args.min_instance_points),
            "min_height": float(args.min_height),
            "max_height": float(args.max_height),
            "max_extent_xy": float(args.max_extent_xy),
            "require_near_tower": bool(args.require_near_tower),
            "require_near_line": bool(args.require_near_line),
            "max_distance_to_tower": float(args.max_distance_to_tower),
            "max_distance_to_line": float(args.max_distance_to_line),
            "use_insulator_class": bool(args.use_insulator_class),
            "attachment_min_along_distance": float(args.attachment_min_along_distance),
            "attachment_max_along_distance": float(args.attachment_max_along_distance),
            "attachment_side_width": float(args.attachment_side_width),
            "attachment_z_gap": float(args.attachment_z_gap),
            "attachment_min_line_points": int(args.attachment_min_line_points),
            "attachment_end_window": float(args.attachment_end_window),
            "attachment_insulator_search_radius": float(
                args.attachment_insulator_search_radius
            ),
            "attachment_tower_search_radius": float(args.attachment_tower_search_radius),
            "attachment_estimated_height": float(args.attachment_estimated_height),
            "attachment_min_tower_z_ratio": float(args.attachment_min_tower_z_ratio),
            "attachment_skip_top_layers": int(args.attachment_skip_top_layers),
            "seed_grow_candidate_radius": float(args.seed_grow_candidate_radius),
            "seed_grow_endpoint_seed_radius": float(
                args.seed_grow_endpoint_seed_radius
            ),
            "seed_grow_line_contact_radius": float(
                args.seed_grow_line_contact_radius
            ),
            "seed_grow_line_contact_seeds": bool(
                not args.no_seed_grow_line_contact_seeds
            ),
            "seed_grow_core_z_window": float(args.seed_grow_core_z_window),
            "seed_grow_core_x_quantile": float(
                args.seed_grow_core_x_quantile
            ),
            "seed_grow_core_x_margin": float(args.seed_grow_core_x_margin),
            "seed_grow_core_entry_margin": float(
                args.seed_grow_core_entry_margin
            ),
            "seed_grow_neighbor_radius": float(args.seed_grow_neighbor_radius),
            "seed_grow_min_seed_points": int(args.seed_grow_min_seed_points),
            "seed_grow_min_points": int(args.seed_grow_min_points),
            "seed_grow_min_length": float(args.seed_grow_min_length),
            "seed_grow_max_length": float(args.seed_grow_max_length),
            "seed_grow_max_width": float(args.seed_grow_max_width),
            "seed_grow_junction_bin_size": float(
                args.seed_grow_junction_bin_size
            ),
            "seed_grow_junction_width_ratio": float(
                args.seed_grow_junction_width_ratio
            ),
            "seed_grow_junction_count_ratio": float(
                args.seed_grow_junction_count_ratio
            ),
            "seed_grow_backward_margin": float(args.seed_grow_backward_margin),
            "seed_grow_min_tower_core_distance": float(
                args.seed_grow_min_tower_core_distance
            ),
            "insulator_visual_radius": float(args.insulator_visual_radius),
            "insulator_visual_step": float(args.insulator_visual_step),
            "draw_keypoint_markers": bool(args.draw_keypoint_markers),
            "draw_insulator_support_points": bool(
                not args.no_draw_insulator_support_points
            ),
            "corrected_las_output": None
            if corrected_las_output_path is None
            else str(corrected_las_output_path),
            "tower_las_output_dir": None
            if tower_las_output_dir is None
            else str(tower_las_output_dir),
            "render_base_las": args.render_base_las,
            "tower_voxel_size": float(args.tower_voxel_size),
            "min_tower_points": int(args.min_tower_points),
            "min_tower_height": float(args.min_tower_height),
            "tower_bind_xy_margin": float(args.tower_bind_xy_margin),
            "tower_bind_z_margin": float(args.tower_bind_z_margin),
            "tower_crop_xy_margin": float(args.tower_crop_xy_margin),
            "tower_crop_z_margin": float(args.tower_crop_z_margin),
        },
        "summary": {
            "total_points": int(coord.shape[0]),
            "raw_insulator_points": int(raw_insulator_coord.shape[0]),
            "used_insulator_points": int(insulator_coord.shape[0]),
            "raw_components": int(raw_component_count),
            "kept_instances": int(len(instances)),
            "seed_grow_reclassified_tower_points": int(
                grown_reclassified_indices.size
            ),
            "tower_instances": int(len(towers)),
            "tower_las_files": int(len(tower_las_files)),
            "elapsed_sec": round(elapsed, 3),
        },
        "tower_las_files": tower_las_files,
        "instances": [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in instances
        ],
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    if visual_output_path is not None:
        marker_count = write_visual_las(
            render_las,
            visual_output_path,
            instances,
            args.marker_size,
            args.marker_step,
            args.marker_shape,
            args.visual_las_mode,
            args.draw_keypoint_markers,
            not args.no_draw_insulator_support_points,
        )
        print(
            f"Wrote visual LAS markers: {visual_output_path} "
            f"(mode={args.visual_las_mode}, marker_points={marker_count})",
            flush=True,
        )

    print(
        f"Wrote {output_path} "
        f"(instances={len(instances)}, elapsed={elapsed:.2f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()





""" 
python tools/infer/extract_insulator_points.py \
  --input /24085403037/24085403037/shixi/dataset/test_lidar/110v12_tile010_pred.ply \
  --output /24085403037/24085403037/shixi/dataset/6_23_demo/test/insulator_tower_ply_110V12.json \
  --visual-las-output /24085403037/24085403037/shixi/dataset/6_23_demo/test/insulator_tower_ply_110V12_markers.las \
  --cluster-method tower_layer_side \
  --insulator-layer-z-gap 2.0 \
  --min-insulator-side-distance 1.0 \
  --marker-shape cube \
  --overwrite
  
  python tools/infer/extract_insulator_points.py \
  --input /24085403037/24085403037/shixi/dataset/6_23_demo/test/original/tower_004_杆塔4.las \
  --output /24085403037/24085403037/shixi/dataset/6_23_demo/test/insulator/tower_004_杆塔4.json \
  --visual-las-output /24085403037/24085403037/shixi/dataset/6_23_demo/test/tower_004_杆塔4_output.las \
  --cluster-method attachment \
  --attachment-min-line-points 80 \
  --attachment-insulator-search-radius 2.5 \
  --attachment-tower-search-radius 3.0 \
  --attachment-skip-top-layers 1 \
  --overwrite
   
   
   python tools/infer/extract_insulator_points.py \
  --input /24085403037/24085403037/shixi/dataset/6_23_demo/test/original/tower_004_杆塔4.las \
  --output /24085403037/24085403037/shixi/dataset/6_23_demo/test/insulator/tower_004_杆塔4.json \
  --visual-las-output /24085403037/24085403037/shixi/dataset/6_23_demo/test/tower_004_杆塔4_output_insulator.las \
  --corrected-las-output /24085403037/24085403037/shixi/dataset/6_23_demo/test/tower_004_杆塔4_output.las \
  --cluster-method tower_shape \
  --attachment-skip-top-layers 1 \
  --expected-insulated-layers 4 \
  --insulators-per-line-group 2 \
  --line-group-sample-distance 6.0 \
  --line-group-sample-window 1.5 \
  --line-group-yz-radius 0.75 \
  --line-group-min-points 20 \
  --line-group-corridor-radius 1.20 \
  --tower-shape-z-bin 0.50 \
  --tower-shape-layer-z-gap 1.20 \
  --tower-shape-cluster-radius 0.25 \
  --tower-shape-min-points 15 \
  --tower-shape-min-linearity 0.60 \
  --tower-shape-min-x-alignment 0.70 \
  --tower-shape-max-line-distance 0.50 \
  --tower-shape-symmetry-x-tolerance 1.00 \
  --tower-shape-symmetry-y-tolerance 0.80 \
  --tower-shape-symmetry-z-tolerance 0.80 \
  --overwrite
"""
