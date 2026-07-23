"""在清理后的 LAS 上绘制杆塔框、档距框和拟合导线。

输入通常是 postprocess_tower_line_clean.py 输出的清理后 LAS：

    classification 0=background, 1=tower, 2=line, 3=insulator

本脚本是拆分后处理流程的第 2 步，主要流程：
1. 根据清理后的语义标签重新构建有效物理杆塔；
2. 为保留杆塔生成杆塔 OBB 框；
3. 为相邻杆塔之间的导线生成档距 OBB 框；
4. 在每个档距内拟合、编号并着色每根导线；
5. 追加 class 30/31 可视化点，并写出框和导线 JSON。

如果 LAS 点格式支持 RGB，脚本会同步更新颜色用于 CloudCompare 查看。
"""
# 后处理的脚本：
# 本脚本不再改写杆塔误检标签，输入应优先使用清理脚本输出的 LAS。
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
    0: (37008, 37008, 37008),  # 背景/地面，灰色
    1: (58880, 16640, 14080),  # 杆塔，红色
    2: (11520, 32000, 65280),  # 导线，蓝色
    3: (65280, 53760, 10240),  # 绝缘子，黄色
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
    parser.add_argument(
        "--preset",
        choices=("transmission-line", "legacy"),
        default="transmission-line",
        help=(
            "Default workflow preset. transmission-line enables the integrated "
            "tower/line/insulator filtering used by this project. legacy restores "
            "the older loose defaults."
        ),
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
            "Origin used for obb[0:3] relative coordinates. For box JSON, "
            "obb_global.las_origin stores this origin. Default: LAS header offsets."
        ),
    )# 控制JSON中的OBB旋转框使用什么相对坐标原点
    # las-offset  使用 LAS 文件头 offset，推荐
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
    parser.add_argument(
        "--json-box-orientation",
        choices=("fitted", "north"),
        default="fitted",
        help=(
            "fitted keeps fitted OBB boxes and exports right-handed rotation "
            "relative to LAS X+ north; north writes an unrotated north-facing "
            "axis-aligned JSON box."
        ),
    )# JSON框朝向。fitted表示右手坐标系下按LAS X+正北导出旋转；north表示强制不旋转的正北轴对齐框。

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
        default=200,
        help="Remove tower components with fewer original points than this.",
    )# 杆塔组件最少点数。低于这个点数的杆塔候选框会被当做误检删掉
    parser.add_argument(
        "--min-tower-height",
        type=float,
        default=6.0,
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
        default=True,
        help="Keep only tower components that have line points nearby.",
    )# 要求杆塔附近必须有导线点
    parser.add_argument(
        "--tower-line-radius",
        type=float,
        default=18.0,
        help="Radius in meters for --require-line-near-tower.",
    )# 判断“杆塔附近是否有导线”的搜索半径，单位为米.
    parser.add_argument(
        "--min-line-points-near-tower",
        type=int,
        default=100,
        help=(
            "Remove tower components with fewer line points than this inside "
            "--tower-line-radius. 0 disables this filter."
        ),
    )# 杆塔附近至少要有多少导线点。0 表示不启用这个过滤。
    parser.add_argument(
        "--require-line-through-tower",
        action="store_true",
        help=(
            "Keep only tower components whose expanded XY footprint contains "
            "enough line points in the upper tower height range."
        ),
    )# 更严格：要求导线点穿过/落在杆塔候选区域上部，适合过滤线路走廊旁边的杆塔误检。
    parser.add_argument(
        "--tower-line-through-xy-margin",
        type=float,
        default=3.0,
        help="XY margin in meters around tower component bbox for through-line filter.",
    )# 判断导线是否穿过杆塔候选区域时，杆塔XY框向外扩的距离。
    parser.add_argument(
        "--tower-line-through-z-margin",
        type=float,
        default=3.0,
        help="Z margin in meters around the upper tower range for through-line filter.",
    )# 判断导线高度时，上下放宽的距离。
    parser.add_argument(
        "--tower-line-through-cross-margin",
        type=float,
        default=1.0,
        help=(
            "Extra along-line distance beyond both sides of the tower footprint "
            "required by --require-line-through-tower."
        ),
    )# 穿越判定：导线点必须超过杆塔沿线路方向的两侧边界。
    parser.add_argument(
        "--min-line-points-through-tower-side",
        type=int,
        default=5,
        help="Minimum through-line points required on each side of a tower.",
    )# 穿越判定：杆塔前后两侧各自最少导线点数。
    parser.add_argument(
        "--tower-line-through-min-height-ratio",
        type=float,
        default=0.45,
        help=(
            "Only count line points above this fraction of the tower component "
            "height for --require-line-through-tower."
        ),
    )# 只统计杆塔高度上部的导线点，避免地面附近误检被保留。
    parser.add_argument(
        "--min-line-points-through-tower",
        type=int,
        default=20,
        help="Minimum line points required by --require-line-through-tower.",
    )# 穿过杆塔候选区域的最少导线点数。
    parser.add_argument(
        "--require-line-touch-tower",
        action="store_true",
        help=(
            "Keep only physical towers whose upper tower points are close "
            "enough to line points in 3D. This removes towers below/near "
            "lines but not actually touched by lines."
        ),
    )# 要求杆塔上部点和导线点在三维空间真正接近/接触，用于删除黄色框这种假塔。
    parser.add_argument(
        "--tower-line-contact-radius",
        type=float,
        default=None,
        help=(
            "3D distance threshold in meters for line-touch filtering. If omitted, "
            "--tower-line-touch-xy-margin is used for backward compatibility."
        ),
    )# 杆塔点到导线点的三维接触距离阈值。越小越严格。
    parser.add_argument(
        "--tower-line-touch-xy-margin",
        type=float,
        default=1.5,
        help=(
            "Backward-compatible contact radius in meters when "
            "--tower-line-contact-radius is not set."
        ),
    )# 兼容旧参数：未设置contact-radius时，用它作为三维接触距离阈值。
    parser.add_argument(
        "--tower-line-touch-z-margin",
        type=float,
        default=2.0,
        help="Deprecated compatibility option; not used by the 3D contact filter.",
    )# 兼容旧命令保留；当前三维接触过滤不再使用它。
    parser.add_argument(
        "--tower-line-touch-min-height-ratio",
        type=float,
        default=0.45,
        help="Only count line points above this fraction of tower height.",
    )# 只统计杆塔上部的导线点，避免地面附近误检影响。
    parser.add_argument(
        "--min-line-points-touch-tower",
        type=int,
        default=20,
        help="Minimum upper tower points close to line points required to keep a tower.",
    )# 杆塔上部至少有多少个点和导线点接近，才保留该杆塔。
    parser.add_argument(
        "--require-line-inside-tower",
        action="store_true",
        help=(
            "Keep only physical towers with enough line points inside the "
            "middle/upper core of the tower OBB."
        ),
    )# 要求杆塔中部/上部核心区域内必须有足够导线点。
    parser.add_argument(
        "--tower-line-inside-xy-scale",
        type=float,
        default=0.45,
        help="Fraction of tower OBB half XY size used as the inside/core region.",
    )# 塔体内部核心区域的XY比例，越小越严格。
    parser.add_argument(
        "--tower-line-inside-min-height-ratio",
        type=float,
        default=0.45,
        help="Only count line points above this fraction of tower height.",
    )# 只统计杆塔高度上部/中部的导线点。
    parser.add_argument(
        "--min-line-points-inside-tower",
        type=int,
        default=80,
        help="Minimum line points inside the tower core required to keep a tower.",
    )# 杆塔核心区域内最少导线点数量。
    parser.add_argument(
        "--require-continuous-line-inside-tower",
        action="store_true",
        help=(
            "Keep only physical towers where line points form a continuous "
            "track inside the tower OBB core."
        ),
    )# 要求线路点在杆塔框核心区域内连续进入，而不是零散点或擦边点。
    parser.add_argument(
        "--continuous-line-inside-xy-scale",
        type=float,
        default=0.35,
        help="Fraction of tower OBB half XY size used by the continuous-line core.",
    )# 连续线路判定使用的塔框XY核心比例，越小越严格。
    parser.add_argument(
        "--continuous-line-inside-min-height-ratio",
        type=float,
        default=0.45,
        help="Only count continuous line points above this fraction of tower height.",
    )# 连续线路判定只看杆塔中上部。
    parser.add_argument(
        "--continuous-line-bin-size",
        type=float,
        default=0.50,
        help="Along-line bin size in meters for continuous inside-line filtering.",
    )# 沿线路方向分箱的长度。
    parser.add_argument(
        "--min-continuous-line-bins",
        type=int,
        default=6,
        help="Minimum consecutive occupied bins required inside the tower core.",
    )# 至少连续占据多少个bin。
    parser.add_argument(
        "--min-continuous-line-length",
        type=float,
        default=3.0,
        help="Minimum continuous along-line length in meters inside the tower core.",
    )# 杆塔框内连续线路段最小长度。
    parser.add_argument(
        "--min-continuous-line-points",
        type=int,
        default=60,
        help="Minimum total line points in the continuous tower-core region.",
    )# 连续线路核心区域内最少线路点数量。
    parser.add_argument(
        "--require-side-line-near-tower",
        action="store_true",
        help=(
            "Keep only physical towers with enough line points near the left "
            "or right side face of the tower box, excluding lines above the box."
        ),
    )# 要求杆塔框左侧或右侧必须有线路点，上方飘过不算。
    parser.add_argument(
        "--tower-side-line-window",
        type=float,
        default=2.0,
        help="Along-line window in meters around each tower side face.",
    )# 杆塔左右侧面附近沿线路方向的统计窗口。
    parser.add_argument(
        "--tower-side-line-xy-margin",
        type=float,
        default=1.5,
        help="Perpendicular XY margin in meters for side-line filtering.",
    )# 侧边线路判定的横向容差。
    parser.add_argument(
        "--tower-side-line-z-margin",
        type=float,
        default=0.3,
        help="Allowed meters above tower box top; small values reject overhead lines.",
    )# 允许超过杆塔框顶部的高度，越小越能排除上方线路。
    parser.add_argument(
        "--tower-side-line-min-height-ratio",
        type=float,
        default=0.35,
        help="Only count side line points above this fraction of tower height.",
    )# 只统计杆塔中上部侧边线路点。
    parser.add_argument(
        "--min-side-line-points",
        type=int,
        default=30,
        help="Minimum line points required on either tower side face.",
    )# 左侧或右侧至少需要多少线路点。
    parser.add_argument(
        "--require-insulator-near-tower",
        action="store_true",
        help=(
            "Keep only physical towers with enough insulator points near the "
            "upper tower box. Useful for removing pole-like false positives "
            "that are under lines but not connected to them."
        ),
    )# 更严格：要求物理杆塔上部附近必须有绝缘子点。
    parser.add_argument(
        "--tower-insulator-xy-margin",
        type=float,
        default=6.0,
        help="XY margin in meters around tower OBB when searching insulator points.",
    )# 搜索绝缘子点时，杆塔旋转框 XY 外扩距离。
    parser.add_argument(
        "--tower-insulator-z-margin",
        type=float,
        default=6.0,
        help="Z margin in meters around the upper tower range for insulator filter.",
    )# 搜索绝缘子点时，高度方向放宽距离。
    parser.add_argument(
        "--tower-insulator-min-height-ratio",
        type=float,
        default=0.35,
        help="Only count insulator points above this fraction of tower height.",
    )# 只统计杆塔上部的绝缘子点，避免地面噪声干扰。
    parser.add_argument(
        "--min-insulator-points-near-tower",
        type=int,
        default=10,
        help="Minimum insulator points required by --require-insulator-near-tower.",
    )# 每座物理杆塔附近至少需要多少绝缘子点。
    parser.add_argument(
        "--require-insulator-line-bridge",
        action="store_true",
        default=True,
        help=(
            "Keep only physical towers with enough nearby insulator points that "
            "are also close to line points."
        ),
    )# 要求杆塔附近的绝缘子必须和导线相邻，作为塔-线连接桥。
    parser.add_argument(
        "--insulator-line-radius",
        type=float,
        default=1.2,
        help="3D radius in meters used to test whether an insulator point is close to line.",
    )# 绝缘子点到导线点的三维距离阈值。
    parser.add_argument(
        "--min-bridged-insulator-points",
        type=int,
        default=20,
        help="Minimum nearby insulator points close to line points required to keep a tower.",
    )# 同时靠近杆塔和导线的绝缘子点最少数量。
    parser.add_argument(
        "--remove-bare-pole-towers",
        action="store_true",
        help=(
            "Remove pole-like tower false positives whose upper tower points "
            "do not spread enough in XY. This targets bare poles under lines."
        ),
    )# 删除上部没有横向展开的光杆/瘦杆误检。
    parser.add_argument(
        "--bare-pole-upper-height-ratio",
        type=float,
        default=0.45,
        help="Use tower points above this height ratio to detect bare poles.",
    )# 判断光杆时，只看杆塔上部点。
    parser.add_argument(
        "--min-upper-tower-width",
        type=float,
        default=4.0,
        help=(
            "Minimum max XY spread in meters for upper tower points. Smaller "
            "upper spread is treated as a bare pole when --remove-bare-pole-towers is enabled."
        ),
    )# 杆塔上部横向展开最小宽度，低于该值认为是光杆。
    parser.add_argument(
        "--min-upper-tower-area",
        type=float,
        default=4.0,
        help=(
            "Minimum upper XY footprint area in square meters. Used together "
            "with --min-upper-tower-width for bare-pole filtering."
        ),
    )# 杆塔上部横向包围面积，低于该值认为是光杆。
    parser.add_argument(
        "--min-upper-tower-points",
        type=int,
        default=200,
        help="Minimum upper tower points required to keep a tower in bare-pole filtering.",
    )# 杆塔上部最少点数，太少也认为不像真实杆塔结构。
    parser.add_argument(
        "--min-tower-height-above-ground",
        type=float,
        default=4.0,
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
        "--require-tower-top-line",
        action="store_true",
        default=True,
        help=(
            "Keep only physical towers whose top area has enough nearby line "
            "points. This is useful for sparse pole-style segmentation where "
            "ground objects are often mislabeled as tower."
        ),
    )# 稀疏杆塔输入适配：杆塔顶部附近必须有导线点。
    parser.add_argument(
        "--tower-top-line-xy-margin",
        type=float,
        default=6.0,
        help="XY margin in meters around the tower footprint for top-line filtering.",
    )# 顶部接线过滤：杆塔XY范围外扩距离。
    parser.add_argument(
        "--tower-top-line-z-below",
        type=float,
        default=5.0,
        help="Meters below tower top used when counting nearby line points.",
    )# 顶部接线过滤：从杆塔顶部向下统计多少米。
    parser.add_argument(
        "--tower-top-line-z-above",
        type=float,
        default=8.0,
        help="Meters above tower top used when counting nearby line points.",
    )# 顶部接线过滤：从杆塔顶部向上统计多少米。
    parser.add_argument(
        "--min-tower-top-line-points",
        type=int,
        default=30,
        help="Minimum line points near tower top required to keep a physical tower.",
    )# 顶部附近至少需要多少导线点。
    parser.add_argument(
        "--merge-tower-xy-radius",
        type=float,
        default=10.0,
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
        default=20.0,
        help="Max XY distance from the tower-to-tower segment when selecting line points.",
    )
    parser.add_argument(
        "--span-end-margin",
        type=float,
        default=12.0,
        help="Extra meters before/after adjacent tower centers when selecting line spans.",
    )
    parser.add_argument(
        "--min-span-line-points",
        type=int,
        default=200,
        help="Skip a between-tower line box if fewer line points are selected.",
    )
    parser.add_argument(
        "--require-connected-line-span",
        action="store_true",
        help=(
            "Keep only physical towers that are used by at least one valid "
            "line span with the left or right neighboring tower."
        ),
    )# 只保留至少和左/右相邻杆塔通过有效线路档距相连的杆塔。
    parser.add_argument(
        "--span-end-contact-window",
        type=float,
        default=8.0,
        help=(
            "Meters from each tower box edge used to verify that line points "
            "connect to both ends of a span."
        ),
    )# 判断档距是否真正接入杆塔时，每个杆塔端部附近统计线路点的窗口长度。
    parser.add_argument(
        "--min-span-end-contact-points",
        type=int,
        default=20,
        help="Minimum line points required near each tower end of a connected span.",
    )# 有效档距左右两端各自至少需要多少线路点。
    parser.add_argument(
        "--edge-step",
        type=float,
        default=0.30,
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
        "--conductor-layer-z-gap",
        type=float,
        default=2.0,
        help=(
            "Vertical gap in meters used to group conductors into height layers "
            "before numbering. Layers are numbered top-to-bottom; conductors in "
            "the same layer are numbered by local side from small to large."
        ),
    )
    parser.add_argument(
        "--no-append-conductor-fit",
        action="store_true",
        help="Color original line points only; do not append fitted conductor points.",
    )
    parser.add_argument(
        "--no-append-conductor-labels",
        action="store_true",
        help="Do not append synthetic point labels showing conductor numbers.",
    )
    parser.add_argument(
        "--conductor-label-size",
        type=float,
        default=1.6,
        help="Height in meters of appended conductor number labels.",
    )
    parser.add_argument(
        "--conductor-label-step",
        type=float,
        default=0.15,
        help="Point spacing in meters for conductor number labels.",
    )
    parser.add_argument(
        "--conductor-label-side-offset",
        type=float,
        default=0.0,
        help="Side-direction offset in meters from the fitted conductor center.",
    )
    parser.add_argument(
        "--conductor-label-z-offset",
        type=float,
        default=0.8,
        help=(
            "Vertical offset in meters from the fitted conductor center to the "
            "center of the appended number label."
        ),
    )
    parser.add_argument(
        "--keep-existing-synthetic-points",
        action="store_true",
        help=(
            "Keep existing synthetic points with classes 30/31 from the input. "
            "By default they are removed before appending new conductor labels, "
            "fitted conductor points, and box edges."
        ),
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
    args = parser.parse_args()
    apply_legacy_preset(args)
    return args


def apply_legacy_preset(args):
    """在显式请求时恢复较宽松的旧版默认参数。

    当前项目默认使用集成后的输电线路工作流，正常生产使用只需要传入
    input/output。legacy preset 仅用于复现实验或调试旧流程。
    """
    if args.preset != "legacy":
        return
    args.min_tower_points = 500
    args.min_tower_height = 4.0
    args.require_line_near_tower = False
    args.tower_line_radius = 8.0
    args.min_line_points_near_tower = 0
    args.require_insulator_line_bridge = False
    args.min_tower_height_above_ground = 0.0
    args.require_tower_top_line = False
    args.merge_tower_xy_radius = 12.0
    args.line_corridor_width = 12.0
    args.span_end_margin = 5.0
    args.min_span_line_points = 50
    args.edge_step = 0.25


def coords_from_las(las):
    return np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)


def has_rgb(las):
    dims = set(las.point_format.dimension_names)
    return {"red", "green", "blue"}.issubset(dims)


def remove_existing_synthetic_points(las, classes=(FITTED_CONDUCTOR_CLASS, BOX_CLASS)):
    cls = np.asarray(las.classification, dtype=np.uint8)
    keep = ~np.isin(cls, np.asarray(classes, dtype=np.uint8))
    removed = int(cls.size - np.count_nonzero(keep))
    if removed == 0:
        return las, 0
    kept = las.points.array[keep]
    las.points = laspy.ScaleAwarePointRecord(
        kept, las.header.point_format, las.header.scales, las.header.offsets
    )
    return las, removed


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
                # 对称邻域只取一半，避免重复建边。
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
        comp["through_line_points"] = 0
        comp["height_above_ground"] = None
        if line_tree is not None and (
            args.require_line_near_tower or args.min_line_points_near_tower > 0
        ):
            hits = line_tree.query_ball_point(comp["center"][:2], args.tower_line_radius)
            comp["nearby_line_points"] = int(len(hits))
        if args.require_line_through_tower and line_coord.shape[0]:
            xy_margin = float(args.tower_line_through_xy_margin)
            z_margin = float(args.tower_line_through_z_margin)
            height_ratio = float(args.tower_line_through_min_height_ratio)
            height_ratio = min(max(height_ratio, 0.0), 1.0)
            lo_xy = comp["bbox_min"][:2] - xy_margin
            hi_xy = comp["bbox_max"][:2] + xy_margin
            z_lo = comp["bbox_min"][2] + float(size[2]) * height_ratio - z_margin
            z_hi = comp["bbox_max"][2] + z_margin
            mask = (
                (line_coord[:, 0] >= lo_xy[0])
                & (line_coord[:, 0] <= hi_xy[0])
                & (line_coord[:, 1] >= lo_xy[1])
                & (line_coord[:, 1] <= hi_xy[1])
                & (line_coord[:, 2] >= z_lo)
                & (line_coord[:, 2] <= z_hi)
            )
            comp["through_line_points"] = int(np.count_nonzero(mask))
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
            args.require_line_through_tower
            and comp["through_line_points"] < args.min_line_points_through_tower
        ):
            comp["keep"] = False
            comp["remove_reason"] = "too_few_through_line_points"
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
    # 主成分分析特征向量的正负号不固定。这里优先选择朝北等价方向，
    # 避免相近杆塔检测结果的四元数出现 180 度翻转。
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


def count_line_points_through_tower_box(line_coord, tower_box, args):
    empty = {"total": 0, "before": 0, "after": 0}
    if line_coord.shape[0] == 0:
        return empty

    center = tower_box["center"]
    half_size = tower_box["half_size"]
    corners = oriented_box_corners(center, tower_box["axes"], half_size)
    line_axis = main_axis_xy(line_coord[:, :2])
    side_axis = np.array([-line_axis[1], line_axis[0]], dtype=np.float64)

    rel_line_xy = line_coord[:, :2] - center[None, :2]
    line_along = rel_line_xy @ line_axis
    line_side = rel_line_xy @ side_axis
    rel_corner_xy = corners[:, :2] - center[None, :2]
    along_extent = float(np.max(np.abs(rel_corner_xy @ line_axis)))
    side_extent = float(np.max(np.abs(rel_corner_xy @ side_axis)))

    xy_margin = float(args.tower_line_through_xy_margin)
    z_margin = float(args.tower_line_through_z_margin)
    cross_margin = float(args.tower_line_through_cross_margin)
    height_ratio = min(max(float(args.tower_line_through_min_height_ratio), 0.0), 1.0)
    z_min = center[2] - half_size[2] + (2.0 * half_size[2]) * height_ratio - z_margin
    z_max = center[2] + half_size[2] + z_margin
    upper_corridor = (
        (np.abs(line_side) <= side_extent + xy_margin)
        & (line_coord[:, 2] >= z_min)
        & (line_coord[:, 2] <= z_max)
    )
    before = upper_corridor & (line_along <= -(along_extent + cross_margin))
    after = upper_corridor & (line_along >= along_extent + cross_margin)
    before_count = int(np.count_nonzero(before))
    after_count = int(np.count_nonzero(after))
    return {
        "total": before_count + after_count,
        "before": before_count,
        "after": after_count,
    }


def filter_towers_by_line_through_box(
    components, towers, tower_box_list, coord, cls, args
):
    if not args.require_line_through_tower:
        return towers, tower_box_list, 0
    line_coord = coord[cls == args.line_class]
    box_by_id = {int(box["tower_id"]): box for box in tower_box_list}
    kept_towers = []
    removed_component_ids = set()
    removed_count = 0
    for tower in towers:
        box = box_by_id.get(int(tower["id"]))
        through_counts = (
            count_line_points_through_tower_box(line_coord, box, args)
            if box is not None
            else {"total": 0, "before": 0, "after": 0}
        )
        tower["through_line_points"] = int(through_counts["total"])
        tower["through_line_points_before"] = int(through_counts["before"])
        tower["through_line_points_after"] = int(through_counts["after"])
        if (
            through_counts["total"] >= args.min_line_points_through_tower
            and through_counts["before"] >= args.min_line_points_through_tower_side
            and through_counts["after"] >= args.min_line_points_through_tower_side
        ):
            kept_towers.append(tower)
            continue
        removed_count += 1
        removed_component_ids.update(int(item) for item in tower["component_ids"])

    if removed_component_ids:
        for comp in components:
            if int(comp["id"]) in removed_component_ids:
                comp["keep"] = False
                comp["remove_reason"] = "physical_tower_no_line_through"

    kept_ids = {int(tower["id"]) for tower in kept_towers}
    kept_boxes = [box for box in tower_box_list if int(box["tower_id"]) in kept_ids]
    return kept_towers, kept_boxes, removed_count


def tower_line_contact_radius(args):
    radius = args.tower_line_contact_radius
    if radius is None:
        radius = args.tower_line_touch_xy_margin
    return max(float(radius), 0.0)


def count_line_points_touch_tower(line_tree, tower_points, tower_box, args):
    empty = {"count": 0, "min_distance": None, "radius": tower_line_contact_radius(args)}
    if line_tree is None or tower_points.shape[0] == 0 or tower_box is None:
        return empty

    center = tower_box["center"]
    axes = tower_box["axes"]
    half_size = tower_box["half_size"]
    height_ratio = min(max(float(args.tower_line_touch_min_height_ratio), 0.0), 1.0)
    local = (tower_points - center[None, :]) @ axes.T
    z_min = -half_size[2] + (2.0 * half_size[2]) * height_ratio
    upper_points = tower_points[local[:, 2] >= z_min]
    if upper_points.shape[0] == 0:
        return empty

    distance, _ = line_tree.query(upper_points, k=1)
    radius = empty["radius"]
    contact_count = int(np.count_nonzero(distance <= radius))
    return {
        "count": contact_count,
        "min_distance": float(np.min(distance)) if distance.size else None,
        "radius": radius,
    }


def filter_towers_by_line_touch_box(
    components, towers, tower_box_list, coord, cls, tower_indices, point_comp, args
):
    if not args.require_line_touch_tower:
        return towers, tower_box_list, 0

    line_coord = coord[cls == args.line_class]
    line_tree = cKDTree(line_coord) if line_coord.shape[0] else None
    box_by_id = {int(box["tower_id"]): box for box in tower_box_list}
    kept_towers = []
    removed_component_ids = set()
    removed_count = 0
    for tower in towers:
        box = box_by_id.get(int(tower["id"]))
        tower_points = points_for_tower(
            coord, tower_indices, point_comp, tower["component_ids"]
        )
        contact = count_line_points_touch_tower(
            line_tree, tower_points, box, args
        )
        tower["touch_line_points"] = int(contact["count"])
        tower["line_touch_min_distance"] = contact["min_distance"]
        tower["line_touch_radius"] = float(contact["radius"])
        if contact["count"] >= int(args.min_line_points_touch_tower):
            kept_towers.append(tower)
            continue

        removed_count += 1
        removed_component_ids.update(int(item) for item in tower["component_ids"])

    if removed_component_ids:
        for comp in components:
            if int(comp["id"]) in removed_component_ids:
                comp["keep"] = False
                comp["remove_reason"] = "physical_tower_no_line_touch"

    kept_ids = {int(tower["id"]) for tower in kept_towers}
    kept_boxes = [box for box in tower_box_list if int(box["tower_id"]) in kept_ids]
    return kept_towers, kept_boxes, removed_count


def count_line_points_inside_tower_core(line_coord, tower_box, args):
    if line_coord.shape[0] == 0 or tower_box is None:
        return 0

    center = tower_box["center"]
    axes = tower_box["axes"]
    half_size = tower_box["half_size"]
    xy_scale = min(max(float(args.tower_line_inside_xy_scale), 0.0), 1.0)
    height_ratio = min(max(float(args.tower_line_inside_min_height_ratio), 0.0), 1.0)

    expanded_half = half_size.copy()
    corners = oriented_box_corners(center, axes, expanded_half)
    world_min = corners.min(axis=0)
    world_max = corners.max(axis=0)
    coarse = np.all(
        (line_coord >= world_min[None, :]) & (line_coord <= world_max[None, :]),
        axis=1,
    )
    if not np.any(coarse):
        return 0

    local = (line_coord[coarse] - center[None, :]) @ axes.T
    z_min = -half_size[2] + (2.0 * half_size[2]) * height_ratio
    inside = (
        (np.abs(local[:, 0]) <= half_size[0] * xy_scale)
        & (np.abs(local[:, 1]) <= half_size[1] * xy_scale)
        & (local[:, 2] >= z_min)
        & (local[:, 2] <= half_size[2])
    )
    return int(np.count_nonzero(inside))


def filter_towers_by_line_inside_box(components, towers, tower_box_list, coord, cls, args):
    if not args.require_line_inside_tower:
        return towers, tower_box_list, 0

    line_coord = coord[cls == args.line_class]
    box_by_id = {int(box["tower_id"]): box for box in tower_box_list}
    kept_towers = []
    removed_component_ids = set()
    removed_count = 0
    for tower in towers:
        box = box_by_id.get(int(tower["id"]))
        inside_count = count_line_points_inside_tower_core(line_coord, box, args)
        tower["inside_tower_line_points"] = int(inside_count)
        if inside_count >= int(args.min_line_points_inside_tower):
            kept_towers.append(tower)
            continue

        removed_count += 1
        removed_component_ids.update(int(item) for item in tower["component_ids"])

    if removed_component_ids:
        for comp in components:
            if int(comp["id"]) in removed_component_ids:
                comp["keep"] = False
                comp["remove_reason"] = "physical_tower_too_few_inside_line_points"

    kept_ids = {int(tower["id"]) for tower in kept_towers}
    kept_boxes = [box for box in tower_box_list if int(box["tower_id"]) in kept_ids]
    return kept_towers, kept_boxes, removed_count


def max_consecutive_run(values):
    if values.size == 0:
        return 0
    values = np.unique(values)
    best = 1
    current = 1
    for index in range(1, values.size):
        if values[index] == values[index - 1] + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return int(best)


def continuous_line_inside_tower_stats(line_coord, tower_box, line_axis, args):
    empty = {
        "point_count": 0,
        "occupied_bins": 0,
        "max_run_bins": 0,
        "max_run_length": 0.0,
    }
    if line_coord.shape[0] == 0 or tower_box is None:
        return empty

    center = tower_box["center"]
    axes = tower_box["axes"]
    half_size = tower_box["half_size"]
    xy_scale = min(max(float(args.continuous_line_inside_xy_scale), 0.0), 1.0)
    height_ratio = min(max(float(args.continuous_line_inside_min_height_ratio), 0.0), 1.0)
    bin_size = max(float(args.continuous_line_bin_size), 1e-3)

    corners = oriented_box_corners(center, axes, half_size)
    world_min = corners.min(axis=0)
    world_max = corners.max(axis=0)
    coarse = np.all(
        (line_coord >= world_min[None, :]) & (line_coord <= world_max[None, :]),
        axis=1,
    )
    if not np.any(coarse):
        return empty

    candidates = line_coord[coarse]
    local = (candidates - center[None, :]) @ axes.T
    z_min = -half_size[2] + (2.0 * half_size[2]) * height_ratio
    inside = (
        (np.abs(local[:, 0]) <= half_size[0] * xy_scale)
        & (np.abs(local[:, 1]) <= half_size[1] * xy_scale)
        & (local[:, 2] >= z_min)
        & (local[:, 2] <= half_size[2])
    )
    inside_points = candidates[inside]
    if inside_points.shape[0] == 0:
        return empty

    along = (inside_points[:, :2] - center[None, :2]) @ line_axis
    bins = np.floor((along - along.min()) / bin_size).astype(np.int64)
    occupied_bins = np.unique(bins)
    max_run_bins = max_consecutive_run(occupied_bins)
    return {
        "point_count": int(inside_points.shape[0]),
        "occupied_bins": int(occupied_bins.size),
        "max_run_bins": int(max_run_bins),
        "max_run_length": float(max_run_bins * bin_size),
    }


def filter_towers_by_continuous_line_inside_box(
    components, towers, tower_box_list, coord, cls, args
):
    if not args.require_continuous_line_inside_tower:
        return towers, tower_box_list, 0

    line_coord = coord[cls == args.line_class]
    line_axis = main_axis_xy(line_coord[:, :2]) if line_coord.shape[0] else np.array([1.0, 0.0])
    box_by_id = {int(box["tower_id"]): box for box in tower_box_list}
    kept_towers = []
    removed_component_ids = set()
    removed_count = 0
    for tower in towers:
        box = box_by_id.get(int(tower["id"]))
        stats = continuous_line_inside_tower_stats(line_coord, box, line_axis, args)
        tower["continuous_inside_line_points"] = int(stats["point_count"])
        tower["continuous_inside_line_bins"] = int(stats["occupied_bins"])
        tower["continuous_inside_line_max_run_bins"] = int(stats["max_run_bins"])
        tower["continuous_inside_line_max_run_length"] = float(stats["max_run_length"])
        if (
            stats["point_count"] >= int(args.min_continuous_line_points)
            and stats["max_run_bins"] >= int(args.min_continuous_line_bins)
            and stats["max_run_length"] >= float(args.min_continuous_line_length)
        ):
            kept_towers.append(tower)
            continue

        removed_count += 1
        removed_component_ids.update(int(item) for item in tower["component_ids"])

    if removed_component_ids:
        for comp in components:
            if int(comp["id"]) in removed_component_ids:
                comp["keep"] = False
                comp["remove_reason"] = "physical_tower_no_continuous_inside_line"

    kept_ids = {int(tower["id"]) for tower in kept_towers}
    kept_boxes = [box for box in tower_box_list if int(box["tower_id"]) in kept_ids]
    return kept_towers, kept_boxes, removed_count


def side_line_near_tower_stats(line_coord, tower_box, line_axis, args):
    empty = {"left": 0, "right": 0, "total": 0}
    if line_coord.shape[0] == 0 or tower_box is None:
        return empty

    center = tower_box["center"]
    half_size = tower_box["half_size"]
    corners = oriented_box_corners(center, tower_box["axes"], half_size)
    rel_corner_xy = corners[:, :2] - center[None, :2]
    side_axis = np.array([-line_axis[1], line_axis[0]], dtype=np.float64)
    along_extent = float(np.max(np.abs(rel_corner_xy @ line_axis)))
    side_extent = float(np.max(np.abs(rel_corner_xy @ side_axis)))

    rel_line_xy = line_coord[:, :2] - center[None, :2]
    along = rel_line_xy @ line_axis
    side = rel_line_xy @ side_axis

    window = max(float(args.tower_side_line_window), 0.0)
    xy_margin = max(float(args.tower_side_line_xy_margin), 0.0)
    z_margin = float(args.tower_side_line_z_margin)
    height_ratio = min(max(float(args.tower_side_line_min_height_ratio), 0.0), 1.0)
    z_min = center[2] - half_size[2] + (2.0 * half_size[2]) * height_ratio
    z_max = center[2] + half_size[2] + z_margin
    height_mask = (line_coord[:, 2] >= z_min) & (line_coord[:, 2] <= z_max)
    side_mask = np.abs(side) <= side_extent + xy_margin
    left_mask = (
        height_mask
        & side_mask
        & (along >= -along_extent - window)
        & (along <= -along_extent + window)
    )
    right_mask = (
        height_mask
        & side_mask
        & (along >= along_extent - window)
        & (along <= along_extent + window)
    )
    left_count = int(np.count_nonzero(left_mask))
    right_count = int(np.count_nonzero(right_mask))
    return {"left": left_count, "right": right_count, "total": left_count + right_count}


def filter_towers_by_side_line_box(components, towers, tower_box_list, coord, cls, args):
    if not args.require_side_line_near_tower:
        return towers, tower_box_list, 0

    line_coord = coord[cls == args.line_class]
    line_axis = main_axis_xy(line_coord[:, :2]) if line_coord.shape[0] else np.array([1.0, 0.0])
    box_by_id = {int(box["tower_id"]): box for box in tower_box_list}
    kept_towers = []
    removed_component_ids = set()
    removed_count = 0
    for tower in towers:
        box = box_by_id.get(int(tower["id"]))
        stats = side_line_near_tower_stats(line_coord, box, line_axis, args)
        tower["side_line_left_points"] = int(stats["left"])
        tower["side_line_right_points"] = int(stats["right"])
        tower["side_line_points"] = int(stats["total"])
        if max(stats["left"], stats["right"]) >= int(args.min_side_line_points):
            kept_towers.append(tower)
            continue

        removed_count += 1
        removed_component_ids.update(int(item) for item in tower["component_ids"])

    if removed_component_ids:
        for comp in components:
            if int(comp["id"]) in removed_component_ids:
                comp["keep"] = False
                comp["remove_reason"] = "physical_tower_no_side_line"

    kept_ids = {int(tower["id"]) for tower in kept_towers}
    kept_boxes = [box for box in tower_box_list if int(box["tower_id"]) in kept_ids]
    return kept_towers, kept_boxes, removed_count


def count_insulator_points_near_tower_box(insulator_coord, tower_box, args):
    if insulator_coord.shape[0] == 0 or tower_box is None:
        return 0
    center = tower_box["center"]
    axes = tower_box["axes"]
    half_size = tower_box["half_size"]
    local = (insulator_coord - center[None, :]) @ axes.T

    xy_margin = float(args.tower_insulator_xy_margin)
    z_margin = float(args.tower_insulator_z_margin)
    height_ratio = min(max(float(args.tower_insulator_min_height_ratio), 0.0), 1.0)
    z_min = -half_size[2] + (2.0 * half_size[2]) * height_ratio - z_margin
    z_max = half_size[2] + z_margin
    mask = (
        (np.abs(local[:, 0]) <= half_size[0] + xy_margin)
        & (np.abs(local[:, 1]) <= half_size[1] + xy_margin)
        & (local[:, 2] >= z_min)
        & (local[:, 2] <= z_max)
    )
    return int(np.count_nonzero(mask))


def bridged_insulator_points_near_tower(insulator_coord, line_tree, tower_box, args):
    if insulator_coord.shape[0] == 0 or line_tree is None or tower_box is None:
        return {"nearby": 0, "bridged": 0}

    center = tower_box["center"]
    axes = tower_box["axes"]
    half_size = tower_box["half_size"]
    local = (insulator_coord - center[None, :]) @ axes.T

    xy_margin = float(args.tower_insulator_xy_margin)
    z_margin = float(args.tower_insulator_z_margin)
    height_ratio = min(max(float(args.tower_insulator_min_height_ratio), 0.0), 1.0)
    z_min = -half_size[2] + (2.0 * half_size[2]) * height_ratio - z_margin
    z_max = half_size[2] + z_margin
    nearby_mask = (
        (np.abs(local[:, 0]) <= half_size[0] + xy_margin)
        & (np.abs(local[:, 1]) <= half_size[1] + xy_margin)
        & (local[:, 2] >= z_min)
        & (local[:, 2] <= z_max)
    )
    nearby = insulator_coord[nearby_mask]
    if nearby.shape[0] == 0:
        return {"nearby": 0, "bridged": 0}

    distance, _ = line_tree.query(nearby, k=1)
    bridged = int(np.count_nonzero(distance <= float(args.insulator_line_radius)))
    return {"nearby": int(nearby.shape[0]), "bridged": bridged}


def filter_towers_by_insulator_line_bridge(
    components, towers, tower_box_list, coord, cls, args
):
    if not args.require_insulator_line_bridge:
        return towers, tower_box_list, 0

    insulator_coord = coord[cls == args.insulator_class]
    line_coord = coord[cls == args.line_class]
    line_tree = cKDTree(line_coord) if line_coord.shape[0] else None
    box_by_id = {int(box["tower_id"]): box for box in tower_box_list}
    kept_towers = []
    removed_component_ids = set()
    removed_count = 0
    for tower in towers:
        box = box_by_id.get(int(tower["id"]))
        stats = bridged_insulator_points_near_tower(insulator_coord, line_tree, box, args)
        tower["nearby_insulator_points"] = int(stats["nearby"])
        tower["bridged_insulator_points"] = int(stats["bridged"])
        if stats["bridged"] >= int(args.min_bridged_insulator_points):
            kept_towers.append(tower)
            continue

        removed_count += 1
        removed_component_ids.update(int(item) for item in tower["component_ids"])

    if removed_component_ids:
        for comp in components:
            if int(comp["id"]) in removed_component_ids:
                comp["keep"] = False
                comp["remove_reason"] = "physical_tower_no_insulator_line_bridge"

    kept_ids = {int(tower["id"]) for tower in kept_towers}
    kept_boxes = [box for box in tower_box_list if int(box["tower_id"]) in kept_ids]
    return kept_towers, kept_boxes, removed_count


def filter_towers_by_insulator(components, towers, tower_box_list, coord, cls, args):
    if not args.require_insulator_near_tower:
        return towers, tower_box_list, 0

    insulator_coord = coord[cls == args.insulator_class]
    box_by_id = {int(box["tower_id"]): box for box in tower_box_list}
    kept_towers = []
    removed_component_ids = set()
    removed_count = 0
    for tower in towers:
        box = box_by_id.get(int(tower["id"]))
        insulator_count = count_insulator_points_near_tower_box(
            insulator_coord, box, args
        )
        tower["nearby_insulator_points"] = int(insulator_count)
        if insulator_count >= int(args.min_insulator_points_near_tower):
            kept_towers.append(tower)
            continue
        removed_count += 1
        removed_component_ids.update(int(item) for item in tower["component_ids"])

    if removed_component_ids:
        for comp in components:
            if int(comp["id"]) in removed_component_ids:
                comp["keep"] = False
                comp["remove_reason"] = "physical_tower_no_nearby_insulator"

    kept_ids = {int(tower["id"]) for tower in kept_towers}
    kept_boxes = [box for box in tower_box_list if int(box["tower_id"]) in kept_ids]
    return kept_towers, kept_boxes, removed_count


def count_line_points_near_tower_top(line_coord, tower, args):
    if line_coord.shape[0] == 0:
        return {
            "count": 0,
            "top_z": float(tower["bbox_max"][2]),
            "z_min": float(tower["bbox_max"][2]),
            "z_max": float(tower["bbox_max"][2]),
        }

    xy_margin = float(args.tower_top_line_xy_margin)
    top_z = float(tower["bbox_max"][2])
    z_min = top_z - float(args.tower_top_line_z_below)
    z_max = top_z + float(args.tower_top_line_z_above)
    xy_min = tower["bbox_min"][:2] - xy_margin
    xy_max = tower["bbox_max"][:2] + xy_margin
    mask = (
        (line_coord[:, 0] >= xy_min[0])
        & (line_coord[:, 0] <= xy_max[0])
        & (line_coord[:, 1] >= xy_min[1])
        & (line_coord[:, 1] <= xy_max[1])
        & (line_coord[:, 2] >= z_min)
        & (line_coord[:, 2] <= z_max)
    )
    return {
        "count": int(np.count_nonzero(mask)),
        "top_z": top_z,
        "z_min": float(z_min),
        "z_max": float(z_max),
    }


def filter_towers_by_top_line(components, towers, tower_box_list, coord, cls, args):
    if not args.require_tower_top_line:
        return towers, tower_box_list, 0

    line_coord = coord[cls == args.line_class]
    kept_towers = []
    removed_component_ids = set()
    removed_count = 0
    for tower in towers:
        stats = count_line_points_near_tower_top(line_coord, tower, args)
        tower["top_line_points"] = int(stats["count"])
        tower["tower_top_z"] = float(stats["top_z"])
        tower["tower_top_line_z_min"] = float(stats["z_min"])
        tower["tower_top_line_z_max"] = float(stats["z_max"])
        if stats["count"] >= int(args.min_tower_top_line_points):
            kept_towers.append(tower)
            continue

        removed_count += 1
        removed_component_ids.update(int(item) for item in tower["component_ids"])

    if removed_component_ids:
        for comp in components:
            if int(comp["id"]) in removed_component_ids:
                comp["keep"] = False
                comp["remove_reason"] = "physical_tower_no_top_line"

    kept_ids = {int(tower["id"]) for tower in kept_towers}
    kept_boxes = [box for box in tower_box_list if int(box["tower_id"]) in kept_ids]
    return kept_towers, kept_boxes, removed_count


def upper_tower_spread(points, tower_box, args):
    empty = {
        "upper_point_count": 0,
        "upper_width": 0.0,
        "upper_length": 0.0,
        "upper_area": 0.0,
    }
    if points.shape[0] == 0 or tower_box is None:
        return empty

    center = tower_box["center"]
    axes = tower_box["axes"]
    half_size = tower_box["half_size"]
    local = (points - center[None, :]) @ axes.T
    ratio = min(max(float(args.bare_pole_upper_height_ratio), 0.0), 1.0)
    z_min = -half_size[2] + (2.0 * half_size[2]) * ratio
    upper = local[local[:, 2] >= z_min]
    if upper.shape[0] == 0:
        return empty

    spread_0 = float(np.ptp(upper[:, 0]))
    spread_1 = float(np.ptp(upper[:, 1]))
    return {
        "upper_point_count": int(upper.shape[0]),
        "upper_width": float(max(spread_0, spread_1)),
        "upper_length": float(min(spread_0, spread_1)),
        "upper_area": float(spread_0 * spread_1),
    }


def filter_bare_pole_towers(
    components, towers, tower_box_list, coord, tower_indices, point_comp, args
):
    if not args.remove_bare_pole_towers:
        return towers, tower_box_list, 0

    box_by_id = {int(box["tower_id"]): box for box in tower_box_list}
    kept_towers = []
    removed_component_ids = set()
    removed_count = 0
    for tower in towers:
        box = box_by_id.get(int(tower["id"]))
        points = points_for_tower(
            coord, tower_indices, point_comp, tower["component_ids"]
        )
        spread = upper_tower_spread(points, box, args)
        tower.update(spread)

        is_bare_pole = (
            spread["upper_point_count"] < int(args.min_upper_tower_points)
            or spread["upper_width"] < float(args.min_upper_tower_width)
            or spread["upper_area"] < float(args.min_upper_tower_area)
        )
        if not is_bare_pole:
            kept_towers.append(tower)
            continue

        removed_count += 1
        removed_component_ids.update(int(item) for item in tower["component_ids"])

    if removed_component_ids:
        for comp in components:
            if int(comp["id"]) in removed_component_ids:
                comp["keep"] = False
                comp["remove_reason"] = "physical_tower_bare_pole"

    kept_ids = {int(tower["id"]) for tower in kept_towers}
    kept_boxes = [box for box in tower_box_list if int(box["tower_id"]) in kept_ids]
    return kept_towers, kept_boxes, removed_count


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
        if args.require_connected_line_span:
            selected_proj = proj[mask]
            end_window = max(float(args.span_end_contact_window), 0.0)
            min_end_points = int(args.min_span_end_contact_points)
            left_end_count = int(
                np.count_nonzero(selected_proj <= span_start + end_window)
            )
            right_end_count = int(
                np.count_nonzero(selected_proj >= span_end - end_window)
            )
            if left_end_count < min_end_points or right_end_count < min_end_points:
                continue
        else:
            left_end_count = 0
            right_end_count = 0
        box = {
            "kind": "line_span",
            "span_id": int(span_id),
            "left_tower_id": int(left["id"]),
            "right_tower_id": int(right["id"]),
            "left_tower_name": left.get("display_name", left.get("name", "")),
            "right_tower_name": right.get("display_name", right.get("name", "")),
            "line_point_count": int(selected.size),
            "left_end_contact_points": int(left_end_count),
            "right_end_contact_points": int(right_end_count),
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


def filter_towers_by_connected_line_span(
    components, towers, tower_box_list, coord, cls, args
):
    if not args.require_connected_line_span:
        return towers, tower_box_list, 0, []

    tower_box_by_id = {int(box["tower_id"]): box for box in tower_box_list}
    span_boxes = line_span_boxes(coord, cls, towers, tower_box_by_id, args)
    connected_ids = set()
    for span in span_boxes:
        left = next(
            (tower for tower in towers if int(tower["id"]) == int(span["left_tower_id"])),
            None,
        )
        right = next(
            (tower for tower in towers if int(tower["id"]) == int(span["right_tower_id"])),
            None,
        )
        min_touch = int(args.min_line_points_touch_tower)
        left_touch_ok = (
            not args.require_line_touch_tower
            or int(left.get("touch_line_points", 0)) >= min_touch
        ) if left is not None else False
        right_touch_ok = (
            not args.require_line_touch_tower
            or int(right.get("touch_line_points", 0)) >= min_touch
        ) if right is not None else False
        if left is not None and left_touch_ok:
            connected_ids.add(int(span["left_tower_id"]))
        if right is not None and right_touch_ok:
            connected_ids.add(int(span["right_tower_id"]))

    for tower in towers:
        tower["connected_span_count"] = 0
    tower_by_id = {int(tower["id"]): tower for tower in towers}
    for span in span_boxes:
        left = tower_by_id.get(int(span["left_tower_id"]))
        right = tower_by_id.get(int(span["right_tower_id"]))
        if left is not None and int(left["id"]) in connected_ids:
            left["connected_span_count"] = int(left.get("connected_span_count", 0)) + 1
        if right is not None and int(right["id"]) in connected_ids:
            right["connected_span_count"] = int(right.get("connected_span_count", 0)) + 1

    kept_towers = []
    removed_component_ids = set()
    removed_count = 0
    for tower in towers:
        if int(tower["id"]) in connected_ids:
            kept_towers.append(tower)
            continue
        removed_count += 1
        removed_component_ids.update(int(item) for item in tower["component_ids"])

    if removed_component_ids:
        for comp in components:
            if int(comp["id"]) in removed_component_ids:
                comp["keep"] = False
                comp["remove_reason"] = "physical_tower_no_connected_line_span"

    kept_ids = {int(tower["id"]) for tower in kept_towers}
    kept_boxes = [box for box in tower_box_list if int(box["tower_id"]) in kept_ids]
    kept_spans = [
        span
        for span in span_boxes
        if int(span["left_tower_id"]) in kept_ids
        and int(span["right_tower_id"]) in kept_ids
    ]
    return kept_towers, kept_boxes, removed_count, kept_spans


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


def sort_conductor_fits_by_height_layer(fits, args):
    if not fits:
        return []
    layer_gap = max(float(args.conductor_layer_z_gap), 0.0)
    high_to_low = sorted(fits, key=lambda fit: fit["sort_z"], reverse=True)
    layers = []
    current = []
    previous_z = None
    for fit in high_to_low:
        if previous_z is not None and previous_z - fit["sort_z"] > layer_gap:
            layers.append(current)
            current = []
        current.append(fit)
        previous_z = fit["sort_z"]
    if current:
        layers.append(current)

    ordered = []
    for layer_no, layer in enumerate(layers, start=1):
        for fit in sorted(layer, key=lambda item: item["sort_side"]):
            fit["sort_layer"] = int(layer_no)
            ordered.append(fit)
    return ordered


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


SEVEN_SEGMENT_DIGITS = {
    "0": ("a", "b", "c", "d", "e", "f"),
    "1": ("b", "c"),
    "2": ("a", "b", "g", "e", "d"),
    "3": ("a", "b", "g", "c", "d"),
    "4": ("f", "g", "b", "c"),
    "5": ("a", "f", "g", "c", "d"),
    "6": ("a", "f", "g", "e", "c", "d"),
    "7": ("a", "b", "c"),
    "8": ("a", "b", "c", "d", "e", "f", "g"),
    "9": ("a", "b", "c", "d", "f", "g"),
}

SEVEN_SEGMENT_ENDPOINTS = {
    "a": ((0.0, 1.0), (1.0, 1.0)),
    "b": ((1.0, 1.0), (1.0, 0.5)),
    "c": ((1.0, 0.5), (1.0, 0.0)),
    "d": ((0.0, 0.0), (1.0, 0.0)),
    "e": ((0.0, 0.5), (0.0, 0.0)),
    "f": ((0.0, 1.0), (0.0, 0.5)),
    "g": ((0.0, 0.5), (1.0, 0.5)),
}


def sample_segment_2d(start, end, step):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    length = float(np.linalg.norm(end - start))
    count = max(int(np.ceil(length / max(step, 1e-3))) + 1, 2)
    alpha = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return start[None, :] * (1.0 - alpha[:, None]) + end[None, :] * alpha[:, None]


def conductor_number_label_world(conductor_no, fit, axes, origin, args):
    text = str(int(conductor_no))
    size = max(float(args.conductor_label_size), 0.1)
    width = size * 0.6
    gap = width * 0.35
    step = max(float(args.conductor_label_step), 0.02)

    along_mid = (float(fit["along_min"]) + float(fit["along_max"])) / 2.0
    side_mid, z_mid = evaluate_fit(fit, np.asarray([along_mid], dtype=np.float64))
    center_side = float(side_mid[0]) + float(args.conductor_label_side_offset)
    center_z = float(z_mid[0]) + float(args.conductor_label_z_offset)

    digit_points = []
    total_width = len(text) * width + max(len(text) - 1, 0) * gap
    side_start = center_side - total_width / 2.0
    for digit_index, char in enumerate(text):
        segments = SEVEN_SEGMENT_DIGITS.get(char)
        if not segments:
            continue
        x0 = side_start + digit_index * (width + gap)
        for segment in segments:
            start, end = SEVEN_SEGMENT_ENDPOINTS[segment]
            start = (
                x0 + start[0] * width,
                center_z + (start[1] - 0.5) * size,
            )
            end = (
                x0 + end[0] * width,
                center_z + (end[1] - 0.5) * size,
            )
            digit_points.append(sample_segment_2d(start, end, step))

    if not digit_points:
        return np.empty((0, 3), dtype=np.float64)
    side_z = np.vstack(digit_points)
    local = np.column_stack(
        (
            np.full(side_z.shape[0], along_mid, dtype=np.float64),
            side_z[:, 0],
            side_z[:, 1],
        )
    )
    return line_local_to_world(local, axes, origin)


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
        return (
            [],
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint16),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint16),
            0,
        )

    conductor_reports = []
    fitted_points = []
    fitted_colors = []
    label_points = []
    label_colors = []
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
            sort_local = np.asarray(
                [[mid, fit["sort_side"], fit["sort_z"]]], dtype=np.float64
            )
            sort_world = line_local_to_world(sort_local, axes, origin)[0]
            fit["sort_x"] = float(sort_world[0])
            fit["sort_y"] = float(sort_world[1])
        fits = sort_conductor_fits_by_height_layer(fits, args)

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
            layer_no = int(fit.get("sort_layer", conductor_no))
            color = CONDUCTOR_COLORS_16[(layer_no - 1) % len(CONDUCTOR_COLORS_16)]
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
            if not args.no_append_conductor_labels:
                label_world = conductor_number_label_world(conductor_no, fit, axes, origin, args)
                if label_world.size:
                    label_points.append(label_world)
                    label_colors.append(
                        np.tile(color[None, :], (label_world.shape[0], 1))
                    )

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
                    "color_by": "sort_layer",
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
                    "sort_x": float(fit["sort_x"]),
                    "sort_y": float(fit["sort_y"]),
                    "sort_layer": int(fit.get("sort_layer", 0)),
                    "sort_side": float(fit["sort_side"]),
                    "sort_z": float(fit["sort_z"]),
                    "side_poly_coef": fit["side_coef"].tolist(),
                    "z_poly_coef": fit["z_coef"].tolist(),
                    "polyline_xyz": json_sample_world.tolist(),
                }
            )
            global_fit_no += 1

    conductor_points = (
        np.vstack(fitted_points)
        if fitted_points
        else np.empty((0, 3), dtype=np.float64)
    )
    conductor_colors = (
        np.vstack(fitted_colors).astype(np.uint16, copy=False)
        if fitted_colors
        else np.empty((0, 3), dtype=np.uint16)
    )
    number_label_points = (
        np.vstack(label_points)
        if label_points
        else np.empty((0, 3), dtype=np.float64)
    )
    number_label_colors = (
        np.vstack(label_colors).astype(np.uint16, copy=False)
        if label_colors
        else np.empty((0, 3), dtype=np.uint16)
    )
    return (
        conductor_reports,
        conductor_points,
        conductor_colors,
        number_label_points,
        number_label_colors,
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
        "through_line_points": int(comp.get("through_line_points", 0)),
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
        "through_line_points": int(tower.get("through_line_points", 0)),
        "through_line_points_before": int(tower.get("through_line_points_before", 0)),
        "through_line_points_after": int(tower.get("through_line_points_after", 0)),
        "touch_line_points": int(tower.get("touch_line_points", 0)),
        "inside_tower_line_points": int(tower.get("inside_tower_line_points", 0)),
        "continuous_inside_line_points": int(tower.get("continuous_inside_line_points", 0)),
        "continuous_inside_line_bins": int(tower.get("continuous_inside_line_bins", 0)),
        "continuous_inside_line_max_run_bins": int(
            tower.get("continuous_inside_line_max_run_bins", 0)
        ),
        "continuous_inside_line_max_run_length": float(
            tower.get("continuous_inside_line_max_run_length", 0.0)
        ),
        "side_line_left_points": int(tower.get("side_line_left_points", 0)),
        "side_line_right_points": int(tower.get("side_line_right_points", 0)),
        "side_line_points": int(tower.get("side_line_points", 0)),
        "connected_span_count": int(tower.get("connected_span_count", 0)),
        "line_touch_min_distance": (
            None
            if tower.get("line_touch_min_distance") is None
            else float(tower["line_touch_min_distance"])
        ),
        "line_touch_radius": float(tower.get("line_touch_radius", 0.0)),
        "top_line_points": int(tower.get("top_line_points", 0)),
        "tower_top_z": float(tower.get("tower_top_z", tower["bbox_max"][2])),
        "tower_top_line_z_min": float(
            tower.get("tower_top_line_z_min", tower["bbox_max"][2])
        ),
        "tower_top_line_z_max": float(
            tower.get("tower_top_line_z_max", tower["bbox_max"][2])
        ),
        "nearby_insulator_points": int(tower.get("nearby_insulator_points", 0)),
        "bridged_insulator_points": int(tower.get("bridged_insulator_points", 0)),
        "upper_point_count": int(tower.get("upper_point_count", 0)),
        "upper_width": float(tower.get("upper_width", 0.0)),
        "upper_length": float(tower.get("upper_length", 0.0)),
        "upper_area": float(tower.get("upper_area", 0.0)),
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


def make_standard_tower_boxes(tower_box_list, origin, orientation):
    return [
        box_to_obb_record(
            box,
            class_name="tower",
            instance_name=f"杆塔{int(box.get('tower_no', 0))}",
            origin=origin,
            orientation=orientation,
        )
        for box in sorted(tower_box_list, key=lambda item: int(item.get("tower_no", 0)))
    ]


def make_standard_line_boxes(span_boxes, origin, orientation):
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
            orientation=orientation,
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


def north_relative_rotation_from_axes(axes):
    """返回从正北朝向 OBB 基到拟合坐标轴的旋转。

    拟合框的 axes 以行向量保存：局部长轴、局部宽轴、局部 Z。
    正北框定义为 length=LAS X+、width=LAS Y+、Z=向上，
    因此已经朝北的框会导出单位旋转。
    """
    fitted_rotation = np.asarray(axes, dtype=np.float64).T
    north_axes = local_axes_from_direction(np.array([1.0, 0.0], dtype=np.float64))
    north_rotation = north_axes.T
    relative_rotation = fitted_rotation @ north_rotation.T
    return rotation_matrix_to_quaternion(relative_rotation)


def box_to_obb_record(box, class_name, instance_name, origin, orientation="fitted"):
    if orientation == "north":
        corners, _ = box_geometry_from_box(box)
        bbox_min = corners.min(axis=0)
        bbox_max = corners.max(axis=0)
        center = (bbox_min + bbox_max) / 2.0
        size_x = float(bbox_max[0] - bbox_min[0])
        size_y = float(bbox_max[1] - bbox_min[1])
        height = float(bbox_max[2] - bbox_min[2])
        extent = np.array([size_x, size_y, height], dtype=np.float64)
        rotation = [0.0, 0.0, 0.0, 1.0]
        extent_order = ["north_x_length", "side_y_width", "height"]
    elif box.get("box_mode") == "oriented":
        center = box["center"].astype(np.float64, copy=False)
        extent = (box["half_size"] * 2.0).astype(np.float64, copy=False)
        axes = box["axes"]
        if box.get("kind") == "line_span":
            # 对档距 JSON，塔到塔方向作为框的 width 轴。
            # 使用 -side、along、up 组成右手局部坐标系。
            axes = np.vstack([-axes[1], axes[0], axes[2]])
            extent = extent[[1, 0, 2]]
            extent_order = [
                "local_x_length_cross_span",
                "local_y_width_between_towers",
                "local_z_height",
            ]
        else:
            extent_order = ["local_x_length", "local_y_width", "local_z_height"]
        rotation = north_relative_rotation_from_axes(axes)
    else:
        center = (box["bbox_min"] + box["bbox_max"]) / 2.0
        extent = box["bbox_max"] - box["bbox_min"]
        rotation = [0.0, 0.0, 0.0, 1.0]
        extent_order = ["x", "y", "z"]
    origin = np.asarray(origin, dtype=np.float64)
    relative_center = center - origin
    relative_center_list = relative_center.tolist()
    center_list = center.tolist()
    origin_list = origin.tolist()
    extent_list = extent.tolist()
    record = {
        "class_name": class_name,
        "instance_name": instance_name,
        "obb": relative_center_list + extent_list + rotation,
        "obb_global": {
            "extent": extent_list,
            "extent_order": extent_order,
            "lat_lng_alt": center_list,
            "las_origin": origin_list,
            "rotation": rotation,
        },
    }
    return record


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
    origin = np.asarray(las.header.offsets, dtype=np.float64)
    header_mins = np.asarray(las.header.mins, dtype=np.float64)
    header_maxs = np.asarray(las.header.maxs, dtype=np.float64)
    if np.all((origin >= header_mins) & (origin <= header_maxs)):
        return origin
    return (header_mins + header_maxs) / 2.0


def build_tower_state(coord, cls, args):
    """构建清理后的杆塔组件和物理杆塔实例。

    这是两个阶段共用的后处理核心：
    1. 清理 class=1 的杆塔误检；
    2. 生成后续画框所需的有效物理杆塔。
    """
    start = time.perf_counter()

    # 将预测为杆塔的点聚类成连通组件。
    components, tower_indices, point_comp = cluster_towers(coord, cls, args)

    # 执行组件级过滤：点数、高度、XY 尺寸、附近导线、局部地面高度等。
    mark_components_to_keep(components, coord, cls, args)

    # 恢复保留杆塔脚印附近的小型低矮组件，避免严格过滤误删杆塔底部。
    seed_towers = merge_physical_towers(components, args)
    recovered_base_components = recover_tower_base_components(
        components, seed_towers, args
    )

    # 将组件合并成物理杆塔实例，并用导线、绝缘子、顶部导线关系进一步验证。
    physical_towers = merge_physical_towers(components, args)
    physical_towers = assign_tower_names(physical_towers, args)
    initial_tower_boxes = tower_boxes(
        physical_towers, coord, tower_indices, point_comp, args.tower_box_margin
    )
    physical_towers, initial_tower_boxes, removed_physical_towers = (
        filter_towers_by_line_through_box(
            components, physical_towers, initial_tower_boxes, coord, cls, args
        )
    )
    physical_towers, initial_tower_boxes, removed_no_touch_towers = (
        filter_towers_by_line_touch_box(
            components,
            physical_towers,
            initial_tower_boxes,
            coord,
            cls,
            tower_indices,
            point_comp,
            args,
        )
    )
    physical_towers, initial_tower_boxes, removed_no_top_line_towers = (
        filter_towers_by_top_line(
            components, physical_towers, initial_tower_boxes, coord, cls, args
        )
    )
    physical_towers, initial_tower_boxes, removed_inside_line_towers = (
        filter_towers_by_line_inside_box(
            components, physical_towers, initial_tower_boxes, coord, cls, args
        )
    )
    physical_towers, initial_tower_boxes, removed_noncontinuous_line_towers = (
        filter_towers_by_continuous_line_inside_box(
            components, physical_towers, initial_tower_boxes, coord, cls, args
        )
    )
    physical_towers, initial_tower_boxes, removed_no_side_line_towers = (
        filter_towers_by_side_line_box(
            components, physical_towers, initial_tower_boxes, coord, cls, args
        )
    )
    physical_towers, initial_tower_boxes, removed_insulatorless_towers = (
        filter_towers_by_insulator(
            components, physical_towers, initial_tower_boxes, coord, cls, args
        )
    )
    physical_towers, initial_tower_boxes, removed_unbridged_insulator_towers = (
        filter_towers_by_insulator_line_bridge(
            components, physical_towers, initial_tower_boxes, coord, cls, args
        )
    )
    physical_towers, initial_tower_boxes, removed_bare_pole_towers = (
        filter_bare_pole_towers(
            components,
            physical_towers,
            initial_tower_boxes,
            coord,
            tower_indices,
            point_comp,
            args,
        )
    )
    physical_towers = assign_tower_names(physical_towers, args)
    physical_towers, initial_tower_boxes, removed_unconnected_span_towers, _ = (
        filter_towers_by_connected_line_span(
            components,
            physical_towers,
            initial_tower_boxes,
            coord,
            cls,
            args,
        )
    )
    physical_towers = assign_tower_names(physical_towers, args)

    return {
        "components": components,
        "tower_indices": tower_indices,
        "point_comp": point_comp,
        "physical_towers": physical_towers,
        "initial_tower_boxes": initial_tower_boxes,
        "recovered_base_components": int(recovered_base_components),
        "removed_physical_towers_without_through_line": int(removed_physical_towers),
        "removed_physical_towers_without_line_touch": int(removed_no_touch_towers),
        "removed_physical_towers_without_top_line": int(removed_no_top_line_towers),
        "removed_physical_towers_with_too_few_inside_line_points": int(
            removed_inside_line_towers
        ),
        "removed_physical_towers_without_continuous_inside_line": int(
            removed_noncontinuous_line_towers
        ),
        "removed_physical_towers_without_side_line": int(removed_no_side_line_towers),
        "removed_physical_towers_without_nearby_insulator": int(
            removed_insulatorless_towers
        ),
        "removed_physical_towers_without_insulator_line_bridge": int(
            removed_unbridged_insulator_towers
        ),
        "removed_bare_pole_physical_towers": int(removed_bare_pole_towers),
        "removed_physical_towers_without_connected_line_span": int(
            removed_unconnected_span_towers
        ),
        "elapsed_sec": float(time.perf_counter() - start),
    }

def default_draw_report_path(output_path):
    return output_path.with_name(output_path.stem + "_draw_report.json")


def default_draw_json_paths(report_path):
    _, default_box_report = default_standard_json_paths(report_path)
    conductor_report = report_path.with_name(
        report_path.stem[: -len("_report")] + "_conductors.json"
        if report_path.stem.endswith("_report")
        else report_path.stem + "_conductors.json"
    )
    return default_box_report, conductor_report


def main():
    args = parse_args()
    global laspy
    import laspy as laspy_module

    laspy = laspy_module
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else default_draw_report_path(output_path)
    default_box_report_path, default_conductor_report_path = default_draw_json_paths(
        report_path
    )
    combined_box_report_path = (
        Path(args.combined_box_report)
        if args.combined_box_report
        else Path(args.line_box_report)
        if args.line_box_report
        else default_box_report_path
    )
    conductor_report_path = (
        Path(args.conductor_report)
        if args.conductor_report
        else default_conductor_report_path
    )
    tower_box_report_path = Path(args.tower_box_report) if args.tower_box_report else None
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists, use --overwrite: {output_path}")

    start = time.perf_counter()
    print(f"Reading cleaned LAS {input_path}", flush=True)
    las = laspy.read(input_path)
    if "classification" not in set(las.point_format.dimension_names):
        raise ValueError("Input LAS has no classification dimension.")

    # 画框阶段只保留当前这次生成的可视化对象。除非用户显式要求保留，
    # 否则先删除输入中已有的 class 30/31 旧绘制点。
    removed_existing_synthetic_points = 0
    if not args.keep_existing_synthetic_points:
        las, removed_existing_synthetic_points = remove_existing_synthetic_points(las)
        if removed_existing_synthetic_points:
            print(
                f"Removed {removed_existing_synthetic_points} existing synthetic points "
                f"(classes {FITTED_CONDUCTOR_CLASS}/{BOX_CLASS})",
                flush=True,
            )

    coord = coords_from_las(las)
    obb_origin = resolve_obb_origin(las, args)
    cls = np.asarray(las.classification, dtype=np.uint8).copy()
    original_counts = {
        str(i): int(v) for i, v in enumerate(np.bincount(cls, minlength=32)) if v
    }
    print(f"Loaded {coord.shape[0]} points; class counts: {original_counts}", flush=True)

    # 绘制导线和框之前，先给原始语义类别重新着色。
    recolor_by_class(las, cls, args)

    # 根据清理后的语义标签重新构建有效物理杆塔。
    # 此处不改写 classification，只决定后续应该画哪些杆塔和档距。
    t0 = time.perf_counter()
    state = build_tower_state(coord, cls, args)
    components = state["components"]
    physical_towers = state["physical_towers"]
    tower_indices = state["tower_indices"]
    point_comp = state["point_comp"]
    kept_components = [comp for comp in components if comp["keep"]]
    print(
        "Tower state: components={} kept={} physical_towers={} in {:.2f}s".format(
            len(components),
            len(kept_components),
            len(physical_towers),
            time.perf_counter() - t0,
        ),
        flush=True,
    )

    # 根据保留下来的物理杆塔生成杆塔框。
    boxes = tower_boxes(
        physical_towers, coord, tower_indices, point_comp, args.tower_box_margin
    )

    # 根据相邻杆塔生成档距框，并只使用塔到塔走廊内的导线点。
    tower_box_by_id = {
        int(box["tower_id"]): box for box in boxes if box["kind"] == "tower"
    }
    span_boxes = line_span_boxes(coord, cls, physical_towers, tower_box_by_id, args)
    boxes.extend(span_boxes)
    print(
        "Boxes: physical_towers={}, line_span={}".format(
            len(physical_towers), len(span_boxes)
        ),
        flush=True,
    )

    # 在每个有效档距内拟合导线。导线按局部高度层和局部 side 位置编号，
    # RGB 颜色按高度层选择，使同层导线颜色一致。
    (
        conductor_reports,
        conductor_points,
        conductor_colors,
        conductor_label_points,
        conductor_label_colors,
        colored_line_points,
    ) = fit_conductors_for_spans(las, coord, span_boxes, args)
    appended_conductor_points = 0
    if conductor_points.size:
        las, appended_conductor_points = append_colored_points(
            las, conductor_points, conductor_colors, FITTED_CONDUCTOR_CLASS
        )
    appended_conductor_label_points = 0
    if conductor_label_points.size:
        las, appended_conductor_label_points = append_colored_points(
            las,
            conductor_label_points,
            conductor_label_colors,
            FITTED_CONDUCTOR_CLASS,
        )
    if conductor_reports:
        print(
            "Conductors: fitted={} colored_original_points={} appended_points={} label_points={}".format(
                len(conductor_reports),
                colored_line_points,
                appended_conductor_points,
                appended_conductor_label_points,
            ),
            flush=True,
        )

    # 追加杆塔框和档距框边线采样点，作为 class 31 可视化点。
    appended_box_points = 0
    if not args.no_append_box_points:
        las, appended_box_points = append_boxes(las, boxes, args.edge_step)
        print(f"Appended {appended_box_points} box-edge points", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(output_path)
    elapsed = time.perf_counter() - start

    debug_conductor_reports = []
    for conductor in conductor_reports:
        item = dict(conductor)
        item.pop("polyline_xyz", None)
        debug_conductor_reports.append(item)
    report = {
        "stage": "draw",
        "input": str(input_path),
        "output": str(output_path),
        "preset": args.preset,
        "original_class_counts": original_counts,
        "tower_components_total": len(components),
        "tower_components_kept": len(kept_components),
        "physical_towers": [json_ready_tower(tower) for tower in physical_towers],
        "physical_tower_count": len(physical_towers),
        "removed_existing_synthetic_points": int(removed_existing_synthetic_points),
        "conductors": debug_conductor_reports,
        "conductor_count": len(conductor_reports),
        "colored_original_line_points": int(colored_line_points),
        "appended_conductor_fit_points": int(appended_conductor_points),
        "appended_conductor_label_points": int(appended_conductor_label_points),
        "boxes": [json_ready_box(box) for box in boxes],
        "appended_box_edge_points": int(appended_box_points),
        "obb_origin_mode": args.obb_origin,
        "obb_origin_xyz": obb_origin.tolist(),
        "json_box_orientation": args.json_box_orientation,
        "elapsed_sec": round(elapsed, 3),
        "parameters": vars(args),
    }
    for key in (
        "recovered_base_components",
        "removed_physical_towers_without_through_line",
        "removed_physical_towers_without_line_touch",
        "removed_physical_towers_without_top_line",
        "removed_physical_towers_with_too_few_inside_line_points",
        "removed_physical_towers_without_continuous_inside_line",
        "removed_physical_towers_without_side_line",
        "removed_physical_towers_without_nearby_insulator",
        "removed_physical_towers_without_insulator_line_bridge",
        "removed_bare_pole_physical_towers",
        "removed_physical_towers_without_connected_line_span",
    ):
        report[key] = int(state.get(key, 0))

    tower_box_report = make_standard_tower_boxes(
        [box for box in boxes if box["kind"] == "tower"],
        obb_origin,
        args.json_box_orientation,
    )
    line_box_report = make_standard_line_boxes(
        span_boxes, obb_origin, args.json_box_orientation
    )
    combined_box_report = tower_box_report + line_box_report
    conductor_render_report = make_conductor_render_report(conductor_reports, obb_origin)

    write_json(report_path, report)
    write_json(combined_box_report_path, combined_box_report)
    write_json(conductor_report_path, conductor_render_report)
    if tower_box_report_path is not None:
        write_json(tower_box_report_path, tower_box_report)

    print(f"Wrote LAS with drawings: {output_path}", flush=True)
    print(f"Wrote draw report: {report_path}", flush=True)
    print(f"Wrote combined boxes: {combined_box_report_path}", flush=True)
    print(f"Wrote conductors: {conductor_report_path}", flush=True)
    print(f"Finished draw stage in {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
