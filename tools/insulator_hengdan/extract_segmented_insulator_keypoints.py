"""从分割完成的点云中提取每串绝缘子的两个几何端点。

输入点云需要已经包含语义类别：默认 classification=3 表示绝缘子。
脚本只对已有绝缘子点做实例聚类，不进行杆塔点恢复、导线拓扑推断等后处理。

示例：
python tools/infer/extract_segmented_insulator_keypoints.py \
  --input input_pred.las \
  --output insulator_keypoints.json \
  --voxel-size 0.20 \
  --min-points 30 \
  --endpoint-percentile 2.0 \
  --overwrite
"""

import argparse
import json
from itertools import product
from pathlib import Path

import laspy
import numpy as np

try:
    from plyfile import PlyData
except ImportError:
    PlyData = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="从分割后的 LAS/LAZ/PLY 中提取每串绝缘子的两个几何端点。"
    )
    parser.add_argument("--input", required=True, help="分割后的 LAS/LAZ/PLY 文件。")
    parser.add_argument("--output", required=True, help="输出 JSON 文件。")
    parser.add_argument(
        "--insulator-class",
        type=int,
        default=3,
        help="绝缘子语义类别，默认是 3。",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.20,
        help="实例连通域使用的体素边长，单位米；点云较稀疏时可以适当增大。",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(6, 18, 26),
        default=26,
        help="体素邻接方式，默认使用 26 邻域。",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=30,
        help="一个绝缘子实例至少包含的原始点数。",
    )
    parser.add_argument(
        "--min-height",
        type=float,
        default=0.0,
        help="绝缘子实例最小 Z 高度，单位米；默认不按高度过滤。",
    )
    parser.add_argument(
        "--endpoint-percentile",
        type=float,
        default=2.0,
        help=(
            "沿绝缘子 PCA 主轴提取端点时使用的两端分位数。默认取 2%% 和 98%%，"
            "可以降低少量离群点对端点的影响。"
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 JSON。")
    return parser.parse_args()


def find_ply_class_field(field_names):
    """识别 PLY 中常见的语义类别字段。"""
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
    lower_to_name = {name.lower(): name for name in field_names}
    for candidate in candidates:
        if candidate in lower_to_name:
            return lower_to_name[candidate]
    raise ValueError(
        "PLY 中没有找到语义类别字段；支持字段名：" + ", ".join(candidates)
    )


def read_segmented_cloud(path):
    """读取分割点云，返回全局 XYZ 坐标和语义类别。"""
    suffix = path.suffix.lower()
    if suffix in (".las", ".laz"):
        las = laspy.read(path)
        if "classification" not in set(las.point_format.dimension_names):
            raise ValueError(f"输入文件没有 classification 字段：{path}")
        coord = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
        classification = np.asarray(las.classification, dtype=np.int32)
        return coord, classification

    if suffix == ".ply":
        if PlyData is None:
            raise ImportError("读取 PLY 需要安装 plyfile：pip install plyfile")
        ply = PlyData.read(str(path))
        if "vertex" not in ply:
            raise ValueError(f"PLY 缺少 vertex 元素：{path}")
        vertex = ply["vertex"].data
        field_names = vertex.dtype.names or ()
        for axis in ("x", "y", "z"):
            if axis not in field_names:
                raise ValueError(f"PLY 缺少坐标字段 {axis}：{path}")
        class_field = find_ply_class_field(field_names)
        coord = np.column_stack((vertex["x"], vertex["y"], vertex["z"]))
        classification = np.asarray(vertex[class_field], dtype=np.int32).reshape(-1)
        return coord.astype(np.float64), classification

    raise ValueError(f"只支持 .las、.laz 或 .ply 输入：{path}")


def neighbor_offsets(connectivity):
    """生成一半体素邻域，避免对同一对体素重复检查。"""
    offsets = []
    for offset in product((-1, 0, 1), repeat=3):
        if offset == (0, 0, 0) or offset <= (0, 0, 0):
            continue
        nonzero_count = sum(value != 0 for value in offset)
        if connectivity == 6 and nonzero_count != 1:
            continue
        if connectivity == 18 and nonzero_count > 2:
            continue
        offsets.append(np.asarray(offset, dtype=np.int64))
    return offsets


def voxel_component_labels(points, voxel_size, connectivity):
    """使用体素连通域将绝缘子语义点划分成独立实例。"""
    if points.shape[0] == 0:
        return np.empty((0,), dtype=np.int32)
    if voxel_size <= 0:
        raise ValueError("--voxel-size 必须大于 0")

    origin = points.min(axis=0)
    voxel_coord = np.floor((points - origin) / float(voxel_size)).astype(np.int64)
    unique_voxels, point_to_voxel = np.unique(
        voxel_coord, axis=0, return_inverse=True
    )
    voxel_lookup = {tuple(voxel): index for index, voxel in enumerate(unique_voxels)}

    # 并查集只保存体素之间的连接，避免在密集点云上建立庞大的点级邻接矩阵。
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

    offsets = neighbor_offsets(connectivity)
    for voxel_id, voxel in enumerate(unique_voxels):
        for offset in offsets:
            neighbor_id = voxel_lookup.get(tuple(voxel + offset))
            if neighbor_id is not None:
                union(voxel_id, neighbor_id)

    roots = np.asarray(
        [find_root(index) for index in range(unique_voxels.shape[0])],
        dtype=np.int64,
    )
    _, voxel_labels = np.unique(roots, return_inverse=True)
    return voxel_labels[point_to_voxel].astype(np.int32, copy=False)


def xyz_list(point):
    return [round(float(value), 6) for value in point]


def extract_instance_endpoints(points, endpoint_percentile):
    """沿实例 PCA 主轴返回两个真实点云端点。

    绝缘子可能倾斜或接近水平，因此不能使用 Z 最小点和 Z 最大点作为端点。
    这里先求实例的三维主方向，再沿主方向寻找两端的稳健分位位置，最后吸附
    到距离目标位置最近的原始点，保证 JSON 坐标确实来自输入点云。
    """
    center = points.mean(axis=0)
    centered = points - center[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]

    # 固定主轴符号，保证同一输入多次运行时 endpoint_1/endpoint_2 顺序稳定。
    dominant_axis = int(np.argmax(np.abs(axis)))
    if axis[dominant_axis] < 0:
        axis = -axis

    projection = centered @ axis
    percentile = float(np.clip(endpoint_percentile, 0.0, 49.0))
    endpoint_1_projection = float(np.percentile(projection, percentile))
    endpoint_2_projection = float(np.percentile(projection, 100.0 - percentile))
    endpoint_1_target = center + endpoint_1_projection * axis
    endpoint_2_target = center + endpoint_2_projection * axis
    endpoint_1 = points[
        int(np.argmin(np.sum((points - endpoint_1_target[None, :]) ** 2, axis=1)))
    ]
    endpoint_2 = points[
        int(np.argmin(np.sum((points - endpoint_2_target[None, :]) ** 2, axis=1)))
    ]
    return endpoint_1, endpoint_2


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
    coord, classification = read_segmented_cloud(input_path)
    insulator_mask = classification == int(args.insulator_class)
    insulator_points = coord[insulator_mask]
    print(
        f"总点数={coord.shape[0]}，绝缘子点数={insulator_points.shape[0]}",
        flush=True,
    )

    labels = voxel_component_labels(
        insulator_points,
        voxel_size=args.voxel_size,
        connectivity=args.connectivity,
    )

    instances = []
    for label in sorted(set(labels.tolist())):
        points = insulator_points[labels == label]
        if points.shape[0] < int(args.min_points):
            continue
        height = float(points[:, 2].max() - points[:, 2].min())
        if height < float(args.min_height):
            continue
        endpoint_1, endpoint_2 = extract_instance_endpoints(
            points, args.endpoint_percentile
        )
        middle_point = (endpoint_1 + endpoint_2) / 2.0
        instances.append(
            {
                "endpoint_1_xyz": xyz_list(endpoint_1),
                "middle_point_xyz": xyz_list(middle_point),
                "endpoint_2_xyz": xyz_list(endpoint_2),
            }
        )

    # 使用空间位置稳定排序，保证相同输入多次运行的实例编号一致。
    instances.sort(
        key=lambda item: (
            (item["endpoint_1_xyz"][0] + item["endpoint_2_xyz"][0]) / 2.0,
            (item["endpoint_1_xyz"][1] + item["endpoint_2_xyz"][1]) / 2.0,
            -(item["endpoint_1_xyz"][2] + item["endpoint_2_xyz"][2]) / 2.0,
        )
    )
    instances = [
        {"id": instance_id, **instance}
        for instance_id, instance in enumerate(instances, start=1)
    ]

    data = {
        "coordinate_system": "input_global_xyz",
        "insulators": instances,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"保存完成：{output_path}，绝缘子实例={len(instances)}", flush=True)


if __name__ == "__main__":
    main()


""" 
python tools/infer/extract_segmented_insulator_keypoints.py \
  --input /24085403037/24085403037/shixi/dataset/6_23_demo/test/test_insulator_hengdan/110v12_merged_4cls_Output.las \
  --output /24085403037/24085403037/shixi/dataset/6_23_demo/test/insulator/tower_004_keypoints.json \
  --voxel-size 0.20 \
  --min-points 30 \
  --overwrite 
  
  """
