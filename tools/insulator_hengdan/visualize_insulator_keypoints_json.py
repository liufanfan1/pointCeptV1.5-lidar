"""把统一 JSON 中的绝缘子端点和横担关键点绘制到 LAS/LAZ 点云中。

颜色约定：
    绝缘子端点1：蓝色，classification=20
    绝缘子中心点：绿色，classification=21
    绝缘子端点2：红色，classification=22
    横担左点：品红，classification=23
    横担右点：青色，classification=24
    横担中点：黄色，classification=25
    横担连线：绿色，classification=26

示例：
python tools/insulator_hengdan/visualize_insulator_keypoints_json.py \
  --input /24085403037/24085403037/shixi/dataset/0617-4Name/110v12/110v12_merged_4cls.las \
  --json /24085403037/24085403037/shixi/dataset/6_23_demo/test/hengdan_insulator/110v12_merged_4cls_hengdan_insulator.json \
  --output /24085403037/24085403037/shixi/dataset/6_23_demo/test/110v12_keypoints_visual.las \
  --marker-shape cube \
  --marker-size 0.50 \
  --marker-step 0.10 \
  --overwrite

目录批处理示例：
python tools/insulator_hengdan/visualize_insulator_keypoints_json.py \
  --input /path/to/las_dir \
  --json /path/to/keypoints_json_dir \
  --output /path/to/visualized_las_dir \
  --marker-shape cube \
  --marker-size 0.50 \
  --marker-step 0.10 \
  --overwrite
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import laspy
import numpy as np


ENDPOINT_STYLE = {
    "endpoint_1_xyz": {
        "name": "端点1",
        "classification": 20,
        "color": (0, 0, 65535),
    },
    "middle_point_xyz": {
        "name": "绝缘子中心点",
        "classification": 21,
        "color": (0, 65535, 0),
    },
    "endpoint_2_xyz": {
        "name": "端点2",
        "classification": 22,
        "color": (65535, 0, 0),
    },
}

CROSSARM_STYLE = {
    "left_point_xyz": {
        "name": "横担左点",
        "classification": 23,
        "color": (65535, 0, 65535),
    },
    "middle_point_xyz": {
        "name": "横担中点",
        "classification": 25,
        "color": (65535, 65535, 0),
    },
    "right_point_xyz": {
        "name": "横担右点",
        "classification": 24,
        "color": (0, 65535, 65535),
    },
}
CROSSARM_LINE_CLASS = 26
CROSSARM_LINE_COLOR = (0, 65535, 0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="读取统一关键点 JSON，并绘制绝缘子端点和横担位置。"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="原始或分割后的 LAS/LAZ 文件，或者递归输入目录。",
    )
    parser.add_argument(
        "--json",
        required=True,
        help="统一关键点 JSON 文件，或者递归 JSON 目录。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="带关键点标记的输出 LAS/LAZ，或者批处理输出目录。",
    )
    parser.add_argument(
        "--json-suffix",
        default="",
        help=(
            "目录模式下 JSON 文件名相对 LAS 文件名增加的后缀。"
            "默认空，例如 tower_001.las 对应 tower_001.json；"
            "设置为 _keypoints 时对应 tower_001_keypoints.json。"
        ),
    )
    parser.add_argument(
        "--marker-shape",
        choices=("cube", "cross"),
        default="cube",
        help="关键点标记形状，默认使用立方体表面。",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.50,
        help="关键点标记边长，单位米。",
    )
    parser.add_argument(
        "--marker-step",
        type=float,
        default=0.10,
        help="标记点采样间隔，单位米。",
    )
    parser.add_argument(
        "--crossarm-line-step",
        type=float,
        default=0.10,
        help="横担左右端点连线的采样间隔，单位米。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出文件。")
    return parser.parse_args()


def read_keypoints(path):
    """读取嵌套 JSON，并展开成现有渲染函数使用的两个列表。"""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("JSON 根节点必须是对象。")

    tower_records = data.get("towers")
    if not isinstance(tower_records, list):
        raise ValueError("JSON 中缺少 towers 数组。")
    insulators = []
    crossarms = []

    def parse_xyz(value, description):
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"{description}必须是三个有效坐标。")
        return point

    for tower_position, tower in enumerate(tower_records, start=1):
        if not isinstance(tower, dict):
            raise ValueError(f"第 {tower_position} 个杆塔记录不是对象。")
        tower_id = int(tower.get("tower_id", tower_position))
        tower_crossarms = tower.get("crossarms")
        if not isinstance(tower_crossarms, list):
            raise ValueError(f"杆塔 {tower_id} 缺少 crossarms 数组。")

        for crossarm_position, crossarm in enumerate(tower_crossarms, start=1):
            if not isinstance(crossarm, dict):
                raise ValueError(
                    f"杆塔 {tower_id} 的第 {crossarm_position} 个横担不是对象。"
                )
            crossarm_id = int(crossarm.get("crossarm_id", crossarm_position))
            left_endpoint = crossarm.get("left_endpoint")
            right_endpoint = crossarm.get("right_endpoint")
            if not isinstance(left_endpoint, dict) or not isinstance(
                right_endpoint, dict
            ):
                raise ValueError(
                    f"杆塔 {tower_id} 横担 {crossarm_id} 缺少左右端点对象。"
                )

            crossarm_points = {
                "left_point_xyz": parse_xyz(
                    left_endpoint.get("point_xyz"),
                    f"杆塔 {tower_id} 横担 {crossarm_id} 的左端点",
                ),
                "middle_point_xyz": parse_xyz(
                    crossarm.get("middle_point_xyz"),
                    f"杆塔 {tower_id} 横担 {crossarm_id} 的中点",
                ),
                "right_point_xyz": parse_xyz(
                    right_endpoint.get("point_xyz"),
                    f"杆塔 {tower_id} 横担 {crossarm_id} 的右端点",
                ),
            }
            crossarms.append(
                {
                    "tower_id": tower_id,
                    "crossarm_id": crossarm_id,
                    "points": crossarm_points,
                }
            )

            # 按 JSON 中固定的左端、右端顺序展开绝缘子，保留端点内编号。
            for side_name, endpoint_record in (
                ("left", left_endpoint),
                ("right", right_endpoint),
            ):
                endpoint_insulators = endpoint_record.get("insulators")
                if not isinstance(endpoint_insulators, list):
                    raise ValueError(
                        f"杆塔 {tower_id} 横担 {crossarm_id} 的 {side_name} "
                        "端点缺少 insulators 数组。"
                    )
                for insulator_position, instance in enumerate(
                    endpoint_insulators, start=1
                ):
                    if not isinstance(instance, dict):
                        raise ValueError(
                            f"杆塔 {tower_id} 横担 {crossarm_id} 的 {side_name} "
                            f"端第 {insulator_position} 个绝缘子不是对象。"
                        )
                    insulator_id = int(
                        instance.get("insulator_id", insulator_position)
                    )
                    points = {
                        key: parse_xyz(
                            instance.get(key),
                            f"杆塔 {tower_id} 横担 {crossarm_id} 的 {side_name} "
                            f"端绝缘子 {insulator_id} 的 {key}",
                        )
                        for key in ENDPOINT_STYLE
                    }
                    insulators.append(
                        {
                            "id": insulator_id,
                            "tower_id": tower_id,
                            "crossarm_id": crossarm_id,
                            "side": side_name,
                            "points": points,
                        }
                    )

    if not insulators and not crossarms:
        raise ValueError("JSON 中没有绝缘子或横担关键点。")
    return insulators, crossarms


def sample_values(size, step):
    if size <= 0:
        raise ValueError("--marker-size 必须大于 0")
    if step <= 0:
        raise ValueError("--marker-step 必须大于 0")
    half = float(size) / 2.0
    values = np.arange(-half, half + float(step) * 0.5, float(step))
    # 无论步长是否整除边长，都确保标记包含两端表面。
    return np.unique(np.concatenate((values, np.asarray([-half, half]))))


def cube_marker(center, size, step):
    """围绕关键点生成立方体六个表面的点。"""
    values = sample_values(size, step)
    half = float(size) / 2.0
    center = np.asarray(center, dtype=np.float64)
    parts = []
    for fixed_axis in range(3):
        free_axes = [axis for axis in range(3) if axis != fixed_axis]
        grid_a, grid_b = np.meshgrid(values, values, indexing="ij")
        for fixed_value in (-half, half):
            offsets = np.zeros((grid_a.size, 3), dtype=np.float64)
            offsets[:, fixed_axis] = fixed_value
            offsets[:, free_axes[0]] = grid_a.ravel()
            offsets[:, free_axes[1]] = grid_b.ravel()
            parts.append(center[None, :] + offsets)
    return np.concatenate(parts, axis=0)


def cross_marker(center, size, step):
    """围绕关键点生成沿 XYZ 三个方向的十字标记。"""
    values = sample_values(size, step)
    center = np.asarray(center, dtype=np.float64)
    parts = []
    for axis in range(3):
        offsets = np.zeros((values.shape[0], 3), dtype=np.float64)
        offsets[:, axis] = values
        parts.append(center[None, :] + offsets)
    return np.concatenate(parts, axis=0)


def sample_segment(start, end, step):
    """在横担左右端点之间均匀采样连线。"""
    if step <= 0:
        raise ValueError("--crossarm-line-step 必须大于 0")
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    length = float(np.linalg.norm(end - start))
    count = max(int(np.ceil(length / float(step))) + 1, 2)
    ratio = np.linspace(0.0, 1.0, count, dtype=np.float64)[:, None]
    return start[None, :] * (1.0 - ratio) + end[None, :] * ratio


def build_marker_arrays(insulators, crossarms, shape, size, step, line_step):
    """生成绝缘子端点、横担三点和横担连线的可视化点。"""
    point_parts = []
    class_parts = []
    color_parts = []
    marker_function = cube_marker if shape == "cube" else cross_marker

    for instance in insulators:
        for key, point in instance["points"].items():
            style = ENDPOINT_STYLE[key]
            points = marker_function(point, size, step)
            point_parts.append(points)
            class_parts.append(
                np.full(points.shape[0], style["classification"], dtype=np.uint8)
            )
            color_parts.append(
                np.tile(np.asarray(style["color"], dtype=np.uint16), (points.shape[0], 1))
            )

    for crossarm in crossarms:
        for key, point in crossarm["points"].items():
            style = CROSSARM_STYLE[key]
            points = marker_function(point, size, step)
            point_parts.append(points)
            class_parts.append(
                np.full(points.shape[0], style["classification"], dtype=np.uint8)
            )
            color_parts.append(
                np.tile(np.asarray(style["color"], dtype=np.uint16), (points.shape[0], 1))
            )

        line_points = sample_segment(
            crossarm["points"]["left_point_xyz"],
            crossarm["points"]["right_point_xyz"],
            line_step,
        )
        point_parts.append(line_points)
        class_parts.append(
            np.full(line_points.shape[0], CROSSARM_LINE_CLASS, dtype=np.uint8)
        )
        color_parts.append(
            np.tile(
                np.asarray(CROSSARM_LINE_COLOR, dtype=np.uint16),
                (line_points.shape[0], 1),
            )
        )

    if not point_parts:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.uint8),
            np.empty((0, 3), dtype=np.uint16),
        )
    return (
        np.concatenate(point_parts, axis=0),
        np.concatenate(class_parts, axis=0),
        np.concatenate(color_parts, axis=0),
    )


def append_markers(input_las, points, classifications, colors):
    """创建标记点记录，并追加到原始点记录之后。"""
    records = laspy.ScaleAwarePointRecord.zeros(points.shape[0], header=input_las.header)
    if points.shape[0]:
        records.x = points[:, 0]
        records.y = points[:, 1]
        records.z = points[:, 2]

        dimensions = set(input_las.point_format.dimension_names)
        if "classification" in dimensions:
            records.classification = classifications
        if {"red", "green", "blue"}.issubset(dimensions):
            records.red = colors[:, 0]
            records.green = colors[:, 1]
            records.blue = colors[:, 2]
        else:
            print("警告：输入 LAS 没有 RGB 字段，只能通过 classification 查看标记。")

    combined = np.concatenate((input_las.points.array, records.array))
    input_las.points = laspy.ScaleAwarePointRecord(
        combined,
        input_las.header.point_format,
        input_las.header.scales,
        input_las.header.offsets,
    )


def is_las_path(path):
    """判断路径是否为支持的点云文件。"""
    return path.is_file() and path.suffix.lower() in (".las", ".laz")


def find_las_files(root):
    """递归查找 LAS/LAZ，同时兼容大写扩展名。"""
    return sorted(
        path for path in root.rglob("*") if is_las_path(path)
    )


def find_json_files(root):
    """递归查找 JSON，同时兼容大写扩展名。"""
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".json"
    )


def build_jobs(input_path, json_path, output_path, json_suffix):
    """建立 LAS、关键点 JSON 和输出 LAS 的一一对应任务。"""
    if input_path.is_file():
        if not is_las_path(input_path):
            raise ValueError(f"--input 只支持 .las 或 .laz：{input_path}")
        if not json_path.is_file() or json_path.suffix.lower() != ".json":
            raise ValueError(f"单文件模式下 --json 必须是 JSON 文件：{json_path}")
        if output_path.suffix.lower() not in (".las", ".laz"):
            raise ValueError(f"单文件模式下 --output 必须是 .las 或 .laz：{output_path}")
        return [(input_path, json_path, output_path)]

    if not input_path.is_dir():
        raise FileNotFoundError(f"LAS 输入不存在：{input_path}")
    if not json_path.is_dir():
        raise ValueError("目录批处理时，--json 必须是 JSON 目录。")
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("目录批处理时，--output 必须是输出目录。")

    las_files = find_las_files(input_path)
    if not las_files:
        raise FileNotFoundError(f"输入目录中没有 LAS/LAZ 文件：{input_path}")

    json_files = find_json_files(json_path)
    if not json_files:
        raise FileNotFoundError(f"JSON 目录中没有 JSON 文件：{json_path}")

    # 相对路径优先，目录结构不同时再按全目录唯一文件名配对。
    json_by_relative_stem = {}
    json_by_name_stem = defaultdict(list)
    for path in json_files:
        relative_stem = path.relative_to(json_path).with_suffix("")
        json_by_relative_stem[str(relative_stem).replace("\\", "/").lower()] = path
        json_by_name_stem[path.stem.lower()].append(path)

    jobs = []
    missing = []
    ambiguous = []
    for source_path in las_files:
        relative_path = source_path.relative_to(input_path)
        expected_stem = f"{source_path.stem}{json_suffix}"
        relative_json_stem = relative_path.parent / expected_stem
        relative_key = str(relative_json_stem).replace("\\", "/").lower()
        matched_json = json_by_relative_stem.get(relative_key)

        if matched_json is None:
            candidates = json_by_name_stem.get(expected_stem.lower(), [])
            if len(candidates) == 1:
                matched_json = candidates[0]
            elif len(candidates) > 1:
                ambiguous.append((source_path, candidates))
                continue

        if matched_json is None:
            missing.append(source_path)
            continue

        jobs.append((source_path, matched_json, output_path / relative_path))

    if ambiguous:
        details = "\n".join(
            f"  {source} -> {len(candidates)} 个同名 JSON"
            for source, candidates in ambiguous[:10]
        )
        raise ValueError(
            "以下 LAS 找到了多个同名 JSON，请让 JSON 保持相同相对目录结构：\n"
            f"{details}"
        )
    if missing:
        details = "\n".join(f"  {path}" for path in missing[:10])
        extra = "" if len(missing) <= 10 else f"\n  ...另有 {len(missing) - 10} 个"
        raise FileNotFoundError(
            f"有 {len(missing)} 个 LAS 找不到对应 JSON。"
            "请检查目录结构、文件名或 --json-suffix：\n"
            f"{details}{extra}"
        )
    return jobs


def process_one(input_path, json_path, output_path, args):
    """把一个 JSON 中的关键点追加到对应 LAS/LAZ。"""
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在，请添加 --overwrite：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    insulators, crossarms = read_keypoints(json_path)
    print(
        f"  读取 JSON：{json_path}，绝缘子={len(insulators)}，横担={len(crossarms)}",
        flush=True,
    )
    input_las = laspy.read(input_path)
    original_count = len(input_las.points)

    points, classifications, colors = build_marker_arrays(
        insulators,
        crossarms,
        shape=args.marker_shape,
        size=args.marker_size,
        step=args.marker_step,
        line_step=args.crossarm_line_step,
    )
    append_markers(input_las, points, classifications, colors)
    input_las.write(output_path)

    print(
        f"  保存完成：{output_path}，原始点={original_count}，"
        f"追加标记点={points.shape[0]}",
        flush=True,
    )
    return original_count, int(points.shape[0])


def main():
    args = parse_args()
    input_path = Path(args.input)
    json_path = Path(args.json)
    output_path = Path(args.output)
    jobs = build_jobs(
        input_path,
        json_path,
        output_path,
        args.json_suffix,
    )

    total_start = time.perf_counter()
    accumulated_file_time = 0.0
    total_input_points = 0
    total_marker_points = 0
    for index, (source_path, source_json, target_path) in enumerate(jobs, start=1):
        file_start = time.perf_counter()
        print(f"[{index}/{len(jobs)}] 处理：{source_path}", flush=True)
        input_count, marker_count = process_one(
            source_path,
            source_json,
            target_path,
            args,
        )
        file_time = time.perf_counter() - file_start
        accumulated_file_time += file_time
        total_input_points += input_count
        total_marker_points += marker_count
        print(f"  单文件耗时：{file_time:.2f}s", flush=True)

    total_time = time.perf_counter() - total_start
    average_time = accumulated_file_time / len(jobs)
    print(
        "全部完成：文件数={}，总耗时={}，平均每个LAS={}，"
        "原始点总数={}，追加标记点总数={}".format(
            len(jobs),
            f"{total_time:.2f}s",
            f"{average_time:.2f}s",
            total_input_points,
            total_marker_points,
        ),
        flush=True,
    )
    print(
        "绝缘子颜色：端点1=蓝色(class 20)，中心点=绿色(class 21)，"
        "端点2=红色(class 22)",
        flush=True,
    )
    print(
        "横担颜色：左点=品红(class 23)，中点=黄色(class 25)，"
        "右点=青色(class 24)，连线=绿色(class 26)",
        flush=True,
    )


if __name__ == "__main__":
    main()
