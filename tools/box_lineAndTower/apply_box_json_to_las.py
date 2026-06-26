"""Draw boxes from JSON back onto an existing LAS file.

Supported JSON formats:
1. Drawable edge format:
   {"boxes": [{"edges": [[[x1,y1,z1], [x2,y2,z2]], ...]}]}

2. OBB format:
   [{"class_name": "tower", "instance_name": "杆塔1",
     "obb": [cx, cy, cz, ex, ey, ez, qx, qy, qz, qw],
     "obb_global": {"lat_lng_alt": [ox, oy, oz]}}]

For the OBB format, obb[0:3] is treated as relative to
obb_global.lat_lng_alt when present.
"""
# 把保存的json文件，重新映射到源.las文件中
import argparse
import json
from pathlib import Path

import numpy as np


laspy = None
BOX_CLASS = 31
DEFAULT_TOWER_COLOR = np.array([65535, 0, 65535], dtype=np.uint16)
DEFAULT_LINE_COLOR = np.array([0, 65535, 65535], dtype=np.uint16)
DEFAULT_BOX_COLOR = np.array([65535, 65535, 0], dtype=np.uint16)
LINE_COLORS = np.array(
    [
        [0, 65535, 65535],
        [65535, 32768, 0],
        [0, 65535, 0],
        [65535, 0, 65535],
        [65535, 65535, 0],
        [0, 32768, 65535],
        [65535, 0, 0],
        [32768, 65535, 0],
    ],
    dtype=np.uint16,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply tower/line box JSON files back onto a LAS file."
    )
    parser.add_argument("--input", required=True, help="Original LAS/LAZ file.")
    parser.add_argument("--output", required=True, help="Output LAS/LAZ with box edges.")
    parser.add_argument(
        "--json",
        nargs="+",
        required=True,
        help="One or more box JSON files. Supports old edge JSON and new OBB JSON.",
    )
    parser.add_argument(
        "--edge-step",
        type=float,
        default=0.30,
        help="Sample spacing in meters along box edges.",
    )
    parser.add_argument(
        "--box-class",
        type=int,
        default=BOX_CLASS,
        help="LAS classification value for appended box-edge points.",
    )
    parser.add_argument(
        "--line-class",
        type=int,
        default=30,
        help="LAS classification value for appended conductor polyline points.",
    )
    parser.add_argument(
        "--line-diameter-mm",
        type=float,
        default=0.0,
        help=(
            "Render conductor polylines as point tubes with this diameter in "
            "millimeters. 0 keeps single-centerline rendering."
        ),
    )
    parser.add_argument(
        "--line-tube-sides",
        type=int,
        default=8,
        help="Number of cross-section points used when --line-diameter-mm > 0.",
    )
    parser.add_argument(
        "--keep-original-rgb",
        action="store_true",
        help="Do not recolor existing points; only appended box points are colored.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def has_rgb(las):
    dims = set(las.point_format.dimension_names)
    return {"red", "green", "blue"}.issubset(dims)


def quaternion_to_rotation_matrix(q):
    x, y, z, w = [float(v) for v in q]
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def color_for_record(record):
    class_name = str(record.get("class_name", "")).lower()
    if class_name == "tower" or str(record.get("name", "")).startswith("杆塔"):
        return DEFAULT_TOWER_COLOR
    if class_name in {"line", "line_span"} or "档距" in str(record.get("instance_name", "")):
        return DEFAULT_LINE_COLOR
    return DEFAULT_BOX_COLOR


def edges_from_corners(corners):
    return [[corners[a], corners[b]] for a, b in BOX_EDGE_PAIRS]


def record_to_edges(record):
    if "edges" in record:
        edges = []
        for edge in record["edges"]:
            if len(edge) != 2:
                continue
            edges.append([np.asarray(edge[0], dtype=np.float64), np.asarray(edge[1], dtype=np.float64)])
        return edges

    if "corners_xyz" in record:
        corners = np.asarray(record["corners_xyz"], dtype=np.float64)
        if corners.shape != (8, 3):
            raise ValueError("corners_xyz must be shape 8x3")
        return edges_from_corners(corners)

    if "obb" in record:
        obb = np.asarray(record["obb"], dtype=np.float64)
        if obb.shape[0] != 10:
            raise ValueError("obb must have 10 values")
        center = obb[:3]
        extent = obb[3:6]
        rotation = obb[6:10]
        origin = np.zeros(3, dtype=np.float64)
        if isinstance(record.get("obb_global"), dict) and "lat_lng_alt" in record["obb_global"]:
            origin = np.asarray(record["obb_global"]["lat_lng_alt"], dtype=np.float64)
        center = origin + center
        rot = quaternion_to_rotation_matrix(rotation)
        corners = center[None, :] + (BOX_CORNER_SIGNS * (extent / 2.0)[None, :]) @ rot.T
        return edges_from_corners(corners)

    return []


def load_box_records(paths):
    records = []
    conductor_lines = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict) and "spans" in data:
            origin = np.asarray(data.get("lat_lng_alt", [0.0, 0.0, 0.0]), dtype=np.float64)
            for span in data.get("spans", []):
                for line in span.get("lines", []):
                    pts = np.asarray(line.get("points", []), dtype=np.float64)
                    if pts.size == 0:
                        continue
                    conductor_lines.append(
                        {
                            "points": pts + origin[None, :],
                            "line_no": int(line.get("line_no", 1)),
                            "color_rgb_16": line.get("color_rgb_16"),
                        }
                    )
        elif isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict):
            if "boxes" in data and isinstance(data["boxes"], list):
                records.extend(data["boxes"])
            else:
                records.append(data)
        else:
            raise ValueError(f"Unsupported JSON root type in {path}: {type(data).__name__}")
    return records, conductor_lines


def sample_edges(edges, colors, step):
    points = []
    point_colors = []
    for edge, color in zip(edges, colors):
        p0, p1 = edge
        length = float(np.linalg.norm(p1 - p0))
        count = max(int(np.ceil(length / max(step, 1e-6))) + 1, 2)
        t = np.linspace(0.0, 1.0, count)
        sampled = p0[None, :] * (1.0 - t[:, None]) + p1[None, :] * t[:, None]
        points.append(sampled)
        point_colors.append(np.tile(color[None, :], (sampled.shape[0], 1)))
    if not points:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint16)
    return np.vstack(points), np.vstack(point_colors).astype(np.uint16, copy=False)


def tube_offsets_for_segment(p0, p1, radius, sides):
    direction = p1 - p0
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12 or radius <= 0:
        return np.empty((0, 3), dtype=np.float64)
    tangent = direction / norm
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(tangent, up))) > 0.95:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    normal = np.cross(tangent, up)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        return np.empty((0, 3), dtype=np.float64)
    normal /= normal_norm
    binormal = np.cross(tangent, normal)
    angles = np.linspace(0.0, 2.0 * np.pi, max(int(sides), 3), endpoint=False)
    return radius * (
        np.cos(angles)[:, None] * normal[None, :]
        + np.sin(angles)[:, None] * binormal[None, :]
    )


def sample_polylines(lines, step, diameter_mm=0.0, tube_sides=8):
    points = []
    colors = []
    radius = max(float(diameter_mm), 0.0) / 2000.0
    for line in lines:
        polyline = line["points"]
        if polyline.shape[0] < 2:
            continue
        if line.get("color_rgb_16"):
            color = np.asarray(line["color_rgb_16"], dtype=np.uint16)
        else:
            color = LINE_COLORS[(line["line_no"] - 1) % len(LINE_COLORS)]
        for p0, p1 in zip(polyline[:-1], polyline[1:]):
            length = float(np.linalg.norm(p1 - p0))
            count = max(int(np.ceil(length / max(step, 1e-6))) + 1, 2)
            t = np.linspace(0.0, 1.0, count)
            sampled = p0[None, :] * (1.0 - t[:, None]) + p1[None, :] * t[:, None]
            if radius > 0:
                offsets = tube_offsets_for_segment(p0, p1, radius, tube_sides)
                if offsets.size:
                    sampled = (sampled[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
            points.append(sampled)
            colors.append(np.tile(color[None, :], (sampled.shape[0], 1)))
    if not points:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint16)
    return np.vstack(points), np.vstack(colors).astype(np.uint16, copy=False)


def append_points(las, points, colors, box_class):
    if points.size == 0:
        return las
    records = laspy.ScaleAwarePointRecord.zeros(points.shape[0], header=las.header)
    records.x = points[:, 0]
    records.y = points[:, 1]
    records.z = points[:, 2]
    if "classification" in set(las.point_format.dimension_names):
        records.classification = np.full(points.shape[0], box_class, dtype=np.uint8)
    if has_rgb(las):
        records.red = colors[:, 0]
        records.green = colors[:, 1]
        records.blue = colors[:, 2]

    combined = np.concatenate([las.points.array, records.array])
    las.points = laspy.ScaleAwarePointRecord(
        combined, las.header.point_format, las.header.scales, las.header.offsets
    )
    return las


def main():
    args = parse_args()
    global laspy
    import laspy as laspy_module

    laspy = laspy_module
    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists, use --overwrite: {output_path}")

    print(f"Reading LAS: {input_path}", flush=True)
    las = laspy.read(input_path)
    records, conductor_lines = load_box_records(args.json)

    edges = []
    colors = []
    for record in records:
        record_edges = record_to_edges(record)
        if not record_edges:
            continue
        color = color_for_record(record)
        edges.extend(record_edges)
        colors.extend([color] * len(record_edges))

    print(f"Loaded {len(records)} box records, {len(edges)} edges", flush=True)
    points, point_colors = sample_edges(edges, colors, args.edge_step)
    print(f"Appending {points.shape[0]} edge points", flush=True)
    las = append_points(las, points, point_colors, args.box_class)

    conductor_points, conductor_colors = sample_polylines(
        conductor_lines,
        args.edge_step,
        diameter_mm=args.line_diameter_mm,
        tube_sides=args.line_tube_sides,
    )
    if conductor_points.size:
        print(
            f"Appending {conductor_points.shape[0]} conductor points",
            flush=True,
        )
        las = append_points(
            las, conductor_points, conductor_colors, args.line_class
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(output_path)
    print(f"Wrote LAS: {output_path}", flush=True)


if __name__ == "__main__":
    main()


""" 
使用实例：
/opt/conda/envs/pointcept/bin/python tools/apply_box_json_to_las.py \
  --input /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb009b5736892392a.las \
  --output /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb_from_json_conductors_v10_1p24mm.las \
  --json /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb_tower_line_boxes_v10_conductors.json \
  --edge-step 0.30 \
  --line-diameter-mm 200 \
  --line-tube-sides 8 \
  --overwrite
# 整体渲染
/opt/conda/envs/pointcept/bin/python tools/apply_box_json_to_las.py \
  --input /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb009b5736892392a.las \
  --output /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb_from_json_boxes_and_conductors_v10_thick.las \
  --json \
    /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb_tower_line_boxes_v10_boxes.json \
    /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb_tower_line_boxes_v10_conductors.json \
  --edge-step 0.30 \
  --line-diameter-mm 200 \
  --line-tube-sides 8 \
  --overwrite
  
  
    --edge-step 0.30 // 每隔0.3米采样一个渲染点
    --line-diameter-mm 200 // 导线渲染成200mm
    --line-tube-sides 8 // 每个导线截面用8个点模拟粗线
"""