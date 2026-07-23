"""批量合并输电线路四类别 LAS，或根据 metadata 恢复原始数据集划分。

Pointcept 数据目录中的 .pth 已经经过分块和数据增强，不适合反向拼回原始场景。
本脚本支持两种模式：
1. 默认读取预处理阶段保存的 metadata.json，按 split 查找和合并原始场景；
2. 使用 --direct-merge 递归扫描源目录，不依赖 metadata，直接批量合并场景。

每个场景目录应包含 0_ground、1_tower、2_line、3_insulator 四类 LAS/LAZ。
合并时根据文件名前缀写入 classification=0/1/2/3。
"""

import argparse
import copy
import json
import re
from pathlib import Path

import laspy
import numpy as np


CLASS_FILE_PATTERN = re.compile(r"^([0-3])[_-].+\.la[sz]$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge four-class transmission-line LAS scenes directly or recover "
            "original train/val/test scenes from Pointcept metadata."
        )
    )
    parser.add_argument(
        "--direct-merge",
        action="store_true",
        help="递归扫描源目录并直接合并四类 LAS，不读取 metadata。",
    )
    parser.add_argument(
        "--metadata",
        default="data/transmission_line_stage1_random/metadata.json",
        help="预处理生成的 metadata.json。",
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="原始场景根目录，每个子目录包含四个类别 LAS。",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="test",
        help="需要恢复的划分，默认 test。",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=None,
        help="只处理指定场景，可重复传入；例如 --scene 110v12。",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="映射报告 JSON；默认写到合并目录或当前目录。",
    )
    parser.add_argument(
        "--merge-output-dir",
        default=None,
        help=(
            "将选中场景的四类 LAS 合并到该目录；--direct-merge 模式下必填。"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2_000_000,
        help="流式合并每次读取的点数，默认 200 万。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "直接合并模式断点续传：跳过点数完整的已有输出，自动重建"
            "异常中断产生的不完整输出。"
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有文件。")
    return parser.parse_args()


def scene_class_files(scene_dir):
    """按类别编号返回原始场景中的四个 LAS/LAZ。"""
    files = {}
    for path in scene_dir.iterdir():
        if not path.is_file():
            continue
        match = CLASS_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        label = int(match.group(1))
        if label in files:
            raise ValueError(
                f"场景 {scene_dir.name} 的类别 {label} 存在多个文件："
                f"{files[label]} 和 {path}"
            )
        files[label] = path
    missing = sorted(set(range(4)) - set(files))
    if missing:
        raise FileNotFoundError(f"场景 {scene_dir} 缺少类别文件：{missing}")
    return [(files[label], label) for label in range(4)]


def discover_direct_scenes(source_root):
    """递归查找至少包含一个类别文件的目录，并验证四个类别是否完整。"""
    candidates = {
        path.parent
        for path in source_root.rglob("*")
        if path.is_file() and CLASS_FILE_PATTERN.match(path.name)
    }
    if not candidates:
        raise FileNotFoundError(
            f"目录中没有找到 0_ground 到 3_insulator 类别文件：{source_root}"
        )

    scenes = []
    for scene_dir in sorted(
        candidates,
        key=lambda path: path.relative_to(source_root).as_posix().lower(),
    ):
        # 在扫描阶段验证完整性，使缺类或重复类别尽早报告，而不是静默跳过。
        scene_class_files(scene_dir)
        scenes.append(scene_dir)
    return scenes


def select_direct_scenes(scene_dirs, source_root, requested_scenes):
    """按目录名或相对路径筛选直接合并的场景。"""
    if not requested_scenes:
        return scene_dirs
    wanted = {
        str(name).replace("\\", "/").strip("/").lower()
        for name in requested_scenes
    }
    selected = []
    for scene_dir in scene_dirs:
        relative_name = scene_dir.relative_to(source_root).as_posix()
        if (
            scene_dir.name.lower() in wanted
            or relative_name.lower() in wanted
        ):
            selected.append(scene_dir)
    if not selected:
        raise ValueError(
            "直接合并模式没有找到 --scene 指定的场景："
            + ", ".join(requested_scenes)
        )
    return selected


def direct_output_path(scene_dir, source_root, merge_root):
    """保留场景父目录的相对层级，生成合并 LAS 输出路径。"""
    relative = scene_dir.relative_to(source_root)
    return merge_root / relative.parent / f"{scene_dir.name}_merged_4cls.las"


def compatible_header(reference, current, path):
    """确认四类 LAS 的基础点格式兼容；额外维度差异会在写出时归一化。"""
    if current.point_format.id != reference.point_format.id:
        raise ValueError(f"点格式不一致，无法直接合并：{path}")
    if current.version != reference.version:
        raise ValueError(f"LAS 版本不一致，无法直接合并：{path}")


def unified_scaling(headers):
    """根据全部类别 LAS 的范围生成不会降低坐标精度的公共 scale/offset。"""
    scales = np.min(np.stack([header.scales for header in headers]), axis=0)
    global_min = np.min(np.stack([header.mins for header in headers]), axis=0)
    global_max = np.max(np.stack([header.maxs for header in headers]), axis=0)
    offsets = (global_min + global_max) / 2.0
    max_integer = np.max(
        np.abs(np.stack((global_min - offsets, global_max - offsets)) / scales),
        axis=0,
    )
    if np.any(max_integer > np.iinfo(np.int32).max):
        raise OverflowError(
            "场景坐标范围在当前 scale 下超出 LAS int32 表示范围："
            f"scale={scales.tolist()}"
        )
    return scales, offsets


def normalize_points(points, output_header, label):
    """将带不同额外维度的点记录转换为输出头要求的统一格式。"""
    normalized = laspy.ScaleAwarePointRecord.zeros(
        len(points), header=output_header
    )
    # 只复制源文件和输出格式共有的字段；输出独有字段保持默认值。
    normalized.copy_fields_from(points)
    # copy_fields_from 会复制整数 XYZ，必须使用真实坐标重新编码 scale/offset。
    normalized.x = np.asarray(points.x)
    normalized.y = np.asarray(points.y)
    normalized.z = np.asarray(points.z)
    normalized.classification = np.full(
        len(points), label, dtype=np.uint8
    )
    return normalized


def source_point_counts(class_files):
    """只读取文件头，获得四个类别源文件的点数。"""
    counts = [0, 0, 0, 0]
    for path, label in class_files:
        with laspy.open(path) as reader:
            counts[label] = int(reader.header.point_count)
    return counts


def output_has_expected_points(output_path, expected_total):
    """通过输出文件头点数判断断点文件是否已经完整写出。"""
    if not output_path.is_file():
        return False
    try:
        with laspy.open(output_path) as reader:
            return int(reader.header.point_count) == int(expected_total)
    except Exception:
        return False


def merge_scene(scene_dir, output_path, chunk_size, overwrite):
    """流式合并一个原始场景，并根据文件名前缀写入 classification。"""
    class_files = scene_class_files(scene_dir)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"输出已存在，请添加 --overwrite：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = []
    for path, _ in class_files:
        with laspy.open(path) as reader:
            headers.append(copy.deepcopy(reader.header))
    reference_header = headers[0]
    for (path, _), header in zip(class_files[1:], headers[1:]):
        compatible_header(reference_header, header, path)
    output_scales, output_offsets = unified_scaling(headers)
    reference_header.scales = output_scales
    reference_header.offsets = output_offsets

    counts = [0, 0, 0, 0]
    with laspy.open(output_path, mode="w", header=reference_header) as writer:
        for path, label in class_files:
            with laspy.open(path) as reader:
                needs_normalization = (
                    reader.header.point_format != reference_header.point_format
                )
                if needs_normalization:
                    source_extra = list(
                        reader.header.point_format.extra_dimension_names
                    )
                    output_extra = list(
                        reference_header.point_format.extra_dimension_names
                    )
                    print(
                        "  normalizing incompatible extra dimensions: "
                        f"{path.name}, source={source_extra}, output={output_extra}",
                        flush=True,
                    )
                for points in reader.chunk_iterator(int(chunk_size)):
                    if needs_normalization:
                        output_points = normalize_points(
                            points, reference_header, label
                        )
                    else:
                        points.classification = np.full(
                            len(points), label, dtype=np.uint8
                        )
                        # 不同类别文件可能使用不同 offset，写出前统一重编码 XYZ。
                        points.change_scaling(
                            scales=output_scales,
                            offsets=output_offsets,
                        )
                        output_points = points
                    writer.write_points(output_points)
                    counts[label] += int(len(points))
    return counts


def write_report(report_path, report, overwrite):
    """写出 JSON 报告，并统一处理覆盖规则。"""
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"报告已存在，请添加 --overwrite：{report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote mapping report: {report_path}", flush=True)


def run_direct_merge(args, source_root):
    """不依赖 metadata，递归发现并批量合并四类别场景。"""
    if not args.merge_output_dir:
        raise ValueError("--direct-merge 必须同时指定 --merge-output-dir")
    if args.resume and args.overwrite:
        raise ValueError("--resume 和 --overwrite 不能同时使用")
    merge_root = Path(args.merge_output_dir)
    if merge_root.exists() and not merge_root.is_dir():
        raise NotADirectoryError(f"合并输出路径不是目录：{merge_root}")
    merge_root.mkdir(parents=True, exist_ok=True)

    report_path = (
        Path(args.report)
        if args.report
        else merge_root / "direct_merge_report.json"
    )
    if report_path.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"报告已存在，请添加 --overwrite：{report_path}")

    scene_dirs = discover_direct_scenes(source_root)
    scene_dirs = select_direct_scenes(scene_dirs, source_root, args.scene)
    records = []
    for index, scene_dir in enumerate(scene_dirs, start=1):
        relative_scene = scene_dir.relative_to(source_root).as_posix()
        output_path = direct_output_path(scene_dir, source_root, merge_root)
        class_files = scene_class_files(scene_dir)
        expected_counts = source_point_counts(class_files)
        expected_total = int(sum(expected_counts))
        if args.resume and output_has_expected_points(output_path, expected_total):
            print(
                f"[{index}/{len(scene_dirs)}] skipping complete {relative_scene}: "
                f"{expected_total} points",
                flush=True,
            )
            records.append(
                {
                    "scene": scene_dir.name,
                    "relative_scene": relative_scene,
                    "source_dir": str(scene_dir),
                    "class_files": {
                        str(label): str(path) for path, label in class_files
                    },
                    "merged_las": str(output_path),
                    "point_counts": expected_counts,
                    "total_points": expected_total,
                    "status": "skipped_complete",
                }
            )
            continue
        replace_incomplete = args.resume and output_path.exists()
        if replace_incomplete:
            print(
                f"[{index}/{len(scene_dirs)}] rebuilding incomplete "
                f"{relative_scene}",
                flush=True,
            )
        else:
            print(
                f"[{index}/{len(scene_dirs)}] merging {relative_scene}",
                flush=True,
            )
        counts = merge_scene(
            scene_dir,
            output_path,
            args.chunk_size,
            args.overwrite or replace_incomplete,
        )
        records.append(
            {
                "scene": scene_dir.name,
                "relative_scene": relative_scene,
                "source_dir": str(scene_dir),
                "class_files": {
                    str(label): str(path) for path, label in class_files
                },
                "merged_las": str(output_path),
                "point_counts": counts,
                "total_points": int(sum(counts)),
                "status": (
                    "rebuilt_incomplete" if replace_incomplete else "merged"
                ),
            }
        )
        print(f"  wrote {output_path}: {sum(counts)} points", flush=True)

    report = {
        "mode": "direct_merge",
        "source_root": str(source_root),
        "merge_output_dir": str(merge_root),
        "scene_count": len(records),
        "scenes": records,
    }
    write_report(report_path, report, args.overwrite or args.resume)


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"原始数据根目录不存在：{source_root}")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size 必须大于 0")

    if args.direct_merge:
        run_direct_merge(args, source_root)
        return

    metadata_path = Path(args.metadata)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata 不存在：{metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = [
        item
        for item in metadata.get("scenes", [])
        if args.split == "all" or item.get("split") == args.split
    ]
    if args.scene:
        wanted = {name.lower() for name in args.scene}
        selected = [
            item
            for item in selected
            if str(item.get("scene", "")).lower() in wanted
            or str(item.get("source_scene", "")).lower() in wanted
        ]
    if not selected:
        raise ValueError(f"metadata 中没有 split={args.split} 的场景")

    merge_root = Path(args.merge_output_dir) if args.merge_output_dir else None
    if merge_root is not None:
        merge_root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, item in enumerate(selected, start=1):
        source_scene = str(item["source_scene"])
        scene_id = str(item["scene"])
        source_dir = source_root / source_scene
        if not source_dir.is_dir():
            raise FileNotFoundError(f"原始场景目录不存在：{source_dir}")
        class_files = scene_class_files(source_dir)
        record = {
            "split": item["split"],
            "scene": scene_id,
            "source_scene": source_scene,
            "source_dir": str(source_dir),
            "tile_count": int(item.get("tiles", 0)),
            "class_files": {
                str(label): str(path) for path, label in class_files
            },
        }
        if merge_root is not None:
            output_path = merge_root / f"{scene_id}_merged_4cls.las"
            print(f"[{index}/{len(selected)}] merging {source_scene}", flush=True)
            counts = merge_scene(
                source_dir, output_path, args.chunk_size, args.overwrite
            )
            record["merged_las"] = str(output_path)
            record["point_counts"] = counts
            print(f"  wrote {output_path}: {sum(counts)} points", flush=True)
        else:
            print(
                f"[{index}/{len(selected)}] {item['split']}: "
                f"{scene_id} -> {source_dir}",
                flush=True,
            )
        records.append(record)

    if args.report:
        report_path = Path(args.report)
    elif merge_root is not None:
        report_path = merge_root / f"original_{args.split}_split_report.json"
    else:
        report_path = Path(f"original_{args.split}_split_report.json")
    report = {
        "mode": "metadata",
        "metadata": str(metadata_path),
        "source_root": str(source_root),
        "split": args.split,
        "split_by": metadata.get("split_by"),
        "split_seed": metadata.get("split_seed"),
        "scene_count": len(records),
        "scenes": records,
    }
    write_report(report_path, report, args.overwrite)


if __name__ == "__main__":
    main()


"""
只查看原始测试集分配：
python tools/infer/prepare_original_split_las.py \
  --source-root /24085403037/24085403037/shixi/dataset/0617-4Name \
  --split val \
  --report /24085403037/24085403037/shixi/dataset/original_val_mapping.json \
  --overwrite

恢复并合并原始测试集 LAS：
python tools/infer/prepare_original_split_las.py \
  --source-root /24085403037/24085403037/shixi/dataset/0617-4Name \
  --split val \
  --merge-output-dir /24085403037/24085403037/shixi/dataset/0617-4Name_original_val_merged \
  --overwrite
  
不进行数据集划分
python tools/infer/prepare_original_split_las.py \
  --direct-merge \
  --source-root /24085403037/24085403037/shixi/dataset/0617-4Name \
  --merge-output-dir /24085403037/24085403037/shixi/dataset/0617-4Name_merged \
  --overwrite
"""
