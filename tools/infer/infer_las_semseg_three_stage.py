"""输电线路 LAS 三阶段推理总控脚本。

三阶段定义：
1. Stage-1：整图快速粗分割，只承担背景、杆塔、导线等大结构定位；
2. Stage-2：根据 Stage-1 杆塔结果生成 ROI，在杆塔局部精细重推绝缘子；
3. Stage-3：基于二阶段语义结果提取绝缘子实例和横担信息。

这个脚本是流水线编排脚本，不重新实现模型和后处理细节：
- Stage-1/Stage-2 调用 tools/infer/infer_las_semseg_two_stage.py；
- Stage-3a 调用 tools/infer/extract_insulator_points.py；
- Stage-3b 调用 tools/infer/extract_crossarm_points.py。
"""

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
TOOL_DIR = SCRIPT_PATH.parent
ROOT_DIR = next(
    (path for path in SCRIPT_PATH.parents if (path / "pointcept").is_dir()),
    SCRIPT_PATH.parents[2],
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run three-stage transmission-line LAS inference."
    )
    parser.add_argument("--input", required=True, help="输入原始 LAS/LAZ。")
    parser.add_argument(
        "--output",
        required=True,
        help="最终输出 LAS/LAZ。默认是经过绝缘子提取阶段修正后的语义分割 LAS。",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="中间文件目录。默认：<output stem>_three_stage_work。",
    )
    parser.add_argument(
        "--summary-report",
        default=None,
        help="三阶段总控报告 JSON。默认：<output stem>_three_stage_summary.json。",
    )
    parser.add_argument("--config-file", default=None, help="Pointcept 配置文件。")
    parser.add_argument("--weight", default=None, help="模型权重。")
    parser.add_argument(
        "--device",
        default=None,
        help="推理设备，例如 cuda、cuda:0 或 cpu。不填则使用二阶段脚本默认值。",
    )
    parser.add_argument(
        "--las-backend",
        choices=("auto", "laspy", "fallback"),
        default="auto",
        help="LAS 读写后端。",
    )
    parser.add_argument("--disable-flash", action="store_true", help="关闭 flash attention。")
    parser.add_argument("--no-colorize", action="store_true", help="不重写 LAS RGB。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出。")

    # Stage-1 快速粗分割参数。
    parser.add_argument("--stage1-tile-size", type=float, default=40.0)
    parser.add_argument("--stage1-tile-stride", type=float, default=40.0)
    parser.add_argument("--stage1-context-margin", type=float, default=10.0)
    parser.add_argument("--stage1-pre-voxel-size", type=float, default=0.08)
    parser.add_argument("--stage1-fragment-batch-size", type=int, default=1)
    parser.add_argument("--stage1-point-max", type=int, default=100000)

    # Stage-2 杆塔 ROI 精细绝缘子分割参数。
    parser.add_argument("--roi-mode", choices=("tower", "connection"), default="tower")
    parser.add_argument("--stage2-pre-voxel-size", type=float, default=0.0)
    parser.add_argument("--stage2-point-max", type=int, default=200000)
    parser.add_argument("--stage2-fragment-batch-size", type=int, default=1)
    parser.add_argument("--tower-roi-xy-margin", type=float, default=10.0)
    parser.add_argument("--tower-roi-z-margin", type=float, default=8.0)
    parser.add_argument("--insulator-score-threshold", type=float, default=0.25)
    parser.add_argument(
        "--recover-insulator-by-structure",
        action="store_true",
        help="二阶段融合时启用结构先验恢复绝缘子。",
    )
    parser.add_argument(
        "--save-stage1-las",
        action="store_true",
        help="保存 Stage-1 粗结构 LAS 到工作目录。",
    )

    # Stage-3a 绝缘子提取参数。
    parser.add_argument(
        "--insulator-cluster-method",
        choices=(
            "tower_shape",
            "attachment",
            "seed_grow",
            "tower_layer_side",
            "voxel_cc",
            "dbscan",
        ),
        default="tower_layer_side",
        help="第三阶段绝缘子实例提取方式。",
    )
    parser.add_argument(
        "--insulator-visual-las-output",
        default=None,
        help="绝缘子关键点可视化 LAS。默认写到工作目录。",
    )
    parser.add_argument(
        "--insulator-tower-las-output-dir",
        default=None,
        help="按杆塔输出绝缘子可视化 LAS 的目录。默认写到工作目录。",
    )

    # Stage-3b 横担提取参数。
    parser.add_argument(
        "--crossarm-layer-source",
        choices=("tower_width", "line_layer", "insulator"),
        default="line_layer",
        help="横担层来源。默认 line_layer：从导线层出发搜索横担宽度峰值。",
    )
    parser.add_argument(
        "--crossarm-layer-z-gap",
        type=float,
        default=2.0,
        help="横担/导线层按高度合并的阈值。",
    )
    parser.add_argument(
        "--line-layer-merge-adjacent-count",
        type=int,
        default=2,
        help="line_layer 模式下，把相邻几个原始导线高度簇合并为一个物理横担层。",
    )
    parser.add_argument(
        "--crossarm-tower-las-output-dir",
        default=None,
        help="按杆塔输出横担可视化 LAS 的目录。默认写到工作目录。",
    )

    # 额外参数用于现场调参，不需要改脚本。
    parser.add_argument(
        "--two-stage-extra",
        default="",
        help='追加给 infer_las_semseg_two_stage.py 的参数字符串，例如 "--max-rois 5"。',
    )
    parser.add_argument(
        "--insulator-extra",
        default="",
        help='追加给 extract_insulator_points.py 的参数字符串，例如 "--merge-same-side-layer-insulators"。',
    )
    parser.add_argument(
        "--crossarm-extra",
        default="",
        help='追加给 extract_crossarm_points.py 的参数字符串，例如 "--crossarm-min-y-span 2.0"。',
    )
    return parser.parse_args()


def ensure_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def split_extra_args(extra):
    if not extra:
        return []
    return shlex.split(extra)


def format_command(command):
    return " ".join(shlex.quote(str(item)) for item in command)


def run_command(name, command):
    start = time.perf_counter()
    print(f"\n===== {name} =====", flush=True)
    print(format_command(command), flush=True)
    result = subprocess.run(command, cwd=str(ROOT_DIR))
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    print(f"{name} finished in {elapsed:.2f}s", flush=True)
    return elapsed


def build_two_stage_command(args, stage12_las, stage12_report, stage1_las):
    command = [
        sys.executable,
        str(TOOL_DIR / "infer_las_semseg_two_stage.py"),
        "--input",
        str(args.input),
        "--output",
        str(stage12_las),
        "--report",
        str(stage12_report),
        "--las-backend",
        args.las_backend,
        "--roi-mode",
        args.roi_mode,
        "--stage1-tile-size",
        str(args.stage1_tile_size),
        "--stage1-tile-stride",
        str(args.stage1_tile_stride),
        "--stage1-context-margin",
        str(args.stage1_context_margin),
        "--stage1-pre-voxel-size",
        str(args.stage1_pre_voxel_size),
        "--stage1-fragment-batch-size",
        str(args.stage1_fragment_batch_size),
        "--stage1-point-max",
        str(args.stage1_point_max),
        "--stage2-pre-voxel-size",
        str(args.stage2_pre_voxel_size),
        "--stage2-point-max",
        str(args.stage2_point_max),
        "--stage2-fragment-batch-size",
        str(args.stage2_fragment_batch_size),
        "--tower-roi-xy-margin",
        str(args.tower_roi_xy_margin),
        "--tower-roi-z-margin",
        str(args.tower_roi_z_margin),
        "--insulator-score-threshold",
        str(args.insulator_score_threshold),
    ]
    if args.config_file:
        command.extend(["--config-file", str(args.config_file)])
    if args.weight:
        command.extend(["--weight", str(args.weight)])
    if args.device:
        command.extend(["--device", str(args.device)])
    if args.disable_flash:
        command.append("--disable-flash")
    if args.no_colorize:
        command.append("--no-colorize")
    if args.overwrite:
        command.append("--overwrite")
    if args.recover_insulator_by_structure:
        command.append("--recover-insulator-by-structure")
    if args.save_stage1_las:
        command.extend(["--save-stage1-las", str(stage1_las)])
    command.extend(split_extra_args(args.two_stage_extra))
    return command


def build_insulator_command(
    args,
    stage12_las,
    output_las,
    insulator_report,
    insulator_visual_las,
    insulator_tower_las_dir,
):
    command = [
        sys.executable,
        str(TOOL_DIR / "extract_insulator_points.py"),
        "--input",
        str(stage12_las),
        "--render-base-las",
        str(args.input),
        "--output",
        str(insulator_report),
        "--visual-las-output",
        str(insulator_visual_las),
        "--tower-las-output-dir",
        str(insulator_tower_las_dir),
        "--corrected-las-output",
        str(output_las),
        "--cluster-method",
        args.insulator_cluster_method,
    ]
    if args.overwrite:
        command.append("--overwrite")
    command.extend(split_extra_args(args.insulator_extra))
    return command


def build_crossarm_command(
    args,
    corrected_las,
    crossarm_report,
    crossarm_tower_las_dir,
):
    command = [
        sys.executable,
        str(TOOL_DIR / "extract_crossarm_points.py"),
        "--input",
        str(corrected_las),
        "--render-base-las",
        str(args.input),
        "--output",
        str(crossarm_report),
        "--tower-las-output-dir",
        str(crossarm_tower_las_dir),
        "--crossarm-layer-source",
        args.crossarm_layer_source,
        "--crossarm-layer-z-gap",
        str(args.crossarm_layer_z_gap),
        "--line-layer-merge-adjacent-count",
        str(args.line_layer_merge_adjacent_count),
    ]
    if args.overwrite:
        command.append("--overwrite")
    command.extend(split_extra_args(args.crossarm_extra))
    return command


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = ensure_path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists, use --overwrite: {output_path}")

    work_dir = (
        Path(args.work_dir)
        if args.work_dir
        else output_path.with_name(output_path.stem + "_three_stage_work")
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    stage12_las = work_dir / f"{input_path.stem}_stage12_semantic.las"
    stage12_report = work_dir / f"{input_path.stem}_stage12_report.json"
    stage1_las = work_dir / f"{input_path.stem}_stage1_coarse.las"
    insulator_report = work_dir / f"{input_path.stem}_insulators.json"
    insulator_visual_las = (
        Path(args.insulator_visual_las_output)
        if args.insulator_visual_las_output
        else work_dir / f"{input_path.stem}_insulator_markers.las"
    )
    insulator_tower_las_dir = (
        Path(args.insulator_tower_las_output_dir)
        if args.insulator_tower_las_output_dir
        else work_dir / "insulator_tower_las"
    )
    crossarm_report = work_dir / f"{input_path.stem}_crossarms.json"
    crossarm_tower_las_dir = (
        Path(args.crossarm_tower_las_output_dir)
        if args.crossarm_tower_las_output_dir
        else work_dir / "crossarm_tower_las"
    )
    summary_report = (
        Path(args.summary_report)
        if args.summary_report
        else output_path.with_name(output_path.stem + "_three_stage_summary.json")
    )
    summary_report.parent.mkdir(parents=True, exist_ok=True)

    all_start = time.perf_counter()
    two_stage_command = build_two_stage_command(args, stage12_las, stage12_report, stage1_las)
    stage12_time = run_command("Stage-1/2 coarse-to-fine inference", two_stage_command)

    insulator_command = build_insulator_command(
        args,
        stage12_las,
        output_path,
        insulator_report,
        insulator_visual_las,
        insulator_tower_las_dir,
    )
    insulator_time = run_command("Stage-3a insulator extraction", insulator_command)

    crossarm_command = build_crossarm_command(
        args,
        output_path,
        crossarm_report,
        crossarm_tower_las_dir,
    )
    crossarm_time = run_command("Stage-3b crossarm extraction", crossarm_command)
    total_time = time.perf_counter() - all_start

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "work_dir": str(work_dir),
        "stage1_coarse_las": str(stage1_las) if args.save_stage1_las else None,
        "stage12_semantic_las": str(stage12_las),
        "stage12_report": str(stage12_report),
        "insulator_report": str(insulator_report),
        "insulator_visual_las": str(insulator_visual_las),
        "insulator_tower_las_dir": str(insulator_tower_las_dir),
        "crossarm_report": str(crossarm_report),
        "crossarm_tower_las_dir": str(crossarm_tower_las_dir),
        "commands": {
            "stage12": two_stage_command,
            "insulator": insulator_command,
            "crossarm": crossarm_command,
        },
        "timing_sec": {
            "stage12": round(stage12_time, 3),
            "insulator": round(insulator_time, 3),
            "crossarm": round(crossarm_time, 3),
            "total": round(total_time, 3),
        },
    }
    with summary_report.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n三阶段推理完成", flush=True)
    print(f"最终语义 LAS: {output_path}", flush=True)
    print(f"绝缘子 JSON: {insulator_report}", flush=True)
    print(f"横担 JSON: {crossarm_report}", flush=True)
    print(f"横担可视化目录: {crossarm_tower_las_dir}", flush=True)
    print(f"总控报告: {summary_report}", flush=True)
    print(f"中间目录: {work_dir}", flush=True)
    print(f"总耗时: {total_time:.2f}s", flush=True)


if __name__ == "__main__":
    main()

""" 


python tools/infer/infer_las_semseg_three_stage.py \
  --input /24085403037/24085403037/shixi/dataset/0617-4Name/110v12/110v12_merged_4cls.las \
  --output /24085403037/24085403037/shixi/dataset/6_23_demo/test/110v12_merged_4cls_three_output.las \
  --overwrite
"""
