#!/usr/bin/env python
"""顺序训练输电线路两阶段模型：先 Stage-1，再 Stage-2。

用途：
    自动调用 tools/train.py 先训练 Stage-1 粗分模型，成功后继续训练
    Stage-2 ROI 精分模型。适合希望一次启动完整两阶段训练流程的情况。
输入：
    Stage-1 config/save_path 和 Stage-2 config/save_path，可通过参数覆盖。
输出：
    两个独立实验目录，各自包含 model_best.pth、model_last.pth 和日志。
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/opt/conda/envs/pointcept/bin/python"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train transmission-line Stage1 coarse segmentation first, then "
            "train Stage2 tower ROI fine segmentation after Stage1 succeeds."
        )
    )
    parser.add_argument(
        "--python",
        default=DEFAULT_PYTHON,
        help="Python executable used to launch tools/train.py.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs per machine for each training stage.",
    )
    parser.add_argument(
        "--stage1-config",
        type=Path,
        default=Path("configs/transmission_line/semseg-pt-v3m1-stage1-4cls.py"),
        help="Stage1 training config.",
    )
    parser.add_argument(
        "--stage1-save-path",
        type=Path,
        default=Path("exp/transmission/stage1_4cls_balance_w8_clean_seq"),
        help="Stage1 experiment output directory.",
    )
    parser.add_argument(
        "--stage2-config",
        type=Path,
        default=Path("configs/transmission_line/semseg-pt-v3m1-stage2-tower.py"),
        help="Stage2 training config.",
    )
    parser.add_argument(
        "--stage2-save-path",
        type=Path,
        default=Path("exp/transmission/stage2_tower_ins_centered_seq"),
        help="Stage2 experiment output directory.",
    )
    parser.add_argument(
        "--stage1-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra Pointcept config override for Stage1. Can be repeated, e.g. "
            "--stage1-option epoch=100 --stage1-option data.train.data_root=..."
        ),
    )
    parser.add_argument(
        "--stage2-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Pointcept config override for Stage2. Can be repeated.",
    )
    parser.add_argument(
        "--resume-stage1",
        action="store_true",
        help="Resume Stage1 training from its save path if Pointcept checkpoint exists.",
    )
    parser.add_argument(
        "--resume-stage2",
        action="store_true",
        help="Resume Stage2 training from its save path if Pointcept checkpoint exists.",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip Stage1 training and run Stage2 directly.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete stage save directories before training. Do not use this if "
            "you want to keep existing checkpoints."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running them.",
    )
    return parser.parse_args()


def format_seconds(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{seconds:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes)}m{seconds:.1f}s"


def checked_path(path, description):
    path = PROJECT_ROOT / path if not path.is_absolute() else path
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def prepare_save_path(path, overwrite):
    path = PROJECT_ROOT / path if not path.is_absolute() else path
    if path.exists() and overwrite:
        print(f"[warn] remove existing output: {path}", flush=True)
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_command(args, stage, config, save_path, extra_options, resume):
    options = [
        f"save_path={save_path}",
        f"resume={str(bool(resume)).lower()}",
    ]
    options.extend(extra_options)
    return [
        args.python,
        "tools/train.py",
        "--config-file",
        str(config),
        "--num-gpus",
        str(args.num_gpus),
        "--options",
        *options,
    ]


def run_command(stage, command, log_path, dry_run):
    command_text = " ".join(str(part) for part in command)
    print(f"\n[{stage}] command:\n{command_text}\n", flush=True)
    if dry_run:
        return 0

    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(command_text + "\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
        elapsed = time.perf_counter() - start
        message = (
            f"\n[{stage}] return_code={return_code} "
            f"elapsed={format_seconds(elapsed)}\n"
        )
        print(message, end="", flush=True)
        log_file.write(message)
    return return_code


def ensure_model_best(stage, save_path):
    model_best = save_path / "model" / "model_best.pth"
    if not model_best.exists():
        raise FileNotFoundError(
            f"{stage} finished but model_best.pth was not found: {model_best}"
        )
    print(f"[{stage}] best checkpoint: {model_best}", flush=True)


def main():
    args = parse_args()
    stage1_config = checked_path(args.stage1_config, "Stage1 config")
    stage2_config = checked_path(args.stage2_config, "Stage2 config")
    stage1_save_path = prepare_save_path(args.stage1_save_path, args.overwrite)
    stage2_save_path = prepare_save_path(args.stage2_save_path, args.overwrite)

    total_start = time.perf_counter()
    if not args.skip_stage1:
        stage1_command = build_command(
            args,
            "stage1",
            stage1_config,
            stage1_save_path,
            args.stage1_option,
            args.resume_stage1,
        )
        code = run_command(
            "stage1",
            stage1_command,
            stage1_save_path / "train_stage1_wrapper.log",
            args.dry_run,
        )
        if code != 0:
            raise SystemExit(f"Stage1 training failed with return code {code}.")
        if not args.dry_run:
            ensure_model_best("stage1", stage1_save_path)
    else:
        print("[stage1] skipped", flush=True)

    stage2_command = build_command(
        args,
        "stage2",
        stage2_config,
        stage2_save_path,
        args.stage2_option,
        args.resume_stage2,
    )
    code = run_command(
        "stage2",
        stage2_command,
        stage2_save_path / "train_stage2_wrapper.log",
        args.dry_run,
    )
    if code != 0:
        raise SystemExit(f"Stage2 training failed with return code {code}.")
    if not args.dry_run:
        ensure_model_best("stage2", stage2_save_path)

    elapsed = time.perf_counter() - total_start
    print(f"\n[done] two-stage training elapsed={format_seconds(elapsed)}", flush=True)
    print(f"[done] stage1_save_path={stage1_save_path}", flush=True)
    print(f"[done] stage2_save_path={stage2_save_path}", flush=True)


if __name__ == "__main__":
    main()


""" 
/opt/conda/envs/pointcept/bin/python tools/train_transmission_line_two_stage.py \
  --num-gpus 1 \
  --stage1-save-path exp/transmission/stage1_4cls_balance_w8_clean_seq_v2 \
  --stage2-save-path exp/transmission/stage2_tower_ins_centered_seq_v2
  """
