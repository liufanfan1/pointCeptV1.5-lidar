"""Profile Pointcept model parameter count and storage cost.

This script builds a Pointcept model from a config file and reports the model
space complexity: parameter count, buffer count, and estimated storage size.
It does not run inference.
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch

SCRIPT_PATH = Path(__file__).resolve()
ROOT_DIR = next(
    (path for path in SCRIPT_PATH.parents if (path / "pointcept").is_dir()),
    SCRIPT_PATH.parents[2],
)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pointcept.models import build_model
from pointcept.utils.config import Config


DEFAULT_CONFIG = (
    "exp/transmission_line/ptv3-4cls-ins-oversample_v2/test_model_best/config.py"
)
DEFAULT_WEIGHT = (
    "exp/transmission_line/ptv3-4cls-ins-oversample_v2/model/model_best.pth"
)


def format_number(value):
    return f"{int(value):,}"


def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def clean_state_dict(state_dict):
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    return cleaned


def load_weight_if_needed(model, weight_path, strict):
    if weight_path is None:
        return None
    weight_path = resolve_path(weight_path)
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = clean_state_dict(state_dict)
    load_info = model.load_state_dict(state_dict, strict=strict)
    return weight_path, load_info


def tensor_bytes(tensors, bytes_per_value):
    return sum(t.numel() for t in tensors) * bytes_per_value


def collect_module_param_counts(model):
    rows = []
    for name, module in model.named_modules():
        if name == "":
            continue
        own_params = sum(p.numel() for p in module.parameters(recurse=False))
        if own_params > 0:
            rows.append((name, module.__class__.__name__, own_params))
    rows.sort(key=lambda item: item[2], reverse=True)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count Pointcept model parameters and estimated storage size."
    )
    parser.add_argument(
        "--config-file",
        default=DEFAULT_CONFIG,
        help="Pointcept config file used to build the model.",
    )
    parser.add_argument(
        "--weight",
        default=DEFAULT_WEIGHT,
        help="Optional checkpoint path. Use '' to skip loading weights.",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Load checkpoint with strict=False.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Print top-k modules by direct parameter count. Use 0 to disable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_file = resolve_path(args.config_file)
    weight = args.weight if args.weight else None

    cfg = Config.fromfile(str(config_file))
    model = build_model(cfg.model)
    model.eval()

    load_result = load_weight_if_needed(model, weight, strict=not args.non_strict)

    params = list(model.parameters())
    buffers = list(model.buffers())

    total_params = sum(p.numel() for p in params)
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    frozen_params = total_params - trainable_params
    total_buffers = sum(b.numel() for b in buffers)

    param_fp32_bytes = tensor_bytes(params, 4)
    param_fp16_bytes = tensor_bytes(params, 2)
    param_int8_bytes = tensor_bytes(params, 1)
    buffer_fp32_bytes = tensor_bytes(buffers, 4)

    print("=" * 80)
    print("Pointcept Model Size Profile")
    print("=" * 80)
    print(f"repo             : {ROOT_DIR}")
    print(f"config           : {config_file}")
    if load_result is not None:
        weight_path, load_info = load_result
        print(f"weight           : {weight_path}")
        print(f"checkpoint size  : {format_bytes(weight_path.stat().st_size)}")
        if args.non_strict:
            print(f"missing keys     : {len(load_info.missing_keys)}")
            print(f"unexpected keys  : {len(load_info.unexpected_keys)}")
    else:
        print("weight           : not loaded")

    print("-" * 80)
    print(f"total params     : {format_number(total_params)}")
    print(f"trainable params : {format_number(trainable_params)}")
    print(f"frozen params    : {format_number(frozen_params)}")
    print(f"buffers          : {format_number(total_buffers)}")
    print("-" * 80)
    print(f"params FP32      : {format_bytes(param_fp32_bytes)}")
    print(f"params FP16      : {format_bytes(param_fp16_bytes)}")
    print(f"params INT8      : {format_bytes(param_int8_bytes)}")
    print(f"buffers FP32 est.: {format_bytes(buffer_fp32_bytes)}")
    print(
        f"params+buffers FP32 est.: "
        f"{format_bytes(param_fp32_bytes + buffer_fp32_bytes)}"
    )

    if args.topk > 0:
        print("-" * 80)
        print(f"Top {args.topk} modules by own parameter count")
        for idx, (name, module_type, count) in enumerate(
            collect_module_param_counts(model)[: args.topk],
            start=1,
        ):
            print(
                f"{idx:>2}. {name:<60} "
                f"{module_type:<20} {format_number(count)}"
            )

    print("=" * 80)


if __name__ == "__main__":
    main()
