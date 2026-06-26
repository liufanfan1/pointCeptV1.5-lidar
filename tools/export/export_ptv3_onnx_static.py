"""Export the PTv3 4-class transmission-line model to a static ONNX file.

This is an experimental ONNX export helper. It exports only the model forward
for one fixed-size point fragment:

    coord, grid_coord, feat, offset -> seg_logits

LAS reading, tiling, GridSample/SphereCrop, merge, and LAS writing stay outside
ONNX. That boundary keeps the first ONNX experiment small and debuggable.
"""

import argparse
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn


DEFAULT_CONFIG = "exp/transmission_line/ptv3-4cls-ins-oversample_v2/config.py"
DEFAULT_WEIGHT = "exp/transmission_line/ptv3-4cls-ins-oversample_v2/model/model_best.pth"
DEFAULT_OUTPUT = "deploy_onnx/ptv3_4cls_static.onnx"


class PTv3ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, coord, grid_coord, feat, offset):
        input_dict = {
            "coord": coord,
            "grid_coord": grid_coord,
            "feat": feat,
            "offset": offset,
        }
        return self.model(input_dict)["seg_logits"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export PTv3 transmission-line checkpoint to static ONNX."
    )
    parser.add_argument("--config-file", default=DEFAULT_CONFIG)
    parser.add_argument("--weight", default=DEFAULT_WEIGHT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sample-npz",
        default=None,
        help=(
            "Optional NPZ with coord, grid_coord, feat, offset arrays. "
            "If omitted, a synthetic static fragment is generated."
        ),
    )
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--grid-size", type=float, default=0.05)
    parser.add_argument("--cube-size", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--disable-flash",
        action="store_true",
        default=True,
        help="Disable flash-attn path. Enabled by default for deployment export.",
    )
    parser.add_argument(
        "--enable-flash",
        action="store_false",
        dest="disable_flash",
        help="Keep enable_flash from config. Usually not recommended for ONNX export.",
    )
    parser.add_argument(
        "--save-sample",
        default=None,
        help="Optional path to save the actual ONNX input sample as NPZ.",
    )
    parser.add_argument(
        "--check-onnxruntime",
        action="store_true",
        help="Run ONNX Runtime once and compare argmax with PyTorch if export succeeds.",
    )
    parser.add_argument(
        "--serialization-order",
        default="z",
        help=(
            "Comma-separated PTv3 serialization order override for ONNX export. "
            "Default 'z' avoids Hilbert dtype-reinterpret view ops. Use 'config' "
            "to keep the original config order."
        ),
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_enable_flash(cfg, enabled):
    try:
        cfg.model.backbone.enable_flash = bool(enabled)
    except Exception:
        pass


def set_serialization_order(cfg, order_text):
    if not order_text or order_text.lower() == "config":
        return
    order = [item.strip() for item in order_text.split(",") if item.strip()]
    if not order:
        raise ValueError("--serialization-order cannot be empty")
    allowed = {"z", "z-trans", "hilbert", "hilbert-trans"}
    unknown = [item for item in order if item not in allowed]
    if unknown:
        raise ValueError(f"Unsupported serialization order: {unknown}")
    cfg.model.backbone.order = order
    cfg.model.backbone.shuffle_orders = False


def load_model(cfg, weight_path, device):
    from pointcept.models import build_model

    model = build_model(cfg.model)
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned_state_dict[key] = value
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def make_synthetic_sample(num_points, grid_size, cube_size, device):
    coord_np = np.random.rand(num_points, 3).astype(np.float32) * float(cube_size)
    coord_np[:, 2] *= 0.4
    grid_coord_np = np.floor((coord_np - coord_np.min(axis=0)) / grid_size).astype(
        np.int64
    )
    color_np = np.random.rand(num_points, 3).astype(np.float32) * 2.0 - 1.0
    feat_np = np.concatenate([coord_np, color_np], axis=1).astype(np.float32)
    offset_np = np.array([num_points], dtype=np.int64)
    return numpy_to_torch(coord_np, grid_coord_np, feat_np, offset_np, device)


def load_npz_sample(path, device):
    data = np.load(path)
    required = ("coord", "grid_coord", "feat", "offset")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"sample NPZ missing keys: {missing}")
    return numpy_to_torch(
        data["coord"],
        data["grid_coord"],
        data["feat"],
        data["offset"],
        device,
    )


def numpy_to_torch(coord_np, grid_coord_np, feat_np, offset_np, device):
    coord = torch.as_tensor(coord_np, dtype=torch.float32, device=device)
    grid_coord = torch.as_tensor(grid_coord_np, dtype=torch.long, device=device)
    feat = torch.as_tensor(feat_np, dtype=torch.float32, device=device)
    offset = torch.as_tensor(offset_np, dtype=torch.long, device=device)
    return coord, grid_coord, feat, offset


def save_sample_npz(path, coord, grid_coord, feat, offset, torch_logits=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(
        coord=coord.detach().cpu().numpy(),
        grid_coord=grid_coord.detach().cpu().numpy(),
        feat=feat.detach().cpu().numpy(),
        offset=offset.detach().cpu().numpy(),
    )
    if torch_logits is not None:
        data["torch_logits"] = torch_logits.detach().cpu().numpy()
    np.savez(path, **data)
    print(f"Saved sample input: {path}")


def export_onnx(wrapper, sample, output_path, opset):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        sample,
        str(output_path),
        input_names=["coord", "grid_coord", "feat", "offset"],
        output_names=["seg_logits"],
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"Exported ONNX: {output_path}")


def check_onnx(path):
    import onnx

    model = onnx.load(path)
    onnx.checker.check_model(model)
    print("ONNX checker: ok")


def check_onnxruntime(path, sample, torch_logits):
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers = [item for item in providers if item in available]
    print(f"ONNX Runtime providers: {providers}")
    sess = ort.InferenceSession(str(path), providers=providers)
    inputs = {
        "coord": sample[0].detach().cpu().numpy(),
        "grid_coord": sample[1].detach().cpu().numpy(),
        "feat": sample[2].detach().cpu().numpy(),
        "offset": sample[3].detach().cpu().numpy(),
    }
    ort_logits = sess.run(["seg_logits"], inputs)[0]
    torch_np = torch_logits.detach().cpu().numpy()
    abs_diff = np.abs(torch_np - ort_logits)
    torch_pred = torch_np.argmax(axis=1)
    ort_pred = ort_logits.argmax(axis=1)
    same = float((torch_pred == ort_pred).mean())
    print(f"ONNX Runtime max_abs_diff: {abs_diff.max():.6f}")
    print(f"ONNX Runtime mean_abs_diff: {abs_diff.mean():.6f}")
    print(f"ONNX Runtime argmax agreement: {same * 100:.4f}%")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    from pointcept.utils.config import Config

    cfg = Config.fromfile(args.config_file)
    if args.disable_flash:
        set_enable_flash(cfg, False)
    set_serialization_order(cfg, args.serialization_order)

    print(f"Device: {device}")
    print(f"Config: {args.config_file}")
    print(f"Weight: {args.weight}")
    print(f"enable_flash: {cfg.model.backbone.get('enable_flash', None)}")
    print(f"serialization_order: {cfg.model.backbone.get('order', None)}")
    print(f"shuffle_orders: {cfg.model.backbone.get('shuffle_orders', None)}")

    model = load_model(cfg, args.weight, device)
    wrapper = PTv3ONNXWrapper(model).to(device).eval()

    if args.sample_npz:
        sample = load_npz_sample(args.sample_npz, device)
        print(f"Loaded sample NPZ: {args.sample_npz}")
    else:
        sample = make_synthetic_sample(
            args.num_points, args.grid_size, args.cube_size, device
        )
        print(f"Generated synthetic sample: N={args.num_points}")

    with torch.no_grad():
        torch_logits = wrapper(*sample)
    print(f"PyTorch forward ok, seg_logits shape: {tuple(torch_logits.shape)}")

    if args.save_sample:
        save_sample_npz(args.save_sample, *sample, torch_logits=torch_logits)

    try:
        export_onnx(wrapper, sample, args.output, args.opset)
        check_onnx(args.output)
    except Exception as exc:
        print("\nONNX export failed.")
        print("This usually means one of the PTv3 operators is not ONNX-exportable,")
        print("commonly spconv, torch_scatter, dynamic indexing, or serialization logic.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")
        raise

    if args.check_onnxruntime:
        check_onnxruntime(args.output, sample, torch_logits)


if __name__ == "__main__":
    main()
