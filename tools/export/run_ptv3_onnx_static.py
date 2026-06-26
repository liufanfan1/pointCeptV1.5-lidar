"""Run a static PTv3 ONNX model on a saved NPZ sample.

This script intentionally does not import Pointcept or PyTorch. It is meant to
verify that an exported ONNX model can run in a lightweight deployment
environment such as native Windows with onnxruntime-gpu installed.

Expected NPZ keys:
    coord, grid_coord, feat, offset

Optional NPZ key:
    torch_logits
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(description="Run PTv3 static ONNX inference.")
    parser.add_argument("--onnx", required=True, help="Path to exported ONNX model.")
    parser.add_argument("--sample", required=True, help="Path to input sample NPZ.")
    parser.add_argument(
        "--providers",
        default="CUDAExecutionProvider,CPUExecutionProvider",
        help="Comma-separated ONNX Runtime providers in priority order.",
    )
    parser.add_argument(
        "--save-pred",
        default=None,
        help="Optional .npy path to save argmax predictions.",
    )
    parser.add_argument(
        "--save-logits",
        default=None,
        help="Optional .npy path to save raw seg_logits.",
    )
    return parser.parse_args()


def load_sample(path):
    data = np.load(path)
    required = ("coord", "grid_coord", "feat", "offset")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"sample NPZ missing keys: {missing}")
    inputs = {
        "coord": data["coord"].astype(np.float32, copy=False),
        "grid_coord": data["grid_coord"].astype(np.int64, copy=False),
        "feat": data["feat"].astype(np.float32, copy=False),
        "offset": data["offset"].astype(np.int64, copy=False),
    }
    torch_logits = data["torch_logits"] if "torch_logits" in data else None
    return inputs, torch_logits


def choose_providers(provider_text):
    requested = [item.strip() for item in provider_text.split(",") if item.strip()]
    available = ort.get_available_providers()
    providers = [item for item in requested if item in available]
    if not providers:
        raise RuntimeError(
            f"None of requested providers are available. requested={requested}, "
            f"available={available}"
        )
    return providers


def main():
    args = parse_args()
    onnx_path = Path(args.onnx)
    sample_path = Path(args.sample)
    inputs, torch_logits = load_sample(sample_path)
    providers = choose_providers(args.providers)

    print(f"ONNX: {onnx_path}")
    print(f"Sample: {sample_path}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Using providers: {providers}")

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    logits = session.run(["seg_logits"], inputs)[0]
    pred = logits.argmax(axis=1).astype(np.uint8)
    counts = np.bincount(pred, minlength=4)

    print(f"seg_logits shape: {logits.shape}")
    print(
        "pred counts: "
        + ", ".join(f"{idx}={int(counts[idx])}" for idx in range(len(counts)))
    )

    if torch_logits is not None:
        abs_diff = np.abs(torch_logits.astype(np.float32) - logits.astype(np.float32))
        torch_pred = torch_logits.argmax(axis=1)
        agreement = float((torch_pred == pred).mean())
        print(f"max_abs_diff: {abs_diff.max():.6f}")
        print(f"mean_abs_diff: {abs_diff.mean():.6f}")
        print(f"argmax agreement: {agreement * 100:.4f}%")

    if args.save_pred:
        Path(args.save_pred).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_pred, pred)
        print(f"Saved pred: {args.save_pred}")

    if args.save_logits:
        Path(args.save_logits).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_logits, logits)
        print(f"Saved logits: {args.save_logits}")


if __name__ == "__main__":
    main()
