"""Run Pointcept semantic segmentation on LAS files and save predicted LAS.

The script reads one LAS file or a directory of LAS files, runs the configured
Pointcept model, and writes predictions into the LAS classification field.
When --keypoints-output is set, it immediately extracts insulator endpoints and
crossarm left/middle/right points from the in-memory predictions into one JSON.
It prefers laspy when installed and falls back to a small LAS 1.2 reader/writer
for common point formats 0-3.
"""

import argparse
import copy
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
ROOT_DIR = next(
    (path for path in SCRIPT_PATH.parents if (path / "pointcept").is_dir()),
    SCRIPT_PATH.parents[2],
)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F

from pointcept.datasets.transform import Compose, TRANSFORMS
from pointcept.datasets.utils import collate_fn
from pointcept.models import build_model
from pointcept.utils.config import Config

# 配置文件
DEFAULT_CONFIG = (
    "exp/transmission_line/ptv3-4cls-ins-oversample_v2/config.py"
)
# 权重
DEFAULT_WEIGHT = (
    "exp/transmission_line/ptv3-4cls-ins-oversample_v2/model/model_best.pth"
)

# RGB colors used for direct visualization in CloudCompare and similar tools.
CLASS_COLOR_8BIT = np.array(
    [
        [145, 145, 145],  # 0 ground
        [230, 65, 55],    # 1 tower
        [45, 125, 255],   # 2 line
        [255, 210, 40],   # 3 insulator
    ],
    dtype=np.uint8,
)


def format_seconds(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{sec:05.2f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{sec:05.2f}s"


def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f}{unit}"
        value /= 1024.0


def format_percent(value):
    if value is None:
        return "N/A"
    return f"{float(value):.1f}%"


def is_cuda_device(device):
    return isinstance(device, torch.device) and device.type == "cuda"


def sync_cuda(device):
    if is_cuda_device(device):
        torch.cuda.synchronize(device)


def reset_cuda_peak_memory(device):
    if is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats(device)


def cuda_peak_memory_summary(device):
    if not is_cuda_device(device):
        return "cuda_peak_allocated=N/A, cuda_peak_reserved=N/A"
    return (
        "cuda_peak_allocated={}, cuda_peak_reserved={}".format(
            format_bytes(torch.cuda.max_memory_allocated(device)),
            format_bytes(torch.cuda.max_memory_reserved(device)),
        )
    )


def cuda_current_memory_summary(device):
    if not is_cuda_device(device):
        return "torch_allocated=N/A, torch_reserved=N/A"
    return (
        "torch_allocated={}, torch_reserved={}".format(
            format_bytes(torch.cuda.memory_allocated(device)),
            format_bytes(torch.cuda.memory_reserved(device)),
        )
    )


def cuda_device_index(device):
    if not is_cuda_device(device):
        return None
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def try_import_psutil():
    try:
        import psutil  # noqa: WPS433
    except Exception:
        return None
    return psutil


def try_init_nvml(device):
    if not is_cuda_device(device):
        return None, None
    try:
        import pynvml  # noqa: WPS433

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_device_index(device))
        return pynvml, handle
    except Exception:
        return None, None


def query_gpu_with_nvidia_smi(device):
    index = cuda_device_index(device)
    if index is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    parts = [part.strip() for part in result.stdout.strip().split(",")]
    if len(parts) < 3:
        return None
    try:
        return {
            "gpu_util": float(parts[0]),
            "gpu_mem_used": int(float(parts[1]) * 1024 * 1024),
            "gpu_mem_total": int(float(parts[2]) * 1024 * 1024),
        }
    except ValueError:
        return None


class ResourceMonitor:
    """后台周期打印 CPU/GPU 利用率，方便定位 Windows/Linux 推理瓶颈。"""

    def __init__(self, device, interval):
        self.device = device
        self.interval = float(interval)
        self.enabled = self.interval > 0
        self.stop_event = threading.Event()
        self.thread = None
        self.start_time = None
        self.psutil = try_import_psutil()
        self.process = None
        self.nvml = None
        self.nvml_handle = None

    def __enter__(self):
        if not self.enabled:
            return self
        self.start_time = time.perf_counter()
        if self.psutil is not None:
            self.process = self.psutil.Process(os.getpid())
            # 预热 cpu_percent，否则第一次通常是 0。
            self.psutil.cpu_percent(interval=None)
            self.process.cpu_percent(interval=None)
        self.nvml, self.nvml_handle = try_init_nvml(self.device)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if not self.enabled:
            return False
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(self.interval, 1.0) + 1.0)
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
        return False

    def _run(self):
        self._print_sample("monitor")
        while not self.stop_event.wait(self.interval):
            self._print_sample("monitor")

    def _cpu_summary(self):
        if self.psutil is None:
            return "cpu_total=N/A, cpu_process=N/A, rss=N/A"
        cpu_total = self.psutil.cpu_percent(interval=None)
        cpu_process = self.process.cpu_percent(interval=None) if self.process else None
        rss = self.process.memory_info().rss if self.process else None
        return (
            f"cpu_total={format_percent(cpu_total)}, "
            f"cpu_process={format_percent(cpu_process)}, "
            f"rss={format_bytes(rss) if rss is not None else 'N/A'}"
        )

    def _gpu_summary(self):
        if not is_cuda_device(self.device):
            return "gpu_util=N/A, gpu_mem=N/A"
        if self.nvml is not None and self.nvml_handle is not None:
            try:
                util = self.nvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                mem = self.nvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                return (
                    f"gpu_util={format_percent(util.gpu)}, "
                    f"gpu_mem={format_bytes(mem.used)}/{format_bytes(mem.total)}, "
                    f"{cuda_current_memory_summary(self.device)}"
                )
            except Exception:
                pass
        smi = query_gpu_with_nvidia_smi(self.device)
        if smi is not None:
            return (
                f"gpu_util={format_percent(smi['gpu_util'])}, "
                f"gpu_mem={format_bytes(smi['gpu_mem_used'])}/"
                f"{format_bytes(smi['gpu_mem_total'])}, "
                f"{cuda_current_memory_summary(self.device)}"
            )
        return f"gpu_util=N/A, gpu_mem=N/A, {cuda_current_memory_summary(self.device)}"

    def _print_sample(self, prefix):
        elapsed = 0.0 if self.start_time is None else time.perf_counter() - self.start_time
        print(
            f"[{prefix} {format_seconds(elapsed)}] "
            f"{self._cpu_summary()}, {self._gpu_summary()}",
            flush=True,
        )


def new_infer_stats():
    return {
        "total_tiles": 0,
        "processed_tiles": 0,
        "skipped_tiles": 0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer transmission-line semantic labels for LAS files."
    )
    parser.add_argument("--input", required=True, help="Input .las file or directory.")
    parser.add_argument("--output", required=True, help="Output .las file or directory.")
    parser.add_argument("--config-file", default=DEFAULT_CONFIG, help="Pointcept config.")
    parser.add_argument("--weight", default=DEFAULT_WEIGHT, help="Checkpoint path.")
    parser.add_argument(
        "--las-backend",
        choices=("auto", "laspy", "fallback"),
        default="auto",
        help="LAS IO backend. auto uses the faster fallback for supported .las files.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--disable-flash",
        action="store_true",
        help="Set cfg.model.backbone.enable_flash=False. Useful on Windows without flash-attn.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help=(
            "Use FP16 model weights and floating-point inputs on CUDA. "
            "Index tensors remain integer and probability accumulation remains FP32."
        ),
    )
    parser.add_argument(
        "--fragment-batch-size",
        type=int,
        default=8,
        help="Number of cropped fragments sent to the model at once.",
    )
    parser.add_argument(
        "--point-max",
        type=int,
        default=None,
        help="Override test SphereCrop point_max from config.",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        default=None,
        help="Override test GridSample grid_size from config.",
    )
    parser.add_argument(
        "--tile-size",
        type=float,
        default=40.0,
        help="XY tile size in meters. Use <=0 to infer the whole LAS at once.",
    )
    parser.add_argument(
        "--tile-stride",
        type=float,
        default=40.0,
        help="XY tile stride in meters. Default matches the training eval tiles.",
    )
    parser.add_argument(
        "--merge-mode",
        choices=("plain", "halo", "overlap"),
        default="plain",
        help=(
            "Tile merge mode. plain is current non-overlap behavior; "
            "halo uses context around each core tile and writes only the core; "
            "overlap averages probabilities from overlapping tiles."
        ),
    )
    parser.add_argument(
        "--context-margin",
        type=float,
        default=0.0,
        help="Halo context margin in meters for --merge-mode halo.",
    )
    parser.add_argument(
        "--min-tile-points",
        type=int,
        default=1024,
        help="Skip tiles with fewer points than this value.",
    )
    parser.add_argument(
        "--pre-voxel-size",
        type=float,
        default=0.0,
        help=(
            "Voxel size before model inference inside each tile. "
            "Use 0.05 to greatly speed up dense LAS inference; "
            "predictions are mapped back to all original points."
        ),
    )
    parser.add_argument(
        "--default-color",
        type=int,
        nargs=3,
        default=(255, 255, 255),
        metavar=("R", "G", "B"),
        help="RGB value used when the input LAS has no color fields.",
    )
    parser.add_argument(
        "--no-colorize",
        action="store_true",
        help="Do not overwrite LAS RGB values with prediction colors.", 
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files."
    )
    parser.add_argument(
        "--resume",
        "--skip-existing",
        dest="resume",
        action="store_true",
        help=(
            "Resume directory inference: skip complete outputs and rebuild missing "
            "or incomplete outputs."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print detailed stage timing for config/model/pipeline/read/infer/write.",
    )
    parser.add_argument(
        "--profile-fragments",
        action="store_true",
        help=(
            "Print fragment construction and model-forward timing inside each tile. "
            "This adds CUDA synchronization overhead, so use it only when diagnosing bottlenecks."
        ),
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.0,
        help=(
            "Print live CPU/GPU utilization every N seconds. 0 disables monitoring. "
            "Example: --monitor-interval 5"
        ),
    )
    keypoint_group = parser.add_argument_group(
        "绝缘子与横担关键点提取",
        "设置 --keypoints-output 后，在语义分割完成后直接从内存预测结果生成统一 JSON。",
    )
    keypoint_group.add_argument(
        "--keypoints-output",
        default=None,
        help=(
            "统一关键点 JSON 路径。输入为目录时，该参数表示 JSON 输出目录；"
            "单文件可设为 auto，自动保存到预测 LAS 同目录；"
            "不设置则只执行原有语义分割。"
        ),
    )
    keypoint_group.add_argument("--tower-class", type=int, default=1)
    keypoint_group.add_argument("--line-class", type=int, default=2)
    keypoint_group.add_argument("--insulator-class", type=int, default=3)
    keypoint_group.add_argument(
        "--insulator-connect-radius",
        "--insulator-voxel-size",
        dest="insulator_connect_radius",
        type=float,
        default=0.20,
        help="绝缘子原始点近邻连通半径；旧参数名仍兼容，但不再进行体素化。",
    )
    keypoint_group.add_argument("--insulator-neighbors", type=int, default=16)
    keypoint_group.add_argument("--min-insulator-points", type=int, default=30)
    keypoint_group.add_argument("--min-insulator-height", type=float, default=0.0)
    keypoint_group.add_argument(
        "--insulator-endpoint-percentile", type=float, default=2.0
    )
    keypoint_group.add_argument("--tower-voxel-size", type=float, default=0.75)
    keypoint_group.add_argument("--min-tower-points", type=int, default=200)
    keypoint_group.add_argument("--min-tower-height", type=float, default=4.0)
    keypoint_group.add_argument("--line-search-radius", type=float, default=35.0)
    keypoint_group.add_argument("--line-along-window", type=float, default=10.0)
    keypoint_group.add_argument("--line-layer-z-gap", type=float, default=2.0)
    keypoint_group.add_argument("--line-layer-merge-count", type=int, default=1)
    keypoint_group.add_argument("--min-line-layer-points", type=int, default=500)
    keypoint_group.add_argument("--scan-z-step", type=float, default=0.5)
    keypoint_group.add_argument("--scan-z-window", type=float, default=1.2)
    keypoint_group.add_argument("--crossarm-along-margin", type=float, default=5.0)
    keypoint_group.add_argument("--min-crossarm-points", type=int, default=50)
    keypoint_group.add_argument("--min-crossarm-width", type=float, default=2.0)
    keypoint_group.add_argument(
        "--crossarm-min-height-ratio", type=float, default=0.45
    )
    keypoint_group.add_argument(
        "--crossarm-min-z-separation", type=float, default=2.5
    )
    keypoint_group.add_argument(
        "--crossarm-min-prominence", type=float, default=0.5
    )
    keypoint_group.add_argument(
        "--crossarm-min-relative-prominence", type=float, default=0.25
    )
    keypoint_group.add_argument(
        "--crossarm-min-width-ratio", type=float, default=0.5
    )
    keypoint_group.add_argument("--endpoint-percentile", type=float, default=2.0)
    keypoint_group.add_argument("--insulator-z-margin", type=float, default=2.0)
    keypoint_group.add_argument(
        "--min-insulator-points-near-layer", type=int, default=3
    )
    keypoint_group.add_argument(
        "--no-require-insulator-after-first-layer",
        dest="require_insulator_after_first_layer",
        action="store_false",
        help="关闭第二层及以下横担必须存在绝缘子的规则。",
    )
    keypoint_group.set_defaults(require_insulator_after_first_layer=True)
    keypoint_group.add_argument("--tower-bind-xy-margin", type=float, default=8.0)
    keypoint_group.add_argument("--tower-bind-z-margin", type=float, default=8.0)
    keypoint_group.add_argument(
        "--insulator-attach-radius",
        type=float,
        default=3.0,
        help="绝缘子端点与横担线段建立挂载关系的最大三维距离，单位米。",
    )
    keypoint_group.add_argument(
        "--downward-vertical-ratio",
        type=float,
        default=0.7,
        help=(
            "同一侧挂载不少于两串绝缘子时，逐串判断并删除满足阈值的"
            "垂直向下绝缘子。"
        ),
    )
    return parser.parse_args()


def try_import_laspy():
    try:
        import laspy  # noqa: WPS433
    except ImportError:
        return None
    return laspy


def read_las_header(path):
    with path.open("rb") as file:
        header = file.read(375)
    if header[:4] != b"LASF":
        raise ValueError(f"Not a LAS file: {path}")
    version = (header[24], header[25])
    if version > (1, 4):
        raise ValueError(f"Unsupported LAS version {version[0]}.{version[1]}: {path}")
    header_size = struct.unpack_from("<H", header, 94)[0]
    point_offset = struct.unpack_from("<I", header, 96)[0]
    point_format = header[104] & 0x3F
    record_length = struct.unpack_from("<H", header, 105)[0]
    point_count = struct.unpack_from("<I", header, 107)[0]
    if point_count == 0 and header_size >= 375:
        point_count = struct.unpack_from("<Q", header, 247)[0]
    scale = np.array(struct.unpack_from("<3d", header, 131), dtype=np.float64)
    offset = np.array(struct.unpack_from("<3d", header, 155), dtype=np.float64)
    return dict(
        point_offset=point_offset,
        point_format=point_format,
        record_length=record_length,
        point_count=point_count,
        scale=scale,
        offset=offset,
    )


def output_has_expected_point_count(source_path, target_path):
    """根据 LAS 文件头点数判断已有推理结果是否完整。"""
    if not target_path.is_file():
        return False
    try:
        source_count = int(read_las_header(source_path)["point_count"])
        target_count = int(read_las_header(target_path)["point_count"])
    except Exception:
        return False
    return source_count == target_count


def color_from_16bit(color_16):
    if color_16.size == 0:
        return color_16.astype(np.uint8)
    if color_16.max(initial=0) > 255:
        return (color_16 >> 8).astype(np.uint8)
    return color_16.astype(np.uint8)


def read_las_fallback(path, default_color):
    header = read_las_header(path)
    point_format = header["point_format"]
    if point_format not in (0, 1, 2, 3):
        raise ValueError(
            "laspy is not installed and the fallback reader only supports "
            f"LAS point formats 0-3, got format {point_format}: {path}"
        )

    count = header["point_count"]
    record_length = header["record_length"]
    raw = np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=header["point_offset"],
        shape=(count * record_length,),
    )
    xyz_integer = np.stack(
        [
            np.ndarray(
                (count,),
                dtype="<i4",
                buffer=raw,
                offset=i * 4,
                strides=(record_length,),
            )
            for i in range(3)
        ],
        axis=1,
    )
    coord = xyz_integer.astype(np.float64) * header["scale"] + header["offset"]
    coord_origin = coord.min(axis=0)
    coord -= coord_origin

    color_offset = 20 if point_format == 2 else 28 if point_format == 3 else None
    if color_offset is None:
        color = np.tile(np.asarray(default_color, dtype=np.uint8), (count, 1))
    else:
        color_16 = np.stack(
            [
                np.ndarray(
                    (count,),
                    dtype="<u2",
                    buffer=raw,
                    offset=color_offset + i * 2,
                    strides=(record_length,),
                )
                for i in range(3)
            ],
            axis=1,
        )
        color = color_from_16bit(color_16)
    return (
        coord.astype(np.float32),
        color.astype(np.float32),
        None,
        coord_origin,
    )


def read_las(path, default_color, backend="auto"):
    if backend == "fallback" or (backend == "auto" and path.suffix.lower() == ".las"):
        try:
            return read_las_fallback(path, default_color)
        except ValueError:
            if backend == "fallback":
                raise

    laspy = try_import_laspy()
    if laspy is None:
        return read_las_fallback(path, default_color)

    las = laspy.read(path)
    coord = np.stack([las.x, las.y, las.z], axis=1)
    coord_origin = coord.min(axis=0)
    coord -= coord_origin
    coord = coord.astype(np.float32)
    dimension_names = set(las.point_format.dimension_names)
    if {"red", "green", "blue"}.issubset(dimension_names):
        color_16 = np.stack([las.red, las.green, las.blue], axis=1)
        color = color_from_16bit(color_16).astype(np.float32)
    else:
        color = np.tile(np.asarray(default_color, dtype=np.float32), (len(las.x), 1))
    return coord, color, las, coord_origin


def pred_to_color16(pred):
    colors = CLASS_COLOR_8BIT[np.clip(pred, 0, len(CLASS_COLOR_8BIT) - 1)]
    return (colors.astype(np.uint16) * 257).astype(np.uint16)


def write_las_fallback(input_path, output_path, pred, colorize=True):
    header = read_las_header(input_path)
    point_format = header["point_format"]
    if point_format not in (0, 1, 2, 3):
        raise ValueError(
            "laspy is not installed and the fallback writer only supports "
            f"LAS point formats 0-3, got format {point_format}: {input_path}"
        )
    if pred.shape[0] != header["point_count"]:
        raise ValueError(
            f"Prediction count {pred.shape[0]} does not match LAS point count "
            f"{header['point_count']}: {input_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, output_path)
    raw = np.memmap(
        output_path,
        dtype=np.uint8,
        mode="r+",
        offset=header["point_offset"],
        shape=(header["point_count"] * header["record_length"],),
    )
    classification = np.ndarray(
        (header["point_count"],),
        dtype=np.uint8,
        buffer=raw,
        offset=15,
        strides=(header["record_length"],),
    )
    classification[:] = pred.astype(np.uint8, copy=False)
    if colorize and point_format in (2, 3):
        color_offset = 20 if point_format == 2 else 28
        colors = pred_to_color16(pred)
        for channel in range(3):
            channel_view = np.ndarray(
                (header["point_count"],),
                dtype="<u2",
                buffer=raw,
                offset=color_offset + channel * 2,
                strides=(header["record_length"],),
            )
            channel_view[:] = colors[:, channel]
    raw.flush()


def write_las(input_path, output_path, las, pred, colorize=True):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if las is None:
        write_las_fallback(input_path, output_path, pred, colorize=colorize)
        return
    las.classification = pred.astype(np.uint8)
    if colorize:
        dimension_names = set(las.point_format.dimension_names)
        if {"red", "green", "blue"}.issubset(dimension_names):
            colors = pred_to_color16(pred)
            las.red = colors[:, 0]
            las.green = colors[:, 1]
            las.blue = colors[:, 2]
        else:
            print(
                "Warning: LAS point format has no RGB fields; only classification was written.",
                flush=True,
            )
    las.write(output_path)


def set_enable_flash(cfg, enabled):
    try:
        cfg.model.backbone.enable_flash = bool(enabled)
    except Exception:
        pass


def load_model(cfg, weight_path, device, use_fp16=False):
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
    if use_fp16:
        # spconv 对显式 FP16 权重和特征的支持比全局 autocast 更稳定。
        model.half()
    model.eval()
    return model


def build_test_pipeline(cfg, point_max=None, grid_size=None):
    test_cfg = copy.deepcopy(cfg.data.test.test_cfg)
    if grid_size is not None:
        test_cfg.voxelize.grid_size = grid_size
    if point_max is not None:
        test_cfg.crop.point_max = point_max
    return dict(
        transform=Compose(cfg.data.test.transform),
        voxelize=TRANSFORMS.build(test_cfg.voxelize),
        crop=TRANSFORMS.build(test_cfg.crop),
        post_transform=Compose(test_cfg.post_transform),
        aug_transform=[Compose(aug) for aug in test_cfg.aug_transform],
    )


def make_fragments(coord, color, pipeline, profile=False):
    stage_times = {
        "transform": 0.0,
        "augment_voxel_crop": 0.0,
        "post_transform": 0.0,
    }
    data_dict = dict(
        coord=coord.copy(),
        color=color.copy(),
        segment=np.full(coord.shape[0], -1, dtype=np.int64),
    )
    stage_start = time.perf_counter()
    data_dict = pipeline["transform"](data_dict)
    stage_times["transform"] = time.perf_counter() - stage_start
    data_dict.pop("segment", None)

    fragment_list = []
    stage_start = time.perf_counter()
    for aug in pipeline["aug_transform"]:
        aug_data = aug(copy.deepcopy(data_dict))
        data_part_list = pipeline["voxelize"](aug_data)
        for data_part in data_part_list:
            fragment_list.extend(pipeline["crop"](data_part))
    stage_times["augment_voxel_crop"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    for idx, fragment in enumerate(fragment_list):
        fragment_list[idx] = pipeline["post_transform"](fragment)
    stage_times["post_transform"] = time.perf_counter() - stage_start
    if profile:
        print(
            "    fragment build profile: transform={}, aug+voxel+crop={}, "
            "post_transform={}, fragments={}".format(
                format_seconds(stage_times["transform"]),
                format_seconds(stage_times["augment_voxel_crop"]),
                format_seconds(stage_times["post_transform"]),
                len(fragment_list),
            ),
            flush=True,
        )
    return fragment_list, stage_times


def move_tensors_to_device(input_dict, device):
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            input_dict[key] = value.to(device, non_blocking=True)
    return input_dict


def fragment_has_serialization_extent(fragment):
    """判断 Fragment 的体素坐标是否能生成非零 PTv3 序列化深度。"""
    grid_coord = fragment.get("grid_coord")
    if grid_coord is None:
        return True
    if isinstance(grid_coord, torch.Tensor):
        if grid_coord.numel() == 0:
            return False
        return int(grid_coord.max().item()) > 0
    grid_coord = np.asarray(grid_coord)
    return grid_coord.size > 0 and int(grid_coord.max()) > 0


@torch.inference_mode()
def infer_las_probs(
    model,
    cfg,
    pipeline,
    coord,
    color,
    device,
    fragment_batch_size,
    use_fp16=False,
    profile_fragments=False,
):
    if coord.shape[0] == 0:
        return np.zeros((0, cfg.data.num_classes), dtype=np.float32)

    fragment_list, fragment_times = make_fragments(
        coord, color, pipeline, profile=profile_fragments
    )
    print(f"    fragments: {len(fragment_list)}", flush=True)
    valid_fragments = [
        fragment
        for fragment in fragment_list
        if fragment_has_serialization_extent(fragment)
    ]
    skipped_fragments = len(fragment_list) - len(valid_fragments)
    if skipped_fragments:
        print(
            f"    warning: skipped {skipped_fragments} degenerate fragments "
            "with zero serialization depth; kept label 0",
            flush=True,
        )
    fragment_list = valid_fragments
    if profile_fragments:
        sync_cuda(device)
    model_start = time.perf_counter()
    batch_count = 0
    pred = torch.zeros((coord.shape[0], cfg.data.num_classes), device=device)
    for start in range(0, len(fragment_list), fragment_batch_size):
        batch_count += 1
        batch = fragment_list[start : start + fragment_batch_size]
        input_dict = move_tensors_to_device(collate_fn(batch), device)
        if use_fp16:
            # 只转换浮点张量，grid_coord、index 和 offset 必须保持整数。
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    input_dict[key] = value.half()
        index = input_dict["index"]
        logits = model(input_dict)["seg_logits"]
        # 使用 FP32 计算 softmax 和多 Fragment 概率累加，减少小类别数值误差。
        prob = F.softmax(logits.float(), dim=-1)
        begin = 0
        for end in input_dict["offset"]:
            pred[index[begin:end], :] += prob[begin:end]
            begin = end
    if profile_fragments:
        sync_cuda(device)
        print(
            "    model profile: batches={}, forward+accumulate={}, fragment_total={}".format(
                batch_count,
                format_seconds(time.perf_counter() - model_start),
                format_seconds(sum(fragment_times.values())),
            ),
            flush=True,
        )
    return pred.cpu().numpy().astype(np.float32, copy=False)


def infer_las(
    model,
    cfg,
    pipeline,
    coord,
    color,
    device,
    fragment_batch_size,
    use_fp16=False,
    profile_fragments=False,
):
    prob = infer_las_probs(
        model,
        cfg,
        pipeline,
        coord,
        color,
        device,
        fragment_batch_size,
        use_fp16=use_fp16,
        profile_fragments=profile_fragments,
    )
    return prob.argmax(axis=1).astype(np.uint8, copy=False)


def tile_starts(min_value, max_value, tile_size, tile_stride):
    if tile_size <= 0:
        return np.array([min_value], dtype=np.float64)
    starts = np.arange(min_value, max_value + 1e-6, tile_stride, dtype=np.float64)
    if starts.size == 0:
        starts = np.array([min_value], dtype=np.float64)
    return starts


def non_overlapping_tile_groups(coord, tile_size):
    min_xy = coord[:, :2].min(axis=0)
    tile_xy = np.floor((coord[:, :2] - min_xy) / tile_size).astype(np.int64)
    x_count = int(tile_xy[:, 0].max()) + 1
    y_count = int(tile_xy[:, 1].max()) + 1
    tile_id = tile_xy[:, 0] * y_count + tile_xy[:, 1]
    order = np.argsort(tile_id, kind="mergesort")
    sorted_id = tile_id[order]
    unique_id, starts, counts = np.unique(
        sorted_id, return_index=True, return_counts=True
    )
    return min_xy, x_count, y_count, order, unique_id, starts, counts


def voxel_reduce_points(coord, color, voxel_size):
    if voxel_size <= 0 or coord.shape[0] == 0:
        return coord, color, None
    grid_coord = np.floor((coord - coord.min(axis=0)) / voxel_size).astype(np.int64)
    _, unique_index, inverse = np.unique(
        grid_coord, axis=0, return_index=True, return_inverse=True
    )
    return coord[unique_index], color[unique_index], inverse


def infer_tile_probs(
    model,
    cfg,
    pipeline,
    coord,
    color,
    device,
    fragment_batch_size,
    pre_voxel_size,
    use_fp16=False,
    profile_fragments=False,
):
    if pre_voxel_size <= 0:
        return infer_las_probs(
            model,
            cfg,
            pipeline,
            coord,
            color,
            device,
            fragment_batch_size,
            use_fp16=use_fp16,
            profile_fragments=profile_fragments,
        )
    reduce_start = time.perf_counter()
    reduced_coord, reduced_color, inverse = voxel_reduce_points(coord, color, pre_voxel_size)
    reduce_time = time.perf_counter() - reduce_start
    print(
        f"    pre-voxel {coord.shape[0]} -> {reduced_coord.shape[0]} "
        f"points in {reduce_time:.2f}s",
        flush=True,
    )
    reduced_prob = infer_las_probs(
        model,
        cfg,
        pipeline,
        reduced_coord,
        reduced_color,
        device,
        fragment_batch_size,
        use_fp16=use_fp16,
        profile_fragments=profile_fragments,
    )
    map_start = time.perf_counter()
    mapped_prob = reduced_prob[inverse].astype(np.float32, copy=False)
    if profile_fragments:
        print(f"    map-back profile: {format_seconds(time.perf_counter() - map_start)}", flush=True)
    return mapped_prob


def infer_tile_points(
    model,
    cfg,
    pipeline,
    coord,
    color,
    device,
    fragment_batch_size,
    pre_voxel_size,
    use_fp16=False,
    profile_fragments=False,
):
    prob = infer_tile_probs(
        model,
        cfg,
        pipeline,
        coord,
        color,
        device,
        fragment_batch_size,
        pre_voxel_size,
        use_fp16=use_fp16,
        profile_fragments=profile_fragments,
    )
    return prob.argmax(axis=1).astype(np.uint8, copy=False)


def tile_grid_bounds(coord, tile_size, tile_stride):
    x_starts = tile_starts(float(coord[:, 0].min()), float(coord[:, 0].max()), tile_size, tile_stride)
    y_starts = tile_starts(float(coord[:, 1].min()), float(coord[:, 1].max()), tile_size, tile_stride)
    return x_starts, y_starts


def points_in_xy_box(coord, x0, x1, y0, y1):
    return (coord[:, 0] >= x0) & (coord[:, 0] <= x1) & (coord[:, 1] >= y0) & (coord[:, 1] <= y1)


def infer_las_tiled(
    model,
    cfg,
    pipeline,
    coord,
    color,
    device,
    fragment_batch_size,
    tile_size,
    tile_stride,
    min_tile_points,
    pre_voxel_size,
    merge_mode="plain",
    context_margin=0.0,
    stats=None,
    use_fp16=False,
    profile_fragments=False,
):
    if tile_size <= 0 or coord.shape[0] <= pipeline["crop"].point_max:
        print(f"  infer whole cloud: {coord.shape[0]} points", flush=True)
        if stats is not None:
            stats["total_tiles"] += 1
            stats["processed_tiles"] += 1
        return infer_tile_points(
            model,
            cfg,
            pipeline,
            coord,
            color,
            device,
            fragment_batch_size,
            pre_voxel_size,
            use_fp16=use_fp16,
            profile_fragments=profile_fragments,
        )

    if tile_stride <= 0:
        raise ValueError("--tile-stride must be > 0 when --tile-size is enabled")

    if merge_mode not in ("plain", "halo", "overlap"):
        raise ValueError(f"Unsupported merge mode: {merge_mode}")
    if merge_mode == "halo" and context_margin <= 0:
        raise ValueError("--context-margin must be > 0 when --merge-mode halo")

    if merge_mode in ("halo", "overlap"):
        x_starts, y_starts = tile_grid_bounds(coord, tile_size, tile_stride)
        total_tiles = len(x_starts) * len(y_starts)
        if stats is not None:
            stats["total_tiles"] += int(total_tiles)
        used_tiles = 0
        done = np.zeros(coord.shape[0], dtype=bool)
        if merge_mode == "overlap":
            prob_sum = np.zeros((coord.shape[0], cfg.data.num_classes), dtype=np.float32)
            vote_count = np.zeros(coord.shape[0], dtype=np.uint16)
        else:
            pred = np.full(coord.shape[0], 0, dtype=np.uint8)
        print(
            f"  tile inference: {coord.shape[0]} points, "
            f"{len(x_starts)} x {len(y_starts)} = {total_tiles} tiles, "
            f"tile_size={tile_size}, stride={tile_stride}, mode={merge_mode}, "
            f"context_margin={context_margin}",
            flush=True,
        )
        tile_idx = 0
        for x0 in x_starts:
            x1 = x0 + tile_size
            for y0 in y_starts:
                tile_idx += 1
                y1 = y0 + tile_size
                if merge_mode == "halo":
                    infer_x0, infer_x1 = x0 - context_margin, x1 + context_margin
                    infer_y0, infer_y1 = y0 - context_margin, y1 + context_margin
                    infer_mask = points_in_xy_box(coord, infer_x0, infer_x1, infer_y0, infer_y1)
                    write_mask = points_in_xy_box(coord, x0, x1, y0, y1) & (~done)
                else:
                    infer_mask = points_in_xy_box(coord, x0, x1, y0, y1)
                    write_mask = infer_mask
                infer_indices = np.flatnonzero(infer_mask)
                if infer_indices.size < min_tile_points:
                    if stats is not None:
                        stats["skipped_tiles"] += 1
                    continue
                if merge_mode == "halo":
                    local_write = np.flatnonzero(write_mask[infer_indices])
                    if local_write.size == 0:
                        if stats is not None:
                            stats["skipped_tiles"] += 1
                        continue
                used_tiles += 1
                if stats is not None:
                    stats["processed_tiles"] += 1
                tile_start = time.perf_counter()
                print(
                    f"  tile {tile_idx}/{total_tiles}: infer={infer_indices.size} "
                    f"write={int(write_mask.sum())} "
                    f"(x={x0:.2f}..{x1:.2f}, y={y0:.2f}..{y1:.2f})",
                    flush=True,
                )
                tile_prob = infer_tile_probs(
                    model,
                    cfg,
                    pipeline,
                    coord[infer_indices],
                    color[infer_indices],
                    device,
                    fragment_batch_size,
                    pre_voxel_size,
                    use_fp16=use_fp16,
                    profile_fragments=profile_fragments,
                )
                if merge_mode == "halo":
                    write_indices = infer_indices[local_write]
                    pred[write_indices] = tile_prob[local_write].argmax(axis=1).astype(np.uint8)
                    done[write_indices] = True
                else:
                    prob_sum[infer_indices] += tile_prob
                    vote_count[infer_indices] += 1
                    done[infer_indices] = True
                print(f"    done in {time.perf_counter() - tile_start:.2f}s", flush=True)
        if merge_mode == "overlap":
            pred = np.full(coord.shape[0], 0, dtype=np.uint8)
            covered = vote_count > 0
            pred[covered] = prob_sum[covered].argmax(axis=1).astype(np.uint8)
            done = covered
        missing = int((~done).sum())
        if missing:
            print(f"  warning: {missing} points were not covered by tiles; kept label 0", flush=True)
        print(f"  finished {used_tiles} non-empty tiles", flush=True)
        return pred

    pred = np.full(coord.shape[0], 0, dtype=np.uint8)
    done = np.zeros(coord.shape[0], dtype=bool)
    used_tiles = 0

    if np.isclose(tile_stride, tile_size):
        group_start = time.perf_counter()
        min_xy, x_count, y_count, order, unique_id, starts, counts = non_overlapping_tile_groups(
            coord, tile_size
        )
        total_tiles = x_count * y_count
        if stats is not None:
            stats["total_tiles"] += int(total_tiles)
        print(
            f"  tile inference: {coord.shape[0]} points, "
            f"{x_count} x {y_count} = {total_tiles} tiles, "
            f"non-empty={len(unique_id)}, tile_size={tile_size}, "
            f"grouping={time.perf_counter() - group_start:.2f}s",
            flush=True,
        )
        for group_idx, (tile_id, start, count) in enumerate(
            zip(unique_id, starts, counts), start=1
        ):
            if count < min_tile_points:
                if stats is not None:
                    stats["skipped_tiles"] += 1
                continue
            used_tiles += 1
            if stats is not None:
                stats["processed_tiles"] += 1
            indices = order[start : start + count]
            ix = int(tile_id // y_count)
            iy = int(tile_id % y_count)
            x0 = float(min_xy[0] + ix * tile_size)
            y0 = float(min_xy[1] + iy * tile_size)
            tile_start = time.perf_counter()
            print(
                f"  tile {group_idx}/{len(unique_id)}: {count} points "
                f"(x={x0:.2f}..{x0 + tile_size:.2f}, "
                f"y={y0:.2f}..{y0 + tile_size:.2f})",
                flush=True,
            )
            tile_pred = infer_tile_points(
                model,
                cfg,
                pipeline,
                coord[indices],
                color[indices],
                device,
                fragment_batch_size,
                pre_voxel_size,
                use_fp16=use_fp16,
                profile_fragments=profile_fragments,
            )
            pred[indices] = tile_pred
            done[indices] = True
            print(f"    done in {time.perf_counter() - tile_start:.2f}s", flush=True)
    else:
        x_starts = tile_starts(float(coord[:, 0].min()), float(coord[:, 0].max()), tile_size, tile_stride)
        y_starts = tile_starts(float(coord[:, 1].min()), float(coord[:, 1].max()), tile_size, tile_stride)
        total_tiles = len(x_starts) * len(y_starts)
        if stats is not None:
            stats["total_tiles"] += int(total_tiles)
        tile_idx = 0
        print(
            f"  tile inference: {coord.shape[0]} points, "
            f"{len(x_starts)} x {len(y_starts)} = {total_tiles} tiles, "
            f"tile_size={tile_size}, stride={tile_stride}",
            flush=True,
        )

        for x0 in x_starts:
            x1 = x0 + tile_size
            x_mask = (coord[:, 0] >= x0) & (coord[:, 0] <= x1)
            for y0 in y_starts:
                tile_idx += 1
                y1 = y0 + tile_size
                mask = x_mask & (coord[:, 1] >= y0) & (coord[:, 1] <= y1)
                indices = np.flatnonzero(mask)
                if indices.size < min_tile_points:
                    if stats is not None:
                        stats["skipped_tiles"] += 1
                    continue
                used_tiles += 1
                if stats is not None:
                    stats["processed_tiles"] += 1
                tile_start = time.perf_counter()
                print(
                    f"  tile {tile_idx}/{total_tiles}: {indices.size} points "
                    f"(x={x0:.2f}..{x1:.2f}, y={y0:.2f}..{y1:.2f})",
                    flush=True,
                )
                tile_pred = infer_tile_points(
                    model,
                    cfg,
                    pipeline,
                    coord[indices],
                    color[indices],
                    device,
                    fragment_batch_size,
                    pre_voxel_size,
                    use_fp16=use_fp16,
                    profile_fragments=profile_fragments,
                )
                pred[indices] = tile_pred
                done[indices] = True
                print(f"    done in {time.perf_counter() - tile_start:.2f}s", flush=True)

    missing = int((~done).sum())
    if missing:
        print(f"  warning: {missing} points were not covered by tiles; kept label 0", flush=True)
    print(f"  finished {used_tiles} non-empty tiles", flush=True)
    return pred


def iter_jobs(input_path, output_path):
    if input_path.is_file():
        if output_path.exists() and output_path.is_dir():
            return [(input_path, output_path / input_path.name)]
        return [(input_path, output_path)]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    return [
        (path, output_path / path.relative_to(input_path))
        for path in sorted(input_path.rglob("*.las"))
    ]


def keypoints_output_path(keypoints_output, input_path, source_path, target_path):
    """根据单文件或目录推理模式生成对应的关键点 JSON 路径。"""
    if str(keypoints_output).strip().lower() == "auto":
        return target_path.with_name(f"{target_path.stem}_keypoints.json")

    root = Path(keypoints_output)
    if input_path.is_file() or source_path == input_path:
        suffix = root.suffix.lower()
        if suffix == ".json":
            return root
        # 容错：如果误把预测 LAS 路径传给关键点参数，改为同目录同名 JSON，
        # 避免 Windows 把已经存在的 .las 文件当成目录创建而触发 WinError 183。
        if suffix in (".las", ".laz"):
            return root.with_name(f"{root.stem}_keypoints.json")
        if suffix:
            raise ValueError(
                "单文件推理时，--keypoints-output 必须是 .json 文件、目录或 auto，"
                f"当前值：{root}"
            )
        return root / f"{target_path.stem}_keypoints.json"

    if root.suffix.lower() == ".json":
        raise ValueError("目录批量推理时，--keypoints-output 必须是目录。")
    relative_path = source_path.relative_to(input_path)
    return (root / relative_path).with_suffix(".json")


def load_keypoint_extractor():
    """仅在请求关键点输出时加载后处理模块，保持原推理依赖不变。"""
    try:
        from tools.insulator_hengdan.extract_segmented_insulator_crossarm_keypoints import (
            extract_keypoints_from_arrays,
            report_counts,
            write_keypoints_json,
        )
    except ModuleNotFoundError as error:
        if error.name not in (
            "tools.insulator_hengdan",
            "tools.insulator_hengdan.extract_segmented_insulator_crossarm_keypoints",
        ):
            raise
        # 兼容早期 Windows 部署中两个脚本都直接放在 tools 目录的布局。
        from tools.extract_segmented_insulator_crossarm_keypoints import (
            extract_keypoints_from_arrays,
            report_counts,
            write_keypoints_json,
        )

    return extract_keypoints_from_arrays, write_keypoints_json, report_counts


def main():
    args = parse_args()
    program_start = time.perf_counter()
    input_path = Path(args.input)
    output_path = Path(args.output)
    config_start = time.perf_counter()
    cfg = Config.fromfile(args.config_file)
    config_time = time.perf_counter() - config_start
    if args.disable_flash:
        set_enable_flash(cfg, False)
    device = torch.device(args.device)

    if args.fragment_batch_size < 1:
        raise ValueError("--fragment-batch-size must be >= 1")
    if args.fp16 and device.type != "cuda":
        raise ValueError("--fp16 仅支持 CUDA 推理，请使用 --device cuda 或取消 --fp16")
    if args.resume and args.overwrite:
        raise ValueError("--resume/--skip-existing 不能与 --overwrite 同时使用")

    model_start = time.perf_counter()
    model = load_model(cfg, args.weight, device, use_fp16=args.fp16)
    sync_cuda(device)
    model_time = time.perf_counter() - model_start
    pipeline_start = time.perf_counter()
    pipeline = build_test_pipeline(cfg, args.point_max, args.grid_size)
    pipeline_time = time.perf_counter() - pipeline_start
    jobs_start = time.perf_counter()
    jobs = iter_jobs(input_path, output_path)
    jobs_time = time.perf_counter() - jobs_start
    if not jobs:
        raise FileNotFoundError(f"No .las files found under: {input_path}")
    keypoint_extractor = None
    keypoint_writer = None
    keypoint_counter = None
    if args.keypoints_output is not None:
        (
            keypoint_extractor,
            keypoint_writer,
            keypoint_counter,
        ) = load_keypoint_extractor()
    print(
        f"Inference precision: {'FP16' if args.fp16 else 'FP32'}",
        flush=True,
    )

    if args.profile:
        print(
            "Startup timing: config={}, model_load={}, pipeline={}, jobs={}, device={}, {}".format(
                format_seconds(config_time),
                format_seconds(model_time),
                format_seconds(pipeline_time),
                format_seconds(jobs_time),
                device,
                cuda_current_memory_summary(device),
            ),
            flush=True,
        )

    all_start = time.perf_counter()
    all_stats = new_infer_stats()
    completed_jobs = 0
    skipped_jobs = 0
    accumulated_job_time = 0.0
    with ResourceMonitor(device, args.monitor_interval):
        for source_path, target_path in jobs:
            keypoint_target = None
            if keypoint_extractor is not None:
                keypoint_target = keypoints_output_path(
                    args.keypoints_output,
                    input_path,
                    source_path,
                    target_path,
                )
            if target_path.exists():
                output_complete = output_has_expected_point_count(
                    source_path, target_path
                )
                keypoints_complete = (
                    keypoint_target is None or keypoint_target.is_file()
                )
                if args.resume and output_complete and keypoints_complete:
                    skipped_jobs += 1
                    print(
                        f"Skipping complete output: {target_path}",
                        flush=True,
                    )
                    continue
                if not args.overwrite and not args.resume:
                    raise FileExistsError(
                        f"Output exists, use --overwrite or --resume: {target_path}"
                    )
                if args.resume:
                    reason = (
                        "missing keypoints JSON"
                        if output_complete and not keypoints_complete
                        else "incomplete output LAS"
                    )
                    print(
                        f"Rebuilding {reason}: {target_path}",
                        flush=True,
                    )
            job_start = time.perf_counter()
            reset_cuda_peak_memory(device)
            print(f"Reading {source_path}", flush=True)
            read_start = time.perf_counter()
            coord, color, las, coord_origin = read_las(
                source_path, args.default_color, args.las_backend
            )
            read_time = time.perf_counter() - read_start
            print(
                f"Loaded {coord.shape[0]} points in {read_time:.2f}s",
                flush=True,
            )
            if args.profile:
                print(
                    "Read profile: coord_shape={}, color_shape={}, las_backend={}, {}".format(
                        tuple(coord.shape),
                        tuple(color.shape),
                        args.las_backend,
                        cuda_current_memory_summary(device),
                    ),
                    flush=True,
                )
            infer_stats = new_infer_stats()
            sync_cuda(device)
            infer_start = time.perf_counter()
            pred = infer_las_tiled(
                model=model,
                cfg=cfg,
                pipeline=pipeline,
                coord=coord,
                color=color,
                device=device,
                fragment_batch_size=args.fragment_batch_size,
                tile_size=args.tile_size,
                tile_stride=args.tile_stride,
                min_tile_points=args.min_tile_points,
                pre_voxel_size=args.pre_voxel_size,
                merge_mode=args.merge_mode,
                context_margin=args.context_margin,
                stats=infer_stats,
                use_fp16=args.fp16,
                profile_fragments=args.profile_fragments,
            )
            sync_cuda(device)
            infer_time = time.perf_counter() - infer_start
            write_start = time.perf_counter()
            write_las(source_path, target_path, las, pred, colorize=not args.no_colorize)
            write_time = time.perf_counter() - write_start
            print(f"Wrote LAS in {write_time:.2f}s", flush=True)

            keypoint_time = 0.0
            if keypoint_extractor is not None:
                keypoint_start = time.perf_counter()
                # 模型推理使用减去 LAS 最小坐标后的局部坐标，以避免大坐标造成
                # 浮点精度损失；JSON 对外统一保存恢复后的 LAS 全局坐标。
                keypoint_coord = coord.astype(np.float64) + coord_origin[None, :]
                report = keypoint_extractor(
                    keypoint_coord, pred, args, verbose=True
                )
                keypoint_writer(
                    report,
                    keypoint_target,
                    overwrite=args.overwrite or args.resume,
                )
                tower_count, crossarm_count, insulator_count = keypoint_counter(report)
                keypoint_time = time.perf_counter() - keypoint_start
                print(
                    f"Saved keypoints {keypoint_target} "
                    f"(towers={tower_count}, crossarms={crossarm_count}, "
                    f"insulators={insulator_count}) in "
                    f"{format_seconds(keypoint_time)}",
                    flush=True,
                )

            counts = np.bincount(pred, minlength=cfg.data.num_classes)
            summary = ", ".join(
                f"{idx}:{cfg.data.names[idx]}={int(counts[idx])}"
                for idx in range(cfg.data.num_classes)
            )
            print(f"Saved {target_path} ({coord.shape[0]} points; {summary})")
            for key in all_stats:
                all_stats[key] += int(infer_stats[key])
            processed_tiles = int(infer_stats["processed_tiles"])
            total_tiles = int(infer_stats["total_tiles"])
            skipped_tiles = max(total_tiles - processed_tiles, 0)
            job_time = time.perf_counter() - job_start
            completed_jobs += 1
            accumulated_job_time += job_time
            print(
                "Timing summary: total={}, read={}, inference={}, write={}, keypoints={}, "
                "tiles_processed={}, tiles_total={}, tiles_skipped={}, {}".format(
                    format_seconds(job_time),
                    format_seconds(read_time),
                    format_seconds(infer_time),
                    format_seconds(write_time),
                    format_seconds(keypoint_time),
                    processed_tiles,
                    total_tiles,
                    skipped_tiles,
                    cuda_peak_memory_summary(device),
                ),
                flush=True,
            )
            if args.profile:
                other_time = max(
                    job_time
                    - read_time
                    - infer_time
                    - write_time
                    - keypoint_time,
                    0.0,
                )
                print(
                    "Stage timing detail: startup_before_jobs={}, read={}, inference={}, "
                    "write={}, keypoints={}, other_job_overhead={}, "
                    "total_program_so_far={}".format(
                        format_seconds(all_start - program_start),
                        format_seconds(read_time),
                        format_seconds(infer_time),
                        format_seconds(write_time),
                        format_seconds(keypoint_time),
                        format_seconds(other_time),
                        format_seconds(time.perf_counter() - program_start),
                    ),
                    flush=True,
                )

    total_processed = int(all_stats["processed_tiles"])
    total_tiles = int(all_stats["total_tiles"])
    program_time = time.perf_counter() - program_start
    average_job_time = (
        accumulated_job_time / completed_jobs if completed_jobs else 0.0
    )
    print(
        "Overall summary: files_processed={}, files_skipped={}, total_elapsed={}, "
        "file_processing_total={}, average_per_las={}, tiles_processed={}, "
        "tiles_total={}, tiles_skipped={}".format(
            completed_jobs,
            skipped_jobs,
            format_seconds(program_time),
            format_seconds(accumulated_job_time),
            format_seconds(average_job_time),
            total_processed,
            total_tiles,
            max(total_tiles - total_processed, 0),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

# 使用教程：

"""   
python tools/infer_las_semseg.py \
    --input /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb009b5736892392a.las \
    --output /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb_outPut.las \
    --overwrite

  也支持目录批量推理：

  python tools/infer_las_semseg.py \
    --input /path/to/las_dir \
    --output /path/to/output_dir \
    --overwrite
    
    python tools/infer_las_semseg.py \
    --input /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb009b5736892392a.las \
    --output /24085403037/24085403037/shixi/dataset/6_23_demo/cloudb_outPut.las \
    --fragment-batch-size 6 \
    --min-tile-points 1024 \
    --pre-voxel-size 0.05 \
    --profile \
    --profile-fragments \
    --monitor-interval 5 \
    --overwrite
    
      1. 当前模式 plain

  python tools/infer/infer_las_semseg.py \
    --input /24085403037/24085403037/shixi/dataset/0617-4Name_original_test_merged/ \
    --output /24085403037/24085403037/shixi/dataset/0617-4Name_Output_test_merged \
    --merge-mode plain \
    --tile-size 40 \
    --tile-stride 40 \
    --pre-voxel-size 0.05 \
    --overwrite

  2. Halo 上下文模式，推荐优先试

  python tools/infer/infer_las_semseg.py \
    --input /24085403037/24085403037/shixi/dataset/6_23_demo/test/test_insulator_hengdan/source_tower/Stage1_tower/tower_004_杆塔4.las \
    --output /24085403037/24085403037/shixi/dataset/6_23_demo/test/test_insulator_hengdan/source_tower/tower_004_杆塔4_test.las \
    --merge-mode halo \
    --tile-size 40 \
    --tile-stride 40 \
    --context-margin 10 \
    --pre-voxel-size 0.01 \
    --overwrite

  含义是：每个核心块仍是 40m，但实际推理用周围额外 10m 上下文，只写回中间 40m 核心区。这个通常比 plain 稳，比 overlap 快。

  3. Overlap 概率平均模式

  python tools/infer_las_semseg.py \
    --input .../cloudb009b5736892392a.las \
    --output .../cloudb_overlap.las \
    --merge-mode overlap \
    --tile-size 40 \
    --tile-stride 20 \
    --pre-voxel-size 0.05 \
    --overwrite

  含义是重叠推理，同一个点可能被多个 tile 预测，脚本会累加 softmax 概率后取平均结果。这个通常最稳，但最慢，也更吃内存。

  建议对比顺序：

  plain -> halo context 10m -> overlap stride 20m

  如果 halo 效果已经够好，就不建议用 overlap，因为 overlap 会重复推理很多点。对于你这种 4800 万点 LAS，halo 更可能是速度和精度的折中最优。

 """
