"""使用 FP16 执行输电线路 LAS 语义分割。

本脚本复用 infer_las_semseg.py 的完整推理流程，只负责默认启用 --fp16。
Windows RTX 4050 运行时可以继续添加 --disable-flash。

示例：
python tools/infer/infer_las_semseg_fp16.py \
  --input /24085403037/24085403037/shixi/dataset/0617-4Name/110v12/110v12_merged_4cls.las \
  --output /24085403037/24085403037/shixi/dataset/6_23_demo/test/infer/110v12_merged_4cls_Output_fp16.las \
  --keypoints-output /24085403037/24085403037/shixi/dataset/6_23_demo/test/hengdan_insulator/110v12_merged_4cls_hengdan_insulator_FP16.json \
  --merge-mode plain \
  --tile-size 40 \
  --tile-stride 40 \
  --pre-voxel-size 0.05 \
  --fragment-batch-size 1 \
  --overwrite
  
  
python tools/infer/infer_las_semseg_fp16.py \
  --input /24085403037/24085403037/shixi/dataset/lidar_original/cloud0.las \
  --output /24085403037/24085403037/shixi/dataset/lidar_test/cloud0.las \
  --merge-mode plain \
  --tile-size 40 \
  --tile-stride 40 \
  --pre-voxel-size 0.05 \
  --fragment-batch-size 1 \
  --overwrite
"""

import inspect
import sys

try:
    from . import infer_las_semseg
except ImportError:
    import infer_las_semseg


if __name__ == "__main__":
    load_model_parameters = inspect.signature(
        infer_las_semseg.load_model
    ).parameters
    if "use_fp16" not in load_model_parameters:
        raise RuntimeError(
            "infer_las_semseg_fp16.py 与 infer_las_semseg.py 版本不匹配。"
            "请同时更新 tools/infer 目录中的这两个脚本；"
            "当前 infer_las_semseg.py 尚未包含 FP16 推理实现。"
        )
    if "--fp16" not in sys.argv:
        sys.argv.append("--fp16")
    infer_las_semseg.main()
