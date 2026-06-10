# Transmission-Line Point Cloud Segmentation Based on Pointcept

本仓库基于 [Pointcept v1.5.1](https://github.com/Pointcept/Pointcept) 和 Point Transformer V3，扩展了面向输电线路点云的语义分割流程。当前代码重点支持：

- 将按类别拆分的 LAS 点云转换为 Pointcept `.pth` 数据。
- 训练 6 类基线分割模型。
- 训练两阶段模型：Stage 1 粗分割 + Stage 2 杆塔 ROI 精分割。
- 使用两阶段权重进行最终 6 类推理。
- 计算最终 mIoU/mAcc/allAcc。
- 导出 Stage1、Stage2 或最终结果为 PLY 方便可视化。

## 1. 类别定义

原始 6 类输电线路语义标签：

| ID | 类别 | 说明 |
| --- | --- | --- |
| 0 | ground | 地面 |
| 1 | tower | 杆塔 |
| 2 | line | 导线 |
| 3 | insulator | 绝缘子 |
| 4 | hengdan | 横担 |
| 5 | other | 其他 |

预处理脚本会将原始标签 `6` 合并到 `other`。

## 2. 目录结构

关键代码如下：

```text
configs/transmission_line/
  semseg-pt-v3m1-0-base.py              # 6 类 PTv3 基线配置
  semseg-pt-v3m1-0-base_v2.py           # 6 类增强权重配置
  semseg-pt-v3m1-stage1-4cls.py         # Stage1 4 类粗分割配置
  semseg-pt-v3m1-stage2-tower.py        # Stage2 杆塔 ROI 精分割配置

pointcept/datasets/preprocessing/transmission_line/
  preprocess_transmission_line.py       # LAS -> Pointcept .pth

tools/
  make_stage1_4cls.py                   # 6 类数据 -> Stage1 4 类数据
  make_stage1_balanced_train.py         # Stage1 训练集过采样
  make_stage2_tower_roi.py              # 从 6 类数据生成 Stage2 杆塔 ROI
  make_stage2_balanced_train.py         # Stage2 训练集过采样
  make_stage2_insulator_centered.py     # 生成绝缘子/横担中心增强 ROI
  make_stage2_insulator_focus.py        # 生成绝缘子增强数据
  infer_transmission_line_two_stage.py  # 两阶段最终 6 类推理
  evaluate_transmission_line_pred.py    # 计算预测结果 mIoU
  export_transmission_line_pred_ply.py  # 6 类预测结果导出 PLY
  export_transmission_line_stage1_pred_ply.py # Stage1 4 类预测导出 PLY
  export_transmission_line_data_ply.py  # 原始 .pth 数据导出 PLY
```

数据、权重和实验输出默认不上传 GitHub：

```text
data/
exp/
pretrained/
weights/
*.pth
*.ply
*.npy
```

这些内容已在 `.gitignore` 中忽略。

## 3. 数据预处理

原始数据目录要求每个场景一个文件夹，每个 LAS 文件只包含一个类别：

```text
<dataset-root>/
  scene01/
    0_ground.las
    1_tower.las
    2_line.las
    3_insulator.las
    4_hengdan.las
    5_other.las
  scene02/
    ...
```

转换命令示例：

```bash
cd /24085403037/PointTransformerV3/Pointcept-v1.5.1

/opt/conda/envs/pointcept/bin/python \
  pointcept/datasets/preprocessing/transmission_line/preprocess_transmission_line.py \
  --dataset-root /path/to/raw_las_dataset \
  --output-root data/transmission_line \
  --voxel-size 0 \
  --tile-size 40 \
  --train-tile-stride 20 \
  --eval-tile-stride 40 \
  --min-points 1024 \
  --overwrite
```

预处理逻辑：

- 读取每个 scene 下的分类 LAS 文件。
- 将大地坐标转换为 scene 局部坐标，降低浮点精度问题。
- 按 XY 空间切成 tile，控制显存和样本规模。
- 过滤少于 `1024` 点的稀疏 tile。
- 保存为 Pointcept 可读取的 `.pth`，字段包括 `coord`、`color`、`semantic_gt`、`origin`、`scene`。

数据划分按场景号固定完成：

| split | scene |
| --- | --- |
| train | scene01 - scene40 |
| val | scene41 - scene48 |
| test | scene49 以后 |

## 4. Stage1 粗分割

Stage1 将原始 6 类合并为 4 类：

| Stage1 ID | 类别 | 来源 |
| --- | --- | --- |
| 0 | ground | ground |
| 1 | tower_structure | tower + insulator + hengdan |
| 2 | line | line |
| 3 | other | other |

生成 Stage1 数据：

```bash
/opt/conda/envs/pointcept/bin/python tools/make_stage1_4cls.py \
  --src-root data/transmission_line \
  --dst-root data/transmission_line_stage1_4cls \
  --overwrite

/opt/conda/envs/pointcept/bin/python tools/make_stage1_balanced_train.py
```

训练 Stage1：

```bash
sh scripts/train.sh \
  -p /opt/conda/envs/pointcept/bin/python \
  -d transmission_line \
  -c semseg-pt-v3m1-stage1-4cls \
  -n stage1_4cls_balance_w8_clean \
  -g 1
```

测试 Stage1：

```bash
sh scripts/test.sh \
  -p /opt/conda/envs/pointcept/bin/python \
  -d transmission \
  -n stage1_4cls_balance_w8_clean \
  -w model_best \
  -g 1
```

Stage1 预测结果会输出到：

```text
exp/transmission/stage1_4cls_balance_w8_clean/result/
```

## 5. Stage2 杆塔 ROI 精分割

Stage2 在杆塔局部 ROI 内做 4 类精分割：

| Stage2 ID | 类别 | 最终 6 类映射 |
| --- | --- | --- |
| 0 | tower | tower |
| 1 | insulator | insulator |
| 2 | hengdan | hengdan |
| 3 | background | 保留 Stage1 或作为背景 |

生成 Stage2 数据：

```bash
/opt/conda/envs/pointcept/bin/python tools/make_stage2_tower_roi.py \
  --src-root data/transmission_line \
  --dst-root data/transmission_line_stage2_tower \
  --overwrite

/opt/conda/envs/pointcept/bin/python tools/make_stage2_balanced_train.py

/opt/conda/envs/pointcept/bin/python tools/make_stage2_insulator_centered.py \
  --src-root data/transmission_line \
  --base-stage2-root data/transmission_line_stage2_tower_balance \
  --dst-root data/transmission_line_stage2_tower_ins_centered \
  --overwrite
```

训练 Stage2：

```bash
sh scripts/train.sh \
  -p /opt/conda/envs/pointcept/bin/python \
  -d transmission_line \
  -c semseg-pt-v3m1-stage2-tower \
  -n stage2_tower_ins_centered_w24 \
  -g 1
```

测试 Stage2：

```bash
sh scripts/test.sh \
  -p /opt/conda/envs/pointcept/bin/python \
  -d transmission \
  -n stage2_tower_ins_centered_w24 \
  -w model_best \
  -g 1
```

Stage2 预测结果会输出到：

```text
exp/transmission/stage2_tower_ins_centered_w24/result/
```

## 6. 两阶段最终推理

最终推理使用：

1. Stage1 权重或已有 Stage1 预测，得到 `ground / tower_structure / line / other`。
2. 从 Stage1 的 `tower_structure` 自动生成 ROI。
3. Stage2 权重细分 ROI 内的 `tower / insulator / hengdan / background`。
4. 将 Stage2 结果回填到原始 tile，得到最终 6 类预测。

推荐复用已有 Stage1 预测，避免重复计算：

```bash
/opt/conda/envs/pointcept/bin/python tools/infer_transmission_line_two_stage.py \
  --data-root data/transmission_line \
  --split test \
  --out exp/transmission/two_stage_infer_v1/result \
  --stage1-pred-dir exp/transmission/stage1_4cls_balance_w8_clean_v1/result \
  --fast-crop
```
不复用的话：
/opt/conda/envs/pointcept/bin/python tools/infer_transmission_line_two_stage.py \
  --data-root data/transmission_line \
  --split test \
  --stage1-config exp/transmission/stage1_4cls_balance_w8_clean_seq_v2/config.py \
  --stage1-weight exp/transmission/stage1_4cls_balance_w8_clean_seq_v2/model/model_best.pth \
  --stage2-config exp/transmission/stage2_tower_ins_centered_seq_v2/config.py \
  --stage2-weight exp/transmission/stage2_tower_ins_centered_seq_v2/model/model_best.pth \
  --out exp/transmission_two_stage_infer/two_stage_infer_v1/result \
  --fast-crop
参数说明：

- `--stage1-pred-dir`：使用已有一阶段 `*_pred.npy`，跳过 Stage1 模型推理。
- `--fast-crop`：百万级大 tile 使用快速分块推理，避免 CPU 端生成大量 fragment。
- 不加 `--overwrite` 时会自动跳过已存在的最终预测，可断点续跑。

最终输出：

```text
exp/transmission/two_stage_infer/result/*_pred.npy
```

最终 6 类映射：

```text
0 ground
1 tower
2 line
3 insulator
4 hengdan
5 other
```

## 7. 计算 mIoU

对最终二阶段 6 类结果计算 mIoU：

```bash
/opt/conda/envs/pointcept/bin/python tools/evaluate_transmission_line_pred.py \
  exp/transmission/two_stage_infer/result \
  --data-root data/transmission_line \
  --split test \
  --out exp/transmission/two_stage_infer/two_stage_metrics.json
```

脚本会输出：

- mIoU
- mAcc
- allAcc
- 每类 IoU
- 每类 accuracy
- intersection / union / target 点数

当前一次测试结果示例：

```text
mIoU/mAcc/allAcc 0.4919/0.6506/0.9118
ground     IoU 0.9257
tower      IoU 0.7134
line       IoU 0.3749
insulator  IoU 0.2180
hengdan    IoU 0.6480
other      IoU 0.0712
```

## 8. 导出 PLY 可视化

### 8.1 导出最终 6 类预测

```bash
/opt/conda/envs/pointcept/bin/python tools/export_transmission_line_pred_ply.py \
  exp/transmission/two_stage_infer/result \
  --data-root data/transmission_line \
  --split test \
  --out exp/transmission/two_stage_infer/result_ply \
  --merge-out exp/transmission/two_stage_infer/two_stage_test_pred.ply \
  --color pred \
  --world-coord
```

### 8.2 导出 Stage1 4 类预测

```bash
/opt/conda/envs/pointcept/bin/python tools/export_transmission_line_stage1_pred_ply.py \
  exp/transmission/stage1_4cls_balance_w8_clean/result \
  --data-root data/transmission_line_stage1_4cls_balance \
  --split test \
  --out exp/transmission/stage1_4cls_balance_w8_clean/stage1_result_ply \
  --merge-out exp/transmission/stage1_4cls_balance_w8_clean/stage1_test_pred.ply \
  --color pred \
  --world-coord
```

Stage1 专用颜色：

| 类别 | 颜色 |
| --- | --- |
| ground | gray |
| tower_structure | red |
| line | blue |
| other | purple |

如果需要看预测对错，可以使用：

```bash
--color correct
```

其中绿色表示预测正确，红色表示预测错误。

## 9. 结果分析要点

当前两阶段结果中，`line` 和 `other` 指标较低，主要原因如下：

- `line`：Stage1 已有较多导线被分到 `ground`；另外如果 Stage2 覆盖 ROI 内所有点，可能会把部分已经预测为 `line` 的点改成 `tower / insulator / hengdan`。
- `other`：Stage1 中大量真实 `other` 被分成 `ground`，说明该类本身混杂且权重较低，Stage2 不会修复该问题。

改进方向：

- 推理融合时只允许 Stage2 覆盖 Stage1 的 `tower_structure` 点，避免误伤 `line`。
- 提高 Stage1 中 `line` 和 `other` 的损失权重。
- 对含 `line` 和 `other` 的 tile 进行更有针对性的过采样。
- 检查 `other` 标注是否过于混杂，必要时拆分类别或明确其语义。

## 10. 注意事项

- 不要把 `data/`、`exp/`、`pretrained/`、`.pth`、`.ply`、`.npy` 上传到普通 GitHub 仓库。
- 权重文件建议通过 GitHub Release、网盘或 Git LFS 管理。
- 两阶段推理时，大点云 tile 建议使用 `--fast-crop`。
- 推理中断后，不加 `--overwrite` 重新运行即可断点续跑。

## 11. 基础框架说明

本项目基于 Pointcept v1.5.1 扩展。Pointcept 原始项目提供了 PTv3、数据加载、训练器、测试器和通用语义分割框架。本仓库的主要新增内容集中在输电线路数据预处理、两阶段数据生成、两阶段推理、mIoU 评估和 PLY 导出脚本。
