# 输电线路模型和工具说明

这个仓库里和输电线路相关的模型大致分成两条路线：一阶段推理和两阶段推理。它们不是因为 Pointcept 框架不同，而是因为任务目标和类别处理方式不同。

## 一阶段版本

一阶段版本只用一个模型直接在原始 tile 上输出目标类别。你当前训练日志里的 `ptv3-4cls-ins-oversample_v1` 属于这个路线。

常见配置：

- `configs/transmission_line/semseg-pt-v3m1-4cls-ins-oversample.py`：4 类一阶段模型，类别是 `ground/tower/line/insulator`。
- `configs/transmission_line/semseg-pt-v3m1-0-base.py`、`semseg-pt-v3m1-0-base_v2.py`：早期 6 类基础配置。

适用场景：

- 想要流程简单，训练、测试、导出都走 Pointcept 标准方式。
- 类别定义已经能覆盖你当前需求，比如只关心 ground、tower、line、insulator 四类。
- 小目标精分压力不太大，或者已经通过过采样改善了绝缘子。

常用命令：

```bash
python tools/train.py --config-file configs/transmission_line/semseg-pt-v3m1-4cls-ins-oversample.py --num-gpus 1
python tools/test.py --config-file exp/transmission_line/ptv3-4cls-ins-oversample_v2/config.py --num-gpus 1 --options save_path=exp/transmission_line/ptv3-4cls-ins-oversample_v2/test_model_best weight=exp/transmission_line/ptv3-4cls-ins-oversample_v2/model/model_best.pth
```

## 两阶段版本

两阶段版本用两个模型协作：

1. Stage-1：先在完整 tile 上做粗分，输出 `ground/tower_structure/line/other`。这里的 `tower_structure` 把 `tower + insulator + hengdan` 合并，目标是高召回地找到杆塔相关区域。
2. Stage-2：只在杆塔 ROI 内做细分，输出 `tower/insulator/hengdan/background`。
3. 合成：`tools/infer_transmission_line_two_stage.py` 把两个阶段结果合成最终 6 类：`ground/tower/line/insulator/hengdan/other`。

常见配置：

- `configs/transmission_line/semseg-pt-v3m1-stage1-4cls.py`：Stage-1 粗分模型。
- `configs/transmission_line/semseg-pt-v3m1-stage2-tower.py`：Stage-2 杆塔 ROI 精分模型。

适用场景：

- 原始 6 类里绝缘子、横担很小，直接一阶段训练容易被背景或塔身淹没。
- 你需要最终 6 类结果，尤其关心 `insulator` 和 `hengdan` 的可视化/指标。
- 可以接受流程更复杂：需要准备 Stage-1 数据、Stage-2 ROI 数据、分别训练两个权重，再用两阶段脚本合成。

## 工具分组

### 通用训练/测试

- `tools/train.py`：Pointcept 标准训练入口。所有一阶段、Stage-1、Stage-2 都通过它训练。
- `tools/test.py`：Pointcept 标准测试入口。输出 `result/*_pred.npy`、`test.log` 和指标。

### 一阶段数据和评估

- `tools/make_insulator_oversampled_train.py`：对一阶段 4 类训练集里含绝缘子的 tile 过采样。
- `tools/count_transmission_line_classes.py`：统计类别点数和比例，检查数据不均衡。
- `tools/evaluate_transmission_line_pred.py`：对已有 `*_pred.npy` 复算 mIoU/mAcc/allAcc。
- `tools/export_transmission_line_data_ply.py`：把原始 `.pth` 数据导出成 PLY，检查数据本身。
- `tools/export_transmission_line_pred_ply.py`：把一阶段或最终 6 类预测导出成 PLY。

### 两阶段数据准备

- `tools/make_stage1_4cls.py`：原始 6 类转 Stage-1 4 类。
- `tools/make_stage1_balanced_train.py`：Stage-1 train 过采样/均衡。
- `tools/make_stage2_tower_roi.py`：从原始 6 类生成 Stage-2 杆塔 ROI 数据。
- `tools/make_stage2_balanced_train.py`：Stage-2 train 过采样/均衡。
- `tools/make_stage2_insulator_centered.py`：额外生成绝缘子中心裁剪 ROI，增强 Stage-2 小目标。
- `tools/make_stage2_insulator_focus.py`：额外生成偏绝缘子/横担关注的 ROI 数据。

### 两阶段训练/推理/可视化

- `tools/train_transmission_line_two_stage.py`：顺序训练 Stage-1 和 Stage-2。
- `tools/infer_transmission_line_two_stage.py`：加载两个阶段权重，合成最终 6 类预测。
- `tools/export_transmission_line_stage1_pred_ply.py`：可视化 Stage-1 粗分 4 类预测。
- `tools/export_transmission_line_stage2_pred_ply.py`：可视化 Stage-2 ROI 4 类预测。

### S3DIS 辅助工具

- `tools/export_s3dis_pred_ply.py`：S3DIS room 预测转 PLY。
- `tools/export_s3dis_area_pred_ply.py`：S3DIS Area 级预测合并转 PLY。

这两个脚本和输电线路模型没有直接关系，是原 Pointcept/S3DIS 可视化辅助工具。
