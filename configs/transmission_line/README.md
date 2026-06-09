# 输电线路点云语义分割流程

本文记录如何将按类别拆分的 LAS 点云转换为 Pointcept 数据，使用 Point Transformer V3 (PTv3) 训练六类语义分割模型，并进行验证和测试。

## 1. 任务与目录

| 原标签 | 训练标签 | 类别 |
| --- | --- | --- |
| 0 | 0 | ground |
| 1 | 1 | tower |
| 2 | 2 | line |
| 3 | 3 | insulator |
| 4 | 4 | hengdan |
| 5、6 | 5 | other |

原始数据应为以下布局，每个 `.las` 文件只包含一个类别的点：

```text
<dataset-root>/
  scene01/
    0_ground.las
    1_tower.las
    2_line.las
    ...
  scene56/
    ...
```

流程按场景号固定划分，防止同一场景附近的瓦片同时进入训练和测试：

| 划分 | 场景 |
| --- | --- |
| train | scene01 - scene40 |
| val | scene41 - scene48 |
| test | scene49 - scene56 |

## 2. 环境

本机已有可运行环境：

```bash
export PROJECT=/24085403037/24085403037/PointTransformerV3/Pointcept-v1.5.1
export PYTHON=/opt/conda/envs/pointcept/bin/python
cd "$PROJECT"
$PYTHON -c "import torch, addict, pointops, spconv; print(torch.cuda.is_available())"
```

检查时环境为 PyTorch `2.1.0+cu118`，可见一张 RTX 3090。

## 3. LAS 预处理

转换脚本为 `pointcept/datasets/preprocessing/transmission_line/preprocess_transmission_line.py`。

```bash
  python pointcept/datasets/preprocessing/transmission_line/preprocess_transmission_line.py \
    --dataset-root /24085403037/24085403037/shixi/dataset/dianyun_cloud \
    --output-root data/transmission_line \
    --voxel-size 0 \
    --tile-size 40 \
    --train-tile-stride 20 \
    --eval-tile-stride 40 \
    --min-points 1024 \
    --overwrite
```

处理逻辑：

1. 读取每个场景按类别拆分的 LAS 文件，将标签 `6` 合并到 `other`。
2. 以场景最小坐标为原点，避免大地坐标降低浮点计算精度。
3. 按 `20 m x 20 m` 切分 XY 瓦片，以限制训练显存并提供局部样本。
4. 按类别执行 `0.02 m` 体素采样，避免少数类在体素冲突中被地面覆盖。
5. 跳过少于 `1024` 点的边缘瓦片；首次数据中已发现仅 `1-4` 个点的瓦片。
6. 保存包含 `coord`、`color`、`semantic_gt` 的 `.pth` 文件。

当前已有数据最初是在加入 `--min-points` 前生成的，其中仅 `1` 个点的瓦片会使 PTv3 无法完成序列化。2026-05-27 已就地修复：所有少于 `1024` 点的旧瓦片已移动并保留在 `data/transmission_line/_sparse_skipped/`，明细见 `legacy_sparse_filter.json`。从原始 LAS 重做数据时，使用上面的 `--min-points` 命令即可得到同类过滤结果。

| 划分 | 场景数 | 当前瓦片数 | 过滤前保留点数 |
| --- | ---: | ---: | ---: |
| train | 40 | 940 | 188,626,673 |
| val | 8 | 201 | 32,697,803 |
| test | 8 | 140 | 36,642,740 |

当前训练集类别分布高度不平衡：

| ground | tower | line | insulator | hengdan | other |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 174,721,243 | 5,007,527 | 4,070,892 | 771,248 | 2,528,783 | 1,526,980 |

因此配置中的交叉熵权重降低 `ground` 贡献，并提高 `insulator` 贡献。

## 4. 配置选择

训练配置为 `configs/transmission_line/semseg-pt-v3m1-0-base.py`。

| 配置 | 值 | 原因 |
| --- | --- | --- |
| 模型 | PT-v3m1 semantic segmentor | 任务需要输出逐点语义类别 |
| 输入 | XYZ + RGB，6 维 | 结合几何形态和颜色信息 |
| `grid_size` | 0.02 m | 与预处理体素分辨率一致 |
| `point_max` | 100000 | 控制注意力显存，先保证稳定启动 |
| `batch_size` | 1 | 约 57k 点单批反向已使用约 6.0 GiB 显存 |
| `enable_amp` | `False` | 首次 AMP 训练在约 0.00194 学习率后出现 NaN 并触发 CUDA 半精度错误 |
| `grad_clip_norm` | `1.0` | 裁剪异常大的梯度更新，降低恢复训练再次发散的风险 |
| 最大学习率 | 0.001 | 从不稳定运行的 0.002 下调，避免直接复用高峰更新 |
| CE 权重 | `[0.2, 1, 1, 2, 1, 1]` | 缓解地面占多数、绝缘子稀少的问题 |
| `epoch` / `eval_epoch` | 100 / 10 | 总计 100 次数据遍历，每 10 次验证一次 |

## 5. 启动训练

正式训练前可先确认配置可解析：

```bash
$PYTHON -c "from pointcept.utils.config import Config; c=Config.fromfile('configs/transmission_line/semseg-pt-v3m1-0-base.py'); print(c.model.num_classes, c.model.backbone.in_channels, c.data.train.split)"
```

2026-05-27 已完成真实训练样本前向/反向检查：普通瓦片变换后约 `57k` 点，损失能够反向传播，峰值显存约 `6.0 GiB`；旧数据中的 1 点边缘瓦片会触发 PTv3 序列化错误。这也是预处理增加 `--min-points 1024` 且未在修复前直接启动训练的原因。

确认训练目录中已无稀疏瓦片后，首次完整训练可以使用一个 GPU 启动。若已经出现本文记录的 AMP 崩溃，不要用 `-r true` 恢复该实验，因为这会继承原优化器和调度器状态；应从有限的 `model_best.pth` 只加载权重，并使用新的稳定配置启动新实验：

```bash
sh scripts/train.sh \
  -p /opt/conda/envs/pointcept/bin/python \
  -d transmission_line \
  -c semseg-pt-v3m1-0-base \
  -n ptv3-fp32-stable \
  -w exp/transmission_line/ptv3-base/model/model_best.pth \
  -g 1
```

产物写入 `exp/transmission_line/ptv3-fp32-stable/`，其中应包括 `train.log`、复制的 `config.py` 以及 `model/model_last.pth`、`model/model_best.pth`。

只有新的 `ptv3-fp32-stable` 实验正常保存检查点后，才从该实验恢复：

```bash
sh scripts/train.sh \
  -p /opt/conda/envs/pointcept/bin/python \
  -d transmission_line \
  -c semseg-pt-v3m1-0-base \
  -n ptv3-fp32-stable \
  -g 1 \
  -r true
```

## 6. 测试与结果判断

使用验证过程中表现最好的权重运行测试集：

```bash
sh scripts/test.sh \
  -p /opt/conda/envs/pointcept/bin/python \
  -d transmission_line \
  -n ptv3-fp32-stable \
  -w model_best \
  -g 1
```

重点检查六类 IoU，不能只看总体准确率。地面点数量极大，即使线路或绝缘子预测较差，总体准确率仍可能看起来较高。

| 现象 | 处理方向 |
| --- | --- |
| CUDA OOM | 将 `point_max` 降至 `80000` |
| 显存余量明显且训练过慢 | 先提高 `point_max`，稳定后再尝试 `batch_size=2` |
| ground 高而 line/insulator IoU 低 | 提高稀有类权重，或对含线路瓦片过采样 |
| val 明显好于 test | 检查不同场景的设备类型及标注分布差异 |

测试完成后保留当次 `config.py` 与 `metadata.json`，以保证结果可复现。

# 数据预处理
### Pointcept-V3需要LAS点云

### 步骤1：读取原始场景数据
    每个scene目录代表一个输电线路场景，目录内的LAS文件按类别分别存放，例如
      scene01/
       0_ground.las
       1_tower.las
       2_line.las
       3_insulator.las
       4_hengdan.las
       5_other.las
### 步骤2:统一类别标签
     训练使用 6 类：
     ┌──────┬───────────────────┐
     │ 标签 │ 类别              │
     ├──────┼───────────────────┤
     │    0 │ ground，地面      │
     │    1 │ tower，杆塔       │
     │    2 │ line，导线        │
     │    3 │ insulator，绝缘子 │
     │    4 │ hengdan，横担     │
     │    5 │ other，其他       │
     └──────┴───────────────────┘
### 步骤3：合并同一场景内的分类点云
    对每个场景读取各类别LAS文件，提取：
      三维坐标XYZ；
      颜色RGB；
      对应语义标签；
    各类别随后组合为一个带逐点标签的场景点云；
### 步骤4：坐标局部化
    每个场景以自身最小坐标作为原点，将坐标转换为局部坐标；
    local_coord = original_coord - scene_origin
    为什么：这样可以避免直接使用较大的地理坐标，减少浮点精度和训练稳定性问题；
### 步骤5：按空间切成瓦片
    每个场景沿XY平面切成20m × 20m的瓦片，步长也是20m。
    为什么？原始的完整线路场景点数太大，不能直接作为一个训练样本。切片后可以
    -控制显存消耗；
    -生成更多局部训练样本；
    -让模型学习局部杆塔、导线以及绝缘子结构。
### 步骤6：按类别进行体素采样
    每个类别单独以0.02m的体素尺寸降采样。
    采用“按类别采样而不是先混合后采样，是为了防止同一体素内数量很多的地面点覆盖导线、绝缘子等少数类别点”。
### 步骤7：过滤小瓦片
    少于1024个点瓦片不会进入训练数据中； 为啥使用1024阈值？排除明显无结构信息的边缘碎片，这个值还需通过实验论证
    因为边缘瓦片可能只有几个点，缺乏结构信息，而且实测只有一个点的瓦片会导致PTv3在序列化阶段报错。
### 步骤8：按场景划分数据集
     不是随机划分瓦片，而是按完整场景划分：

     ┌────────┬───────────────────┐
     │ 数据集 │ 场景              │
     ├────────┼───────────────────┤
     │ 训练集 │ scene01 - scene40 │
     │ 验证集 │ scene41 - scene48 │
     │ 测试集 │ scene49 - scene56 │
     └────────┴───────────────────┘

     这样避免同一个场景的相邻瓦片同时进入训练集和测试集，造成数据泄漏。
### 步骤9： 保存为 Pointcept 格式
     每个有效瓦片保存成一个 .pth 文件，主要字段是：

     {
         "coord": ...,       # N x 3 局部三维坐标
         "color": ...,       # N x 3 RGB
         "semantic_gt": ..., # N 个逐点语义标签
         "origin": ...,      # 场景原始坐标原点
         "scene": ...        # 所属场景名称
     }

### 步骤10😂 送入 PTv3 训练
     训练时模型输入使用：

     XYZ + RGB = 6 维特征

     配置中进一步进行中心化、轻度旋转/缩放/翻转/扰动，以及最大点数裁剪，再输入 Point Transformer V3 完成六类逐点分割。

  当前过滤后的有效样本数为：

  ┌────────┬────────────┐
  │ 数据集 │ 有效瓦片数 │
  ├────────┼────────────┤
  │ train  │        940 │
  │ val    │        201 │
  │ test   │        140 │
  └────────┴────────────┘
python tools/train.py \
  --config-file configs/transmission_line/semseg-pt-v3m1-0-base_v2.py \
  --num-gpus 1 \
  --options save_path=exp/transmission/ptv3_v2_sanity