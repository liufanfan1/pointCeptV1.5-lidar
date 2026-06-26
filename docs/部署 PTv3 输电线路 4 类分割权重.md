# Windows 笔记本部署 PTv3 输电线路 4 类分割权重

### 背景：目标是在一台带 NVIDIA 显卡的 Windows 笔记本上运行本仓库的输电线路点云分割推理。

目标权重：

```text
exp/transmission_line/ptv3-4cls-ins-oversample_v2/model/model_best.pth
```

对应类别：

| ID | 类别 | 颜色 |
| --- | --- | --- |
| 0 | ground | 灰色 |
| 1 | tower | 红色 |
| 2 | line | 蓝色 |
| 3 | insulator | 黄色 |

推荐结论：

- **最稳方案：Windows + WSL2 Ubuntu + NVIDIA GPU**。
- **原生 Windows 方案可以尝试，但更容易卡在 `spconv`、`torch_scatter`、`flash-attn` 这些 CUDA 依赖上**。
- 如果只是部署推理，不建议在 Windows 上重新训练。

---

## 1. 原生 Windows 部署方案

如果你不想用 WSL2，可以尝试原生 Windows。这个方案更容易踩依赖坑。

### 3.1 安装软件

需要安装：

1. NVIDIA 驱动。
2. Miniconda for Windows。
3. Git for Windows。
4. Visual Studio Build Tools 2022。

Visual Studio Build Tools 安装时勾选：

```text
Desktop development with C++
MSVC
Windows 10/11 SDK
```

### 3.2 创建环境

打开 Anaconda Prompt：

```bat
conda create -n pointcept python=3.8 -y
conda activate pointcept
```

安装 PyTorch：

```bat
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

检查：

```bat
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### 3.3 安装依赖

```bat
pip install numpy scipy addict yapf timm laspy
pip install spconv-cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

如果 `torch-scatter` 或 `spconv-cu118` 安装失败，优先改用 WSL2。

原生 Windows 上不建议安装 `flash-attn`。推理时使用：

```text
--disable-flash
```

### 3.4 运行推理

假设项目在：

```text
D:\pointCeptV1.5-lidar
```

输入 LAS：

```text
D:\data\input.las
```

运行：

```bat
cd /d D:\pointCeptV1.5-lidar
conda activate pointcept

python tools\infer_las_semseg.py ^
  --input D:\data\input.las ^
  --output D:\data\input_pred.las ^
  --merge-mode halo ^
  --tile-size 40 ^
  --tile-stride 40 ^
  --context-margin 10 ^
  --pre-voxel-size 0.05 ^
  --fragment-batch-size 1 ^
  --min-tile-points 1024 ^
  --disable-flash ^
  --overwrite
```

---

## 4. 推理参数怎么选

### 4.1 最稳但较慢

```bash
--merge-mode plain
--tile-size 40
--tile-stride 40
--fragment-batch-size 1
--disable-flash
```

不加 `--pre-voxel-size`，最接近原始测试流程，但大 LAS 会慢。

### 4.2 推荐部署参数

```bash
--merge-mode halo
--tile-size 40
--tile-stride 40
--context-margin 10
--pre-voxel-size 0.05
--fragment-batch-size 1
--disable-flash
```

这是速度和效果的折中方案。

### 4.3 更稳的边界融合

```bash
--merge-mode overlap
--tile-size 40
--tile-stride 20
--pre-voxel-size 0.05
--fragment-batch-size 1
--disable-flash
```

这个会更慢，因为重叠区域会重复推理。

### 4.4 显存够时提速

可以尝试：

```bash
--fragment-batch-size 2
```

如果报显存不足，再改回：

```bash
--fragment-batch-size 1
```

`fragment-batch-size` 通常不明显影响精度，主要影响速度和显存。

---

## 5. 常见问题

### 5.1 `ModuleNotFoundError: No module named 'spconv'`

没有安装 `spconv`：

```bash
pip install spconv-cu118
```

如果 Windows 原生安装失败，建议切换 WSL2。

### 5.2 `ModuleNotFoundError: No module named 'torch_scatter'`

安装：

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

注意 PyTorch 和 CUDA 版本必须对应。

### 5.3 `Make sure flash_attn is installed`

说明配置里启用了 flash attention，但环境没有 `flash-attn`。

推理命令加：

```bash
--disable-flash
```

### 5.4 `torch.cuda.is_available()` 是 False

检查：

```bash
nvidia-smi
```

如果 `nvidia-smi` 正常，但 PyTorch 不可用，通常是安装了 CPU 版 PyTorch。重新安装 CUDA 版：

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### 5.5 推理很慢

优先使用：

```bash
--pre-voxel-size 0.05
--merge-mode halo
--context-margin 10
```

大 LAS 文件不要放在 WSL 的 `/mnt/c/...` 下推理，建议放在 WSL Linux 文件系统里，例如：

```text
~/data/
```

### 5.6 CloudCompare 看不到分割颜色

确认你没有加：

```bash
--no-colorize
```

新版推理脚本默认会把预测类别写成 RGB。

CloudCompare 中选择：

```text
RGB colors
```

或查看：

```text
Scalar fields -> Classification
```

---

## 6. 推荐目录结构

WSL 推荐：

```text
~/projects/pointCeptV1.5-lidar/
~/data/input.las
~/data/input_pred.las
```

Windows 原生推荐：

```text
D:\pointCeptV1.5-lidar\
D:\data\input.las
D:\data\input_pred.las
```

路径里尽量不要有中文、空格、括号。

---

## 7. 最小验证流程

1. 检查 GPU：

```bash
nvidia-smi
```

2. 检查 PyTorch：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

3. 进入仓库：

```bash
cd ~/projects/pointCeptV1.5-lidar
```

4. 跑推理：

```bash
python tools/infer_las_semseg.py \
  --input ~/data/input.las \
  --output ~/data/input_pred.las \
  --merge-mode halo \
  --tile-size 40 \
  --tile-stride 40 \
  --context-margin 10 \
  --pre-voxel-size 0.05 \
  --fragment-batch-size 1 \
  --min-tile-points 1024 \
  --disable-flash \
  --overwrite
```

5. 用 CloudCompare 打开：

```text
input_pred.las
```

能看到灰、红、蓝、黄四色点云，就说明部署成功。
