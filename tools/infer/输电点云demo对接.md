# postprocess_tower_line_boxes.py 处理流程说明

本文档说明 `tools/infer/postprocess_tower_line_boxes.py` 的当前处理逻辑、坐标系约定、杆塔框/线框定义以及 JSON 输出格式。

## 1. 输入类别定义

脚本输入是语义分割后的 LAS/LAZ 点云，默认使用 `classification` 字段表示类别：

```text
0 = background / 背景
1 = tower / 杆塔
2 = line / 导线
3 = insulator / 绝缘子
```

这些类别可以通过命令参数修改：

```text
--background-class
--tower-class
--line-class
--insulator-class
```

## 2. 总体处理流程

脚本主流程如下：

```text
读取 LAS/LAZ
  -> 读取点坐标和 classification
  -> 对杆塔点进行体素连通域聚类
  -> 清理明显的杆塔误检组件
  -> 恢复可能被误删的杆塔底部小组件
  -> 合并同一物理杆塔的多个组件
  -> 按线路主方向给杆塔排序并编号
  -> 生成杆塔 OBB 框
  -> 生成相邻杆塔之间的线框 OBB
  -> 可选：拟合每个档距内的单根导线
  -> 可选：向 LAS 追加框边缘采样点
  -> 输出 LAS、调试 report JSON、标准 OBB JSON
```

## 3. 杆塔点聚类

脚本首先取出所有 `classification == tower_class` 的点，默认即 `classification == 1`。

聚类方式：

1. 使用 `--tower-voxel-size` 对杆塔点体素化，默认体素大小为 `0.50m`。
2. 使用 3D 连通域算法连接相邻体素。
3. 连通方式由 `--tower-connectivity` 控制，支持 `6`、`18`、`26`，默认 `26`。

每个杆塔组件会统计：

```text
id
point_count
bbox_min
bbox_max
size
center
keep
remove_reason
```

## 4. 杆塔误检清理

每个杆塔组件会根据参数进行过滤。基础过滤包括：

```text
--min-tower-points          点数太少则删除
--min-tower-height          高度太低则删除
--min-tower-xy-size         XY 尺寸太小则删除
--min-tower-height-above-ground  离地高度不足则删除
```

可选过滤包括：

```text
--require-line-near-tower              要求杆塔附近有导线点
--require-line-through-tower           要求导线穿过杆塔上部区域
--require-line-touch-tower             要求杆塔上部点与导线点三维接近
--require-line-inside-tower            要求杆塔 OBB 内部有足够导线点
--require-continuous-line-inside-tower 要求杆塔内部导线点连续
--require-side-line-near-tower         要求杆塔侧面附近有导线点
--require-insulator-near-tower         要求杆塔附近有绝缘子点
--require-insulator-line-bridge        要求绝缘子点同时靠近导线点
--remove-bare-pole-towers              删除上部没有横向展开的光杆误检
--require-connected-line-span          只保留连接到有效档距的杆塔
```

被删除的杆塔点会被改为 `background_class`，默认是 `0`。

## 5. 杆塔底部恢复

为了避免严格过滤导致杆塔底部缺失，脚本默认会尝试恢复低矮的小组件。

恢复逻辑：

```text
如果一个被过滤的小组件位于已保留杆塔 XY 范围附近，
并且高度位于杆塔底部附近，
则将其恢复为杆塔组件。
```

相关参数：

```text
--no-recover-tower-base      关闭底部恢复
--recover-base-xy-margin     杆塔 XY 范围外扩距离
--recover-base-z-margin      杆塔底部高度容差
--recover-base-min-points    可恢复组件的最少点数
```

## 6. 物理杆塔合并和排序

真实杆塔可能被聚类成多个组件。脚本会使用 `--merge-tower-xy-radius` 按 XY 中心距离合并组件。

合并后，每个物理杆塔包含：

```text
id
component_ids
point_count
bbox_min
bbox_max
center
size
```

随后脚本会根据所有杆塔中心点的 XY 主方向排序，并生成：

```text
tower_no
tower_1, tower_2, ...
杆塔1, 杆塔2, ...
```

如果需要反向编号，可以使用：

```text
--reverse-tower-order
```

## 7. 杆塔框定义

杆塔框是旋转 3D OBB，不是普通 XYZ 轴对齐框。

生成逻辑：

1. 取该物理杆塔包含的所有杆塔点。
2. 在 XY 平面上做 PCA。
3. 使用 PCA 主方向作为杆塔框的局部 X 轴。
4. 与局部 X 轴垂直的水平向量作为局部 Y 轴。
5. LAS Z+ 作为局部 Z 轴。
6. 将杆塔点投影到该局部坐标系，取 min/max。
7. 每个方向外扩 `--tower-box-margin`，默认 `1.0m`。

杆塔框尺寸定义：

```text
local_x_length = 杆塔 PCA 主方向尺寸
local_y_width  = 杆塔侧向尺寸
local_z_height = 杆塔高度
```

## 8. 线框定义

线框表示相邻两个杆塔之间的整体档距框，不是每根导线一个框。

例如：

```text
杆塔1 - 杆塔2 -> 一个 line_span 框
杆塔2 - 杆塔3 -> 一个 line_span 框
```

选取线点时，脚本只使用 `classification == line_class` 的点，默认即 `classification == 2`。

线点选择条件：

```text
点位于两个相邻杆塔框之间
点到两个杆塔中心连线的 XY 垂直距离 <= --line-corridor-width
选中点数 >= --min-span-line-points
```

默认 `--line-box-mode oriented`，即生成旋转 OBB。内部计算时：

```text
along = 两个杆塔中心连线方向
side  = 垂直档距方向，即横跨多根导线方向
z     = 高度方向
```

线框范围：

```text
along 范围：左杆塔框出口到右杆塔框入口，中间扣除 --line-tower-gap
side 范围：导线点侧向分布的百分位范围，再加 --line-box-margin
z 范围：导线点高度分布的百分位范围，再加 --line-box-margin
```

相关参数：

```text
--line-box-mode oriented|axis
--line-corridor-width
--line-box-margin
--line-fit-percentile
--line-min-box-width
--line-min-box-height
--line-tower-gap
--min-span-line-points
```

标准 JSON 导出时，线框尺寸顺序为：

```text
sx = 横跨线路方向尺寸，即多根导线左右展开的距离
sy = 两个杆塔之间的档距方向尺寸
sz = 高度方向尺寸
```

## 9. 坐标系约定

当前脚本输出采用 LAS/UTM 坐标，单位为米。

当前约定：

```text
LAS X+ = 正北方向 / north / 0°
LAS Y+ = 侧向方向
LAS Z+ = 向上 / up
```

这是右手坐标系：

```text
X × Y = Z
```

注意：这是当前脚本导出 JSON OBB 的旋转基准约定。也就是说，当 OBB 的 local X 轴指向 LAS X+ 时，它被认为是朝正北。

## 10. JSON OBB 输出格式

标准 OBB JSON 由 `--combined-box-report` 输出，包含杆塔框和线框。

每个框的核心格式：

```text
obb = [cx, cy, cz, sx, sy, sz, qx, qy, qz, qw]
```

字段含义：

```text
cx, cy, cz = OBB 中心点坐标
sx, sy, sz = OBB 在局部坐标系下的尺寸
qx, qy, qz, qw = 旋转四元数
```

默认情况下，`obb[0:3]` 是相对坐标：

```text
obb[0:3] = center - las_origin
```

全局信息保存在：

```text
obb_global.lat_lng_alt = OBB 中心点的 LAS/UTM 全局坐标
obb_global.las_origin  = 相对坐标原点
obb_global.extent      = 尺寸
obb_global.rotation    = 四元数
obb_global.extent_order = 尺寸顺序说明
```

注意：字段名 `lat_lng_alt` 是历史命名，在这里实际表示 LAS/UTM 的 XYZ 坐标，不是经纬度。

## 11. OBB 原点选择逻辑

`--obb-origin` 控制 `obb[0:3]` 使用哪个相对坐标原点。

可选值：

```text
las-offset = 使用 LAS header offsets，默认值
las-min    = 使用 LAS header mins
zero       = 使用 [0, 0, 0]
custom     = 使用 --obb-origin-xyz 指定的自定义原点
```

当前默认 `las-offset` 的具体逻辑是：

```text
1. 先取 las.header.offsets 作为 LAS 原点。
2. 如果 offset 的 x/y/z 都在 header.mins 到 header.maxs 范围内，继续使用该 offset。
3. 如果 offset 不在点云范围内，则改用 (header.mins + header.maxs) / 2.0。
```

也就是说，默认情况下脚本会优先使用 LAS 文件自身的 header offset；如果该 offset 明显不适合作为当前点云的相对原点，就退回到 header min/max 的中心点。

这样可以避免输出 JSON 中 `obb[0:3]` 因为 LAS offset 远离当前点云而出现过大的相对坐标。

## 12. 旋转定义

当前脚本使用 `--json-box-orientation fitted` 时，会保留真实拟合朝向。

旋转四元数表示：

```text
从“正北标准框”旋转到当前 OBB 局部坐标系
```

当前正北标准框定义为：

```text
local X = LAS X+，即正北方向
local Y = LAS Y+，即侧向方向
local Z = LAS Z+，即向上方向
```

因此，当框朝正北时：

```text
rotation = [0, 0, 0, 1]
```

如果使用：

```text
--json-box-orientation north
```

则 JSON 会输出不旋转的正北轴对齐框，旋转固定为：

```text
[0, 0, 0, 1]
```

如果需要保留真实 OBB 朝向，应使用：

```text
--json-box-orientation fitted
```

## 13. LAS 输出中的框边缘点

如果没有设置 `--no-append-box-points`，脚本会将杆塔框和线框的 12 条边采样成点，并追加到输出 LAS。

追加的框边缘点：

```text
classification = 31
```

颜色：

```text
杆塔框 = 洋红色 [65535, 0, 65535]
线框   = 青色   [0, 65535, 65535]
```

边缘点采样间距由以下参数控制：

```text
--edge-step
```

## 14. 可选导线拟合

脚本还支持在每个档距内拟合单根导线。

导线拟合逻辑大致为：

```text
对每个 line_span 内的导线点建立局部坐标
沿档距方向分箱
在每个横截面里聚类导线点
跨分箱跟踪导线中心
对每根导线拟合 side 和 z 关于 along 的多项式
输出每根导线的 polyline JSON
可选追加拟合导线点到 LAS
```

导线拟合点的类别：

```text
classification = 30
```

可用以下参数关闭：

```text
--no-fit-conductors
--no-append-conductor-fit
```

## 15. 推荐命令

保留真实 OBB 朝向的推荐命令：

```bash
python tools/infer/postprocess_tower_line_boxes.py \
  --input your_segmented.las \
  --output your_output_with_boxes.las \
  --report your_report.json \
  --combined-box-report your_boxes.json \
  --obb-origin las-offset \
  --json-box-orientation fitted \
  --line-box-mode oriented \
  --tower-box-margin 1.0 \
  --line-box-margin 1.0 \
  --overwrite
```

如果需要更严格的杆塔过滤，可以按数据情况增加：

```bash
--require-line-near-tower \
--tower-line-radius 18.0 \
--min-line-points-near-tower 100 \
--require-insulator-line-bridge \
--tower-insulator-xy-margin 6.0 \
--tower-insulator-z-margin 6.0 \
--insulator-line-radius 1.2
```

## 16. 对接说明

给下游系统解析时，需要明确：

```text
1. 坐标单位是米。
2. 当前 JSON OBB 以 LAS X+ 作为正北 0°。
3. 正北时 quaternion = [0, 0, 0, 1]。
4. obb[0:3] 默认是相对 las_origin 的坐标。
5. obb_global.lat_lng_alt 是中心点全局 LAS/UTM 坐标。
6. 杆塔框 sx/sy/sz = PCA 主方向尺寸 / 侧向尺寸 / 高度。
7. 线框 sx/sy/sz = 横跨线路尺寸 / 档距方向尺寸 / 高度。
```
