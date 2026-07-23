# 分割点云中的绝缘子与横担关键点提取说明

本文档说明脚本：

```text
tools/insulator_hengdan/extract_segmented_insulator_crossarm_keypoints.py
```

## 1. 脚本作用

该脚本从已经完成语义分割的输电线路 LAS/LAZ 点云中提取：

- 独立杆塔实例；
- 每座杆塔上部的横担层；
- 每层横担的左端点、中心点、右端点；
- 每串绝缘子的挂载端、中点和自由端；
- 绝缘子与具体杆塔、横担层及横担左右侧的对应关系。

最终只输出一个 JSON 文件，不输出新的 LAS/LAZ，也不会修改输入点云。

## 2. 输入分类约定

输入 LAS/LAZ 必须包含 `classification` 字段，默认类别约定为：

| classification | 类别 | 脚本中的用途 |
|---:|---|---|
| `1` | 杆塔 | 聚类杆塔实例并检测横担 |
| `2` | 导线 | 当前只统计数量，不参与横担和绝缘子计算 |
| `3` | 绝缘子 | 聚类绝缘子实例并提取端点 |

类别编号可以通过下面的参数修改：

```bash
--tower-class 1
--line-class 2
--insulator-class 3
```

脚本读取的是 `las.x`、`las.y`、`las.z`，因此 JSON 中保存的是输入 LAS 解码后的全局 XYZ 坐标，不是减去 LAS offset 后的局部坐标。

## 3. 总体处理流程

```text
读取 LAS/LAZ 和 classification
  -> 分离杆塔点、导线点和绝缘子点
  -> 使用原始绝缘子点近邻图提取绝缘子实例
  -> 使用杆塔体素连通域提取独立杆塔
  -> 扫描每座杆塔上部的水平切片
  -> 从切片宽度曲线中寻找横担峰值
  -> 提取每层横担的左、中、右关键点
  -> 将每串绝缘子唯一挂载到最近横担及其左侧或右侧
  -> 对杆塔、横担和绝缘子排序编号
  -> 输出嵌套 JSON
```

## 4. 绝缘子实例提取

### 4.1 原始点近邻聚类

脚本对全部 `class=3` 原始点建立 `cKDTree`，不对绝缘子点做体素化。

对每个绝缘子点，最多查询默认 `16` 个邻居，只保留三维距离不超过默认 `0.20m` 的邻接关系，然后通过无向图连通域得到绝缘子候选实例。

主要参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--insulator-connect-radius` | `0.20` | 绝缘子原始点之间的最大三维连接距离，单位 m |
| `--insulator-neighbors` | `16` | 每个点最多查询的近邻数 |
| `--min-insulator-points` | `30` | 一个绝缘子实例至少包含的原始点数 |
| `--min-insulator-height` | `0.0` | 绝缘子实例最小 Z 高度，`0` 表示不限制 |

旧参数名 `--insulator-voxel-size` 仍能使用，但它只是 `--insulator-connect-radius` 的别名，当前代码不会对绝缘子点执行体素化。

参数影响：

- 连接半径太小：同一串绝缘子可能断成多个实例；
- 连接半径太大：相邻绝缘子可能粘连成一个实例；
- 邻居数太小：高密度点云中可能出现不必要的断裂；
- 最小点数太大：稀疏绝缘子容易被过滤；
- 最小点数太小：绝缘子噪点容易被保留。

### 4.2 PCA 主轴与端点

每个有效绝缘子实例使用三维 PCA 估计长度主方向：

1. 计算实例点云均值；
2. 对去中心化坐标执行 SVD；
3. 取第一主成分作为绝缘子的主轴；
4. 固定主轴符号，保证相同输入多次运行时结果稳定；
5. 将所有绝缘子点投影到主轴；
6. 默认使用投影值的第 `2%` 和第 `98%` 分位数作为两端目标位置；
7. 将两个目标位置分别吸附到最近的真实绝缘子点。

```bash
--insulator-endpoint-percentile 2.0
```

使用分位数而不是最小值和最大值，可以减弱少量离群点对端点的影响。输出端点仍然来自真实点云，不是凭空生成的拟合点。

## 5. 杆塔实例提取

### 5.1 体素连通域

脚本对 `class=1` 杆塔点进行三维体素化：

1. 默认体素边长为 `0.75m`；
2. 使用 26 邻域连接相邻非空体素；
3. 使用并查集得到杆塔连通域；
4. 将体素标签映射回原始杆塔点；
5. 按点数和高度过滤过小的杆塔组件。

主要参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--tower-voxel-size` | `0.75` | 杆塔聚类体素边长，单位 m |
| `--min-tower-points` | `200` | 一个杆塔实例最少原始点数 |
| `--min-tower-height` | `4.0` | 一个杆塔实例最小 Z 高度，单位 m |

体素过小可能把一座杆塔拆成多个实例；体素过大可能把相邻物体或误分点合并进杆塔。

### 5.2 杆塔编号

每个有效杆塔的中心使用杆塔点 XYZ 中位数计算。杆塔按以下顺序排序：

```text
先按全局 X 从小到大，再按全局 Y 从小到大
```

排序后依次命名为 `杆塔1`、`杆塔2` 等。

注意：该规则不是沿输电线路主方向排序。如果线路方向与全局 X 轴差异较大，编号顺序可能与 CloudCompare 当前视角下的“从左到右”不一致。

## 6. 横担提取

### 6.1 扫描范围

横担检测只使用杆塔点，不使用导线点和绝缘子点辅助判断。

脚本默认只扫描杆塔高度上部 `45%` 至塔顶的范围：

```text
搜索起始高度 = 杆塔最低 Z + 杆塔高度 x 0.45
```

然后以默认 `0.5m` 的步长移动水平切片，每个切片使用默认 `1.2m` 厚度，即扫描中心上下各约 `0.6m` 的杆塔点。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--crossarm-min-height-ratio` | `0.45` | 横担搜索起始相对高度 |
| `--scan-z-step` | `0.5` | 相邻扫描中心的 Z 间隔，单位 m |
| `--scan-z-window` | `1.2` | 单个水平切片的总厚度，单位 m |
| `--min-crossarm-points` | `50` | 一个切片至少需要的杆塔点数 |

### 6.2 切片宽度测量

对每个满足点数要求的水平切片：

1. 使用切片点 XY 中位数作为水平中心；
2. 对 XY 坐标执行二维 PCA；
3. 最大方差方向定义为该层横担的 `side_axis`，即横担展开方向；
4. 与 `side_axis` 垂直的方向定义为 `along_axis`；
5. 将点投影到 `side_axis`；
6. 默认取第 `2%` 和第 `98%` 分位位置作为左右目标端；
7. 两个目标端分别吸附到最近的真实杆塔点；
8. 两端点中点作为横担中心点。

主要参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--min-crossarm-width` | `2.0` | 有效横担切片最小宽度，单位 m |
| `--endpoint-percentile` | `2.0` | 横担左右端点使用第 `2%/98%` 分位 |

### 6.3 从宽度曲线寻找横担层

每个扫描高度都会得到一个横担宽度，所有宽度共同构成“高度-宽度曲线”。脚本使用 `scipy.signal.find_peaks` 寻找局部宽度峰值。

峰值需要同时满足：

- 与相邻保留峰的 Z 间距至少为 `2.5m`；
- 相对周围切片的绝对突出宽度至少为 `0.5m`；
- 突出宽度与自身宽度的比例至少为 `0.25`；
- 横担宽度至少达到整座杆塔上部最宽切片的 `50%`；
- 横担宽度不能小于 `2.0m`。

主要参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--crossarm-min-z-separation` | `2.5` | 相邻横担层最小高度间隔，单位 m |
| `--crossarm-min-prominence` | `0.5` | 宽度峰最小绝对突出值，单位 m |
| `--crossarm-min-relative-prominence` | `0.25` | 宽度峰最小相对突出比例 |
| `--crossarm-min-width-ratio` | `0.5` | 横担宽度相对最宽切片的最小比例 |

为使塔顶边界处的横担也能形成峰值，宽度曲线两端会补零。搜索区间的第一个切片会被排除，避免把人为截取边界处的塔身宽度当成横担。

如果完全没有找到峰值候选，但存在满足宽度条件的切片，脚本会使用最宽切片作为兜底横担。若已经存在峰值候选，只是候选因为突出度不足被拒绝，则不会兜底，以避免把缓慢变宽的塔身误认为横担。

### 6.4 横担编号和左右方向

同一座杆塔的横担按照中心点 Z 从高到低排序，并依次编号：

```text
crossarm_id=1：最上层横担
crossarm_id=2：第二层横担
...
```

横担的左右由二维 PCA 主轴决定。代码会固定 `side_axis` 的符号以保证重复运行稳定：绝对值最大的全局轴分量被调整为正方向。

因此：

- `left_endpoint` 对应 `side_axis` 投影较小的一端；
- `right_endpoint` 对应 `side_axis` 投影较大的一端；
- 这里的“左/右”是稳定的几何编号，不一定等于用户当前观察视角中的屏幕左/右，也不固定表示东/西。

## 7. 绝缘子与横担挂载

### 7.1 确定挂载端和自由端

对每串绝缘子的两个 PCA 端点，分别计算其到每条横担三维线段的最近距离。

- 距离横担更近的绝缘子端点定义为 `endpoint_1_xyz`，即挂载端；
- 另一个端点定义为 `endpoint_2_xyz`，即自由端；
- 两个端点的中点定义为 `middle_point_xyz`。

只有绝缘子挂载端到横担线段的最近距离不大于默认 `3.0m` 时，该绝缘子才有资格挂载：

```bash
--insulator-attach-radius 3.0
```

每串绝缘子只会分配一次。如果它同时靠近多个横担，优先选择挂载距离最小的横担；距离相同时再按杆塔编号、横担高度和左右侧稳定选择。

### 7.2 确定横担左侧或右侧

绝缘子挂载端相对横担中心在 `side_axis` 上的投影决定左右侧：

```text
投影 <= 0：left_insulators
投影 > 0：right_insulators
```

如果挂载端恰好位于横担中部，则使用绝缘子中点的投影消除左右歧义。

### 7.3 同一端点内的绝缘子排序

同一横担同一侧挂载的绝缘子按以下顺序排列：

1. 绝缘子中点在横担局部 `along_axis` 上的投影，从小到大；
2. 投影相同时，按中点 Z 从高到低；
3. 再按全局 X、Y 从小到大保证稳定。

排序后在各自的 `left_endpoint.insulators` 或 `right_endpoint.insulators` 中从 `1` 开始生成 `insulator_id`。因此绝缘子编号只在“某一横担的某一侧”内部有效，不是整个文件的全局唯一编号。

### 7.4 朝下绝缘子过滤

脚本会计算从挂载端指向自由端的方向中，向下的垂直分量比例：

```text
downward_ratio = max(0, -方向Z / 绝缘子长度)
```

当同一横担同一侧挂载了至少两串绝缘子时，默认删除 `downward_ratio >= 0.7` 的绝缘子：

```bash
--downward-vertical-ratio 0.7
```

注意：这个逻辑会排除明显向下悬挂的绝缘子，只在同一侧至少有两串时触发。如果业务上需要保留悬垂绝缘子，应把该阈值调高到 `1.0`，此时正常非零长度绝缘子通常不会因该条件被删除。

## 8. JSON 输出结构

输出 JSON 的基本结构如下：

```json
{
  "coordinate_system": "input_global_xyz",
  "towers": [
    {
      "tower_id": 1,
      "tower_name": "杆塔1",
      "crossarms": [
        {
          "crossarm_id": 1,
          "left_endpoint": {
            "point_xyz": [526000.0, 3514000.0, 80.0],
            "insulators": [
              {
                "insulator_id": 1,
                "endpoint_1_xyz": [526000.1, 3514000.2, 79.9],
                "middle_point_xyz": [526000.2, 3514000.4, 78.8],
                "endpoint_2_xyz": [526000.3, 3514000.6, 77.7]
              }
            ]
          },
          "middle_point_xyz": [526005.0, 3514000.0, 80.0],
          "right_endpoint": {
            "point_xyz": [526010.0, 3514000.0, 80.0],
            "insulators": []
          }
        }
      ]
    }
  ]
}
```

字段含义：

| 字段 | 含义 |
|---|---|
| `coordinate_system` | 固定为 `input_global_xyz`，表示坐标来自输入 LAS 全局 XYZ |
| `tower_id` | 按全局 X、Y 排序后的杆塔编号 |
| `tower_name` | 中文杆塔名称，例如 `杆塔1` |
| `crossarm_id` | 单座杆塔内按 Z 从高到低的横担编号 |
| `left_endpoint.point_xyz` | 横担几何左端的真实杆塔点坐标 |
| `middle_point_xyz` | 横担左右端点的中点坐标 |
| `right_endpoint.point_xyz` | 横担几何右端的真实杆塔点坐标 |
| `insulator_id` | 当前横担当前侧内部的绝缘子序号 |
| `endpoint_1_xyz` | 距横担更近的绝缘子挂载端 |
| `middle_point_xyz` | 绝缘子两个端点的中点 |
| `endpoint_2_xyz` | 远离横担的绝缘子自由端 |

所有坐标保存到小数点后 6 位。

JSON 不保存点云索引、分类置信度、PCA 方向、横担宽度、扫描峰值和未挂载绝缘子。这些内容只在计算过程或终端统计中存在。

## 9. 当前未参与计算的参数

当前版本保留了下面这些命令行参数，但函数主体没有读取它们，因此调整这些参数不会改变输出：

| 参数 | 当前状态 |
|---|---|
| `--line-search-radius` | 未使用 |
| `--line-along-window` | 未使用 |
| `--line-layer-z-gap` | 未使用 |
| `--line-layer-merge-count` | 未使用 |
| `--min-line-layer-points` | 未使用 |
| `--crossarm-along-margin` | 未使用 |
| `--insulator-z-margin` | 未使用 |
| `--min-insulator-points-near-layer` | 未使用 |
| `--no-require-insulator-after-first-layer` | 未使用 |
| `--tower-bind-xy-margin` | 未使用 |
| `--tower-bind-z-margin` | 未使用 |

`class=2` 导线点当前也只用于打印输入点数，没有参与横担高度分层、线路方向估计或绝缘子挂载。

## 10. 运行命令

基本命令：

```bash
python tools/insulator_hengdan/extract_segmented_insulator_crossarm_keypoints.py \
  --input /path/to/segmented_input.las \
  --output /path/to/tower_keypoints.json \
  --overwrite
```

项目中的示例：

```bash
python tools/insulator_hengdan/extract_segmented_insulator_crossarm_keypoints.py \
  --input /24085403037/24085403037/shixi/dataset/6_23_demo/test/infer/110v12_merged_4cls_Output.las \
  --output /24085403037/24085403037/shixi/dataset/6_23_demo/test/hengdan_insulator/tower_004_keypoints.json \
  --overwrite
```

提高绝缘子断裂容忍度：

```bash
--insulator-connect-radius 0.30 \
--insulator-neighbors 24
```

减少相邻绝缘子粘连：

```bash
--insulator-connect-radius 0.12
```

检测间距较近的横担层：

```bash
--scan-z-step 0.25 \
--crossarm-min-z-separation 1.5
```

横担漏检时，可适当放宽：

```bash
--min-crossarm-points 30 \
--min-crossarm-width 1.5 \
--crossarm-min-prominence 0.3 \
--crossarm-min-relative-prominence 0.15
```

横担误检较多时，可适当收紧：

```bash
--min-crossarm-width 3.0 \
--crossarm-min-prominence 0.8 \
--crossarm-min-relative-prominence 0.35 \
--crossarm-min-width-ratio 0.65
```

## 11. 依赖环境

脚本依赖：

```text
laspy
numpy
scipy
```

可以使用下面的命令检查：

```bash
python -c "import laspy, numpy, scipy; print('dependencies ok')"
```

读取 LAZ 时，`laspy` 还需要可用的 LAZ 后端，例如 `lazrs`。

## 12. 常见问题与调参建议

### 一个绝缘子被拆成多个实例

优先增大 `--insulator-connect-radius`，其次增大 `--insulator-neighbors`。连接半径应根据点间距逐步增加，不要一次调得过大。

### 多串绝缘子被合并成一个实例

减小 `--insulator-connect-radius`。如果相邻绝缘子的点本身已经接触，仅靠连通域无法稳定分开，需要增加几何分裂逻辑。

### 杆塔没有被提取

检查杆塔类别是否正确，然后适当减小 `--min-tower-points`、`--min-tower-height`，或增大 `--tower-voxel-size` 以连接稀疏塔体。

### 横担层数少于实际层数

适当减小扫描步长、峰值最小间距、最小突出度和最小相对突出度。也要检查分割结果中横担是否仍被预测为 `class=1`。

### 塔身被误认为横担

提高 `--crossarm-min-prominence`、`--crossarm-min-relative-prominence` 或 `--crossarm-min-width-ratio`。这些参数用于区分局部横向突出的横担与逐渐变宽的塔身。

### 绝缘子没有挂载到横担

先检查绝缘子是否成功聚类，再适当增大 `--insulator-attach-radius`。该距离是绝缘子两个端点中较近端到横担三维线段的距离。

### CloudCompare 中的左右和 JSON 左右不一致

JSON 左右由 PCA 主轴及固定符号规则决定，不依赖 CloudCompare 相机方向。旋转观察点云后，屏幕左右可能改变，但 JSON 编号不会改变。

## 13. 使用限制

- 横担检测完全依赖杆塔分割质量；横担点被分成背景时无法恢复；
- 当前不利用导线确定线路方向，因此复杂塔型中的 `along_axis` 只是横担 PCA 方向的垂线；
- 相邻绝缘子点云发生连接时，近邻连通域可能无法把它们分开；
- 横担左右是几何稳定方向，不是观察视角方向；
- 绝缘子只按端点到横担线段的距离挂载，没有验证绝缘子是否与导线连接；
- 当前朝下绝缘子过滤可能不适合需要保留悬垂绝缘子的场景；
- 输出坐标直接继承输入 LAS 坐标系，脚本不执行 CRS、经纬度或 Unity 坐标转换。
