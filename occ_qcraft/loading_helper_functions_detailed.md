# loading.py 辅助函数详细技术文档

> **相关文档**：
> - 七个类的详细说明：`loading_detailed_explanation.md`
> - 概述性说明：`loading_and_data_structure_explanation.md`
> - 代码流程图：`loading_code_flowchart.md`

## 目录
1. [概述](#概述)
2. [六个辅助函数详细说明](#六个辅助函数详细说明)
3. [函数之间的关系](#函数之间的关系)
4. [与七个类的关系](#与七个类的关系)
5. [数学公式详解](#数学公式详解)
6. [输入输出详细说明](#输入输出详细说明)

---

## 概述

本文档详细说明 `loading.py` 中6个核心辅助函数的功能、实现细节、数学公式和输入输出。这些函数为数据加载类提供基础功能支持。

### 函数列表
1. `get_focal_length` - 焦距计算
2. `get_image_width_based_center` - 基于中心的图像宽度计算
3. `get_image_width_based_edge` - 基于边缘的图像宽度计算
4. `crop_and_resize_image` - 基于FOV的图像裁剪和缩放
5. `numba_label_mapping` - 标签聚合（Numba加速）
6. `priority_height_sampling` - 优先级高度采样（Numba加速）

---

## 五个辅助函数详细说明

### 1. get_focal_length

#### 1.1 功能与动机

**功能**：根据图像宽度和视场角（Field of View, FOV）计算相机的等效焦距。

**动机**：
- 在图像处理中，需要根据FOV和图像尺寸计算焦距
- 焦距是相机内参的关键参数
- 用于后续的图像裁剪和缩放计算

**应用场景**：
- 当已知图像宽度和FOV，需要计算焦距时
- 用于 `crop_and_resize_image` 函数中

#### 1.2 实现细节

**核心逻辑**：
```python
input_fov_rad = math.radians(input_fov)  # 将角度转换为弧度
focal_length = (input_width * 0.5) / math.tan(input_fov_rad * 0.5)
```

**数学原理**：
- 在针孔相机模型中，FOV与焦距的关系为：`tan(FOV/2) = (W/2) / f`
- 因此：`f = (W/2) / tan(FOV/2)`

#### 1.3 流程图

```
输入: input_width, input_fov
  │
  ├─ 1. 角度转弧度
  │   └─ input_fov_rad = radians(input_fov)
  │
  ├─ 2. 计算焦距
  │   └─ focal_length = (input_width / 2) / tan(input_fov_rad / 2)
  │
  └─ 输出: focal_length
```

#### 1.4 数学公式

**焦距计算公式**：
```
f = (W / 2) / tan(FOV / 2)
```

其中：
- `f`: 焦距（像素单位）
- `W`: 图像宽度（像素）
- `FOV`: 视场角（度）

**推导过程**：
```
在针孔相机模型中：
tan(θ) = (W/2) / f

其中 θ = FOV / 2

因此：
f = (W/2) / tan(FOV/2)
```

---

### 2. get_image_width_based_center

#### 2.1 功能与动机

**功能**：基于中心参考点，根据焦距和FOV计算图像宽度。

**动机**：
- 当需要根据目标FOV和已知焦距计算图像宽度时
- 用于以图像中心为参考点的裁剪场景
- 与 `get_focal_length` 互为逆运算

**应用场景**：
- `crop_and_resize_image` 中，当 `ref_pos="center"` 时使用

#### 2.2 实现细节

**核心逻辑**：
```python
input_fov_rad = math.radians(input_fov)
output_width = 2 * focal_length * math.tan(input_fov_rad * 0.5)
return int(output_width)
```

**数学原理**：
- 从焦距公式反推：`W = 2 * f * tan(FOV/2)`

#### 2.3 流程图

```
输入: focal_length, input_fov
  │
  ├─ 1. 角度转弧度
  │   └─ input_fov_rad = radians(input_fov)
  │
  ├─ 2. 计算图像宽度
  │   └─ output_width = 2 * focal_length * tan(input_fov_rad / 2)
  │
  └─ 输出: int(output_width)
```

#### 2.4 数学公式

**图像宽度计算公式**：
```
W = 2 * f * tan(FOV / 2)
```

其中：
- `W`: 图像宽度（像素）
- `f`: 焦距（像素单位）
- `FOV`: 视场角（度）

**与焦距公式的关系**：
```
f = (W / 2) / tan(FOV / 2)  ←→  W = 2 * f * tan(FOV / 2)
```

---

### 3. get_image_width_based_edge

#### 3.1 功能与动机

**功能**：基于边缘参考点，根据焦距、输入FOV和输出FOV计算图像宽度。

**动机**：
- 当需要以图像边缘（左边缘或右边缘）为参考点进行裁剪时
- 计算在保持焦距不变的情况下，改变FOV后的图像宽度
- 用于非中心裁剪场景

**应用场景**：
- `crop_and_resize_image` 中，当 `ref_pos="left_edge"` 或 `"right_edge"` 时使用

#### 3.2 实现细节

**核心逻辑**：
```python
diff_fov_rad = math.radians(input_fov * 0.5 - output_fov)
output_width = input_width * 0.5 - focal_length * math.tan(diff_fov_rad)
return int(output_width)
```

**数学原理**：
- 从图像中心到边缘的距离：`W/2`
- 从图像中心到新FOV边缘的距离：`f * tan(FOV_out/2)`
- 从原边缘到新边缘的距离：`f * tan((FOV_in/2 - FOV_out))`
- 因此：`W_out = W_in/2 - f * tan((FOV_in/2 - FOV_out))`

#### 3.3 流程图

```
输入: focal_length, input_fov, input_width, output_fov
  │
  ├─ 1. 计算FOV差值（角度）
  │   └─ diff_fov = input_fov / 2 - output_fov
  │
  ├─ 2. 角度转弧度
  │   └─ diff_fov_rad = radians(diff_fov)
  │
  ├─ 3. 计算输出宽度
  │   └─ output_width = input_width / 2 - focal_length * tan(diff_fov_rad)
  │
  └─ 输出: int(output_width)
```

#### 3.4 数学公式

**基于边缘的图像宽度计算公式**：
```
W_out = W_in/2 - f * tan((FOV_in/2 - FOV_out))
```

其中：
- `W_out`: 输出图像宽度（像素）
- `W_in`: 输入图像宽度（像素）
- `f`: 焦距（像素单位）
- `FOV_in`: 输入视场角（度）
- `FOV_out`: 输出视场角（度）

**几何示意图**：
```
        图像中心
           |
    ┌──────┼──────┐  ← 输入图像 (FOV_in)
    |      |      |
    |      |      |
    └──────┼──────┘
           |
    ┌──────┼──────┐  ← 输出图像 (FOV_out)
    |      |      |
    └──────┼──────┘
           |
        边缘参考点
```

**推导过程**：
```
从图像中心到输入边缘的距离：d_in = f * tan(FOV_in/2)
从图像中心到输出边缘的距离：d_out = f * tan(FOV_out/2)
从输入边缘到输出边缘的距离：Δd = f * tan(FOV_in/2 - FOV_out)

因此：
W_out = W_in/2 - Δd = W_in/2 - f * tan(FOV_in/2 - FOV_out)
```

---

### 4. crop_and_resize_image

#### 4.1 功能与动机

**功能**：基于输入/输出FOV进行图像裁剪或填充，并计算裁剪/填充参数。

**动机**：
- 不同相机具有不同的FOV，需要统一到目标FOV
- 支持三种参考位置：中心、左边缘、右边缘
- 处理裁剪和填充两种情况
- 为后续的图像处理和相机内参更新提供参数

**应用场景**：
- `LoadMultiViewUndistortImageWithResize` 类的 `image_resize_based_on_fov` 方法中调用

#### 4.2 实现细节

**核心流程**：

1. **计算焦距**
   ```python
   focal_length = get_focal_length(input_width, input_fov)
   ```

2. **计算输出图像宽度**
   - 如果 `ref_pos="center"`：
     ```python
     output_width = get_image_width_based_center(focal_length, output_fov)
     ```
   - 否则（`"left_edge"` 或 `"right_edge"`）：
     ```python
     output_width = get_image_width_based_edge(focal_length, input_fov, input_width, output_fov)
     ```

3. **宽度对齐**（确保是4的倍数）
   ```python
   output_width = floor(output_width // 4) * 4
   ```

4. **计算输出高度**（保持宽高比）
   ```python
   output_height = output_width // output_wh_ratio  # output_wh_ratio = 2
   ```

5. **计算裁剪位置**
   - `ref_pos="center"`: `crop_x = (input_width - output_width) / 2`
   - `ref_pos="left_edge"`: `crop_x = 0`
   - `ref_pos="right_edge"`: `crop_x = input_width - output_width`

6. **处理填充或裁剪**
   - 如果 `output_height > input_height`：需要填充（`is_padding=True`）
     ```python
     crop_y = floor((output_height - input_height) / 2)
     crop_x = 0  # 填充时x方向不裁剪
     ```
   - 如果 `padding_x=True`：x方向填充
     ```python
     crop_x = floor((input_height * output_wh_ratio - input_width) / 2)
     crop_y = 0
     ```
   - 否则：裁剪y方向（`is_padding=False`）
     ```python
     crop_y = input_height - output_height
     crop_y = floor(crop_y * crop_y_scale + crop_y_shift)
     ```

#### 4.3 流程图

```
输入: input_fov, input_width, input_height, output_fov, ...
  │
  ├─ 1. 计算焦距
  │   └─ focal_length = get_focal_length(input_width, input_fov)
  │
  ├─ 2. 计算输出宽度
  │   ├─ ref_pos == "center"?
  │   │   ├─ YES → output_width = get_image_width_based_center(focal_length, output_fov)
  │   │   └─ NO  → output_width = get_image_width_based_edge(...)
  │   │
  │   └─ 宽度对齐: output_width = floor(output_width // 4) * 4
  │
  ├─ 3. 计算输出高度
  │   └─ output_height = output_width // output_wh_ratio
  │
  ├─ 4. 计算裁剪位置x
  │   ├─ ref_pos == "center" → crop_x = (input_width - output_width) / 2
  │   ├─ ref_pos == "left_edge" → crop_x = 0
  │   └─ ref_pos == "right_edge" → crop_x = input_width - output_width
  │
  ├─ 5. 判断填充或裁剪
  │   ├─ output_height > input_height?
  │   │   └─ YES → 填充: crop_y = floor((output_height - input_height) / 2), crop_x = 0
  │   │
  │   ├─ padding_x == True?
  │   │   └─ YES → x方向填充: crop_x = floor((input_height * ratio - input_width) / 2)
  │   │
  │   └─ NO → 裁剪: crop_y = floor((input_height - output_height) * scale + shift)
  │
  └─ 输出: crop_x, crop_y, output_width, output_height, is_padding
```

#### 4.4 数学公式

**输出宽度计算（中心参考）**：
```
W_out = 2 * f * tan(FOV_out / 2)
```

**输出宽度计算（边缘参考）**：
```
W_out = W_in/2 - f * tan((FOV_in/2 - FOV_out))
```

**输出高度计算**：
```
H_out = W_out / ratio  (ratio = 2)
```

**裁剪位置计算（中心）**：
```
crop_x = (W_in - W_out) / 2
```

**裁剪位置计算（左边缘）**：
```
crop_x = 0
```

**裁剪位置计算（右边缘）**：
```
crop_x = W_in - W_out
```

**填充大小计算**：
```
pad_y = floor((H_out - H_in) / 2)
```

**裁剪位置计算（带微调）**：
```
crop_y = floor((H_in - H_out) * scale + shift)
```

---

### 5. numba_label_mapping

#### 5.1 功能与动机

**功能**：将多个点映射到同一体素时的标签聚合，使用Numba JIT加速。

**动机**：
- 体素化过程中，多个点可能映射到同一个体素
- 需要将多个点的标签聚合为单个体素标签
- 使用Numba加速以提高性能（处理大量点云数据时）

**应用场景**：
- `LoadOccGTFromPCD` 类的 `get_downsampled_labels` 方法中调用
- `LoadOccGTFromLCData` 类的 `get_bev_labels` 方法中调用（如果启用）

#### 5.2 实现细节

**核心逻辑**：

1. **初始化**
   - 创建标签计数器数组（256个类别）
   - 初始化当前体素位置和可见性标志

2. **遍历所有点**
   - 对每个点，检查是否属于当前体素
   - 如果属于：累加标签计数，更新可见性（OR操作）
   - 如果不属于：处理上一个体素，重置计数器

3. **标签聚合策略**（按优先级）：
   - **策略1**：如果所有点都是 `ignore_label` → 输出 `ignore_label`
   - **策略2**：如果所有点都是 `free_label` → 输出 `free_label`
   - **策略3**：如果只有 `free_label` 和 `ignore_label` → 输出 `free_label`
   - **策略4**：否则 → 输出出现次数最多的非free/ignore标签

4. **可见性聚合**：
   - 只要有一个点可见，体素就可见（OR操作）

#### 5.3 流程图

```
输入: labels, valid, voxel_indices, labels_out, valid_out
  │
  ├─ 1. 初始化
  │   ├─ counter_label = zeros(256)  # 标签计数器
  │   ├─ counter_valid = False  # 可见性标志
  │   └─ cur_sea_pos = voxel_indices[0]  # 当前体素位置
  │
  ├─ 2. 遍历所有点 (idx = 1 to len-1)
  │   │
  │   ├─ 3. 检查是否属于当前体素
  │   │   ├─ cur_pos == cur_sea_pos?
  │   │   │   ├─ YES → 累加标签和可见性
  │   │   │   │   ├─ counter_label[labels[idx]] += 1
  │   │   │   │   └─ counter_valid = counter_valid | valid[idx]
  │   │   │   │
  │   │   │   └─ NO → 处理上一个体素
  │   │   │       ├─ 4. 标签聚合
  │   │   │       │   ├─ 全部ignore? → labels_out = ignore_label
  │   │   │       │   ├─ 全部free? → labels_out = free_label
  │   │   │       │   ├─ 只有free+ignore? → labels_out = free_label
  │   │   │       │   └─ 否则 → labels_out = argmax(counter_label[free+1:ignore])
  │   │   │       │
  │   │   │       ├─ 5. 可见性聚合
  │   │   │       │   └─ valid_out = counter_valid
  │   │   │       │
  │   │   │       └─ 6. 重置计数器
  │   │   │           ├─ counter_label = zeros(256)
  │   │   │           ├─ counter_valid = False
  │   │   │           └─ cur_sea_pos = cur_pos
  │   │   │
  │   │   └─ 继续下一个点
  │   │
  └─ 7. 处理最后一个体素（同步骤4-5）
  │
  └─ 输出: labels_out, valid_out
```

#### 5.4 数学公式

**标签聚合公式**：

设体素 `v` 内有 `N` 个点，标签为 `{l₁, l₂, ..., lₙ}`，则：

```
如果 ∀i: lᵢ = ignore_label → label(v) = ignore_label
如果 ∀i: lᵢ = free_label → label(v) = free_label
如果 ∀i: lᵢ ∈ {free_label, ignore_label} → label(v) = free_label
否则 → label(v) = argmax_{l ∈ [free+1, ignore-1]} count(l)
```

其中 `count(l)` 是标签 `l` 在体素内的出现次数。

**可见性聚合公式**：
```
valid(v) = ∨ᵢ valid(pᵢ)
```

即体素的可见性为所有点可见性的逻辑或（OR）。

**标签计数**：
```
count(l) = Σᵢ [lᵢ == l]
```

其中 `[·]` 是指示函数。

---

## 函数之间的关系

### 调用关系图

```
crop_and_resize_image
  ├─ get_focal_length
  ├─ get_image_width_based_center (当ref_pos="center")
  └─ get_image_width_based_edge (当ref_pos="left_edge"或"right_edge")

LoadMultiViewUndistortImageWithResize.image_resize_based_on_fov
  └─ crop_and_resize_image

LoadOccGTFromPCD.get_downsampled_labels
  └─ numba_label_mapping

LoadOccGTFromLCData.get_bev_labels (如果启用)
  └─ numba_label_mapping

LoadOccGTFromLCData.__call__
  └─ priority_height_sampling
```

### 功能分组

**FOV相关函数组**：
- `get_focal_length` - 焦距计算
- `get_image_width_based_center` - 中心参考宽度计算
- `get_image_width_based_edge` - 边缘参考宽度计算
- `crop_and_resize_image` - 综合裁剪/填充函数

**标签处理函数组**：
- `numba_label_mapping` - 体素标签聚合（点云到体素）
- `priority_height_sampling` - 高度维度压缩（体素到体素）

### 数据流关系

```
FOV函数组的数据流：
  input_width, input_fov
    ↓
  get_focal_length
    ↓
  focal_length
    ↓
  get_image_width_based_center/edge
    ↓
  output_width
    ↓
  crop_and_resize_image
    ↓
  crop_x, crop_y, output_width, output_height, is_padding

标签处理的数据流：
  points → voxel_indices (已排序)
    ↓
  numba_label_mapping
    ↓
  labels_out, valid_out

高度压缩的数据流：
  target_voxel (H, W, Z)
    ↓
  priority_height_sampling
    ↓
  cls_label (H, W, Z//2)
```

---

## 与七个类的关系

### 1. LoadOccGTFromPCD

**使用 `numba_label_mapping`**：
- 在 `get_downsampled_labels` 方法中调用
- 用于将点云标签聚合到下采样体素网格
- 位置：第480行

**调用链**：
```
LoadOccGTFromPCD.__call__
  └─ get_downsampled_labels
      └─ numba_label_mapping
```

### 2. LoadOccGTFromLCData

**使用 `numba_label_mapping`**：
- 在 `get_bev_labels` 方法中调用（如果启用BEV标签生成）
- 用于生成BEV（Bird's Eye View）标签
- 位置：第525行

**使用 `priority_height_sampling`**：
- 在 `__call__` 方法中直接调用
- 用于压缩高分辨率占用标签的高度维度
- 位置：第1169行

**调用链**：
```
LoadOccGTFromLCData.__call__
  ├─ get_bev_labels (如果启用)
  │   └─ numba_label_mapping
  └─ priority_height_sampling (压缩高度维度)
```

### 3. LoadMultiViewUndistortImageWithResize

**使用 `crop_and_resize_image`**：
- 在 `image_resize_based_on_fov` 方法中调用
- 用于基于FOV进行图像裁剪和缩放
- 位置：第1483行

**调用链**：
```
LoadMultiViewUndistortImageWithResize.__call__
  └─ image_resize_based_on_fov
      └─ crop_and_resize_image
          ├─ get_focal_length
          ├─ get_image_width_based_center
          └─ get_image_width_based_edge
```

### 4. 其他类

**LoadDepthGTFromPCD**、**LoadMultiViewVirtualImageWithResize**、**LoadMultiViewSegLabel**、**LoadMultiViewDepthLabel**：
- 不直接使用这5个辅助函数
- 但它们可能使用这些函数处理后的结果（如裁剪后的图像）

---

## 数学公式详解

### FOV相关公式总结

**1. 焦距公式**：
```
f = (W / 2) / tan(FOV / 2)
```

**2. 图像宽度公式（中心参考）**：
```
W = 2 * f * tan(FOV / 2)
```

**3. 图像宽度公式（边缘参考）**：
```
W_out = W_in/2 - f * tan((FOV_in/2 - FOV_out))
```

**4. 图像高度公式**：
```
H = W / ratio  (ratio = 2)
```

**5. 裁剪位置公式（中心）**：
```
crop_x = (W_in - W_out) / 2
```

**6. 裁剪位置公式（边缘）**：
```
crop_x = 0  (左边缘)
crop_x = W_in - W_out  (右边缘)
```

### 标签聚合公式总结

**标签聚合规则**：
```
label(v) = {
    ignore_label,  if ∀p ∈ v: label(p) = ignore_label
    free_label,    if ∀p ∈ v: label(p) = free_label
    free_label,    if ∀p ∈ v: label(p) ∈ {free_label, ignore_label}
    argmax_l count(l),  otherwise
}
```

**可见性聚合规则**：
```
valid(v) = ∨_{p ∈ v} valid(p)
```

**标签计数**：
```
count(l) = Σ_{p ∈ v} [label(p) == l]
```

**优先级高度采样公式**：
```
Z_out = Z_in // 2
z1 = z_out * 2
z2 = z_out * 2 + 1
label_out = argmin_{i} [priority[i] ∈ {label1, label2}]
```

---

## 输入输出详细说明

### 1. get_focal_length

#### 输入参数

| 参数名 | 类型 | 取值范围 | 含义 |
|--------|------|----------|------|
| `input_width` | int/float | > 0 | 输入图像宽度（像素） |
| `input_fov` | float | (0, 180) | 输入视场角（度） |

#### 输出

| 返回值 | 类型 | 取值范围 | 含义 |
|--------|------|----------|------|
| `focal_length` | float | > 0 | 计算得到的焦距（像素单位） |

**示例**：
- 输入：`input_width=1920, input_fov=110`
- 输出：`focal_length ≈ 1024.5`（像素）

---

### 2. get_image_width_based_center

#### 输入参数

| 参数名 | 类型 | 取值范围 | 含义 |
|--------|------|----------|------|
| `focal_length` | float | > 0 | 焦距（像素单位） |
| `input_fov` | float | (0, 180) | 视场角（度） |

#### 输出

| 返回值 | 类型 | 取值范围 | 含义 |
|--------|------|----------|------|
| `output_width` | int | > 0 | 计算得到的图像宽度（像素） |

**示例**：
- 输入：`focal_length=1024.5, input_fov=90`
- 输出：`output_width ≈ 2049`（像素）

---

### 3. get_image_width_based_edge

#### 输入参数

| 参数名 | 类型 | 取值范围 | 含义 |
|--------|------|----------|------|
| `focal_length` | float | > 0 | 焦距（像素单位） |
| `input_fov` | float | (0, 180) | 输入视场角（度） |
| `input_width` | int/float | > 0 | 输入图像宽度（像素） |
| `output_fov` | float | (0, input_fov) | 输出视场角（度） |

#### 输出

| 返回值 | 类型 | 取值范围 | 含义 |
|--------|------|----------|------|
| `output_width` | int | > 0 | 计算得到的图像宽度（像素） |

**约束**：`output_fov < input_fov`（否则输出宽度可能为负）

**示例**：
- 输入：`focal_length=1024.5, input_fov=110, input_width=1920, output_fov=90`
- 输出：`output_width ≈ 960`（像素）

---

### 4. crop_and_resize_image

#### 输入参数

| 参数名 | 类型 | 取值范围 | 含义 |
|--------|------|----------|------|
| `input_fov` | float | (0, 180) | 输入视场角（度） |
| `input_width` | int | > 0 | 输入图像宽度（像素） |
| `input_height` | int | > 0 | 输入图像高度（像素） |
| `output_fov` | float | (0, 180) | 输出视场角（度） |
| `output_wh_ratio` | int | 默认2 | 输出图像宽高比（宽/高） |
| `ref_pos` | str | "center", "left_edge", "right_edge" | 参考位置 |
| `crop_y_scale` | float | [0, 1] | Y方向裁剪位置缩放因子 |
| `crop_y_shift` | int | 任意整数 | Y方向裁剪位置偏移（像素） |
| `padding_x` | bool | True/False | 是否在X方向填充 |

#### 输出

| 返回值 | 类型 | 取值范围 | 含义 |
|--------|------|----------|------|
| `crop_x` | int | >= 0 | X方向裁剪/填充起始位置（像素） |
| `crop_y` | int | >= 0 | Y方向裁剪/填充起始位置（像素） |
| `output_width` | int | > 0, 4的倍数 | 输出图像宽度（像素） |
| `output_height` | int | > 0 | 输出图像高度（像素） |
| `is_padding` | bool | True/False | 是否为填充模式（True=填充，False=裁剪） |

**输出值含义**：

- **`crop_x`**：
  - 裁剪模式：从输入图像的第 `crop_x` 列开始裁剪
  - 填充模式：在输入图像左右各填充 `crop_x` 列
  - 范围：`[0, input_width]`

- **`crop_y`**：
  - 裁剪模式：从输入图像的第 `crop_y` 行开始裁剪
  - 填充模式：在输入图像上下各填充 `crop_y` 行
  - 范围：`[0, input_height]`

- **`output_width`**：
  - 输出图像的宽度（像素）
  - 保证是4的倍数（用于对齐）
  - 范围：`[4, input_width]`（裁剪）或 `[input_width, ...]`（填充）

- **`output_height`**：
  - 输出图像的高度（像素）
  - 等于 `output_width / output_wh_ratio`
  - 范围：`[1, ...]`

- **`is_padding`**：
  - `True`：需要填充（输出尺寸大于输入）
  - `False`：需要裁剪（输出尺寸小于输入）

**示例1（中心裁剪）**：
- 输入：`input_fov=110, input_width=1920, input_height=1080, output_fov=90, ref_pos="center"`
- 输出：`crop_x=240, crop_y=270, output_width=1440, output_height=720, is_padding=False`

**示例2（填充）**：
- 输入：`input_fov=90, input_width=1440, input_height=720, output_fov=110, ref_pos="center"`
- 输出：`crop_x=0, crop_y=180, output_width=1920, output_height=960, is_padding=True`

---

### 5. numba_label_mapping

#### 输入参数

| 参数名 | 类型 | 形状 | 数据类型 | 取值范围 | 含义 |
|--------|------|------|----------|----------|------|
| `labels` | np.ndarray | [N] | uint8 | [0, 255] | 每个点的标签 |
| `valid` | np.ndarray | [N] | bool | True/False | 每个点的可见性 |
| `voxel_indices` | np.ndarray | [N, 3] | int32 | [0, grid_size[i]) | 每个点对应的体素索引 |
| `labels_out` | np.ndarray | [H, W, D] | uint8 | 初始化为0 | 输出标签数组（会被修改） |
| `valid_out` | np.ndarray | [H, W, D] | bool | 初始化为True | 输出可见性数组（会被修改） |
| `ignore_label` | int | - | int | 默认255 | 忽略标签值 |
| `free_label` | int | - | int | 默认0 | 自由空间标签值 |

**约束**：
- `voxel_indices` 必须已按 `(x, y, z)` 字典序排序
- `labels_out` 和 `valid_out` 的形状必须匹配网格大小
- `N` 是点的数量，`H, W, D` 是网格的高度、宽度、深度

#### 输出

| 返回值 | 类型 | 形状 | 数据类型 | 取值范围 | 含义 |
|--------|------|------|----------|----------|------|
| `labels_out` | np.ndarray | [H, W, D] | uint8 | [0, 255] | 聚合后的标签数组（原地修改） |
| `valid_out` | np.ndarray | [H, W, D] | bool | True/False | 聚合后的可见性数组（原地修改） |

**输出值含义**：

- **`labels_out[x, y, z]`**：
  - 体素 `(x, y, z)` 的聚合标签
  - 值范围：`[0, 255]`
  - `0`：自由空间
  - `1-254`：各类别标签
  - `255`：忽略/未知

- **`valid_out[x, y, z]`**：
  - 体素 `(x, y, z)` 的可见性
  - `True`：至少有一个点可见
  - `False`：所有点都不可见

**标签聚合规则示例**：

假设体素内有5个点，标签为 `[0, 0, 2, 2, 255]`，可见性为 `[True, True, True, False, True]`：

1. 标签计数：`count(0)=2, count(2)=2, count(255)=1`
2. 不满足全部ignore或全部free的条件
3. 不满足只有free+ignore的条件（因为有标签2）
4. 选择出现次数最多的非free/ignore标签：`argmax(count(2)) = 2`
5. 输出：`labels_out = 2, valid_out = True`

**性能特点**：
- 使用Numba JIT编译，执行速度快
- `cache=True`：缓存编译结果
- `nogil=True`：释放GIL，支持多线程

---

### 6. priority_height_sampling

#### 6.1 功能与动机

**功能**：将占用网格的高度维度从Z压缩到Z/2，通过优先级策略选择每对相邻z层的标签。

**动机**：
- 减少计算和存储开销：将高度维度减半
- 保留重要信息：使用优先级策略，优先保留障碍物标签而非道路标签
- 处理重叠标签：当同一(x,y)位置的多个z层有不同的标签时，选择优先级最高的
- 提高模型效率：减少输入维度可以加速训练和推理

**应用场景**：
- `LoadOccGTFromLCData` 类中，对高分辨率占用标签进行高度压缩
- 用于减少内存占用和提高处理速度

#### 6.2 实现细节

**核心逻辑**：

1. **初始化**
   ```python
   H, W, Z = target_voxel.shape
   new_z = Z // 2  # 压缩后的高度维度
   cls_label = np.ones((H, W, new_z)) * ignore_label  # 初始化为忽略标签
   ```

2. **遍历所有体素位置**
   - 对每个 `(x, y)` 位置
   - 对每个压缩后的z层 `z`（范围 `[0, new_z)`）

3. **处理每对z层**
   ```python
   z1, z2 = z * 2, z * 2 + 1  # 原始的两个z层
   label1 = target_voxel[x, y, z1]
   label2 = target_voxel[x, y, z2] if z2 < Z else -1
   ```

4. **优先级选择**
   - 按优先级顺序遍历 `priority_order` 数组
   - 如果 `label1` 或 `label2` 等于当前优先级标签，则选择该标签并跳出循环
   - 如果都不匹配，继续下一个优先级

5. **输出**
   - 将选中的标签写入 `cls_label[x, y, z]`

**关键特点**：
- 使用Numba JIT编译加速
- 优先级顺序：从高到低（数组第一个元素优先级最高）
- 如果两个z层都没有匹配的优先级标签，输出保持为 `ignore_label`

#### 6.3 流程图

```
输入: target_voxel, priority_order, ignore_label
  │
  ├─ 1. 初始化
  │   ├─ H, W, Z = target_voxel.shape
  │   ├─ new_z = Z // 2
  │   └─ cls_label = ones((H, W, new_z)) * ignore_label
  │
  ├─ 2. 遍历所有位置 (x, y, z)
  │   │
  │   ├─ 3. 计算原始z层索引
  │   │   ├─ z1 = z * 2
  │   │   └─ z2 = z * 2 + 1 (如果 < Z)
  │   │
  │   ├─ 4. 获取两个z层的标签
  │   │   ├─ label1 = target_voxel[x, y, z1]
  │   │   └─ label2 = target_voxel[x, y, z2] (如果存在)
  │   │
  │   ├─ 5. 按优先级选择标签
  │   │   ├─ 遍历 priority_order (从高到低)
  │   │   │   ├─ label1 == priority?
  │   │   │   │   └─ YES → cls_label[x, y, z] = priority, break
  │   │   │   ├─ label2 == priority?
  │   │   │   │   └─ YES → cls_label[x, y, z] = priority, break
  │   │   │   └─ NO → 继续下一个优先级
  │   │   │
  │   │   └─ 如果都不匹配 → 保持 ignore_label
  │   │
  │   └─ 继续下一个位置
  │
  └─ 输出: cls_label (H, W, new_z)
```

#### 6.4 数学公式

**高度压缩公式**：
```
Z_out = Z_in // 2
```

**z层映射公式**：
```
z1 = z_out * 2
z2 = z_out * 2 + 1
```

**优先级选择公式**：
```
label_out = {
    priority[0],  if label1 == priority[0] or label2 == priority[0]
    priority[1],  else if label1 == priority[1] or label2 == priority[1]
    ...
    priority[n-1], else if label1 == priority[n-1] or label2 == priority[n-1]
    ignore_label,  otherwise
}
```

其中 `priority[i]` 是优先级数组中的第i个元素（优先级从高到低）。

**优先级选择逻辑**：
```
label_out = argmin_{i} [priority[i] ∈ {label1, label2}]
```

即选择在 `{label1, label2}` 中出现且优先级最高（索引最小）的标签。

#### 6.5 示例

**输入**：
- `target_voxel[x, y, :] = [0, 0, 2, 2, 1, 1, 3, 3]` (Z=8)
- `priority_order = [2, 3, 1, 0]` (优先级：2 > 3 > 1 > 0)

**处理过程**：
- `z=0`: `z1=0, z2=1`, `label1=0, label2=0`
  - 检查优先级2：不匹配
  - 检查优先级3：不匹配
  - 检查优先级1：不匹配
  - 检查优先级0：匹配！→ `cls_label[x, y, 0] = 0`
  
- `z=1`: `z1=2, z2=3`, `label1=2, label2=2`
  - 检查优先级2：匹配！→ `cls_label[x, y, 1] = 2`
  
- `z=2`: `z1=4, z2=5`, `label1=1, label2=1`
  - 检查优先级2：不匹配
  - 检查优先级3：不匹配
  - 检查优先级1：匹配！→ `cls_label[x, y, 2] = 1`
  
- `z=3`: `z1=6, z2=7`, `label1=3, label2=3`
  - 检查优先级2：不匹配
  - 检查优先级3：匹配！→ `cls_label[x, y, 3] = 3`

**输出**：
- `cls_label[x, y, :] = [0, 2, 1, 3]` (new_z=4)

#### 6.6 输入参数

| 参数名 | 类型 | 形状 | 数据类型 | 取值范围 | 含义 |
|--------|------|------|----------|----------|------|
| `target_voxel` | np.ndarray | [H, W, Z] | uint8 | [0, 255] | 输入占用标签数组 |
| `priority_order` | np.ndarray | [N] | int/uint8 | 标签值数组 | 优先级顺序（从高到低） |
| `ignore_label` | int | - | int | 默认255 | 忽略标签值 |

**约束**：
- `Z` 必须是偶数（否则最后一对z层可能不完整）
- `priority_order` 中的值必须是有效的标签值
- `H, W, Z` 分别是网格的高度、宽度、深度

#### 输出

| 返回值 | 类型 | 形状 | 数据类型 | 取值范围 | 含义 |
|--------|------|------|----------|----------|------|
| `cls_label` | np.ndarray | [H, W, Z//2] | uint8 | [0, 255] | 压缩后的标签数组 |

**输出值含义**：

- **`cls_label[x, y, z]`**：
  - 压缩后体素 `(x, y, z)` 的标签
  - 值范围：`[0, 255]`
  - 值来源：从原始体素 `(x, y, z*2)` 和 `(x, y, z*2+1)` 中按优先级选择
  - `0`：自由空间
  - `1-254`：各类别标签（按优先级选择）
  - `255`：忽略/未知（如果两个z层都不匹配任何优先级）

**优先级选择规则示例**：

假设：
- `priority_order = [2, 3, 1, 0]`（优先级：2 > 3 > 1 > 0）
- `target_voxel[x, y, z*2] = 1`（可行驶区域）
- `target_voxel[x, y, z*2+1] = 2`（可移动物体）

选择过程：
1. 检查优先级2：`label2 == 2` → 匹配！→ `cls_label[x, y, z] = 2`

结果：选择优先级更高的可移动物体标签（2），而不是可行驶区域标签（1）。

**性能特点**：
- 使用Numba JIT编译，执行速度快
- 时间复杂度：O(H × W × Z/2 × N)，其中N是优先级数量
- 空间复杂度：O(H × W × Z/2)

**使用场景**：
- 减少内存占用：高度维度减半
- 保留重要信息：优先保留障碍物标签
- 加速处理：减少后续处理的维度

---

## 总结

本文档详细说明了 `loading.py` 中6个核心辅助函数的功能、实现、数学公式和输入输出。这些函数为数据加载类提供基础功能：

1. **FOV相关函数组**：处理相机FOV、焦距和图像尺寸的计算
2. **标签处理函数组**：
   - `numba_label_mapping`：高效处理点云到体素的标签聚合
   - `priority_height_sampling`：高效处理体素高度维度的压缩

这些函数通过Numba加速和优化的算法设计，确保了数据加载管道的高效执行。

