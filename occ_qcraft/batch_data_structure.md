# train_dataloader Batch 数据结构详细说明

本文档详细说明 `vis.py` 中遍历 `train_dataloader` 时，每个 batch 的数据结构。

## 一、Batch 整体结构

**类型**: `Dict[str, Any]`（字典）

**说明**: batch 是一个字典，包含多个键值对。每个键对应一种数据类型，值可能是：
- `list`: 列表类型（多个样本的列表，每个元素对应一个样本）
- `numpy.ndarray`: 数组类型（多个样本已stack成数组）
- `torch.Tensor`: 张量类型（多个样本已stack成张量）
- 其他基本类型（字符串、整数等）

---

## 二、各键值对详细说明

### 1. `data_format`

**类型**: `str`（字符串）

**含义**: 数据格式标识符，用于区分不同的数据格式

**取值范围**: 
- `"qds"`: QDS格式数据（当前使用的格式）
- `"lc"`: LC格式数据（其他格式）

**示例值**: `"qds"`

**说明**: 用于transforms判断是否处理当前数据

---

### 2. `img_filename`

**类型**: `list[list[str]]`（列表的列表）

**含义**: 每个样本的多视角图像文件路径列表

**结构说明**:
- 外层列表长度 = batch_size（样本数量）
- 内层列表长度 = 该样本的相机数量（通常为多个视角）
- 每个字符串是图像文件的相对路径

**取值范围**: 
- 路径字符串，例如: `"image/CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99/raw_image/000.jpg"`

**示例值**:
```python
[
    ["image/CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99/raw_image/000.jpg", 
     "image/CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99/raw_image/000.jpg", ...],
    ["image/CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99/raw_image/001.jpg", ...],
    ...
]
```

**说明**: 用于定位和加载图像文件，也用于推断其他数据文件（如占用标签、分割标签等）的路径

---

### 3. `sample_idx`

**类型**: `list[str]`（字符串列表）

**含义**: 每个样本的帧ID（样本标识符）

**结构说明**:
- 列表长度 = batch_size
- 每个元素是帧ID字符串

**取值范围**: 
- 字符串，通常是数字字符串，例如: `"000"`, `"001"`, `"065"`

**示例值**: `["000", "001", "002", "003"]`

**说明**: 用于标识不同的样本帧，也用于构建其他数据文件的路径

---

### 4. `data_dir`

**类型**: `str`（字符串）

**含义**: 数据根目录路径（FILELIST所在目录）

**取值范围**: 
- 有效的文件系统路径字符串，例如: `"/mnt/data/user/zhongyurong/dz/occ/occ_qcraft/data/train"`

**示例值**: `"/mnt/data/user/zhongyurong/dz/occ/occ_qcraft/data/train"`

**说明**: 用于构建完整的数据文件路径

---

### 5. `intrinsic`

**类型**: `numpy.ndarray`（numpy数组）

**含义**: 每个样本的每个相机的内参矩阵

**形状**: `(batch_size, num_cameras, 3, 3)`
- `batch_size`: 批次大小
- `num_cameras`: 每个样本的相机数量（可能不同，但通常相同）
- `3, 3`: 3x3内参矩阵

**数据类型**: `numpy.float32` 或 `numpy.float64`

**数值范围**: 
- 内参矩阵元素通常为正值
- `fx, fy`: 焦距（像素单位），通常范围 [100, 10000]
- `cx, cy`: 主点坐标（像素单位），通常范围 [0, 图像宽度/高度]
- 矩阵格式: `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`

**示例值**:
```python
array([
    [[[1000.0, 0.0, 512.0],
      [0.0, 1000.0, 512.0],
      [0.0, 0.0, 1.0]],
     [[1000.0, 0.0, 512.0],
      [0.0, 1000.0, 512.0],
      [0.0, 0.0, 1.0]], ...],
    ...
], dtype=float32)
```

**说明**: 用于图像到3D空间的投影变换

---

### 6. `resized_intrinsic`

**类型**: `list[list[numpy.ndarray]]`（列表的列表，每个元素是3x3数组）

**含义**: 每个样本的每个相机在图像resize后的内参矩阵

**结构说明**:
- 外层列表长度 = batch_size
- 内层列表长度 = 该样本的相机数量
- 每个元素是形状为 `(3, 3)` 的numpy数组

**数据类型**: `numpy.float32` 或 `numpy.float64`

**数值范围**: 与 `intrinsic` 相同，但值会根据图像resize比例调整

**说明**: 图像经过resize后，内参需要相应调整（焦距和主点坐标按比例缩放）

---

### 7. `img`

**类型**: `list[list[numpy.ndarray]]`（列表的列表，每个元素是图像数组）

**含义**: 每个样本的每个相机的图像数据

**结构说明**:
- 外层列表长度 = batch_size
- 内层列表长度 = 该样本的相机数量
- 每个元素是形状为 `(H, W, 3)` 的numpy数组（RGB图像）

**形状**: 每个图像数组的形状为 `(height, width, 3)`
- `height`: 图像高度（像素），例如: 512
- `width`: 图像宽度（像素），例如: 1024
- `3`: RGB三个通道

**数据类型**: 
- `numpy.uint8`: 如果 `to_float32=False`（默认）
- `numpy.float32`: 如果 `to_float32=True`

**数值范围**:
- `uint8`: [0, 255]，表示RGB像素值
- `float32`: [0.0, 255.0] 或归一化后的范围

**示例值**:
```python
[
    [array([[[255, 255, 255], ...], ...], shape=(512, 1024, 3), dtype=uint8), ...],
    ...
]
```

**说明**: 多视角图像数据，用于模型输入

---

### 8. `img_shape`

**类型**: `list[list[tuple]]`（列表的列表，每个元素是形状元组）

**含义**: 每个样本的每个图像的形状（高度、宽度、通道数）

**结构说明**:
- 外层列表长度 = batch_size
- 内层列表长度 = 该样本的相机数量
- 每个元素是形状元组 `(H, W, C)` 或 `(H, W)`

**取值范围**: 
- 元组，例如: `(512, 1024, 3)` 或 `(512, 1024)`

**示例值**: `[[(512, 1024, 3), (512, 1024, 3), ...], ...]`

**说明**: 记录每个图像的尺寸信息

---

### 9. `ori_shape`

**类型**: `list[list[tuple]]`（列表的列表，每个元素是形状元组）

**含义**: 每个样本的每个图像的原始形状（resize前）

**结构说明**: 与 `img_shape` 相同

**说明**: 记录原始图像尺寸，用于后续处理（如坐标变换）

---

### 10. `pad_shape`

**类型**: `list[list[tuple]]`（列表的列表，每个元素是形状元组）

**含义**: 每个样本的每个图像的填充后形状（如果进行了padding）

**结构说明**: 与 `img_shape` 相同

**说明**: 记录填充后的图像尺寸，用于对齐不同尺寸的图像

---

### 11. `filename`

**类型**: `list[list[str]]`（列表的列表）

**含义**: 与 `img_filename` 相同，是 `img_filename` 的副本

**说明**: 保留字段，用于兼容性

---

### 12. `img_norm_cfg`

**类型**: `dict`（字典）

**含义**: 图像归一化配置

**结构**:
```python
{
    "mean": numpy.ndarray,  # 均值数组，形状 (num_channels,)
    "std": numpy.ndarray,   # 标准差数组，形状 (num_channels,)
    "to_rgb": bool          # 是否转换为RGB
}
```

**取值范围**:
- `mean`: 浮点数数组，通常为 `[0.0, 0.0, 0.0]`（未归一化）
- `std`: 浮点数数组，通常为 `[1.0, 1.0, 1.0]`（未归一化）
- `to_rgb`: `False`（图像已转换为RGB）

**说明**: 记录图像归一化参数，用于后续数据增强

---

### 13. `scale_factor`

**类型**: `float`（浮点数）

**含义**: 图像缩放因子

**取值范围**: 
- 通常为 `1.0`（当前实现中未使用缩放）

**说明**: 记录图像缩放比例

---

### 14. `gt_masks_3d`

**类型**: `list[numpy.ndarray]`（列表，每个元素是3D占用标签数组）

**含义**: 每个样本的3D占用网格Ground Truth标签（下采样版本）

**结构说明**:
- 列表长度 = batch_size
- 每个元素是形状为 `(H, W, D)` 的numpy数组
  - `H`: Y方向体素数（横向）
  - `W`: X方向体素数（前向）
  - `D`: Z方向体素数（高度）

**形状计算**:
- 根据 `occ_grid_config` 计算:
  - `xbound = [-21.0, 110.0, 0.3]` (min, max, voxel_size_downsample)
  - `ybound = [-24.0, 24.0, 0.3]`
  - `zbound = [-1.0, 2.6, 0.15]`
  - `W = ceil((110.0 - (-21.0)) / 0.3) = 437`
  - `H = ceil((24.0 - (-24.0)) / 0.3) = 160`
  - `D = ceil((2.6 - (-1.0)) / 0.15) = 24`
  - 因此形状约为 `(160, 437, 24)`（注意：代码中会transpose，实际可能是 `(437, 160, 24)`）

**数据类型**: `numpy.int64`

**数值范围**: 
- `0`: 自由空间（free）
- `1`: 可行驶区域（drivable）
- `2`: 可移动物体（moveable_object）
- `3`: 不可移动物体（unmoveable_object）
- `4`: 植被（vegetation）
- `5`: 其他（other）
- `255`: 忽略/未知（ignore）

**示例值**:
```python
[
    array([[[0, 0, 1, ...], ...], ...], shape=(160, 437, 24), dtype=int64),
    ...
]
```

**说明**: 3D占用网格标签，用于监督学习。每个体素对应一个类别标签。

---

### 15. `voxel_valid_mask`

**类型**: `list[numpy.ndarray]`（列表，每个元素是布尔数组）

**含义**: 每个样本的3D占用网格有效性掩码（下采样版本）

**结构说明**:
- 列表长度 = batch_size
- 每个元素是形状为 `(H, W, D)` 的布尔数组（与 `gt_masks_3d` 形状相同）

**数据类型**: `numpy.bool_`

**数值范围**: 
- `True`: 该体素有效（有标签数据）
- `False`: 该体素无效（无标签数据，应忽略）

**说明**: 标记哪些体素有有效的标签数据，用于损失计算时过滤无效体素

---

### 16. `gt_masks_3d_upscale`

**类型**: `list[numpy.ndarray]`（列表，每个元素是3D占用标签数组）

**含义**: 每个样本的3D占用网格Ground Truth标签（高分辨率版本）

**结构说明**:
- 列表长度 = batch_size
- 每个元素是形状为 `(H_upscale, W_upscale, D_upscale)` 的numpy数组
  - 分辨率是 `gt_masks_3d` 的2倍（X和Y方向，Z方向相同）

**形状计算**:
- 使用 `voxel_size`（0.15m）而不是 `voxel_size_downsample`（0.3m）
- `W_upscale = ceil((110.0 - (-21.0)) / 0.15) = 874`
- `H_upscale = ceil((24.0 - (-24.0)) / 0.15) = 320`
- `D_upscale = D = 24`
- 因此形状约为 `(320, 874, 24)`

**数据类型**: `numpy.int64`

**数值范围**: 与 `gt_masks_3d` 相同

**说明**: 高分辨率的3D占用网格标签，用于更精细的监督学习

---

### 17. `voxel_valid_mask_upscale`

**类型**: `list[numpy.ndarray]`（列表，每个元素是布尔数组）

**含义**: 每个样本的3D占用网格有效性掩码（高分辨率版本）

**结构说明**:
- 列表长度 = batch_size
- 每个元素是形状为 `(H_upscale, W_upscale, D_upscale)` 的布尔数组（与 `gt_masks_3d_upscale` 形状相同）

**数据类型**: `numpy.bool_`

**数值范围**: 
- `True`: 该体素有效
- `False`: 该体素无效

**说明**: 高分辨率版本的有效性掩码

---

### 18. `fine_gt_masks_3d`

**类型**: `list[numpy.ndarray]`（列表，每个元素是3D占用标签数组）

**含义**: 每个样本的精细类别3D占用网格标签（下采样版本）

**结构说明**:
- 列表长度 = batch_size
- 每个元素是形状为 `(H, W, D)` 的numpy数组（与 `gt_masks_3d` 形状相同）

**数据类型**: `numpy.int64`

**数值范围**: 
- `0`: 无精细类别（使用粗类别）
- `1, 2, 3, ...`: 精细类别ID（根据 `forground_classes` 配置）
  - 例如: `1` = 第一个前景类别，`2` = 第二个前景类别，等等
- 特殊值（如果配置了）:
  - `unmanned_bicycle_label`: 无人自行车类别
  - `other_unmovable_label`: 其他不可移动物体类别

**说明**: 精细类别的占用标签，用于更细粒度的分类任务。如果未配置 `forground_classes`，则全为0。

---

### 19. `fine_gt_masks_3d_upscale`

**类型**: `list[numpy.ndarray]`（列表，每个元素是3D占用标签数组）

**含义**: 每个样本的精细类别3D占用网格标签（高分辨率版本）

**结构说明**:
- 列表长度 = batch_size
- 每个元素是形状为 `(H_upscale, W_upscale, D_upscale)` 的numpy数组（与 `gt_masks_3d_upscale` 形状相同）

**数据类型**: `numpy.int64`

**数值范围**: 与 `fine_gt_masks_3d` 相同

**说明**: 高分辨率版本的精细类别标签

---

### 20. `gt_depth`（可选）

**类型**: `list[list[numpy.ndarray]]`（列表的列表，每个元素是深度图数组）

**含义**: 每个样本的每个相机的深度Ground Truth标签（如果启用了 `LoadDepthGTFromPCD`）

**结构说明**:
- 外层列表长度 = batch_size
- 内层列表长度 = 该样本的相机数量
- 每个元素是形状为 `(H_downsampled, W_downsampled)` 的numpy数组

**形状计算**:
- 原始图像尺寸: `(512, 1024)` 或 `(1024, 2048)`
- 下采样率: 16（如果高度为512）或 8（如果高度为1024）
- `H_downsampled = H // downsample`
- `W_downsampled = W // downsample`
- 例如: `(512, 1024)` → `(32, 64)`

**数据类型**: `numpy.float32`

**数值范围**: 
- 归一化后的深度值: `(depth - bound_d_min) / depth_grid_size`
- `bound_d_min = 0.1`（米）
- `depth_grid_size = 0.5`（米）
- 实际深度范围: `[0.1, 100.0]` 米
- 归一化后范围: `[0.0, 199.8]`（近似）
- `0.0`: 表示无效深度（无深度数据）

**说明**: 从激光雷达点云投影到图像平面生成的深度图，用于深度监督学习

---

### 21. `seg`（可选）

**类型**: `list[list[numpy.ndarray]]`（列表的列表，每个元素是分割标签数组）

**含义**: 每个样本的每个相机的语义分割Ground Truth标签（如果启用了 `LoadMultiViewSegLabel`）

**结构说明**:
- 外层列表长度 = batch_size
- 内层列表长度 = 该样本的相机数量
- 每个元素是形状为 `(H, W)` 的numpy数组（与对应图像的高度和宽度相同）

**数据类型**: `numpy.int64`

**数值范围**: 
- `0, 1, 2, ...`: 语义类别ID（根据 `cate_mapping` 配置）
- `255`: 忽略/背景（`ignore_label`）

**说明**: 多视角语义分割标签，用于语义分割监督学习

---

### 22. `seg_ignore_label`（可选）

**类型**: `int`（整数）

**含义**: 语义分割标签中的忽略值

**取值范围**: 
- 通常为 `255`

**说明**: 用于标识分割标签中应忽略的像素

---

### 23. `depth`（可选）

**类型**: `list[list[numpy.ndarray]]`（列表的列表，每个元素是深度标签数组）

**含义**: 每个样本的每个相机的深度标签（如果启用了 `LoadMultiViewDepthLabel`）

**结构说明**:
- 外层列表长度 = batch_size
- 内层列表长度 = 该样本的相机数量
- 每个元素是形状为 `(H_downsampled, W_downsampled)` 的numpy数组

**数据类型**: `numpy.float32` 或 `numpy.int64`（取决于配置）

**数值范围**: 
- 如果使用线性归一化: 与 `gt_depth` 相同
- 如果使用对数归一化: 根据 `grid_config["dbound"]` 计算的对数值

**说明**: 从预存的深度标签图像加载的深度标签（与 `gt_depth` 不同，`gt_depth` 是从点云实时计算的）

---

### 24. `lidar_points`（可选）

**类型**: `list[numpy.ndarray]`（列表，每个元素是点云数组）

**含义**: 每个样本的激光雷达点云数据（如果加载了）

**结构说明**:
- 列表长度 = batch_size
- 每个元素是形状为 `(N, 3)` 的numpy数组
  - `N`: 点云中的点数（每个样本不同）
  - `3`: x, y, z坐标

**数据类型**: `numpy.float32`

**数值范围**: 
- x, y, z坐标（米），通常在 `[-100, 100]` 范围内

**说明**: 原始激光雷达点云，用于生成深度标签或可见性掩码

---

### 25. `gt_bboxes_3d`（可选）

**类型**: `list[torch.Tensor]` 或 `list[Qcraft3DBoxes]`（列表，每个元素是3D检测框）

**含义**: 每个样本的3D检测框Ground Truth（如果加载了）

**结构说明**:
- 列表长度 = batch_size
- 每个元素是形状为 `(N, 7)` 或 `(N, 9)` 的torch.Tensor或Qcraft3DBoxes对象
  - `N`: 检测框数量（每个样本不同）
  - `7`: [x, y, z, w, l, h, yaw]（中心点、尺寸、旋转角）
  - `9`: [x, y, z, w, l, h, yaw, vx, vy]（包含速度）

**数据类型**: `torch.float32`

**数值范围**: 
- x, y, z: 中心点坐标（米）
- w, l, h: 宽度、长度、高度（米）
- yaw: 旋转角（弧度）
- vx, vy: 速度（米/秒，如果存在）

**说明**: 3D目标检测框，用于占用标签的精细标注（如果配置了 `forground_classes`）

---

### 26. `gt_labels_3d`（可选）

**类型**: `list[torch.Tensor]`（列表，每个元素是类别标签张量）

**含义**: 每个样本的3D检测框对应的类别标签（如果加载了）

**结构说明**:
- 列表长度 = batch_size
- 每个元素是形状为 `(N,)` 的torch.Tensor
  - `N`: 检测框数量（与 `gt_bboxes_3d` 对应）

**数据类型**: `torch.int64`

**数值范围**: 
- `0, 1, 2, ...`: 类别ID（根据 `forground_classes` 配置）

**说明**: 3D检测框的类别标签，与 `gt_bboxes_3d` 一一对应

---

## 三、Batch Collate 说明

`custom_collate_fn` 函数会将多个样本（字典）合并成一个batch字典：

1. **列表类型**: 保持为列表（不进行stack）
   - 例如: `img`, `img_filename`, `gt_masks_3d` 等

2. **numpy数组类型**: 尝试stack成数组
   - 如果形状一致，则stack
   - 如果形状不一致，则保持为列表
   - 例如: `intrinsic` 可能被stack

3. **torch.Tensor类型**: 尝试stack成张量
   - 如果形状一致，则stack
   - 如果形状不一致，则保持为列表

4. **字典类型**: 保持为列表（不进行collate）

5. **其他类型**: 尝试转换为tensor，失败则保持原样

---

## 四、典型Batch示例

假设 `batch_size=4`，每个样本有6个相机视角：

```python
batch = {
    "data_format": "qds",
    "img_filename": [
        ["image/CAM_.../000.jpg", "image/CAM_.../000.jpg", ...],  # 样本0的6个相机
        ["image/CAM_.../001.jpg", "image/CAM_.../001.jpg", ...],  # 样本1的6个相机
        ["image/CAM_.../002.jpg", "image/CAM_.../002.jpg", ...],  # 样本2的6个相机
        ["image/CAM_.../003.jpg", "image/CAM_.../003.jpg", ...],  # 样本3的6个相机
    ],
    "sample_idx": ["000", "001", "002", "003"],
    "data_dir": "/mnt/data/.../train",
    "intrinsic": array([...], shape=(4, 6, 3, 3)),  # 如果stack成功
    "resized_intrinsic": [[array(3x3), ...], ...],  # 列表的列表
    "img": [
        [array(512, 1024, 3), array(512, 1024, 3), ...],  # 样本0的6个图像
        [array(512, 1024, 3), array(512, 1024, 3), ...],  # 样本1的6个图像
        ...
    ],
    "img_shape": [[(512, 1024, 3), ...], ...],
    "gt_masks_3d": [
        array(160, 437, 24),  # 样本0的占用标签
        array(160, 437, 24),  # 样本1的占用标签
        ...
    ],
    "voxel_valid_mask": [
        array(160, 437, 24, dtype=bool),  # 样本0的有效性掩码
        array(160, 437, 24, dtype=bool),  # 样本1的有效性掩码
        ...
    ],
    "gt_masks_3d_upscale": [
        array(320, 874, 24),  # 样本0的高分辨率占用标签
        ...
    ],
    "voxel_valid_mask_upscale": [
        array(320, 874, 24, dtype=bool),  # 样本0的高分辨率有效性掩码
        ...
    ],
    "fine_gt_masks_3d": [
        array(160, 437, 24),  # 样本0的精细类别标签
        ...
    ],
    "fine_gt_masks_3d_upscale": [
        array(320, 874, 24),  # 样本0的高分辨率精细类别标签
        ...
    ],
    # 可选字段（如果启用了相应transform）:
    # "gt_depth": [[array(32, 64), ...], ...],
    # "seg": [[array(512, 1024), ...], ...],
    # "depth": [[array(32, 64), ...], ...],
    # "lidar_points": [array(N, 3), ...],
    # "gt_bboxes_3d": [tensor(N, 7), ...],
    # "gt_labels_3d": [tensor(N,), ...],
}
```

---

## 五、坐标系统说明

### 3D占用网格坐标系统

- **X轴**: 前向（车辆前进方向），范围 `[-21.0, 110.0]` 米
- **Y轴**: 横向（车辆左侧为正），范围 `[-24.0, 24.0]` 米
- **Z轴**: 高度（向上为正），范围 `[-1.0, 2.6]` 米

### 体素索引

- 体素索引 `(w, h, d)` 对应物理坐标:
  - `x = w * voxel_size_x + min_x + voxel_size_x / 2`
  - `y = h * voxel_size_y + min_y + voxel_size_y / 2`
  - `z = d * voxel_size_z + min_z + voxel_size_z / 2`

### 图像坐标系统

- **u轴**: 图像宽度方向（从左到右），范围 `[0, W-1]`
- **v轴**: 图像高度方向（从上到下），范围 `[0, H-1]`

---

## 六、注意事项

1. **列表 vs 数组**: 由于不同样本可能有不同数量的相机或不同尺寸，许多字段保持为列表而不是stack成数组。

2. **形状转置**: 注意 `gt_masks_3d` 等占用标签在代码中会进行transpose操作，实际存储顺序可能与物理坐标顺序不同。

3. **可选字段**: 某些字段（如 `gt_depth`, `seg`, `depth`）只有在相应transform启用时才会存在。

4. **数据类型**: 注意区分 `numpy.ndarray` 和 `torch.Tensor`，某些字段可能是numpy数组，某些可能是torch张量。

5. **batch_size**: 实际batch大小可能小于配置的 `batch_size`（特别是在最后一个batch或使用 `drop_last=True` 时）。

---

## 七、快速参考表

| 键名 | 类型 | 形状/结构 | 含义 |
|------|------|-----------|------|
| `data_format` | str | - | 数据格式标识 |
| `img_filename` | list[list[str]] | (batch_size, num_cameras) | 图像文件路径 |
| `sample_idx` | list[str] | (batch_size,) | 帧ID |
| `data_dir` | str | - | 数据目录 |
| `intrinsic` | ndarray/list | (batch_size, num_cameras, 3, 3) | 相机内参 |
| `resized_intrinsic` | list[list[ndarray]] | (batch_size, num_cameras, 3, 3) | resize后内参 |
| `img` | list[list[ndarray]] | (batch_size, num_cameras, H, W, 3) | 图像数据 |
| `img_shape` | list[list[tuple]] | (batch_size, num_cameras) | 图像形状 |
| `gt_masks_3d` | list[ndarray] | (batch_size, H, W, D) | 占用标签（下采样） |
| `voxel_valid_mask` | list[ndarray] | (batch_size, H, W, D) | 有效性掩码（下采样） |
| `gt_masks_3d_upscale` | list[ndarray] | (batch_size, H*2, W*2, D) | 占用标签（高分辨率） |
| `voxel_valid_mask_upscale` | list[ndarray] | (batch_size, H*2, W*2, D) | 有效性掩码（高分辨率） |
| `fine_gt_masks_3d` | list[ndarray] | (batch_size, H, W, D) | 精细类别标签（下采样） |
| `fine_gt_masks_3d_upscale` | list[ndarray] | (batch_size, H*2, W*2, D) | 精细类别标签（高分辨率） |

---

**文档版本**: 1.0  
**最后更新**: 2024年  
**基于代码**: `dataloader.py`, `loading.py`, `config/dataloader_config.py`

