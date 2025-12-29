# loading.py 代码与 nas/ 数据目录说明文档

> **注意**：本文档为概述性说明。如需详细了解每个类的功能、实现细节、数学公式和输入输出，请参考 `loading_detailed_explanation.md`。

## 一、概述

`loading.py` 是一个数据加载和处理模块，主要用于**自动驾驶占用网格（Occupancy Grid）**任务的训练数据加载。`nas/` 目录下存放的是与该代码配套的训练数据集。

本文档提供：
- 代码模块的总体概述
- 数据目录结构说明
- 基本的数据流程
- 快速参考信息

详细的技术文档（包括数学公式、详细流程图、输入输出规范）请参阅 `loading_detailed_explanation.md`。

## 二、loading.py 代码结构解析

### 2.1 主要功能模块

`loading.py` 包含以下几个核心的数据加载类：

#### 1. **LoadOccGTFromPCD** (行 173-418)
- **功能**：从 PCD（Point Cloud Data）文件加载占用网格的 Ground Truth 标签
- **数据格式**：`qds`（Qcraft Data Format）
- **主要处理流程**：
  1. 从 `occ/labels/` 目录读取 `.pcd` 格式的占用标签文件
  2. 将点云数据转换为体素网格（voxel grid）
  3. 生成两种分辨率的标签：
     - `gt_masks_3d`：下采样版本（体素大小 ×2）
     - `gt_masks_3d_upscale`：高分辨率版本（原始体素大小）
  4. 处理可见性掩码、类别映射、可行驶区域过滤等

**关键代码路径**：
```python
# 第 256-258 行：构建标签文件路径
root_path = results["img_filename"][0].split("/image/")[0]
pcd_name = results["img_filename"][0].split("/")[-1].replace(".jpg", ".pcd")
occ_gt_path = os.path.join(root_path, "occ/labels", pcd_name)
```

#### 2. **LoadOccGTFromLCData** (行 1044-1367)
- **功能**：从 LC（Lidar Camera）格式数据加载占用网格标签
- **数据格式**：`lc`
- **主要特点**：
  - 从 `.npz` 文件加载预处理的占用标签
  - 支持优先级高度采样（priority_height_sampling）
  - 处理不同体素尺寸的标签对齐

#### 3. **LoadDepthGTFromPCD** (行 925-1013)
- **功能**：从点云数据生成深度图的 Ground Truth
- **处理流程**：
  1. 将激光雷达点投影到相机坐标系
  2. 生成每个相机的深度图
  3. 进行下采样处理

#### 4. **LoadMultiViewUndistortImageWithResize** (行 1370-1632)
- **功能**：加载多视角图像并进行去畸变、裁剪、缩放处理
- **数据格式**：`lc`
- **主要处理**：
  - 虚拟相机转换（convert_virtual_camera）
  - 基于 FOV 的图像裁剪和缩放
  - 相机内参和外参的更新

#### 5. **LoadMultiViewVirtualImageWithResize** (行 1635-1730)
- **功能**：加载多视角虚拟图像并调整大小
- **数据格式**：`qds`
- **主要处理**：
  - 从 `img_root` 路径加载图像
  - 根据相机分辨率调整图像大小
  - 更新相机内参

#### 6. **LoadMultiViewSegLabel** (行 1733-1844)
- **功能**：加载多视角语义分割标签
- **处理流程**：
  1. 从 `occ/image_seg/` 目录加载分割标签图像
  2. 将 BGR 编码的标签转换为类别 ID
  3. 与图像大小对齐

#### 7. **LoadMultiViewDepthLabel** (行 1847-1992)
- **功能**：加载多视角深度标签
- **处理流程**：
  1. 从 `occ/depth/` 目录加载深度图
  2. 进行下采样和编码处理

### 2.2 辅助函数

#### **numba_label_mapping** (行 118-170)
- 使用 Numba JIT 加速的标签映射函数
- 将多个点映射到同一个体素时的标签聚合逻辑

#### **priority_height_sampling** (行 1016-1041)
- 按优先级对高度维度进行采样
- 优先保留障碍物标签而非道路标签

#### **crop_and_resize_image** (行 44-115)
- 基于输入/输出视场角（FOV）进行图像裁剪和缩放

## 三、nas/ 目录数据结构

### 3.1 目录结构

```
nas/
└── ad-labeldata/
    └── qmlp/
        └── occ_data_driving/
            └── 2772/
                └── fusion_251011/
                    └── 20250818_083149_Q3701-1150_1160/
                        └── train/
                            └── 20250818_083149_Q3701/
                                └── 20250818_083149_Q3701-1150_1160/
                                    ├── FILELIST                    # 数据文件列表
                                    ├── image/                      # 多视角相机图像
                                    │   ├── CAM_PBQ_FRONT_*_HR/     # 高分辨率相机
                                    │   └── CAM_PBQ_*/              # 普通分辨率相机
                                    ├── lidar/                      # 激光雷达点云
                                    │   ├── LDR_CENTER/             # 中心激光雷达
                                    │   ├── LDR_FRONT/               # 前向激光雷达
                                    │   └── LDR_*_BLIND/             # 盲区激光雷达
                                    ├── occ/                        # 占用网格相关数据
                                    │   ├── labels/                 # 占用标签（.pcd格式）
                                    │   ├── filtered_lidar/         # 过滤后的点云
                                    │   ├── lidar_seg/              # 点云分割结果
                                    │   └── occ-config.json         # 占用网格配置
                                    ├── label/                      # 3D检测标签
                                    │   └── lidar_label/            # 激光雷达检测框（.json）
                                    ├── pose/                       # 位姿信息
                                    │   ├── image_pose.json         # 图像位姿
                                    │   └── lidar_pose.json         # 激光雷达位姿
                                    ├── info/                       # 元信息
                                    │   ├── meta_info.json
                                    │   └── lite_run.json
                                    └── params/                     # 车辆参数
                                        └── vehicle_params.json
```

### 3.2 关键文件说明

#### **FILELIST**
- **格式**：每行一个相对路径，指向图像文件
- **示例**：
  ```
  image/CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99/raw_image/000.jpg
  image/CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99/raw_image/001.jpg
  ...
  ```
- **用途**：数据加载器读取此文件获取所有训练样本的路径列表

#### **occ/labels/*.pcd**
- **格式**：点云数据格式（PCD），包含：
  - `x, y, z`：点云坐标
  - `label`：占用类别标签（uint8）
  - `is_visible_in_fisheye_cameras`：相机可见性标记
  - `label_hit_num`：标签命中次数
  - `camera_visibility_mask`：相机可见性掩码（可选）
- **用途**：`LoadOccGTFromPCD` 类读取这些文件生成占用网格标签

#### **occ/occ-config.json**
- **内容**：占用网格的配置参数
  - `voxel_size`: 0.15（米）
  - `min_x, max_x, min_y, max_y, min_z, max_z`：空间范围
  - `mask_camera_ids`：用于掩码的相机ID列表
  - `valid_lidar_ids`：有效的激光雷达ID列表

#### **image/** 目录
- **结构**：每个相机一个子目录，包含 `raw_image/` 子目录
- **相机类型**：
  - `CAM_PBQ_FRONT_*`：前向相机
  - `CAM_PBQ_REAR_*`：后向相机
  - `*_HR`：高分辨率版本
- **文件格式**：`.jpg` 图像文件

#### **lidar/** 目录
- **结构**：每个激光雷达一个子目录
- **文件格式**：`.pcd` 点云文件
- **用途**：用于生成深度标签和可见性掩码

## 四、数据格式对应关系

### 4.1 数据格式标识

代码中通过 `results["data_format"]` 区分两种数据格式：

1. **`qds`** (Qcraft Data Format)
   - 使用 `LoadOccGTFromPCD` 加载占用标签
   - 使用 `LoadMultiViewVirtualImageWithResize` 加载图像
   - 数据路径结构：`{root}/image/{camera_id}/raw_image/{frame_id}.jpg`

2. **`lc`** (Lidar Camera Format)
   - 使用 `LoadOccGTFromLCData` 加载占用标签
   - 使用 `LoadMultiViewUndistortImageWithResize` 加载图像
   - 数据可能来自不同的预处理流程

### 4.2 路径推断逻辑

代码中通过路径推断来定位相关文件：

```python
# 从图像路径推断占用标签路径（LoadOccGTFromPCD）
root_path = results["img_filename"][0].split("/image/")[0]
pcd_name = results["img_filename"][0].split("/")[-1].replace(".jpg", ".pcd")
occ_gt_path = os.path.join(root_path, "occ/labels", pcd_name)

# 从图像路径推断分割标签路径（LoadMultiViewSegLabel）
root, tail = img_path.split("/image/", 1)
parts = tail.split("/")
cam_id = parts[0]
if not cam_id.endswith("_HR"):
    cam_id = cam_id + "_HR"
mask_path = os.path.join(root, "occ", "image_seg", cam_id, fname_wo_ext + suffix)
```

## 五、数据加载流程示例

### 5.1 QDS 格式数据加载流程

```
1. 读取 FILELIST，获取图像路径列表
   └─> image/CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99/raw_image/000.jpg

2. LoadMultiViewVirtualImageWithResize
   └─> 加载图像，调整大小，更新内参

3. LoadOccGTFromPCD
   └─> 推断路径：{root}/occ/labels/000.pcd
   └─> 读取点云，转换为体素网格
   └─> 生成 gt_masks_3d 和 gt_masks_3d_upscale

4. LoadMultiViewSegLabel（可选）
   └─> 推断路径：{root}/occ/image_seg/CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99_HR/000-panonet.png
   └─> 加载语义分割标签

5. LoadMultiViewDepthLabel（可选）
   └─> 推断路径：{root}/occ/depth/CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99_HR/000.png
   └─> 加载深度标签
```

## 六、关键概念说明

### 6.1 占用网格（Occupancy Grid）
- **定义**：将3D空间划分为规则的体素（voxel）网格，每个体素标记为占用/空闲/未知
- **用途**：用于自动驾驶中的障碍物检测和路径规划
- **分辨率**：
  - 下采样版本：体素大小 ×2（如 0.3m）
  - 高分辨率版本：原始体素大小（如 0.15m）

### 6.2 体素化（Voxelization）
- **过程**：将点云数据映射到3D网格中
- **处理**：多个点可能映射到同一个体素，需要聚合策略（如 majority voting）

### 6.3 可见性掩码（Visibility Mask）
- **作用**：标记哪些体素在相机视野内可见
- **来源**：
  - 点云中的 `is_visible_in_fisheye_cameras` 字段
  - 激光雷达点云投影到体素网格

### 6.4 类别映射（Label Mapping）
- **目的**：将原始数据中的类别ID映射到模型训练所需的类别ID
- **实现**：通过 `class_names_mapping` 字典进行转换

## 七、总结

1. **`loading.py`** 是一个完整的数据加载和处理管道，支持多种数据格式和任务类型
2. **`nas/` 目录** 包含完整的训练数据集，包括图像、点云、标签和配置文件
3. **数据格式**：主要支持 `qds` 和 `lc` 两种格式
4. **路径结构**：代码通过路径推断自动定位相关文件，要求数据目录结构符合约定

## 八、常见问题

### Q1: 如何确认数据是否与代码配套？
**A**: 检查以下几点：
- `FILELIST` 文件是否存在且格式正确
- `occ/labels/` 目录下是否有对应的 `.pcd` 文件
- 图像路径是否与 `FILELIST` 中的路径一致
- `occ/occ-config.json` 中的配置是否与代码中的 `occ_grid_config` 匹配

### Q2: 数据格式 `qds` 和 `lc` 的区别？
**A**: 
- `qds`：使用 PCD 格式的占用标签，图像路径结构为 `image/{camera}/raw_image/{frame}.jpg`
- `lc`：使用 NPZ 格式的预处理标签，可能包含不同的图像处理流程

### Q3: 如何添加新的数据加载器？
**A**: 
1. 继承或参考现有的加载器类
2. 使用 `@TRANSFORMS.register_module()` 装饰器注册
3. 实现 `__call__` 方法，接收和返回 `results` 字典
4. 在配置文件中引用新的加载器

