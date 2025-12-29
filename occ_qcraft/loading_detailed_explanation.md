# loading.py 详细技术文档

> **相关文档**：
> - 辅助函数详细说明：`loading_helper_functions_detailed.md`
> - 概述性说明：`loading_and_data_structure_explanation.md`
> - 代码流程图：`loading_code_flowchart.md`

## 目录
1. [概述](#概述)
2. [七个类的详细说明](#七个类的详细说明)
3. [类之间的关系与区别](#类之间的关系与区别)
4. [数学公式详解](#数学公式详解)
5. [输入输出详细说明](#输入输出详细说明)

---

## 概述

`loading.py` 实现了自动驾驶占用网格（Occupancy Grid）任务的数据加载管道。该模块包含7个核心数据加载类，分别处理不同类型的输入数据和标签。

### 数据格式
- **qds** (Qcraft Data Format): 使用PCD格式的占用标签，图像路径结构为 `image/{camera}/raw_image/{frame}.jpg`
- **lc** (Lidar Camera Format): 使用NPZ格式的预处理标签，可能包含不同的图像处理流程

---

## 七个类的详细说明

### 1. LoadOccGTFromPCD

#### 1.1 功能与动机

**功能**：从PCD（Point Cloud Data）文件加载占用网格的Ground Truth标签，并转换为体素网格格式。

**动机**：
- 占用网格预测需要将3D点云数据转换为规则的体素网格
- 点云数据包含每个点的类别标签和可见性信息
- 需要处理多个点映射到同一体素的情况（标签聚合）
- 支持多分辨率输出（下采样和高分辨率版本）

#### 1.2 实现细节

**核心流程**：

1. **路径推断与文件读取**
   ```python
   root_path = results["img_filename"][0].split("/image/")[0]
   pcd_name = results["img_filename"][0].split("/")[-1].replace(".jpg", ".pcd")
   occ_gt_path = os.path.join(root_path, "occ/labels", pcd_name)
   ```
   - 从图像路径推断对应的PCD标签文件路径
   - 读取PCD文件，提取点云坐标、标签、可见性等信息

2. **坐标变换**
   - 如果存在 `bda_transform`（Beam Data Augmentation变换），对点云坐标进行变换
   - 公式：`xyz' = BDA @ [xyz, 1]^T`

3. **标记点处理**
   - 识别标记点：`z == -10000.0` 的点（特殊标记）
   - 未标记区域：标记点且 `label == unlabeled_pillar` 的区域

4. **体素化**
   - **高分辨率版本**：使用 `voxel_size`（如0.15m）
     - 体素索引计算：`voxel_idx = floor((point - min_range) / voxel_size)`
   - **下采样版本**：使用 `voxel_size_downsample = voxel_size * 2`（如0.3m）
     - 对体素索引排序后，使用 `numba_label_mapping` 聚合标签

5. **标签聚合策略**（`numba_label_mapping`）
   - 如果体素内所有点都是 `ignore_label` → 输出 `ignore_label`
   - 如果体素内所有点都是 `free_label` → 输出 `free_label`
   - 如果只有 `free_label` 和 `ignore_label` → 输出 `free_label`
   - 否则 → 输出出现次数最多的非free/ignore标签

6. **后处理**
   - 范围过滤：过滤ego车辆范围内的点和超出网格范围的点
   - 可见性处理：合并激光雷达可见性掩码（可选）
   - 不可见区域映射：将不可见体素设置为指定标签
   - 可行驶区域过滤：在可行驶区域内过滤特定类别
   - 前景类别填充：在检测框内将类别0填充为moveable_object
   - 精细类别标注：根据3D检测框标注精细类别

#### 1.3 关键参数

- `occ_grid_config`: 占用网格配置
  - `xbound`: [min_x, max_x, voxel_size_x]
  - `ybound`: [min_y, max_y, voxel_size_y]
  - `zbound`: [min_z, max_z, voxel_size_z]
- `ignore_num_thre_per_voxel`: 每个体素的最小标签命中次数阈值
- `invisible_mapping_label`: 不可见区域的标签映射（-1: 使用原始标签, 255: 忽略, 0: 自由空间）

---

### 2. LoadDepthGTFromPCD

#### 2.1 功能与动机

**功能**：从激光雷达点云实时计算每个相机的深度图Ground Truth。

**动机**：
- 深度估计是占用网格预测的重要辅助任务
- 当没有预处理的深度标签文件时，从点云实时计算深度
- 需要将3D激光雷达点投影到2D图像平面
- 处理多个点投影到同一像素的情况（取最近深度）
- 支持多相机视图

**与LoadMultiViewDepthLabel的区别**：
- **LoadDepthGTFromPCD**：计算型，从点云实时计算，输出到`gt_depth`
- **LoadMultiViewDepthLabel**：加载型，从预存文件加载，输出到`depth`
- 两者可以同时使用，服务于不同的监督任务或不同的数据源

#### 2.2 实现细节

**核心流程**：

1. **坐标变换链**
   ```
   激光雷达坐标系 → 相机坐标系 → 图像坐标系
   ```
   - 变换矩阵：`lidar2img = e2c = inv(c2e) = inv(img2pose @ inv(intrinsic))`
   - 点云投影：`points_img = lidar2img @ [points_lidar, 1]^T`
   - 透视投影：`uv = points_img[:, :2] / points_img[:, 2]`

2. **深度图生成**
   - 过滤：保留在图像范围内且深度在有效范围内的点
   - 排序：按 `(u + v * width + depth / 100.0)` 排序，确保同一像素取最近深度
   - 去重：对同一像素的多个点，只保留第一个（最近的）

3. **下采样处理**
   - 根据图像高度选择下采样率：512高度用16倍，否则用8倍
   - 对每个下采样块取最小深度值
   - 归一化：`depth_norm = (depth - bound_d_min) / depth_grid_size`

#### 2.3 数学公式

**坐标变换**：
```
points_cam = lidar2img @ [points_lidar, 1]^T
uv = points_cam[:2] / points_cam[2]
```

**深度归一化**：
```
depth_norm = (depth - d_min) / d_grid_size
```

---

### 3. LoadOccGTFromLCData

#### 3.1 功能与动机

**功能**：从LC格式的预处理NPZ文件加载占用网格标签。

**动机**：
- LC格式数据已经预处理为体素网格格式，无需从点云转换
- 支持不同空间范围的标签对齐（`lpai_gt_padding`）
- 使用优先级高度采样压缩高度维度

#### 3.2 实现细节

**核心流程**：

1. **NPZ文件读取**
   - `labels`: 下采样版本的标签数组
   - `invalid`: 无效体素的字节数组（需要解包为位掩码）
   - `upsample`: 高分辨率版本的标签数组

2. **标签对齐**（`lpai_gt_padding`）
   - 计算原始标签空间范围与目标空间范围的交集
   - 将原始标签复制到目标网格的对应位置
   - 未覆盖区域填充为 `lpai_padding_label`（200）

3. **优先级高度采样**（`priority_height_sampling`）
   - 将高度维度从Z压缩到Z/2
   - 对每对相邻的z层（z*2和z*2+1），按优先级选择标签
   - 优先级顺序（从高到低）：moveable_object > unmoveable_object > fence > vegetation > undrivable > drivable > other

#### 3.3 与LoadOccGTFromPCD的区别

| 特性 | LoadOccGTFromPCD | LoadOccGTFromLCData |
|------|------------------|---------------------|
| 输入格式 | PCD点云文件 | NPZ预处理数组 |
| 数据格式 | qds | lc |
| 体素化 | 需要从点云转换 | 已预处理 |
| 高度采样 | 无 | 有（优先级采样） |
| 标签对齐 | 无 | 有（lpai_gt_padding） |

---

### 4. LoadMultiViewUndistortImageWithResize

#### 4.1 功能与动机

**功能**：加载多视角图像，进行去畸变、虚拟相机转换、裁剪和缩放处理。

**动机**：
- 鱼眼相机存在畸变，需要去畸变处理
- 支持虚拟相机转换，统一不同相机的视角
- 基于FOV进行图像裁剪和缩放，统一图像尺寸

#### 4.2 实现细节

**核心流程**：

1. **虚拟相机转换**（`convert_virtual_camera`）
   - 构建虚拟相机内参：`K_vir = [[vfl, 0, vcx], [0, vfl, vcy], [0, 0, 1]]`
   - 构建虚拟相机外参：`E_vir`（使用欧拉角旋转）
   - 计算虚拟相机到源相机的旋转：`vir2src = inv(E_src) @ E_vir`
   - 生成畸变映射：`get_mapping_distortion`
   - 使用 `cv2.remap` 进行图像重映射

2. **畸变映射计算**（`get_mapping_distortion`）
   - 从虚拟图像像素坐标反投影到虚拟相机坐标系
   - 转换到源相机坐标系
   - 使用 `cv2.fisheye.projectPoints` 投影到源图像平面

3. **基于FOV的图像裁剪和缩放**（`image_resize_based_on_fov`）
   - 计算焦距：`f = (width / 2) / tan(FOV / 2)`
   - 根据输出FOV计算输出图像宽度
   - 进行裁剪或填充
   - 更新相机内参

#### 4.3 数学公式

**焦距计算**：
```
f = (W / 2) / tan(FOV_rad / 2)
```

**输出图像宽度**（基于中心）：
```
W_out = 2 * f * tan(FOV_out_rad / 2)
```

**输出图像宽度**（基于边缘）：
```
W_out = W_in / 2 - f * tan((FOV_in / 2 - FOV_out) * π / 180)
```

**虚拟相机坐标反投影**：
```
y_cam = -(u - cx) * RECOVERED_X / f
z_cam = -(v - cy) * RECOVERED_X / f
x_cam = RECOVERED_X
```

**坐标变换**：
```
pts_src_cam = vir2src @ pts_vir_cam
```

---

### 5. LoadMultiViewVirtualImageWithResize

#### 5.1 功能与动机

**功能**：加载多视角图像并调整大小，适用于qds格式数据。

**动机**：
- qds格式数据可能已经预处理过，不需要去畸变
- 只需要根据目标分辨率调整图像大小
- 更新相机内参以匹配新的图像尺寸

#### 5.2 实现细节

**核心流程**：

1. **图像加载**
   - 从 `img_root` 路径加载图像
   - 支持磁盘和Petrel存储后端
   - BGR → RGB 转换

2. **分辨率调整**
   - 从 `camera_resolution` 字典获取目标分辨率
   - 计算缩放比例：`scale = original_size / target_size`
   - 使用 `cv2.resize` 调整图像大小
   - 更新内参：`intrinsic[:2, :] = intrinsic[:2, :] / scale`

#### 5.3 与LoadMultiViewUndistortImageWithResize的区别

| 特性 | LoadMultiViewUndistortImageWithResize | LoadMultiViewVirtualImageWithResize |
|------|----------------------------------------|-------------------------------------|
| 数据格式 | lc | qds |
| 去畸变 | 是（鱼眼去畸变） | 否 |
| 虚拟相机转换 | 是 | 否 |
| FOV裁剪 | 是 | 否 |
| 处理复杂度 | 高 | 低 |

---

### 6. LoadMultiViewSegLabel

#### 6.1 功能与动机

**功能**：加载多视角语义分割标签图像。

**动机**：
- 语义分割是占用网格预测的辅助任务
- 标签以BGR编码的图像格式存储，需要解码
- 需要与图像大小对齐

**数据格式兼容性**：
- 不检查`data_format`，可以在qds和lc两种格式中使用
- 只需要`img_filename`和`img`字段，这些由图像加载器提供

#### 6.2 实现细节

**核心流程**：

1. **路径推断**
   ```python
   root, tail = img_path.split("/image/", 1)
   cam_id = tail.split("/")[0]
   if not cam_id.endswith("_HR"):
       cam_id = cam_id + "_HR"
   mask_path = os.path.join(root, "occ", "image_seg", cam_id, fname + suffix)
   ```

2. **BGR到ID解码**
   ```python
   ID = R + 256 * G + 256 * 256 * B
   semantic_id = ID // 1000
   ```

3. **大小对齐**
   - 使用最近邻插值调整标签大小以匹配图像
   - 处理填充（如果需要）

4. **类别映射**（可选）
   - 使用 `cate_mapping` 字典将原始类别ID映射到目标类别ID

#### 6.3 数学公式

**BGR到ID解码**：
```
ID = R + 256 * G + 256² * B
semantic_id = floor(ID / 1000)
```

---

### 7. LoadMultiViewDepthLabel

#### 7.1 功能与动机

**功能**：从预存的深度标签图像文件加载多视角深度标签。

**动机**：
- 深度监督是占用网格预测的重要辅助任务
- 当数据集中有预处理的深度标签文件时，直接加载（通常质量更高）
- 深度标签以BGR编码的图像格式存储
- 需要下采样和归一化处理

**与LoadDepthGTFromPCD的区别**：
- **LoadMultiViewDepthLabel**：加载型，从预存文件加载，输出到`depth`
- **LoadDepthGTFromPCD**：计算型，从点云实时计算，输出到`gt_depth`
- 两者可以同时使用，服务于不同的监督任务或不同的数据源
- 需要设置`pv_depth_supervision=True`才会执行

#### 7.2 实现细节

**核心流程**：

1. **深度图加载与解码**
   - 从 `occ/depth/{camera_id}_HR/{frame}.png` 加载
   - BGR解码：`depth = (R + 256*G + 256²*B) / 1000`

2. **下采样处理**（`get_downsampled_gt_depth`）
   - 将图像划分为 `downsample × downsample` 的块
   - 对每个块取最小深度值（忽略0值）
   - 归一化深度值

3. **深度归一化**
   - **线性归一化**（`sid=False`）：
     ```
     depth_norm = (depth - (d_min - d_grid_size)) / d_grid_size
     ```
   - **对数归一化**（`sid=True`）：
     ```
     depth_norm = (log(depth) - log(d_min)) * (D-1) / log((d_max-1)/d_min) + 1
     ```

4. **分辨率自适应**
   - 低分辨率图像使用 `downsample_lowres`
   - 高分辨率图像使用 `downsample`

#### 7.3 数学公式

**深度归一化（线性）**：
```
depth_norm = (depth - d_min + d_grid_size) / d_grid_size
```

**深度归一化（对数）**：
```
depth_norm = (ln(depth) - ln(d_min)) * (D-1) / ln((d_max-1)/d_min) + 1
```

**下采样（取最小值）**：
```
depth_downsampled[i, j] = min(depth[i*ds:(i+1)*ds, j*ds:(j+1)*ds])
```

---

## 类之间的关系与区别

### 数据格式对应关系

```
qds格式数据流：
  LoadMultiViewVirtualImageWithResize (图像)
    ↓
  LoadOccGTFromPCD (占用标签)
    ↓
  LoadDepthGTFromPCD (深度标签-计算型，可选)
    ↓
  LoadMultiViewSegLabel (分割标签，可选)
    ↓
  LoadMultiViewDepthLabel (深度标签-加载型，可选)

lc格式数据流：
  LoadMultiViewUndistortImageWithResize (图像)
    ↓
  LoadOccGTFromLCData (占用标签)
    ↓
  LoadMultiViewSegLabel (分割标签，可选)
    ↓
  LoadMultiViewDepthLabel (深度标签-加载型，可选)
```

**重要说明**：

1. **LoadDepthGTFromPCD vs LoadMultiViewDepthLabel 的区别**：
   - **LoadDepthGTFromPCD**（计算型深度）：
     - **数据来源**：从激光雷达点云实时计算
     - **输入**：`lidar_points`（激光雷达点云）
     - **处理方式**：将点云投影到图像平面，计算每个像素的深度
     - **输出键**：`gt_depth`
     - **归一化**：`(depth - bound_d_min) / depth_grid_size`（线性归一化）
     - **用途**：当没有预处理的深度标签文件时，从点云生成深度
     - **数据格式要求**：qds（因为需要lidar_points）
   
   - **LoadMultiViewDepthLabel**（加载型深度）：
     - **数据来源**：从预存的深度标签图像文件加载
     - **输入**：`occ/depth/{camera_id}_HR/{frame}.png`（BGR编码的深度图像）
     - **处理方式**：解码BGR图像，下采样，归一化
     - **输出键**：`depth`
     - **归一化**：支持线性和对数两种方式（由`sid`参数控制）
     - **用途**：加载预处理的深度标签（通常质量更高或包含额外信息）
     - **数据格式要求**：qds或lc（不检查data_format）
     - **启用条件**：需要设置`pv_depth_supervision=True`

   **使用场景**：
   - 如果数据集中有预处理的深度标签文件，使用`LoadMultiViewDepthLabel`
   - 如果没有预处理文件，使用`LoadDepthGTFromPCD`从点云计算
   - 两者可以同时使用，输出到不同的键（`gt_depth`和`depth`），用于不同的监督任务

   **监督任务区别详解**：
   
   - **LoadDepthGTFromPCD (`gt_depth`)** - **深度回归监督**：
     - **任务类型**：连续深度值回归（Regression）
     - **输出格式**：连续浮点数值，范围 `[0, (d_max - d_min) / d_grid_size]`
     - **归一化方式**：线性归一化 `(depth - d_min) / d_grid_size`
     - **损失函数**：通常使用L1/L2损失、Smooth L1损失等回归损失
     - **模型输出**：模型预测连续的深度值，与ground truth进行数值比较
     - **应用场景**：用于训练模型预测精确的深度值，适用于需要精确深度估计的任务
     - **下采样率**：16倍（图像高度512）或8倍
     - **特点**：值域连续，可以表示任意深度值（在范围内）
   
   - **LoadMultiViewDepthLabel (`depth`)** - **深度分类监督**：
     - **任务类型**：离散深度区间分类（Classification）
     - **输出格式**：离散化的深度值，范围 `[0, depth_channels]`，被限制为整数区间
     - **归一化方式**：线性或对数归一化后，被限制在 `depth_channels` 范围内
     - **损失函数**：通常使用交叉熵损失（Cross-Entropy Loss）等分类损失
     - **模型输出**：模型预测深度所属的离散区间（bin），与ground truth进行类别比较
     - **应用场景**：用于训练模型预测深度区间，适用于占用网格预测等需要深度先验的任务
     - **下采样率**：可配置（`downsample`或`downsample_lowres`），支持不同分辨率
     - **特点**：值域离散，深度被划分为有限个区间（bins）
     - **参数说明**：`depth_channels` 参数定义了深度区间的数量，类似于分类任务的类别数
   
   **任务差异程度**：
   
   **1. 本质差异**：
   - **回归任务** (`gt_depth`)：预测连续的深度值，目标是尽可能接近真实的深度数值
   - **分类任务** (`depth`)：预测深度所属的离散区间，目标是正确分类到对应的深度bin
   
   **2. 输出格式差异**：
   - **`gt_depth`**：
     - 数据类型：`float32`
     - 值域：连续，`[0, (d_max-d_min)/d_grid_size]`
     - 示例：`[0.5, 1.2, 3.7, 8.9, ...]`（任意浮点数）
     - 精度：可以表示任意精度的深度值
   
   - **`depth`**：
     - 数据类型：`float32`（但值被离散化）
     - 值域：离散，`[0, depth_channels]`，通常为整数或接近整数的值
     - 示例：`[0, 1, 2, 5, 10, ...]`（深度区间的索引）
     - 精度：只能表示有限个深度区间（由`depth_channels`决定，如80个区间）
   
   **3. 损失函数差异**：
   - **回归损失**（用于`gt_depth`）：
     - L1 Loss: `|pred - gt|`
     - L2 Loss: `(pred - gt)²`
     - Smooth L1 Loss: 结合L1和L2的优点
     - 特点：直接比较预测值和真实值的数值差异
   
   - **分类损失**（用于`depth`）：
     - Cross-Entropy Loss: `-log(P(gt_bin))`
     - Focal Loss: 处理类别不平衡问题
     - 特点：比较预测的概率分布和真实类别
   
   **4. 模型架构差异**：
   - **回归头**（用于`gt_depth`）：
     - 输出层：单个神经元，输出连续值
     - 激活函数：通常无激活或ReLU（确保非负）
     - 输出形状：`[B, N, H, W, 1]`（每个像素一个深度值）
   
   - **分类头**（用于`depth`）：
     - 输出层：`depth_channels + 1`个神经元（+1可能用于无效深度）
     - 激活函数：Softmax（输出概率分布）
     - 输出形状：`[B, N, H, W, depth_channels+1]`（每个像素一个概率分布）
   
   **5. 精度要求差异**：
   - **回归任务**：要求精确的深度值，误差越小越好
     - 例如：真实深度10.5米，预测10.3米，误差0.2米
     - 评估指标：MAE（平均绝对误差）、RMSE（均方根误差）
   
   - **分类任务**：只需要正确的深度区间，不关心区间内的具体值
     - 例如：真实深度在[10, 11)米区间（bin=10），预测也在bin=10即可
     - 评估指标：准确率（Accuracy）、IoU（如果按区间计算）
   
   **6. 应用场景差异**：
   - **`gt_depth`（回归）**：
     - 用于需要精确深度估计的下游任务
     - 3D目标检测：需要精确的3D位置
     - SLAM：需要精确的深度进行定位和建图
     - 路径规划：需要精确的距离信息
     - 特点：深度是主要输出，需要高精度
   
   - **`depth`（分类）**：
     - 用于占用网格预测等需要深度先验的任务
     - 占用网格预测：深度作为辅助信息，帮助理解场景结构
     - 多任务学习：深度分类作为辅助任务，提升主任务性能
     - 特点：深度是辅助信息，只需要粗略的深度区间即可
   
   **7. 计算复杂度差异**：
   - **回归任务**：计算简单，直接输出一个值
   - **分类任务**：需要计算所有类别的概率，计算量更大（但通常`depth_channels`不会太大，如80）
   
   **8. 训练稳定性差异**：
   - **回归任务**：对异常值敏感，需要仔细设计损失函数
   - **分类任务**：相对稳定，但可能存在类别不平衡问题
   
   **同时使用的意义**：
   - **多任务学习**：可以同时进行两种监督，提高模型的鲁棒性和泛化能力
   - **互补性**：
     - 回归监督提供精确的深度信息，帮助模型学习细粒度的深度特征
     - 分类监督提供深度区间的先验知识，帮助模型理解场景的深度结构
   - **训练策略**：
     - 可以设置不同的损失权重，平衡两种监督的影响
     - 可以在不同训练阶段使用不同的监督方式
   - **实际效果**：
     - 两种监督互补，帮助模型更好地理解场景的深度结构
     - 提高模型在深度相关任务上的性能

2. **lc格式数据流扩展**：
   - `LoadMultiViewSegLabel`和`LoadMultiViewDepthLabel`不检查`data_format`，因此可以在lc格式中使用
   - 它们只需要`img_filename`和`img`字段，这些在lc格式中由`LoadMultiViewUndistortImageWithResize`提供
   - 因此lc格式数据流可以完整支持分割和深度标签加载

### 功能对比表

| 类名 | 输入数据 | 输出数据 | 主要处理 | 数据格式 |
|------|----------|----------|----------|----------|
| LoadOccGTFromPCD | PCD点云 | 占用网格标签 | 体素化、标签聚合 | qds |
| LoadOccGTFromLCData | NPZ数组 | 占用网格标签 | 标签对齐、高度采样 | lc |
| LoadDepthGTFromPCD | 激光雷达点云 | 深度图 | 投影、下采样 | qds |
| LoadMultiViewUndistortImageWithResize | 原始图像 | 去畸变图像 | 去畸变、虚拟相机、FOV裁剪 | lc |
| LoadMultiViewVirtualImageWithResize | 原始图像 | 调整大小图像 | 缩放、内参更新 | qds |
| LoadMultiViewSegLabel | BGR标签图像 | 分割标签数组 | BGR解码、大小对齐 | qds/lc（不检查格式） |
| LoadMultiViewDepthLabel | BGR深度图像 | 深度图数组 | BGR解码、下采样、归一化 | qds/lc（不检查格式） |

### 依赖关系

```
LoadOccGTFromPCD
  ├─ 依赖: img_filename (来自图像加载器)
  ├─ 可选依赖: lidar_points (用于可见性掩码)
  ├─ 可选依赖: gt_bboxes_3d, gt_labels_3d (用于精细标注)
  └─ 输出: gt_masks_3d, gt_masks_3d_upscale, fine_gt_masks_3d

LoadDepthGTFromPCD
  ├─ 依赖: img, img_shape, img2pose, intrinsic (来自图像加载器)
  ├─ 依赖: lidar_points
  └─ 输出: gt_depth

LoadOccGTFromLCData
  ├─ 依赖: occupancy.occ_label_path (来自数据配置)
  ├─ 可选依赖: lidar_points
  └─ 输出: gt_masks_3d, gt_masks_3d_upscale

LoadMultiViewSegLabel
  ├─ 依赖: img_filename, img (来自图像加载器)
  ├─ 可选依赖: pad_shape
  └─ 输出: seg, seg_ignore_label
  └─ 数据格式: qds或lc（不检查格式）

LoadMultiViewDepthLabel
  ├─ 依赖: img_filename, img (来自图像加载器)
  ├─ 可选依赖: pad_shape
  ├─ 启用条件: pv_depth_supervision=True
  └─ 输出: depth
  └─ 数据格式: qds或lc（不检查格式）

LoadMultiViewUndistortImageWithResize
  ├─ 依赖: img_filename, intrinsic, cam_distort, extrinsic, cams_load_order
  └─ 输出: img, intrinsic, img2pose, resized_intrinsic

LoadMultiViewVirtualImageWithResize
  ├─ 依赖: img_filename, intrinsic
  └─ 输出: img, resized_intrinsic

LoadMultiViewSegLabel
  ├─ 依赖: img_filename, img (来自图像加载器)
  └─ 输出: seg, seg_ignore_label

LoadMultiViewDepthLabel
  ├─ 依赖: img_filename, img (来自图像加载器)
  └─ 输出: depth
```

---

## 数学公式详解

### 1. 体素化公式

**体素索引计算**：
```
voxel_idx[i] = floor((point[i] - min_range[i]) / voxel_size[i])
```

**体素中心坐标**：
```
voxel_center[i] = voxel_idx[i] * voxel_size[i] + min_range[i] + voxel_size[i] / 2
```

**网格大小计算**：
```
grid_size[i] = round((max_range[i] - min_range[i]) / voxel_size[i])
```

### 2. 坐标变换公式

**BDA变换（齐次坐标）**：
```
[x', y', z', 1]^T = BDA @ [x, y, z, 1]^T
```

**激光雷达到图像投影**：
```
points_cam = lidar2img @ [points_lidar, 1]^T
uv = points_cam[:2] / points_cam[2]
```

**虚拟相机坐标反投影**：
```
y_cam = -(u - cx) * X_rec / f
z_cam = -(v - cy) * X_rec / f
x_cam = X_rec
```

**虚拟相机到源相机变换**：
```
pts_src_cam = vir2src @ pts_vir_cam
vir2src = (inv(E_src) @ E_vir)[:3, :3]
```

### 3. 相机模型公式

**针孔相机模型**：
```
u = f * x_cam / z_cam + cx
v = f * y_cam / z_cam + cy
```

**焦距计算**：
```
f = (W / 2) / tan(FOV / 2)
```

**FOV到图像宽度**：
```
W = 2 * f * tan(FOV / 2)
```

### 4. 深度处理公式

**深度归一化（线性）**：
```
depth_norm = (depth - d_min + d_grid_size) / d_grid_size
```

**深度归一化（对数）**：
```
depth_norm = (ln(depth) - ln(d_min)) * (D-1) / ln((d_max-1)/d_min) + 1
```

**下采样（最小深度）**：
```
depth_down[i, j] = min(depth[i*ds:(i+1)*ds, j*ds:(j+1)*ds])
```

### 5. 标签编码/解码公式

**BGR到ID编码**：
```
ID = R + 256 * G + 256² * B
```

**ID到语义类别**：
```
semantic_id = floor(ID / 1000)
instance_id = ID % 1000
```

### 6. 图像处理公式

**缩放后内参更新**：
```
K_new = K_old / scale
```

**裁剪后内参更新**：
```
cx_new = cx_old - crop_x
cy_new = cy_old - crop_y
```

**填充后内参更新**：
```
cx_new = cx_old + pad_x
cy_new = cy_old + pad_y
```

---

## 输入输出详细说明

### LoadOccGTFromPCD

#### 输入（results字典）

| 键 | 类型 | 形状/格式 | 取值范围 | 含义 |
|----|------|-----------|----------|------|
| `data_format` | str | - | "qds" | 数据格式标识 |
| `img_filename` | list[str] | [N] | 图像文件路径列表 | 图像文件路径，用于推断标签路径 |
| `bda_transform` | np.ndarray | [4, 4] (可选) | 齐次变换矩阵 | BDA数据增强变换矩阵 |
| `lidar_points` | np.ndarray | [M, 3] (可选) | 3D点坐标 | 激光雷达点云，用于可见性掩码 |
| `gt_bboxes_3d` | Qcraft3DBoxes | [N_bbox, 7] (可选) | 3D检测框 | 用于精细类别标注 |
| `gt_labels_3d` | np.ndarray | [N_bbox] (可选) | 类别ID | 检测框对应的类别 |

#### 输出（results字典，新增/更新）

| 键 | 类型 | 形状 | 数据类型 | 取值范围 | 含义 |
|----|------|------|----------|----------|------|
| `gt_masks_3d` | np.ndarray | [H, W, D] | int64 | 0-254, 255 | 下采样占用标签，255=忽略 |
| `voxel_valid_mask` | np.ndarray | [H, W, D] | bool | True/False | 下采样可见性掩码 |
| `gt_masks_3d_upscale` | np.ndarray | [H_up, W_up, D] | int64 | 0-254, 255 | 高分辨率占用标签 |
| `voxel_valid_mask_upscale` | np.ndarray | [H_up, W_up, D] | bool | True/False | 高分辨率可见性掩码 |
| `fine_gt_masks_3d` | np.ndarray | [H, W, D] | int64 | 0-N_fine, 255 | 精细类别标签（下采样） |
| `fine_gt_masks_3d_upscale` | np.ndarray | [H_up, W_up, D] | int64 | 0-N_fine, 255 | 精细类别标签（高分辨率） |

**标签值含义**：
- `0`: 自由空间（free）
- `1`: 可行驶区域（drivable）
- `2`: 可移动物体（moveable_object）
- `3`: 不可移动物体（unmoveable_object）
- `4`: 植被（vegetation）
- `5`: 其他（other）
- `255`: 忽略/未知（ignore）

**维度说明**：
- `H, W`: 下采样网格的高度和宽度（体素大小×2）
- `H_up, W_up`: 高分辨率网格的高度和宽度（原始体素大小）
- `D`: 高度方向的体素数

---

### LoadDepthGTFromPCD

#### 输入（results字典）

| 键 | 类型 | 形状/格式 | 取值范围 | 含义 |
|----|------|-----------|----------|------|
| `img` | list[np.ndarray] | [N] | 图像数组列表 | 多视角图像 |
| `img_shape` | list[tuple] | [N, 2] | (H, W)元组列表 | 每个图像的形状 |
| `img2pose` | np.ndarray | [N, 4, 4] | 齐次变换矩阵 | 图像到姿态的变换 |
| `intrinsic` | np.ndarray | [N, 3, 3] | 相机内参矩阵 | 每个相机的内参 |
| `lidar_points` | np.ndarray | [M, 3] (可选) | 3D点坐标 | 激光雷达点云 |

#### 输出（results字典，新增）

| 键 | 类型 | 形状 | 数据类型 | 取值范围 | 含义 |
|----|------|------|----------|----------|------|
| `gt_depth` | list[np.ndarray] | [N] | float32 | [0, (d_max-d_min)/d_grid_size] | 每个相机的深度图列表（连续值，用于回归） |

**深度图形状**：
- 下采样率：图像高度512用16倍，否则用8倍
- 形状：`[H // downsample, W // downsample]`
- 值：归一化后的深度值，0表示无效深度

---

### LoadOccGTFromLCData

#### 输入（results字典）

| 键 | 类型 | 形状/格式 | 取值范围 | 含义 |
|----|------|-----------|----------|------|
| `data_format` | str | - | "lc" | 数据格式标识 |
| `occupancy` | dict (可选) | - | - | 占用标签配置 |
| `occupancy.occ_label_path` | str | - | NPZ文件路径 | 标签文件路径 |
| `occupancy.min_extent` | list[float] | [3] | 空间范围最小值 | [min_x, min_y, min_z] |
| `occupancy.max_extent` | list[float] | [3] | 空间范围最大值 | [max_x, max_y, max_z] |
| `occupancy.voxel_size` | float | - | 体素大小（米） | 下采样体素大小 |
| `occupancy.upsample_voxel_size` | float | - | 体素大小（米） | 高分辨率体素大小 |
| `lidar_points` | np.ndarray | [M, 3] (可选) | 3D点坐标 | 激光雷达点云 |

#### 输出（results字典，新增/更新）

| 键 | 类型 | 形状 | 数据类型 | 取值范围 | 含义 |
|----|------|------|----------|----------|------|
| `gt_masks_3d` | np.ndarray | [H, W, D] | int64 | 0-254, 255 | 下采样占用标签 |
| `voxel_valid_mask` | np.ndarray | [H, W, D] | bool | True/False | 下采样可见性掩码 |
| `gt_masks_3d_upscale` | np.ndarray | [H_up, W_up, D//2] | int64 | 0-254, 255 | 高分辨率占用标签（高度压缩） |
| `voxel_valid_mask_upscale` | np.ndarray | [H_up, W_up, D//2] | bool | True/False | 高分辨率可见性掩码 |

**注意**：`gt_masks_3d_upscale` 的高度维度被压缩为 `D//2`（优先级高度采样）。

---

### LoadMultiViewUndistortImageWithResize

#### 输入（results字典）

| 键 | 类型 | 形状/格式 | 取值范围 | 含义 |
|----|------|-----------|----------|------|
| `data_format` | str | - | "lc" | 数据格式标识 |
| `img_filename` | list[str] | [N] | 图像文件路径列表 | 原始图像文件路径 |
| `intrinsic` | np.ndarray | [N, 3, 3] | 相机内参矩阵 | 每个相机的内参 |
| `cam_distort` | list[np.ndarray] | [N] | 畸变系数 | 每个相机的畸变系数 |
| `extrinsic` | np.ndarray | [N, 4, 4] | 齐次变换矩阵 | 每个相机的外参 |
| `cams_load_order` | list[str] | [N] | 相机名称列表 | 相机加载顺序 |

#### 输出（results字典，新增/更新）

| 键 | 类型 | 形状 | 数据类型 | 取值范围 | 含义 |
|----|------|------|----------|----------|------|
| `img` | list[np.ndarray] | [N_vcam] | float32/uint8 | [0, 255] | 处理后的图像列表 |
| `intrinsic` | np.ndarray | [N_vcam, 3, 3] | float32 | 相机内参矩阵 | 更新后的内参 |
| `img2pose` | np.ndarray | [N_vcam, 4, 4] | float32 | 齐次变换矩阵 | 图像到姿态的变换 |
| `resized_intrinsic` | np.ndarray | [N_vcam, 3, 3] | float32 | 相机内参矩阵 | 调整大小后的内参 |
| `img_shape` | list[tuple] | [N_vcam, 2] | (H, W) | 图像形状 | 每个图像的形状 |
| `ori_shape` | list[tuple] | [N_vcam, 2] | (H, W) | 原始形状 | 原始图像形状 |
| `pad_shape` | list[tuple] | [N_vcam, 2] | (H, W) | 填充后形状 | 填充后的形状 |
| `filename` | list[str] | [N_vcam] | 文件路径 | 图像文件路径 |
| `img_norm_cfg` | dict | - | - | 图像归一化配置 | mean, std, to_rgb |

**注意**：`N_vcam` 可能大于 `N`，因为一个源相机可能生成多个虚拟相机视图。

---

### LoadMultiViewVirtualImageWithResize

#### 输入（results字典）

| 键 | 类型 | 形状/格式 | 取值范围 | 含义 |
|----|------|-----------|----------|------|
| `data_format` | str | - | "qds" | 数据格式标识 |
| `img_filename` | list[str] | [N] | 图像文件路径列表 | 图像文件路径（相对路径） |
| `intrinsic` | np.ndarray | [N, 3, 3] | 相机内参矩阵 | 每个相机的内参 |

#### 输出（results字典，新增/更新）

| 键 | 类型 | 形状 | 数据类型 | 取值范围 | 含义 |
|----|------|------|----------|----------|------|
| `img` | list[np.ndarray] | [N] | float32/uint8 | [0, 255] | 调整大小后的图像列表 |
| `resized_intrinsic` | np.ndarray | [N, 3, 3] | float32 | 相机内参矩阵 | 调整大小后的内参 |
| `img_shape` | list[tuple] | [N, 2] | (H, W) | 图像形状 | 每个图像的形状 |
| `ori_shape` | list[tuple] | [N, 2] | (H, W) | 原始形状 | 原始图像形状 |
| `pad_shape` | list[tuple] | [N, 2] | (H, W) | 填充后形状 | 填充后的形状 |
| `filename` | list[str] | [N] | 文件路径 | 图像文件路径 |
| `img_norm_cfg` | dict | - | - | 图像归一化配置 | mean, std, to_rgb |
| `scale_factor` | float | - | 1.0 | 缩放因子 | 固定为1.0 |

---

### LoadMultiViewSegLabel

#### 输入（results字典）

| 键 | 类型 | 形状/格式 | 取值范围 | 含义 |
|----|------|-----------|----------|------|
| `img_filename` | list[str] | [N] | 图像文件路径列表 | 图像文件路径，用于推断标签路径 |
| `img` | list[np.ndarray] | [N] | 图像数组 | 图像数组，用于获取目标大小 |
| `pad_shape` | list[tuple] | [N, 2] (可选) | (H, W) | 填充后形状 | 用于填充标签 |

#### 输出（results字典，新增）

| 键 | 类型 | 形状 | 数据类型 | 取值范围 | 含义 |
|----|------|------|----------|----------|------|
| `seg` | list[np.ndarray] | [N] | int64 | 0-N_class, ignore_label | 每个相机的分割标签 |
| `seg_ignore_label` | int | - | int | 通常为255 | 忽略标签值 |

**分割标签形状**：与对应图像相同 `[H, W]`

**标签值含义**：
- `0` 到 `N_class-1`: 语义类别ID
- `ignore_label`（通常255）: 忽略/背景

---

### LoadMultiViewDepthLabel

#### 输入（results字典）

| 键 | 类型 | 形状/格式 | 取值范围 | 含义 |
|----|------|-----------|----------|------|
| `img_filename` | list[str] | [N] | 图像文件路径列表 | 图像文件路径，用于推断深度路径 |
| `img` | list[np.ndarray] | [N] | 图像数组 | 图像数组，用于获取目标大小 |
| `pad_shape` | list[tuple] | [N, 2] (可选) | (H, W) | 填充后形状 | 用于填充深度图 |

#### 输出（results字典，新增）

| 键 | 类型 | 形状 | 数据类型 | 取值范围 | 含义 |
|----|------|------|----------|----------|------|
| `depth` | list[np.ndarray] | [N] | float32 | [0, depth_channels] | 每个相机的深度图（离散值，用于分类） |

**深度图形状**：`[H // downsample, W // downsample]`

**深度值含义**：
- `0`: 无效深度
- `> 0`: 归一化后的深度值（线性或对数归一化）
- **注意**：值被限制在 `[0, depth_channels]` 范围内，表示深度所属的离散区间索引
- **用途**：用于深度分类任务，模型需要预测每个像素的深度属于哪个区间（bin）

---

## 总结

本文档详细说明了 `loading.py` 中7个核心数据加载类的功能、实现细节、数学公式和输入输出格式。这些类共同构成了占用网格预测任务的数据加载管道，支持多种数据格式和处理流程。

关键要点：
1. **数据格式区分**：qds和lc格式使用不同的加载器
2. **多分辨率支持**：占用标签支持下采样和高分辨率两个版本
3. **多任务支持**：同时支持占用、深度、分割等多个任务
4. **坐标变换链**：完整的3D到2D投影和变换流程
5. **标签聚合策略**：处理点云到体素的标签聚合问题

