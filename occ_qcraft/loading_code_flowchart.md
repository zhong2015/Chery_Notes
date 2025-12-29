# loading.py 代码执行流程图

> **注意**：本文档提供简化的流程图。详细的执行流程、数学公式和输入输出说明请参考 `loading_detailed_explanation.md`。

## 一、LoadOccGTFromPCD 执行流程（QDS 格式）

```
输入: results 字典
  ├─ data_format == "qds"?
  │   └─ NO → 直接返回 results
  │
  ├─ occ_enable == True?
  │   └─ NO → 直接返回 results
  │
  └─ YES → 开始处理
      │
      ├─ 1. 构建标签文件路径
      │   ├─ 从 img_filename[0] 提取 root_path
      │   ├─ 提取图像文件名，替换为 .pcd
      │   └─ 构建: root_path/occ/labels/{pcd_name}
      │
      ├─ 2. 检查文件是否存在
      │   └─ 不存在 → 返回全 ignore_label 的占位标签
      │
      ├─ 3. 读取 PCD 文件
      │   ├─ 提取 xyz 坐标
      │   ├─ 提取 label 标签
      │   ├─ 提取 is_visible_in_fisheye_cameras（可见性）
      │   └─ 提取 label_hit_num（命中次数）
      │
      ├─ 4. 应用 BDA 变换（如果存在）
      │   └─ 对点云坐标进行坐标变换
      │
      ├─ 5. 处理标记点和未标记点
      │   ├─ marker_mask: z == -10000.0 的点（标记点）
      │   └─ unlabel_mask: 标记点且 label == unlabeled_pillar
      │
      ├─ 6. 范围过滤（range_filter）
      │   ├─ 过滤 ego 车辆范围内的点
      │   ├─ 将范围内的非道路类别改为 "other"
      │   └─ 过滤超出网格范围的点
      │
      ├─ 7. 生成高分辨率标签（get_high_resolution_label）
      │   ├─ 使用 voxel_size（原始体素大小，如 0.15m）
      │   ├─ 计算体素索引
      │   └─ 生成 upsample_labels 和 upsample_valid
      │
      ├─ 8. 生成下采样标签（get_downsampled_labels）
      │   ├─ 使用 voxel_size_downsample（×2，如 0.3m）
      │   ├─ 对体素索引排序
      │   ├─ 调用 numba_label_mapping 聚合标签
      │   └─ 生成 labels 和 valid
      │
      ├─ 9. 标签映射（occupancy_label_mapping）
      │   └─ 将原始类别ID映射到训练类别ID
      │
      ├─ 10. 生成激光雷达可见性掩码（可选）
      │    └─ 如果 with_lidar_visible == True
      │        └─ 将 lidar_points 投影到体素网格
      │
      ├─ 11. 处理不可见区域
      │    └─ 如果 invisible_mapping_label >= 0
      │        └─ 将不可见体素设置为指定标签
      │
      ├─ 12. 处理未标记区域
      │    └─ 将 unlabel_mask 对应的体素设置为 ignore_label
      │
      ├─ 13. 可行驶区域过滤（可选）
      │    └─ 如果 drivable_filter_classes 不为空
      │        └─ filter_labels_according_to_drivable
      │            └─ 在可行驶区域内过滤特定类别
      │
      ├─ 14. 前景类别填充（可选）
      │    └─ 如果 forground_fill_classes 不为空
      │        └─ fill_forground_classes
      │            └─ 在检测框内将类别0填充为 moveable_object
      │
      ├─ 15. 精细类别标注（可选）
      │    └─ 如果 forground_classes 不为空
      │        ├─ fine_occ_classes_by_3d_bboxes
      │        │   └─ 根据3D检测框标注精细类别
      │        └─ fine_occ_classes_by_unmovable
      │            └─ 处理不可移动物体和无人自行车
      │
      └─ 16. 输出结果
          ├─ gt_masks_3d: 下采样标签 (H, W, D)
          ├─ voxel_valid_mask: 下采样可见性掩码
          ├─ gt_masks_3d_upscale: 高分辨率标签
          ├─ voxel_valid_mask_upscale: 高分辨率可见性掩码
          ├─ fine_gt_masks_3d: 精细类别标签（下采样）
          └─ fine_gt_masks_3d_upscale: 精细类别标签（高分辨率）
```

## 二、LoadOccGTFromLCData 执行流程（LC 格式）

```
输入: results 字典
  ├─ data_format == "lc"?
  │   └─ NO → 直接返回 results
  │
  └─ YES → 开始处理
      │
      ├─ 1. 检查 occupancy 配置
      │   └─ 不存在或没有 occ_label_path → 返回占位标签
      │
      ├─ 2. 读取 NPZ 文件
      │   ├─ labels: 占用标签数组
      │   ├─ invalid: 无效体素的字节数组
      │   └─ upsample: 高分辨率标签数组
      │
      ├─ 3. 解析无效掩码
      │   └─ 将字节数组解包为位掩码
      │
      ├─ 4. 标签对齐（lpai_gt_padding）
      │   ├─ 计算原始标签和当前配置的空间范围交集
      │   ├─ 将原始标签复制到新网格的对应位置
      │   └─ 处理高分辨率版本
      │
      ├─ 5. 生成激光雷达可见性掩码（可选）
      │   └─ 同 LoadOccGTFromPCD
      │
      ├─ 6. 标签映射
      │   └─ 同 LoadOccGTFromPCD
      │
      ├─ 7. 优先级高度采样
      │   └─ priority_height_sampling
      │       └─ 将高度维度从 Z 压缩到 Z/2
      │           └─ 按优先级保留标签（障碍物优先于道路）
      │
      ├─ 8. 处理不可见区域
      │   └─ 同 LoadOccGTFromPCD
      │
      └─ 9. 输出结果
          ├─ gt_masks_3d: 下采样标签
          ├─ voxel_valid_mask: 下采样可见性掩码
          ├─ gt_masks_3d_upscale: 高分辨率标签（已高度采样）
          └─ voxel_valid_mask_upscale: 高分辨率可见性掩码
```

## 三、LoadMultiViewVirtualImageWithResize 执行流程

```
输入: results 字典
  ├─ data_format == "qds"?
  │   └─ NO → 直接返回 results
  │
  └─ YES → 开始处理
      │
      ├─ 1. 遍历所有图像路径
      │   │
      │   ├─ 2. 构建完整路径
      │   │   └─ os.path.join(img_root, img_path)
      │   │
      │   ├─ 3. 读取图像
      │   │   ├─ 从磁盘或 Petrel 存储读取
      │   │   └─ BGR → RGB 转换
      │   │
      │   ├─ 4. 获取目标分辨率
      │   │   └─ 从 camera_resolution 字典获取
      │   │
      │   ├─ 5. 调整图像大小
      │   │   ├─ 计算缩放比例
      │   │   ├─ cv2.resize 调整图像
      │   │   └─ 更新相机内参（按缩放比例）
      │   │
      │   └─ 6. 添加到列表
      │
      └─ 7. 输出结果
          ├─ img: 图像列表
          ├─ resized_intrinsic: 调整后的内参
          ├─ img_shape: 图像形状列表
          └─ 其他元数据
```

## 四、LoadMultiViewSegLabel 执行流程

```
输入: results 字典
  │
  ├─ 1. 遍历所有图像路径
  │   │
  │   ├─ 2. 推断分割标签路径
  │   │   ├─ 从图像路径提取 root 和 tail
  │   │   ├─ 提取相机ID，添加 "_HR" 后缀
  │   │   └─ 构建: root/occ/image_seg/{cam_id}/{fname}-panonet.png
  │   │
  │   ├─ 3. 检查并加载标签图像
  │   │   └─ 不存在 → 返回全 ignore_label 的占位标签
  │   │
  │   ├─ 4. 转换标签格式
  │   │   ├─ BGR 图像 → ID（R + 256*G + 256*256*B）
  │   │   └─ ID // 1000 → 语义类别ID
  │   │
  │   ├─ 5. 调整大小
  │   │   └─ cv2.resize（最近邻插值）对齐到目标图像大小
  │   │
  │   ├─ 6. 类别映射（可选）
      │   └─ 如果 cate_mapping 不为空
      │       └─ 将原始类别ID映射到目标类别ID
      │   │
      │   └─ 7. 填充（可选）
          │   └─ 如果 pad_shape 存在
          │       └─ 使用 ignore_label 填充到目标大小
      │
      └─ 8. 输出结果
          └─ seg: 分割标签列表
```

## 五、关键函数说明

### numba_label_mapping
```
输入: 
  - labels: 点云标签数组
  - valid: 点云可见性数组
  - voxel_indices: 体素索引数组（已排序）
  - labels_out: 输出标签数组（初始化为0）
  - valid_out: 输出可见性数组（初始化为True）

处理逻辑:
  1. 遍历所有点，按体素索引分组
  2. 对每个体素：
     - 统计所有点的标签分布
     - 如果全部是 ignore_label → 输出 ignore_label
     - 如果全部是 free_label → 输出 free_label
     - 如果只有 free_label 和 ignore_label → 输出 free_label
     - 否则 → 输出出现次数最多的非 free/ignore 标签
  3. 可见性：只要有一个点可见，体素就可见（OR 操作）

输出:
  - labels_out: 聚合后的标签数组
  - valid_out: 聚合后的可见性数组
```

### priority_height_sampling
```
输入:
  - target_voxel: 3D标签数组 (H, W, Z)
  - priority_order: 优先级数组（从高到低）
  - ignore_label: 忽略标签值

处理逻辑:
  1. 将高度维度从 Z 压缩到 Z/2
  2. 对每对相邻的 z 层（z*2 和 z*2+1）：
     - 按优先级顺序查找标签
     - 如果找到匹配的优先级标签，则使用该标签
     - 否则继续查找下一个优先级

输出:
  - cls_label: 压缩后的标签数组 (H, W, Z/2)
```

### filter_labels_according_to_drivable
```
输入:
  - label: 3D标签数组 (W, H, D)

处理逻辑:
  1. 确定体素大小（下采样或高分辨率）
  2. 计算 ego 车辆位置（w, h）
  3. 定义过滤区域：
     - 横向范围：ego_h ± lateral_voxels
     - 前向范围：ego_w 到 ego_w + forward_voxels
  4. 在区域内查找可行驶体素（drivable label）
  5. 对每个可行驶体素：
     - 检查是否包含目标过滤类别
     - 如果包含：
       - 对于特定类别（unmoveable_object, vegetation）：
         - 只过滤高度 < threshold 的部分
       - 对于其他类别：
         - 全部改为 "other"

输出:
  - modified_label: 过滤后的标签数组
```

## 六、数据流图

```
原始数据文件
    │
    ├─ FILELIST ──────────────┐
    │                          │
    ├─ image/*.jpg ────────────┤
    │                          │
    ├─ occ/labels/*.pcd ───────┤
    │                          │
    ├─ lidar/*.pcd ────────────┤
    │                          │
    └─ occ/image_seg/*.png ────┤
                                │
                                ▼
                        数据加载器管道
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
LoadMultiViewVirtualImage  LoadOccGTFromPCD    LoadMultiViewSegLabel
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                                ▼
                          results 字典
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
    img (图像列表)      gt_masks_3d (标签)        seg (分割标签)
    intrinsic (内参)    voxel_valid_mask (掩码)
    img2pose (位姿)      gt_masks_3d_upscale
                        fine_gt_masks_3d
```

## 七、关键参数说明

### occ_grid_config
```python
{
    "xbound": [min_x, max_x, voxel_size_x],  # 如 [-51.2, 51.2, 0.2]
    "ybound": [min_y, max_y, voxel_size_y],  # 如 [-51.2, 51.2, 0.2]
    "zbound": [min_z, max_z, voxel_size_z],  # 如 [-5.0, 3.0, 0.2]
    "dbound": [min_d, max_d, depth_size]     # 深度范围（用于深度标签）
}
```

### class_names_mapping
```python
{
    "drivable": 1,
    "moveable_object": 2,
    "unmoveable_object": 3,
    "vegetation": 4,
    "other": 5,
    # ... 更多类别
}
```

### index_to_class_names
```python
{
    0: "free",
    1: "drivable",
    2: "moveable_object",
    # ... 原始数据中的类别索引到名称的映射
}
```

