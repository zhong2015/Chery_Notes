# Occ3D-nuScenes 数据集生成完整实施方案

**文档版本**: v1.1  
**创建日期**: 2025-01-27  
**最后更新**: 2025-01-27  
**项目路径**: `E:\Chery\dz\Occ\Occ3D-master`  
**状态**: 已实现核心功能，支持单进程和多进程并行处理

---

## 目录

1. [项目背景与目标](#1-项目背景与目标)
2. [数据流总览](#2-数据流总览)
3. [技术方案详细设计](#3-技术方案详细设计)
4. [CUDA Algorithm 3 实现细节](#4-cuda-algorithm-3-实现细节)
5. [文件结构与输出格式](#5-文件结构与输出格式)
6. [配置参数与命令行接口](#6-配置参数与命令行接口)
7. [验证与检查点](#7-验证与检查点)
8. [依赖与环境要求](#8-依赖与环境要求)

---

## 1. 项目背景与目标

### 1.1 项目目标

根据论文 [Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving](https://arxiv.org/pdf/2304.14365) 和官方仓库 [Tsinghua-MARS-Lab/Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D)，实现 **Occ3D-nuScenes** 基准数据集的完整生成流程。

**核心要求**：
- 生成的数据集目录结构必须与官方 README 完全一致
- 实现论文中描述的**三阶段标签生成流水线**（Voxel Densification → Occlusion Reasoning → Image-guided Refinement）
- 支持多 GPU 并行处理（2~4 张 GPU）
- 服务器上存放完整 nuScenes v1.0-trainval 数据，本地仅建立目录架构

### 1.2 输入数据源

- **nuScenes v1.0-trainval** 完整数据（服务器端）
- 包含：
  - LiDAR 点云（`samples/LIDAR_TOP/` 和 `sweeps/LIDAR_TOP/`）
  - 6 个相机图像（`samples/CAM_*/`）
  - LiDAR 语义分割标签（`lidarseg/v1.0-trainval/`）
  - 元数据 JSON（`v1.0-trainval/*.json`）

### 1.3 输出数据集规格

根据 [Occ3D 官方 README](https://github.com/Tsinghua-MARS-Lab/Occ3D)：

| 属性 | 数值 |
|------|------|
| 训练/验证/测试 | 600 / 150 / 250 scenes |
| 相机数量 | 6 |
| 体素尺寸 | [0.4m, 0.4m, 0.4m] |
| 空间范围 | [-40m, -40m, -1m, 40m, 40m, 5.4m] |
| 网格大小 | [200, 200, 16] |
| 类别数 | 18 (0-16: nuScenes-lidarseg, 17: free) |

**输出目录结构**（必须严格对齐官方）：
```
Occpancy3D-nuScenes-V1.0/
├── trainval/
│   ├── imgs/
│   │   ├── CAM_BACK/
│   │   ├── CAM_BACK_LEFT/
│   │   ├── CAM_BACK_RIGHT/
│   │   ├── CAM_FRONT/
│   │   ├── CAM_FRONT_LEFT/
│   │   └── CAM_FRONT_RIGHT/
│   ├── gts/
│   │   ├── [scene_name]/
│   │   │   ├── [frame_token]/
│   │   │   │   └── labels.npz
│   │   │   └── ...
│   │   └── ...
│   └── annotations.json
└── test/
    ├── imgs/
    └── annotations.json
```

---

## 2. 数据流总览

### 2.1 处理流程（每个 keyframe）

```
输入: nuScenes keyframe (sample_token)
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Voxel Densification（体素密集化）                   │
│ • Multi-frame Aggregation: 聚合 21 个 keyframe（前后各10+当前）│
│ • (可选) Sweeps 加入: 引入未标注 sweeps 提升覆盖              │
│ • (可选) Label Assignment (KNN): 给 sweeps 点分配语义标签     │
│ • Dynamic objects alignment: 动态物体实例对齐减少拖影         │
│ • (可选) Mesh Reconstruction: TSDF/mesh 补洞得到更致密表面     │
│ 输出: densified semantic points（稠密语义点云）               │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Occlusion Reasoning（遮挡推理）                      │
│ • LiDAR Visibility (Algorithm 2): ray casting 生成 free/occ   │
│   - 得到 semantics（含 free=17）                              │
│   - 得到 mask_lidar（LiDAR view observed mask）               │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Camera Visibility (Algorithm 3, CUDA)               │
│ • 对 6 相机每像素发射射线（默认原始分辨率）                    │
│ • 3D DDA 遍历：遇到第一个 occupied voxel 后停止（遮挡终止）   │
│ • 得到 mask_camera_rays（camera rays observed mask）          │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Image-guided Voxel Refinement（图像引导体素细化）    │
│ • 使用 2D 语义（来自 2D 分割模型或标注）修剪 3D 边界           │
│ • 沿像素射线：遇到首个语义一致 voxel 前的 voxels 置为 free      │
│ • (建议) 细化后重算一次 mask_camera_rays 保持一致性            │
│ 输出: refined semantics                                      │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Export: 写盘与元信息                                          │
│ • mask_camera = mask_lidar AND mask_camera_rays               │
│ • 写入 labels.npz（XYZ）与 annotations.json                   │
│ • imgs/ 用 symlink/hardlink/copy 对齐官方目录结构              │
└─────────────────────────────────────────────────────────────┘
  ↓
输出: 完整的Occ3D-nuScenes格式数据
```

### 2.2 关键设计决策

1. **多帧聚合窗口**：nuScenes 使用 **21 个 keyframe**（当前帧 + 前后各 10 个）
2. **(可选) Label Assignment (KNN)**：当引入无标签 sweeps 时，通过 KNN 从 keyframe 传播语义标签
3. **(可选) Mesh Reconstruction**：TSDF/mesh 补洞得到更致密表面（更接近论文 Stage 1）
4. **动态物体处理**：只对 vehicle/pedestrian/bicycle/motorcycle 做 instance 对齐，movable_object（barrier/traffic_cone）按静态处理
5. **LiDAR Visibility Mask**：`mask_lidar` 由 LiDAR ray casting 得到（Stage 2 核心产物之一）
6. **mask_camera 定义**：`mask_camera = mask_lidar AND mask_camera_rays`（确保只在 LiDAR observed 区域内评测）
7. **相机分辨率**：默认使用 nuScenes **原始分辨率**（与论文保持一致），可选切换到 928×1600（同时缩放 K）
8. **Stage 3 Image-guided refinement**：使用 2D 语义修剪 3D 边界（更接近论文 Stage 3）
9. **数据格式**：统一使用 **XYZ 顺序**，dtype 为 `uint8`

---

## 3. 技术方案详细设计

### 3.1 Stage 1: Voxel Densification（体素密集化）

#### 3.1.1 输入数据读取

- 使用 `nuscenes-devkit` 读取：
  - `sample.json`: keyframe 列表
  - `sample_data.json`: LiDAR 和相机数据路径
  - `ego_pose.json`: 车辆位姿
  - `calibrated_sensor.json`: 传感器标定参数
  - `lidarseg.json`: LiDAR 语义标签文件路径

- 对每个 target keyframe，收集：
  - 当前帧 + 前后各 10 个 keyframe（共 21 帧）
  - 每帧的 LiDAR 点云（`.pcd.bin`）和 lidarseg 标签（`.bin`）

#### 3.1.2 坐标变换

**目标**：将所有 source keyframe 的点云变换到 **target ego 坐标系**

**变换链**：
```
p_lidar_s → p_ego_s → p_global → p_ego_t
```

**数学表示**：
$$ p^{\text{ego}_t} = \left( T^{\text{ego}_t}_{\text{global}} \right)^{-1} \cdot T^{\text{global}}_{\text{ego}_s} \cdot T^{\text{ego}_s}_{\text{lidar}_s} \cdot p^{\text{lidar}_s} $$

**实现**：

- 使用 `pyquaternion` 处理四元数旋转
- 使用 `nuscenes-devkit` 的 `transform_matrix` 工具函数

#### 3.1.3 动态物体对齐

**动态类别列表**（根据 nuScenes-lidarseg 类别定义）：
- `vehicle.car` (lidarseg index: 4)
- `vehicle.truck` (10)
- `vehicle.bus` (3)
- `vehicle.trailer` (9)
- `vehicle.construction` (5)
- `human.pedestrian.*` (7)
- `vehicle.bicycle` (2)
- `vehicle.motorcycle` (6)

**对齐策略**：
1. 对每个 source keyframe：
   - 读取该帧的 `sample_annotation`（3D bounding boxes）
   - 使用 `utils/points_in_bbox.py:points_in_rbbox` 筛选 box 内的点
2. 对动态类别的点：
   - 检查该 `instance_token` 是否在 target keyframe 也存在
   - 如果存在：
     - 将 source box 内的点变换到 **box-local 坐标系**
     - 再用 target box 的 pose 变换回 target ego 坐标系
   - 如果不存在：丢弃这些点（避免幽灵残影）
3. 对静态点：直接使用 ego/global 变换

**复用现有代码**：
- `utils/points_in_bbox.py:points_in_rbbox` - 旋转 3D box 内点筛选
- `utils/points_in_bbox.py:center_to_corner_box3d` - box 角点计算

#### 3.1.4 输出

- `P_agg`: 聚合后的点云坐标 `(N, 3)`，在 target ego 坐标系
- `L_agg`: 对应的 lidarseg 语义标签 `(N,)`，取值 0~16

#### 3.1.5 (可选) Label Assignment (KNN) —— 给 sweeps 点赋语义标签

对照 `cursor_gen_files/Occ3D_Paper_Detailed_Interpretation.md`：论文流水线在存在“未标注帧/非 keyframe”的情况下，需要通过 KNN 给这些点赋语义标签（Waymo 10Hz vs 2Hz 的场景更典型）。\n
在 nuScenes 上，如果我们为了 densification 引入 `sweeps/LIDAR_TOP`（通常没有 lidarseg 标注），也必须实现这一步。

**输入**：
- `P_key (N1,3)`: 已有语义的 keyframe 点云（已对齐到 target ego）
- `L_key (N1,)`: `P_key` 的 lidarseg 语义（0~16）
- `P_sweep (N2,3)`: sweeps 点云（已对齐到 target ego）

**输出**：
- `L_sweep (N2,)`: 给每个 sweep 点分配的伪标签（0~16）

**算法建议（工程可落地）**：
- 在 `P_key` 上构建 KDTree（`sklearn.neighbors.KDTree` 或 `faiss`）
- 对每个 sweep 点找 `k` 个最近邻 key 点，语义取多数投票（mode）
- 典型超参数：`k=5`，可加 `max_radius` 防止远距离错误传播

**整合**：
- `P_dense = concat(P_key, P_sweep)`
- `L_dense = concat(L_key, L_sweep)`

#### 3.1.6 (可选) Mesh Reconstruction —— TSDF/mesh 补洞得到更致密表面

对照 `cursor_gen_files/Occ3D_Paper_Detailed_Interpretation.md`：论文在 Stage 1 中提到 mesh/TSDF 重建用于补洞，进一步提高 recall。\n
该步骤目标是把稀疏点云补成更连续的表面，再进行后续体素与可见性推理。

**输入**：`P_dense, L_dense`\n
**输出**：`P_mesh_dense, L_mesh_dense`（从 mesh 表面采样出的更致密点云，仍带语义）

**实现路径（建议分两档）**：
- **实现档 A（易落地）**：Open3D TSDF integration 或 Poisson reconstruction → 表面采样 → 语义继承
- **实现档 B（更接近论文）**：集成 TSDF/VDBFusion 等实现；对地面类单独做拟合/补点，避免 TSDF 的 ground artifact

**开关建议**：
- `--enable-sweeps-densification`：是否引入 sweeps 并对 sweeps 做 KNN 语义赋值（这两个通常需要绑定）
- `--enable-mesh-recon`：是否做 mesh/TSDF 补洞
- `--mesh-recon-mode {tsdf, poisson}`：mesh 重建模式
  

---

### 3.2 Stage 2: Occlusion Reasoning —— LiDAR Visibility Mask（Algorithm 2）

#### 3.2.1 体素网格参数（固定）

```python
pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]  # [x_min, y_min, z_min, x_max, y_max, z_max]
voxel_size = [0.4, 0.4, 0.4]  # [vx, vy, vz]
grid_shape = (200, 200, 16)  # (X, Y, Z)
```

#### 3.2.2 输入准备（Algorithm 2 Inputs）

对照 `cursor_gen_files/Occ3D_Paper_Detailed_Interpretation.md` 的 Algorithm 2：这一步的输入不仅是点云，还需要**每条 LiDAR beam 的起点（LiDAR origin）**以及统一的体素网格定义。

**输入**：
- `P_dense (N, 3)`: Stage 1 输出的稠密语义点云（在 target ego 坐标系）
- `L_dense (N,)`: `P_dense` 的语义标签（0~16）
- `lidar_origins`: 聚合窗口内每个 keyframe 的 LiDAR origin（均已变换到 target ego 坐标系）
- `pc_range, voxel_size, grid_shape`: 体素网格定义（固定为 Occ3D-nuScenes 规格）

**输出**：
- `semantics[X,Y,Z] uint8`
- `mask_lidar[X,Y,Z] uint8`

#### 3.2.2 Ray Casting 算法（Algorithm 2 思想）

**核心统计量**：
- `voxel_occ_count[X,Y,Z]`: 被点"命中"的次数
- `voxel_free_count[X,Y,Z]`: 被射线"穿过"的次数

**处理流程**（对聚合窗口内每个 keyframe）：
1. 读取该帧的 LiDAR origin（从 `calibrated_sensor` 和 `ego_pose` 计算）
2. 将该 origin 变换到 target ego 坐标系
3. 对该帧的每个点（已在 target ego）：
   - 计算点所在的 voxel 索引 `target_voxel`
   - `atomicAdd(voxel_occ_count[target_voxel], 1)`
   - 使用 **3D DDA 算法**（Algorithm 1）从 origin 到 `target_voxel` 遍历：
     - 对路径上的每个 voxel：`atomicAdd(voxel_free_count[voxel_index], 1)`

**3D DDA 实现要点**（参考论文解读文档）：
- 使用 `EPS = 1e-9` 避免边界条件问题
- 计算 `tDelta`（跨过体素边长所需的参数 t 增量）
- 使用 `tMax` 判断下一次跨越体素边界的 t 值

#### 3.2.3 语义标签分配

**规则**：
- 对每个 voxel：
  - 如果 `voxel_occ_count > 0`：
    - 该 voxel 为 **occupied**
    - 语义标签 = 该 voxel 内所有点的 lidarseg 标签的 **多数投票**（mode）
  - 否则如果 `voxel_free_count > 0`：
    - 该 voxel 为 **free**（语义标签 = 17）
  - 否则：
    - 该 voxel 为 **unobserved**（`mask_lidar = 0`，语义可置 0 或 17，但会被 mask 忽略）

#### 3.2.4 输出

- `semantics[X,Y,Z] uint8`: 体素语义标签（0~17，其中 17 为 free）
- `mask_lidar[X,Y,Z] uint8`: **LiDAR Visibility Mask**（0/1）
  - 定义：`mask_lidar = (voxel_free_count > 0) OR (voxel_occ_count > 0)`
  - 含义：`mask_lidar==0` 表示该 voxel 在 LiDAR view **unobserved**（遮挡或超出覆盖）
  - 与 Occ3D README 对 `[mask_lidar]` 的定义一致，见 [Tsinghua-MARS-Lab/Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D)

**复用现有代码**：
- `utils/custom.py:sparse2dense` - 可作为参考，但需要扩展为支持 ray casting 的 free space 推理

---

### 3.3 Stage 2: Occlusion Reasoning —— Camera Visibility Mask（Algorithm 3, CUDA）

#### 3.3.1 输入准备

- `occupied_grid[X,Y,Z] uint8`: 从 `semantics` 和 `mask_lidar` 得到
  - `occupied_grid = (semantics != 17) & (mask_lidar == 1)`
- 6 个相机的参数：
  - `K`: 内参矩阵 `(3, 3)`
  - `T_cam2ego`: 外参矩阵 `(4, 4)`（相机到 ego 的变换）
  - `img_w, img_h`: 图像分辨率（默认使用 nuScenes 原始分辨率）

#### 3.3.2 射线构造

对每个像素 `(u, v)`：
1. 将像素坐标转换为相机坐标系方向：
   - 使用 `K^{-1}` 计算射线方向（归一化）
2. 变换到 ego 坐标系：
   - `ray_origin_ego = T_cam2ego[:3, 3]`（相机光心）
   - `ray_dir_ego = T_cam2ego[:3, :3] @ ray_dir_cam`

#### 3.3.3 CUDA Kernel 设计

**Kernel 签名**：
```cuda
__global__ void camera_ray_casting_kernel(
    const uint8_t* occupied_grid,      // [X*Y*Z]
    uint8_t* mask_camera_rays,          // [X*Y*Z], output
    const float* K_inv,                // [3*3], 内参逆矩阵
    const float* T_cam2ego,            // [4*4], 外参矩阵
    int img_w, int img_h,              // 图像分辨率
    const float* pc_range_min,        // [3], (x_min, y_min, z_min)
    const float* voxel_size,          // [3], (vx, vy, vz)
    const int* grid_size               // [3], (X, Y, Z)
)
```

**并行策略**：
- **Grid/Block 配置**：
  - `blockDim = (16, 16)`（256 线程/block）
  - `gridDim = ((img_w + 15)/16, (img_h + 15)/16)`
  - 每个线程处理一个像素 `(u, v)`

**每线程流程**：
1. 计算像素坐标：`u = blockIdx.x * blockDim.x + threadIdx.x`, `v = blockIdx.y * blockDim.y + threadIdx.y`
2. 检查边界：`if (u >= img_w || v >= img_h) return;`
3. 构造射线：
   - `ray_dir_cam = K_inv @ [u, v, 1]^T`（归一化）
   - `ray_origin_ego = T_cam2ego[:3, 3]`
   - `ray_dir_ego = T_cam2ego[:3, :3] @ ray_dir_cam`
4. Ray-Box 相交测试（与 `pc_range` 的 AABB）：
   - 计算 `t_enter` 和 `t_exit`
   - 如果 `t_enter >= t_exit`，射线不穿过体素网格，直接返回
5. 3D DDA 遍历（从 `t_enter` 开始）：
   - 对每个遍历到的 voxel：
     - 计算 voxel 索引 `idx = x + X*(y + Y*z)`
     - `atomicOr(&mask_camera_rays[idx], 1)`（标记为 camera-observed）
     - 如果 `occupied_grid[idx] == 1`：**break**（遮挡终止，后方不再标记）

#### 3.3.3 多相机融合

对 6 个相机分别 launch kernel，然后：
- `mask_camera_rays = OR(mask_camera_rays_cam0, ..., mask_camera_rays_cam5)`

#### 3.3.4 最终 mask_camera

```
mask_camera = mask_lidar AND mask_camera_rays
```

**实现**：

```python
mask_camera = np.logical_and(mask_lidar, mask_camera_rays).astype(np.uint8)
```

---

### 3.4 Stage 3: Image-guided Voxel Refinement（图像引导体素细化）

对照 `cursor_gen_files/Occ3D_Paper_Detailed_Interpretation.md`：Stage 3 的目标是修复 3D-2D misalignment（姿态漂移/噪声造成的 3D 外扩），通过 2D 语义沿像素射线“修剪” 3D 边界。

#### 3.4.1 输入/输出

**输入**：
- Stage 2 输出的 `semantics[X,Y,Z]`、`mask_lidar[X,Y,Z]`、以及相机参数（K、T_cam2ego、原始分辨率）
- 每个相机的 2D 语义 `seg2d_cam[h,w]`（来源可为：2D 分割模型推理结果，或你们已有的 2D 标注/伪标注）

**输出**：
- `semantics_refined[X,Y,Z]`：主要变化是将部分“前景外扩”的 occupied voxels 置为 `free=17`

#### 3.4.2 核心算法（与论文描述对齐）

对每个相机、每个像素（建议在 ROI 内）：
1. 读取像素语义 `c_2d = seg2d_cam[v,u]`
2. 沿像素射线在体素网格内从近到远 DDA 遍历 voxels
3. 当遇到**第一个语义与 `c_2d` 一致**的 occupied voxel 时：
   - 将该 voxel 之前遍历过的 voxels（若被标成 occupied）全部置为 `free=17`

这一步会显著改善物体边界的精细度，并提升 3D-2D semantic consistency。

#### 3.4.3 ROI 与类别映射

- **ROI（推荐）**：按论文的 2D ROI 思路（单帧 LiDAR 可投影覆盖的区域），避免在无 LiDAR 覆盖区域引入噪声修剪
- **类别映射**：2D 分割输出类别需映射到 nuScenes-lidarseg 的 0~16 体系（与 `semantics` 对齐），`free` 不来自 2D

#### 3.4.4 与 mask_camera 的一致性（建议）

Stage 3 会改变 occupied 的空间分布，进而改变 “camera ray 的第一个 occupied voxel” 位置。为了与论文口径更一致，建议：
- 先 Stage 2 得到 `mask_camera_rays`
- Stage 3 得到 `semantics_refined`
- **再基于 `semantics_refined` 重跑一次 CUDA Algorithm 3** 得到新的 `mask_camera_rays`
- 最终：`mask_camera = mask_lidar AND mask_camera_rays`

#### 3.4.5 开关建议

- `--enable-image-guided-refine`：是否启用 Stage 3
- `--seg2d-mode {none,model,lidar_project,annotation}`：2D 语义生成模式（新增接口：选择方案A/方案B/都不选/使用已有标注）
- `--seg2d-model / --seg2d-weights`：2D 分割模型与权重（若 seg2d-mode=model）
- `--seg2d-cache-dir`：2D 语义缓存目录（保存 `seg2d_cam`，供复用/断点续跑）
- `--refine-roi {lidar_roi, full_image}`：ROI 策略

#### 3.4.6 如果本项目/数据侧没有提供 2D 语义（seg2d_cam），怎么办？

这是一个现实问题：Occ3D 论文的 Stage 3 依赖 `seg2d_cam[h,w]`，但 **nuScenes 官方数据本身并不直接提供与 Occ3D 同口径的 2D 语义**。\n
因此在实施时必须提供一个“2D 语义生成接口”，用于选择：方案 A / 方案 B / 都不选择（关闭 Stage 3）/ 使用已有标注（若你们未来补齐）。

为避免 Stage 3 实现与 2D 语义来源强耦合，建议新增统一接口：

```python
def build_or_load_seg2d_cam(
    sample_token: str,
    cam_name: str,
    nusc,
    seg2d_mode: str,  # 'none' | 'model' | 'lidar_project' | 'annotation'
    seg2d_cache_dir: str,
    *,
    model=None,
    class_mapper=None,
    lidar_points_ego=None,   # (N,3) in ego
    lidar_labels=None,       # (N,) 0..16
) -> "np.ndarray | None":
    \"\"\"返回 seg2d_cam[h,w] (uint8, 0..16)，若 seg2d_mode='none' 则返回 None。\"\"\n
```

并在 Stage 3 主流程中显式分支：

```python
if not args.enable_image_guided_refine or args.seg2d_mode == 'none':
    # 不做 Stage 3（等价于方案 C）
    semantics_refined = semantics
else:
    # 使用 seg2d_mode 生成/加载 seg2d_cam，然后做 voxel refinement
    semantics_refined = image_guided_voxel_refine(...)
```

下面给出方案 A/B 的“明确代码实现要点（伪代码级别）”，确保后续可直接落地。

##### 3.4.6-A 方案 A：2D 语义分割模型推理生成 `seg2d_cam`（seg2d_mode='model'）

**目标**：对每个相机图像推理得到 `seg2d_cam[h,w]`，并映射到 nuScenes-lidarseg 的 0..16 类。\n
**输入**：相机图像路径、模型权重、类别映射表。\n
**输出**：`seg2d_cam[h,w] uint8`（缓存到 `seg2d_cache_dir`）。

**建议新增模块**：`occ3d_nuscenes/seg2d_model.py`\n
**建议函数签名**：

```python
def infer_seg2d_from_model(
    img_path: str,
    *,
    model,
    class_mapper,
    out_hw: tuple[int, int] | None = None,  # None=原始分辨率；否则 resize 并同步处理
) -> "np.ndarray":
    \"\"\"返回 seg2d_cam[h,w] uint8, 值域 0..16（对齐 nuScenes-lidarseg）。\"\"\n
```

**缓存约定（强烈建议）**：\n
- `seg2d_cache_dir/{split}/{scene_name}/{frame_token}/{CAM_NAME}.npy`\n
- dtype: `uint8`\n
- shape: `[h,w]`（与该相机原始分辨率一致，或与 `--camera-ray-image-size` 一致）

**推理伪代码**：

```python
def build_or_load_seg2d_cam(..., seg2d_mode='model', model=None, class_mapper=None, ...):
    cache_path = make_cache_path(seg2d_cache_dir, split, scene_name, frame_token, cam_name)
    if os.path.exists(cache_path):
        return np.load(cache_path).astype(np.uint8)

    img_path = get_cam_img_path_from_nusc(sample_token, cam_name)
    seg_raw = model_infer(img_path, model=model)          # seg_raw: [h,w] in model label space
    seg2d = class_mapper.to_lidarseg(seg_raw)             # seg2d: [h,w] in 0..16
    np.save(cache_path, seg2d.astype(np.uint8))
    return seg2d
```

> 说明：这里 `class_mapper` 的职责是把 2D 模型输出类别映射到 nuScenes-lidarseg 的 16 类（0..16，0 通常为 ignore/void）。类别表可参考 `nuscenes-devkit lidarseg README`：`https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/fcc41628d41060b3c1a86928751e5a571d2fc2fa/python-sdk/nuscenes/eval/lidarseg/README.md`。\n

##### 3.4.6-B 方案 B：LiDAR 语义投影到图像生成 `seg2d_cam`（seg2d_mode='lidar_project'）

**目标**：不用外部 2D 模型，仅利用 LiDAR 语义点云投影到每个相机平面得到 2D 伪标签。\n
**输入**：\n
- `lidar_points_ego (N,3)`（建议用当前 keyframe 的 LiDAR 点，不用聚合点，避免时序错位）\n
- `lidar_labels (N,)`（0..16）\n
- 相机内参 `K`、外参 `T_cam2ego`、图像尺寸 `h,w`\n
**输出**：`seg2d_cam[h,w] uint8`（缓存）。

**建议新增模块**：`occ3d_nuscenes/seg2d_lidar_project.py`\n
**建议函数签名**：

```python
def project_lidarseg_to_image(
    points_ego: "np.ndarray",   # (N,3)
    labels: "np.ndarray",       # (N,)
    K: "np.ndarray",            # (3,3)
    T_cam2ego: "np.ndarray",    # (4,4)
    img_hw: tuple[int, int],    # (h,w)
) -> "np.ndarray":
    \"\"\"返回 seg2d_cam[h,w] uint8，未覆盖像素填 0（ignore/void）。\"\"\n
```

**实现伪代码（核心逻辑）**：

```python
def project_lidarseg_to_image(points_ego, labels, K, T_cam2ego, img_hw):
    h, w = img_hw
    seg = np.zeros((h, w), dtype=np.uint8)
    depth = np.full((h, w), np.inf, dtype=np.float32)  # z-buffer: 取最近点

    T_ego2cam = np.linalg.inv(T_cam2ego)
    pts_cam = transform_points(points_ego, T_ego2cam)  # (N,3)

    # 只保留相机前方
    mask = pts_cam[:, 2] > 1e-3
    pts_cam = pts_cam[mask]
    lbl = labels[mask].astype(np.uint8)

    # 投影到像素
    uvw = (K @ pts_cam.T).T  # (N,3)
    u = (uvw[:, 0] / uvw[:, 2]).astype(np.int32)
    v = (uvw[:, 1] / uvw[:, 2]).astype(np.int32)
    z = pts_cam[:, 2].astype(np.float32)

    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z, lbl = u[in_img], v[in_img], z[in_img], lbl[in_img]

    # z-buffer 更新：同一像素保留最近点语义
    for i in range(u.shape[0]):
        if z[i] < depth[v[i], u[i]]:
            depth[v[i], u[i]] = z[i]
            seg[v[i], u[i]] = lbl[i]

    # 可选：做一次简单的形态学膨胀/闭运算填洞（弱替代）
    # seg = postprocess(seg)
    return seg
```

**说明**：\n
- 方案 B 的 `seg2d_cam` 会很稀疏，必须依赖后处理（膨胀/插值）才能在 Stage 3 产生较明显效果；这也是它弱于论文设定的原因。\n
- 但它的好处是：不引入外部模型，能尽快让 Stage 3 代码跑通并做对齐实验。

---

### 3.5 Export（非论文 Stage）: 最终输出与写盘

说明：论文流水线是 3 个阶段（Stage 1/2/3）。这里的“写盘与元信息导出”是工程化的导出步骤，不属于论文定义的一个新 stage，因此不命名为 Stage 4。

#### 3.5.1 labels.npz 格式

**文件路径**：`gts/[scene_name]/[frame_token]/labels.npz`

**内容**（XYZ 顺序，uint8）：
```python
{
    'semantics': np.ndarray,      # [200, 200, 16] uint8, 0-17
    'mask_lidar': np.ndarray,     # [200, 200, 16] uint8, 0/1
    'mask_camera': np.ndarray,    # [200, 200, 16] uint8, 0/1
}
```

**写入代码示例**：
```python
np.savez_compressed(
    output_path,
    semantics=semantics.astype(np.uint8),
    mask_lidar=mask_lidar.astype(np.uint8),
    mask_camera=mask_camera.astype(np.uint8)
)
```

#### 3.5.2 annotations.json 格式

**完全对齐官方 README 的 schema**：

```json
{
    "train_split": ["scene-0001", "scene-0002", ...],
    "val_split": ["scene-0003", "scene-0004", ...],
    "scene_infos": {
        "scene-0001": {
            "n015-2018-07-18-11-07-57+0800__CAM_BACK__1531883530437525": {
                "timestamp": "1531883530437525",
                "camera_sensor": {
                    "ca4d3d9de242603dae34ba357e07be62b": {
                        "img_path": "imgs/CAM_BACK/n015-2018-07-18-11-07-57+0800__CAM_BACK__1531883530437525.jpg",
                        "intrinsic": [[...], [...], [...]],
                        "extrinsic": {
                            "translation": [x, y, z],
                            "rotation": [w, x, y, z]
                        },
                        "ego_pose": {
                            "translation": [x, y, z],
                            "rotation": [w, x, y, z]
                        }
                    },
                    ...
                },
                "ego_pose": {
                    "translation": [x, y, z],
                    "rotation": [w, x, y, z]
                },
                "gt_path": "gts/scene-0001/n015-2018-07-18-11-07-57+0800__CAM_BACK__1531883530437525/labels.npz",
                "next": "next_frame_token",
                "prev": "prev_frame_token"
            },
            ...
        },
        ...
    }
}
```

#### 3.5.3 图像文件处理

**策略**（按优先级）：
1. **symlink**（推荐，Linux 服务器）：
   ```python
   os.symlink(src_path, dst_path)
   ```
2. **hardlink**（同盘更省空间）：
   ```python
   os.link(src_path, dst_path)
   ```
3. **copy**（最慢，但最通用）：
   ```python
   shutil.copy2(src_path, dst_path)
   ```

**路径映射**：
- 源：`nuScenes/samples/CAM_*/xxx.jpg`
- 目标：`Occpancy3D-nuScenes-V1.0/trainval/imgs/CAM_*/xxx.jpg`

---

## 4. CUDA Algorithm 3 实现细节

### 4.1 3D DDA 算法（Algorithm 1）

**输入**：
- `ray_start`: 射线起点 `(x, y, z)`
- `ray_end`: 射线终点（或方向 + 最大深度）
- `pc_range`: `[x_min, y_min, z_min, x_max, y_max, z_max]`
- `voxel_size`: `[vx, vy, vz]`
- `grid_size`: `[X, Y, Z]`

**算法步骤**（参考论文解读文档）：
1. 将射线移入网格坐标系：`new_ray_start = ray_start - pc_range[0:3]`
2. 计算每个轴的 step（向前/向后走）
3. 计算 `tDelta`（跨过体素边长所需的参数 t 增量）
4. 计算 `cur_voxel` / `last_voxel`（起止体素）
5. 计算 `tMax`（下一次在轴 k 上跨越体素边界的 t 值）
6. 使用 3D DDA 算法遍历体素

**CUDA 实现伪代码**：
```cuda
__device__ void ray_casting_3d_dda(
    float3 ray_start, float3 ray_dir, float t_max,
    float3 pc_range_min, float3 voxel_size, int3 grid_size,
    const uint8_t* occupied_grid, uint8_t* mask_camera_rays
) {
    // 移入网格坐标系
    float3 start = make_float3(
        ray_start.x - pc_range_min.x,
        ray_start.y - pc_range_min.y,
        ray_start.z - pc_range_min.z
    );
    
    // 计算体素索引
    int3 cur_voxel = make_int3(
        (int)floorf(start.x / voxel_size.x),
        (int)floorf(start.y / voxel_size.y),
        (int)floorf(start.z / voxel_size.z)
    );
    
    // 计算 step 和 tDelta
    int3 step = make_int3(
        ray_dir.x > 0 ? 1 : -1,
        ray_dir.y > 0 ? 1 : -1,
        ray_dir.z > 0 ? 1 : -1
    );
    
    float3 tDelta = make_float3(
        abs(voxel_size.x / ray_dir.x),
        abs(voxel_size.y / ray_dir.y),
        abs(voxel_size.z / ray_dir.z)
    );
    
    // 计算 tMax（到下一个体素边界的距离）
    float3 tMax = make_float3(
        ((cur_voxel.x + (step.x > 0 ? 1 : 0)) * voxel_size.x - start.x) / ray_dir.x,
        ((cur_voxel.y + (step.y > 0 ? 1 : 0)) * voxel_size.y - start.y) / ray_dir.y,
        ((cur_voxel.z + (step.z > 0 ? 1 : 0)) * voxel_size.z - start.z) / ray_dir.z
    );
    
    // DDA 遍历
    while (t < t_max) {
        // 检查边界
        if (cur_voxel.x < 0 || cur_voxel.x >= grid_size.x ||
            cur_voxel.y < 0 || cur_voxel.y >= grid_size.y ||
            cur_voxel.z < 0 || cur_voxel.z >= grid_size.z) {
            break;
        }
        
        // 计算线性索引
        int idx = cur_voxel.x + grid_size.x * (cur_voxel.y + grid_size.y * cur_voxel.z);
        
        // 标记为 camera-observed
        atomicOr(&mask_camera_rays[idx], 1);
        
        // 如果遇到 occupied voxel，停止（遮挡）
        if (occupied_grid[idx] == 1) {
            break;
        }
        
        // 选择下一个要跨越的轴
        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            cur_voxel.x += step.x;
            tMax.x += tDelta.x;
        } else if (tMax.y < tMax.z) {
            cur_voxel.y += step.y;
            tMax.y += tDelta.y;
        } else {
            cur_voxel.z += step.z;
            tMax.z += tDelta.z;
        }
    }
}
```

### 4.2 多 GPU 并行策略（已实现）

**实现方式**：使用 Python `multiprocessing.Pool` + GPU 绑定

**核心模块**：
- `occ3d_nuscenes/camera_visibility_parallel.py`: 实现 `process_sample_chunk_worker()` worker 函数
- `generate_occ3d_nuscenes.py`: 实现 `process_samples_parallel()` 主调度函数

**任务分配策略**：
1. 收集所有 `(scene_name, sample_token)` 对
2. 按 `--chunk-size` 分块（默认 10 个 sample_token/chunk）
3. 使用 `multiprocessing.Pool` 启动多个 worker（worker 数量 = min(num_gpus, num_chunks)）
4. 每个 worker 绑定到一个 GPU（通过 `CUDA_VISIBLE_DEVICES`）

**Worker 函数**（`process_sample_chunk_worker`）：
```python
def process_sample_chunk_worker(
    worker_id: int,
    gpu_id: int,
    sample_tokens_chunk: List[Tuple[str, str]],  # [(scene_name, sample_token), ...]
    args_dict: Dict[str, Any],  # 序列化后的参数
) -> Dict[str, Any]:
    # 1. 绑定 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 2. 重建复杂对象（避免跨进程共享）
    grid = VoxelGridSpec()
    reader = NuScenesReader(...)
    
    # 3. 处理每个 sample_token（完整 pipeline）
    annotations = init_annotations()
    for scene_name, sample_token in sample_tokens_chunk:
        # Stage 1 → Stage 2 → Stage 3 → Export
        ...
        update_annotations_for_frame(annotations, ...)
    
    # 4. 返回结果
    return {
        "annotations": annotations,
        "processed_count": ...,
        "errors": [...],
    }
```

**主进程调度**（`process_samples_parallel`）：
```python
def process_samples_parallel(args, reader, grid):
    # 1. 收集所有样本
    all_samples = [(scene_name, sample_token) for ...]
    
    # 2. 分块
    chunks = [all_samples[i:i+chunk_size] for i in range(0, len(all_samples), chunk_size)]
    
    # 3. 启动 worker pool
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(process_sample_chunk_worker, worker_args)
    
    # 4. 合并 annotations
    annotations = init_annotations()
    for result in results:
        merge_annotations(annotations, result["annotations"])
    
    # 5. 写入最终 annotations.json
    write_annotations_json(out_ann_path, annotations)
```

**关键特性**：
- ✅ GPU 绑定：每个 worker 通过 `CUDA_VISIBLE_DEVICES` 绑定到指定 GPU
- ✅ 参数序列化：只传递可序列化的配置，复杂对象在 worker 内部重建
- ✅ 错误处理：每个 worker 捕获异常并返回错误列表
- ✅ 进度跟踪：汇总所有 worker 的处理数量
- ✅ 文件安全：每个 worker 写入独立的 `labels.npz`，主进程统一合并 `annotations.json`

---

## 5. 文件结构与输出格式

### 5.1 项目代码结构（已实现）

```
Occ3D-master/
├── occ3d_nuscenes/              # Occ3D-nuScenes 生成模块（已实现）
│   ├── __init__.py
│   ├── nusc_io.py               # nuScenes 数据读取与坐标变换
│   ├── accumulate.py            # Stage 1: 多帧聚合
│   ├── voxel_grid.py            # 体素网格定义与工具函数
│   ├── lidar_visibility.py      # Stage 2: LiDAR ray casting (Algorithm 2)
│   ├── camera_visibility.py     # Stage 2: Camera visibility (Algorithm 3, CPU/CUDA)
│   ├── camera_visibility_parallel.py  # 多进程并行处理 worker
│   ├── cuda/                    # CUDA 扩展
│   │   ├── camera_visibility_ext.cpp
│   │   └── camera_visibility_ext.cu
│   ├── image_guided_refine.py   # Stage 3: Image-guided voxel refinement
│   ├── seg2d_provider.py       # 2D 语义统一接口
│   ├── seg2d_model.py          # 方案A: 2D 分割模型推理
│   ├── seg2d_lidar_project.py  # 方案B: LiDAR 投影伪标签
│   └── export_occ3d.py          # Export: 输出与写盘（含 merge_annotations）
├── generate_occ3d_nuscenes.py  # 主入口脚本（根目录）
├── utils/                       # 现有：复用现有工具
│   ├── custom.py               # sparse2dense (参考)
│   ├── points_in_bbox.py       # 动态物体对齐（复用）
│   └── vis_occ.py              # 可视化验证（复用）
└── ...
```

### 5.2 输出数据集结构（严格对齐官方）

```
Occpancy3D-nuScenes-V1.0/
├── trainval/
│   ├── imgs/
│   │   ├── CAM_BACK/
│   │   │   ├── n015-2018-07-18-11-07-57+0800__CAM_BACK__1531883530437525.jpg
│   │   │   └── ...
│   │   ├── CAM_BACK_LEFT/
│   │   ├── CAM_BACK_RIGHT/
│   │   ├── CAM_FRONT/
│   │   ├── CAM_FRONT_LEFT/
│   │   └── CAM_FRONT_RIGHT/
│   ├── gts/
│   │   ├── scene-0001/
│   │   │   ├── n015-2018-07-18-11-07-57+0800__CAM_BACK__1531883530437525/
│   │   │   │   └── labels.npz
│   │   │   └── ...
│   │   └── ...
│   └── annotations.json
└── test/
    ├── imgs/
    └── annotations.json
```

### 5.3 labels.npz 字段说明

| 字段名 | Shape | Dtype | 说明 |
|--------|-------|-------|------|
| `semantics` | (200, 200, 16) | uint8 | 体素语义标签，0-16 对应 nuScenes-lidarseg，17 为 free |
| `mask_lidar` | (200, 200, 16) | uint8 | LiDAR 可见性 mask，0=unobserved，1=observed |
| `mask_camera` | (200, 200, 16) | uint8 | 相机可见性 mask，0=unobserved，1=observed |

**注意**：
- Shape 为 **XYZ 顺序**：`(X=200, Y=200, Z=16)`
- 索引 `(i, j, k)` 对应的物理坐标：
  - \(x = -40 + (i + 0.5) \times 0.4\)
  - \(y = -40 + (j + 0.5) \times 0.4\)
  - \(z = -1 + (k + 0.5) \times 0.4\)

---

## 6. 配置参数与命令行接口

### 6.1 命令行参数（已实现）

```python
# 数据路径
parser.add_argument('--nusc-root', type=str, default='/mnt/data/.../nuscenes/',
                    help='nuScenes 数据根目录（服务器端完整路径，有默认值）')
parser.add_argument('--nusc-version', type=str, default='v1.0-trainval',
                    help='nuScenes 版本')
parser.add_argument('--out-root', type=str, required=True,
                    help='输出 Occ3D-nuScenes 数据集根目录')
parser.add_argument('--split', type=str, choices=['trainval', 'test', 'mini'],
                    default='trainval', help='处理的数据集 split')
parser.add_argument('--scene-name', type=str, default='',
                    help='如果设置，只处理指定的场景')

# Stage 1 选项
parser.add_argument('--window-size', type=int, default=21,
                    help='多帧聚合窗口大小（默认 21，当前帧+前后各10）')
parser.add_argument('--enable-sweeps-densification', action='store_true',
                    help='是否引入 sweeps 并对 sweeps 做 KNN 语义赋值')
parser.add_argument('--enable-mesh-recon', action='store_true',
                    help='是否启用 mesh/TSDF 补洞（Stage 1 可选步骤）')
parser.add_argument('--mesh-recon-mode', type=str, choices=['tsdf', 'poisson'], default='tsdf',
                    help='mesh 重建模式（tsdf 或 poisson）')

# Stage 2 选项
parser.add_argument('--camera-mask-cuda', action='store_true',
                    help='使用 CUDA 扩展加速 Algorithm 3（如果可用）')

# 并行处理选项
parser.add_argument('--num-gpus', type=int, default=1,
                    help='并行处理使用的 GPU 数量')
parser.add_argument('--use-parallel-camera-visibility', action='store_true',
                    help='启用多进程并行处理（Stage 2 Algorithm 3）')
parser.add_argument('--chunk-size', type=int, default=10,
                    help='每个 chunk 的 sample_token 数量（并行处理时）')

# Stage 3 选项
parser.add_argument('--enable-image-guided-refine', action='store_true',
                    help='是否启用 Stage 3: Image-guided Voxel Refinement')
parser.add_argument('--seg2d-mode', type=str, 
                    choices=['none', 'model', 'lidar_project', 'annotation'], 
                    default='none',
                    help='2D 语义生成模式：none=关闭Stage3, model=方案A, lidar_project=方案B, annotation=已有标注')
parser.add_argument('--seg2d-cache-dir', type=str, default='',
                    help='2D 语义缓存目录（保存/读取 seg2d_cam）')

# Export 选项
parser.add_argument('--link-method', type=str,
                    choices=['symlink', 'hardlink', 'copy'], default='symlink',
                    help='图像文件链接方式（默认 symlink）')
```

**注意**：
- `--num-train-scenes`, `--num-val-scenes`, `--seed` 已删除（未使用）
- `--camera-ray-image-size`, `--seg2d-model`, `--seg2d-weights`, `--refine-roi`, `--gpus`, `--workers-per-gpu` 未实现（当前版本）

### 6.2 使用示例

**单进程模式（默认）**：
```bash
python generate_occ3d_nuscenes.py \
    --nusc-root /path/to/nuscenes \
    --out-root /path/to/Occpancy3D-nuScenes-V1.0 \
    --split trainval
```

**多进程并行模式**：
```bash
python generate_occ3d_nuscenes.py \
    --nusc-root /path/to/nuscenes \
    --out-root /path/to/Occpancy3D-nuScenes-V1.0 \
    --split trainval \
    --use-parallel-camera-visibility \
    --num-gpus 4 \
    --chunk-size 10 \
    --camera-mask-cuda
```

**启用 Stage 3（使用 LiDAR 投影方案B）**：
```bash
python generate_occ3d_nuscenes.py \
    --nusc-root /path/to/nuscenes \
    --out-root /path/to/Occpancy3D-nuScenes-V1.0 \
    --enable-image-guided-refine \
    --seg2d-mode lidar_project \
    --seg2d-cache-dir /path/to/cache
```

---

## 7. 验证与检查点

### 7.1 数据完整性检查

1. **文件数量**：
   - 检查 `gts/` 下每个 scene 的 frame 数量是否与 `annotations.json` 一致
   - 检查 `imgs/` 下每个相机的图像数量

2. **labels.npz 格式**：
   ```python
   data = np.load('labels.npz')
   assert 'semantics' in data
   assert 'mask_lidar' in data
   assert 'mask_camera' in data
   assert data['semantics'].shape == (200, 200, 16)
   assert data['semantics'].dtype == np.uint8
   assert np.all(data['mask_camera'] <= data['mask_lidar'])  # mask_camera 是 mask_lidar 的子集
   ```

### 7.2 语义合理性检查

1. **free 类比例**：
   - `semantics == 17` 的比例应该很高（通常 > 50%），这是正常的（大部分空间是 free）

2. **mask 关系**：
   - `mask_camera` 应该是 `mask_lidar` 的子集（因为 AND 操作）
   - `mask_camera` 的覆盖率应该明显小于 `mask_lidar`（相机盲区和遮挡导致）

3. **语义分布**：
   - 检查 `semantics` 的类别分布是否合理（0-16 对应 nuScenes-lidarseg 类别）

### 7.3 可视化验证（复用现有工具）

使用 `utils/vis_occ.py` 可视化生成的 `labels.npz`：
```python
data = np.load('labels.npz')
semantics = data['semantics']
mask_lidar = data['mask_lidar']
mask_camera = data['mask_camera']

# 使用 vis_occ.py 的可视化函数
from utils.vis_occ import main as vis_occ
vis_occ(semantics, mask_lidar, mask_camera, voxel_size=[0.4, 0.4, 0.4])
```

**预期结果**：
- `mask_lidar` 覆盖范围较大（包含 free space）
- `mask_camera` 覆盖范围明显更小（被遮挡和盲区裁掉）
- `semantics` 中 free（17）占大部分，occupied 类别集中在物体表面

### 7.4 与官方数据对比（如果有）

如果可以获得官方 Occ3D-nuScenes 的样本数据：
- 对比 `mask_lidar` 和 `mask_camera` 的覆盖率
- 对比语义标签的分布
- 对比 `annotations.json` 的结构

---

## 8. 依赖与环境要求

### 8.1 Python 依赖（需要新增）

在 `requirement.txt` 基础上添加：
```
nuscenes-devkit>=1.1.0
pyquaternion>=0.9.0
```

### 8.2 CUDA 要求

- **CUDA 版本**：>= 10.2（支持 PyTorch CUDA extension）
- **GPU 显存**：建议每张 GPU >= 8GB（处理 200×200×16 的体素网格）
- **PyTorch**：已包含在现有依赖中

### 8.3 CUDA Extension（已实现）

**实现位置**：
- `occ3d_nuscenes/cuda/camera_visibility_ext.cpp`: C++ 绑定
- `occ3d_nuscenes/cuda/camera_visibility_ext.cu`: CUDA kernel 实现

**编译方式**：
- 使用 PyTorch 的 JIT（Just-In-Time）编译
- 首次运行时自动编译，后续运行会复用已编译的扩展
- 如果编译失败，自动回退到 CPU 参考实现

**使用方式**：
- 通过 `--camera-mask-cuda` 参数启用 CUDA 加速
- `camera_visibility.py` 中的 `_try_load_cuda_ext()` 函数负责加载扩展
- 如果 CUDA 扩展不可用，自动使用 CPU 实现（`camera_visibility_cpu()`）

**注意**：
- JIT 编译需要 CUDA 工具链（nvcc）
- 首次编译可能需要几分钟时间
- 编译后的扩展会缓存在 PyTorch 的缓存目录中

---

## 9. 实施状态

### ✅ Phase 1: 基础框架搭建（已完成）
- ✅ 创建 `occ3d_nuscenes/` 目录结构
- ✅ 实现 `nusc_io.py`（数据读取与坐标变换）
- ✅ 实现 `voxel_grid.py`（体素网格工具函数）
- ✅ 实现 `accumulate.py`（多帧聚合，复用 `points_in_bbox.py`）

### ✅ Phase 2: LiDAR Ray Casting（已完成）
- ✅ 实现 `lidar_visibility.py`（CPU 版本）
- ✅ 验证 `semantics` 和 `mask_lidar` 的生成

### ✅ Phase 3: CUDA Algorithm 3（已完成）
- ✅ 编写 `cuda/camera_visibility_ext.cu`（CUDA kernel）
- ✅ 编写 `cuda/camera_visibility_ext.cpp`（C++ 绑定）
- ✅ 实现 `camera_visibility.py`（Python 接口，支持 CPU/CUDA 自动切换）
- ✅ CUDA extension 支持 JIT 编译
- ✅ 验证 `mask_camera_rays` 的生成

### ✅ Phase 4: 输出与整合（已完成）
- ✅ 实现 `export_occ3d.py`（写 `labels.npz` 和 `annotations.json`）
- ✅ 实现 `merge_annotations()` 函数（合并多进程结果）
- ✅ 实现图像文件处理（symlink/hardlink/copy）
- ✅ 实现命令行工具 `generate_occ3d_nuscenes.py`（根目录）

### ✅ Phase 5: 多 GPU 并行与优化（已完成）
- ✅ 实现 `camera_visibility_parallel.py`（多进程 worker）
- ✅ 实现 `process_samples_parallel()`（主调度函数）
- ✅ 使用 `multiprocessing.Pool` 进行任务分配
- ✅ GPU 绑定和参数序列化

### ✅ Phase 6: Stage 3 实现（已完成）
- ✅ 实现 `image_guided_refine.py`（CPU 参考实现）
- ✅ 实现 `seg2d_provider.py`（统一接口）
- ✅ 实现 `seg2d_model.py`（方案A：2D 模型推理，占位符）
- ✅ 实现 `seg2d_lidar_project.py`（方案B：LiDAR 投影）

### 🔄 Phase 7: 验证与文档（进行中）
- ✅ 使用 `utils/vis_occ.py` 可视化验证（可用）
- ⏳ 数据完整性检查（待大规模测试）
- ⏳ 与官方格式对比（如果有官方数据）
- ✅ 更新实现计划文档（本文档）

---

## 10. 参考文献与链接

1. **论文**：
   - [Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving](https://arxiv.org/pdf/2304.14365)

2. **官方仓库**：
   - [Tsinghua-MARS-Lab/Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D)

3. **nuScenes 相关**：
   - [nuScenes-lidarseg README](https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/fcc41628d41060b3c1a86928751e5a571d2fc2fa/python-sdk/nuscenes/eval/lidarseg/README.md)
   - [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

4. **本地文档**：
   - `cursor_gen_files/Occ3D_Paper_Detailed_Interpretation.md` - 论文详细解读

---

## 附录：关键设计决策总结

| 决策项 | 选择 | 理由 | 状态 |
|--------|------|------|------|
| 多帧聚合窗口 | 21 keyframes（当前+前后各10） | 论文明确说明，nuScenes 使用 21 帧 | ✅ 已实现 |
| 动态物体类别 | vehicle/pedestrian/bicycle/motorcycle | movable_object（barrier/cone）按静态处理，避免对齐错误 | ✅ 已实现 |
| mask_camera 定义 | `mask_lidar & mask_camera_rays` | 确保只在 LiDAR observed 区域评测，符合 vision-centric 任务 | ✅ 已实现 |
| 相机分辨率 | 默认 native | 与论文保持一致，像素射线与内参严格绑定 | ✅ 已实现 |
| 数据格式 | XYZ 顺序，uint8 dtype | 统一约定，便于后续训练和可视化 | ✅ 已实现 |
| 图像文件处理 | 优先 symlink | 节省空间，适合服务器环境 | ✅ 已实现 |
| 多GPU并行 | multiprocessing.Pool + GPU绑定 | Python 标准库，易于实现和维护 | ✅ 已实现 |
| train/val split | 暂不实现 | 当前版本处理所有场景，split 由用户通过 `--split` 指定 | ⏳ 待实现 |
| Stage 3 seg2d | 方案A/B统一接口 | 支持模型推理和LiDAR投影两种方式 | ✅ 已实现 |

---

## 11. 已知限制与未来改进

### 11.1 当前限制

1. **train/val split 未实现**：
   - 当前版本处理所有场景，不自动划分 train/val
   - 用户需要手动指定 `--split` 或 `--scene-name`
   - 未来可添加基于场景名称的自动划分逻辑

2. **Stage 3 方案A（2D模型推理）**：
   - `seg2d_model.py` 目前为占位符实现
   - 需要集成具体的 2D 分割模型（如 InternImage）
   - 需要实现类别映射表

3. **Mesh Reconstruction**：
   - `--enable-mesh-recon` 参数已定义，但实现可能不完整
   - 需要验证 TSDF/Poisson 重建的实际效果

4. **CUDA Extension 编译**：
   - 当前使用 JIT 编译，首次运行可能较慢
   - 建议预编译或提供预编译版本

### 11.2 性能优化建议

1. **I/O 优化**：
   - 考虑使用异步 I/O 或线程池处理图像链接
   - 批量写入 `labels.npz` 减少文件系统调用

2. **内存优化**：
   - 对于大规模数据集，考虑流式处理
   - 及时释放中间结果的内存

3. **并行优化**：
   - 当前使用进程级并行，可考虑线程级并行（GIL 限制）
   - 对于 I/O 密集型任务，可增加 worker 数量

---

**文档结束**

