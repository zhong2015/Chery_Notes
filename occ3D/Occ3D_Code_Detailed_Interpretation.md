# Occ3D-nuScenes 代码实现详细解读

**文档版本**: v1.1  
**创建日期**: 2025-01-27  
**更新日期**: 2025-01-XX  
**项目路径**: `E:\Chery\dz\Occ\Occ3D-master`  
**基于论文**: [Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving](https://arxiv.org/pdf/2304.14365)

**重要更新**：
- 已移除CUDA支持，camera visibility现在使用CPU实现
- 删除了`CameraVisibilityOptions`类和`--camera-mask-cuda`参数
- 简化了`camera_visibility_mask_camera_rays`函数实现
- 通过多进程并行实现加速，每个worker独立处理不同的sample chunk

---

## 目录

1. [总体架构与数据流](#1-总体架构与数据流)
2. [核心模块详解](#2-核心模块详解)
3. [数学公式与算法](#3-数学公式与算法)
4. [输入输出数据结构](#4-输入输出数据结构)
5. [模块间依赖关系](#5-模块间依赖关系)
6. [并行处理机制](#6-并行处理机制)

---

## 1. 总体架构与数据流

### 1.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        generate_occ3d_nuscenes.py                    │
│                         (主入口脚本，根目录)                          │
│  • 命令行参数解析                                                    │
│  • 单进程/多进程模式选择                                              │
│  • 场景遍历与任务分发                                                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    occ3d_nuscenes/ 模块包                            │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │   voxel_grid.py │  │    nusc_io.py   │  │  accumulate.py  │   │
│  │  (体素网格定义)  │  │  (数据读取与变换) │  │  (Stage 1 聚合)  │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │lidar_visibility │  │camera_visibility │  │image_guided_    │   │
│  │     .py         │  │     .py          │  │  refine.py      │   │
│  │ (Stage 2 Algo 2)│  │ (Stage 2 Algo 3)│  │ (Stage 3 细化)   │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │  seg2d_provider  │  │  seg2d_lidar_   │  │  seg2d_model.py │   │
│  │     .py          │  │  project.py     │  │  (占位符)        │   │
│  │ (2D语义统一接口)  │  │ (方案B:投影)    │  │ (方案A:模型)     │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐                        │
│  │  export_occ3d.py │  │camera_visibility│                        │
│  │  (导出与合并)    │  │  _parallel.py    │                        │
│  │                  │  │  (多进程worker) │                        │
│  └─────────────────┘  └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 完整处理流程图

```
开始: generate_occ3d_nuscenes.py::main()
  ↓
[1] 初始化
  • 创建 VoxelGridSpec (voxel_grid.py)
  • 创建 NuScenesReader (nusc_io.py)
  • 解析命令行参数
  ↓
[2] 选择处理模式
  • 单进程: process_samples_sequential()
  • 多进程: process_samples_parallel()
  ↓
[3] 遍历场景和样本
  for scene_name in scene_names:
    for sample_token in sample_tokens:
      ↓
      [3.1] 读取相机信息 (nusc_io.py::get_camera_infos)
      [3.2] 导出图像链接 (export_occ3d.py::export_images)
      ↓
      ┌─────────────────────────────────────────────────────┐
      │ Stage 1: Voxel Densification                        │
      │ (accumulate.py::voxel_densification)                │
      │                                                       │
      │ [1.1] 获取窗口内keyframes (nusc_io.py::get_window_   │
      │        keyframes, window_size=21)                    │
      │ [1.2] 对每个keyframe:                                 │
      │   • 加载LiDAR点云 (nusc_io.py::load_lidar_points)    │
      │   • 加载语义标签 (nusc_io.py::load_lidarseg_labels)   │
      │   • 计算坐标变换 (nusc_io.py::compute_T_lidar_s_to_  │
      │     ego_t)                                           │
      │   • 变换点到target ego (nusc_io.py::transform_points)│
      │ [1.3] 拼接所有点云和标签                              │
      │                                                       │
      │ 输出: P_dense (N,3), L_dense (N,), lidar_origins    │
      └─────────────────────────────────────────────────────┘
      ↓
      ┌─────────────────────────────────────────────────────┐
      │ Stage 2: LiDAR Visibility (Algorithm 2)             │
      │ (lidar_visibility.py::lidar_visibility_ray_casting)   │
      │                                                       │
      │ [2.1] 对每个点 P_dense[i]:                            │
      │   • 从对应lidar_origin到P_dense[i]做DDA遍历          │
      │   • 路径上的voxel标记为free                           │
      │   • 终点voxel标记为occupied                          │
      │ [2.2] 语义投票:                                       │
      │   • 每个occupied voxel收集语义标签投票                │
      │   • 多数投票决定最终语义                              │
      │ [2.3] 生成mask_lidar:                                │
      │   • observed = (occ_count > 0) | (free_count > 0)    │
      │                                                       │
      │ 输出: semantics (X,Y,Z) uint8 0..17,                 │
      │      mask_lidar (X,Y,Z) uint8 0/1                    │
      └─────────────────────────────────────────────────────┘
      ↓
      ┌─────────────────────────────────────────────────────┐
      │ Stage 2: Camera Visibility (Algorithm 3)             │
      │ (camera_visibility.py::camera_visibility_mask_       │
      │  camera_rays)                                         │
      │                                                       │
      │ [3.1] 构建occupied_grid:                              │
      │   occupied_grid = (mask_lidar == 1) &                │
      │                   (semantics != FREE_LABEL)           │
      │ [3.2] 调用 camera_visibility_mask_camera_rays()      │
      │   • CPU实现，对每个相机每个像素进行DDA遍历           │
      │   • 标记所有被相机射线扫到的体素为observed          │
      │                                                       │
      │ 输出: mask_camera_rays (X,Y,Z) uint8 0/1              │
      └─────────────────────────────────────────────────────┘
      ↓
      ┌─────────────────────────────────────────────────────┐
      │ Stage 3: Image-guided Refinement (可选)              │
      │ (image_guided_refine.py::image_guided_voxel_refine_  │
      │  cpu)                                                │
      │                                                       │
      │ [4.1] 获取2D语义 (seg2d_provider.py::build_or_load_  │
      │        seg2d_cam):                                   │
      │   • 方案A (model): 调用2D分割模型                     │
      │   • 方案B (lidar_project): LiDAR投影伪标签            │
      │   • 方案C (none): 跳过Stage 3                        │
      │ [4.2] 对每个相机像素射线:                             │
      │   • 沿射线DDA遍历                                     │
      │   • 找到首个语义匹配的voxel                           │
      │   • 将该voxel之前的occupied voxels置为free           │
      │ [4.3] (建议) 重新计算mask_camera_rays                 │
      │                                                       │
      │ 输出: refined semantics (X,Y,Z) uint8 0..17          │
      └─────────────────────────────────────────────────────┘
      ↓
      ┌─────────────────────────────────────────────────────┐
      │ Export: 保存结果                                      │
      │ (export_occ3d.py)                                     │
      │                                                       │
      │ [5.1] 计算mask_camera:                                 │
      │   mask_camera = mask_lidar & mask_camera_rays         │
      │ [5.2] 保存labels.npz:                                 │
      │   save_labels_npz(path, semantics, mask_lidar,        │
      │                   mask_camera)                        │
      │ [5.3] 更新annotations:                                 │
      │   update_annotations_for_frame(...)                   │
      └─────────────────────────────────────────────────────┘
      ↓
[4] 写入annotations.json (单进程) 或 合并annotations (多进程)
  ↓
结束
```

### 1.3 多进程并行流程图

```
process_samples_parallel()
  ↓
[1] 收集所有 (scene_name, sample_token) 对
  ↓
[2] 分块 (chunk_size个样本一块)
  chunks = [chunk_0, chunk_1, ..., chunk_N]
  ↓
[3] 创建multiprocessing.Pool (num_workers = min(num_gpus, len(chunks)))
  ↓
[4] 为每个chunk分配worker
  worker_args = [
    (worker_id=0, gpu_id=0, chunk_0, args_dict),
    (worker_id=1, gpu_id=1, chunk_1, args_dict),
    ...
  ]
  ↓
[5] 并行执行 (pool.starmap)
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ Worker 0     │  │ Worker 1     │  │ Worker 2     │
  │ GPU 0        │  │ GPU 1        │  │ GPU 2        │
  │              │  │              │  │              │
  │ process_     │  │ process_     │  │ process_     │
  │ sample_      │  │ sample_      │  │ sample_      │
  │ chunk_worker │  │ chunk_worker │  │ chunk_worker │
  │              │  │              │  │              │
  │ 返回:        │  │ 返回:        │  │ 返回:        │
  │ {            │  │ {            │  │ {            │
  │   annotations│  │   annotations│  │   annotations│
  │   processed  │  │   processed  │  │   processed  │
  │   errors     │  │   errors     │  │   errors     │
  │ }            │  │ }            │  │ }            │
  └──────────────┘  └──────────────┘  └──────────────┘
  ↓
[6] 合并结果
  • merge_annotations(target, source) 合并annotations
  • 汇总processed_count和errors
  ↓
[7] 写入最终annotations.json
```

---

## 2. 核心模块详解

### 2.1 voxel_grid.py - 体素网格定义

#### 2.1.1 功能与Motivation

**功能**：
- 定义Occ3D-nuScenes的固定体素网格参数
- 提供点云到体素索引的转换工具函数
- 提供体素索引的线性化/反线性化函数

**Motivation**：
- **统一性**：确保整个pipeline使用相同的体素网格定义，避免不一致
- **可维护性**：集中管理网格参数，便于修改和验证
- **性能**：提供高效的索引转换函数，避免重复计算

#### 2.1.2 核心数据结构

**VoxelGridSpec (dataclass, frozen=True)**

```python
@dataclass(frozen=True)
class VoxelGridSpec:
    pc_range: Tuple[float, float, float, float, float, float] = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)
    voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4)
    grid_shape: Tuple[int, int, int] = (200, 200, 16)  # (X,Y,Z)
```

**参数详解**：

| 参数 | 类型 | 数值 | 含义 |
|------|------|------|------|
| `pc_range` | `Tuple[6]` | `(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)` | 点云范围 `[x_min, y_min, z_min, x_max, y_max, z_max]` (单位: 米) |
| `voxel_size` | `Tuple[3]` | `(0.4, 0.4, 0.4)` | 体素尺寸 `[vx, vy, vz]` (单位: 米) |
| `grid_shape` | `Tuple[3]` | `(200, 200, 16)` | 网格大小 `[X, Y, Z]` (体素数量) |

**验证关系**：
$$(x_{max} - x_{min}, y_{max} - y_{min}, z_{max} - z_{min}) = (vx \cdot X, vy \cdot Y, vz \cdot Z)$$

即：$80 = 0.4 \times 200$, $80 = 0.4 \times 200$, $6.4 = 0.4 \times 16$

#### 2.1.3 核心函数

**1. points_to_voxel_indices()**

**功能**：将ego坐标系下的3D点转换为体素索引

**数学公式**：
对于点 $p = (x, y, z)$，体素索引 $(i_x, i_y, i_z)$ 计算为：
$$i_x = \lfloor \frac{x - x_{min}}{v_x} \rfloor$$
$$i_y = \lfloor \frac{y - y_{min}}{v_y} \rfloor$$
$$i_z = \lfloor \frac{z - z_{min}}{v_z} \rfloor$$

**输入**：
- `points_xyz`: `np.ndarray` shape `(N, 3)`, dtype `float64`
  - 每行是一个3D点 `[x, y, z]` (ego坐标系，单位: 米)
  - 范围：理论上无限制，但通常 $x \in [-40, 40]$, $y \in [-40, 40]$, $z \in [-1, 5.4]$

**输出**：
- `np.ndarray` shape `(N, 3)`, dtype `int32`
  - 每行是体素索引 `[ix, iy, iz]`
  - 范围：有效索引 $i_x \in [0, X)$, $i_y \in [0, Y)$, $i_z \in [0, Z)$
  - 超出范围的索引会被标记，需要后续过滤

**2. xyz_to_linear()**

**功能**：将3D体素索引转换为线性索引

**数学公式**：
对于体素索引 $(i_x, i_y, i_z)$，线性索引 $idx$ 为：
$$idx = i_x + X \cdot (i_y + Y \cdot i_z)$$

这是**Z-major**存储顺序（Z轴变化最快）。

**输入**：
- `ix, iy, iz`: `np.ndarray` shape `(N,)`, dtype `int64`
  - 体素索引，范围：$i_x \in [0, X)$, $i_y \in [0, Y)$, $i_z \in [0, Z)$

**输出**：
- `np.ndarray` shape `(N,)`, dtype `int64`
  - 线性索引，范围：$idx \in [0, X \times Y \times Z) = [0, 640000)$

**3. linear_to_xyz()**

**功能**：将线性索引转换为3D体素索引（`xyz_to_linear`的逆操作）

**数学公式**：
$$i_z = \lfloor \frac{idx}{X \times Y} \rfloor$$
$$rem = idx - i_z \times X \times Y$$
$$i_y = \lfloor \frac{rem}{X} \rfloor$$
$$i_x = rem - i_y \times X$$

**输入**：
- `idx`: `np.ndarray` shape `(N,)`, dtype `int64`
  - 线性索引，范围：$idx \in [0, X \times Y \times Z)$

**输出**：
- `np.ndarray` shape `(N, 3)`, dtype `int32`
  - 体素索引 `[ix, iy, iz]`

**4. voxel_center_xyz()**

**功能**：计算体素中心的世界坐标

**数学公式**：
$$x = x_{min} + (i_x + 0.5) \cdot v_x$$
$$y = y_{min} + (i_y + 0.5) \cdot v_y$$
$$z = z_{min} + (i_z + 0.5) \cdot v_z$$

**输入**：
- `ix, iy, iz`: `int`
  - 体素索引

**输出**：
- `Tuple[float, float, float]`
  - 体素中心坐标 `(x, y, z)` (单位: 米)

---

### 2.2 nusc_io.py - nuScenes数据读取与坐标变换

#### 2.2.1 功能与Motivation

**功能**：
- 封装`nuscenes-devkit`，提供简化的数据访问接口
- 实现坐标系统之间的变换（lidar_s → ego_s → global → ego_t → cam）
- 提供多帧聚合的窗口获取功能

**Motivation**：
- **抽象层**：隐藏nuScenes复杂的JSON结构，提供清晰的接口
- **稳定性**：显式实现坐标变换，便于审计和调试
- **复用性**：统一的变换接口，避免重复代码

#### 2.2.2 核心数据结构

**1. CameraInfo (dataclass, frozen=True)**

```python
@dataclass(frozen=True)
class CameraInfo:
    cam_name: str                    # 相机名称，如 "CAM_FRONT"
    sample_data_token: str           # nuScenes sample_data token
    img_path: str                    # 图像文件路径（绝对或相对）
    width: int                       # 图像宽度（像素）
    height: int                      # 图像高度（像素）
    intrinsic: np.ndarray            # (3,3) 相机内参矩阵 K
    T_cam2ego: np.ndarray           # (4,4) 从相机到ego的变换矩阵
    ego_pose: Dict[str, Any]        # ego位姿（translation + rotation quaternion）
```

**详细说明**：

| 字段 | 类型 | 取值范围/含义 |
|------|------|-------------|
| `cam_name` | `str` | `"CAM_FRONT"`, `"CAM_FRONT_LEFT"`, `"CAM_FRONT_RIGHT"`, `"CAM_BACK"`, `"CAM_BACK_LEFT"`, `"CAM_BACK_RIGHT"` |
| `sample_data_token` | `str` | nuScenes唯一标识符（UUID格式） |
| `img_path` | `str` | 图像文件路径，可以是绝对路径或相对路径 |
| `width` | `int` | 图像宽度，通常 1600 像素 |
| `height` | `int` | 图像高度，通常 900 像素 |
| `intrinsic` | `np.ndarray` | `(3,3)` float64，相机内参矩阵 $K$，格式：<br>$\begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$ |
| `T_cam2ego` | `np.ndarray` | `(4,4)` float64，齐次变换矩阵，将相机坐标系点变换到ego坐标系 |
| `ego_pose` | `Dict` | `{"translation": [x,y,z], "rotation": [w,x,y,z]}`，ego在全局坐标系中的位姿 |

**2. LidarInfo (dataclass, frozen=True)**

```python
@dataclass(frozen=True)
class LidarInfo:
    sample_data_token: str           # nuScenes sample_data token
    lidar_path: str                  # LiDAR点云文件路径
    lidarseg_path: Optional[str]     # LiDAR语义标签文件路径（None表示无标签）
    T_lidar2ego: np.ndarray          # (4,4) 从LiDAR到ego的变换矩阵
    ego_pose: Dict[str, Any]         # ego位姿
```

**详细说明**：

| 字段 | 类型 | 取值范围/含义 |
|------|------|-------------|
| `sample_data_token` | `str` | nuScenes唯一标识符 |
| `lidar_path` | `str` | `.pcd.bin`文件路径 |
| `lidarseg_path` | `Optional[str]` | `.bin`文件路径，包含每个点的语义标签（uint8，0..16），keyframe通常有，sweep通常无 |
| `T_lidar2ego` | `np.ndarray` | `(4,4)` float64，齐次变换矩阵 |
| `ego_pose` | `Dict` | 同CameraInfo |

#### 2.2.3 核心函数

**1. compute_T_lidar_s_to_ego_t()**

**功能**：计算从source LiDAR坐标系到target ego坐标系的变换矩阵

**数学公式**：
变换链：$p^{lidar_s} \rightarrow p^{ego_s} \rightarrow p^{global} \rightarrow p^{ego_t}$

$$T^{ego_t}_{lidar_s} = (T^{ego_t}_{global})^{-1} \cdot T^{global}_{ego_s} \cdot T^{ego_s}_{lidar_s}$$

其中：
- $T^{ego_s}_{lidar_s}$: 从source LiDAR到source ego（来自calibrated_sensor）
- $T^{global}_{ego_s}$: 从source ego到global（来自ego_pose）
- $T^{ego_t}_{global}$: 从global到target ego（来自target ego_pose）

**输入**：
- `lidar_sd_token_s`: `str` - source LiDAR的sample_data token
- `lidar_sd_token_t`: `str` - target LiDAR的sample_data token

**输出**：
- `np.ndarray` shape `(4, 4)`, dtype `float64`
  - 齐次变换矩阵，可直接用于`transform_points()`

**2. transform_points()**

**功能**：使用齐次变换矩阵变换3D点

**数学公式**：
对于点 $p = (x, y, z)$，变换后的点 $p'$ 为：
$$\begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = T \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$

**输入**：
- `points_xyz`: `np.ndarray` shape `(N, 3)`, dtype `float64`
  - 源坐标系下的点
- `T`: `np.ndarray` shape `(4, 4)`, dtype `float64`
  - 齐次变换矩阵

**输出**：
- `np.ndarray` shape `(N, 3)`, dtype `float64`
  - 目标坐标系下的点

**3. get_window_keyframes()**

**功能**：获取以target keyframe为中心的窗口内的所有keyframe tokens

**算法**：
1. 从target向后遍历`prev`指针，收集`half = window_size // 2`个keyframe
2. 从target向前遍历`next`指针，收集`half`个keyframe
3. 返回：`[prev_half, ..., prev_1, target, next_1, ..., next_half]`

**输入**：
- `sample_token`: `str` - target keyframe的token
- `window_size`: `int` - 窗口大小（必须为奇数，默认21）

**输出**：
- `List[str]` - keyframe token列表，长度 $\leq$ `window_size`（如果到达场景边界会截断）

**4. load_lidar_points()**

**功能**：加载LiDAR点云文件

**文件格式**：
- nuScenes使用`.pcd.bin`格式
- 每个点5个float32：`[x, y, z, intensity, ring_index]`
- 只读取前3个维度（xyz）

**输入**：
- `lidar_path`: `str` - `.pcd.bin`文件路径

**输出**：
- `np.ndarray` shape `(N, 3)`, dtype `float64`
  - 点云坐标（LiDAR坐标系，单位: 米）

**5. load_lidarseg_labels()**

**功能**：加载LiDAR语义分割标签

**文件格式**：
- `.bin`文件，每个点一个uint8标签
- 标签范围：0..16（nuScenes-lidarseg类别）

**输入**：
- `lidarseg_path`: `str` - `.bin`文件路径

**输出**：
- `np.ndarray` shape `(N,)`, dtype `uint8`
  - 语义标签，范围：0..16
  - 0通常表示"ignore"或"void"

---

### 2.3 accumulate.py - Stage 1: 体素密集化

#### 2.3.1 功能与Motivation

**功能**：
- 实现Stage 1的核心逻辑：多帧聚合
- 将多个keyframe的LiDAR点云聚合到target keyframe的ego坐标系
- 输出密集的点云和对应的语义标签

**Motivation**：
- **解决稀疏性问题**：单帧LiDAR点云只覆盖约4.7%的体素，多帧聚合可显著提高覆盖率
- **时间一致性**：通过聚合历史帧和未来帧，获得更完整的场景表示
- **模块化设计**：将聚合逻辑独立出来，便于测试和扩展（未来可加入sweeps、KNN、mesh重建）

#### 2.3.2 核心数据结构

**DensifyOptions (dataclass)**

```python
@dataclass
class DensifyOptions:
    window_size: int = 21                          # 聚合窗口大小（keyframe数量）
    enable_sweeps_densification: bool = False       # 是否启用sweeps + KNN（未实现）
    enable_mesh_recon: bool = False                 # 是否启用mesh重建（未实现）
    mesh_recon_mode: str = "tsdf"                  # mesh重建模式：'tsdf' | 'poisson'
```

#### 2.3.3 核心函数

**1. densify_keyframes_only()**

**功能**：仅聚合keyframe（当前实现，未包含sweeps和mesh重建）

**算法流程**：

```
输入: target_sample_token, window_size=21
  ↓
[1] 获取窗口内所有keyframe tokens
  window_samples = get_window_keyframes(target_sample_token, window_size)
  ↓
[2] 获取target keyframe的LiDAR信息
  tgt_lidar = get_lidar_keyframe(target_sample_token, with_lidarseg=True)
  tgt_sd_token = tgt_lidar.sample_data_token
  ↓
[3] 对每个source keyframe:
  for s_token in window_samples:
    [3.1] 加载source LiDAR点云和标签
      pts = load_lidar_points(src_lidar.lidar_path)
      lbl = load_lidarseg_labels(src_lidar.lidarseg_path)
    [3.2] 计算变换矩阵
      T = compute_T_lidar_s_to_ego_t(src_lidar.sample_data_token, tgt_sd_token)
    [3.3] 变换点到target ego坐标系
      pts_t = transform_points(pts, T)
    [3.4] 记录LiDAR原点（用于后续ray casting）
      origin = transform_points([0,0,0], T)[0]
    [3.5] 添加到列表
      all_points.append(pts_t)
      all_labels.append(lbl)
      lidar_origins.append(origin)
  ↓
[4] 拼接所有点云和标签
  P_dense = concatenate(all_points)  # (N_total, 3)
  L_dense = concatenate(all_labels)  # (N_total,)
  ↓
输出: P_dense, L_dense, lidar_origins
```

**数学公式**：

对于source keyframe $s$ 中的点 $p^{lidar_s}$，变换到target ego坐标系：
$$p^{ego_t} = T^{ego_t}_{lidar_s} \cdot p^{lidar_s}$$

其中 $T^{ego_t}_{lidar_s}$ 的计算见`nusc_io.py::compute_T_lidar_s_to_ego_t()`。

**输入**：
- `reader`: `NuScenesReader` - 数据读取器
- `target_sample_token`: `str` - target keyframe的token
- `window_size`: `int` - 窗口大小（默认21，必须为奇数）

**输出**：
- `P_dense`: `np.ndarray` shape `(N, 3)`, dtype `float64`
  - 聚合后的点云（target ego坐标系，单位: 米）
  - $N$ 是所有keyframe的点数总和，通常 $N \approx 21 \times 30000 = 630000$
- `L_dense`: `np.ndarray` shape `(N,)`, dtype `uint8`
  - 对应的语义标签，范围：0..16
- `lidar_origins`: `List[np.ndarray]` length `window_size`
  - 每个keyframe的LiDAR原点在target ego坐标系中的位置
  - 每个元素shape `(3,)`, dtype `float64`

**2. voxel_densification()**

**功能**：Stage 1的统一入口，根据选项选择不同的实现路径

**当前实现**：
- 如果`enable_sweeps_densification=True`或`enable_mesh_recon=True`，抛出`NotImplementedError`
- 否则调用`densify_keyframes_only()`

**未来扩展**：
- 当实现sweeps + KNN时，会先调用`densify_keyframes_only()`，然后：
  1. 加载sweeps点云（无标签）
  2. 使用KNN从keyframe点传播标签
  3. 合并到`P_dense`和`L_dense`
- 当实现mesh重建时，会在点云聚合后：
  1. 对非地面点使用TSDF重建
  2. 对地面点使用虚拟网格拟合
  3. 生成更密集的表面点

---

### 2.4 lidar_visibility.py - Stage 2: LiDAR可见性（Algorithm 2）

#### 2.4.1 功能与Motivation

**功能**：
- 实现论文Algorithm 2：LiDAR ray casting
- 生成`semantics`（包含free标签=17）和`mask_lidar`（LiDAR可见性mask）

**Motivation**：
- **解决遮挡问题**：区分"free"（射线穿过）和"unobserved"（未被射线扫到）
- **语义传播**：通过ray casting将点云的语义标签传播到体素网格
- **可见性标记**：生成`mask_lidar`，标记哪些体素被LiDAR观测到

#### 2.4.2 核心数据结构

**LidarVisibilityResult (dataclass)**

```python
@dataclass
class LidarVisibilityResult:
    semantics: np.ndarray    # (X,Y,Z) uint8, 0..17
    mask_lidar: np.ndarray   # (X,Y,Z) uint8, 0/1
    occ_count: np.ndarray     # (X,Y,Z) uint16
    free_count: np.ndarray    # (X,Y,Z) uint16
```

**详细说明**：

| 字段 | 形状 | 类型 | 取值范围 | 含义 |
|------|------|------|---------|------|
| `semantics` | `(200, 200, 16)` | `uint8` | 0..17 | 体素语义标签：0..16为nuScenes类别，17为free |
| `mask_lidar` | `(200, 200, 16)` | `uint8` | 0/1 | LiDAR可见性mask：1=observed（被射线扫到），0=unobserved |
| `occ_count` | `(200, 200, 16)` | `uint16` | 0..65535 | 每个体素被标记为occupied的次数（用于投票） |
| `free_count` | `(200, 200, 16)` | `uint16` | 0..65535 | 每个体素被标记为free的次数 |

**常量**：
- `FREE_LABEL = 17`: free体素的语义标签

#### 2.4.3 核心算法：3D DDA (Digital Differential Analyzer)

**DDA算法**：高效遍历射线穿过的所有体素

**数学原理**：

给定射线起点 $o = (o_x, o_y, o_z)$ 和终点 $p = (p_x, p_y, p_z)$，射线方向：
$$d = p - o = (d_x, d_y, d_z)$$

参数化射线：
$$r(t) = o + t \cdot d, \quad t \in [0, 1]$$

**体素空间坐标**（连续）：
$$o_{vox} = \frac{o - pc_{min}}{voxel_{size}} = (o_x', o_y', o_z')$$
$$d_{vox} = \frac{d}{voxel_{size}} = (d_x', d_y', d_z')$$

**步进计算**：
- 沿X轴步进距离：$\Delta t_x = \frac{1}{|d_x'|}$（如果$d_x' \neq 0$）
- 沿Y轴步进距离：$\Delta t_y = \frac{1}{|d_y'|}$
- 沿Z轴步进距离：$\Delta t_z = \frac{1}{|d_z'|}$

**当前体素索引**：
$$i_x = \lfloor o_x' + t \cdot d_x' \rfloor$$
$$i_y = \lfloor o_y' + t \cdot d_y' \rfloor$$
$$i_z = \lfloor o_z' + t \cdot d_z' \rfloor$$

**到下一个体素边界的距离**：
$$tMax_x = \frac{(i_x + 1) - (o_x' + t \cdot d_x')}{d_x'} \quad \text{(如果} d_x' > 0\text{)}$$
$$tMax_y = \frac{(i_y + 1) - (o_y' + t \cdot d_y')}{d_y'}$$
$$tMax_z = \frac{(i_z + 1) - (o_z' + t \cdot d_z')}{d_z'}$$

**步进规则**：选择$tMax$最小的轴进行步进。

#### 2.4.4 核心函数

**1. _dda_traverse_voxels()**

**功能**：使用3D DDA算法遍历从origin到target的所有体素

**算法流程**：

```
输入: origin_xyz, target_xyz, grid
  ↓
[1] 转换到体素空间坐标
  o = (origin - pc_min) / voxel_size
  p = (target - pc_min) / voxel_size
  d = p - o
  ↓
[2] 射线-包围盒相交检测
  if not ray_box_intersect(o, d, [0,0,0], [X,Y,Z]):
    return []  # 射线不经过体素网格
  ↓
[3] 计算进入和退出参数
  t_enter = max(t0, 0.0)
  t_exit = min(t1, 1.0)
  if t_enter > t_exit:
    return []  # 射线不经过有效段
  ↓
[4] 初始化当前体素索引
  cur = o + d * (t_enter + eps)
  ix, iy, iz = floor(cur)
  ↓
[5] 计算步进方向和步进距离
  step_x = sign(d_x)
  tDeltaX = 1.0 / |d_x|  (if d_x != 0)
  ↓
[6] DDA遍历
  while ix, iy, iz in bounds:
    visited.append((ix, iy, iz))
    if (ix, iy, iz) == target_voxel:
      break
    # 选择最小tMax的轴步进
    if tMaxX <= tMaxY and tMaxX <= tMaxZ:
      ix += step_x
      tMaxX += tDeltaX
    elif tMaxY <= tMaxZ:
      iy += step_y
      tMaxY += tDeltaY
    else:
      iz += step_z
      tMaxZ += tDeltaZ
  ↓
输出: visited = [(ix1,iy1,iz1), (ix2,iy2,iz2), ..., (ix_target,iy_target,iz_target)]
```

**输入**：
- `origin_xyz`: `np.ndarray` shape `(3,)`, dtype `float64`
  - 射线起点（ego坐标系，单位: 米）
- `target_xyz`: `np.ndarray` shape `(3,)`, dtype `float64`
  - 射线终点（ego坐标系，单位: 米）
- `grid`: `VoxelGridSpec` - 体素网格定义

**输出**：
- `List[Tuple[int, int, int]]`
  - 遍历到的体素索引列表，包括终点体素
  - 最后一个元素是包含target的体素

**2. lidar_visibility_ray_casting()**

**功能**：Stage 2 Algorithm 2的主函数

**算法流程**：

```
输入: P_dense (N,3), L_dense (N,), lidar_origins, grid
  ↓
[1] 初始化计数器
  occ_count = zeros(X, Y, Z) uint16
  free_count = zeros(X, Y, Z) uint16
  votes = zeros(X*Y*Z, num_classes) uint16  # 语义投票表
  ↓
[2] 过滤边界外的点
  idx = points_to_voxel_indices(P_dense, grid)
  keep = in_bounds(idx, grid)
  P_dense = P_dense[keep]
  L_dense = L_dense[keep]
  idx = idx[keep]
  ↓
[3] 对每个点进行ray casting
  for i in range(N):
    origin = lidar_origins[i % len(lidar_origins)]  # 轮询分配origin
    target = P_dense[i]
    path = _dda_traverse_voxels(origin, target, grid)
    if path is empty:
      continue
    ↓
    [3.1] 终点体素标记为occupied
      ix_t, iy_t, iz_t = path[-1]
      occ_count[ix_t, iy_t, iz_t] += 1
      lin = xyz_to_linear(ix_t, iy_t, iz_t, grid)
      c = L_dense[i]  # 语义类别
      if 0 <= c < num_classes:
        votes[lin, c] += 1  # 语义投票
    ↓
    [3.2] 路径上的其他体素标记为free
      for (ix, iy, iz) in path[:-1]:
        free_count[ix, iy, iz] += 1
  ↓
[4] 生成mask_lidar
  observed = (occ_count > 0) | (free_count > 0)
  mask_lidar = observed.astype(uint8)
  ↓
[5] 生成semantics
  semantics = zeros(X, Y, Z) uint8
  ↓
  [5.1] occupied体素：多数投票决定语义
    occ_mask = occ_count > 0
    occ_lin = xyz_to_linear(*nonzero(occ_mask), grid)
    occ_votes = votes[occ_lin]  # (K, num_classes)
    sem_occ = argmax(occ_votes, axis=1)  # (K,)
    semantics[occ_mask] = sem_occ
  ↓
  [5.2] observed但非occupied → free
    free_mask = observed & (~occ_mask)
    semantics[free_mask] = FREE_LABEL (17)
  ↓
  [5.3] unobserved → free（方便起见）
    semantics[~observed] = FREE_LABEL (17)
  ↓
输出: LidarVisibilityResult(semantics, mask_lidar, occ_count, free_count)
```

**数学公式**：

**语义投票**：
对于occupied体素 $v$，其语义标签为：
$$sem(v) = \arg\max_{c \in \{0,1,...,16\}} \sum_{i: \text{点}i\text{命中}v} \mathbb{1}[L_{dense}[i] = c]$$

**可见性判断**：
$$mask_{lidar}(v) = \begin{cases} 1 & \text{if } occ_{count}(v) > 0 \text{ or } free_{count}(v) > 0 \\ 0 & \text{otherwise} \end{cases}$$

**输入**：
- `P_dense`: `np.ndarray` shape `(N, 3)`, dtype `float64`
  - 密集点云（target ego坐标系）
- `L_dense`: `np.ndarray` shape `(N,)`, dtype `uint8`
  - 语义标签，范围：0..16
- `lidar_origins`: `List[np.ndarray]` length `window_size`
  - 每个keyframe的LiDAR原点
- `grid`: `VoxelGridSpec` - 体素网格定义

**输出**：
- `LidarVisibilityResult` - 包含所有结果字段

---

### 2.5 camera_visibility.py - Stage 2: 相机可见性（Algorithm 3）

#### 2.5.1 功能与Motivation

**功能**：
- 实现论文Algorithm 3：相机像素射线可见性
- 生成`mask_camera_rays`（相机射线可见性mask）
- 使用CPU实现，通过多进程并行加速

**Motivation**：
- **多视角可见性**：LiDAR只能提供单视角可见性，相机提供多视角可见性
- **遮挡推理**：沿相机射线，第一个occupied voxel后的voxel被遮挡
- **性能优化**：通过多进程并行处理，每个worker独立处理不同的sample chunk

#### 2.5.2 核心算法：Algorithm 3

**算法原理**：

对每个相机 $c$ 的每个像素 $(u, v)$：
1. 构造像素射线：从相机原点出发，经过像素 $(u, v)$ 的方向
2. 使用3D DDA沿射线遍历体素
3. 标记所有遍历到的体素为observed
4. 遇到第一个occupied voxel后停止（遮挡终止）

**数学公式**：

**像素到相机射线的转换**：
对于像素坐标 $(u, v)$，齐次坐标为 $\tilde{p} = (u, v, 1)^T$。

相机坐标系下的射线方向：
$$\vec{d}_{cam} = K^{-1} \cdot \tilde{p} = K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

归一化：
$$\vec{d}_{cam}^{norm} = \frac{\vec{d}_{cam}}{||\vec{d}_{cam}||}$$

**转换到ego坐标系**：
$$\vec{o}_{ego} = T_{cam2ego}[:3, 3] \quad \text{(相机原点在ego坐标系)}$$
$$\vec{d}_{ego} = T_{cam2ego}[:3, :3] \cdot \vec{d}_{cam}^{norm} \quad \text{(旋转)}$$

**射线参数化**：
$$r(t) = \vec{o}_{ego} + t \cdot \vec{d}_{ego}, \quad t \geq 0$$

**遮挡终止条件**：
沿射线DDA遍历，当遇到第一个occupied voxel（即`occupied_grid[ix, iy, iz] == 1`）时停止。

#### 2.5.3 核心函数

**camera_visibility_mask_camera_rays()**

**功能**：Algorithm 3的主入口，使用CPU实现

**算法流程**：

```
输入: occupied_grid (X,Y,Z) uint8, cams, grid
  ↓
[1] 初始化输出
  out = zeros(X, Y, Z) uint8
  ↓
[2] 对每个相机
  for cam in cams:
    Kinv = inv(cam.intrinsic)
    T = cam.T_cam2ego
    R = T[:3, :3]
    t = T[:3, 3]
    ↓
    [2.1] 对每个像素
      for v in range(cam.height):
        for u in range(cam.width):
          [2.1.1] 构造像素射线
            ray_cam = Kinv @ [u, v, 1]
            ray_cam = normalize(ray_cam)
            o_ego = t
            d_ego = R @ ray_cam
          [2.1.2] 转换到体素空间
            o = (o_ego - pc_min) / voxel_size
            d = d_ego / voxel_size
          [2.1.3] 射线-包围盒相交
            if not ray_box_intersect(o, d, [0,0,0], [X,Y,Z]):
              continue
          [2.1.4] DDA遍历
            while in_bounds(ix, iy, iz):
              out[ix, iy, iz] = 1  # 标记为observed
              if occupied_grid[ix, iy, iz] == 1:
                break  # 遮挡终止
              # 步进到下一个体素
              ...
  ↓
输出: out (X,Y,Z) uint8 0/1
```

**函数签名**：
```python
def camera_visibility_mask_camera_rays(
    occupied_grid: np.ndarray,  # (X,Y,Z) uint8
    cams: List[CameraInfo],
    grid: VoxelGridSpec,
    *,
    eps: float = 1e-9,
) -> np.ndarray
```

**输入**：
- `occupied_grid`: `np.ndarray` shape `(X, Y, Z)`, dtype `uint8`
  - occupied体素mask，1=occupied，0=free/unobserved
  - 通常计算为：`occupied_grid = (mask_lidar == 1) & (semantics != FREE_LABEL)`
- `cams`: `List[CameraInfo]` - 相机信息列表（6个相机）
- `grid`: `VoxelGridSpec` - 体素网格定义

**输出**：
- `np.ndarray` shape `(X, Y, Z)`, dtype `uint8`
  - `mask_camera_rays`，1=observed（被至少一条相机射线扫到），0=unobserved

**性能特点**：
- CPU实现：顺序处理所有像素，时间复杂度 $O(N_{cams} \times H \times W \times L_{ray})$，其中$L_{ray}$是平均射线长度
- 适合多进程并行：每个worker进程独立处理，通过多进程实现并行化

---

### 2.6 image_guided_refine.py - Stage 3: 图像引导细化

#### 2.6.1 功能与Motivation

**功能**：
- 实现论文Stage 3：使用2D语义分割结果细化3D体素边界
- 沿相机像素射线，找到首个语义匹配的voxel，将之前的occupied voxels置为free

**Motivation**：
- **解决3D-2D错位问题**：传感器噪声和位姿误差导致3D体素投影到2D时出现错位
- **边界精细化**：利用2D分割的精确边界信息，修正3D体素的语义边界
- **一致性保证**：确保3D体素语义与2D像素语义沿射线方向一致

#### 2.6.2 核心数据结构

**ImageGuidedRefineOptions (dataclass)**

```python
@dataclass
class ImageGuidedRefineOptions:
    refine_roi: str = "lidar_roi"  # 'lidar_roi' | 'full_image'
    free_label: int = 17           # free体素的标签
```

**详细说明**：

| 字段 | 类型 | 取值范围 | 含义 |
|------|------|---------|------|
| `refine_roi` | `str` | `"lidar_roi"` \| `"full_image"` | 细化区域：`lidar_roi`=只在LiDAR投影区域细化，`full_image`=整幅图像 |
| `free_label` | `int` | 17 | free体素的语义标签（与`FREE_LABEL`一致） |

#### 2.6.3 核心算法

**算法原理**：

对每个相机 $c$ 的每个像素 $(u, v)$：
1. 获取2D语义标签：$c_{2d} = seg2d_cam[v, u]$
2. 如果 $c_{2d} = 0$（ignore/void），跳过
3. 构造像素射线（同Algorithm 3）
4. 沿射线DDA遍历，记录所有遍历到的voxel
5. 找到首个语义匹配的voxel：$semantics[ix, iy, iz] == c_{2d}$
6. 将该voxel之前的所有occupied voxels（在`mask_lidar`内）置为free

**数学公式**：

**细化规则**：
对于像素 $(u, v)$ 的射线 $r(t)$，遍历到的voxel序列为 $v_1, v_2, ..., v_k$。

找到首个匹配voxel $v_j$，使得：
$$semantics[v_j] = seg2d_cam[v, u] \neq free_{label}$$

则对于所有 $i < j$，如果 $mask_{lidar}[v_i] = 1$ 且 $semantics[v_i] \neq free_{label}$：
$$semantics[v_i] \leftarrow free_{label}$$

**物理意义**：
- 如果2D图像显示某个像素是"car"，但3D体素中该射线路径上第一个occupied voxel是"road"，说明3D边界过于靠前
- 通过将前面的occupied voxels置为free，使3D边界与2D边界对齐

#### 2.6.4 核心函数

**image_guided_voxel_refine_cpu()**

**功能**：Stage 3的CPU参考实现

**算法流程**：

```
输入: semantics (X,Y,Z) uint8, mask_lidar (X,Y,Z) uint8, 
      cams, seg2d_list, grid, opts
  ↓
[1] 初始化输出
  out = semantics.copy()
  ↓
[2] 对每个相机和对应的2D语义
  for cam, seg2d in zip(cams, seg2d_list):
    if seg2d is None:
      continue
    ↓
    [2.1] 验证尺寸
      assert seg2d.shape == (cam.height, cam.width)
    ↓
    [2.2] 对每个像素
      for v in range(cam.height):
        for u in range(cam.width):
          c2d = seg2d[v, u]
          if c2d == 0:  # ignore
            continue
          ↓
          [2.2.1] 构造像素射线（同Algorithm 3）
            ray_cam = Kinv @ [u, v, 1]
            ray_cam = normalize(ray_cam)
            o_ego = T[:3, 3]
            d_ego = R @ ray_cam
          ↓
          [2.2.2] DDA遍历，记录路径
            traversed = []
            while in_bounds(ix, iy, iz):
              traversed.append((ix, iy, iz))
              if out[ix, iy, iz] != free_label and 
                 out[ix, iy, iz] == c2d:
                # 找到首个匹配voxel
                for (px, py, pz) in traversed[:-1]:
                  if mask_lidar[px, py, pz] == 1 and 
                     out[px, py, pz] != free_label:
                    out[px, py, pz] = free_label
                break
              # 步进...
  ↓
输出: out (X,Y,Z) uint8 0..17
```

**输入**：
- `semantics`: `np.ndarray` shape `(X, Y, Z)`, dtype `uint8`
  - Stage 2输出的语义标签，范围：0..17
- `mask_lidar`: `np.ndarray` shape `(X, Y, Z)`, dtype `uint8`
  - LiDAR可见性mask，用于限制细化范围（只在observed区域细化）
- `cams`: `List[CameraInfo]` - 相机信息列表
- `seg2d_list`: `List[Optional[np.ndarray]]` length `len(cams)`
  - 每个相机的2D语义分割结果
  - 每个元素shape `(h, w)`, dtype `uint8`，范围：0..16（nuScenes-lidarseg类别）
  - `None`表示该相机无2D语义（跳过）
- `grid`: `VoxelGridSpec` - 体素网格定义
- `opts`: `ImageGuidedRefineOptions` - 细化选项

**输出**：
- `np.ndarray` shape `(X, Y, Z)`, dtype `uint8`
  - 细化后的语义标签，范围：0..17
  - 部分原本occupied的voxel被置为free（17）

**性能注意**：
- 当前实现是CPU串行版本，非常慢（$O(N_{cams} \times H \times W \times L_{ray})$）
- 当前使用CPU实现，通过多进程并行实现加速

---

### 2.7 seg2d_provider.py - 2D语义统一接口

#### 2.7.1 功能与Motivation

**功能**：
- 提供统一的接口获取2D语义分割结果（用于Stage 3）
- 支持三种方案：模型推理、LiDAR投影、已有标注
- 支持缓存机制，避免重复计算

**Motivation**：
- **灵活性**：不同用户可能有不同的2D语义来源（模型、投影、标注）
- **统一接口**：隐藏实现细节，Stage 3只需调用统一接口
- **性能优化**：通过缓存避免重复计算（特别是模型推理）

#### 2.7.2 核心数据结构

**Seg2DOptions (dataclass)**

```python
@dataclass
class Seg2DOptions:
    seg2d_mode: str = "none"        # 'none' | 'model' | 'lidar_project' | 'annotation'
    seg2d_cache_dir: str = ""       # 缓存目录（空字符串表示不缓存）
    seg2d_model_name: str = ""      # 模型名称（用于model模式）
    seg2d_weights: str = ""         # 模型权重路径（用于model模式）
```

**详细说明**：

| 字段 | 类型 | 取值范围 | 含义 |
|------|------|---------|------|
| `seg2d_mode` | `str` | `"none"` \| `"model"` \| `"lidar_project"` \| `"annotation"` | 2D语义获取方式：<br>`none`=跳过Stage 3<br>`model`=使用2D分割模型推理<br>`lidar_project`=LiDAR投影伪标签<br>`annotation`=使用已有标注（未实现） |
| `seg2d_cache_dir` | `str` | 任意路径字符串 | 缓存目录，如果非空，会缓存2D语义结果到`.npy`文件 |
| `seg2d_model_name` | `str` | 模型名称字符串 | 用于model模式（当前未使用，预留接口） |
| `seg2d_weights` | `str` | 权重文件路径 | 用于model模式（当前未使用，预留接口） |

#### 2.7.3 核心函数

**build_or_load_seg2d_cam()**

**功能**：获取或构建单个相机的2D语义分割结果

**算法流程**：

```
输入: reader, split, scene_name, frame_token, cam, opts, ...
  ↓
[1] 检查模式
  if opts.seg2d_mode == "none":
    return None
  ↓
[2] 检查缓存
  if opts.seg2d_cache_dir:
    cache_path = f"{cache_dir}/{split}/{scene_name}/{frame_token}/{cam_name}.npy"
    if os.path.exists(cache_path):
      return np.load(cache_path).astype(uint8)
  ↓
[3] 根据模式构建
  if opts.seg2d_mode == "model":
    seg2d = infer_seg2d_from_model(cam.img_path, model, class_mapper, ...)
  elif opts.seg2d_mode == "lidar_project":
    seg2d = project_lidarseg_to_image(
      lidar_points_ego, lidar_labels,
      K=cam.intrinsic, T_cam2ego=cam.T_cam2ego,
      img_hw=(cam.height, cam.width)
    )
  elif opts.seg2d_mode == "annotation":
    raise NotImplementedError(...)
  ↓
[4] 保存缓存（如果启用）
  if cache_path:
    os.makedirs(dirname(cache_path), exist_ok=True)
    np.save(cache_path, seg2d.astype(uint8))
  ↓
输出: seg2d (h, w) uint8 0..16 或 None
```

**输入**：
- `reader`: `NuScenesReader` - 数据读取器
- `split`: `str` - 数据集split（`"trainval"`, `"test"`, `"mini"`）
- `scene_name`: `str` - 场景名称
- `frame_token`: `str` - frame token（sample_token）
- `cam`: `CameraInfo` - 相机信息
- `opts`: `Seg2DOptions` - 选项
- `class_mapper`: 类别映射器（用于model模式，将模型输出映射到nuScenes-lidarseg类别）
- `model`: `Seg2DModelWrapper`（用于model模式）
- `lidar_points_ego`: `Optional[np.ndarray]` shape `(N, 3)`（用于lidar_project模式）
- `lidar_labels`: `Optional[np.ndarray]` shape `(N,)` dtype `uint8`（用于lidar_project模式）

**输出**：
- `Optional[np.ndarray]` shape `(h, w)`, dtype `uint8`
  - 2D语义分割结果，范围：0..16（nuScenes-lidarseg类别）
  - 0表示ignore/void
  - 如果`seg2d_mode == "none"`，返回`None`

---

### 2.8 seg2d_lidar_project.py - 方案B: LiDAR投影

#### 2.8.1 功能与Motivation

**功能**：
- 将LiDAR点云的语义标签投影到相机图像平面
- 生成2D语义伪标签（用于Stage 3方案B）

**Motivation**：
- **无需外部模型**：不依赖2D分割模型，直接使用LiDAR语义标签
- **快速实现**：实现简单，计算快速
- **局限性**：只能覆盖LiDAR点投影到的像素，覆盖率较低

#### 2.8.2 核心函数

**project_lidarseg_to_image()**

**功能**：将LiDAR语义标签投影到图像平面

**数学公式**：

**投影变换链**：
$$p^{ego} \rightarrow p^{cam} \rightarrow p^{img}$$

**步骤1：ego → cam**：
$$p^{cam} = T_{ego2cam} \cdot p^{ego} = (T_{cam2ego})^{-1} \cdot p^{ego}$$

**步骤2：cam → 图像坐标**：
$$\begin{bmatrix} u \\ v \\ w \end{bmatrix} = K \cdot p^{cam} = K \begin{bmatrix} x^{cam} \\ y^{cam} \\ z^{cam} \end{bmatrix}$$

**步骤3：齐次坐标 → 像素坐标**：
$$u_{pixel} = \frac{u}{w}, \quad v_{pixel} = \frac{v}{w}$$

**深度缓冲（Z-buffer）**：
对于每个像素 $(u, v)$，保留深度最小的点的标签：
$$seg2d[v, u] = label(\arg\min_{i: (u_i, v_i) = (u, v)} z_i^{cam})$$

**算法流程**：

```
输入: points_ego (N,3), labels (N,), K (3,3), T_cam2ego (4,4), img_hw
  ↓
[1] 初始化输出
  seg = zeros(h, w) uint8 (fill_value=void_label=0)
  depth = inf(h, w) float32
  ↓
[2] 变换到相机坐标系
  T_ego2cam = inv(T_cam2ego)
  pts_cam = transform_points(points_ego, T_ego2cam)
  ↓
[3] 过滤相机后的点
  mask = pts_cam[:, 2] > 1e-3  # z > 0 (在相机前方)
  pts_cam = pts_cam[mask]
  labels = labels[mask]
  ↓
[4] 投影到图像平面
  uvw = K @ pts_cam.T  # (3, N)
  u = (uvw[0] / uvw[2]).astype(int32)
  v = (uvw[1] / uvw[2]).astype(int32)
  z = pts_cam[:, 2].astype(float32)
  ↓
[5] 过滤图像外的点
  inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
  u, v, z, labels = u[inside], v[inside], z[inside], labels[inside]
  ↓
[6] Z-buffer：保留最近的点
  for i in range(N):
    uu, vv = int(u[i]), int(v[i])
    if z[i] < depth[vv, uu]:
      depth[vv, uu] = z[i]
      seg[vv, uu] = labels[i]
  ↓
输出: seg (h, w) uint8 0..16
```

**输入**：
- `points_ego`: `np.ndarray` shape `(N, 3)`, dtype `float64`
  - LiDAR点在ego坐标系下的坐标（单位: 米）
  - 通常使用当前keyframe的LiDAR点（不是聚合后的点，避免时间错位）
- `labels`: `np.ndarray` shape `(N,)`, dtype `uint8`
  - LiDAR语义标签，范围：0..16
- `K`: `np.ndarray` shape `(3, 3)`, dtype `float64`
  - 相机内参矩阵
- `T_cam2ego`: `np.ndarray` shape `(4, 4)`, dtype `float64`
  - 从相机到ego的变换矩阵
- `img_hw`: `Tuple[int, int]` - 图像尺寸 `(height, width)`

**输出**：
- `np.ndarray` shape `(h, w)`, dtype `uint8`
  - 2D语义伪标签，范围：0..16
  - 0表示未覆盖的像素（void）
  - 其他值表示nuScenes-lidarseg类别

**局限性**：
- **覆盖率低**：只覆盖LiDAR点投影到的像素，通常只有10-30%的像素有标签
- **稀疏性**：未覆盖的像素保持为0（void），在Stage 3中会被跳过

---

### 2.9 seg2d_model.py - 方案A: 2D模型推理（占位符）

#### 2.9.1 功能与Motivation

**功能**：
- 提供2D语义分割模型推理的接口框架
- 当前为占位符实现，需要用户集成具体的2D分割模型

**Motivation**：
- **扩展性**：预留接口，支持用户集成自己的2D分割模型（如InternImage、SegFormer等）
- **灵活性**：不强制依赖特定的模型库（mmseg/detectron2/torchvision等）

#### 2.9.2 核心数据结构

**Seg2DModelWrapper (class)**

```python
class Seg2DModelWrapper:
    def __init__(self, predict_fn: Callable[[str], np.ndarray]):
        self.predict_fn = predict_fn
    
    def predict(self, img_path: str) -> np.ndarray:
        seg = self.predict_fn(img_path)
        # 验证: shape (H, W), dtype任意
        return seg
```

**设计说明**：
- 使用函数式接口，用户可以传入任何预测函数
- 不强制模型格式，只要返回`(H, W)`的numpy数组即可
- 类别映射由`class_mapper`处理（将模型输出映射到nuScenes-lidarseg类别）

**Seg2DModelConfig (dataclass, 未使用)**

```python
@dataclass
class Seg2DModelConfig:
    model_name: str
    weights_path: str
    device: str = "cpu"  # 注意：实际代码中此参数已不再使用
```

**说明**：预留的配置类，当前未使用，未来可用于统一配置。

#### 2.9.3 核心函数

**infer_seg2d_from_model()**

**功能**：使用2D分割模型推理单张图像

**算法流程**：

```
输入: img_path, model, class_mapper, out_hw
  ↓
[1] 模型推理
  seg_raw = model.predict(img_path)
  # seg_raw: (H, W) 任意dtype，模型原始输出空间
  ↓
[2] 类别映射
  seg2d = class_mapper.to_lidarseg(seg_raw)
  # seg2d: (H, W) uint8 0..16 (nuScenes-lidarseg空间)
  ↓
[3] 尺寸验证
  if out_hw is not None:
    assert seg2d.shape == out_hw
  ↓
输出: seg2d (H, W) uint8 0..16
```

**输入**：
- `img_path`: `str` - 图像文件路径
- `model`: `Seg2DModelWrapper` - 模型包装器
- `class_mapper`: 类别映射器（需实现`to_lidarseg()`方法）
- `out_hw`: `Optional[Tuple[int, int]]` - 期望输出尺寸

**输出**：
- `np.ndarray` shape `(H, W)`, dtype `uint8`
  - 2D语义分割结果，范围：0..16

**使用示例**（用户需要实现）：

```python
# 用户代码示例
def my_predict_fn(img_path: str) -> np.ndarray:
    img = load_image(img_path)
    seg = my_model(img)  # 模型推理
    return seg.cpu().numpy()  # (H, W) 模型输出空间

model = Seg2DModelWrapper(my_predict_fn)
class_mapper = MyClassMapper()  # 实现 to_lidarseg() 方法

seg2d = infer_seg2d_from_model(
    img_path, model=model, class_mapper=class_mapper, out_hw=(900, 1600)
)
```

---

### 2.10 export_occ3d.py - 导出与合并

#### 2.10.1 功能与Motivation

**功能**：
- 保存`labels.npz`文件（semantics, mask_lidar, mask_camera）
- 管理`annotations.json`的构建和合并
- 处理图像文件的链接/复制

**Motivation**：
- **格式一致性**：确保输出格式与Occ3D官方格式完全一致
- **多进程支持**：提供annotations合并功能，支持多进程并行处理
- **存储优化**：使用symlink/hardlink减少存储占用

#### 2.10.2 核心数据结构

**ExportOptions (dataclass)**

```python
@dataclass
class ExportOptions:
    link_method: str = "symlink"  # 'symlink' | 'hardlink' | 'copy'
```

**详细说明**：

| 字段 | 类型 | 取值范围 | 含义 |
|------|------|---------|------|
| `link_method` | `str` | `"symlink"` \| `"hardlink"` \| `"copy"` | 图像文件处理方式：<br>`symlink`=符号链接（节省空间，跨设备可能失败）<br>`hardlink`=硬链接（节省空间，同设备）<br>`copy`=复制（占用空间，最可靠） |

#### 2.10.3 核心函数

**1. save_labels_npz()**

**功能**：保存`labels.npz`文件

**文件格式**：
- 使用`np.savez_compressed()`压缩保存
- 包含三个数组：`semantics`, `mask_lidar`, `mask_camera`
- 所有数组shape `(X, Y, Z) = (200, 200, 16)`，dtype `uint8`

**输入**：
- `path`: `str` - 保存路径（如`gts/scene-0001/sample_token/labels.npz`）
- `semantics`: `np.ndarray` shape `(X, Y, Z)`, dtype `uint8`
  - 语义标签，范围：0..17
- `mask_lidar`: `np.ndarray` shape `(X, Y, Z)`, dtype `uint8`
  - LiDAR可见性mask，0/1
- `mask_camera`: `np.ndarray` shape `(X, Y, Z)`, dtype `uint8`
  - 相机可见性mask，0/1
  - 通常计算为：`mask_camera = mask_lidar & mask_camera_rays`

**输出**：
- 无返回值，直接保存文件

**2. init_annotations()**

**功能**：初始化`annotations.json`字典结构

**输出**：
- `Dict[str, Any]`：
  ```python
  {
    "train_split": [],      # List[str] - train scene名称列表
    "val_split": [],        # List[str] - val scene名称列表
    "scene_infos": {}       # Dict[str, Dict] - 场景信息
  }
  ```

**3. update_annotations_for_frame()**

**功能**：更新annotations，添加一个frame的信息

**输入**：
- `annotations`: `Dict[str, Any]` - 待更新的annotations字典
- `scene_name`: `str` - 场景名称
- `frame_token`: `str` - frame token（sample_token）
- `timestamp`: `int` - 时间戳
- `cams`: `List[CameraInfo]` - 相机信息列表
- `gt_relpath`: `str` - `labels.npz`的相对路径
- `prev_token`: `Optional[str]` - 前一帧token
- `next_token`: `Optional[str]` - 后一帧token

**输出**：
- 无返回值，直接修改`annotations`字典

**annotations结构**：
```python
{
  "train_split": ["scene-0001", "scene-0002", ...],
  "val_split": ["scene-0101", ...],
  "scene_infos": {
    "scene-0001": {
      "sample_token_1": {
        "timestamp": "1234567890",
        "camera_sensor": {
          "sample_data_token_cam1": {
            "img_path": "imgs/CAM_FRONT/xxx.jpg",
            "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "extrinsic": {
              "translation": [x, y, z],
              "rotation": [w, x, y, z]  # 四元数（当前为占位符）
            },
            "ego_pose": {"translation": [...], "rotation": [...]}
          },
          ...
        },
        "ego_pose": {...},
        "gt_path": "gts/scene-0001/sample_token_1/labels.npz",
        "next": "sample_token_2",
        "prev": "sample_token_0"
      },
      ...
    },
    ...
  }
}
```

**4. merge_annotations()**

**功能**：合并多个worker返回的annotations片段（用于多进程并行）

**合并逻辑**：
- `train_split`/`val_split`：合并列表并去重
- `scene_infos`：合并字典，如果scene_name已存在，则合并frame_token字典

**输入**：
- `target`: `Dict[str, Any]` - 目标annotations（会被修改）
- `source`: `Dict[str, Any]` - 源annotations（待合并）

**输出**：
- 无返回值，直接修改`target`

**5. export_images()**

**功能**：将图像文件链接/复制到Occ3D目录结构

**算法流程**：

```
输入: out_imgs_root, cams, link_method, make_relative
  ↓
[1] 对每个相机
  for cam in cams:
    [1.1] 构建目标路径
      fname = basename(cam.img_path)
      dst = f"{out_imgs_root}/{cam.cam_name}/{fname}"
    [1.2] 链接/复制
      if link_method == "symlink":
        os.symlink(cam.img_path, dst)
      elif link_method == "hardlink":
        os.link(cam.img_path, dst)
      elif link_method == "copy":
        shutil.copy2(cam.img_path, dst)
    [1.3] 更新CameraInfo（如果make_relative=True）
      rel_path = relpath(dst, dirname(out_imgs_root))
      new_cam = CameraInfo(..., img_path=rel_path)
  ↓
输出: new_cams List[CameraInfo]
```

**输入**：
- `out_imgs_root`: `str` - 输出图像根目录（如`trainval/imgs`）
- `cams`: `List[CameraInfo]` - 相机信息列表
- `link_method`: `str` - 链接方式
- `make_relative`: `bool` - 是否将`img_path`改为相对路径

**输出**：
- `List[CameraInfo]` - 更新后的相机信息列表（`img_path`可能已改为相对路径）

---

### 2.11 camera_visibility_parallel.py - 多进程Worker

#### 2.11.1 功能与Motivation

**功能**：
- 实现多进程并行处理的worker函数
- 处理一个sample_token chunk，返回annotations片段

**Motivation**：
- **多GPU并行**：通过multiprocessing实现多GPU并行处理
- **任务分发**：将大量sample_token分块，分配给不同GPU worker
- **结果合并**：每个worker返回独立的annotations片段，主进程合并

#### 2.11.2 核心函数

**process_sample_chunk_worker()**

**功能**：处理一个sample_token chunk的worker函数

**算法流程**：

```
输入: worker_id, gpu_id, sample_tokens_chunk, args_dict
  ↓
[1] 重建对象（multiprocessing需要）
  grid = VoxelGridSpec()
  reader = NuScenesReader(args_dict["nusc_root"], ...)
  ↓
[3] 初始化本地annotations
  annotations = init_annotations()
  processed_count = 0
  errors = []
  ↓
[4] 处理每个sample_token
  for scene_name, sample_token in sample_tokens_chunk:
    try:
      [4.1] 读取相机信息并导出图像
        cams = reader.get_camera_infos(sample_token)
        cams = export_images(...)
      ↓
      [4.2] Stage 1: Voxel Densification
        P_dense, L_dense, lidar_origins = voxel_densification(...)
      ↓
      [4.3] Stage 2: LiDAR Visibility
        vis = lidar_visibility_ray_casting(...)
        semantics = vis.semantics
        mask_lidar = vis.mask_lidar
      ↓
      [4.4] Stage 2: Camera Visibility
        occupied_grid = (mask_lidar == 1) & (semantics != FREE_LABEL)
        mask_camera_rays = camera_visibility_mask_camera_rays(...)
      ↓
      [4.5] Stage 3: Image-guided Refinement (可选)
        if enable_image_guided_refine:
          seg2d_list = build_or_load_seg2d_cam(...)  # 对每个相机
          semantics = image_guided_voxel_refine_cpu(...)
          # 重新计算mask_camera_rays
          mask_camera_rays = camera_visibility_mask_camera_rays(...)
      ↓
      [4.6] 计算mask_camera
        mask_camera = mask_lidar & mask_camera_rays
      ↓
      [4.7] 保存labels.npz
        save_labels_npz(gt_path, semantics, mask_lidar, mask_camera)
      ↓
      [4.8] 更新annotations
        update_annotations_for_frame(...)
      ↓
      processed_count += 1
    except Exception as e:
      errors.append(str(e))
  ↓
输出: {
    "annotations": annotations,      # Dict - 该chunk的annotations片段
    "processed_count": int,          # 成功处理的sample_token数量
    "errors": List[str]              # 错误信息列表
  }
```

**输入**：
- `worker_id`: `int` - Worker ID（用于日志）
- `gpu_id`: `int` - GPU ID（0, 1, 2, ...）
- `sample_tokens_chunk`: `List[Tuple[str, str]]` - `[(scene_name, sample_token), ...]`
- `args_dict`: `Dict[str, Any]` - 序列化后的参数（所有命令行参数和路径）

**输出**：
- `Dict[str, Any]`：
  ```python
  {
    "annotations": Dict,      # 该chunk的annotations片段
    "processed_count": int,   # 成功处理的sample_token数量
    "errors": List[str]       # 错误信息列表
  }
  ```

**关键设计**：
- **对象重建**：在worker进程内重建`NuScenesReader`和`VoxelGridSpec`（multiprocessing无法序列化复杂对象）
- **错误处理**：捕获异常并记录，不中断整个chunk的处理
- **进度报告**：每处理2个样本或完成chunk时输出进度信息

---

### 2.12 generate_occ3d_nuscenes.py - 主入口脚本

#### 2.12.1 功能与Motivation

**功能**：
- 命令行入口，解析参数并启动处理流程
- 支持单进程和多进程两种模式
- 协调所有模块的执行

**关键导入**：
```python
```
所有必要的导入都已包含在代码中。

#### 2.12.2 核心函数

**1. parse_args()**

**功能**：解析命令行参数

**参数列表**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--nusc-root` | `str` | `/mnt/data/.../nuscenes/` | nuScenes数据根目录 |
| `--nusc-version` | `str` | `"v1.0-trainval"` | nuScenes版本 |
| `--out-root` | `str` | **必需** | 输出根目录 |
| `--split` | `str` | `"trainval"` | 数据集split：`trainval` \| `test` \| `mini` |
| `--scene-name` | `str` | `""` | 如果设置，只处理该场景 |
| `--window-size` | `int` | `21` | Stage 1聚合窗口大小 |
| `--enable-sweeps-densification` | `flag` | `False` | 启用sweeps + KNN（未实现） |
| `--enable-mesh-recon` | `flag` | `False` | 启用mesh重建（未实现） |
| `--mesh-recon-mode` | `str` | `"tsdf"` | mesh重建模式：`tsdf` \| `poisson` |
| `--num-gpus` | `int` | `1` | 并行处理的worker数量（实际不再使用GPU） |
| `--use-parallel-camera-visibility` | `flag` | `False` | 启用多进程并行 |
| `--chunk-size` | `int` | `10` | 每个chunk的sample_token数量 |
| `--enable-image-guided-refine` | `flag` | `False` | 启用Stage 3 |
| `--seg2d-mode` | `str` | `"none"` | 2D语义模式：`none` \| `model` \| `lidar_project` |
| `--seg2d-cache-dir` | `str` | `""` | 2D语义缓存目录 |
| `--link-method` | `str` | `"symlink"` | 图像链接方式：`symlink` \| `hardlink` \| `copy` |

**2. process_samples_sequential()**

**功能**：单进程顺序处理模式

**算法流程**：
- 遍历所有场景和sample_token
- 对每个sample_token执行完整pipeline（Stage 1→2→3→Export）
- 每个场景处理完成后打印进度信息：`"{len(sample_tokens)} samples of the scene-{scene_name} have been processed"`
- 最后写入`annotations.json`

**3. process_samples_parallel()**

**功能**：多进程并行处理模式

**算法流程**：
- 收集所有`(scene_name, sample_token)`对
- 分块（`chunk_size`个样本一块）
- 确定worker数量：`num_workers = min(num_gpus, len(chunks))`
- 创建进程池：使用`multiprocessing.get_context('spawn').Pool(processes=num_workers)`
  - 使用`'spawn'`启动方法，确保跨平台兼容性（特别是Windows系统）
  - 每个worker进程独立启动，不共享内存空间
- 并行执行`process_sample_chunk_worker`
- 合并所有worker返回的annotations
- 写入最终`annotations.json`
- 打印处理摘要（总处理数、错误数等）

**4. main()**

**功能**：主函数

**算法流程**：

```
[1] 解析参数
  args = parse_args()
  ↓
[2] 初始化
  grid = VoxelGridSpec()
  grid.validate()
  reader = NuScenesReader(args.nusc_root, version=args.nusc_version, verbose=True)
  ↓
[3] 选择处理模式
  if args.use_parallel_camera_visibility and args.num_gpus > 1:
    process_samples_parallel(args, reader, grid)
  else:
    process_samples_sequential(args, reader, grid)
```

**关键更新**：
- `NuScenesReader`初始化时设置`verbose=True`，输出详细日志
- 使用`multiprocessing.get_context('spawn')`创建进程池，确保跨平台兼容性（特别是Windows系统）
- Camera visibility使用CPU实现，通过多进程并行实现加速

---

## 3. 数学公式与算法

### 3.1 坐标变换公式

#### 3.1.1 多帧聚合坐标变换

**变换链**：$p^{lidar_s} \rightarrow p^{ego_s} \rightarrow p^{global} \rightarrow p^{ego_t}$

**完整公式**：
$$p^{ego_t} = (T^{ego_t}_{global})^{-1} \cdot T^{global}_{ego_s} \cdot T^{ego_s}_{lidar_s} \cdot p^{lidar_s}$$

其中：
- $T^{ego_s}_{lidar_s}$：从source LiDAR到source ego（来自`calibrated_sensor`）
- $T^{global}_{ego_s}$：从source ego到global（来自`ego_pose`）
- $T^{ego_t}_{global}$：从global到target ego（来自target `ego_pose`）

**齐次坐标形式**：
$$\begin{bmatrix} p^{ego_t} \\ 1 \end{bmatrix} = T^{ego_t}_{lidar_s} \begin{bmatrix} p^{lidar_s} \\ 1 \end{bmatrix}$$

#### 3.1.2 相机投影公式

**3D点 → 2D像素**：

**步骤1：ego → cam**：
$$p^{cam} = T_{ego2cam} \cdot p^{ego} = (T_{cam2ego})^{-1} \cdot p^{ego}$$

**步骤2：cam → 图像坐标**：
$$\begin{bmatrix} u \\ v \\ w \end{bmatrix} = K \cdot p^{cam} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x^{cam} \\ y^{cam} \\ z^{cam} \end{bmatrix}$$

**步骤3：齐次坐标 → 像素坐标**：
$$u_{pixel} = \frac{u}{w} = \frac{f_x \cdot x^{cam} + c_x \cdot z^{cam}}{z^{cam}}$$
$$v_{pixel} = \frac{v}{w} = \frac{f_y \cdot y^{cam} + c_y \cdot z^{cam}}{z^{cam}}$$

**2D像素 → 3D射线**（用于Algorithm 3）：

**步骤1：像素 → 相机射线方向**：
$$\vec{d}_{cam} = K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

**步骤2：归一化**：
$$\vec{d}_{cam}^{norm} = \frac{\vec{d}_{cam}}{||\vec{d}_{cam}||}$$

**步骤3：转换到ego坐标系**：
$$\vec{o}_{ego} = T_{cam2ego}[:3, 3]$$
$$\vec{d}_{ego} = T_{cam2ego}[:3, :3] \cdot \vec{d}_{cam}^{norm}$$

### 3.2 体素化公式

#### 3.2.1 点云到体素索引

对于点 $p = (x, y, z)$，体素索引 $(i_x, i_y, i_z)$：
$$i_x = \lfloor \frac{x - x_{min}}{v_x} \rfloor$$
$$i_y = \lfloor \frac{y - y_{min}}{v_y} \rfloor$$
$$i_z = \lfloor \frac{z - z_{min}}{v_z} \rfloor$$

#### 3.2.2 体素索引线性化

**Z-major顺序**：
$$idx = i_x + X \cdot (i_y + Y \cdot i_z)$$

**逆变换**：
$$i_z = \lfloor \frac{idx}{X \times Y} \rfloor$$
$$rem = idx - i_z \times X \times Y$$
$$i_y = \lfloor \frac{rem}{X} \rfloor$$
$$i_x = rem - i_y \times X$$

#### 3.2.3 体素中心坐标

$$x = x_{min} + (i_x + 0.5) \cdot v_x$$
$$y = y_{min} + (i_y + 0.5) \cdot v_y$$
$$z = z_{min} + (i_z + 0.5) \cdot v_z$$

### 3.3 3D DDA算法

#### 3.3.1 射线参数化

给定起点 $o = (o_x, o_y, o_z)$ 和终点 $p = (p_x, p_y, p_z)$：
$$d = p - o = (d_x, d_y, d_z)$$
$$r(t) = o + t \cdot d, \quad t \in [0, 1]$$

#### 3.3.2 体素空间坐标

$$o_{vox} = \frac{o - pc_{min}}{voxel_{size}} = (o_x', o_y', o_z')$$
$$d_{vox} = \frac{d}{voxel_{size}} = (d_x', d_y', d_z')$$

#### 3.3.3 步进计算

**步进距离**：
$$\Delta t_x = \frac{1}{|d_x'|}, \quad \Delta t_y = \frac{1}{|d_y'|}, \quad \Delta t_z = \frac{1}{|d_z'|}$$

**到下一个体素边界的距离**：
$$tMax_x = \frac{(i_x + 1) - (o_x' + t \cdot d_x')}{d_x'} \quad \text{(如果} d_x' > 0\text{)}$$
$$tMax_y = \frac{(i_y + 1) - (o_y' + t \cdot d_y')}{d_y'}$$
$$tMax_z = \frac{(i_z + 1) - (o_z' + t \cdot d_z')}{d_z'}$$

**步进规则**：选择$tMax$最小的轴进行步进。

### 3.4 语义投票公式

对于occupied体素 $v$，其语义标签为：
$$sem(v) = \arg\max_{c \in \{0,1,...,16\}} \sum_{i: \text{点}i\text{命中}v} \mathbb{1}[L_{dense}[i] = c]$$

其中 $\mathbb{1}[\cdot]$ 是指示函数。

---

## 4. 输入输出数据结构

### 4.1 labels.npz文件格式

**文件路径**：`gts/{scene_name}/{sample_token}/labels.npz`

**文件内容**（使用`np.savez_compressed()`保存）：

| 键名 | 形状 | 类型 | 取值范围 | 含义 |
|------|------|------|---------|------|
| `semantics` | `(200, 200, 16)` | `uint8` | 0..17 | 体素语义标签：0..16=nuScenes类别，17=free |
| `mask_lidar` | `(200, 200, 16)` | `uint8` | 0/1 | LiDAR可见性mask：1=observed，0=unobserved |
| `mask_camera` | `(200, 200, 16)` | `uint8` | 0/1 | 相机可见性mask：1=observed，0=unobserved |

**读取示例**：
```python
data = np.load("labels.npz")
semantics = data["semantics"]      # (200, 200, 16) uint8
mask_lidar = data["mask_lidar"]    # (200, 200, 16) uint8
mask_camera = data["mask_camera"]  # (200, 200, 16) uint8
```

### 4.2 annotations.json文件格式

**文件路径**：`{split}/annotations.json`

**JSON结构**：

```json
{
  "train_split": ["scene-0001", "scene-0002", ...],
  "val_split": ["scene-0101", ...],
  "scene_infos": {
    "scene-0001": {
      "sample_token_1": {
        "timestamp": "1234567890",
        "camera_sensor": {
          "sample_data_token_cam1": {
            "img_path": "imgs/CAM_FRONT/xxx.jpg",
            "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
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
        "ego_pose": {...},
        "gt_path": "gts/scene-0001/sample_token_1/labels.npz",
        "next": "sample_token_2",
        "prev": "sample_token_0"
      },
      ...
    },
    ...
  }
}
```

**字段说明**：

| 字段路径 | 类型 | 说明 |
|---------|------|------|
| `train_split` | `List[str]` | 训练集场景名称列表 |
| `val_split` | `List[str]` | 验证集场景名称列表 |
| `scene_infos[scene_name][frame_token].timestamp` | `str` | 时间戳（字符串格式） |
| `scene_infos[scene_name][frame_token].camera_sensor[sample_data_token].img_path` | `str` | 图像相对路径 |
| `scene_infos[scene_name][frame_token].camera_sensor[sample_data_token].intrinsic` | `List[List[float]]` | 相机内参矩阵（3×3） |
| `scene_infos[scene_name][frame_token].camera_sensor[sample_data_token].extrinsic.translation` | `List[float]` | 相机外参平移向量 `[x, y, z]` |
| `scene_infos[scene_name][frame_token].camera_sensor[sample_data_token].extrinsic.rotation` | `List[float]` | 相机外参旋转四元数 `[w, x, y, z]`（当前为占位符`[1,0,0,0]`） |
| `scene_infos[scene_name][frame_token].camera_sensor[sample_data_token].ego_pose` | `Dict` | ego位姿 `{"translation": [...], "rotation": [...]}` |
| `scene_infos[scene_name][frame_token].ego_pose` | `Dict` | 同camera_sensor中的ego_pose（冗余，保持兼容性） |
| `scene_infos[scene_name][frame_token].gt_path` | `str` | `labels.npz`的相对路径 |
| `scene_infos[scene_name][frame_token].next` | `Optional[str]` | 下一帧token（如果存在） |
| `scene_infos[scene_name][frame_token].prev` | `Optional[str]` | 前一帧token（如果存在） |

---

## 5. 模块间依赖关系

### 5.1 依赖关系图

```
generate_occ3d_nuscenes.py (主入口)
  ├── voxel_grid.py (体素网格定义)
  │   └── 无依赖（基础工具）
  │
  ├── nusc_io.py (数据读取)
  │   ├── 依赖: nuscenes-devkit, pyquaternion
  │   └── 被依赖: accumulate.py, seg2d_lidar_project.py, export_occ3d.py
  │
  ├── accumulate.py (Stage 1)
  │   ├── 依赖: nusc_io.py, voxel_grid.py
  │   └── 被依赖: generate_occ3d_nuscenes.py, camera_visibility_parallel.py
  │
  ├── lidar_visibility.py (Stage 2 Algorithm 2)
  │   ├── 依赖: voxel_grid.py
  │   └── 被依赖: generate_occ3d_nuscenes.py, camera_visibility_parallel.py
  │
  ├── camera_visibility.py (Stage 2 Algorithm 3)
  │   ├── 依赖: nusc_io.py, voxel_grid.py
  │   ├── 可选依赖: torch (仅用于其他模块，camera_visibility不使用)
  │   └── 被依赖: generate_occ3d_nuscenes.py, camera_visibility_parallel.py
  │
  ├── seg2d_provider.py (2D语义统一接口)
  │   ├── 依赖: nusc_io.py, seg2d_lidar_project.py, seg2d_model.py
  │   └── 被依赖: generate_occ3d_nuscenes.py, camera_visibility_parallel.py
  │
  ├── seg2d_lidar_project.py (方案B)
  │   └── 被依赖: seg2d_provider.py
  │
  ├── seg2d_model.py (方案A占位符)
  │   └── 被依赖: seg2d_provider.py
  │
  ├── image_guided_refine.py (Stage 3)
  │   ├── 依赖: nusc_io.py, voxel_grid.py
  │   └── 被依赖: generate_occ3d_nuscenes.py, camera_visibility_parallel.py
  │
  ├── export_occ3d.py (导出)
  │   ├── 依赖: nusc_io.py
  │   └── 被依赖: generate_occ3d_nuscenes.py, camera_visibility_parallel.py
  │
  └── camera_visibility_parallel.py (多进程Worker)
      ├── 依赖: 所有Stage模块 (accumulate, lidar_visibility, camera_visibility, 
      │                        image_guided_refine, seg2d_provider, export_occ3d)
      └── 被依赖: generate_occ3d_nuscenes.py
```

### 5.2 数据流依赖

**Stage 1 → Stage 2 → Stage 3 → Export**：

```
Stage 1 (accumulate.py)
  ↓ P_dense (N,3), L_dense (N,), lidar_origins
Stage 2 Algorithm 2 (lidar_visibility.py)
  ↓ semantics (X,Y,Z), mask_lidar (X,Y,Z)
Stage 2 Algorithm 3 (camera_visibility.py)
  ↓ mask_camera_rays (X,Y,Z)
Stage 3 (image_guided_refine.py, 可选)
  ↓ refined semantics (X,Y,Z)
Export (export_occ3d.py)
  ↓ labels.npz, annotations.json
```

**Stage 3的额外依赖**：
- 需要`seg2d_provider.py`提供2D语义
- `seg2d_provider.py`可能依赖`seg2d_lidar_project.py`或`seg2d_model.py`

### 5.3 关键接口

**1. VoxelGridSpec**：
- 所有体素相关操作的基础
- 被`lidar_visibility.py`, `camera_visibility.py`, `image_guided_refine.py`使用

**2. NuScenesReader**：
- 所有数据读取的基础
- 被`accumulate.py`, `seg2d_provider.py`, `export_occ3d.py`使用

**3. CameraInfo / LidarInfo**：
- 传感器信息的统一表示
- 在`nusc_io.py`中定义，被多个模块使用

---

## 6. 并行处理机制

### 6.1 多进程架构

**设计模式**：Master-Worker模式

**启动方法**：使用`'spawn'`方法
- **原因**：确保跨平台兼容性，特别是Windows系统
- **特点**：
  - Windows默认不支持`'fork'`方法（仅Unix系统支持）
  - `'spawn'`方法会启动全新的Python解释器进程
  - 每个worker进程独立，不共享内存空间
  - 所有对象都需要序列化传递（通过`args_dict`）

```
主进程 (generate_occ3d_nuscenes.py)
  │
  ├── [1] 收集所有任务
  │     all_samples = [(scene_name, sample_token), ...]
  │
  ├── [2] 分块
  │     chunks = [chunk_0, chunk_1, ..., chunk_N]
  │
  ├── [3] 创建进程池
  │     pool = multiprocessing.get_context('spawn').Pool(num_workers)
  │     # 使用'spawn'方法确保跨平台兼容性（特别是Windows）
  │
  ├── [4] 分发任务
  │     results = pool.starmap(process_sample_chunk_worker, worker_args)
  │
  └── [5] 合并结果
        merge_annotations(target, source) for each result
```

### 6.2 数据序列化

**可序列化参数**（通过`args_dict`传递）：
- 字符串：路径、版本号等
- 基本类型：int, float, bool
- 列表/字典：包含基本类型

**不可序列化对象**（在worker内重建）：
- `NuScenesReader`：在worker内重新创建
- `VoxelGridSpec`：在worker内重新创建
- `Seg2DModelWrapper`：如果使用model模式，需要在worker内加载

**注意**：由于使用`'spawn'`启动方法，所有worker进程都是全新启动的Python解释器，因此所有复杂对象都必须在worker内重建。这与`'fork'`方法不同，`'fork'`会复制父进程的内存空间。

### 6.4 结果合并

**annotations合并**：
- `train_split`/`val_split`：列表合并并去重
- `scene_infos`：字典合并，同一scene的frame_token合并

**错误处理**：
- 每个worker独立捕获异常
- 错误信息收集到`errors`列表
- 主进程汇总所有错误并输出

### 6.5 性能考虑

**并行度**：
- Worker数量 = `min(num_gpus, len(chunks))`
- 虽然参数名为`num_gpus`，但实际不再使用GPU，仅用于控制worker数量
- 每个worker使用CPU处理，通过多进程并行实现加速

**多进程启动方法**：
- 使用`multiprocessing.get_context('spawn')`创建进程池
- `spawn`方法确保每个worker进程是全新的Python进程，避免fork带来的问题
- 每个worker进程独立运行，不会相互干扰

**负载均衡**：
- 当前实现：简单分块，不保证负载均衡
- 改进方向：可以根据每个sample_token的处理时间动态分配

**内存管理**：
- 每个worker独立内存空间
- 避免共享大对象，减少序列化开销
- 及时释放中间结果

---

## 7. 总结

### 7.1 核心设计理念

1. **模块化**：每个Stage独立实现，便于测试和扩展
2. **统一接口**：使用dataclass定义选项，统一参数传递
3. **向后兼容**：支持单进程模式，便于调试
4. **性能优化**：多进程并行处理，每个worker独立处理不同的sample chunk

### 7.2 关键算法

1. **3D DDA**：高效遍历射线穿过的体素
2. **语义投票**：多数投票决定体素语义
3. **遮挡推理**：通过ray casting区分free/unobserved
4. **3D-2D一致性**：使用2D语义细化3D边界

### 7.3 扩展方向

1. **Stage 1扩展**：
   - Sweeps + KNN标签分配
   - Mesh重建（TSDF/Poisson）

2. **Stage 3优化**：
   - 更精细的边界细化策略
   - 性能优化

3. **性能优化**：
   - 动态负载均衡
   - 异步I/O
   - 内存池管理

4. **功能扩展**：
   - 支持Waymo数据集
   - 支持其他体素分辨率
   - 支持实例分割标签

---

**文档结束**