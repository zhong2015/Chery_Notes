# Drive-JEPA 论文详细解读

**论文标题**: Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving  
**作者**: Linhan Wang, Zichong Yang, Chen Bai, Guoxiang Zhang, Xiaotong Liu, Xiaoyin Zheng, Xiao-Xiao Long, Chang-Tien Lu, Cheng Lu  
**arXiv ID**: 2601.22032  
**发表日期**: 2026年1月  

---

## 一、Abstract（摘要）解读

### 1.1 原文与结构

摘要指出：端到端自动驾驶越来越多地利用**自监督视频预训练**来学习可迁移的规划表示；然而，为场景理解预训练视频世界模型迄今只带来**有限改进**。这一局限与驾驶的**内在歧义性**叠加：每个场景通常只提供**一条人类轨迹**，难以学习多模态行为。

本文提出 **Drive-JEPA**：将 **Video Joint-Embedding Predictive Architecture (V-JEPA)** 与**多模态轨迹蒸馏**结合，用于端到端驾驶。

- **第一点**：将 V-JEPA 适配到端到端驾驶，在**大规模驾驶视频**上预训练 ViT 编码器，产生与**轨迹规划对齐**的预测表示。
- **第二点**：引入**以提案为中心的规划器**，将**模拟器生成的多条轨迹**与人类轨迹一起蒸馏；并设计**动量感知选择机制**，促进稳定、安全的行为。

在 **NAVSIM** 上评估：
- **感知无关设置**：V-JEPA 表示 + 简单基于 Transformer 的解码器，比先前方法高 **3 PDMS**。
- **完整框架**：在 v1 上达到 **93.3 PDMS**，在 v2 上达到 **87.8 EPDMS**，创下新的 **SOTA**。

### 1.2 关键概念释义

| 概念 | 含义 |
|------|------|
| 端到端自动驾驶 | 从原始传感器观测直接映射到驾驶动作的统一神经模型 |
| 自监督视频预训练 | 不依赖人工标注，从视频中学习表示 |
| 监督瓶颈 | 每场景仅一条人类轨迹，无法覆盖多模态合理行为 |
| V-JEPA | 联合嵌入预测架构，预测未来潜在表示而非像素，有效防止模式崩溃 |
| 多模态轨迹蒸馏 | 用人类轨迹 + 模拟器生成的多条轨迹同时监督，增加行为多样性 |
| 动量感知选择 | 考虑历史轨迹连续性，减少帧间轨迹抖动，提升舒适性 |
| PDMS / EPDMS | 预测性驾驶员模型分数 / 扩展版，NAVSIM 的规划质量综合指标 |

### 1.3 摘要中的数字含义

- **3 PDMS**：在感知无关设置下，相对先前最佳方法的 PDMS 提升量。
- **93.3 PDMS**：在 NAVSIM v1 上的综合得分（满分倾向 100 附近）。
- **87.8 EPDMS**：在 NAVSIM v2 上的扩展综合得分，考虑更多规则与舒适性维度。

---

## 二、Introduction（引言）解读

### 2.1 端到端自动驾驶与动机

- **端到端自动驾驶**（Pomerleau 1988; Chitta et al. 2022; Hu et al. 2023b）：直接从原始观测映射到驾驶动作，去掉传统模块化流水线中的手工中间表示，减少信息损失、提高可扩展性。
- **近期趋势**：越来越多地借助**自监督视频预训练**学习可迁移的规划表示；但为场景理解预训练视频世界模型**改进有限**。

### 2.2 现有视频预训练方法的两类局限

**（1）视频生成类方法**（如 VaVAM、Epona）  
- 通过重建或生成视频学习表示，再迁移到规划。  
- **问题**：像素级目标计算重，且可能过度强调与决策无关的视觉细节。

**（2）潜在世界模型**（如 LAW、World4Drive）  
- 预测紧凑的特征动力学（如从特征 T 预测 T+1），降低计算成本。  
- **问题**：多作为辅助目标，未明显体现**大规模预训练**的收益，且可能存在**表示坍塌**。

### 2.3 监督瓶颈：单轨迹与多模态需求

- **现象**：每个场景通常只有**一条人类轨迹**，但驾驶本质是**多模态**的（同一场景有多种合理轨迹）。
- **离散方法**（VAD v2、Hydra-MDP）：将轨迹聚类为固定词表并预测安全/舒适分数；**局限**在于词表的覆盖与质量，在词表外场景泛化差。
- **扩散方法**（DiffusionDrive、GoalFlow）：通过迭代采样建模多模态轨迹分布；**局限**在于仍受「每场景单条人类轨迹」监督，学到的行为多样性受限。

### 2.4 Drive-JEPA 的应对思路

- **第一**：将 **V-JEPA** 适配到驾驶域，从大规模原始视频中学习**与规划对齐的预测表示**，提升迁移效果。
- **第二**：提出**多模态轨迹蒸馏**，将**模拟器知识**蒸馏到**以提案为中心的规划器**，提供超越单条人类轨迹的多样监督，支持更安全的多模态决策。

### 2.5 框架三组件概览

1. **Driving Video Pretraining**：在大规模驾驶视频上用 V-JEPA 预训练 ViT 编码器，学习预测未来潜在表示，并有效防止模式崩溃。  
2. **Multimodal Trajectory Distillation**：Waypoint 锚定的提案生成（可变形注意力在轨迹 waypoint 处聚合 BEV 特征、迭代细化）+ 人类轨迹与模拟器生成的多模态轨迹联合监督，实现从模拟器到规划器的知识蒸馏。  
3. **Momentum-aware Trajectory Selection**：对候选轨迹打分（碰撞风险、交通规则遵守、舒适度），并加入**动量感知惩罚**以降低帧间轨迹失真。

### 2.6 实验与贡献总结

- **数据集**：NAVSIM v1、NAVSIM v2、Bench2Drive。  
- **结果**：93.3 PDMS（v1）、87.8 EPDMS（v2）SOTA；仅前视相机 + 轻量 Transformer 规划器在感知无关设置下优于先前工作 3 PDMS；Bench2Drive 上多模态轨迹蒸馏持续提升驾驶质量。

**贡献归纳**：  
- 将 V-JEPA 预训练引入端到端自动驾驶，在感知相关与感知无关设置下均提升性能。  
- 提出多模态轨迹监督，将模拟器知识蒸馏到以提案为中心的框架，生成多样多模态轨迹。  
- 设计动量感知轨迹选择模块，提升驾驶舒适性。  
- 在 NAVSIM v1/v2 上达到新 SOTA，且在无感知标注的 NAVSIM 上仍有强表现。

---

## 三、第 3 节 Method（方法）详细解读

### 3.1 Preliminary（预备知识）

#### 3.1.1 端到端自动驾驶任务形式化

- **输入**：  
  - $ I_t = \{I^1_t, I^2_t, \ldots, I^N_t\} $：时刻 $ t $ 的 **N** 个多视角环视图像。  
  - 自车状态：驾驶指令（左/直/右）、速度、加速度等。
- **输出**：  
  未来轨迹以 **waypoint 序列** 表示：  
  $$
  W_t = \{w^1_t, w^2_t, \ldots, w^M_t\}
  $$
  其中每个 waypoint  
  $$
  w^i_t = (x^i_t, y^i_t, \psi^i_t)
  $$
  表示在 **BEV** 下、对应未来时刻 $ t+i $ 的预测位置与航向角。

**参数与符号**：

| 符号 | 含义 | 典型/说明 |
|------|------|-----------|
| $ N $ | 相机/视角数量 | 文中完整框架可仅用前视，即 1 |
| $ M $ | 未来 waypoint 数量 | 预测的时间步数 |
| $ x^i_t, y^i_t $ | 第 $ i $ 个 waypoint 的 BEV 坐标 | 米 |
| $ \psi^i_t $ | 第 $ i $ 个 waypoint 的航向角 | 弧度或度 |

#### 3.1.2 V-JEPA 详细解析

**（1）Motivation：为什么用 V-JEPA？**

- **预测潜在表示而非像素**：视频生成/重建类方法要预测或重建像素，计算重且容易学到与规划无关的纹理、光照等细节；V-JEPA 只预测**高维潜在表示**，目标更紧凑、与高层语义/动力学更相关，便于迁移到规划。
- **防止表示坍塌（mode collapse）**：若只做“上下文编码器预测目标编码器”且两端都梯度更新，容易退化为常数输出（所有输入映射到同一表示）。V-JEPA 通过 **stop-gradient + EMA 目标编码器** 让目标端稳定、不随预测端一起塌缩，从而学到有区分度的预测表示。
- **与规划对齐**：在驾驶视频上预训练后，被预测的“未来表示”与场景演化、自车运动强相关，因此迁移到轨迹预测时，仅用 ViT 特征 + 简单解码器即可取得良好效果（如免感知 89.0 PDMS）。

**（2）公式与数学符号逐项说明**

目标（仅在被掩码位置计算损失）：
$$
\min_{\theta,\phi,\Delta y} \ \bigl\| P_\phi(\Delta y, E_\theta(x)) - \mathrm{sg}(E_{\bar{\theta}}(y)) \bigr\|_1
$$

| 符号 | 含义 | 作用 |
|------|------|------|
| $ x $ | 被掩码后的视图 | 输入视频中随机丢弃一部分时空 patch 后的结果，编码器只看到部分内容。 |
| $ y $ | 完整目标视图 | 同一段视频的完整帧，用于生成“要预测”的目标表示。 |
| $ E_\theta(\cdot) $ | 在线编码器（可训练） | 对 $ x $ 编码得到上下文特征；参数 $ \theta $ 随梯度更新。 |
| $ E_{\bar{\theta}}(\cdot) $ | 目标编码器（EMA） | 对 $ y $ 编码得到目标表示；参数 $ \bar{\theta }$ 为 $ \theta $ 的指数滑动平均，**不反传梯度**，提供稳定目标。 |
| $ P_\phi(\cdot) $ | 预测器（可训练） | 输入：掩码位置信息 $ \Delta y $ + $ E_\theta(x) $；输出：对**被掩码位置**的预测表示。 |
| $ \Delta y $ | 可学习 mask token | 标记哪些时空位置被掩码，预测器据此知道要预测哪些位置。 |
| $ \mathrm{sg}(\cdot) $ | Stop-gradient | 括号内不参与求导；即目标 $ E_{\bar{\theta}}(y) $ 只提供监督信号，不更新 $ E_{\bar{\theta}} $（$ \bar{\theta }$ 仅通过 EMA 从 $ \theta $ 更新）。 |
| $ \|\cdot\|_1 $ | L1 范数 | 在预测表示与目标表示之间算 L1 距离，仅在被掩码位置计算，其余位置不参与损失。 |

**（3）训练流程与作用效果**

- **前向**：对视频片段随机掩码得 $ x $，完整视图为 $ y $；用 $ E_\theta(x) $ 与 $ \Delta y $ 输入 $ P_\phi $ 得到被掩码处的预测；用 $ E_{\bar{\theta}}(y) $ 在相同位置取出目标表示。
- **损失**：预测与目标在掩码位置上的 L1 误差；只对 $ \theta,\phi,\Delta y $ 反传，$ E_{\bar{\theta}} $ 不反传。
- **EMA 更新**：通常 $ \bar{\theta } \leftarrow \tau \bar{\theta } + (1-\tau)\theta $（$ \tau $ 接近 1），使目标缓慢跟踪在线编码器，既稳定又避免坍塌。
- **效果**：编码器学到“根据过去与当前可见内容预测未来/缺失位置的表示”，且该表示具有多样性（不坍缩），适合作为下游规划器的输入特征。

---

### 3.2 Driving Video Pretraining（驾驶视频预训练）

#### 3.2.1 数据与规模

- **数据来源**：CoVLA、DrivingDojo、OpenScene。  
- **设定**：前视相机、**8 帧**片段、分辨率 **512×256**、**2 Hz**。  
- **规模**：预训练约 **208 小时**（见表 1），大于 LAW/World4Drive（约 20h），计算上因潜在预测任务高效且防坍塌而可行。

#### 3.2.2 Table 1 解读（免感知规划器对比）

| 列 | 含义 |
|----|------|
| Encoder size | 编码器参数量：LAW/World4Drive 21M，Epona 1.1B，Ours 307M（ViT/L） |
| Data scale | 预训练数据时长：Ours 208h |
| PDMS | 免感知设置下的 PDMS：Ours 89.0，优于 83.8/85.1/86.1 |

**数字含义**：  
- **208h**：预训练视频时长，用于与规划对齐的表示学习。  
- **512×256**：单帧分辨率。  
- **8-frame, 2 Hz**：每段 8 帧、2 Hz 采样，对应 3.5 秒左右片段。

#### 3.2.3 免感知端到端驾驶流程与公式（含动机与符号解析）

给定前视 $ I^1_t, I^1_{t-1} $（当前帧 + 前一帧），目标是预测未来 $ M $ 个 waypoint 的序列。

**（1）特征提取**
$$
F_t \in \mathbb{R}^{N_f \times D}
$$

- **Motivation**：用 V-JEPA 预训练好的 ViT 编码器把两帧图像编码成**时空特征**，不依赖感知标注（无检测/分割标签），即“免感知”。
- **符号**：$ N_f $ 为 ViT 输出 token 数（由 patch 划分与时间维度决定），$ D $ 为特征维度（如 1024）。$ F_t $ 的每一行是一个 token 的 $ D $ 维向量，整体承载当前与近期场景的表示。

**（2）可学习 query**

$ Q \in \mathbb{R}^{M \times D} $：$ M $ 个可学习向量，每个对应一个**未来时刻**的 waypoint。

- **Motivation**：借鉴 DETR 式设计——用固定数量的 query 向编码特征“发问”，通过 cross-attention 从 $ F_t $ 中聚合与各未来时刻相关的信息；$ M $ 个 query 自然对应 $ M $ 个 waypoint，结构清晰且易扩展。

**（3）Transformer 解码**

$$
H = \mathrm{TransformerDecoder}(Q, F_t)
$$

- **含义**：标准 Transformer 解码器；$ Q $ 为 decoder 的 query，$ F_t $ 为 encoder 的 key/value。每个 query 通过 self-attention 与其它 query 交互、通过 cross-attention 从 $ F_t $ 中取信息，输出 $ H \in \mathbb{R}^{M \times D} $。
- **Motivation**：解码器负责把“未来各时刻”的语义从视觉特征里解析出来，为下一步回归 waypoint 提供 per-waypoint 的表示。
- **符号**：$ H $ 的第 $ i $ 行 $ H_i $ 对应第 $ i $ 个 waypoint 的隐藏表示。

**（4）Waypoint 预测**

$$
\hat{W}_t = \mathrm{MLP}(H), \quad \hat{W}_t = \{\hat{w}^1_t, \ldots, \hat{w}^M_t\}, \quad \hat{w}^i_t = (\hat{x}^i_t, \hat{y}^i_t, \hat{\psi}^i_t)
$$

- **含义**：对 $ H $ 的每一行做同一 MLP（或共享的逐点 MLP），输出 3 维：$ \hat{x}^i_t, \hat{y}^i_t $（BEV 坐标），$ \hat{\psi}^i_t $（航向）。整体得到 $ \hat{W}_t $。
- **Motivation**：解码器输出是高层表示，用轻量 MLP 映射到规划所需的几何量（位置+航向），便于与真值轨迹做 MSE 监督。
- **符号小结**：$ \hat{w}^i_t $ 为第 $ i $ 个预测 waypoint，对应未来某固定时间步（如 $ t+i $ 的离散时间）。

**（5）训练损失**

$$
\mathcal{L}_{\mathrm{traj}}^{\mathrm{simple}} = \|\hat{W}_t - W_t\|^2_{\mathrm{MSE}}
$$

- **含义**：预测轨迹 $ \hat{W}_t $ 与真实人类轨迹 $ W_t $ 在 waypoint 序列上的均方误差（对 $ M $ 个点的 $ (x,y,\psi) $ 逐维求平方差再平均或求和，具体以实现为准）。
- **Motivation**：在免感知设定下，唯一监督就是人类轨迹；MSE 直接约束几何一致，且对 ViT 特征+解码器端到端可导，使预训练表示朝“有利于轨迹预测”的方向微调。
- **作用效果**：整体流程简单（编码器 + 解码器 + MLP），却能在 NAVSIM 上显著优于其他感知无关方法（见表 1），说明 V-JEPA 预训练学到的表示与规划任务高度对齐。

**参数与符号汇总**

| 符号 | 含义 | 典型/说明 |
|------|------|-----------|
| $ F_t $ | ViT 输出的时空特征矩阵 | $ N_f \times D $ |
| $ N_f $ | 特征 token 数 | 由 ViT patch 与时间决定 |
| $ D $ | 特征维度 | 如 1024 |
| $ Q $ | 可学习 waypoint query | $ M \times D $，$ M $ 为 waypoint 数 |
| $ H $ | 解码器输出 | $ M \times D $，每行对应一个 waypoint |
| $ \hat{W}_t, W_t $ | 预测轨迹与真实轨迹 | 各 $ M $ 个 $ (x,y,\psi) $ |

---

### 3.3 Waypoint-anchored Proposals Generation（Waypoint 锚定提案生成）

#### 3.3.1 输入与初始化

- **输入**：视觉特征 $ F_t \in \mathbb{R}^{N_f \times D} $，时刻 $ t $ 的自车状态。  
- **自车特征**：$ e_t \in \mathbb{R}^{1 \times D} $，由线性层从自车状态得到。  
- **提案 query 初始化**：  
  $$
  Q^0 \in \mathbb{R}^{N_p \times M \times D}
  $$
  由 $ e_t $ 与**可学习位置嵌入**相加得到。  
  - $ N_p $：提案（轨迹）数量。  
  - $ M $：每条轨迹的 waypoint 数。

#### 3.3.2 迭代细化与 WADA

对 $ \ell = 0, 1, \ldots, L-1 $：

1. **从 query 解码出 waypoint 轨迹**：  
   $$
   \tilde{W}_\ell = \{\tilde{W}^{(n)}_\ell\}_{n=1}^{N_p}, \quad \tilde{W}_\ell \in \mathbb{R}^{N_p \times M \times 3}
   $$
   每个 waypoint 为 $ (x, y, \psi) $；即当前迭代下每条提案的 $ M $ 个 BEV 位置与航向。

2. **Waypoint 锚定的可变形注意力（WADA）**：见下文详细解析。

3. **更新 query**：  
   $$
   Q^{\ell+1} = \mathrm{MLP}\bigl(\mathrm{WADA}(Q^\ell, \tilde{W}_\ell, F_t)\bigr)
   $$

**参数**：$ N_p $ 提案数（文中 32），$ M $ 每条轨迹 waypoint 数，$ L $ 迭代次数。

---

**可变形注意力（Deformable Attention）原理简述**

标准 Transformer 的 cross-attention 对**所有** key 位置计算注意力，复杂度为序列长度的平方。**可变形注意力**（Deformable DETR / ViT with Deformable Attention，Xia et al.）只在一小部分**采样点**上计算，且采样位置由网络预测的**偏移**决定，从而：

- **Reference（参考点）**：每个 query 对应一个或多个参考点（如 2D 坐标）。
- **Offset（偏移）**：由 query 经过线性层预测若干偏移量，得到 $ K $ 个采样点 = 参考点 + 偏移。
- **采样**：在特征图上对这几个采样点做双线性插值取特征，再对 $ K $ 个特征做加权求和（权重由 query 与采样特征计算），得到该 query 的更新特征。
- **效果**：计算量与 $ K $ 成线性关系，且网络能“主动”把采样点移到最相关的区域，适合高分辨率特征或稀疏关注（如只关心轨迹经过的局部）。

**Waypoint-anchored Deformable Attention（WADA）在 Drive-JEPA 中的含义**

- **“Waypoint-anchored”**：参考点不再任意，而是**当前预测的 waypoint 在 BEV 上的位置**。即对第 $ n $ 条提案的第 $ m $ 个 waypoint，参考点就是 $ \tilde{W}^{(n)}_{\ell,m} $ 的 $ (x,y) $（或再加 $ \psi $ 的编码）。
- **输入**：$ Q^\ell $（当前提案 query）、$ \tilde{W}_\ell $（当前 waypoint 坐标）、$ F_t $（视觉特征）。$ F_t $ 可能对应 2D 时空 feature map，或已通过某种方式与 BEV 对齐。
- **过程**：对每个 (提案, waypoint) 的 query，以其 waypoint 的 BEV 坐标为参考点，用可变形注意力在 $ F_t $（或由 $ F_t $ 得到的 BEV 特征图）上预测偏移并采样、聚合特征；同时在提案维度做 self-attention，使不同提案之间交换信息。输出与 $ Q^\ell $ 同形的特征，再经 MLP 得到 $ Q^{\ell+1} $。
- **Motivation**：轨迹质量依赖“沿轨迹线”的局部场景信息；把参考点锚定在 waypoint 上，让每条提案在每个未来点处直接聚合该位置附近的视觉/BEV 信息，实现**以轨迹为中心的感知**，并随 $ \tilde{W}_\ell $ 的迭代细化而逐步对准更合理的区域。

**Lift-Splat BEV 特征采样是什么？**

**Lift-Splat-Shoot**（Philion & Fidler, ECCV 2020）是把**多视角相机图像**编码成 **Bird’s-Eye-View (BEV)** 表示的方法：

- **Lift（抬升）**：对每张图像，为每个像素预测多个深度上的特征（或沿视锥取离散深度），得到“图像 × 深度”的 3D 视锥特征，相当于把 2D 图像沿射线“抬”到 3D。
- **Splat（投影）**：根据相机内外参，把各视角的 3D 特征投影到统一的 BEV 网格上（俯视图 2D 网格，每格对高度方向池化或求和），得到 BEV 特征图。
- **Shoot**：在 BEV 上做规划/控制；与“在 waypoint 处采样”可结合。

在 Drive-JEPA 中，“**aggregating features from $ F_t $ around each predicted waypoint via lift-splat BEV feature sampling**”含义是：若框架中有由 lift-splat 生成的 **BEV 特征图**，则 waypoint 的 $ (x,y) $ 对应 BEV 上的位置；WADA 在该 waypoint 附近（通过偏移）采样时，就是在 BEV 特征图上围绕该位置取特征。若 $ F_t $ 仍是 ViT 的 2D 特征图，则可能通过 waypoint 的 BEV 坐标反推到图像/特征上的区域再做可变形采样，或中间有一层 image-view 到 BEV 的转换后再以 waypoint 为锚做 WADA。**实现上**：用当前 waypoint 的 BEV 坐标作为参考点，在 BEV 特征上做可变形采样，把轨迹经过位置附近的场景信息聚合到 query 中，再配合提案间 self-attention 与 MLP 完成一次迭代更新。

---

#### 3.3.3 仅人类轨迹时的轨迹损失（最小过 N 的监督）

若仅用人类轨迹 $ W_t $ 监督，采用**最小过 N** 的损失，并按迭代做折扣：  
$$
\mathcal{L}_{\mathrm{traj}} = \sum_{\ell=0}^{L-1} \lambda^{L-\ell-1} \min_{n \in \{1,\ldots,N_p\}} \bigl\| W_t - \tilde{W}^{(n)}_\ell \bigr\|^2
$$

- **$ \lambda = 0.1 $**：对更早的迭代降权，实现从粗到细的细化。  
- **$\min_n$**：只惩罚与人类轨迹最近的那条提案，避免强迫所有提案都拟合同一条轨迹，保留一定多样性；但单轨迹监督仍会限制多模态。

---

### 3.4 Multimodal Trajectories Distillation（多模态轨迹蒸馏）

#### 3.4.1 轨迹词表与伪教师构建（实现向步骤）

**第一步：构建轨迹词表（离线，只做一次）**

1. **收集轨迹池**：从整个训练集中收集所有人类驾驶轨迹（每条轨迹为 $ M $ 个 waypoint，如 $ M=8 $），得到超过 **100k** 条轨迹，每条形状为 $ (M, 3) $（$ x, y, \psi $）。
2. **轨迹表示**：为便于 k-means，将每条轨迹展平或保持为固定长度向量（例如 $ 8 \times 3 = 24 $ 维，或加上时间戳等）。
3. **K-means 聚类**：对上述轨迹做 **k-means**（论文用 FAISS 等库），聚类数 $ K = 8192 $，得到 $ 8192 $ 个聚类中心。
4. **词表**：这 $ 8192 $ 个中心即为**轨迹词表** $ \mathcal{V} $；每个中心是一条“典型”轨迹，用于在任意场景下作为候选轨迹之一。

**第二步：对「每个训练场景 × 词表中每条轨迹」计算 EPDMS（离线）**

对训练集中**每一个场景**（对应某一时刻 $ t $ 的观测与自车状态、周围车辆、红绿灯等），需要为**词表中的每条轨迹**（8192 条）打一个 EPDMS 分数，以便后续筛选高质量轨迹。流程如下。

1. **输入**：  
   - 场景 $ t $：当前帧图像、自车状态、地图片段、其他交通参与者轨迹、交通灯状态等（与 NAVSIM v2 的数据格式一致）。  
   - 候选轨迹：词表中的一条轨迹 $ P $，形式为 8 个 waypoint（与模型输出的 $ M=8 $ 一致）。

2. **从 8 waypoint 到可仿真的稠密轨迹**：  
   - 8 个 waypoint 只是稀疏控制点，仿真需要更密的时间步。  
   - 用 **PID 控制器**（或类似跟踪器）将 8 waypoint **插值/跟踪**成更密的轨迹，论文中为 **41 个点**（对应一段时间的离散时间步，如约 4 秒、10 Hz）。  
   - 这样得到一条可用于“逐时间步 replay”的轨迹 $ P^{\mathrm{dense}} $。

3. **规则模拟器 Replay**：  
   - 在**开环**下，固定场景中其他车辆、红绿灯等按真实数据 replay。  
   - 自车按照 $ P^{\mathrm{dense}} $ 的 41 个点**逐时间步移动**，模拟执行这条轨迹。  
   - 在每一步检查：是否碰撞、是否驶出可行驶区域、是否闯红灯、是否逆行、车道保持、舒适度（加速度/曲率等）等。  
   - 这些检查对应 EPDMS 的各个子项：NC, DAC, DDC, TLC, EP, TTC, LK, HC, EC（见第七节式 4）。  
   - 按 NAVSIM v2 的公式把子项聚合成一个 **EPDMS** 标量（0～1 之间）。

4. **输出**：对当前场景 $ t $ 和词表中的轨迹 $ P $，得到一个 EPDMS 分数 $ s(t, P) $。  
5. **规模化**：对**所有**训练场景 $ t $ 和**所有**词表轨迹 $ P \in \mathcal{V} $（8192 条）重复上述过程。论文提到对 NAVSIM v2 的规则模拟器做了**向量化/效率优化**，以支持大规模离线打分。

**第三步：为每个场景选出伪教师集合 $ \mathcal{P}_t $（离线）**

1. **排序与阈值**：对场景 $ t $，词表中 8192 条轨迹各有分数 $ s(t, P) $。按 $ s(t, P) $ 从高到低排序，设**阈值** $ \tau = 0.95 $（附录 B）。  
2. **高质量集合**：所有满足 $ s(t, P) \geq \tau $ 的轨迹 $ P $ 组成“高质量集合”；若超过 $ N_{\mathrm{pseudo}} $ 条，则从中**均匀随机抽样** $ N_{\mathrm{pseudo}} $ 条（论文消融中 $ N_{\mathrm{pseudo}} \in \{0,1,2,4,8\} $）。  
3. **伪教师集合**：  
   $$
   \mathcal{P}_t = \{P^1_t, \ldots, P^{N_{\mathrm{pseudo}}}_t\}
   $$
   即为该场景 $ t $ 的**伪教师轨迹**，在训练多模态蒸馏时与人类轨迹 $ W_t $ 一起监督提案（见式 2）。

**实现要点小结**

| 步骤 | 输入 | 输出 |
|------|------|------|
| 词表构建 | 训练集所有轨迹（>100k） | 8192 个聚类中心 $ \mathcal{V} $ |
| 单次 EPDMS | 场景 $ t $ + 词表中一条 $ P $（8 waypoint） | 标量 EPDMS $ s(t,P) $ |
| 稠密轨迹 | 8 waypoint | 41 点轨迹（PID 等） |
| 伪教师选择 | 场景 $ t $，$ s(t,P) $ 及阈值 0.95 | $ \mathcal{P}_t $，大小为 $ N_{\mathrm{pseudo}} $ |

训练时：每个 batch 的样本带有其预计算好的 $ \mathcal{P}_t $（和人类轨迹 $ W_t $），损失按式 2 用 $ W_t $ 与 $ \mathcal{P}_t $ 一起监督 $ \tilde{W}_\ell $。

#### 3.4.2 多模态轨迹蒸馏损失（式 2）

最终轨迹损失为：  
$$
\mathcal{L}_{\mathrm{traj}} = \sum_{\ell=1}^{L} \lambda^{L-\ell} \left( \min_n \|W_t - \tilde{W}^{(n)}_\ell\|^2 + \sum_{P \in \mathcal{P}_t} \min_n \|P - \tilde{W}^{(n)}_\ell\|^2 \right).
$$

- **$\min_n$**：对提案索引 $ n \in \{1,\ldots,N_p\} $ 取最小。  
- **第一项**：人类轨迹 $ W_t $ 与最近提案的 $ L_2 $ 距离。  
- **第二项**：每个伪教师轨迹 $ P \in \mathcal{P}_t $ 与最近提案的 $ L_2 $ 距离之和。  
- **$\lambda^{L-\ell}$**：同样对早期迭代降权。  
- **效果**：提案分布同时拟合人类轨迹与多条模拟器高质量轨迹，缓解单轨迹监督导致的**模式坍塌**，使提案呈现多模态（见 Figure 3）。

#### 3.4.3 Figure 3 解读

- **无 MTD**：提案在 BEV 上**坍缩为单一模式**（几乎重合）。  
- **有 MTD**：提案在 BEV 上**多模态分布**，覆盖不同合理路径。  
- 说明多模态轨迹蒸馏对**提案多样性**至关重要。

---

### 3.5 Momentum-aware Trajectory Selection（动量感知轨迹选择）

#### 3.5.1 打分器与 BCE 监督（MLP 设计、输入与 $ \hat{S} $ 的获取）

**（1）输入到打分器的特征**

- 最终提案对应的 query 为 $ Q^L \in \mathbb{R}^{N_p \times M \times D} $：$ N_p $ 条提案，每条 $ M $ 个 waypoint，每个 waypoint 一个 $ D $ 维向量。  
- 在 **waypoint 维度**上做 **max pooling**（即对每个提案的 $ M $ 个向量逐维取最大值），得到 $ \bar{Q}^L \in \mathbb{R}^{N_p \times D} $。  
- **含义**：每条提案用一条 $ D $ 维向量表示，聚合了整条轨迹上的语义，作为该提案的“描述子”。

**（2）MLP 设计与输出 $ S $**

- **输入**：$ \bar{Q}^L $，形状 $ (N_p, D) $；即逐条提案，每条一个 $ D $ 维向量。  
- **MLP 结构**：论文未给层数与宽度，常见做法为 2～3 层全连接 + 激活（如 ReLU），最后一层输出维度 1，再经过 **sigmoid**，得到每条提案的分数在 $ (0,1) $ 之间。  
  - 例如：$ \bar{Q}^L \to \mathrm{Linear}(D \to D') \to \mathrm{ReLU} \to \mathrm{Linear}(D' \to 1) \to \sigma \to S $。  
- **输出**：$ S \in \mathbb{R}^{N_p \times 1} $，$ S_n $ 表示第 $ n $ 条提案的“质量”分数（标量）。

**（3）监督标签 $ \hat{S} $ 如何由模拟器 EPDMS 得到**

- **与伪教师同一套规则模拟器**：对**同一条提案轨迹** $ \tilde{W}^{(n)}_L $（即模型在训练时对当前场景预测的第 $ n $ 条提案），用与 3.4.1 相同的流程在规则模拟器中跑一遍——8 waypoint → PID 得到 41 点 → 在当帧场景下 replay → 计算 EPDMS，得到标量 $ \mathrm{EPDMS}(t, \tilde{W}^{(n)}_L) $。  
- **二值化为 $ \hat{S}_n $**：设定阈值（与伪教师一致，如 0.95），若 $ \mathrm{EPDMS}(t, \tilde{W}^{(n)}_L) \geq 0.95 $，则 $ \hat{S}_n = 1 $（高质量），否则 $ \hat{S}_n = 0 $（低质量）。  
- **向量**：$ \hat{S} \in \mathbb{R}^{N_p \times 1} $，与 $ S $ 逐元素对应。  
- **训练**：用 BCE 让 $ S $ 逼近 $ \hat{S} $，即学会“在给定场景下，哪条提案在模拟器里会得高分”，从而在推理时无需再跑模拟器，直接用网络输出的 $ S $ 选轨迹。

**（4）损失**

$$
\mathcal{L}_{\mathrm{score}} = \mathrm{BCE}(S, \hat{S}), \quad \mathrm{BCE}(x,y) = -y\log x - (1-y)\log(1-x).
$$

---

#### 3.5.2 舒适项与分数重校准（$ S_c $ 如何计算）

- **问题**：多模态蒸馏会增加**时间上的不一致**，相邻帧所选轨迹差异大，影响舒适性。  
- **思路**：引入**舒适分数** $ S_c \in \mathbb{R}^{N_p \times 1} $，度量“当前每条提案与**上一帧已选轨迹** $ \hat{W}_{t-1} $ 的接近程度”；越接近则帧间抖动越小、舒适度越高，$ S_c $ 应越大。

**$ S_c $ 的计算方式（论文表述与常见实现）**

- 论文写：*“We compute a distortion-based comfort score $ S_c $ by comparing $ \hat{W}_{t-1} $ with each current proposal in $ \{ \tilde{W}^{(n)}_L \}_{n=1}^{N_p} $”*，即基于**失真/差异**的舒适分数。  
- **典型做法**：对第 $ n $ 条提案 $ \tilde{W}^{(n)}_L $，与 $ \hat{W}_{t-1} $ 在**同一组 waypoint 上**逐点比较（两者都是 $ M $ 个 $ (x,y,\psi) $），计算某种“距离”$ d_n $，例如：  
  - $ d_n = \sum_{i=1}^{M} \bigl( (\Delta x_i)^2 + (\Delta y_i)^2 + \alpha (\Delta \psi_i)^2 \bigr) $，其中 $ \Delta $ 为当前提案与上一帧轨迹在对应 waypoint 上的差；或  
  - 使用与 NAVSIM v2 中 **History Comfort (HC)** 类似的平滑度/连续性指标（如加速度、曲率变化等）在“上一帧轨迹 + 当前提案”的拼接轨迹上计算，得到不舒适度再取反或归一化。  
- **从距离到分数**：距离越大越不舒适，因此可将 $ d_n $ 映射为“舒适分数”，例如 $ S_{c,n} = \exp(-\beta \cdot d_n) $ 或 $ S_{c,n} = 1 / (1 + d_n) $，使 $ S_c $ 与 $ S $ 同尺度（如都在 $ [0,1] $），便于线性混合。  
- **权重**：按 NAVSIM v2 的设定，重校准时  
  $$
  S \leftarrow \frac{7}{8}S + \frac{1}{8}S_c.
  $$
  即 7/8 来自学习到的 EPDMS 相关分数 $ S $，1/8 来自舒适项 $ S_c $，在保证安全与规则遵守的前提下，鼓励与上一帧轨迹更连续的提案被选中。

#### 3.5.3 最终轨迹选择

$$
\hat{W}_t = \tilde{W}^{(n^*)}_L, \qquad n^* = \arg\max_{n \in \{1,\ldots,N_p\}} S_n.
$$

- $ S_n $：重校准后第 $ n $ 个提案的分数。  
- 选得分最高的提案作为当前帧输出轨迹。

---

### 3.6 Losses（总损失与辅助任务）

#### 3.6.1 辅助任务

- **Proposal-centric mapping**：预测 $ \tilde{W}_\ell $ 中各 waypoint 的**在路**与**在规划路线**概率 $ R \in \mathbb{R}^{N_p \times M \times 2} $。  
  $$
  \mathcal{L}_{\mathrm{map}} = \mathrm{BCE}(R, \hat{R}).
  $$
- **Proposal-centric collision**：用 log-replay 仿真估计 waypoint 的碰撞概率 $ A_v $，监督 $ \hat{A}_v $。  
  $$
  \mathcal{L}_{\mathrm{colli}} = \|A_v - \hat{A}_v\| + 0.1 \cdot \mathrm{BCE}(A_v, \hat{A}_v).
  $$
  - **0.1**：BCE 项权重，避免碰撞预测被 BCE 主导。

#### 3.6.2 总损失

$$
\mathcal{L} = \mathcal{L}_{\mathrm{traj}} + w_{\mathrm{score}}\mathcal{L}_{\mathrm{score}} + w_{\mathrm{map}}\mathcal{L}_{\mathrm{map}} + w_{\mathrm{colli}}\mathcal{L}_{\mathrm{colli}},
$$

文中：  
- $ w_{\mathrm{score}} = 1 $，  
- $ w_{\mathrm{map}} = 2 $，  
- $ w_{\mathrm{colli}} = 1 $。

---

## 四、Method 部分涉及的图表汇总

### Figure 2：Drive-JEPA 整体架构（深入解析）

Figure 2 从左到右、从上到下大致分为四块：**预训练**、**场景编码与提案生成**、**多模态轨迹蒸馏**、**动量感知轨迹选择**。下面按数据流与模块逐一说明。

**（1）左上：Driving Video Pretraining（驾驶视频预训练）**

- **输入**：大规模驾驶视频（8 帧片段、512×256、2 Hz）。  
- **流程**：  
  - 对视频做**随机掩码**（Mask），得到上下文视图 $ x $ 与目标视图 $ y $。  
  - **上方分支**：$ x $ 经 **Vision Encoder**（ViT）得到 $ E_\theta(x) $，再与 **Mask predictor** 一起预测被掩码位置的表示；预测器输出与**目标**做 $ L_1 $ 损失。  
  - **下方分支**：$ y $ 经 **EMA Vision Encoder**（$ E_{\bar{\theta}} $）得到目标表示，经 **stop-gradient** 后作为预测目标，不反传梯度。  
- **输出**：预训练好的 ViT 编码器，用于下游的 Driving Scene Encoding。

**（2）中间偏左：Driving Scene Encoding + Waypoint-anchored Proposal Generation**

- **输入**：  
  - **History Images**（历史图像，如 $ I_t, I_{t-1} $）从 **Frame Buffer** 来；  
  - **Ego History**（自车历史状态）经 **Linear Ego Status** 得到 ego 特征。  
- **场景编码**：History Images 输入 **Vision Encoder**（与预训练共享或微调），得到 **Image Features**（即 $ F_t $）。  
- **提案生成**：  
  - **Initial BEV Proposal Queries** + **Learnable Embeddings** 得到初始 $ Q^0 $（$ N_p \times M \times D $）。  
  - 经过 **Self Attention**（提案间交互）与 **Spatial Cross Attention**（即 WADA：以 waypoint 为锚在 BEV/特征上采样），重复 **× $ L $** 次，得到 **Proposals** $ \tilde{W}_L $（$ N_p $ 条轨迹，每条 $ M $ 个 waypoint）。  
- **数据流**：Frame Buffer + Ego History → Vision Encoder → Image Features；Image Features + Initial Queries → [Self Attn + WADA] × $ L $ → Proposals。

**（3）右侧：Multimodal Trajectory Distillation**

- **输入**：  
  - **Driving Scenario**（当前驾驶场景，即训练样本）；  
  - **Human Trajectories**（该场景的人类轨迹 $ W_t $）；  
  - **Trajectory Vocabularies**：由 **K-means Clustering** 在大量轨迹上得到 8192 个中心，构成词表。  
- **流程**：对词表中的轨迹在**当前场景**下用规则模拟器算 **EPDM Score**，经 **EPDM Score Filter**（阈值 0.95 等）筛选，得到 **Diverse High Score Pseudo-Teacher Trajectories** $ \mathcal{P}_t $。  
- **与主干的联系**：训练时，$ W_t $ 与 $ \mathcal{P}_t $ 一起监督左侧生成的 **Proposals**（式 2），使提案分布既拟合人类轨迹又覆盖模拟器认可的多条高质量轨迹，避免模式坍塌。

**（4）下方：Momentum-aware Trajectory Selection**

- **输入**：当前帧的 **Proposals** $ \tilde{W}^{(n)}_L $，以及上一帧的已选轨迹 $ \hat{W}_{t-1} $（图中标为 $ t-1 $ 与 $ t $）。  
- **流程**：  
  - 对每条提案用 **EPDM**（或 EPDMS）相关逻辑得到基础质量分数（训练时由 BCE 与 $ \hat{S} $ 学习）；  
  - **Inter-frame Comfort Scorer** 比较 $ \hat{W}_{t-1} $ 与每条 $ \tilde{W}^{(n)}_L $，得到舒适分数 $ S_c $；  
  - 将基础分数与 $ S_c $ 按 7/8 与 1/8 混合，得到最终分数，**选最高分**的提案作为当前帧输出轨迹。  
- **输出**：最终规划轨迹 $ \hat{W}_t $。

**（5）整体数据流小结**

- **离线**：预训练 ViT（左上）；构建轨迹词表 + 对每场景每词表轨迹算 EPDMS，得到伪教师（右侧，可离线预计算）。  
- **在线/训练**：历史图像 + 自车状态 → 编码 → 提案 query 初始化 → WADA × $ L $ → Proposals；Proposals 受 $ W_t $ 与 $ \mathcal{P}_t $ 监督（多模态蒸馏）；同时打分器学 EPDMS 对应分数，推理时再叠加舒适项并选取最高分提案。

### Table 1（见 3.2.2）

感知无关规划器对比：编码器规模、数据规模、PDMS。

### Figure 3（见 3.4.3）

BEV 下提案分布：无 MTD 单峰坍塌 vs 有 MTD 多模态。

---

## 五、Method 中数学公式与参数总表

| 公式/位置 | 含义 | 主要参数 |
|-----------|------|----------|
| 式 (1) | V-JEPA 编码器+预测器损失 | $ \theta,\phi,\Delta y $; $ E_{\bar{\theta}} $ EMA |
| $ W_t = \{w^1_t,\ldots,w^M_t\} $ | 未来 waypoint 序列 | $ M $ waypoint 数 |
| $ w^i_t = (x^i_t,y^i_t,\psi^i_t) $ | 单点 BEV 位置+航向 | - |
| $ F_t \in \mathbb{R}^{N_f \times D} $ | ViT 时空特征 | $ N_f $ token 数，$ D $ 维度 |
| $ H = \mathrm{TransformerDecoder}(Q, F_t) $ | 解码器输出 | $ Q \in \mathbb{R}^{M \times D} $ |
| $ \hat{W}_t = \mathrm{MLP}(H) $ | 预测 waypoints | - |
| $ Q^0 \in \mathbb{R}^{N_p \times M \times D} $ | 提案 query 初始化 | $ N_p=32 $, $ M $, $ D $ |
| $ Q^{\ell+1} = \mathrm{MLP}(\mathrm{WADA}(\cdots)) $ | 迭代提案更新 | $ L $ 迭代次数 |
| $ \lambda^{L-\ell-1} $, $ \lambda^{L-\ell} $ | 迭代折扣 | $ \lambda=0.1 $ |
| 式 (2) $ \mathcal{L}_{\mathrm{traj}} $ | 多模态轨迹蒸馏损失 | $ N_p $, $ N_{\mathrm{pseudo}} $, $ L $ |
| $ S \leftarrow \frac{7}{8}S + \frac{1}{8}S_c $ | 动量感知分数重校准 | 7/8、1/8 为 NAVSIM v2 权重 |
| $ n^* = \arg\max_n S_n $ | 轨迹选择 | - |
| $ \mathcal{L} = \mathcal{L}_{\mathrm{traj}} + \cdots $ | 总损失 | $ w_{\mathrm{score}}=1 $, $ w_{\mathrm{map}}=2 $, $ w_{\mathrm{colli}}=1 $ |

---

## 六、关键超参与实现数字小结

| 名称 | 值 | 含义 |
|------|-----|------|
| 前视分辨率 | 512×256 | 输入图像 |
| 输入帧 | $ I_t, I_{t-1} $ | 当前+前一帧，2×512×256 |
| 预训练视频 | 208h, 8-frame, 2Hz | 驾驶视频预训练 |
| 轨迹词表大小 | 8192 | k-means 聚类中心数 |
| EPDMS 阈值 | 0.95 | 伪教师轨迹筛选 |
| $ N_{\mathrm{pseudo}} $ | 0/1/2/4/8 消融 | 每场景伪教师数量 |
| $ N_p $ | 32 | 提案数量 |
| $ \lambda $ | 0.1 | 迭代折扣系数 |
| 分数混合 | 7/8 学习分数 + 1/8 舒适 | 动量感知重校准 |
| PID 输出点数 | 41 | 从 8 waypoint 插密后用于仿真 |

---

## 七、与 Method 相关的评估公式（论文式 3、式 4）

伪教师筛选与打分器监督均基于模拟器 EPDMS。NAVSIM 定义如下。

**PDMS（NAVSIM v1，式 3）**：
$$
\mathrm{PDMS} = NC \times DAC \times \frac{5 \times (EP + TTC) + 2 \times C}{12}
$$
- $ NC $：无责任碰撞；$ DAC $：可行驶区域遵守；$ EP $：自车进度；$ TTC $：时间到碰撞；$ C $：舒适度。  
- 分子权重和为 $ 5+5+2=12 $，$ NC $ 与 $ DAC $ 为硬约束（任一为 0 则 PDMS=0）。

**EPDMS（NAVSIM v2，式 4）**：
$$
\mathrm{EPDMS} = NC \times DAC \times DDC \times TLC \times \frac{5 \times (EP + TTC) + 2 \times (LK + HC + EC)}{16}
$$
- 新增：$ DDC $ 行驶方向遵守、$ TLC $ 红绿灯遵守、$ LK $ 车道保持、$ HC $ 历史舒适、$ EC $ 扩展舒适。  
- 分母 16 对应权重和 $ 5+5+2+2+2=16 $。  
- Method 中 EPDMS 阈值 0.95 用于从词表中筛选高质量轨迹构成 $ \mathcal{P}_t $。

---

## 八、第 4 节 Experiments（实验）详细解读

本节对应论文 **4. Experiments**，包含数据集与指标定义（4.1）、实现细节（4.2）、主结果（4.3）与消融实验（4.4），以及所涉表格与图示的逐项说明。

---

### 8.1 Dataset and Metrics（4.1 数据集与指标）

**（1）评估基准概览**

论文在三个基准上评估：**NAVSIM v1**、**NAVSIM v2**、**Bench2Drive**。前两者为**开环**评估（模型输出轨迹后由规则模拟器打分），后者为**闭环**仿真（自车在 CARLA 中执行规划、与场景交互）。

**（2）NAVSIM v1**

- **数据来源**：基于 **OpenScene**（Sima et al., 2023）与 **NuPlan**（Caesar et al., 2021）的真实世界驾驶数据。  
- **规模**：**103k** 场景用于训练（Navtrain），**12k** 场景用于评估（Navtest）；场景多样且具挑战性。  
- **评估方式**：**开环**——模型对每帧输出一条轨迹，不真正控制车辆；该轨迹交给**规则模拟器**在场景中 replay，得到各项子指标后再聚合成 PDMS。  
- **动机**：用仿真指标在开环下近似衡量“若按该轨迹执行”的闭环表现，便于大规模对比与消融，且与真实驾驶数据一致（无闭环执行误差）。

**指标与式 (3)**：  
- **NC**（No at-fault Collisions）：无责任碰撞，0/0.5/1。  
- **DAC**（Drivable Area Compliance）：可行驶区域遵守，0/1。  
- **TTC**（Time to Collision with bounds）：时间到碰撞（有界），0/1。  
- **EP**（Ego Progress）：自车进度，[0,1]。  
- **C**（Comfort）：舒适度，0/1。  
- **PDMS**（PDM Score）：
  $$
  \mathrm{PDMS} = NC \times DAC \times \frac{5 \times (EP + TTC) + 2 \times C}{12}
  $$
  $ NC $、$ DAC $ 为硬约束（任一为 0 则 PDMS=0）；分子为加权和，分母 12 对应权重 $ 5+5+2 $。

**（3）NAVSIM v2**

- **相对 v1 的加强**：将 PDMS 扩展为 **EPDMS**，增加**规则遵守**与**舒适性**的细粒度评估。  
- **新增子指标**：  
  - **DDC**（Driving Direction Compliance）：行驶方向合法（不逆行等）。  
  - **TLC**（Traffic Light Compliance）：红绿灯遵守。  
  - **LK**（Lane Keeping）：车道保持。  
  - **HC**（History Comfort）：短时舒适（与人类近期轨迹拼接后评估，见附录 A）。  
  - **EC**（Extended Comfort）：跨时间步的平滑性（相邻帧轨迹连续性）。  
- **EPDMS 公式**（论文式 4）：  
  $$
  \mathrm{EPDMS} = NC \times DAC \times DDC \times TLC \times \frac{5 \times (EP + TTC) + 2 \times (LK + HC + EC)}{16}
  $$
  分母 16 对应权重和 $ 5+5+2+2+2 $；$ DDC $、$ TLC $ 也为硬约束。

**（4）Bench2Drive**

- **设定**：基于 **CARLA** 的**闭环**评估，面向交互式城市场景。  
- **规模**：**220** 条路线，覆盖 **44** 个交互场景。  
- **指标**：  
  - **DS**（Driving Score）：综合驾驶分数（路线完成 + 违规惩罚）。  
  - **SR**（Success Rate）：成功完成路线比例。  
  - **Efficiency**：效率（与周围车流速度比较等）。  
  - **Comfortness**：舒适度（遵循 nuPlan 的平滑度协议）。  
- **动机**：检验在闭环、多车交互下的实际表现，与开环 NAVSIM 形成互补。

---

### 8.2 Implementation Details（4.2 实现细节）

**（1）驾驶视频预训练阶段**

| 项目 | 设定 | 含义 |
|------|------|------|
| 硬件 | 8 × H800 GPU | 预训练算力 |
| 训练轮数 | 50 epochs | 在约 208h 驾驶视频上 |
| 耗时 | 约 3 天 | 与数据规模、V-JEPA 效率相关 |

**（2）Drive-JEPA 规划器训练阶段**

| 项目 | 设定 | 含义 |
|------|------|------|
| 硬件 | 2 × NVIDIA A30 | 规划器与编码器微调 |
| 优化器 | Adam | - |
| 总 batch size | 64 | - |
| 训练轮数 | 20 epochs | - |
| 规划器学习率 | $ 1 \times 10^{-4} $ | 提案、打分器等 |
| ViT 编码器学习率 | $ 1 \times 10^{-5} $ | 较小，避免破坏预训练表示 |
| 提案数 $ N_p $ | 32 | 消融中兼顾效率与性能 |
| 输入 | 仅前视相机 | 分辨率 512×256；无 LiDAR、无多视角堆叠 |

**（3）设计动机简述**

- **编码器小学习率**：预训练表示已与规划对齐，微调时主要更新规划头与 WADA，编码器仅小幅适配。  
- **单前视 + 512×256**：相比 Transfuser/GoalFlow 等的多视角或更高分辨率，Drive-JEPA 在更少输入下仍达 SOTA，突出 V-JEPA 预训练与多模态蒸馏的贡献。

---

### 8.3 Main Results（4.3 主结果）

**（1）Table 2：NAVSIM v1 定量对比**

- **表格结构**：  
  - **第一块**：**感知无关**设置（无感知标注，仅人类轨迹监督）；方法、Backbone、Inputs、NC/DAC/EP/C/TTC、**PDMS**。  
  - **第二块**：**感知相关**设置（可使用检测/分割等）；同样列出各子指标与 PDMS。  
- **符号与列**：  
  - **Inputs**：C & L = Camera + LiDAR；Camera = 仅相机。  
  - **↑**：该列数值越高越好。  
  - **NC, DAC, EP, C, TTC**：多为百分比或 0–100 刻度，PDMS 为综合分（越高越好）。

**主要结论**：  
- **感知无关**：Ours (ViT/L, Camera) 达到 **89.0 PDMS**，优于 LAW 83.8、World4Drive 85.1、Epona 86.2；说明 V-JEPA 驾驶视频预训练 + 简单解码器即可显著超越先前方法。  
- **感知相关**：  
  - ResNet34：Drive-JEPA **91.5** PDMS，优于 iPad 91.1、DiffusionDrive 88.1 等。  
  - ViT/L：Drive-JEPA **93.3** PDMS，仅次于 DriveSuprim 93.5（后者使用更强数据增强）；且 Drive-JEPA 在 **EP**（90.8）上最佳，体现更积极的驾驶风格。  
- **安全与舒适**：NC、DAC、TTC 等保持高位（约 98–99），说明 SOTA 并非以牺牲安全换进度。

**（2）Table 3：NAVSIM v2 定量对比**

- **列**：在 v1 基础上增加 **DDC, TL（TLC）, LK, HC, EC**，综合分为 **EPDMS**。  
- **结论**：  
  - Drive-JEPA (ResNet34) **85.4** EPDMS；(ViT/L) **87.8** EPDMS，均为表中最佳。  
  - 其他方法在 **EC**（扩展舒适）上普遍偏低（约 68–78），Drive-JEPA 达 **84.8**，与**动量感知轨迹选择**（M4）直接相关。  
  - DDC、TLC、LK 等规则遵守项 Drive-JEPA 也接近或达到最高，说明多模态蒸馏与打分器并未牺牲规则遵守。

**（3）Table 4：Bench2Drive 定量对比**

- **列**：**Effi.**（Efficiency）、**Comf.**（Comfortness）、**SR**、**DS**。  
- **结论**：  
  - Drive-JEPA **DS 64.52**，为表中最高；**SR 36.82** 也最佳。  
  - Efficiency 157.85，与 VAD 等相当；Comfort 30.24，低于部分方法，但 DS 综合领先。  
  - 相对同属 proposal-centric 的 **iPad**（DS 60.52），Drive-JEPA 高约 **4** 分，论文归因于**多模态轨迹蒸馏**带来的多样性与安全性。

**（4）感知无关小结**

- 在 NAVSIM v1 上，仅用 V-JEPA 预训练 ViT + 简单 Transformer 解码器（3.2 节），不依赖感知标注即可达到 89.0 PDMS，接近甚至超过部分依赖感知的方法，说明**预训练表示与规划任务高度对齐**。

---

### 8.4 Ablation Studies（4.4 消融实验）

**（1）Table 5：各模块消融（NAVSIM v2）**

- **模块定义**：  
  - **M1**：使用 V-JEPA 2 发布的 ViT  checkpoint（替换 ResNet34）。  
  - **M2**：Driving Video Pretraining（在驾驶视频上用 V-JEPA 目标预训练）。  
  - **M3**：Multimodal Trajectory Distillation（MTD）。  
  - **M4**：Momentum-aware Trajectory Selection。  
- **表格读法**：每行对应一组开关（% = 关闭，! = 开启）；最后一列 **EPDMS** 及相对增量（如 +1.7, +2.0）。  
- **数值要点**：  
  - 基线（全关，即 iPad 式）：EPDMS **84.1**，EC **68.2**，D（Diversity）**25%**。  
  - 仅 M1（ViT）：**85.8**（+1.7）。  
  - 仅 M2（+ 驾驶视频预训练）：**86.1**（+2.0）。  
  - M1+M2+M3（无 M4）：**84.5**（+0.4）；**EC 骤降至 47.9**，**D 升至 40%**——多模态蒸馏带来多样性，但帧间一致性变差。  
  - 全开 M1+M2+M3+M4：**87.8**（+3.7），**EC 恢复至 84.8**，**D 保持 40%**。  
- **结论**：M4（动量感知选择）在保持多模态多样性的同时显著修复 EC，是 EPDMS 创新高的关键；M2 缩小域差，M3 提升多样性，二者与 M4 互补。

**（2）Table 6：伪教师数量 $ N_{\mathrm{pseudo}} $**

- **设定**：$ N_{\mathrm{pseudo}} \in \{0, 1, 2, 4, 8\} $。  
- **结果**：0 时 EPDMS **87.2**；1/2/4/8 时在 **87.5–87.8** 之间，略优于 0，但彼此差异不大。  
- **含义**：使用伪教师（$ \geq 1 $）一致优于不用；数量在 1–8 间对 EPDMS 影响不敏感，实现时可按计算成本选适中值（如 2 或 4）。

**（3）Table 7：驾驶视频预训练 vs 其他视觉预训练**

- **设定**：**同一简单解码器**（3.2 节），仅替换**编码器**为不同预训练方法得到的 ViT/L 或 ResNet34。  
- **结果**：  
  - ImageNet ResNet34 **76.0**；Dinov2 ViT/L **76.1**；Sigclip ViT/L **83.4**；V-JEPA 2 ViT/L **86.1**；**Ours（驾驶视频 + V-JEPA 目标）89.0**。  
  - MAE、DepthAnything 的 ViT/L **未收敛**（表中 “-”）。  
- **结论**：  
  - V-JEPA 2 在通用视频上已优于图像/深度预训练；在**驾驶域数据**上进一步用 V-JEPA 目标预训练，再提升约 **3 PDMS**，超过 Epona。  
  - 说明**任务与域对齐的预训练**对规划表示至关重要；MAE 等重建类目标在该设定下难以收敛或迁移。

**（4）Figure 4：定性对比**

- **内容**：不同场景下，**Human Trajectory**、**Drive-JEPA**、**iPad**、**Transfuser** 的轨迹在前视相机视图与 BEV 上的对比。  
- **作用**：直观展示 Drive-JEPA 与人类轨迹的贴合度、与 iPad/Transfuser 的差异（如更顺滑、更少抖动或更合理超车/变道）。

**（5）Figure 5：多模态轨迹蒸馏对 PDMS 的影响**

- **内容**：验证集上** PDM 分数随训练变化**的曲线；应包含“有 MTD”与“无 MTD”或不同配置的对比。  
- **作用**：支撑“多模态轨迹蒸馏提升 PDMS”的结论（与 Table 5 中 M3 带来的多样性、以及 M4 对 EC 的修复一致）。

---

### 8.5 实验部分涉及的图表与数字汇总

| 图表 | 内容 | 主要结论/用途 |
|------|------|----------------|
| Table 2 | NAVSIM v1 主结果（感知无关 + 感知相关） | 感知无关 89.0；感知相关 93.3；EP 最佳 |
| Table 3 | NAVSIM v2 主结果 | EPDMS 87.8；EC 84.8 领先 |
| Table 4 | Bench2Drive 主结果 | DS 64.52，SR 36.82 最佳；相对 iPad +4 DS |
| Table 5 | 模块消融 M1–M4 | M4 修复 EC、提升 EPDMS；M3 提升 D |
| Table 6 | $ N_{\mathrm{pseudo}} $ 消融 | 有伪教师优于无；1–8 条差异不大 |
| Table 7 | 视觉预训练方法对比 | 驾驶 V-JEPA 预训练 89.0，超 Epona 3 PDMS |
| Figure 4 | 轨迹定性对比 | 多模型轨迹可视化 |
| Figure 5 | MTD 与 PDMS 曲线 | 多模态蒸馏提升验证分数 |

---

*以上解读基于论文 arXiv:2601.22032 的 Abstract、Introduction、Section 3 Method 与 Section 4 Experiments，公式与图表编号与原文对应。*
