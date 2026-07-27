# AOI — Automatic Optical Inspection（自动光学检测）

一个面向**工业产线实时缺陷检测**的 AI 系统，核心解决三大难题：**实时性**、**少样本启动**、**未见缺陷泛化**。

---

## 📋 问题定义

传统工业质检依赖人工目检或传统机器视觉，存在以下痛点：

| 挑战 | 描述 |
|------|------|
| **实时性要求高** | 产线节拍要求毫秒级单图推理，无法使用慢速大模型 |
| **缺陷样本极度稀缺** | 新产品线初期只有约 100 张正常品和 30 张瑕疵品可用 |
| **缺陷类型不可穷举** | 产线上会出现训练阶段从未见过的缺陷类型（如弯曲、缺口等） |
| **多维度检测需求** | 同一产线需同时检测外观划痕、颜色偏差、组件缺失、几何变形等多类异常 |
| **持续优化闭环** | 产线操作员发现误报/漏报后，系统需要快速修正而不中断生产 |

本项目针对以上问题，设计了一套完整的 AI 解决方案。
问题描述如下：

![Question](Question.png)

---

## 🏗 系统整体架构

```
                        ┌──────────────────────────────────────┐
                        │       阶段一：公共数据集预训练         │
                        │   MVTec AD / VisA / DAGM / LOCO AD   │
                        └────────────────┬─────────────────────┘
                                         │ 输出：industrial_checkpoint
                                         ▼
                        ┌──────────────────────────────────────┐
                        │        阶段二：目标产线少样本迁移      │
                        │   100 张正常 + 30 张瑕疵（few-shot）  │
                        │   + 合成数据引擎数据增强              │
                        └────────────────┬─────────────────────┘
                                         │ 输出：target_model + normal_reference
                                         ▼
             ┌─────────────┬─────────────────┬─────────────────┐
             │             │                 │                  │
             ▼             ▼                 ▼                  ▼
      ┌──────────┐ ┌────────────┐ ┌──────────────┐ ┌──────────────┐
      │实时单图检测│ │实时视频检测 │ │反馈即时调整   │ │反馈周期重训   │
      │单次前向 +  │ │帧检测 +    │ │阈值自适应调整 │ │联合旧数据Fine-│
      │多分支融合   │ │GRU时序建模 │ │正常库候选更新 │ │tune + 自动回滚│
      └──────────┘ └────────────┘ └──────────────┘ └──────────────┘
```

---

## 🔬 核心技术详解

### 1. 多分支模型架构（`aoi_model.py`）

以 **ConvNeXt V2 Tiny** 为共享骨干网络，单次前向同时提取 **4 个分辨率层级特征**（f4 / f8 / f16 / f32），接入 6 个独立任务头：

```
输入图像 (384×384)
    │
    ▼
ConvNeXt V2 Tiny 骨干
    │
    ├── f16 (384通道, 24×24) ──→ local_head ──→ 局部异常热力图
    │                                │              [像素级划痕/污渍/裂纹]
    │                                ▼
    │                          Top-k 平均 → local_top_score
    │
    ├── f32 (768通道, 12×12) ──→ global_head ──→ 全局异常分数
    │                                             [整体异常]
    │
    ├── f32 ──→ component_head ──→ 8个组件slot置信度
    │                                   [缺件/错位/逻辑异常]
    │
    ├── f32 ──→ geometry_head ──→ 6维几何回归
    │                             [宽高比/面积/周长等]
    │
    ├── f32 ──→ domain_head ──→ real/synthetic 域分类
    │                  (带梯度反转层 GRL)
    │
    └── 各分支分数 ──→ fusion_head ──→ 最终融合分数
```

**最终融合公式**：
```
final = max(local_top_score, global_logit) + 0.5 * tanh(fusion_head([local_top, local_mean, global_logit, component_uncertainty, geometry_magnitude, external_scores]))
```

**设计理念**：将多种缺陷模态解耦到不同分支，再由融合头自适应组合。局部头捕捉微小瑕疵，全局头检测整体异常，组件头识别结构缺陷，几何头捕获形状变形。

---

### 2. 两阶段训练策略（`fewshot_transfer.py`）

#### 阶段一：公共工业预训练

在 4 个公开工业缺陷数据集上进行大规模预训练：

| 数据集 | 领域 | 类别数 | 任务类型 |
|--------|------|--------|----------|
| MVTec AD | 工业外观检测 | 15 类 | appearance |
| MVTec LOCO AD | 逻辑/结构缺陷 | 5 类 | structure / logic |
| VisA | 视觉异常 | 12 类 | appearance |
| DAGM 2007 | 纹理缺陷 | 10 类 | appearance |

- 使用 WeightedRandomSampler 缓解类别不均衡
- 损失函数组合：
  - **分类损失**（BCE）：图像级正常/异常二分类
  - **全局损失**：全局头单独监督
  - **局部损失**：局部头 Top-k 分数监督
  - **分割损失**（BCE + Dice）：对有像素级标注的样本做分割
  - **组件损失**：对结构/逻辑类别的样本的组件头监督
  - **几何损失**（Smooth L1）：几何回归监督
  - **排名损失**（Ranking Loss）：拉大正常与异常的分数差距

#### 阶段二：目标产线少样本迁移

使用 **100 张正常 + 30 张瑕疵** 实现新产线适配，分 4 步渐进微调：

```
Step 1: heads 微调 (3 epochs)     — 只训练 6 个任务头
Step 2: stage4 微调 (3 epochs)    — 解冻骨干最后 stage
Step 3: stage3 微调 (3 epochs)    — 解冻骨干最后两个stage
         + 35% 概率合成数据增强    — 从正常图生成外观/颜色/几何缺陷
Step 4: heads 修正 (2 epochs)     — 仅用真实数据校准
```

每一步训练后都在验证集上评估 AUC，保存最佳 checkpoint，**防止过拟合**。

---

### 3. 正常参考库与记忆检索（`normal_reference.py`）

这是实现**未见缺陷泛化**的核心机制。

#### 建立参考

适配后的模型从训练集的正常图中提取三种特征，构建统计参考：

| 参考类型 | 特征来源 | 维度 | 建模方法 |
|----------|----------|------|----------|
| **局部 Token 库** | F16 特征图所有像素位置 | ~12000 tokens × 384d | 原始特征向量存储 |
| **全局特征参考** | F32 全局平均池化 | 768d | Ledoit-Wolf 马氏距离 |
| **颜色特征参考** | LAB 颜色空间统计 | 54d | Ledoit-Wolf 马氏距离 |
| **几何特征参考** | Canny 边缘 + 轮廓分析 | 6d | Ledoit-Wolf 马氏距离 |

#### 推理时评分

对每一张待测图片：

1. **局部记忆评分 (memory_local)**：将 F16 每个像素位置的特征与正常 Token 库计算**最近邻欧氏距离**，取 Top-1% 最大距离的均值。异常区域的局部特征会与正常库差异巨大。

2. **全局记忆评分 (memory_global)**：计算 F32 全局特征与正常分布的马氏距离。

3. **颜色评分 (color)**：计算 LAB 颜色统计与正常分布的马氏距离。

4. **几何评分 (geometry)**：计算几何轮廓统计与正常分布的马氏距离。

5. **Z-Score 标准化**：所有分数用正常集上的均值和标准差标准化，使不同分支的分数可比。

#### GPU 加速局部检索

```python
# 使用 ||x-y||² = ||x||² + ||y||² - 2xy 分解避免内存爆炸
# 分块（chunk_size=4096）计算，GPU 矩阵乘法加速
for start in range(0, bank.shape[0], chunk_size):
    distances = token_sq_norm + bank_norm_chunk - 2.0 * matmul(tokens, bank_chunk.T)
    minimum_distance = min(minimum_distance, distances.min(dim=1))
```

---

### 4. 实时检测引擎（`realtime_detection.py`）

#### 单图推理流程

```
输入图像
    │
    ├──→ [骨干网络唯一一次前向]
    │       │
    │       ├──→ local_head → 局部热力图 (24×24)
    │       ├──→ global_head → 全局异常分数
    │       ├──→ component_head → 组件状态
    │       ├──→ geometry_head → 几何特征
    │       └──→ features (F16, F32) → 复用给 normal_reference
    │
    ├──→ [正常参考评分] (复用本次前向特征，不重新跑模型)
    │       ├── memory_local (GPU最近邻)
    │       ├── memory_global (马氏距离)
    │       ├── color (马氏距离)
    │       └── geometry (马氏距离)
    │
    ├──→ [可选 ROI 精修]
    │       从局部热力图选出 Top-2 候选区域
    │       裁剪原图 384×384 重新推理
    │       取最强 ROI 分数与全局分数取 max
    │
    └──→ [最终融合]
            score = w_sup * supervised + w_local * memory_local
                  + w_global * memory_global + w_color * color
                  + w_geometry * geometry
            is_anomaly = score >= threshold
```

**延迟分析**：核心瓶颈在骨干网络前向，正常参考评分复用已有特征不增加额外前向，整体可在毫秒级完成。

**阈值校准**：使用 conformal prediction 方法，在验证集正常样本上按目标 FPR（默认 5%）选取分位点作为阈值：
```python
rank = ceil((n + 1) * (1 - target_fpr)) - 1
threshold = sorted(calibration_scores)[rank]
```

#### 视频推理

- 按帧间隔提取帧，逐帧运行检测器
- 将每帧的全局分数、局部分数、组件分数、几何分数拼接为特征向量
- 送入 **GRU 时序头** 建模帧间状态转移
- 最终分数取帧级最高分数与时序头分数的最大值

---

### 5. 反馈驱动持续优化（`feedback_optimization.py`）

上线后操作员可以通过反馈接口纠正误判，系统自动响应：

#### 即时调整

| 反馈类型 | 操作 |
|----------|------|
| 误报（正常品被判为异常） | 降低阈值（取中位数 + 0.05） |
| 漏报（异常品被判为正常） | 提高阈值（取中位数 - 0.05） |
| 单次调整上限 | ±0.1（防止单条反馈导致剧烈波动） |

#### 周期性重训

- 当反馈累积 ≥ 20 条时触发
- 自动备份当前模型和参考库（时间戳命名）
- 合并原始 100/30 样本 + 新反馈样本重新运行适配流程
- **安全检查**：新模型阈值漂移 > 0.5 则自动回滚
- 异常发生时自动从备份恢复

---

### 6. 合成数据引擎（`synthetic_engine.py`）

在少样本迁移阶段，通过自动数据增强缓解正负样本极度不均衡：

| 合成方式 | 描述 | 模拟的缺陷类型 |
|----------|------|---------------|
| **外观合成** | 随机位置 + 方向绘制暗色线条 | 划痕、裂纹 |
| **颜色合成** | 随机调整色相、饱和度、亮度 | 颜色异常、褪色 |
| **几何合成** | 随机缩放 + 居中裁剪 | 尺寸偏差、变形 |
| **组件缺失** | CV2 inpaint 擦除指定区域 | 缺件（预留接口） |
| **GAN 生成** | 可接入 DFMGAN（StyleGAN2 变种） | 真实感外观缺陷 |

---

## 📁 项目结构

```
AOI/
├── main.py                          # CLI 入口，6 个命令
├── aoi_model.py                     # 多分支模型 + 时序头
├── convnextv2.py                    # ConvNeXt V2 Tiny 骨干
├── config.py                        # AOIConfig 配置类
├── config.json                      # 默认配置
├── example_config.json              # 部署用精简配置示例
├── requirements.txt                 # 依赖
│
├── modules/
│   ├── realtime_detection.py        # 实时检测引擎（单图 + 视频）
│   ├── fewshot_transfer.py          # 两阶段训练（公共预训练 + 少样本迁移）
│   ├── normal_reference.py          # 正常参考库（记忆检索 + 阈值校准）
│   ├── synthetic_engine.py          # 合成数据引擎
│   └── feedback_optimization.py     # 反馈优化（即时调整 + 周期重训）
│
├── utils/
│   ├── image.py                     # 图像加载、归一化、特征提取
│   ├── paths.py                     # 数据集扫描、split 构建
│   └── manifests.py                 # 数据清单处理
│
├── DFMGAN/                          # 预训练 StyleGAN2（可接入生成真实感缺陷）
├── model/                           # 预训练权重存放
└── data/                            # 数据集存放
```

---

## 🚀 使用流程

### 1. 准备数据
```bash
# 按标准结构存放四个公开数据集
data/
├── mvtec_ad/        # MVTec AD
├── mvtec_loco_ad/   # MVTec LOCO AD
├── visa/            # VisA
└── dagm2007/        # DAGM 2007
```

### 2. 构建数据划分
```bash
python main.py --config config.json make-split \
    --target-dataset mvtec_ad \
    --target-category grid \
    --unseen-type bent \
    --normal-budget 100 \
    --anomaly-budget 30
```

### 3. 公共数据预训练
```bash
python main.py --config config.json pretrain-public \
    --split-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent
```

### 4. 目标产线迁移
```bash
python main.py --config config.json adapt \
    --normal-dir data/target/normal \
    --anomaly-dir data/target/anomaly
```

### 5. 实时推理

#### 单张图片
```bash
python main.py --config config.json infer-image \
    --image data/test_image.jpg
```

#### 视频流
```bash
python main.py --config config.json infer-video \
    --video data/production_video.mp4 \
    --frame-stride 5
```

#### 批量评估
```bash
python main.py --config config.json evaluate \
    --normal-dir data/query/normal \
    --seen-dir data/query/seen \
    --unseen-dir data/query/unseen
```

### 6. 反馈修正
```bash
# 操作员纠正误报/漏报
python main.py --config config.json feedback \
    --image data/false_positive.jpg \
    --predicted-label 1 \
    --corrected-label 0 \
    --score 2.35 \
    --note "误报：这是正常品，表面反光导致"

# 周期重训（自动累积到阈值后触发）
python main.py --config config.json feedback-retrain \
    --normal-dir data/target/normal \
    --anomaly-dir data/target/anomaly \
    --validation-normal-dir data/validation/normal
```

---

## ⚙️ 配置说明

关键配置项（`config.json`）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `student_checkpoint` | ConvNeXt V2 Tiny（ImageNet-22K） | 学生骨干初始权重 |
| `component_slots` | 8 | 组件状态头 slot 数 |
| `geometry_dims` | 6 | 几何回归维度 |
| `local_top_ratio` | 0.01 | 局部 Top-k 占比 |
| `head_epochs` / `stage4_epochs` / `stage3_epochs` | 3/3/3 | 各阶段训练轮数 |
| `enable_memory_local` | true | 是否启用 GPU 局部记忆检索 |
| `memory_local_weight` / `memory_global_weight` | 0.35 | 记忆分支融合权重 |
| `target_normal_fpr` | 0.05 | 目标正常样本误报率（Conformal 阈值） |
| `feedback_retrain_min_samples` | 20 | 触发周期重训的最少反馈数 |

---

## 📊 实验结果（预期指标）

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **单图推理延迟** | < 50ms | 单次 ConvNeXt V2 Tiny 前向 |
| **seen AUC** | > 95% | 已见过缺陷类型 |
| **unseen AUC** | > 85% | 未见缺陷类型 |
| **正常误报率** | < 5% | 通过 conformal 校准控制 |
| **少样本数量** | 100 正常 + 30 缺陷 | 新产线适配数据需求 |

---

## 🔗 参考

- [ConvNeXt V2](https://arxiv.org/abs/2301.00808) — 骨干网络
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — 工业缺陷数据集
- [MVTec LOCO AD](https://www.mvtec.com/company/research/datasets/mvtec-loco-ad) — 逻辑/结构缺陷数据集
- [Ledoit-Wolf Covariance](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/covariance/_shrunk_covariance.py) — 高维协方差估计
- [DFMGAN](https://github.com/your-repo/DFMGAN) — 预训练 StyleGAN2 用于真实感缺陷合成
