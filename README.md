# AOI — Automatic Optical Inspection

面向工业产线的实时 AI 质检系统，重点解决以下三个问题：

1. **实时图片与视频异常检测**
2. **少样本或无异常样本条件下的新产线快速启动**
3. **用户反馈驱动的安全持续优化**

系统面向划痕、裂纹、污点、崩边、颜色偏差、尺寸变化、组件缺失、装配异常和工序逻辑异常等多种 AOI 场景。

---

## 1. 项目目标

传统工业质检通常依赖人工目检或固定规则机器视觉系统，存在以下问题：

| 挑战 | 说明 |
|---|---|
| 实时性要求高 | 2500×2500 高分辨率图像需要在产线节拍内完成检测 |
| 缺陷样本稀缺 | 新产品上线时通常只有约 100 张正常图和 30 张异常图 |
| 未见缺陷难以覆盖 | 测试阶段可能出现训练阶段从未出现过的异常类型 |
| 异常类型异质 | 外观、颜色、几何、组件、装配和时序异常不能由单一分数完整描述 |
| 用户反馈更新存在风险 | 直接在线更新容易造成模型漂移和正常模式污染 |

本项目采用 ConvNeXt 多尺度特征、开放集异常训练、目标产线少样本适配、异质异常增强和反馈安全更新，构建完整的 AOI 检测流程。

题目描述：

![Question](Question.png)

---

## 2. 赛题约束

当前方案按照以下部署条件设计：

- 输入支持图片和视频；
- 原始工业图像分辨率可达到约 2500×2500；
- RTX 2060 或以下 GPU 单图推理目标小于 200 ms；
- CPU 单图推理目标小于 2 s；
- 新产线适配阶段允许使用 100 张正常样本和 30 张异常样本；
- 完成适配后，在后续 1000+ 测试样本上冻结模型；
- 测试数据不加入训练集，不更新模型参数；
- 用户反馈只在独立更新阶段使用；
- 更新后的模型必须经过历史回放、验证和回滚检查后再发布。

---

## 3. 项目目录

```
AOI/
│
├── main.py                              # 命令行统一入口（8 个命令）
├── aoi_model.py                         # 多尺度模型、6 个任务头 + 时序头
├── convnextv2.py                        # ConvNeXtV2-Tiny 骨干实现
├── config.py                            # 配置读取、路径解析和验证
├── config.json                          # 默认训练与部署配置
├── example_config.json                  # 部署用精简配置示例
├── requirements.txt                     # Python 依赖
├── __init__.py
├── Question.png                         # 题目描述图
│
├── vis_infer.py                         # 推理可视化脚本（生成热力图对比）
│
├── modules/
│   ├── __init__.py
│   ├── realtime_detection.py            # 实时检测引擎（单图/视频/批量评估）
│   ├── fewshot_transfer.py              # 公共工业预训练 + 目标产线少样本适配
│   ├── normal_reference.py              # 正常参考库（聚类/马氏距离/GPU最近邻）
│   ├── synthetic_engine.py              # 异质异常合成（外观/颜色/几何/缺件）
│   └── feedback_optimization.py         # 用户反馈（即时调阈/周期重训/自动回滚）
│
├── utils/
│   ├── __init__.py
│   ├── image.py                         # 图像加载、归一化、LAB/几何特征提取
│   ├── paths.py                         # 数据集扫描、seen/unseen 划分、JSONL 管理
│   └── manifests.py                     # CSV/JSONL 数据清单读取
│
├── model/                               # 预训练权重目录
│   ├── convnextv2_tiny_22k_384_ema.pt   # ConvNeXtV2-Tiny (ImageNet-22K, 110 MB)
│   ├── convnext_large_22k_1k_224.pth    # ConvNeXt-Large (22K→1K, 755 MB)
│   └── model.txt
│
├── data/                                # 工业数据集
│   ├── mvtec_ad/                        # MVTec AD（bottle/cable/.../grid/.../zipper 共15类）
│   │   ├── bottle/
│   │   ├── cable/
│   │   ├── capsule/
│   │   ├── grid/                        # 当前实验目标类别
│   │   │   ├── train/good/              #  正常训练图
│   │   │   └── test/                    #  测试：good/bent/broken/glue/...
│   │   └── ...
│   └── dagm2007/                        # DAGM 2007（10 类纹理缺陷）
│       ├── Class1/
│       ├── Class2/
│       └── ...
│
├── DFMGAN/                              # StyleGAN2 代码（预留，未集成）
│   ├── train.py
│   ├── generate.py
│   ├── training/
│   └── ...
│
└── aoi_full_workspace/                  # 运行产出（自动生成，可删除重建）
    ├── checkpoints/
    │   ├── industrial_pretrained.pth    # 公共预训练权重
    │   └── industrial_pretrained.history.json  # 训练历史
    │
    ├── splits/
    │   └── mvtec_ad_grid_unseen_bent/   # 实验划分（grid 为目标，bent 为未见）
    │       ├── public_train.jsonl       # MVTec+DAGM 公共训练清单
    │       ├── public_val.jsonl         # 公共验证清单
    │       ├── summary.json             # 划分统计
    │       ├── support/
    │       │   ├── normal/              # ~100 张正常图
    │       │   └── anomaly/             # ~30 张异常图（broken/glue/metal/thread）
    │       ├── query/
    │       │   ├── normal/              # ~21 张测试正常图
    │       │   ├── seen/                # ~15 张已见过缺陷图
    │       │   └── unseen/              # ~12 张未见缺陷图（bent）
    │       ├── query_normal.jsonl
    │       ├── query_seen.jsonl
    │       └── query_unseen.jsonl
    │
    ├── deployment/                      # 最终部署产物
    │   ├── target_model.pth             # 迁移后的完整模型权重
    │   └── normal_reference.pth         # 正常参考库（阈值 + 特征库）
    │
    ├── evaluation/                      # 批量评估结果
    │   ├── metrics.json                 # 汇总指标（AUROC/F1/延迟等）
    │   ├── predictions.csv              # 每张图的预测详情
    │   └── failed_cases.csv             # 预测错误的样本
    │
    ├── feedback/                        # 用户反馈（运行时生成）
    │   ├── feedback.jsonl               # 反馈日志
    │   └── rollback/                    # 重训前的模型备份
    │
    └── logs/                            # 训练日志
```

---

## 4. 使用的数据集

离线工业训练使用以下四个公开数据集：

| 数据集 | 主要内容 | 在本项目中的作用 |
|---|---|---|
| MVTec AD | 15 类工业对象和纹理异常 | 划痕、裂纹、污点、缺口和局部外观缺陷 |
| DAGM 2007 | 10 类工业纹理缺陷 | 细粒度纹理异常和小面积缺陷 |
| VisA | 12 类复杂工业对象 | 外观异常、多组件异常和电子元件异常 |
| MVTec LOCO AD | 5 类结构和逻辑异常 | 缺件、多件、错位、错误组合和逻辑异常 |

数据目录包括：

- `data/mvtec_ad`
- `data/dagm2007`
- `data/visa`
- `data/mvtec_loco_ad`

四个数据集共同用于：

- 公共工业预训练；
- 跨类别异常泛化训练；
- 已见异常与未见异常划分；
- 纹理、颜色、几何和组件异常训练；
- 后续 0/1/2/5/30-shot 开放集 episode 构造。

---

## 5. 系统整体流程

系统分为四个阶段。

### 阶段一：公共工业数据训练

使用 MVTec AD、DAGM 2007、VisA 和 MVTec LOCO AD 进行离线训练，学习通用的工业纹理、局部外观、全局结构和异常判别能力。

训练完成后输出：

- `industrial_pretrained.pth`

### 阶段二：目标产线少样本适配

使用目标产品的：

- 100 张正常样本；
- 30 张真实异常样本；
- 纹理、颜色、几何和组件合成异常。

更新目标产品相关的正常性参数、异常分类头、轻量适配参数、融合权重和检测阈值。

适配完成后输出：

- `target_model.pth`
- `normal_reference.pth`

### 阶段三：冻结检测

完成少样本适配后，在正式测试数据上冻结模型，支持：

- 实时单图检测；
- 视频逐帧检测；
- 异常热力图输出；
- 已见异常和未见异常识别；
- 颜色、几何和组件异常分析。

### 阶段四：用户反馈优化

用户反馈不会直接用于正式测试阶段在线学习，而是在独立更新流程中完成：

- 反馈样本记录；
- 困难正常样本和困难异常样本缓存；
- 阈值平滑调整；
- 轻量参数周期更新；
- 历史样本回放；
- 新旧模型验证；
- 性能下降时自动回滚。

---

## 6. 骨干网络

### 5.1 当前可运行模型

当前代码使用：

- ConvNeXtV2-Tiny-384

输入尺寸为 384×384。

核心多尺度特征包括：

| 特征层 | 特征尺寸 | 主要作用 |
|---|---|---|
| F16 | B×384×24×24 | 局部纹理、小缺陷和未见外观异常 |
| F32 | B×768×12×12 | 全局结构、组件关系和装配异常 |

当前使用 ConvNeXtV2-Tiny 的原因：

- 参数量较小；
- 预训练权重已经成功加载；
- 已完成公共训练、目标迁移和实际延迟测试；
- 适合 RTX 2060 等低算力设备；
- 能为局部检测、全局检测和视频逻辑分支保留计算预算。

### 5.2 C²-FFN 快速骨干（实验模式）

仓库同时提供 `c2_hard_b` 快速模式。该模式保留 ConvNeXtV2-Tiny
的稠密深度卷积、残差路径和完整 `C→4C→C` 通道 FFN，但在 Stage 1–3
中先按局部区域聚合 token，只在 128/32/8 个 centroid 上执行通道 FFN，
再将更新写回原空间；Stage 4 仍保持原始 Dense ConvNeXt。

该模式不改变多尺度接口，仍输出 F4、F8、F16 和 F32。启用方式：

```json
{
  "backbone_mode": "c2_hard_b"
}
```

默认值仍为 `dense`。C²-FFN 当前定位是可切换的实验加速模式，不作为
Dense ConvNeXt 的无损替代；其训练权重应与 Dense 模式分别保存和加载。

在 RTX 3060、2500×2500 单图输入、相同 AMP/channels-last 测试协议下，
严格匹配结果如下：

| 骨干模式 | P95 延迟 | AUROC | 定位 |
|---|---:|---:|---|
| Dense ConvNeXtV2-Tiny | 193.01 ms | 0.6896 | 默认精度模式 |
| C²-FFN Hard-B（128/32/8） | 122.67 ms | 0.6178 | 实验快速模式 |

C²-FFN 将 P95 延迟降低约 36.4%（约 1.57× 加速），但当前 AUROC 下降
约 7.18 个百分点。因此已停止继续修改 C² 结构；后续 AOI 准确率方法默认
在 Dense 骨干上开发，最终系统完成后再统一审计 Dense/C² 的端到端取舍。

### 5.3 ConvNeXt-L 的定位

项目中同时保留公开可用的 ConvNeXt-L 预训练权重：

- `model/convnext_large_22k_1k_224.pth`

ConvNeXt-L 后续可用于：

- 更强的离线工业预训练；
- 教师模型；
- 特征蒸馏；
- 与 ConvNeXtV2-Tiny 进行精度对比；
- 验证更大骨干是否能提升未见异常泛化。

当前实测结果均来自 ConvNeXtV2-Tiny，ConvNeXt-L 尚未完成相同条件下的精度和延迟验证。

---

## 7. F16/F32 多尺度主线

F16 和 F32 是当前方案不可删除的两条核心分支。

### 6.1 F16 局部外观分支

F16 具有较高空间分辨率，主要处理：

- 划痕；
- 裂纹；
- 污点；
- 崩边；
- 小孔洞；
- 局部纹理破坏；
- 局部色差；
- 小面积未见外观异常。

F16 分支输出局部异常热力图，并通过高响应区域聚合得到图像级局部异常分数。

此前实验已经证明，缺少 F16 局部正常性建模时，未见异常检测性能会明显下降，因此 F16 是当前系统中不可替代的局部异常主线。

### 6.2 F32 全局结构分支

F32 具有更大的感受野，主要处理：

- 整体结构异常；
- 大范围异常；
- 组件缺失；
- 组件数量变化；
- 装配错误；
- 全局颜色变化；
- 几何形态变化；
- 逻辑结构异常。

F32 连接全局异常头、组件状态头、几何头和融合模块，用于补充 F16 无法可靠描述的全局结构信息。

---

## 8. 当前模型结构

当前模型以 ConvNeXtV2-Tiny 为共享骨干，并包含以下任务分支：

| 分支 | 主要作用 |
|---|---|
| F16 局部异常头 | 输出局部异常热力图和局部异常分数 |
| F32 全局异常头 | 判断整体图像是否异常 |
| 组件状态头 | 为缺件、错位和组件关系异常提供接口 |
| 几何异常头 | 为尺寸变化、轮廓变化和形态异常提供接口 |
| 域对齐头 | 缓解真实异常和合成异常之间的分布差异 |
| 融合头 | 联合局部、全局、组件、几何和颜色证据 |
| 时序逻辑头 | 为视频状态序列建模提供接口 |

需要注意：

- 组件状态头必须结合组件标签、组件框、组件 Mask 或可控组件异常合成进行训练；
- 几何异常头必须定义明确的几何监督目标；
- 仅定义输出头不能证明模型已经具备完整的缺件或尺寸检测能力。

---

## 9. 当前强基线：F16 局部记忆检索

当前已经跑通的强基线使用 F16 局部 Token 建立目标产品正常参考库。

推理时，测试图像的每个 F16 局部 Token 与正常 Token 库进行最近邻距离比较，并取距离最高的一部分区域作为局部异常分数。

正常参考还包括：

| 参考类型 | 特征来源 | 建模方法 |
|---|---|---|
| F16 局部参考 | 局部 Token | GPU 最近邻检索 |
| F32 全局参考 | 全局池化特征 | 正常均值、协方差和 Mahalanobis 距离 |
| 颜色参考 | LAB 颜色统计 | 正常分布距离 |
| 几何参考 | 边缘和轮廓统计 | 正常分布距离 |

该实验已经证明：

> F16 局部正常性建模是未见局部缺陷检测的关键。

但是，F16 Token 记忆库属于非参数检索方法，需要额外保存目标产品的正常局部特征，因此不作为最终模型的核心算法。

当前版本保留为：

- Memory-bank Baseline

其主要作用是为后续参数化模型提供强基线和消融对照。

---

## 10. 最终局部模型：LNBR

最终计划使用模型内部的可学习正常性参数，替代外部 F16 Token 记忆库。

模块名称：

- Learnable Normal Basis Residual Module
- LNBR
- 可学习正常基残差模块

### 9.1 可学习正常基

LNBR 在模型内部维护固定数量的可训练正常基。

这些正常基：

- 是模型参数；
- 参与反向传播；
- 保存到 `target_model.pth`；
- 参数数量固定；
- 推理时不读取外部局部特征库；
- 用户反馈后可以通过轻量微调更新。

它们用于表示目标产品常见的正常局部模式，例如正常纹理、边缘、孔位、接缝和局部结构。

### 9.2 正常特征重构

F16 局部 Token 根据与正常基的相似度进行软分配，再由正常基重构局部特征。

正常区域通常能够被正常基较好地解释，因此重构残差较低。

异常区域与正常模式不一致，重构残差通常较高。

局部残差可以恢复为 24×24 的异常图，用于定位：

- 划痕；
- 裂纹；
- 污点；
- 崩边；
- 未见局部异常。

### 9.3 正常区域压缩

大量低残差正常 Token 不再全部进入后续全局建模，而是根据正常基聚合为固定数量的正常聚合 Token。

该过程可以：

- 压缩重复正常区域；
- 降低高分辨率特征的全局建模开销；
- 保留不同类型的正常局部模式；
- 避免所有正常位置被等权处理。

### 9.4 高残差 Token 保留

残差最高的少量 Token 被保留为独立疑似异常 Token。

例如，原始 F16 包含 576 个局部 Token，可以压缩为：

- 32 个正常聚合 Token；
- 16 个高残差疑似异常 Token。

该设计借鉴 C²SSM 的簇中心思想，但不照搬完整恢复网络，重点实现：

- 大面积正常区域压缩；
- 高残差异常位置保留；
- 小缺陷不被全局平均；
- 固定长度的轻量后续建模。

### 9.5 LNBR 训练目标

LNBR 的训练目标包括：

- 降低正常样本的局部重构残差；
- 提高异常样本高风险区域的残差；
- 防止多个正常基退化为相同模式；
- 防止所有 Token 只使用少数正常基；
- 使用异常 Mask 对局部残差图进行监督；
- 使用真实异常和合成异常共同训练异常边界。

最终目标是让正常局部区域能够被参数化正常基稳定解释，而未见异常区域产生明显残差。

---

## 11. F16/F32 跨尺度建模

最终模型不只分别输出 F16 和 F32 分数，还会联合：

- F16 正常聚合 Token；
- F16 高残差疑似异常 Token；
- F32 全局结构 Token。

这些 Token 输入轻量跨尺度关系模块，用于：

- 将 F16 局部异常证据传递到 F32 全局结构表示；
- 建模局部缺陷与整体装配结构之间的关系；
- 防止小缺陷被全局特征平均；
- 控制高分辨率全局建模开销；
- 提升组件缺失、局部错位和装配异常检测能力。

跨尺度模块可以采用轻量 Attention、Token Mixer、低秩关系建模或轻量状态空间结构。

---

## 12. F32 全局正常性建模

当前强基线使用 F32 正常均值和协方差计算全局异常距离。

最终模型计划增加参数化全局正常性头，将目标产品的全局正常中心和正常变化范围表示为模型参数。

该分支主要处理：

- 缺件；
- 大范围结构变化；
- 装配异常；
- 全局形态变化；
- 组件组合异常；
- 整体颜色和分布变化。

F32 全局正常性参数最终保存到 `target_model.pth`，不依赖高维外部全局特征库。

---

## 13. 多分支融合

最终融合模块联合以下异常证据：

- F16 局部残差分数；
- F32 全局正常性分数；
- 监督异常分类分数；
- 组件异常分数；
- 几何异常分数；
- 颜色异常分数。

不同分支首先进行尺度标准化，再由轻量融合头输出最终异常分数。

融合头主要通过：

- 100 张目标正常样本；
- 30 张目标异常样本；

进行新产线快速校准。

相比固定手工权重，学习式融合可以根据不同产品的缺陷特点自动调整局部、全局、颜色、几何和组件分支的重要性。

---

## 14. 公共工业训练

### 13.1 当前已实现训练

当前 `pretrain-public` 已经支持在四个公开工业数据集上进行联合训练，主要包含：

- 图像级正常/异常分类；
- F16 局部异常监督；
- F32 全局异常监督；
- 像素级异常区域监督；
- 正常与异常排名约束；
- 真实异常与合成异常域对齐；
- 类别不平衡采样。

### 13.2 开放集少样本 Episode

后续计划增加：

- 0-shot；
- 1-shot；
- 2-shot；
- 5-shot；
- 30-shot。

每个 episode 包含：

- 支持集正常样本；
- 支持集异常样本；
- 查询集正常样本；
- 查询集已见异常；
- 查询集未见异常。

该训练用于模拟新产线在不同异常样本数量下的启动过程。

本项目不采用额外的 MAML 或 Reptile 模型，也不生成独立的 `meta_model.pth`。

计划训练流程为：

1. 普通公共工业预训练；
2. 开放集少样本 episode 训练；
3. 目标产线 100/30 适配。

最终仍然输出统一的工业预训练模型。

---

## 15. 目标产线 100/30 少样本适配

### 14.1 100 张正常样本

100 张正常样本用于：

- 适配 F16 正常基；
- 适配局部轻量参数；
- 降低正常样本的 F16 重构残差；
- 适配 F32 全局正常性参数；
- 建立颜色统计；
- 建立几何统计；
- 校准正常分数分布；
- 确定最终异常阈值。

最终 LNBR 版本不再使用 100 张正常图建立 F16 原始 Token 记忆库。

### 14.2 30 张异常样本

30 张异常样本用于：

- 训练监督异常分类头；
- 训练 F16 异常残差边界；
- 训练 F32 全局异常边界；
- 训练跨尺度融合头；
- 训练轻量 Adapter；
- 校准正常和异常之间的决策边界。

### 14.3 渐进式解冻

当前目标适配采用渐进式训练：

1. 先训练任务头；
2. 再解冻 Stage 4；
3. 再解冻 Stage 3 后部；
4. 最后仅使用真实样本进行收尾校准。

每个阶段均使用验证集 AUC 保存最佳模型，以降低少样本过拟合风险。

---

## 16. 异质异常增强与 GAN

`synthetic_engine.py` 用于生成四类异质异常：

| 类型 | 示例 | 主要作用 |
|---|---|---|
| 纹理异常 | 划痕、裂纹、污点、孔洞 | 强化 F16 局部外观异常检测 |
| 颜色异常 | 色偏、褪色、亮度变化 | 强化局部颜色和整体颜色异常检测 |
| 几何异常 | 缩放、形变、错位、旋转 | 强化 F16/F32 和几何分支 |
| 语义或组件异常 | 缺件、替换、多件、错误组合 | 强化 F32 和组件分支 |

合成数据需要统一返回：

- 合成异常图；
- 异常区域 Mask；
- 异常类型；
- 对应任务类型。

GAN 的定位是：

> 扩大 30 张真实异常无法覆盖的缺陷空间，而不是替代真实异常样本。

当前规则合成已经接入基础训练流程。

DFM/StyleGAN 类真实感异常生成仍属于后续扩展，在未完成真实实验前不作为已验证模块。

---

## 17. 实时单图检测

### 16.1 当前 Memory-bank Baseline

当前基线流程为：

1. 输入图像经过 ConvNeXtV2-Tiny 单次前向；
2. F16 输出局部异常图；
3. F32 输出全局异常分数；
4. 复用 F16/F32 特征进行正常参考评分；
5. 计算局部、全局、颜色和几何异常分数；
6. 进行多分支融合；
7. 根据校准阈值输出异常判断。

当前 ROI 精修分支默认关闭，因为现有实验没有显示有效收益。

### 16.2 最终 LNBR 模型

最终模型流程为：

1. 输入图像经过 ConvNeXt 单次前向；
2. F16 进入 LNBR；
3. 输出局部残差图、正常聚合 Token 和高残差 Token；
4. F32 输出全局异常、组件和几何状态；
5. F16/F32 Token 进入跨尺度关系模块；
6. 多分支融合头输出最终异常分数；
7. 同时输出局部异常位置。

最终版推理不再访问外部 F16 局部 Token 库。

---

## 18. 阈值校准

系统使用独立正常校准集确定最终异常阈值。

当前采用有限样本 conformal 分位点进行正常阈值估计。

需要注意：

- `target_normal_fpr=0.05` 是目标值；
- 该配置不代表测试集实际 FPR 一定等于 5%；
- 正常校准样本较少时，阈值估计存在较大方差；
- 必须在独立目标产品测试集上报告实际正常误报率。

---

## 19. 视频检测

视频检测分为两个阶段。

### 18.1 当前可实现能力

当前支持：

- 视频逐帧读取；
- 帧级异常检测；
- 连续异常帧过滤；
- 帧级组件状态记录；
- 视频异常分数汇总；
- 基于人工规则的状态机接口。

### 18.2 后续时序训练

具有真实产线视频序列后，可进一步训练：

- GRU；
- TCN；
- 轻量时序 Transformer；
- 显式状态转移模型。

需要的视频标签包括：

- 正常工序序列；
- 缺步序列；
- 错序序列；
- 重复步骤序列；
- 异常持续时间；
- 组件状态变化。

在没有真实时序监督数据前，GRU 仅作为模型接口，不能宣称已经完成可靠的缺步、错序和重复步骤识别。

---

## 20. 用户反馈驱动优化

用户反馈分为：

- 误报：正常产品被判为异常；
- 漏报：异常产品被判为正常。

### 19.1 阈值调整方向

异常分数越高表示越异常，因此：

- 误报增加时，阈值应适当提高；
- 漏报增加时，阈值应适当降低。

阈值采用平滑更新，并限制单次变化幅度，避免少量反馈导致决策边界剧烈变化。

### 19.2 反馈样本池

反馈样本分别进入：

- 困难正常样本池；
- 困难异常样本池。

单条反馈不直接更新模型参数或正常基，避免模型污染。

### 19.3 周期安全更新

反馈积累到一定数量后执行：

1. 备份当前模型；
2. 合并原始 100/30 数据和反馈数据；
3. 更新 Adapter、LNBR、异常头和融合头；
4. 在历史验证集上回放；
5. 检查 AUROC、F1、正常误报率和异常召回率；
6. 验证通过后发布；
7. 性能下降时自动回滚。

安全检查至少包括：

- 正常误报率；
- 异常召回率；
- F1；
- 已见异常 AUROC；
- 未见异常 AUROC；
- 推理延迟。

---

## 21. 当前真实实验结果

当前实验设置：

| 项目 | 设置 |
|---|---|
| 目标数据集 | MVTec AD |
| 目标类别 | grid |
| 完全未见异常类型 | bent |
| Query 样本数 | 48 |

当前最强结果来自：

- ConvNeXtV2-Tiny；
- F16 GPU 局部 Token 记忆检索；
- F32 全局统计；
- 颜色和几何统计。

实验结果：

| 指标 | 实测结果 |
|---|---:|
| Overall AUROC | 0.9718 |
| Seen AUROC | 0.9873 |
| Unseen AUROC | 0.9524 |
| Accuracy | 0.8958 |
| Precision | 0.8667 |
| Recall | 0.9630 |
| F1 | 0.9123 |
| Normal FPR | 0.1905 |
| TN | 17 |
| FP | 4 |
| FN | 1 |
| TP | 26 |
| Mean Latency | 101.7 ms |
| P95 Latency | 113.3 ms |

关闭 F16 局部正常建模后的结果：

| 指标 | 实测结果 |
|---|---:|
| Overall AUROC | 约 0.7354 |
| Unseen AUROC | 约 0.5992 |
| Mean Latency | 约 97.8 ms |

该实验说明：

> F16 局部正常性建模对未见缺陷检测起关键作用。

但当前高精度依赖非参数 Memory-bank，因此下一阶段需要使用 LNBR 替代，并与当前强基线进行公平对比。

当前延迟仅在现有服务器和 384×384 输入上验证，尚未完成：

- RTX 2060 正式测速；
- CPU 正式测速；
- 2500×2500 原图推理验证；
- 多类别重复实验；
- 多数据集重复实验。

---

## 22. 使用流程

### 23.1 构建目标划分

使用 `make-split` 指定：

- 目标数据集；
- 目标类别；
- 完全未见异常类型；
- 正常样本预算；
- 异常样本预算。

当前测试划分为：

- 目标数据集：MVTec AD；
- 目标类别：grid；
- 未见异常类型：bent；
- 正常样本预算：100；
- 异常样本预算：30。

划分结果保存在：

- `aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent`

### 23.2 公共工业训练

使用 `pretrain-public` 读取划分目录中的公共训练清单，完成四个公开数据集联合训练。

输出模型：

- `aoi_full_workspace/checkpoints/industrial_pretrained.pth`

### 23.3 目标产线适配

使用 `adapt` 指定：

- 100 张正常样本目录；
- 30 张异常样本目录；
- 是否启用合成异常。

输出：

- `aoi_full_workspace/deployment/target_model.pth`
- `aoi_full_workspace/deployment/normal_reference.pth`

### 23.4 批量评估

使用 `evaluate` 指定：

- 正常查询集；
- 已见异常查询集；
- 未见异常查询集。

输出包括：

- Overall AUROC；
- Seen AUROC；
- Unseen AUROC；
- Accuracy；
- Precision；
- Recall；
- F1；
- Normal FPR；
- TP、FP、TN、FN；
- 平均延迟；
- P95 延迟。

### 23.5 图片和视频推理

使用：

- `infer-image` 完成单张图片推理；
- `infer-video` 完成视频逐帧推理。

### 23.6 用户反馈

使用：

- `feedback` 记录误报和漏报；
- `feedback-retrain` 执行反馈样本安全重训。

---

## 23. 关键配置

### 24.1 基础配置

| 配置项 | 作用 |
|---|---|
| `student_checkpoint` | 当前 ConvNeXtV2-Tiny 初始权重 |
| `global_size` | 模型全图输入尺寸 |
| `component_slots` | 组件状态输出数量 |
| `geometry_dims` | 几何输出维度 |
| `local_top_ratio` | 局部异常高响应区域比例 |
| `head_epochs` | 任务头训练轮数 |
| `stage4_epochs` | Stage 4 解冻训练轮数 |
| `stage3_epochs` | Stage 3 解冻训练轮数 |
| `target_normal_fpr` | 正常校准目标误报率 |
| `feedback_retrain_min_samples` | 触发反馈重训的最少样本数 |
| `enable_roi_refinement` | 是否启用 ROI 精修 |

### 24.2 Memory-bank Baseline 配置

| 配置项 | 作用 |
|---|---|
| `enable_memory_local` | 是否启用 F16 局部 Token 检索 |
| `memory_local_weight` | 局部记忆分数权重 |
| `memory_global_weight` | F32 全局统计分数权重 |
| `local_bank_chunk_size` | GPU 最近邻检索分块大小 |

### 24.3 LNBR 计划配置

| 配置项 | 建议值 | 作用 |
|---|---:|---|
| `normal_basis_count` | 32 | F16 可学习正常基数量 |
| `abnormal_token_count` | 16 | 保留的高残差 Token 数量 |
| `normal_basis_temperature` | 0.1 | 正常基软分配温度 |
| `normal_loss_weight` | 1.0 | 正常残差损失权重 |
| `basis_diversity_weight` | 0.05 | 正常基多样性约束权重 |
| `basis_balance_weight` | 0.05 | 正常基使用均衡约束权重 |
| `residual_margin_weight` | 1.0 | 异常残差间隔损失权重 |

---

## 24. 当前开发状态

### 25.1 已完成

- 四个公开工业数据集扫描；
- 目标类别排除式公共训练划分；
- 100 正常 + 30 异常支持集构建；
- seen/unseen query 划分；
- ConvNeXtV2-Tiny 权重加载；
- 公共工业训练；
- 渐进式目标适配；
- F16 局部异常头；
- F32 全局异常头；
- GPU 局部 Token 记忆检索；
- 正常阈值校准；
- 单图推理；
- 视频逐帧推理接口；
- 批量评估；
- 反馈记录和备份框架；
- 2500×2500 Dense ConvNeXt 延迟验证；
- C²-FFN Hard-B（Stage 1–3 使用 128/32/8 centroids，Stage 4 Dense）；
- Dense/C² 严格匹配速度与 AUROC 对照；
- C² 模式与 Dense 模式统一 F4/F8/F16/F32 输出接口。

### 25.2 正在重构

- 使用 LNBR 替代最终版 F16 局部 Token 记忆库；
- F16 正常 Token 压缩；
- F16 高残差 Token 保留；
- F16/F32 跨尺度关系建模；
- 参数化 F32 全局正常性头；
- 学习式多分支融合。

### 25.3 待完成

- 0/1/2/5/30-shot 开放集 episode 训练；
- 组件头真实监督；
- 几何头真实监督；
- 四类异质异常统一 Mask 输出；
- DFM/GAN 真实感缺陷生成；
- 真实产线视频序列训练；
- RTX 2060 延迟验证；
- CPU 延迟验证；
- 多目标类别重复实验；
- LNBR 与 Memory-bank 公平消融实验；
- 完整 AOI 系统完成后的 Dense/C² 端到端速度—精度复测。

---

## 25. 下一阶段实验

LNBR 完成后，首先在相同 `grid/bent` 划分上与当前 Memory-bank baseline 比较：

| 指标 | Memory-bank | LNBR |
|---|---:|---:|
| Overall AUROC | 0.9718 | 待测试 |
| Seen AUROC | 0.9873 | 待测试 |
| Unseen AUROC | 0.9524 | 待测试 |
| F1 | 0.9123 | 待测试 |
| Normal FPR | 0.1905 | 待测试 |
| Mean Latency | 101.7 ms | 待测试 |
| P95 Latency | 113.3 ms | 待测试 |
| 外部局部特征库 | 需要 | 不需要 |
| 参数量 | 模型参数加 Token 库 | 固定模型参数 |
| 反馈更新 | 更新特征库 | 更新轻量模型参数 |

随后扩展到：

- MVTec AD / grid；
- MVTec AD / screw；
- VisA / PCB 或 capsules；
- MVTec LOCO AD / breakfast_box。

分别验证：

- 局部外观异常；
- 小目标异常；
- 组件异常；
- 结构异常；
- 未见异常泛化；
- 推理速度；
- 反馈更新稳定性。

---

## 26. 最终方案定位

最终目标不是单一异常分类器，也不是单纯的最近邻记忆检索系统，而是：

- ConvNeXt 多尺度工业视觉骨干；
- F16 可学习正常基残差；
- 正常区域 Token 压缩；
- 高残差疑似异常 Token 保留；
- F32 全局结构与装配建模；
- 四类异质异常增强；
- 100/30 少样本快速适配；
- 视频状态检测；
- 用户反馈安全更新。

核心主线为：

> F16 局部正常性建模  
> + F32 全局结构建模  
> + 开放集少样本迁移  
> + 用户反馈闭环

---

## 27. 参考资料

- [ConvNeXt V2](https://arxiv.org/abs/2301.00808)
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [MVTec LOCO AD](https://www.mvtec.com/company/research/datasets/mvtec-loco-ad)
- [VisA](https://github.com/amazon-science/spot-diff)
- [DAGM 2007](https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection)
- [Ledoit-Wolf Covariance](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html)

---

## 28. 说明

当前 Memory-bank 结果是已经完成并验证的强基线。

以下模块仍处于设计或开发阶段：

- LNBR；
- F16/F32 跨尺度关系建模；
- 参数化 F32 全局正常性；
- 0/1/2/5/30-shot episode；
- 真实视频时序训练；
- 完整反馈参数更新；
- DFM/GAN 真实感异常生成。

README 中对这些模块的描述表示项目设计目标，不代表相关实验已经全部完成。

---

## 29. 完整命令参考

以下是从零开始运行训练到推理的完整命令流程。

### 30.1 环境准备

```bash
# 安装依赖
pip install torch>=2.1 torchvision>=0.16 "numpy<2" pandas pillow opencv-python-headless scikit-learn tqdm
```

### 30.2 数据划分

```bash
# 构建 public/support/query 划分
# --target-dataset: 目标产线数据集 (mvtec_ad / dagm2007 / visa / mvtec_loco_ad)
# --target-category: 目标类别 (如 grid, bottle, screw)
# --unseen-type: 完全未见过的缺陷类型 (如 bent, broken_large)
python main.py --config config.json make-split \
    --target-dataset mvtec_ad \
    --target-category grid \
    --unseen-type bent \
    --normal-budget 100 \
    --anomaly-budget 30
```

### 30.3 公共工业预训练

```bash
# 在四个公开数据集上联合训练，学习通用工业异常特征
# --split-dir: 指向 make-split 的输出目录
python main.py --config config.json pretrain-public \
    --split-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent

# 可选：指定训练轮数和每轮步数
python main.py --config config.json pretrain-public \
    --split-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent \
    --epochs 10 \
    --steps-per-epoch 500
```

### 30.4 目标产线少样本迁移

```bash
# 用 100 张正常 + 30 张异常适配新产品线
# 输出 target_model.pth 和 normal_reference.pth
python main.py --config config.json adapt \
    --normal-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/support/normal \
    --anomaly-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/support/anomaly

# 禁用合成数据增强（可选）
python main.py --config config.json adapt \
    --normal-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/support/normal \
    --anomaly-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/support/anomaly \
    --disable-synthetic
```

### 30.5 批量评估

```bash
# 在 query 集上评估 seen/unseen 异常检测精度
python main.py --config config.json evaluate \
    --normal-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/query/normal \
    --seen-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/query/seen \
    --unseen-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/query/unseen
```

### 30.6 单张图片推理

```bash
# 正常品
python main.py --config config.json infer-image \
    --image data/mvtec_ad/grid/test/good/000.png

# 见过的缺陷类型
python main.py --config config.json infer-image \
    --image data/mvtec_ad/grid/test/broken/000.png

# 未见过的缺陷类型
python main.py --config config.json infer-image \
    --image data/mvtec_ad/grid/test/bent/000.png
```

### 30.7 视频推理

```bash
python main.py --config config.json infer-video \
    --video /path/to/video.mp4 \
    --frame-stride 5
```

### 30.8 可视化推理结果

```bash
# 单张图片可视化
python vis_infer.py \
    --image data/mvtec_ad/grid/test/bent/000.png

# 按类别批量可视化
python vis_infer.py \
    --normal-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/query/normal \
    --seen-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/query/seen \
    --unseen-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/query/unseen \
    --save-dir vis_results \
    --max-per-type 6
```

### 30.9 用户反馈

```bash
# 记录误报（模型判为异常，实际是正常）
python main.py --config config.json feedback \
    --image /path/to/false_positive.png \
    --predicted-label 1 \
    --corrected-label 0 \
    --score 2.35 \
    --defect-type false_positive \
    --note "表面反光导致误判"

# 记录漏报（模型判为正常，实际是异常）
python main.py --config config.json feedback \
    --image /path/to/false_negative.png \
    --predicted-label 0 \
    --corrected-label 1 \
    --score 0.50 \
    --defect-type missing_defect \
    --note "细小裂纹未检出"

# 周期重训（累积反馈满 20 条后触发）
python main.py --config config.json feedback-retrain \
    --normal-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/support/normal \
    --anomaly-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/support/anomaly \
    --validation-normal-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/query/normal
```

### 30.10 完整从零开始脚本

```bash
#!/bin/bash
# 从零到评估的完整流程

# Step 1: 数据划分
python main.py --config config.json make-split \
    --target-dataset mvtec_ad \
    --target-category grid \
    --unseen-type bent

# Step 2: 公共预训练
python main.py --config config.json pretrain-public \
    --split-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent

# Step 3: 目标迁移
python main.py --config config.json adapt \
    --normal-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/support/normal \
    --anomaly-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/support/anomaly

# Step 4: 评估
python main.py --config config.json evaluate \
    --normal-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/query/normal \
    --seen-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/query/seen \
    --unseen-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/query/unseen

# Step 5: 可视化
python vis_infer.py \
    --normal-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/query/normal \
    --seen-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/query/seen \
    --unseen-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/query/unseen \
    --save-dir vis_results \
    --max-per-type 6
```
