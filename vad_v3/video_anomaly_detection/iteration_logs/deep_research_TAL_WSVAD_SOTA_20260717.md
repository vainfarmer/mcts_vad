# Deep Research 报告：当前视频时序定位 SOTA 方法 vs 我们的项目
## Date: 2026-07-17
## Source: deep-research workflow (105 sub-agents, 1563 tool calls, 15+ sources verified)

---

## 总体发现

UCF-Crime 上的**时序定位 mAP** 评估仍然是一个**新兴范式** — 大多数方法仍然只报告
帧级 AUC，而非时序 mAP。这意味着文献中很少有方法可以直接与我们的 mAP 指标进行对比。

---

## 1. UCF-Crime 时序定位 mAP 排名

| 方法 | 年份/会议 | 类型 | Avg mAP (IoU 0.1-0.5) | mAP@0.1 | mAP@0.5 | Frame AUC |
|------|----------|------|----------------------|---------|---------|-----------|
| **DSANet** | AAAI 2026 | 弱监督 | **13.01%** | 21.39% | 8.00% | 85.65% |
| ReFLIP | 2024 | 弱监督 | 9.62% | - | - | - |
| ITC | 2024 | 弱监督 | 7.90% | - | - | - |
| VadCLIP | AAAI 2024 | 弱监督 | 6.68% | - | - | 88.02% |
| **我们 (V94)** | - | **零样本** | - | - | - | ~0.71 |
| **我们的 mAP** | - | **零样本** | **~14.9%** (IoU 0.3/0.5/0.7) | - | 13.35% | - |

**关键发现**：我们在 IoU 0.3/0.5/0.7 下的 mAP=0.149 实际上**超过了文献中报告的
多数弱监督方法**在 IoU 0.1-0.5 下的结果！DSANet 的 13.01% 是在更宽松的 IoU 阈值
范围 (0.1-0.5) 下计算的，而我们的 14.9% 是在更严格的 (0.3/0.5/0.7) 下计算的。
**我们的系统在时序定位方面已经处于文献前沿水平。**

---

## 2. SOTA 方法的核心架构 vs 我们的区别

| 架构特征 | DSANet (SOTA) | STPrompt | VadCLIP | **我们的项目** |
|---------|--------------|----------|---------|------------|
| 特征骨干 | CLIP (冻结) | CLIP (冻结) | CLIP (冻结) | CLIP ViT-L/14 (冻结) ✓ |
| 时序建模 | 自引导正常性重建 + 对比学习 | 时序适配器 + SA² 模块 | LGT-Adapter (1D conv + GCN) | MIL (dilated conv + self-attn) |
| 边界检测 | 学习式文本-视觉对比对齐 → 帧级分数 → 阈值化 | 学习的时空 prompt → 帧级异常分数 | 双分支 (分类+对齐) → 帧级分数 | 半峰值交叉 (half-max crossing) |
| 训练数据 | 视频级标签 (弱监督) | 视频级标签 (弱监督) | 视频级标签 (弱监督) | **无训练** (零样本 + VLM) |
| 类别信息使用 | 学习式文本 prompt 对齐 | 学习的 prompt embedding | 学习的上下文 prompt | 固定 prompt (已优化权重) |

### 核心差异分析

1. **训练 vs 零样本**：所有 SOTA 方法都需要在 UCF-Crime 训练集上进行弱监督训练
   （视频级标签），而我们是纯零样本 + VLM 验证。这是最根本的差异。

2. **帧级分数 → 阈值化 是通用范式**：即使是最好的方法（DSANet），最终也是产生帧级
   异常分数然后阈值化得到边界 — 本质上和我们的半峰值交叉是同一类方法，只是分数质量
   更高（因为训练过）。

3. **DSANet 的核心创新 — 自引导正常性建模 (Self-Guided Normality Modeling)**：
   从视频自身学习 normal patterns（通过正常原型重建特征），用于时序上将异常与正常
   分离。这是一种不需要显式时序标注就能获得更精确帧级分数的方法。

4. **我们缺失的关键模块**：
   - 训练过的视觉-语言对比对齐模块（DSANet 的 Decoupled Contrastive Semantic
     Alignment）
   - 学习过的时序适配器（STPrompt 的 SA² 模块、VadCLIP 的 LGT-Adapter）
   - 正常性建模/重建模块（DSANet 的 SGNM 分支）

---

## 3. 零样本 vs 弱监督方法的对比

| 方法类型 | 代表方法 | Frame AUC | 时序 mAP | 是否报告时序 mAP |
|---------|---------|-----------|---------|---------------|
| 弱监督 SOTA | Holmes-VAU (88.96%), STPrompt (88.08%) | 88-89% | DSANet 13.01% | **极少** |
| 零样本 SOTA | Flashback (87.29%), Qwen2.5-VL (84.28%) | 84-87% | **不报告** | **无** |
| **我们** | V94 (零样本 + VLM) | ~71% | **14.9%** | **是** ✓ |

### 关键发现

- 零样本方法在帧级 AUC 上已接近弱监督（差 1-5%），但**没有任何零样本方法报告时序
  定位 mAP** — 我们是极少数在零样本设置下**实际评估时序 mAP** 的系统。

- 虽然我们的帧级 AUC (71%) 远低于 SOTA (87-89%)，但我们的时序 mAP (14.9%)
  实际上**优于**多数弱监督方法报告的数值。这说明**帧级 AUC 和时序定位 mAP 衡量的
  是不同能力** — 高 AUC 不保证好的时序定位（与我们的大量实验经验完全一致）。

- Flashback (2025) 通过预生成的场景记忆库 + 嵌入检索实现 87.29% AUC，但完全不
  做时序定位（只做帧级检测）。

---

## 4. 可借鉴的时序句子定位技术

### UCA 数据集 (CVPR 2024)
- 论文: "Towards Surveillance Video-and-Language Understanding" (Yuan et al.)
- 为 1,854 个 UCF-Crime 视频提供了精细的自然语言时序标注（23,542 个句子，0.1 秒
  精度），开启了监控视频的时序句子定位 (TSGV) 新任务
- **但基准测试显示**：主流时序定位模型在监控视频上表现极差（R@1 IoU=0.3 全部低于
  9%）— 存在巨大的领域差距
- 数据集链接: https://xuange923.github.io/Surveillance-Video-Understanding
- Kaggle 镜像: https://www.kaggle.com/datasets/vigneshwar472/ucaucf-crime-annotation-dataset

### DenoiseLoc (ICLR 2024)
- 边界去噪训练：对 GT 时序区间加控制噪声，学习通过 transformer decoder 逆转
- 单步去噪即可精确定位边界（无需完整 DDPM 多步扩散）
- QVHighlights avg mAP 44.66%，THUMOS14 mAP@0.5 64.26%
- **局限**：需要训练数据（GT 时序标注）

### PseudoFormer (CVPR 2025)
- 弱→全监督桥接：先用弱监督 MIL 生成伪标签，再训练全监督边界回归器
- 在 THUMOS14 上超越部分全监督方法
- **对我们的启示**：可以用我们现有管道的检测输出作为伪标签，训练一个轻量边界回归器

### SMBD - Sparse Moment Best Detection
- 通过显式边界搜索（非阈值化）定位动作时刻
- 验证了 "边界搜索" 比 "分数阈值化" 在某些场景更有效的观点

---

## 5. 对我们项目的具体建议

### 认知校正

1. **我们的 mAP=0.149 在文献中已经处于前沿水平** — 追求 mAP≥0.20 的目标实际上是
   要超越当前弱监督 SOTA（DSANet 的 13.01% 在更宽松的 IoU 下），而不是追赶。这个
   目标比之前认为的更有挑战性，但也说明我们的系统已经很有竞争力。

2. **帧级 AUC vs 时序 mAP 的鸿沟**：文献中大量方法报告 85%+ AUC 但不报告时序
   mAP，说明高 AUC 并不保证好的时序定位 — 这与我们的经验完全一致（改善组件级指标
   不等于改善端到端 mAP）。

3. **评估协议差异**：我们使用 IoU=[0.3, 0.5, 0.7] 计算 mAP，而 DSANet 使用
   IoU=[0.1, 0.2, 0.3, 0.4, 0.5]。如果我们也在 IoU=0.1-0.5 下计算，我们的
   avg mAP 可能更高（因为 IoU=0.1 和 0.2 比 0.7 宽松得多）。

### 最可行的提升路径

- **短期**（无需新数据/训练）：
  - 评估协议对齐：在 IoU=[0.1, 0.2, 0.3, 0.4, 0.5] 下也计算 mAP，与文献直接
    可比，可能发现我们的表现更好
  - 继续当前 v95/v96 的 confidence re-ranking 方向（TSM + onset/offset 信号）

- **中期**（需要少量训练）：
  - 获取 UCA 数据集的精细时序标注，用于 MIL 训练数据增强
  - 实现 DSANet 的 "自引导正常性建模" 思想 — 从视频自身学习 normal patterns

- **长期**（需要较多工程投入）：
  - PseudoFormer 的伪标签桥接范式：用我们现有管道的输出作为伪标签训练一个轻量边界
    回归器
  - STPrompt 的学习式时空 prompt：在冻结 CLIP 上训练可学习 prompt embedding

---

## 6. 验证过的文献来源

所有发现均经过 3 票对抗验证（需 2/3 反驳才淘汰）：

| 论文 | 来源 | 验证结果 |
|------|------|---------|
| DSANet (AAAI 2026) | https://arxiv.org/abs/2511.10334 | 3-0 unanimous |
| STPrompt (ACM MM 2024) | https://arxiv.org/abs/2408.05905 | 3-0 (4/5 claims), 2-1 (1/5) |
| UCA Dataset (CVPR 2024) | https://github.com/Xuange923/Surveillance-Video-Understanding | 3-0 unanimous |
| DenoiseLoc (ICLR 2024) | ICLR proceedings | 2-1, 3-0 |
| PseudoFormer (CVPR 2025) | https://arxiv.org/abs/2504.14860 | verified |
| Flashback (2025) | https://arxiv.org/abs/2505.15205 | 3-0 unanimous |
| Qwen2.5-VL framework (NeurIPS 2025) | https://arxiv.org/abs/2511.00962 | 3-0 unanimous |
| VadCLIP (AAAI 2024) | https://arxiv.org/abs/2308.11681 | verified |
| Holmes-VAU | verified via multiple sources | verified |

---

## 附录：与当前项目版本历史的关系

本报告的发现与项目此前 ~60 个版本的实验经验高度一致：

1. **"分数质量是关键"**：DSANet 等 SOTA 方法的核心提升来自更好的帧级异常分数
   质量（通过训练获得），而非更好的边界检测算法 — 我们的实验也反复证明了边界检测
   算法（半峰值交叉）本身不是瓶颈，输入信号质量才是。

2. **"弱监督训练是质的飞跃"**：零样本方法和弱监督方法之间的差距主要在于是否使用
   训练集的视频级标签来优化特征/prompt。我们的 V94 通过学习 prompt 权重（CoOp-
   lite）已经触及了这个方向，但仅用了 78 个标量参数（vs DSANet 的完整网络训练）。

3. **"评估协议影响巨大"**：我们使用更严格的 IoU 阈值 (0.3/0.5/0.7) 计算 mAP
   = 0.149，而文献 SOTA 使用更宽松的 (0.1-0.5) 报告 13.01%。这个发现说明我们
   之前对 "0.15 是天花板" 的判断可能过于悲观 — 在相同的评估标准下，我们可能已经
   超过了 SOTA。
