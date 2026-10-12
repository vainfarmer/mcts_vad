# VAD 时间定位 mAP > 50% 深度优化方案

## 总体诊断

### 当前性能基线
| 指标 | 当前值(v75/v76) | 历史最佳(v51) | 目标 |
|------|----------------|--------------|------|
| Temporal mAP | 0.1388 | 0.1492 | **>0.50** |
| AP@0.5 | 0.1283 | 0.1213 | >0.35 |
| Frame AUC-ROC | 0.6714 | 0.7220 | >0.80 |
| Video Accuracy | 92.19% | ~92% | >95% |

### 核心结论：0.14 是当前架构的结构性天花板

**关键证据**（来自 `iteration_logs/summary_loops4-7_20260709.md`）：
- 20+版本(v50-v78)跨7种架构范式（MCTS→CLIP半峰值→MIL融合→特征重中心→梯度精修→LogOIC→比较式VLM），全部收敛到 0.14 ±0.03
- ±0.03 已被证明是纯 VLM API 随机性（v51 vs v69 参数相同，mAP差0.008）
- **推论**：任何不改变基础范式的增量改进，最多达到 ~0.17-0.18

### 突破天花板的三大瓶颈

1. **28.8% 候选位置完全错误** — 低CLIP类别(Shoplifting/Stealing/Arson/Burglary) CLIP曲线平坦(cv<0.03)，零样本无区分力
2. **39% 检测边界精度不足(IoU 0.1-0.5)** — 单信号CLIP曲线固有宽度限制半峰值交叉精度
3. **MIL时序判别力丧失** — v1 45%视频collapsed，v3/v4 75-83% collapsed，当前对边界检测无贡献

### 达到50%需要的根本性变化

参考学术文献指出的可行路径：
- **VadCLIP (AAAI 2024)**：在冻结CLIP特征上训练时序适配器(LGT-Adapter)，弱监督视频异常检测
- **PE-MIL (CVPR 2024)**：提示增强MIL，边界感知片段检测
- **MGFN (CVPR 2023)**：多粒度特征融合网络，magnitude-contrastive loss
- **GS-MoE (ICCV 2025)**：时序高斯溅射混合专家，AUC 91.58%
- **AnomalyCLIP (ICLR 2024)**：特征重中心化增强异常区分度
- **FreeZAD (arXiv 2025)**：LogOIC无训练时序定位

---

## 实施方案（5阶段，按依赖序）

### Phase 0：文献深度调研（1周）

**目标**：使用 academic-research-skills 系统搜索最新 WTAL/VAD 文献，确定最适合本项目的技术路线

**Step 0.1**：搜索 2023-2025 弱监督时序异常定位 (W-TAL for VAD) 论文
- 搜索关键词：Weakly-supervised Temporal Action Localization, Video Anomaly Detection, CLIP temporal adapter
- 重点关注：在UCF-Crime/XD-Violence数据集上的mAP结果
- 文件参考：`/Users/zzl/PycharmProjects/vad/academic-research-skills/deep-research/SKILL.md`

**Step 0.2**：搜索 MIL 训练稳定性和退化解避免技术
- 搜索关键词：MIL degenerate solution, temporal score collapse prevention, range regularization
- 重点关注：RTFM后续改进、contrastive MIL训练

**Step 0.3**：搜索多模态时序定位技术
- 搜索关键词：multimodal temporal grounding, optical flow for boundary detection, motion-guided localization

**预期产出**：技术路线选型报告，确定2-3个核心技术方向

---

### Phase 1：MIL 训练修复 + 时序判别力恢复（2-3周）

**核心思路**：MIL是突破天花板的关键杠杆——v1的45% collapse率意味着一半视频完全没有时序信号。修复MIL就能解锁其作为时序定位主信号（而非常数偏移）的潜力。

**Step 1.1**：实现对抗性范围损失（Range Adversarial Loss）
- 文件：`/Users/zzl/PycharmProjects/vad/video_anomaly_detection/mil_model.py` (L82-105)
- 改动：在 `MILLoss.forward()` 中新增范围正则项
```python
# 在 MILLoss 中添加
range_loss = -torch.log(torch.std(abn_scores, dim=1) + 1e-8).mean()
total_loss = existing_loss + lam_range * range_loss  # lam_range=0.05
```
- 原理：直接惩罚per-video异常分数方差为零的退化解，迫使模型学习时序内判别
- 监控指标：per-video score range (目标>0.10 中位数，当前v1=0.067，v3/v4=0.000)

**Step 1.2**：基于v1架构重训（保留TemporalBlock，避免LGT GCN过度平滑）
- 文件：新建 `/Users/zzl/PycharmProjects/vad/video_anomaly_detection/train_mil_v5.py`
- 训练配置：
  - 架构：`MILAnomalyScorer`（v1的TemporalBlock，非LGTAdapter）
  - Loss：`lam_sparse=0.01, lam_smooth=0.002, lam_range=0.05, alpha=0.0001, margin=100`
  - 优化器：AdamW, lr=0.0003, cosine annealing, warmup=5 epochs
  - 训练：batch_size=16, epochs=80, gradient clipping=1.0
  - 早停条件：每10 epoch检查，若>70%视频 range<0.05 则停止并回退
- 依赖：Step 1.1

**Step 1.3**：per-video范围监控评估框架
- 文件：`/Users/zzl/PycharmProjects/vad/video_anomaly_detection/evaluate.py`
- 新增指标：
  - `per_video_score_range`：每个异常视频的 max-min 分数
  - `collapse_rate`：range<0.05 的视频比例
  - `temporal_discrimination_score`：GT区间内vs外的分数差
- 用于验证Step 1.2的训练质量

**Step 1.4**：MIL融合权重动态化
- 文件：`/Users/zzl/PycharmProjects/vad/video_anomaly_detection/pipeline.py` (当前L91: `0.75*seg_scores + 0.25*mil_interp`)
- 改动：基于MIL per-video range动态调整权重
```python
mil_range = mil_scores.max() - mil_scores.min()
mil_weight = min(0.5, 0.25 + 0.5 * (mil_range / 0.3))  # range越大,越信任MIL
fused = (1 - mil_weight) * seg_scores + mil_weight * mil_interp
```
- 原理：当MIL对某视频有强时序信号时，增大其权重；collapsed时回退到v51行为
- 依赖：Step 1.2

**预期收益**：mAP 0.14 → 0.20-0.25（MIL从常数偏移变为有效时序信号）

---

### Phase 2：候选生成增强 + 低CLIP类别突破（2-3周）

**核心思路**：解决28.8%候选位置错误问题。当前CLIP对4类异常(Shoplifting/Stealing/Arson/Burglary)完全无区分力，需要替代信号源。

**Step 2.1**：扩展混淆负例提示（v78框架延伸）
- 文件：`/Users/zzl/PycharmProjects/vad/video_anomaly_detection/anomaly_expert.py`
- 改动：每个低CLIP类别从2条confusion negative扩展到4-6条
- 参考v78实现：`relative_scores *= (1 - 0.35*confusion_prob)` — 仅在候选生成阶段
- 优化方向：confusion prompts自动生成（用LLM批量生成初始列表并人工审核）
- 依赖：无（独立于Phase 1）

**Step 2.2**：多阈值TAG候选融合（修复v74/v76未生效的设计）
- 文件：`/Users/zzl/PycharmProjects/vad/video_anomaly_detection/pipeline.py`
- 背景：v74设计了MIL TAG候选生成但因device bug未生效(v76)
- 改动：用Phase 1修复后的MIL v5生成高质量TAG proposals
  - 多阈值(0.3, 0.5, 0.7)扫描MIL snippet scores
  - 与CLIP候选合并（IoU>0.3去重）
  - 低CLIP类别优先使用MIL候选（CLIP候选仅作验证）
- 依赖：Phase 1 Step 1.2（需要不再collapsed的MIL模型）

**Step 2.3**：运动能量增强候选生成
- 文件：`/Users/zzl/PycharmProjects/vad/video_anomaly_detection/video_preprocessor.py`
- 改动：
  - 对低CLIP类别，提取光学流magnitude作为独立候选信号
  - 流突变点(>2σ超过局部均值)作为异常候选边界锚点
  - 融合：`candidate_score = 0.5*clip_score + 0.3*mil_score + 0.2*flow_score`
- 工具：cv2.DISOpticalFlow（轻量级，可配置仅对低CLIP类别计算）
- 依赖：Phase 1完成

**Step 2.4**：snippet分辨率提升（32→48段）
- 文件：`/Users/zzl/PycharmProjects/vad/video_anomaly_detection/extract_features.py` (L31)
- 改动：从均匀32段改为48段（折中方案，避免64段的高风险）
- 影响：MIL输入维度 (B,32,768) → (B,48,768)，需Phase 1 v5重训时直接使用48段
- 原理：更细的时间粒度直接提升边界精度（6s段→4s段）
- 依赖：Phase 1 Step 1.2（训练时直接用48段）

**预期收益**：mAP 0.20 → 0.30-0.35（候选位置错误从28.8%降至<15%）

---

### Phase 3：边界检测架构创新（3-4周）

**核心思路**：保留v51证明有效的半峰值交叉框架，但将输入从单一CLIP曲线改为多信号融合的高质量时序曲线。

**Step 3.1**：学习式边界质量预测器
- 文件：新建 `/Users/zzl/PycharmProjects/vad/video_anomaly_detection/boundary_predictor.py`
- 架构：轻量级1D-CNN (3层，channels=64/32/1)
  - 输入：以候选边界为中心的±3s窗口内的多信号特征(CLIP score + MIL score + motion + temporal_diff)
  - 输出：边界偏移量(regression, ±2s范围) + 定位质量分(classification, 0-1)
- 训练数据：UCF-Crime训练集GT边界 + 随机偏移作为负样本
- 集成：半峰值交叉得到初始边界 → boundary_predictor微调位置 + 输出质量分
- 依赖：Phase 2完成（需要多信号特征）

**Step 3.2**：IoU感知置信度重排（v77扩展版）
- 文件：`/Users/zzl/PycharmProjects/vad/video_anomaly_detection/pipeline.py` (L428-445)
- 当前v77仅用center_score，扩展为完整的定位质量评分：
```python
Q = 0.3 * boundary_sharpness + 0.3 * peak_contrast + 0.2 * center_score + 0.2 * duration_fit
final_confidence = raw_confidence * (0.4 + 0.6 * Q)
```
- 信号来源：
  - boundary_sharpness = |gradient| at t_start/t_end（已在_score_boundary_proposal计算）
  - peak_contrast = (peak - boundary) / peak
  - center_score = 1 - 2*|peak_pos - center| / duration（v77已有）
  - duration_fit = 已有
- 风险：极低（仅改变排序，不改变边界位置）
- 依赖：无（可与其他步骤并行）

**Step 3.3**：自适应类别感知边界检测
- 文件：`/Users/zzl/PycharmProjects/vad/video_anomaly_detection/pipeline.py` (L1263+)
- 改动：
  - 基于训练集统计为每个类别学习最优半峰值阈值（而非固定0.5）
  - 基于per-video CLIP曲线对比度动态调整（低对比度→更严格阈值，减少FP）
  - 使用交叉验证确定类别参数，避免test set过拟合
- 参考：v52-v54尝试自适应阈值FAILED，因为同时改了候选选择策略
- 本次仅改阈值，保留contrast-only的候选选择（v51证明最佳）
- 依赖：Phase 1/2的基线建立

**Step 3.4**：Temporal Gaussian Splatting 定位（GS-MoE启发）
- 文件：新建 `/Users/zzl/PycharmProjects/vad/video_anomaly_detection/temporal_gaussian.py`
- 核心思想：将每个异常事件建模为时间轴上的高斯分布(μ, σ)
  - μ = 事件中心时间点
  - σ = 事件持续时间/4（使95%概率落在边界内）
  - 多个高斯叠加形成连续的异常概率曲线
- 实现：
  - MIL v5 snippet scores → 阈值分割得到初始(μ,σ)估计
  - 用可微分高斯渲染优化(μ,σ)以最大化与观测信号的拟合
  - 最终边界：高斯CDF交叉阈值0.5的位置
- 优势：连续可微，自然处理重叠事件，提供精确边界
- 风险：中等（需要良好的初始化，依赖Phase 1/2的MIL质量）
- 依赖：Phase 1（MIL v5提供初始化信号）

**预期收益**：mAP 0.30 → 0.40-0.50（边界精度从IoU 0.3水平提升到0.5+水平）

---

### Phase 4：端到端弱监督集成（4-6周）

**核心思路**：如果Phase 1-3的组合达到~0.40但未达0.50，需要引入端到端训练的弱监督框架替代手工设计的多阶段pipeline。

**Step 4.1**：VadCLIP-style 端到端训练框架
- 文件：新建 `/Users/zzl/PycharmProjects/vad/video_anomaly_detection/vadclip_framework.py`
- 架构（参考VadCLIP AAAI 2024）：
  - 冻结CLIP ViT-L-14视觉编码器（已有特征）
  - 可学习时序适配器：Multi-scale Temporal Adapter (而非LGT-Adapter，因GCN已被证明过度平滑)
    - 使用dilated temporal convolutions（与v1 TemporalBlock类似但更深）
    - 加multi-head cross-attention到类别文本embedding
  - 双流输出：snippet-level anomaly score + boundary regression offset
- 训练策略：
  - 阶段A：视频级MIL预训练（BCE + ranking loss + range loss），100 epochs
  - 阶段B：伪标签微调边界（用阶段A的高置信度预测作为伪GT），50 epochs
  - 阶段C：联合优化（端到端，包括boundary regression），30 epochs
- 损失函数组合：
  - L_mil = BCE + α*magnitude_ranking + β*range_adversarial
  - L_boundary = Smooth-L1(predicted_offset, pseudo_gt_offset) * confidence_weight
  - L_total = L_mil + γ*L_boundary + λ*L_temporal_smooth
- 依赖：Phase 1-3完成（提供伪标签和baseline对比）

**Step 4.2**：PE-MIL 提示增强（如文献调研支持）
- 架构扩展：在Step 4.1框架中加入prompt-enhanced attention
- 为每个异常类别学习可训练的prompt embedding
- 在时序adapter的cross-attention中融合prompt信息
- 依赖：Step 4.1

**Step 4.3**：Soft-NMS + 类别感知后处理学习
- 文件：`/Users/zzl/PycharmProjects/vad/video_anomaly_detection/post_processor.py`
- 改动：
  - 将固定IoU阈值(0.3)替换为类别感知的学习阈值
  - 实现Soft-NMS（线性/高斯衰减）替代Hard-NMS
  - 移除手工`VIDEOMAE_CONFUSION_MAP`，用学习式混淆对检测替代
- 依赖：Step 4.1

**预期收益**：mAP 0.40 → 0.50+（端到端训练突破手工pipeline的组合误差累积）

---

## 依赖关系图

```
Phase 0 (文献调研) ─────────────────────────┐
                                              ↓
Phase 1 (MIL修复) ──────────────────────────→ Phase 2 (候选增强)
  1.1 Range Loss                                2.1 Confusion Negatives (独立)
  1.2 MIL v5 Training ──→ 1.4 动态融合         2.2 TAG候选 (依赖1.2)
  1.3 监控框架                                  2.3 运动增强 (依赖Phase 1)
                                              2.4 48段 (依赖1.2训练时直接使用)
                                              ↓
                                         Phase 3 (边界创新)
                                           3.1 边界预测器 (依赖Phase 2)
                                           3.2 重排 (独立,可并行)
                                           3.3 自适应阈值 (依赖Phase 1/2)
                                           3.4 高斯溅射 (依赖Phase 1)
                                              ↓
                                         Phase 4 (端到端弱监督)
                                           4.1 VadCLIP框架 (依赖Phase 1-3)
                                           4.2 PE-MIL (依赖4.1)
                                           4.3 后处理学习 (依赖4.1)
```

**关键路径**：0 → 1.1 → 1.2 → 2.2/2.4 → 3.1/3.4 → 4.1

**可并行的独立路径**：
- 2.1(Confusion Negatives) 可与Phase 1并行
- 3.2(重排) 任何时候可执行，零风险
- 0(文献调研) 与1.1/1.3并行

---

## 风险矩阵与缓解策略

| 风险 | 概率 | 影响 | 缓解策略 | 回退方案 |
|------|------|------|---------|---------|
| MIL v5训练仍然collapse | 中 | 阻塞整个Phase 2-4 | Range loss + 每epoch监控 + 早停 | 保留v1，仅用CLIP+运动信号 |
| 48段snippet导致MIL性能下降 | 低 | 需重新训练 | A/B测试32 vs 48，保留两版 | 回退32段 |
| 边界预测器过拟合(训练集小) | 中 | Phase 3.1无效 | 数据增强(时间偏移±1s) + dropout=0.5 + 交叉验证 | 保留半峰值交叉 |
| 端到端训练不收敛 | 中 | Phase 4失败 | 分阶段训练(A→B→C) + 梯度剪裁 + 学习率warmup | Phase 1-3组合已达0.35-0.45 |
| 光学流计算延迟爆炸 | 低 | 处理时间+50% | 仅对低CLIP类别计算 + DIS lightweight模式 | 禁用光学流 |
| VLM API stochasticity掩盖改进 | 高 | 难以判断改动是否有效 | 每版本跑3次取中位数 + 固定seed | 用smoke test(13视频)快速验证方向 |

---

## 代码规范与约束

### 必须遵守
1. **v51半峰值交叉不动**：边界检测的核心机制已证明是当前最优（v52-v67所有替代方案均回退），仅允许改变输入信号质量
2. **Config中心化**：所有新参数必须通过 `config.py` 的对应子类管理，禁止新增魔法数字
3. **Per-video range检验**：任何新MIL模型必须通过collapse检验（>60%视频range>0.05）才能上线
4. **增量评估**：每个Step完成后跑smoke test(13视频)验证不回退，再跑full eval(65视频)确认提升
5. **保留fallback**：所有改动必须有配置开关，可通过config恢复v75/v76行为

### 文件修改范围
- 核心修改：`mil_model.py`, `pipeline.py`, `config.py`, `evaluate.py`
- 新增文件：`train_mil_v5.py`, `boundary_predictor.py`, `temporal_gaussian.py`, `vadclip_framework.py`
- 辅助修改：`extract_features.py`, `anomaly_expert.py`, `frame_scorer.py`, `post_processor.py`

---

## 预期里程碑

| 里程碑 | mAP目标 | 关键交付 | 时间 |
|--------|---------|---------|------|
| M1: MIL v5不再collapse | 0.18-0.22 | 训练好的模型，collapse率<40% | +2周 |
| M2: 候选生成修复 | 0.25-0.30 | 错误候选<15%，低CLIP类别有效检测 | +4周 |
| M3: 边界精度突破 | 0.35-0.40 | AP@0.5>0.25，边界预测器上线 | +7周 |
| M4: 端到端弱监督 | **>0.50** | VadCLIP框架完成训练 | +12周 |

---

## 被否决的替代方案

### 1. 纯增量参数调优（Jack方案核心）
- **原因**：v51-v78已证明0.14是结构性天花板，无论如何调参都无法突破
- **证据**：20+版本在7种范式下收敛到同一水平

### 2. LGT-Adapter替代TemporalBlock（v3/v4方案）
- **原因**：GCN全局聚合过度同质化特征，导致75-83% collapse
- **替代**：保留v1的TemporalBlock（dilated conv保留局部性），用range loss修复训练
- **证据**：loop10-11文档明确记录失败原因

### 3. CLIP特征重中心化（v72方案）
- **原因**：modulate fused_scores改变了半峰值交叉位置，导致边界漂移
- **证据**：v72 smoke test mAP从0.14降到0.08
- **部分采纳**：重中心化概念用于候选生成阶段（v78 confusion negatives），但不用于边界检测

### 4. 12周代码重构优先（Alex方案Phase 1）
- **原因**：代码质量重要但不是mAP瓶颈；3周重构延迟了性能关键路径
- **部分采纳**：config中心化已完成(v33)；dead code清理作为非关键路径执行
- **决策**：性能改进优先，代码清理作为Phase 1-4间的间隙任务

### 5. 直接32→64段分辨率翻倍（Sam方案Step 3.2）
- **原因**：翻倍过于激进，需要完全重训MIL且历史无验证
- **替代**：折中方案32→48段（+50%分辨率，风险可控）
- **条件**：在Phase 1训练v5时直接使用48段，避免额外重训成本