> V3 使用说明及新 MCTS 开关见 [V3 README](../README.md)。下文保留 V2 的基础说明，历史评测数字不代表 V3。

# Video Anomaly Detection — LLM + MCTS 时间定位系统

基于 **CLIP 零样本分类** + **VLM 视觉推理** + **MCTS 时间搜索** 的视频异常检测与时间定位系统。

无需训练即可检测监控视频中的异常事件，并精确定位异常发生的时间边界。

## 目录

- [系统架构](#系统架构)
- [环境安装](#环境安装)
- [快速开始](#快速开始)
- [Pipeline 详解](#pipeline-详解)
  - [Stage 1: 视频预处理](#stage-1-视频预处理)
  - [Stage 2: 粗粒度异常发现](#stage-2-粗粒度异常发现)
  - [Stage 3: MCTS 时间边界定位](#stage-3-mcts-时间边界定位)
  - [Stage 4: 边界精修](#stage-4-边界精修)
  - [Stage 5: 后处理](#stage-5-后处理)
- [核心组件](#核心组件)
  - [CLIP 异常专家](#clip-异常专家-anomaly_expertpy)
  - [语义帧采样器](#语义帧采样器-token_compressorpy)
- [评估系统](#评估系统)
- [配置参数](#配置参数)
- [项目结构](#项目结构)
- [设计思路](#设计思路)

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                        MultiAnomalyVAD Pipeline                     │
│                                                                      │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐              │
│  │  Stage 1     │   │  Stage 2     │   │  Stage 3     │              │
│  │  视频预处理   │──▶│  异常发现     │──▶│  MCTS 定位   │              │
│  │              │   │              │   │              │              │
│  │ - 抽帧(4fps) │   │ - CLIP 预筛  │   │ - 二叉搜索   │              │
│  │ - 运动过滤   │   │ - LLM 并行   │   │ - LLM 验证   │              │
│  │ - 等长分段   │   │   扫描       │   │ - CLIP 引导   │              │
│  └─────────────┘   └──────────────┘   └──────┬───────┘              │
│                                               │                      │
│  ┌─────────────┐   ┌──────────────┐          │                      │
│  │  Stage 5     │   │  Stage 4     │◀─────────┘                      │
│  │  后处理      │◀──│  边界精修     │                                 │
│  │              │   │              │                                  │
│  │ - 合并重叠段 │   │ - 二分收缩   │                                  │
│  │ - CLIP 重分类│   │ - 二分扩展   │                                  │
│  │ - 输出结果   │   │ - 并行左右   │                                  │
│  └─────────────┘   └──────────────┘                                  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────┐            │
│  │                   共享组件                            │            │
│  │  ┌──────────────────┐  ┌──────────────────────────┐  │            │
│  │  │ VideoAnomalyExpert│  │ SemanticFrameSampler     │  │            │
│  │  │ (CLIP ViT-B-32)  │  │ (动态帧数 + 语义去重)     │  │            │
│  │  └──────────────────┘  └──────────────────────────┘  │            │
│  └──────────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────┘
```

**核心思路**：将视频异常检测类比为 DyFo（Dynamic Focus）框架的时间维度版本：
- DyFo 用 MCTS 在图像空间中搜索包含目标的最小区域
- 本系统用 MCTS 在时间轴上搜索包含异常的最短时间段

## 环境安装

### 依赖

```bash
# 基础依赖
pip install opencv-python-headless numpy openai

# CLIP 模型（推荐，提供零样本异常分类能力）
pip install open_clip_torch torch pillow

# 评估（仅评估时需要）
pip install scikit-learn
```

### Python 版本

- Python >= 3.9

### 硬件要求

- **CPU 模式**：无需 GPU，CLIP ViT-B-32 可在 CPU 上运行（~350MB 内存）
- **GPU 模式**：支持 CUDA / Apple MPS 加速 CLIP 推理
- **网络**：需要访问 DashScope API（qwen-vl-max 视觉语言模型）

## 快速开始

### 1. 设置 API Key

```bash
export DASHSCOPE_API_KEY=sk-your-api-key
```

### 2. 运行检测

```bash
# 基本用法
python run.py --video_path /path/to/video.mp4

# 完整参数示例
python run.py \
  --video_path data/video/sample.mp4 \
  --output_file data/results/result.json \
  --model qwen-vl-max \
  --clip_model ViT-B-32 \
  --device cpu \
  --mcts_simulations 8 \
  --mcts_max_depth 4 \
  --fps 4 \
  --segment_duration 8.0
```

### 3. 输出格式

```json
{
  "video_path": "data/video/sample.mp4",
  "duration": 211.1,
  "processing_time": 28.2,
  "is_normal": false,
  "anomalies": [
    {
      "t_start": 11.1,
      "t_end": 21.1,
      "confidence": 0.91,
      "anomaly_type": "Stealing"
    }
  ]
}
```

### 4. 评估

```bash
python evaluate.py \
  --label_file data/test_labels.txt \
  --result_dir data/results/ \
  --fps 30
```

> **注意**：`--fps` 必须使用原始视频帧率（通常为 30），而非采样帧率（4），因为 GT 标注使用原始帧号。

## Pipeline 详解

### Stage 1: 视频预处理

**文件**: `video_preprocessor.py` — `VideoPreprocessor`

将原始视频转化为结构化数据，供后续所有阶段使用。

```
原始视频 → 按帧率采样 → 运动过滤 → 等长分段 → video_data dict
```

**处理步骤**：

1. **帧采样**：按指定帧率（默认 4fps）从视频中均匀抽帧，输出 RGB numpy 数组
2. **运动过滤**：计算相邻帧的像素差分（在 64x64 缩小图上），跳过帧差均值低于 `motion_threshold` 的静止帧。保底机制确保静态区域至少每 `min_motion_interval` 秒保留 1 帧
3. **等长分段**：按 `segment_duration`（默认 8 秒）将时间轴切分为初始段，每段记录对应的帧索引

**输出 `video_data` 结构**：

```python
{
    'video_path': str,          # 视频文件路径
    'duration': float,          # 视频总时长（秒）
    'all_frames': np.ndarray,   # [N, H, W, 3] 所有采样帧（RGB）
    'timestamps': np.ndarray,   # 每帧对应的时间戳
    'segments': [               # 初始分段列表
        {'t_start': float, 't_end': float, 'frame_indices': [int, ...]},
        ...
    ],
    'video_fps': float,         # 原始视频帧率
}
```

**CLIP 特征预计算**：在 Stage 1 完成后，Pipeline 会调用 `anomaly_expert.precompute_video_features()` 一次性编码所有帧的 CLIP 特征，后续阶段通过索引查表复用，避免重复编码。

### Stage 2: 粗粒度异常发现

**文件**: `anomaly_discovery.py` — `AnomalyDiscovery`

快速筛选出可能包含异常的时间段，减少后续精细搜索的范围。

```
初始分段 → CLIP 预筛 → LLM 并行扫描 → 合并相邻 → 候选列表
```

**处理步骤**：

1. **CLIP 预筛选**：
   - 对所有初始段批量计算 CLIP 异常分数（`batch_score_segments`，单次矩阵运算）
   - 使用自适应阈值（所有段分数的均值）过滤明显正常的段
   - 典型效果：50-70% 的段被跳过，大幅减少 LLM API 调用

2. **LLM 并行扫描**：
   - 对通过预筛的段，通过 `asyncio.gather` 并发调用 VLM（qwen-vl-max）
   - VLM 接收关键帧图片 + prompt，返回 JSON 格式的判断结果
   - 包含：是否异常、置信度、异常类型

3. **候选合并**：
   - 相邻的同类型可疑段合并为一个候选区域
   - 每个候选带有起止时间、异常类型和粗粒度分数

### Stage 3: MCTS 时间边界定位

**文件**: `temporal_mcts.py` — `TemporalBoundaryMCTS`

对每个候选区域独立运行 MCTS，精确定位异常的时间边界。

```
候选区域 [t_start, t_end] → MCTS 搜索 → 最精确的异常时间段
```

**MCTS 搜索空间**：

| 维度 | 说明 |
|------|------|
| 状态 | 时间段 `(t_start, t_end)` |
| 动作 | `shrink_left`（左边界右移）、`shrink_right`（右边界左移） |
| 奖励 | `clip_score × 0.8 + shrink_bonus + boundary_bonus` |
| 终止 | 达到最大深度或时间段过短 |

**四个阶段**：

1. **Selection（选择）**：从根节点沿 UCB 最高的子节点向下遍历，探索系数随模拟次数衰减（`c_puct × e^(-λ × sim_i)`）
2. **Expansion（扩展）**：在叶节点执行一个未尝试的动作，创建新子节点。动作选择由 CLIP `peak_time` 引导——异常峰值在左半部分则优先 `shrink_right`，反之优先 `shrink_left`
3. **Simulation（模拟）**：调用 VLM 判断当前时间段是否仍包含目标异常，并结合 CLIP 异常分数计算奖励。VLM 还返回边界信息（异常是否从段首开始、是否延伸到段尾），用于引导后续搜索方向
4. **Backpropagation（回传）**：将奖励沿路径向上传播，更新每个节点的访问次数和 Q 值

**收缩比例**：深度自适应，`ratio = 0.4 / (1 + 0.3 × depth)`，浅层大步收缩快速缩小范围，深层精细收缩精确定位。

**结果提取**：从所有标记为"包含异常"的节点中，选择 `leaf_reward` 最高的作为最终结果。异常类型优先使用 CLIP 的 `predicted_type`。

### Stage 4: 边界精修

**文件**: `pipeline.py` — `_refine_boundary()`

对 MCTS 输出的时间段做双向二分搜索精修，修正边界误差。

```
MCTS 输出 → Pass 1: 收缩修剪 → Pass 2: 扩展发现 → 精修结果
```

**Pass 1 — 收缩**：
- 从两端向内二分搜索，找到 CLIP 异常分数首次 >= `core_score × 0.6` 的位置
- 去除两端低分"噪声"帧，最大收缩量为段长的 30%
- 左右方向通过 `asyncio.gather` 并行执行

**Pass 2 — 扩展**：
- 从修剪后的边界向外二分搜索，找到 CLIP 分数骤降到 `core_score × 0.8` 以下的位置
- 发现 MCTS 可能遗漏的异常边缘，最大扩展量为段长的 75%
- 同样左右并行

### Stage 5: 后处理

**文件**: `pipeline.py` — `_global_merge()` + `_reclassify_anomalies()`

合并重叠段、统一分类，输出最终结果。

**合并策略**（迭代至稳定）：

| 条件 | 合并规则 |
|------|---------|
| 同类型 + 有重叠或间隔 < 3s | 合并，保留较高置信度 |
| 不同类型 + IoU > 0.3 | 合并，保留高置信度的类型 |
| 其他 | 不合并 |

**CLIP 重分类**：
- 合并后的段可能包含来自不同候选的混合类型
- 用 CLIP 对每个最终段重新评估异常类型，解决分类冲突

## 核心组件

### CLIP 异常专家 (`anomaly_expert.py`)

**类**: `VideoAnomalyExpert`

零样本异常分类引擎，基于 CLIP ViT-B-32（通过 open_clip 加载）。

**工作原理**：
1. 预计算文本特征：每个异常类别 2 条描述性 prompt + 3 条正常活动 prompt，各自做均值集成
2. 视频帧编码：通过 CLIP 视觉编码器提取帧特征（预计算后索引查表）
3. 相似度计算：帧特征与文本特征的余弦相似度 × 学习到的 `logit_scale`
4. 异常评分：`top3_anomaly / (top3_anomaly + normal_prob)`
   - `top3_anomaly`：softmax 后前 3 个最高异常类别概率之和
   - `normal_prob`：正常类别的概率
   - 使用 top-3 而非 max，因为相似类别（如 Robbery/Stealing）的概率可能分散

**Prompt 设计**：

每个类别配有 2 条差异化描述，强调该类别的独特视觉特征：
```
Robbery  → "armed person threatening and forcibly taking valuables"
Stealing → "secretly taking someone else's unattended belongings"
```

**关键方法**：

| 方法 | 用途 | 性能 |
|------|------|------|
| `precompute_video_features()` | 批量编码全部帧 | 一次计算，全程复用 |
| `batch_score_segments()` | 批量评估多个段 | 单次矩阵运算 |
| `score_segment()` | 评估单个段（含详细结果） | 返回各类别分数和峰值帧 |

### 语义帧采样器 (`token_compressor.py`)

**类**: `SemanticFrameSampler`

动态帧数 + 语义去重的两阶段帧选择，控制发送给 VLM 的图片数量。

**阶段 1 — 动态帧数**：
```
K = clamp(duration × fps_effective, min_frames, max_for_depth)

其中：
  fps_effective = fps_sample × (1 + 0.5 × n_shrinks)  # 越深越密
  max_for_depth = max_frames / (1 + depth × 0.3)      # 越深上限越低
```

- 浅层（MCTS 早期）：段长、帧数少，粗采样做概览
- 深层（MCTS 后期，多次 shrink）：段短、帧数密，精采样做精判

**阶段 2 — 语义去重**：
1. 用 CLIP 特征计算帧间余弦相似度矩阵
2. 贪心聚类：相似度 > τ（默认 0.92）的帧归入同一簇
3. 从每簇中选最独特的帧（与簇内其他帧平均相似度最低者）

## 评估系统

**文件**: `evaluate.py`

支持三级评估指标，兼容 UCF-Crime 标注格式。

### GT 标注格式

```
# video_id category start_frame1 end_frame1 [start_frame2 end_frame2 ...]
video_name Robbery 450 1050 1050 1500
normal_video Normal
```

### 评估指标

| 级别 | 指标 | 说明 |
|------|------|------|
| 帧级 | AUC-ROC | UCF-Crime 标准指标，所有视频帧拼接后计算全局 AUC |
| 段级 | mAP (AP@0.3/0.5/0.7) | 时间段级别的平均精度，按不同 IoU 阈值评估边界精度 |
| 视频级 | Accuracy/Precision/Recall/F1/AUC | 视频级二分类（正常 vs 异常） |
| 类别级 | Category Accuracy | 异常类型分类准确率 |

### 运行评估

```bash
# 评估特定结果目录
python evaluate.py \
  --label_file data/test_labels.txt \
  --result_dir data/results/ \
  --fps 30 \
  --output_file data/eval_report.json
```

## 配置参数

### 视频预处理

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--fps` | 4 | 采样帧率（每秒抽取帧数） |
| `--segment_duration` | 8.0 | 初始分段长度（秒） |
| `--motion_threshold` | 1.0 | 运动过滤阈值（0-255，0 禁用） |
| `--min_motion_interval` | 2.0 | 静态区域最低帧保留间隔（秒） |

### MCTS 搜索

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mcts_simulations` | 8 | 每个候选的 MCTS 模拟次数（越多越精确但越慢） |
| `--mcts_max_depth` | 4 | MCTS 搜索最大深度 |

### CLIP 模型

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--clip_model` | ViT-B-32 | CLIP 模型架构 |
| `--clip_pretrained` | openai | 预训练权重来源（openai / webli） |
| `--device` | cpu | 计算设备（cpu / cuda / mps） |

### 帧采样

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_frames` | 200 | 单次 LLM 请求的帧数硬上限 |
| `--min_frames` | 3 | 帧数下限（确保 LLM 能识别内容） |
| `--fps_sample` | 2.0 | 动态帧数计算的目标 fps |

### LLM

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | qwen-vl-max | VLM 模型名称 |
| `--api_key` | 环境变量 | DashScope API Key |
| `--api_base_url` | DashScope URL | API 端点 |
| `--temperature` | 0.0 | 生成温度（0 = 确定性输出） |
| `--max_tokens` | 256 | 最大生成 token 数 |

## 项目结构

```
video_anomaly_detection/
├── __init__.py              # 包导出
├── run.py                   # 入口脚本（参数解析 + LLM 客户端 + 调用 Pipeline）
├── pipeline.py              # 主 Pipeline（五阶段编排 + 边界精修 + 合并策略）
├── video_preprocessor.py    # Stage 1: 视频抽帧 + 运动过滤 + 分段
├── anomaly_discovery.py     # Stage 2: CLIP 预筛 + LLM 并行扫描
├── temporal_mcts.py         # Stage 3: MCTS 时间边界搜索
├── anomaly_expert.py        # CLIP 零样本异常分类专家
├── token_compressor.py      # 语义帧采样器（动态帧数 + CLIP 去重）
├── evaluate.py              # UCF-Crime 评估（帧AUC / 段mAP / 视频F1）
├── analyze_clip_similarity.py  # CLIP 类别相似度分析工具
└── data/
    ├── test_labels.txt      # GT 标注文件
    └── video/               # 测试视频
```

### 模块依赖关系

```
run.py
 └── pipeline.py (MultiAnomalyVAD)
      ├── video_preprocessor.py (VideoPreprocessor)
      ├── anomaly_discovery.py (AnomalyDiscovery)
      │    ├── anomaly_expert.py (VideoAnomalyExpert)  ← CLIP
      │    └── token_compressor.py (SemanticFrameSampler)
      ├── temporal_mcts.py (TemporalBoundaryMCTS)
      │    ├── anomaly_expert.py (VideoAnomalyExpert)  ← CLIP
      │    └── token_compressor.py (SemanticFrameSampler)
      └── anomaly_expert.py (VideoAnomalyExpert)       ← 精修/重分类
```

## 设计思路

### 为什么用 MCTS 做时间定位？

传统方法（滑动窗口、动态规划）在时间轴上搜索效率低。MCTS 的优势：
- **自适应搜索**：先粗后细，快速缩小范围
- **Expert 引导**：CLIP 异常峰值指导收缩方向，避免盲目搜索
- **LLM 验证**：每步搜索都有 VLM 语义验证，不会丢失真正的异常

### 为什么需要 CLIP + LLM 双系统？

| 系统 | 优势 | 劣势 |
|------|------|------|
| CLIP | 快速、本地、无 API 成本、可批量 | 分类粗糙、容易误报 |
| LLM (VLM) | 语义理解深、判断准确 | 慢、有 API 成本、受帧数限制 |

两者配合：CLIP 做粗筛减少 LLM 调用量，LLM 做精判确保结果准确。

### 运动预过滤的作用

监控视频中 60-90% 的帧是静态背景。通过帧差分检测运动：
- 减少 CLIP 编码量（905 帧 → 475 帧，减少 47%）
- 减少 LLM 接收的冗余帧
- 对活跃视频无负面影响（运动帧全部保留）

### 支持的异常类别

系统默认支持 UCF-Crime 数据集的 13 个异常类别：

| 类别 | 说明 |
|------|------|
| Abuse | 身体虐待 |
| Arrest | 逮捕行为 |
| Arson | 纵火 |
| Assault | 暴力攻击 |
| Burglary | 入室盗窃 |
| Explosion | 爆炸 |
| Fighting | 打架斗殴 |
| RoadAccidents | 交通事故 |
| Robbery | 抢劫 |
| Shooting | 枪击 |
| Shoplifting | 商店偷窃 |
| Stealing | 偷盗 |
| Vandalism | 故意破坏 |

类别列表可自定义，通过 `anomaly_categories` 参数传入。

### 编程接口

```python
from video_anomaly_detection import MultiAnomalyVAD

vad = MultiAnomalyVAD(
    llm_client=your_llm_client,       # 需实现 async generate(prompt, images) 方法
    clip_model_name="ViT-B-32",
    device="cpu",
)

result = await vad.detect("video.mp4", context="surveillance camera")

for anomaly in result['anomalies']:
    print(f"{anomaly['t_start']:.1f}s - {anomaly['t_end']:.1f}s: "
          f"{anomaly['anomaly_type']} (conf={anomaly['confidence']:.2f})")
```

LLM 客户端需实现以下接口：

```python
class YourLLMClient:
    async def generate(self, prompt: str, images: np.ndarray = None) -> str:
        """
        Args:
            prompt: 文本提示
            images: [N, H, W, 3] RGB numpy 数组，或 None
        Returns:
            str: 模型生成的文本
        """
        ...
```
