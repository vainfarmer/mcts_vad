# CBT 后执行 MCTS 的修改方案与推理纠正

日期：2026-10-28。状态：**设计评审稿，尚未实现末端 MCTS**。本文不修改 v1/v2/v3 的当前推理行为，不启动新的付费实验。

## 1. 最终结论与范围

建议在现有 v3 中新增可配置的 `post_cbt` 搜索位置，复用当前有预算、引用检查和回退机制的搜索器。保留 `stage5` 位置作为对照，保留 v2 作为历史基线。第一阶段不直接移植 v1 搜索代码，不同时启用早期和末端两轮 MCTS，不启用 v216 的几何边界保护，不重算检测置信度。

预期解决的问题是：**让搜索器评价的最终提案直接成为输出区间，避免提案之后再次被 BUE/CBT 改写。** 这并不证明末端搜索比当前流程更准确、更快或更便宜。

本阶段只优化已保留检测的时间边界，不负责恢复早退视频、已删除候选、未发现事件；不通过 MCTS 删除检测或改变类别。最终 NMS 仍可以抑制重复检测，其效果单独记录。

## 2. 对上一版推理的纠正

| 原建议中的隐含推断 | 纠正后的结论 |
|---|---|
| 把 MCTS 移到最后就更好 | 只消除了“搜索后再次改边界”的冲突；目标函数、图像证据、错误种子等问题仍存在，效果须实测。 |
| A/B/C 可以固定相同最终 confidence，做纯位置消融 | 只有末端 on/off 能按检测 ID 继承同一组最终 confidence。早期 MCTS 会改变合并、筛选、Q-score、CBT 校准，最终检测集合也可能不同，不能强行配对或复制分数。 |
| 相同搜索器等于仅改变一个独立变量 | 位置改变会连带改变种子长度、候选数量、类别上下文和后续处理。早期与末端比较是“插入位置策略的端到端比较”，不是只替换一个端点的微观因果实验。 |
| v216 几何保护退化证明末端 MCTS 会改善 | 仅说明早期已接受区间不宜作为所有后处理的硬边界。v216 没有运行末端 MCTS，不能推出其收益。 |
| CBT 输出可视为正确完整事件 | 它可能过短、偏移、包含多个事件或仍有背景。只作为比较基准和失败回退，不视为真值。 |
| 原始候选范围/CLIP 峰值应成为不可越过的边界 | 粗候选可能不完整，峰值也可能是背景。保留它们作为来源和辅助证据，不能要求所有提案必须覆盖峰值或锁死在粗候选内。 |
| 末端候选更少，所以耗时必然减少 | 可能更少，但长区间、无效证据、串行评价、API 排队和重试都能抵消收益；配对缓存还会扭曲分支耗时。 |
| 冻结 frame_scores，AUC 一定不变 | 只有存在可用的 frame_scores 且时间戳一致时成立；当前 evaluator 在 frame_scores 缺失或过短时会用检测区间构造分数，AUC 仍可能变化。 |
| v1 `search()` 自带扩窗 | 扩窗在 v1 pipeline：左右各扩原段长度的 75%。只接 `search()` 不会扩窗；它启用的动作只有收缩。 |
| 复制 v1 只是换一个边界优化器 | 原版还可能返回空列表、改变类别，并把 reward 当 confidence，不能用它直接做位置比较。 |
| v3根节点判断无异常就一定停止 | 当前target_present_inside=False可返回有效reward=0，搜索仍能继续；只有评价无效等条件才停止。不能把“必须根部支持目标”暗加成末端新门槛。 |

## 3. 现有证据与不能下的结论

- v215，65 视频：MCTS off/on 的 mAP 为 0.152019/0.155292。对冻结结果仅加最终 NMS，变为 0.152725/0.155802。该试验支持去重的局部收益，未测试末端搜索。
- v216，14 视频：旧 on 与仅去重 on 的 mAP 均为 0.169408；去重加几何保护为 0.161616。Arson035 改善，但 Vandalism015 的有效 BUE 扩展被阻止，IoU 从 0.6667 变为 0。
- v215 与 v216 是不同运行、不同样本范围和分支安排，不能把二者 mAP 差归因于某一个代码变化。
- 14/65 视频已经多次用于开发评估，不能再称为独立未见测试集。选择最终策略后需要核对历史使用记录，另行确定未参与调参的验证数据。

证据文件：

- [v216 报告](D:/Users/zzl/project/mcts_vad/vad_v3/experiments/v216/REPORT.md)
- [65 视频离线去重结果](D:/Users/zzl/project/mcts_vad/vad_v3/experiments/v216/offline_nms_65.json)
- [v215 对比](D:/Users/zzl/project/mcts_vad/vad_v3/experiments/v215/COMPARISON.md)

## 4. 目标架构与精确的插入点

### 4.1 公共分叉点

公共上游运行至 Stage5 的 CLIP 定位完成，冻结 `Stage5Snapshot`。它包含视频数据引用、候选来源、CLIP 区间、类别上下文及原本 Stage5b 触发集合。实验中所有分支从该快照出发。

```text
共享预处理、VideoMAE、候选发现、CLIP 初始定位
                         │
                  Stage5Snapshot
                         │
         ┌───────────────┴──────────────────┐
         │                                  │
  不执行早期 MCTS                      执行早期 MCTS
  完整旧尾流程至 CBT                    完整旧尾流程至 CBT
         │                                  │
       R_cbt                             R_early
         │                                  │
    ┌────┴──────────┐                    最终 NMS
    │               │                        │
 最终 NMS     末端 MCTS(R_cbt)          early_system
    │               │
baseline_cbt     R_late_pre_nms
                    │
                 最终 NMS
                    │
              late_boundary_only
```

**末端搜索使用 CBT 后、最终 NMS 前的全部保留检测。** baseline 与 late 分别在输出端执行相同最终 NMS。不能只对 late 预先 NMS：那会提前删除可能通过搜索移开的候选，并混入不同筛选策略。第一阶段宁可按预算跳过额外候选，也不偷偷增加只对某分支生效的候选过滤。

现有早期合并/NMS 仍属于旧尾流程；这里新增的是末端最终 NMS。若以后需要“末端搜索前先去重”来省成本，应作为另一个明确的消融项，保存被跳过候选及原因。

### 4.2 拆分尾流程

将现有 `_finish_from_stage5()` 拆成两个职责：

1. `finish_to_cbt(...) -> TailSnapshot`：执行原 Stage5b、合并/NMS、收缩、分类、CLIP 修正、置信度筛选、帧分数、Q-score、BUE、CBT，**不运行新增最终 NMS、不立即丢弃来源信息**。
2. `finalize_result(snapshot, candidates, ...) -> DetectionResult`：做区间合法性检查、最终 NMS、ID/附属数组对齐、结果序列化。禁止在这里再次执行 BUE、CBT、分类或置信度门控。

末端搜索插在两者之间。单模式运行只执行必要分支；消融模式 baseline 的 `TailSnapshot` 计算一次后复制给 late，不从零重复跑尾流程。

无候选、Normal 高置信早退仍然早退。消融输出为这些视频生成独立的空/原样分支，并记录 `not_applicable_reason`，不从评测分母删除它们。

### 4.3 “v2 基准”的准确命名

本实现落在 v3，以 MCTS off 的完整尾流程为工程基准，沿用 v2 定位/后处理逻辑。它还包含路径兼容、重试、日志和最终 NMS 等修复，因此报告称 `baseline_cbt`，不能直接冒充历史 v2 实验值。

如需与实际 v2 代码比较，应使用相同上游快照或已冻结请求回放，逐项核对配置和修复差异。不能拿两次独立 API 运行当作严格配对。

## 5. 数据契约：参考区间、搜索域、身份与分数分离

新增 `RefinementSeed` / `TailSnapshot` 内部模型，搜索器不再依赖临时凑齐的裸字典。建议字段：

```python
RefinementSeed(
    detection_id,                 # 本次分支内稳定ID，不使用输出列表下标代替
    source_candidate_ids,         # 支持合并后的多个来源
    reference_interval,           # 本轮真正的基准：stage5或CBT后区间
    output_confidence,            # 保留的检测分数；不是reward
    output_category,              # 保留的输出标签
    target_category,              # 评价使用的类别，必须显式记录
    source_descriptions,          # 有时间范围的原始描述；不当作事实/指令
    source_coarse_scores,         # 可缺失；不得用最终confidence伪造
    lineage_intervals,            # CLIP、pre/post BUE、pre/post CBT历史
    search_domain,               # 可到达端点范围，独立于reference
    evidence_pool_id,
)
```

实施要求：

- Stage5 来源表不可变，合并合并来源 ID，抑制单独记审计，不按排序后的下标猜来源。
- 多来源事件描述必须标注其时间区间，不能把多个不同事件的文本拼成一个“已确认同一事件”。
- 当前 refiner 的 `missing_coarse_score` 是早期接口假设。末端输入允许原粗分缺失，并明确 `missing`，不能用最终分数填充。首轮候选排序继续采用现有“对比值、开始时间、稳定ID”规则；改成最终confidence优先是后续可选策略，不能混入第一轮。
- 当前 `AnomalySegment.from_dict()/to_dict()` 会丢失 source IDs 等附加字段，不能等序列化后再适配。需在内部快照和可选输出追踪结构中明确保留。
- `post_cbt`模式的`reference_interval`为原CBT结果；`stage5`模式仍为原CLIP seed。扩大的搜索窗口不是reference，也不能替代原结果成为fallback。
- v1 后续适配时即使以扩大区间为树根，也必须独立评价原 CBT 结果并与它比较；否则“比扩大根更好”不代表比 v2 输出更好。

类别策略也须明示：**首轮控制协议使用公共快照中的VideoMAE类别作为early/late的评价目标，避免同时改变类别来源。** 输出category依然继承各分支原结果，不用评价目标强行改标签。对末端输出标签与评价目标不一致的检测记录`category_context_mismatch`，单列其覆盖率/行为；不能把它们的变化宣称为分类改善。目标未知/Normal或根部支持不足时，首轮仍按现有调用及证据校验规则处理并记录状态，不对late暗加资格过滤；更保守的支持性筛选另见下文独立策略。

部署候选方案可改用末端最终类别，但须作为`target_category_source=final_detection`的独立消融，并清楚承认相对早期分支增加了类别上下文变化。现有refiner一次只接受一个类别；将来分组搜索必须共享同一arm的视频总预算对象，不能每组重新获得40次请求。来源文本冲突时也不得凭第一个描述强行选择事件。

## 6. 搜索与接受规则

### 6.1 第一轮：先比较位置，不同时调搜索器

使用同一 v3 搜索实现、`shrink_expand_shift`、同一 reward 与 `relative_nonloss`、相同逻辑请求/图像/超时上限。参考 v214–v216配置：每候选10次逻辑评价、每视频40次、最多4候选、单次最多32图、12 simulations、单候选90秒。root评价及最终比较均计入预算，引用修复最多一次且不额外加预算。

初始搜索域采用现有相对规则（参考区间左右各25%），内移限制等也保持不变。短 CBT 种子可能因此可搜索空间不足，需报告该现象，而不能在位置比较中悄悄扩大域。

逻辑预算相同不代表实际 token、评估次数或耗时相同；必须同时报告资源上限与实际用量。原早期分支的 Stage5b 绕过规则保留，末端不再设置任何绕过标志，因为 Stage5b 已完成。

### 6.2 第二轮独立消融：解决短区间搜索域问题

若第一轮表明大量短区间缺少可达动作/外部证据，再单独增加 `bounded_context` 域策略。试验初值而非已验证最优值：

```text
L = reference_end - reference_start
pad = min(max_pad_seconds, max(min_pad_seconds, padding_ratio * L))
domain = [max(0, start-pad), min(video_duration, end+pad)]
min_pad_seconds = 2，max_pad_seconds = 8，padding_ratio = 0.25
```

必须同时调整合法动作的外移限制与采样范围：不能只把 evidence 窗口扩大，而 `legal()` 仍限定外移0.25L，导致新域实际不可达。内移仍有独立比例限制、最短时长及事件保留要求，外移不越过 domain；位移始终相对 reference，防止多层累计越界。

候选的原 CLIP/BUE 范围用于提示和审计，不能作为硬边界阻止补全。超过本轮有界范围的漏检属于本方法能力边界，不自动跳到视频其它事件。

帧不足时返回 `insufficient_evidence` 并保留原区间。局部密集解码是另一个明确配置项，默认不启用；启用后维护独立帧池、真实时间戳、图像上限、解码/编码耗时，不能污染 off 的帧数据。`boundary_dense.py` 也必须使用新域策略，不能残留旧比例窗口。

### 6.3 接受、拒绝与失败

1. 根评价不等于原结果正确。首轮采用`root_support_policy=legacy`：目标不存在但评价有效、reward=0时，保持现有允许探索子节点的行为；评价无效时按当前逻辑停止。单独记录`root_support`及之后是否接受提案，不能把根分数0写成技术失败。若没有合格提案则保留原输出。后续可增加`supported_only`（未知/Normal目标、根部目标或身份不获支持时跳过），但这是新的资格策略，须单独消融，不能只对late悄悄开启。无论哪种策略，都不以GT寻找其它事件，不把跨事件迁移解释为同事件精修。
2. 提案必须有限、合法、实际改变区间，属于同一事件；不得丢弃 reference 中可见的目标活动。reference 可以不完整，提案允许通过外扩补全。
3. 使用 reference/proposal 共用的时间戳证据做最终 A/B；列出每段新增/删除范围与请求内图像索引，保持当前引用有效性检查及相对完整性检查。
4. `relative_nonloss` 中 tie可接受是现有协议。首轮不悄悄改为必须获胜；后续“更保守的末端接受策略”作为独立参数/消融，不能因期待提升而反复追问模型改判。
5. 不启用 v216 基于几何保留率/峰值的硬保护。原区间和曲线峰值都可能错误；同事件判断依赖图像证据与明确语义，不用最大IoU、GT或标签时间段指导搜索。
6. reward仅用于树搜索排序，不覆盖输出 confidence/type，不新增存在性门控。MCTS改变区间后，这个confidence不再声称是对新边界重新校准的概率，输出中标注继承策略。
7. 搜索无效、提案被拒绝、证据不足、搜索预算/候选时间耗尽：保留原检测并分别记录原因。
8. API瞬时失败沿用请求重试；重试耗尽时保留诊断trace，但按现有API补偿契约将视频标为不完整并进入批末重诊断，不能把技术失败混入正常的“提案被拒绝”。禁止让末端模块吞掉该状态。

## 7. 评分、最终去重与评测口径

### 7.1 末端 on/off 的边界消融

`late_boundary_only` 在最终 NMS 前与 `R_cbt` 一一对应：检测ID、confidence、类别、数量相同，只允许区间以及追踪字段变化。明确保存 `R_cbt` 和 `R_late_pre_nms`。

之后分别执行相同 NMS。区间变化会改变IoU和抑制关系，**因此最终 on/off 数量不一定相同**，不能要求NMS后仍一一对应。用ID记录抑制关系，同时报告：

- NMS前按ID的区间改变及定位质量诊断；
- NMS后的正式temporal AP/重复抑制数。

早期分支允许按其正常尾流程产生不同分数、类别和检测集合，不强行复制baseline分数。其指标用于端到端系统比较，不能只用“按索引IoU差”解释。

### 7.2 帧分数

首轮 late继承baseline的frame_scores与timestamps，明确 `frame_score_policy=inherited_pre_late`。不重新运行Q-score、CBT recal或置信度门控，以免混入另一项改动。

有效帧分数相同的late/off应有相同Frame AUC；缺失/长度不足时当前evaluator会按区间补分，必须标注 `frame_score_source=interval_fallback`，不能声称AUC被固定。不得以全零数组替代缺失值来制造相同AUC。

如后续要使帧分数对应最终边界，另设 `recompute_after_late` 实验；baseline与late均按相同新规则重算并单独报告，不能只重算on。

### 7.3 最终NMS与历史数组

最终NMS采用已有规则和阈值0.3，在所有分支一致运行；保持现有置信度稳定排序及该分支输入顺序，ID用于追踪，不在首轮另外按ID重排平局。保存输入顺序；评测文件/视频加载顺序也固定到manifest名单，避免同分数排序漂移。若改变平局规则，单独升级协议并重建基准，不能悄悄归因于末端搜索。

保存按ID的 pre-BUE、pre-CBT、pre-late、post-late边界。NMS筛选/排序后生成与最终输出对齐的兼容数组。**历史边界数组不再能重建完整off结果**，因为搜索/排序/NMS改变了集合；必须保存明确的配对分支，更新相关旧分析脚本的适用性检查。

## 8. 配置与运行模式（均为拟新增接口）

避免把已有`paired_eval`和`postprocess_ablation`不断叠加为笛卡尔积。新增命名实验模式，旧开关仅做兼容映射，冲突时报错。

| 参数/字段 | 初始建议 | 行为 |
|---|---|---|
| `mcts_placement` | `stage5`，保持旧默认 | `stage5` / `post_cbt`；与现有mode=off/shadow/refine配合 |
| `mcts_experiment_mode` | `none` | `none` / `postprocess_ablation`（旧v216）/ `placement_ablation` |
| `mcts_backend` | `v3_boundary` | 首轮只开放当前实现；v1 adapter后续独立添加 |
| `target_category_source` | `shared_video` | 首轮early/late共用VideoMAE目标；`final_detection`另做末端类别来源消融 |
| `candidate_priority` | `contrast_time_id` | 保留当前排序规则，不能给末端暗加最终confidence优先 |
| `root_support_policy` | `legacy` | 保留有效零分根继续探索的旧行为；`supported_only`作为后续独立资格策略 |
| `mcts_search_domain_policy` | `legacy_ratio` | `bounded_context`只在后续域策略消融使用 |
| `late_score_policy` | `inherit_reference` | 不用reward覆盖confidence |
| `late_frame_score_policy` | `inherit_reference` | 显式记录数据来源及fallback |
| `late_delete_policy` | `preserve_input` | 无证据/语义否定不自动删原检测 |
| `mcts_final_nms_enabled` | 实验显式true | 所有分支一致，不能仅给on去重 |
| `mcts_boundary_guard_enabled` | false | 首轮与post_cbt不兼容；拒绝静默启用 |

`mode=off`不构建搜索器、不增加API请求；`mode=shadow`保存末端提案但输出仍等于baseline；`mode=refine`才替换区间。placement_ablation显式构建baseline/early/late三条结果，单独记录每条的配置，不用全局配置突变切换并发视频。

现有 API 并发4、视频并发2、每请求最多3次尝试及随机退避先保持不变。若要比较API并发4/6，另开性能实验，不与位置或算法切换混在一次归因中。

表内无前缀项为拟新增配置字段，归入MCTS实验配置后由CLI统一使用`--mcts_`前缀；具体名称在实现时一次性固定并写入schema，不同时维护多个含义相近的布尔开关。以下是**实现后的预期调用，不是当前可执行命令**：

```powershell
conda run --no-capture-output -n common python `
  D:\Users\zzl\project\mcts_vad\vad_v3\video_anomaly_detection\batch_evaluate.py `
  --dataset_dir F:\dataset\ucf_crime `
  --output_dir D:\Users\zzl\project\mcts_vad\vad_v3\experiments\v217\results `
  --max_per_category 1 --concurrency 2 --api_concurrency 4 `
  --clip_model ViT-L-14 --device cuda `
  --detect_model qwen3-vl-plus --classify_model qwen3-vl-plus `
  --mcts_mode refine --mcts_placement post_cbt `
  --mcts_experiment_mode placement_ablation `
  --mcts_target_category_source shared_video `
  --mcts_action_set shrink_expand_shift --mcts_trigger_policy all_valid `
  --mcts_acceptance_policy relative_nonloss --mcts_trace `
  --mcts_final_nms_enabled --no-mcts_boundary_guard_enabled `
  --mcts_max_images_per_evaluation 32 `
  --mcts_max_boundary_gap_seconds 1.5 --mcts_max_evidence_gap_seconds 3.0 `
  --mcts_max_candidates_per_video 4 `
  --mcts_max_vlm_evaluations_per_candidate 10 --mcts_max_vlm_evaluations_per_video 40
```

实验运行器还须核对14名单与v214/v216一致、65名单与v215一致，保存所选视频绝对路径及索引哈希。实际输出目录在启动时选取未占用编号；上例的v217仅为当前规划。API密钥继续从现有环境配置读取，不写入命令或manifest。

## 9. 运行编排与输出

建议编排伪代码：

```python
snapshot = await detect_to_stage5(video)
if snapshot.early_return:
    return independent_unchanged_arms(snapshot)

cbt = await finish_to_cbt(copy(snapshot), early_refinement=None)
baseline = finalize_result(copy(cbt), copy(cbt.candidates))

late_seeds = adapt_post_cbt(copy(cbt), snapshot.source_registry)
late_outcome = await refine(late_seeds, shared_video_context, budget_for_late)
late = finalize_result(copy(cbt), late_outcome.candidates)

if experiment_mode == 'placement_ablation':
    early_outcome = await refine(adapt_stage5(copy(snapshot)), shared_video_context, budget_for_early)
    early_cbt = await finish_to_cbt(copy(snapshot), early_refinement=early_outcome)
    early = finalize_result(early_cbt, early_cbt.candidates)
return explicit_named_results(baseline, early, late)
```

上述顺序为功能示意；正式顺序写入manifest，预算为各arm独立计数。MCTS实际跑两次（early/late各一次），因为种子不同，不能用同一搜索树假装进行了位置比较。单独production的post_cbt模式不执行early分支。

输出采用版本化schema，不复用含义容易混淆的`original_off`：

```text
paired_results.baseline_cbt
paired_results.early_system
paired_results.late_boundary_only
candidate_registry
boundary_history_by_detection_id
mcts_traces.early
mcts_traces.late
evaluation_protocol
```

顶层结果由所选运行模式明确指定，并记录`primary_arm`；不能只凭是否有`paired_results`猜on是哪一条。早退时每个arm深拷贝，避免嵌套对象共享。

日志继续输出Root/Sim/Search end/API-QUEUE，补充 `arm`、`placement`、`detection_id`、`logical_request_id`、`attempt`、`cache_hit`、`request_role=root/node/pair/repair`。输出域、图片覆盖、原/新边界、接受/拒绝原因、最终抑制关系。

trace与对照结果无API key；不向普通计时日志写完整图片或提示词。需要复现证据时用哈希与本地受控文件引用。

## 10. 缓存、计时、token与实验公平性

- 同一次视频诊断共享只读帧/特征，所有可变候选及尾流程状态分支隔离。
- 只对模型参数、提示词、图片及相关生成选项完全一致的请求复用成功回复；不能复用“区间看起来一样但身份/上下文不同”的评价。
- 缓存键不因为arm名字而强行拆开，但事件记录包含调用arm。跨arm命中计逻辑调用、计搜索逻辑预算，不计新的SDK/token。
- 异步并发共享缓存必须合并同一key的在途请求；失败条目清理，不能把异常当成功回复。候选局部无效证据缓存不得跨reference/arm复用。
- 缓存、队列、视频用量上下文随整视频重诊断重建；成功视频跳过时不能记作本次零成本完成的全量测试。
- 分别记录preprocess、CLIP编码、旧尾流程、early/late MCTS、最终NMS、API排队/请求/退避、候选及视频墙钟。
- API累计等待可能大于视频墙钟，不能把并发请求累计值当作延迟；`search_seconds`也包含排队，不能与queue_seconds再次相加当总时间。
- 功能配对运行共享缓存，先执行arm会承担其它arm后来命中的请求成本，**不能用其分支墙钟或SDK token直接声称哪种位置更快**。总token按整次实验报告，逻辑请求/图像预算按arm报告。
- 比较生产成本时，另跑独立单模式off/early/late，同一名单/配置，或使用冻结Stage5数据进行独立回放；各模式缓存隔离、模型预热一致、API顺序轮换，报告重复试验的中位数和P95。原始模型请求波动不能归因于算法。

## 11. 分文件修改清单

| 文件 | 拟修改内容 |
|---|---|
| `models.py` | 新增内部RefinementSeed/TailSnapshot与稳定ID、来源表、边界历史；可选版本化输出字段，保持旧读者可用。 |
| `pipeline.py` | 拆分finish_to_cbt/finalize；在Stage5保留来源表；选择单模式或三arm编排；统一处理早退；末端无Stage5b绕过。 |
| `mcts_refiner.py` | 使用明确的reference/search_domain/target输入；早期coarse_score检查与末端输入适配分离；返回同ID候选及状态；保留预算、API错误传播和引用校验。 |
| `boundary_search.py` | 不改变第一轮的树搜索算法；域策略独立注入，后续扩大域时同时改legal/grid/动作端点限制。 |
| `mcts_boundary_scorer.py` / `mcts_prompts.py` | reward不当confidence；明确reference可能不完整，来源文本是非可信参考；所有新增/删除区域都映射图片。协议变化时升级prompt_version。 |
| `boundary_dense.py` | 若启用新域，按实际domain＋证据上下文采样；保持独立帧池，不改变baseline。 |
| `post_processor.py` | 稳定最终NMS、按ID审计；原合并保留所有来源ID，避免替换字典丢字段。 |
| `mcts_config.py` / `config.py` | 新位置/实验/域策略配置及兼容检查；默认不改现有stage5行为。 |
| `experiment.py` | 新命名arms、早退深拷贝、分支预算和请求归属、在途缓存合并；manifest记录schema/代码/提示词/域规则。 |
| `llm_usage.py` / `api_recovery.py` | 计时与请求记录增加arm/placement/role；实际API只记一次；耗尽失败保持视频级补偿语义。 |
| `run.py` / `batch_evaluate.py` | 暴露新参数；批评测显式评估每个arm；视频集合与manifest校验；禁止不完整结果静默进入比较。 |
| `evaluate.py` 或独立分析器 | 沿用正式AP定义；新增按ID变化、NMS前后对照、分数来源标注、分支覆盖检查；不为抬高结果改变现有AP算法。 |
| `tests/` | 覆盖下述契约与回归，不依赖付费API。 |

## 12. 测试与验收

### 12.1 实现前必须固定的回归

- `mode=off`与旧完整流程在预测字段上保持一致（忽略新增追踪/耗时字段），最终NMS开关相同。
- `placement=stage5`保持已有搜索结果协议、预算、回退、Stage5b规则；相同模拟响应下不改变预测。
- root/sim/pair/repair、缓存命中和SDK重试计数能相互核对。

### 12.2 新单元/集成测试

1. 末端no-op提案输出与baseline完全一致，包括confidence、类别、数量和分数数组。
2. 有效末端提案替换区间后，BUE/CBT不会再次被调用。
3. 有效零分根在legacy策略下仍可探索；无证据、无可行动作、比较拒绝、预算用尽且没有合格提案时回退，状态可区分。另测supported_only按其显式规则跳过，不能影响legacy回归。
4. API耗尽触发视频补偿；搜索超时取消并回收任务，不残留请求或串用量。
5. 多来源合并、来源缺失、原描述冲突、未知类别不伪造字段；末端类别策略被记录；多类别分组不重置每视频预算。
6. 短区间/视频起止边缘/非均匀时间戳/缺帧/图像上限严格处理；扩大域后合法动作实际可到达。
7. 两个提案搜到同一事件，最终NMS正确抑制；按ID重排历史数组，平局次序稳定。
8. NMS前late/off同ID且同分数，NMS后允许集合不同，不通过数组位置错误配对。
9. 两视频并发不共享可变trace、预算、guard或候选；在途相同请求仅调用一次，失败后可重试。
10. 所有早退路径输出完整独立arms，14/65视频集合一致；`frame_scores=None`的fallback被显式标注。
11. 解析旧结果保持兼容；旧pre_bue/pre_cbt重构脚本对新schema给出明确不适用提示，而非错误重建off。

### 12.3 分阶段实验

**阶段一：14视频功能试验。** 使用原14名单、ViT-L-14、相同VLM、视频并发2/API并发4、同预算，运行baseline/early/late；几何保护关、最终NMS开、旧比例域。检查trace和失败原因。只证明可运行与行为，不据此宣布普遍提升。

**阶段二：固定65视频位置比较。** 参数冻结后运行同一65名单，使用已有`F:/dataset/ucf_crime/data_parquet`索引，不生成新parquet。不因结果不好更换视频。已有v215是参考，不替代本轮共享上游的baseline。14视频是65视频的子集，分别报告14、其余51、全部65；其余51也已在v215参与开发观察，不能仅因不在14子集中就称独立未见验证集。

**阶段三：一次只改一个策略。** 如短区间证据不足，再比较legacy_ratio/bounded_context；如需更保守接受，再比较tie是否允许；若需v1搜索，再做backend消融。不得把扩大域、改reward、重评分、改模型同时归因于末端位置。

**阶段四：生产成本与独立验证。** 单模式重复测时/token；在未参与选择策略的数据上验证。实验目录使用下一个未占用编号，当前可规划v217但启动时重新检查，不覆盖v215/v216。

### 12.4 指标和判断规则

- 主指标：现有全局temporal mAP及AP@0.3/0.5/0.7。该代码是按confidence排序、逐视频匹配GT的11点插值AP，区间变好不保证AP必然变化。
- 分支诊断：NMS前按ID的区间变化、最终检测数、重复抑制、接受率、改善/退化案例、失败/跳过原因；最佳IoU只作辅助，不把重复检测都计作独立真阳性。
- 分类别/正常视频：报告误报和漏检，区分类别正确性与当前temporal AP（当前匹配不要求预测类别与GT相同）。
- 资源：逻辑请求、SDK尝试、图像、已知token、未知usage请求、排队/服务/退避累计、视频墙钟、中位数/P95。
- 统计：对相同视频做配对bootstrap，并保证重采样重复视频拥有独立虚拟ID、GT分母一致；区间跨0则报告证据不足，不能只用平均值正向宣布优越。
- 不设置“必须所有候选都采用”或“必须提高接受率”的目标。优先判断最终定位指标和资源代价。

上线条件：契约测试通过；65视频无不完整样本混入；定位收益可解释且无显著高IoU退化；独立验证支持收益后再考虑改变默认位置。若没有稳定收益，保留post_cbt为可选实验，默认stage5不变。

## 13. v1 搜索作为后续算法消融

适配器必须固定原检测ID/confidence/type，空返回解释为不修改而不是删除；原CBT区间独立保留。v1的收缩奖励、40%浅层步长、扩窗、类别改写都要分别记录。

这已经是 `legacy_v1_adapter`，不能称为未经修改的v1。第一轮比较位置时不引入它；也不在同一个结果中既用v1奖励替换confidence，又声称仅比较边界优化。

## 14. 关键代码依据

- [v1扩窗调用](D:/Users/zzl/project/mcts_vad/vad_v1/video_anomaly_detection/pipeline.py:271)
- [v1动作与收缩](D:/Users/zzl/project/mcts_vad/vad_v1/video_anomaly_detection/temporal_mcts.py:90)
- [v1奖励写入confidence](D:/Users/zzl/project/mcts_vad/vad_v1/video_anomaly_detection/temporal_mcts.py:610)
- [v2帧分数计算位置](D:/Users/zzl/project/mcts_vad/vad_v2/vad-6-8.13-169/video_anomaly_detection/pipeline.py:953)
- [v2 CBT及结果返回](D:/Users/zzl/project/mcts_vad/vad_v2/vad-6-8.13-169/video_anomaly_detection/pipeline.py:981)
- [v3当前refiner](D:/Users/zzl/project/mcts_vad/vad_v3/video_anomaly_detection/mcts_refiner.py)
- [v3结果模型](D:/Users/zzl/project/mcts_vad/vad_v3/video_anomaly_detection/models.py)
- [v3配对缓存](D:/Users/zzl/project/mcts_vad/vad_v3/video_anomaly_detection/experiment.py)
- [评测定义及帧分数fallback](D:/Users/zzl/project/mcts_vad/vad_v3/video_anomaly_detection/evaluate.py)

本文定义了实现和验证范围；不声称末端MCTS已实现或已取得提升。下一次代码任务应按“数据/尾流程拆分 → 输入适配与位置开关 → 分支输出与审计 → 测试 → 14视频 → 冻结参数后65视频”的顺序执行。
