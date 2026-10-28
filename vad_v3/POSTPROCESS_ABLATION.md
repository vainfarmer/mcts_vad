# 搜索日志、API排队计时与后处理消融

## 日志与计时

- `[MCTS] Root eval`：根节点评价。
- `[MCTS] Sim i/N`：每次子节点扩展评价，包含候选、视频、深度、区间、奖励及缓存命中。无效评价显示原因，不伪装成零分。
- `[MCTS] Search end`：实际子节点评估数、总评价数、停止原因、接受结果与耗时。根评价不计入simulation数；A/B比较也不是子节点simulation。
- `[API-QUEUE]`：每次SDK尝试取得共享API槽位前的等待；重试重新排队。取消排队不虚增请求次数。
- 后处理计时保留在JSON trace的`stage_seconds`和`tail_seconds`中，包含阶段内的异步等待；不再逐阶段打印`[TAIL-TIMING]`到控制台/run.log。
- `[BOUNDARY-GUARD]`：保护回退阶段、来源候选、拟修改区间、恢复区间及原因。
- `[FINAL-NMS]`：所有边界处理之后的去重前后数量。

逐API调用的`queue_wait_seconds`和`elapsed_seconds`分别记录排队与SDK调用。视频/批次token统计新增`api_queue_seconds`、`api_request_seconds`、`api_backoff_seconds`和`api_cancelled_queue_seconds`。这些都是并发任务的累计时间，不能直接相加作为墙钟耗时或独立生产分支成本。

## 开关

以下参数同时支持单视频和批处理入口，统一使用现有`--mcts_`配置前缀；最终NMS对off/on均生效。

| 参数 | 默认 | 作用 |
|---|---|---|
| `--mcts_final_nms_enabled` | false | 在BUE/CBT和保护之后增加最终NMS |
| `--mcts_final_nms_iou` | 0.3 | 最终NMS的IoU阈值 |
| `--mcts_boundary_guard_enabled` | false | 保护MCTS已接受候选的区间 |
| `--mcts_guard_min_retained_ratio` | 0.5 | 至少保留已接受区间的比例 |
| `--mcts_guard_max_shift_ratio` | 0.25 | 相对已接受来源包络长度的端点位移上限 |
| `--mcts_guard_tolerance_seconds` | 0.5 | 端点位移最小容差、证据范围及锚点检查容差 |
| `--mcts_postprocess_ablation` | false | 保存旧流程、仅去重、去重加保护分支 |

布尔开关可用`--no-mcts_final_nms_enabled`、`--no-mcts_boundary_guard_enabled`显式关闭。消融要求同时启用`--mcts_paired_eval`、`--mcts_mode refine`、最终NMS和保护。

初始保留默认关闭，避免未经验证改变全部实验。v216显式启用进行消融，不改变v215原始结果。

`run_relaxed_search.ps1` 已显式启用最终NMS：v215冻结的65视频离线对照中，off的mAP从0.152019提升到0.152725，on从0.155292提升到0.155802。底层配置仍允许关闭以复现旧流程。边界保护不在该常规入口自动启用，需依据独立消融结果判断。

v216已完成14/14：旧on与仅去重on均为mAP 0.169408；去重加保护降至0.161616。故当前保留保护开关及审计，但不推荐默认启用。报告位于`experiments/v216/REPORT.md`。

## 保护规则

MCTS接受提案时保存接受区间、A/B证据时间范围、接受区间内的CLIP曲线峰值锚点。锚点是启发式定位线索，并非VLM逐帧认证。后处理在Stage5b、PostTighten、短事件收缩、CLIP修正、BUE及CBT之后检查：

1. 时间区间有效。
2. 端点相对已接受来源的包络位移不过大。
3. 区间不超出证据时间包络及容差。
4. 每个已接受来源至少保留指定区间比例，并覆盖其曲线峰值锚点。

不满足时恢复该阶段修改前的完整检测，包含置信度，避免保留基于被拒绝区间计算的CBT校准分数。不会新增API请求，不会根据GT判断是否回退。

通过`source_candidate_ids`跟踪合并来源。多来源合并时端点/证据包络纳入全部来源，包括未搜索候选；同时逐个检查已保护候选的保留率和锚点。来源匹配缺失会记录诊断信息，不按列表下标猜测对应关系。

最终NMS保留较高置信度检测，不扩大区间；对`pre_bue_boundaries`、`pre_cbt_boundaries`同步筛选和排序。启用保护后，不能用这两个历史数组代替明确运行的消融分支。

## 配对分支

- `paired_results.legacy_off`：无MCTS、旧后处理。
- `paired_results.legacy_on`：有MCTS、旧后处理。
- `paired_results.original_off`：无MCTS、增加最终去重（分析中命名dedup_off）。
- `paired_results.dedup_on`：有MCTS、增加最终去重。
- 顶层结果：有MCTS、增加最终去重和边界保护（分析中命名guard_on）。
- `paired_results.skip_only`：保留原边界，仅采用MCTS接受集合的Stage5b绕过策略，并增加最终去重。

同次诊断共享上游及一次MCTS搜索，候选和分支可变状态深拷贝隔离，已完成的视频帧/CLIP特征只读共享，相同API请求复用成功回复。早退视频也保存所有分支，避免只评估进入MCTS的子集。

帧级分数继续沿用旧流程的post_conf_gate计算位置。最终NMS不会重算帧级分数；其直接影响主要在最终检测集合及temporal AP。为单独验证去重与保护，本次不同时改变帧级评分定义。

## 配对分支的数据复制优化（2026-10-28）

`video_state.fork_video_state()`替代整份`video_data`深拷贝。普通paired模式的三个尾分支、完整消融的六个尾分支均使用该函数；非paired模式继续直接使用本视频上下文。

- 共享白名单：`all_frames`、`timestamps`、`clip_features`、`motion_energy`、`mil_scores`、`tsm_novelty`，以及已有`_cat_scores_*`数值曲线。
- NumPy数组使用共享存储的只读view，不修改源数组的可写标志；object数组不共享。
- PyTorch的`clip_features`共享存储。Tensor没有只读标志，尾流程必须遵守只读契约；需要原地修改的新代码应先`clone()`。
- 分支字典、segments及其嵌套列表、未知字段、未知数组仍深拷贝。新增类别缓存写入本分支字典，不传播给其它分支或原始上下文。
- 候选、来源、trace和保护状态继续独立，未修改搜索、评分或边界规则。

已新增5项测试，覆盖共享存储/只读保护、可变缓存与未知字段隔离、真实FrameScorer和尾流程预测等价、六分支预测等价、异步分支隔离；全套125项测试通过。这里的等价验证使用模拟API，不发起付费评测。

六尾流程消融中，旧实现复制5份完整视频数据，on尾流程使用原数据；新实现6个分支均共享大数组。以64帧360×640 RGB及对应CPU特征约42.38 MiB为例，可避免每视频约211.88 MiB的累计载荷复制。这是合成数据的复制量说明，不代表进程峰值内存下降同样数值，也不是端到端视频耗时提升的测量。

## v216复现

在common环境执行`python vad_v3/experiments/v216/run_experiment.py`。实际命令保存于该目录的`COMMAND.json`，参数固定为之前的14个视频、ViT-L-14、CUDA、视频并发2、API并发4及v214的宽松MCTS配置。请保留`.env`中的API配置。

完成后执行`python vad_v3/experiments/v216/analyze_ablation.py`，只读结果、离线生成`REPORT.md`、`ablation.json`和各分支导出文件。勿在旧实验目录用不同源码或参数续跑；manifest会拒绝混用。
