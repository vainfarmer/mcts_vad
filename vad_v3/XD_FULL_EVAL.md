# XD 完整 800 视频评测与日志

入口 `video_anomaly_detection/xd_full_ablation.py`，本轮输出 `experiments/v220`。使用本地完整测试集（500 异常、300 Normal），不抽样、不读取 v218/v219 的预测结果。配置继承 v219，以便保持相同的检测、MCTS 和尾处理参数。

每视频执行一次 XD VideoMAE 类别先验、CLIP 曲线、窗口 VLM 检查及 CLIP 边界定位；再执行一次共享的 MCTS 搜索。五组分别为 MCTS、MCTS+Q-score、MCTS+Q-score+CBT、MCTS+Q-score+CBT+BUE、MCTS+CBT+BUE；同时保存共享上游的 off 对照。Normal 七类预测沿用已有视频级 Normal 判定及整视频区间规则。

本轮启动参数：视频并发 4，检测/分类共用 API 并发上限 8。CLIP/VideoMAE 的 GPU 预处理在同一锁内执行，以控制显存；等待 API 时允许其他视频继续。MCTS 使用 `shrink_expand_shift`、`force_best`、`all_valid`，每视频最多 4 个候选、40 次逻辑评价，与 v219 一致。

```powershell
# 已在当前进程设置 DASHSCOPE_API_KEY 后，运行或断点续跑：
& 'D:\Users\zzl\project\mcts_vad\vad_v3\run_xd_full_eval.ps1'
```

## 日志

- 批次启动：`VAD batch evaluation started`、类别选取数量、模型加载、视频/API 并发、检测/分类模型。
- 逐视频：`[i/800] START`、`DONE`、`FAIL`、`SKIP`，以及耗时、五组区间数、MCTS 搜索/采用数。
- MCTS：保留 `[MCTS] Root eval`、simulation、结束及采用原因日志。
- 批次结束：Succeeded/Skipped/Failed、总墙钟时间、累计处理时间、成功视频平均耗时、并行加速比。
- 每个分支分别输出 Level 1 帧级 AUC/AP、Level 2 七类时序 mAP 及六类对照、Level 3 二分类视频指标、Level 4 多标签类别指标。
- `[API-QUEUE]`、`[LLM-TOKENS-CALL]` 不输出控制台；每次调用仍写 `metadata/llm_calls_*.jsonl`。视频和批次 token 汇总保留；批次完整逐视频账本写 JSON，不在结尾打印整个 800 视频数组。

XD 使用多标签 GT，类别覆盖数允许重叠，不能将各类别行直接相加当成唯一视频数。日志不会伪造 UCF learned prompt weights 或 MIL 的加载；这两个模块在 XD 分支没有启用。

## 指标口径

- 帧级 AP：所有源视频帧拼接后使用 `average_precision_score`；AUC 为 ROC-AUC。
- 每类别帧级指标：含该 GT 类别的视频子集上的二分类异常分数，子集可以重叠；不是每个类别的逐帧 one-vs-rest 分类器评测。Normal-only 子集没有正样本，AP/AUC 显示 N/A。
- 七类时序 mAP：包含 Normal，IoU 阈值 0.1～0.5；保留六异常类全视频及仅异常视频两份对照。
- 视频级 Accuracy/Precision/Recall/F1：有无最终异常区间；AUC 使用该视频最大源帧异常分数。
- 类别 Overall：预测类别集合与 GT 集合完全相同。每类 `correct/total` 为 TP/GT，另列 precision/recall/F1，允许同视频多个真实类别。
- Normal 预测由模型无异常的判定生成完整视频区间，GT 仅在真实 Normal 视频上生成整段。这不是 VadCLIP 的 Normal 曲线提案复现。

## 保存、恢复与失败处理

每视频一个 `result/<视频名>.json`：五组在 `arms`，off 对照在 `baseline_off`；`anomalies` 是六异常类区间，`detections` 包含可选 Normal。共享先验、曲线、VLM 检查和 MCTS 树保留在文件中。

`metadata/run_status.json` 在每视频结束后更新；`video_attempts_*.jsonl` 保留每次诊断；`upstream/` 保存成功上游检查点。API 随机退避重试后仍失败的视频，以及耗尽 JSON 修复的视频，会在批末补偿一轮。只评估全部成功的视频集合；有失败时不伪报完整 800 视频指标。

断点续跑要求源码、配置、数据文件元信息及标注指纹相同。配置变化应使用新的输出目录。`--evaluate_only` 不调用 API、不覆盖 token 账本。最后输出 `evaluation_report.json` 与 `RESULTS.md`。
