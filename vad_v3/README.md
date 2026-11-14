# V3：V2 pipeline + 局部 MCTS 边界优化

代码位于 `video_anomaly_detection/`，以 V2 为基础保留完整检测尾流程，在 Stage 5 CLIP 验证之后、Stage 5b 之前增加局部边界优化。设计依据为 [MCTS 融合方案](../vad_v2/vad-6-8.13-169/MCTS_FUSION_DESIGN.md)。V1、V2 不加入这条新搜索路径，仅同步 token 统计。

## 使用

### 本地视频分析系统

已接入 SQLite、独立串行 Worker、视频库、任务中心、过程与复核、实验评价和报告导出。使用下面的控制器同时启动两个服务：

~~~powershell
.\vad_v3\webapp\scripts\start.ps1
.\vad_v3\webapp\scripts\status.ps1
.\vad_v3\webapp\scripts\stop.ps1
~~~

访问 http://127.0.0.1:7860。Web 新任务默认 CUDA、MCTS refine、接收策略 force_best；命令行及算法配置的默认接收策略也为 force_best。研究参数中可显式切换其他策略，四组对比的各分支独立指定策略，历史任务使用其保存的配置。VLM 原始描述不要求中文。密钥由 Web 和 Worker 分别从 webapp/.env 加载，修改后重启。详见 [操作与接口手册](webapp/README.md) 和 [本次实施验收记录](../docs/系统实施与验收记录.md)。

2026-11-11 已重新确认本机 PyTorch 2.7.1+cu128 可导入，RTX 5060 Ti 的 CUDA 推理可用；此前 WinError 4551 环境障碍已不再复现。

### 命令行

先在 common 环境安装 `pip install -r vad_v3/requirement.txt`，配置环境变量 `DASHSCOPE_API_KEY`。从项目根目录执行：

```powershell
conda run -n common python vad_v3/video_anomaly_detection/run.py --video_path "D:\data\example.mp4" --output_file "D:\results\shadow.json" --mcts_mode shadow --mcts_trace
conda run -n common python vad_v3/video_anomaly_detection/run.py --video_path "D:\data\example.mp4" --output_file "D:\results\refine.json" --mcts_mode refine --mcts_trace
```

- `--mcts_mode off`（默认）：原 V2 检测尾流程。
- `shadow`：运行搜索并记录候选、证据、比较和拒绝原因，不修改输出边界。
- `refine`：应用符合所选接收策略的边界，默认 force_best 直接接收最佳有效非根节点，不执行最终 A/B 比较；这些候选跳过冻结的 Stage 5b 处理，后续合并、NMS、tighten、分类和过滤继续执行。
- `--mcts_search_strategy mcts|greedy|uniform`：使用同一评价器、接受器和逻辑预算。
- `--mcts_paired_eval`：配合 refine 执行 original_off、skip_only、on 完整尾流程，并复用相同请求。
- `--mcts_stage4_direct_eval`：仅用于配对实验，与 refine、paired_eval 同用。两组共享 Stage 4 的全部原始候选，一组直接 BUE→CBT，另一组先运行原有 MCTS 再 BUE→CBT；跳过 Stage 5、中间精修、top-k、NMS、类别复核、置信度过滤及 QSCORE。置信度固定为发现分数，类别固定为上游预测；不改变默认完整流程。
- `--mcts_acceptance_policy relaxed_node`：接收消融，保留节点目标存在、同一事件和 CLIP 质量下降上限，取消完整性/切断边界否决及最终 A/B 比较。每个候选采用最高奖励的合格非根节点。
- `--mcts_acceptance_policy force_best`（默认）：强制采用每个候选最高奖励的有效非根节点，允许低于根奖励；取消额外语义、CLIP 质量和最终 A/B 接收门槛。失败/未评分节点、非法区间仍不采用。force_best 和 relaxed_node 都要求 `comparison_gate=best_valid`，保留搜索阶段预算和证据检查。
- `--mcts_stage4_acceptance_eval`：在 Stage 4 配对实验基础上同时运行 off、strict、relaxed_node、force_best 四组。要求同时开启 refine、paired_eval、stage4_direct_eval，并显式传入 `--mcts_acceptance_policy strict`，comparison_gate 为 best_valid。off 在搜索前完成固定后处理；三个搜索分支独立运行各自策略，仅完全相同的 VLM 请求共享回复。分支耗时受缓存影响，不能作为独立部署耗时对照。
- `--mcts_dense_fps`：默认 0；大于零时创建独立局部密集帧池，不改变原流水线全局采样。密集解码及编码耗时另行记录，不在单候选搜索超时内。

单视频和批次参数均可用 `--help` 查看。批次入口是 `video_anomaly_detection/batch_evaluate.py`。输出包含实验 manifest；配置、代码、模型或视频列表不一致时需使用新的输出目录。

## 实现内容

`mcts_config.py` 定义开关与预算；`mcts_boundary_scorer.py` 提供独立时间加权边界对比、固定核心及变更区域证据校验；`boundary_search.py` 实现三种搜索；`mcts_refiner.py` 控制触发、预算、严格 JSON 解析、按策略接收及回退。默认 force_best 不执行最终 A/B 比较；无有效候选、超时或根评价失败时保留原候选。

`iteration_logs/analysis_mcts_coverage.py` 用于开发集可达性与证据覆盖检查；`iteration_logs/analysis_mcts_fusion.py` 汇总匹配 manifest 的完整结果并做视频级 bootstrap 对照。搜索不能读取测试集标注。

修正 VideoMAE 参数加载以匹配实际 Transformers state dict，加载失败立即报错；MIL 分支要求 768 维 CLIP 特征，因此 V3 CLI 默认 ViT-L-14。保留旧版搜索实现用于兼容，新的局部搜索走独立路径。

三个版本的 token 统计说明见 [TOKEN_USAGE.md](../TOKEN_USAGE.md)。

## 评估口径

V3 的 evaluate.py 主报告采用 VadCLIP 的非插值 AP：按置信度排序并逐类一对一匹配，AP = sum(Precision@k × TP_k) / N_GT。预测秒数按各视频真实 FPS 转为整数半开帧区间，主 mAP 对 13 个异常类别平均，IoU 为 0.1—0.5；正常视频误报单独统计。已有最终区间和置信度保持原样，不重新执行 VadCLIP 的候选生成。

排序使用 VadCLIP 官方的 `np.argsort(-confidence)` 默认算法，不追加视频 ID 或区间边界作为并列排序键。输入先按视频 ID 汇总，并保留视频内预测顺序；相同置信度的排列可能受输入顺序、数量及 NumPy 版本影响。`compute_vadclip_map(..., include_normal_videos=True)` 将正常视频的异常预测纳入对应类别 FP，仍对 13 个异常类别平均。六组全 290 视频实验保存在 `experiments/stage4_six_groups_290/`，同时复算 140 异常视频子集，以分离排序变化和正常视频误报的影响。

每次评估直接打开结果 `video_path` 指向的原视频读取 FPS 和帧数，不读取 FPS map/清单文件。视频搬迁后使用 `--video_root F:/dataset/ucf_crime`；文件缺失、无法解码或同名定位有歧义时明确报错。`--legacy_fixed_fps --fps 30` 仅用于显式复现旧固定帧率指标。各轮 Stage 4 对比汇总脚本同样重新读取原视频，输入清单中的历史 FPS 只作记录。

temporal_map 是上述类别敏感主指标；temporal_map_class_agnostic 是保留 0.3/0.5/0.7 阈值和原匹配范围的辅助指标，也已取消 11 点插值。历史报告不被覆盖，旧实验脚本再次运行时会使用新公式。V1/V2/V4 的独立评估文件及显式历史复现工具不在此次修改范围内。

2026-11-11 的 [Stage 4 配对实验报告](experiments/stage4_fixed_vs_mcts/report.html) 使用 13 类异常各一个视频及一个正常视频，全部 14 个执行成功。直接 BUE→CBT 与先进行原有 shrink MCTS 再 BUE→CBT 的最终区间完全一致：AVG mAP 均为 27.56%，mAP@0.5 均为 9.62%。19 个候选实际进入搜索，83 次逻辑评估，接受 0 个修改；累计候选搜索耗时 262.07 秒。正常视频两组均有 1 个误报。该结论仅适用于本轮小样本、收缩动作和严格接受配置，不代表全部 MCTS 配置；本实验路径未设为默认。

复现命令、配置、输入文件哈希、实际执行源码及逐视频结果保存在该实验目录。可运行 `python vad_v3/experiments/stage4_fixed_vs_mcts/compare.py` 重新汇总已有结果，无须再次调用 VLM。

随后完成的 [接收条件消融报告](experiments/stage4_acceptance_ablation/report.html) 复用上述 14 个视频的原始候选、搜索树和 VLM 评分，仅改变接收规则，并重建真实 CLIP 曲线、重新执行原有 BUE→CBT。直接后处理与严格接收基线逐视频精确复现。`relaxed_node` 与 `force_best` 均接受 19 个提案，输出完全一致：AVG mAP 为 35.26%（+7.69 个百分点），mAP@0.5 为 17.31%。其中 9 个修改被固定后处理恢复到原最终边界；8 个视频最终区间改变。按不计类别的最佳 tIoU 诊断，13 个异常视频中 3 个改善、1 个退化、9 个不变；AP 收益来自 Explosion 和 Fighting 两个样本。新 VLM 请求为 0。该结果属于固定搜索树回放，不是不同策略重新搜索后的整流程测试，不能用来比较独立推理耗时；仍需扩大样本，未改默认接收策略。

重新汇总：`python vad_v3/experiments/stage4_acceptance_ablation/compare.py`。重建曲线及回放：`python vad_v3/experiments/stage4_acceptance_ablation/run_replay.py`（需要 CUDA 和原视频，自动暂时释放空闲 Worker 的 GPU 锁）。

2026-11-11 随后完成 [65 视频四组实时配对实验](experiments/stage4_acceptance_65/report.html)：60 个异常视频、5 个正常视频，全部成功。四组共享 Stage 4 上游，三个搜索策略分别执行，仅精确相同的请求复用回复。关闭 MCTS / strict 的 AVG mAP 均为 23.88%，relaxed_node / force_best 均为 27.40%（+3.51 个百分点）；mAP@0.5 从 9.65% 升至 12.20%，mAP@0.1 从 42.83% 降至 41.95%。strict 接受 0/89 个已搜索候选，放宽和强制均接受 81/89，二者搜索后及最终区间完全相同；26 个修改被固定后处理恢复，36 个视频最终边界改变。60 个异常视频的最佳 tIoU 为 16 个改善、9 个退化、35 个不变。新增 51 视频子集 AVG 为 25.05% → 26.77%；不能与原 14 视频子集简单加权为总体 mAP。四组均在 5 个正常视频中误报 4 个。全批实际耗时约 45.2 分钟，989 次实际 API 请求、1,506,000 Token；共享缓存使后两组的新增请求为零，这不是独立运行成本。默认策略仍为 strict。

该轮命令、输入哈希、源码快照和逐视频结果均在 `experiments/stage4_acceptance_65/`；运行 `python vad_v3/experiments/stage4_acceptance_65/compare.py` 可重新汇总，`run_experiment.py` 可按匹配 manifest 断点续跑。65 视频包含此前试验样本，不是完整测试集或独立留出集。

## 验证范围

```powershell
conda run -n common python -m pytest tests/test_llm_usage.py vad_v3/video_anomaly_detection/tests -q
conda run -n common python -m pytest tests/test_native_fps_ap.py vad_v3/webapp/tests -q
```

已完成模拟 API 的搜索、回退、预算、配对及原尾流程等价性测试，以及本机真实 CLIP/VideoMAE/MIL 加载检查、VideoMAE CUDA 前向检查。实验目录保留各轮真实推理结果，具体配置和执行源码以对应记录为准；历史实验不代表当前启用功能，也不能据此断言当前 MCTS 提升 mAP。目录中继承自 V2 的历史日志和文档不是 V3 新实验结果。

## v212 激活配置对齐

v3 的 CLIP 已移除 `force_quick_gelu` 强制覆盖，与 v2 保持相同激活配置。14 视频 MCTS 开关配对测试见 [v212 报告](experiments/v212/REPORT.md)。
