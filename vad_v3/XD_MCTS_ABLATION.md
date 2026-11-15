# XD 五组配对消融（v219）

输入为 v218 同一批 35 个视频（六类异常和 Normal 各指定 5 个）。读取 v218 已保存的 `diagnostics.postprocess_off` 作为共同种子；不重新抽样、不重跑 VideoMAE 和窗口分类。重新解码和提取 ViT-L-14 类别曲线，并要求时间戳、时长及原始 frame_scores 复现误差不超过 1e-4。

五组为 `mcts`、`mcts_qscore`、`mcts_qscore_cbt`、`mcts_qscore_cbt_bue`、`mcts_cbt_bue`。每视频只执行一次 MCTS，五组复用相同搜索输出。

## 搜索配置

- `mode=refine`，`action_set=shrink_expand_shift`，`acceptance_policy=force_best`，`trigger_policy=all_valid`。
- 每视频最多 4 个候选、40 次逻辑评价；每候选最多 10 次，12 次 simulation，最大深度 4。现有 refiner 仍保留 1 次比较请求预算，force_best 不使用该预留请求，因此通常每候选最多实际发出 9 次逻辑请求。
- 不同预测类别轮转，类别顺序按预测置信度；类内按边界对比值和时间排序。无效证据不占搜索候选名额。没有候选的视频不凭空创建异常。
- 搜索范围和每端外扩均为种子长度的 25%；端点内移上限 25%。启用 expand 不等于无限扩张。
- 每评价最多 32 图，边界覆盖间隔 1.5 秒、内部覆盖间隔 3 秒，单候选超时 120 秒。
- `force_best` 接收有效的最佳非根节点，不做最终 A/B 比较。仍可能因为根评价无效、无合法动作、证据不足、预算或超时而不采用。不会把失败伪装成 applied。
- 诊断并发 2，API 上限 4。每个候选内部评价有依赖，API 上限不是保证同时 4 个请求。
- MCTS 使用经实际探测可用的 `qwen3-vl-plus`。两个替代模型返回 403，探测记录保存在 metadata。v218 的分类同样使用 qwen3-vl-plus；本轮未重复分类。

## XD 后处理的明确含义

这些是 XD 专用实验适配，不等同于 UCF 中使用类别时长上限的 CBT 或全视频 min/max 的 Q-score。没有使用本批 GT 调参；参数在运行前固定。不会改变原 UCF 流程或普通 XD 默认推理。

顺序固定为：共同 MCTS → 可选 CBT → 可选 BUE → 同类相邻合并 → 可选 Q-score → 同类 NMS。组名表示所含组件，不表示 Q-score 先于边界操作。

- **CBT**：局部类别曲线半峰阈值，只裁掉低支持的首尾背景，每端最多收缩当前区间的 25%；保留区间内所有高支持片段，不只保留最高峰。不使用 UCF 的类别时长 cap。
- **BUE**：沿当前左右边界连续的同类支持向外扩，遇到低分帧停止；每端最多当前长度的 50%，且不超过 8 秒。属于视觉曲线启发式，不能证明外扩部分一定是同一真实事件。
- **Q-score**：使用未叠加检测置信度的类别 CLIP 曲线。固定尺度的内部支持、内外对比、边界对比三项取平均；`confidence *= 0.5 + 0.5*Q`。不重新标注类别、不额外按置信度过滤。所有几何变化和相邻合并结束后计算，避免分数对应旧区间。
- **NMS**：IoU 阈值 0.5，仅同类抑制；不同类别可以同时出现。Q-score 改变排序也可能改变 NMS 留下的区间。

## 文件与复现

`experiments/v219/result/<video_id>.json` 每视频一个文件：`baseline_off` 是 v218 引用对照；`shared.seed_events`、`shared.mcts_events`、`shared.mcts_trace` 为共享证据；`arms.<组名>.anomalies` 为该组最终结果。每组保存 operator audit、source_candidate_ids、最终 frame_scores 和 operation_order。

`evaluation_report.json` 含五组及 v218 对照的原生 FPS 时间 IoU AP、六类 mAP、帧级 AP/AUC、区间数量和操作次数；metadata 保存每组匹配审计、配置、上游哈希、源码快照、token 统计和任务状态。

```powershell
# 在已设置 DASHSCOPE_API_KEY 的 PowerShell 进程中运行：
& 'D:\Users\zzl\project\mcts_vad\vad_v3\run_xd_mcts_ablation.ps1'
```

入口是 `video_anomaly_detection/xd_mcts_ablation.py`，可通过 `--source_dir` 和 `--output_dir` 指定新实验。相同源码/配置支持跳过已完成视频；任何 manifest 不一致都会拒绝混入。`--evaluate_only` 仅重算指标，不调用 API，也不覆盖 token 账本。

这是已观察过的 35 视频上的探索性配对消融，不能作为完整 800 视频测试集性能。类别错误和上游漏检不会被只调整边界的 MCTS 修复；Normal 没有候选时各组结果自然相同。

## 将 Normal 纳入第七个检测类别

已新增 `experiments/v219/reevaluation_normal7/`，保留原六类报告。重评入口：

```powershell
conda run -n common python -m evaluation_tools.reevaluate_xd_seven_class --experiment vad_v3/experiments/v219 --dataset_root F:/dataset/xd-violence
```

Normal 预测由原结果的 `is_normal=True` 生成整段区间，置信度为 `1 - 全源视频帧的平均异常分数`；预测函数不接收 GT 或视频文件名。真实 Normal 视频的 GT 为整段，异常视频中的正常背景不另外标 Normal。所有输入视频统一应用该预测规则。

七类结果的逐视频字段为 `arms.<组名>.detections`，同时含六类异常检测和可选 Normal 检测。原 `anomalies` 未被改写。七类均值与六类均值是不同指标，不应将加入 Normal 后的数值上涨解释为异常定位能力提升。本适配没有复现 VadCLIP 的 Normal 曲线阈值提案。
