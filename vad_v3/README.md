# V3：V2 pipeline + 局部 MCTS 边界优化

代码位于 `video_anomaly_detection/`，以 V2 为基础保留完整检测尾流程，在 Stage 5 CLIP 验证之后、Stage 5b 之前增加局部边界优化。设计依据为 [MCTS 融合方案](../vad_v2/vad-6-8.13-169/MCTS_FUSION_DESIGN.md)。V1、V2 不加入这条新搜索路径，仅同步 token 统计。

## 使用

先在 common 环境安装 `pip install -r vad_v3/requirement.txt`，配置环境变量 `DASHSCOPE_API_KEY`。从项目根目录执行：

```powershell
conda run -n common python vad_v3/video_anomaly_detection/run.py --video_path "D:\data\example.mp4" --output_file "D:\results\shadow.json" --mcts_mode shadow --mcts_trace
conda run -n common python vad_v3/video_anomaly_detection/run.py --video_path "D:\data\example.mp4" --output_file "D:\results\refine.json" --mcts_mode refine --mcts_trace
```

- `--mcts_mode off`（默认）：原 V2 检测尾流程。
- `shadow`：运行搜索并记录候选、证据、比较和拒绝原因，不修改输出边界。
- `refine`：只应用通过最终比较的边界；这些候选跳过冻结的 Stage 5b 处理，后续合并、NMS、tighten、分类和过滤继续执行。
- `--mcts_search_strategy mcts|greedy|uniform`：使用同一评价器、接受器和逻辑预算。
- `--mcts_paired_eval`：配合 refine 执行 original_off、skip_only、on 完整尾流程，并复用相同请求。
- `--mcts_dense_fps`：默认 0；大于零时创建独立局部密集帧池，不改变原流水线全局采样。密集解码及编码耗时另行记录，不在单候选搜索超时内。

单视频和批次参数均可用 `--help` 查看。批次入口是 `video_anomaly_detection/batch_evaluate.py`。输出包含实验 manifest；配置、代码、模型或视频列表不一致时需使用新的输出目录。

## 实现内容

`mcts_config.py` 定义开关与预算；`mcts_boundary_scorer.py` 提供独立时间加权边界对比、固定核心及变更区域证据校验；`boundary_search.py` 实现三种搜索；`mcts_refiner.py` 控制触发、预算、严格 JSON 解析、最终 A/B 比较及回退。证据覆盖不足、超时、接口失败或比较不通过均保留原候选。

`iteration_logs/analysis_mcts_coverage.py` 用于开发集可达性与证据覆盖检查；`iteration_logs/analysis_mcts_fusion.py` 汇总匹配 manifest 的完整结果并做视频级 bootstrap 对照。搜索不能读取测试集标注。

修正 VideoMAE 参数加载以匹配实际 Transformers state dict，加载失败立即报错；MIL 分支要求 768 维 CLIP 特征，因此 V3 CLI 默认 ViT-L-14。保留旧版搜索实现用于兼容，新的局部搜索走独立路径。

三个版本的 token 统计说明见 [TOKEN_USAGE.md](../TOKEN_USAGE.md)。

## 验证范围

```powershell
conda run -n common python -m pytest tests/test_llm_usage.py vad_v3/video_anomaly_detection/tests -q
```

已完成模拟 API 的搜索、回退、预算、配对及原尾流程等价性测试，以及本机真实 CLIP/VideoMAE/MIL 加载检查、VideoMAE CUDA 前向检查。尚未进行真实 DashScope 调用或全数据集精度评测，不能据此断言 MCTS 提升 mAP。目录中继承自 V2 的历史日志和文档不是 V3 新实验结果。

## v212 激活配置对齐

v3 的 CLIP 已移除 `force_quick_gelu` 强制覆盖，与 v2 保持相同激活配置。14 视频 MCTS 开关配对测试见 [v212 报告](experiments/v212/REPORT.md)。
