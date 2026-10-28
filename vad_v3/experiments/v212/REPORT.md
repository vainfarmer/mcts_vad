# v212：V3 对齐 V2 CLIP 激活后的 MCTS 配对评测

## 结果

14/14 成功；关闭、开启 MCTS 及 skip_only 三个分支结果指标相同。本次没有观察到 MCTS 带来收益。

| 指标 | MCTS off | MCTS on |
|---|---:|---:|
| mAP | 0.171236 | 0.171236 |
| AP@0.3 | 0.232323 | 0.232323 |
| AP@0.5 | 0.190476 | 0.190476 |
| AP@0.7 | 0.090909 | 0.090909 |
| 帧 AUC | 0.793339 | 0.793339 |
| 帧 AUPR | 0.489890 | 0.489890 |
| 视频准确率 | 0.928571 | 0.928571 |

CLIP 已移除 force_quick_gelu，与 V2 的 create_model_and_transforms 调用参数完全一致；ViT-L-14/openai、CUDA、qwen3-vl-plus。直接读取 F:/dataset/ucf_crime/data_parquet，启动时核对为相同14个视频，无新建 parquet。

与前次比较：v210 mAP=0.062290；本次 v212=0.171236，接近 v211 V2=0.169408。该结果支持 CLIP 激活配置是重要差异因素，但跨次 VLM 回复并未固定，不能把全部差值严格归因于激活配置。当前开启/关闭对照共享上游与完全相同请求缓存，更适合判断 MCTS 的实际影响。

## MCTS 触发与回退

22 个候选，最终接受修改 0 个；55 次逻辑请求，558 张证据图像，候选搜索累计约166.67秒。视频级配对bootstrap 1000次的mAP差值95%区间为[0,0]，这反映当前输出相同，不代表所有配置或全数据集上 MCTS 都无效。

- `boundary_gap_left_out`：2
- `boundary_gap_right_in`：2
- `missing_changed_region_citation`：7
- `image_budget_insufficient`：4
- `comparison_rejected`：3
- `contrast_not_selected`：2
- `no_evidenced_action`：1
- `coverage_gap_0_inside`：1

## 时间与用量

批次耗时375.83秒，平均26.84秒/视频。207次API请求全部成功，262,817 token（输入247,401，输出15,416），没有未知usage。时间与token为包含配对分支的整次实验，不能直接作为单独on分支的部署成本。

## 文件

- v212_console.log：完整执行日志。
- results/metadata/manifest.json：配置、模型和源码指纹。
- comparison.json：三分支指标和bootstrap。
- summary.json：包含逐视频指标和token的汇总。
- results/metadata/token_usage.json：批次用量。
- selected_videos.json：本次视频及原始标注。

此次只改变CLIP激活配置，不根据测试标注调优MCTS预算或接受条件。后续应先在开发数据上解决证据覆盖和变更区域引用问题，再评测边界优化效果。
