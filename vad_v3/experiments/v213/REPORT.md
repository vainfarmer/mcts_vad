# v213：新版提示词、并发度2验证

## 配置

直接读取现有data_parquet，相同14视频。ViT-L-14/openai（V2一致激活）、CUDA、qwen3-vl-plus。MCTS refine + paired_eval，提示词boundary-v3-2，citation_repair_attempts=1。动作保持shrink，本次未启用shrink_expand_shift。

## 指标

14/14成功，程序退出码0。

| 指标 | off | on |
|---|---:|---:|
| mAP | 0.173737 | 0.173737 |
| AP@0.3 | 0.233333 | 0.233333 |
| AP@0.5 | 0.196970 | 0.196970 |
| AP@0.7 | 0.090909 | 0.090909 |
| 帧 AUC | 0.793025 | 0.793025 |
| 帧 AUPR | 0.489536 | 0.489536 |

skip_only同样相同。视频级bootstrap1000次：on-off mAP差值95%区间[0,0]，因为没有任何提案被采用。

## 接受过程

共23候选；11个开始搜索，合计47次逻辑请求、501张证据图像；最终接受0个。

- 7个 no_valid_proposal：未找到满足完整性/事件/曲线要求的提案；节点评价中出现大量partial。
- 1个 invalid_root：Shooting根回复出现invalid_evidence_indices，未获得有效根评分。
- 3个进入最终比较：Normal、Stealing、Vandalism候选。三个均未选择新提案、完整无损为false、删除内容含目标事件；其中前两个引用合格，Vandalism还缺一段引用。
- 引用纠正触发0次：没有出现“语义全通过但引用有错”的情形。没有为了获得接受而重问语义拒绝。

新版映射已经能产出有效引用，但本次仍无有效接受或mAP收益。只能说明当前收缩提案及完整性判断仍是瓶颈，不能据此声称提示词优化已成功。若进一步检验补全能力，应单独启用已实现的shrink_expand_shift并保持其他配置固定；不得删除完整性要求来增加接受数。

## 并发与用量

并发2；批次墙钟262.29秒（约4.37分钟），视频处理时间合计486.76秒，平均34.77秒/视频。相比v212墙钟375.83秒减少约30.2%，但两次提示词、API耗时和请求数量不同，不能将其当作严格的纯并发加速对照。

199次实际API请求全部成功，无未知usage；264,919 token（输入248,718，输出16,201）。已逐请求核对各视频及batch token总和一致，未出现并发归属串账。耗时和用量包含配对分支，不能作为单独on分支成本。

v212 mAP=0.171236，v213=0.173737；由于两次独立API回复和候选有变化，且本次on/off相同，不能把跨次小幅增加归因于MCTS。

## 文件

- v213_console.log：完整日志，含[MCTS]候选和摘要。
- summary.json / comparison.json：全部指标。
- acceptance_audit.json：搜索候选逐项拒绝记录。
- results/metadata/token_usage.json：batch和视频token。
- COMMAND.json / PLAN.md / results/metadata/manifest.json：可复核配置。
