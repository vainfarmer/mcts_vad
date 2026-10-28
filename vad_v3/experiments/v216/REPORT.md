# v216 后处理配对消融

同一批14视频，共享上游和MCTS搜索；相同API请求复用。legacy为旧尾流程，dedup增加最终NMS，guard在dedup基础上增加边界保护。

| 分支 | mAP | AP@0.3 | AP@0.5 | AP@0.7 | Frame AUC | 检测数 |
|---|---:|---:|---:|---:|---:|---:|
| legacy_off | 0.169408 | 0.226840 | 0.190476 | 0.090909 | 0.792614 | 18 |
| legacy_on | 0.169408 | 0.226840 | 0.190476 | 0.090909 | 0.792233 | 18 |
| dedup_off | 0.169408 | 0.226840 | 0.190476 | 0.090909 | 0.792614 | 16 |
| dedup_on | 0.169408 | 0.226840 | 0.190476 | 0.090909 | 0.792233 | 16 |
| guard_on | 0.161616 | 0.217631 | 0.176309 | 0.090909 | 0.792233 | 16 |
| skip_only | 0.169408 | 0.226840 | 0.190476 | 0.090909 | 0.792257 | 16 |

## 控制变量比较

- legacy_on_minus_legacy_off: mAP差值 +0.000000，边界或数量变化视频 3。
- dedup_off_minus_legacy_off: mAP差值 +0.000000，边界或数量变化视频 2。
- dedup_on_minus_legacy_on: mAP差值 +0.000000，边界或数量变化视频 2。
- guard_on_minus_dedup_on: mAP差值 -0.007792，边界或数量变化视频 4。
- guard_on_minus_dedup_off: mAP差值 -0.007792，边界或数量变化视频 5。

## 保护与耗时

接受候选数：8；子节点评估次数：102。
保护回退阶段：{'post_bue': 4, 'post_cbt': 1}。原因计数（一次可有多个原因）：{'endpoint_shift': 4, 'outside_evidence': 4, 'insufficient_retained_interval': 1}。

API时间均为所有请求的累计秒数，不等于墙钟耗时；请求等待已在CALL日志queue_wait_seconds独立记录。
```json
{
  "requests": 274,
  "successful_requests": 274,
  "failed_requests": 0,
  "unknown_usage_requests": 0,
  "prompt_tokens": 394261,
  "completion_tokens": 23793,
  "total_tokens": 418054,
  "api_queue_seconds": 783.8010599998524,
  "api_request_seconds": 1074.4170999999997,
  "api_backoff_seconds": 0.0,
  "api_cancelled_queue_seconds": 0.0,
  "by_model": {
    "qwen3-vl-plus": {
      "requests": 274,
      "successful_requests": 274,
      "failed_requests": 0,
      "unknown_usage_requests": 0,
      "prompt_tokens": 394261,
      "completion_tokens": 23793,
      "total_tokens": 418054,
      "api_queue_seconds": 783.8010599998524,
      "api_request_seconds": 1074.4170999999997,
      "api_backoff_seconds": 0.0,
      "api_cancelled_queue_seconds": 0.0,
      "by_model": {}
    }
  }
}
```

Frame scores沿用原有post_conf_gate计算位置；BUE/CBT之后的最终NMS不重新计算frame scores。因此边界或重复检测改善可能不改变Frame AUC。
本实验验证14视频，不能据此确认65视频或完整测试集收益。总token属于整个多分支实验，不能视为单个on/off生产分支成本。
最终NMS后的pre_bue/pre_cbt数组已按保留结果重新排列，但开启保护时历史边界不再能直接重构“关闭保护”分支；应使用明确保存的消融分支。

完整逐视频区间见ablation.json；comparison_arms下保存每个分支的完整结果。

## 当前配置建议

常规run_relaxed_search.ps1启用最终NMS，边界保护保持关闭。14视频最终NMS减少2条检测且mAP不变；v215冻结65视频的离线对照中，on mAP由0.155292升至0.155802，off由0.152019升至0.152725，详情见offline_nms_65.json。
当前几何保护使mAP从0.169408降至0.161616，不应默认启用，也不在本次小样本上反复调阈值追求提升。
Arson035的最终区间从4.5–45回退为22–33.5，最佳GT IoU从0.2469升至0.5926；但Vandalism015从55.5–90回退为55.5–67，IoU从0.6667降至0。说明已接受的局部区间可能仍缺失事件，不适合作为所有后处理的硬上限。
后续应优先考虑条件式保护：大幅迁移时重新核对事件身份/核心证据，而非仅凭几何范围否定BUE补全。该方向尚未实现，当前保留关闭开关和明确的消融结果。
