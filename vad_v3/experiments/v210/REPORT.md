# v210 评测结论

14 类各 1 个视频，ViT-L-14/openai，qwen3-vl-plus，CUDA。配对对照共享上游检测和完全相同请求的回复。

**本次没有观察到 MCTS 带来性能改善：24 个候选中 0 个边界修改被接受，off / skip_only / on 的评测指标相同。**

| 指标 | off | on |
|---|---:|---:|
| Temporal mAP | 0.062290 | 0.062290 |
| AP@0.3 | 0.106061 | 0.106061 |
| AP@0.5 | 0.075758 | 0.075758 |
| AP@0.7 | 0.005051 | 0.005051 |
| 帧 AUC | 0.785573 | 0.785573 |
| 帧 AUPR | 0.400629 | 0.400629 |
| 视频准确率 | 0.928571 | 0.928571 |

## MCTS 运行情况

44 次逻辑请求，发送 462 张证据图像，候选搜索累计约 121.4 秒。拒绝/跳过原因：

- `boundary_gap_left_out`：2
- `image_budget_insufficient`：9
- `boundary_gap_right_in`：2
- `missing_changed_region_citation`：8
- `contrast_not_selected`：1
- `no_evidenced_action`：1
- `coverage_gap_0_inside`：1

## 用量与运行状态

14/14 推理成功，181 次 API 请求全部成功，总计 233,664 token（输入 219,990；输出 13,674）。批处理推理耗时 340.1 秒，平均每视频 24.3 秒。时间与 token 覆盖全部配对分支，不能直接视为单独 on 分支成本。

原批次于 2026-10-27 12:49:38 完成推理，但指标计算读取 UTF-8 JSON 时使用 Windows GBK，导致 UnicodeDecodeError。已修复 evaluate.py 的读取编码，使用已有 14 个结果离线补算全部指标，无新增 API 调用。原日志、exit_status.json、token 汇总中的 error 状态均保留；恢复状态见 post_eval_status.json。源代码修复发生在推理之后，不修改原 manifest。

这只是 14 视频小样本；不能推断所有数据上的 MCTS 效果。下一步应在开发数据上检查证据预算与模型引用格式，再预设新配置进行独立评测，不能因本次结果直接放宽测试集接受门槛。

## 每视频用量

| 类别 | 视频 | token |
|---|---|---:|
| Abuse | Abuse030_x264 | 12881 |
| Arrest | Arrest007_x264 | 7664 |
| Arson | Arson035_x264 | 20525 |
| Assault | Assault006_x264 | 16314 |
| Burglary | Burglary079_x264 | 14421 |
| Explosion | Explosion002_x264 | 16245 |
| Fighting | Fighting018_x264 | 10574 |
| Normal | Normal_Videos_782_x264 | 18804 |
| RoadAccidents | RoadAccidents123_x264 | 10195 |
| Robbery | Robbery050_x264 | 18832 |
| Shooting | Shooting032_x264 | 34133 |
| Shoplifting | Shoplifting001_x264 | 18507 |
| Stealing | Stealing036_x264 | 27035 |
| Vandalism | Vandalism015_x264 | 7534 |
