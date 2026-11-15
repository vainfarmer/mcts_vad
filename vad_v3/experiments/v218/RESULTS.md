# v218：XD-Violence 五视频/类别适配验证

正式测评 35/35 成功，0 个视频失败。六类异常和 Normal 各指定 5 个视频，固定 seed=42。
多标签视频不重复执行；实际 Fighting 覆盖 7 个、Shooting 8 个，其余各 5 个。
本轮 MCTS 关闭，CLIP 为 ViT-L-14，分类 VLM 为 qwen3-vl-plus，视频并发 2、API 并发 4。
检测/MCTS 客户端配置为 qwen2.5-vl-72b-instruct，但本轮没有调用，不能据此认定其 API 可用。

## 整体指标

| 指标 | 结果 |
|---|---:|
| 帧级 AP | 64.31% |
| 帧级 AUC | 73.85% |
| mAP@0.1 | 36.87% |
| mAP@0.2 | 25.99% |
| mAP@0.3 | 16.15% |
| mAP@0.4 | 7.54% |
| mAP@0.5 | 3.55% |
| AVG | 18.02% |
| Normal 视频误报 | 0/5 |
| GT / 预测区间数 | 72 / 257 |
| 类别集合完全匹配的视频 | 10/35 |
| 测评总墙钟时间（不含模型初始化） | 374.66 秒 |
| 平均单视频诊断耗时（含排队） | 20.82 秒 |

Normal 视频没有异常预测，因此本轮 full 与 abnormal-only 的 temporal mAP 相同。
以上都是这 35 个视频的结果，不是完整 800 视频的 benchmark。帧级 AP 不含 tIoU；temporal mAP 按类别和区间匹配。

## 类别定位结果

AP 均为百分数，AVG 是五个 tIoU 阈值的平均。

| 类别 | GT | 预测 | AP@0.1 | AP@0.3 | AP@0.5 | AVG |
|---|---:|---:|---:|---:|---:|---:|
| Fighting | 14 | 66 | 58.26 | 20.57 | 6.07 | 26.11 |
| Shooting | 20 | 33 | 39.21 | 21.18 | 7.07 | 20.58 |
| Riot | 10 | 30 | 37.60 | 6.54 | 0.00 | 13.94 |
| Abuse | 6 | 64 | 14.40 | 6.43 | 0.98 | 8.53 |
| CarAccident | 17 | 35 | 32.30 | 22.43 | 7.17 | 20.16 |
| Explosion | 5 | 29 | 39.43 | 19.73 | 0.00 | 18.79 |

## VideoMAE 先验诊断

下表统计各类别指定的五个样本中，该类别是否为视频汇总先验 Top-1。一个多标签视频的其他正确类别也可能排第一，所以该统计不能称为完整的多标签分类准确率。

| 指定类别 | Top-1 命中该类别 |
|---|---:|
| Fighting | 3/5 |
| Shooting | 2/5 |
| Riot | 5/5 |
| Abuse | 0/5 |
| CarAccident | 4/5 |
| Explosion | 4/5 |
| Normal | 3/5 |

- Riot 先验在指定的五个视频中均为 Top-1，说明新模型已提供该类别入口；但 Riot AP@0.5 为零，类别识别与边界定位必须分别评估。
- Abuse 先验在指定的五个视频中均未排第一，不能把 VideoMAE Top-1 当作搜索硬门槛。
- Normal 先验 Top-1 命中 3/5，最终 VLM 确认后的五个 Normal 视频均没有异常输出。
- 预测 257 段、GT 72 段，且 tIoU 从 0.1 升到 0.5 时 mAP 显著下降，存在重复、碎片化和边界不准问题。
- 本样本 Riot GT 段时长中位数约 34.40 秒、预测 Riot 段约 7.50 秒；Explosion 分别约 7.83 秒和 2.00 秒。两组并非一一匹配，不能把中位数差值当作定位误差，但它们提示需要检查短窗口对完整事件覆盖的限制。
- 当前只检查每类两个优先窗口和两个均匀覆盖窗口，长事件可能只被看到局部。后续应独立验证连续曲线候选、相邻窗口连续性合并及多标签类别校准，不应根据本测试 GT 调阈值后仍将其当作未见测试集。

## 可靠性、预处理与用量

- 相关回归用例累计 81 项通过（最终新增两项为端到端多标签/无文件名泄漏，以及跨类别 MCTS 总预算）；原有引用缺失 vad_v4 目录的用例单独排除，未修改该历史问题。
- 35 个视频全部读取原视频 FPS，均为 24 FPS，标注帧范围合法；源时间轴保存在 metadata/video_fps.json。变帧率尚未额外验证。
- XD 模型缺少专属 processor 配置，采用其基座 VideoMAE 官方 ImageNet 预处理；这是明确记录的实验假设，尚未核实微调作者的实际处理方式。
- 初轮 32/35 成功，三个视频因模型返回非法区间而失败；已保留在 ../v218_initial_attempt。修复零时长提示及格式纠正反馈、增加窗口内交集处理后，以同一 35 视频完整重跑。
- 正式轮 287 次 API 请求，成功 287，失败 0，已知 token 672,120。
- 初轮已知 token 668,739；初轮与正式轮合计 1,340,859，两轮用量分开保存在各自 metadata/token_usage.json。

## 文件与复现

- evaluation_report.json：完整指标、每视频类别和区间数量。
- analysis_summary.json：先验诊断和两轮 token 汇总。
- run.log / v218_console.log：正式执行日志。
- metadata/selection.json：抽样名单与各类配额。
- metadata/temporal_matching_audit.json：各类别、各阈值的逐预测 TP/FP 匹配。
- metadata/source_snapshot：与 manifest 哈希一致的 45 个推理源码快照。
- ../v218_initial_attempt：初轮记录，未混入正式指标。

设置 DASHSCOPE_API_KEY 后，从项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\vad_v3\run_xd_eval.ps1
```

该命令默认尝试同配置续跑 v218；代码或参数变化后，应通过 -OutputDir 指定新目录。
完整说明见 ../../XD_VIOLENCE.md。
