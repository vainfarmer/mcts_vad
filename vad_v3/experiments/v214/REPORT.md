# v214 宽松MCTS配对测试

相同14视频，ViT-L-14/openai（V2激活），qwen3-vl-plus，CUDA，并发2；all_valid + relative_nonloss + shrink_expand_shift，32图/请求，每视频4候选/40逻辑请求，时间覆盖1.5秒/3秒。

## 结果

14/14结果齐全。首次Burglary视频API超时，续跑仅重试该视频，其他13个结果跳过；结果指纹一致。

| 指标 | off | on | skip_only |
|---|---:|---:|---:|
| mAP | 0.181145 | 0.181145 | 0.181145 |
| AP@0.3 | 0.246465 | 0.246465 | 0.246465 |
| AP@0.5 | 0.206061 | 0.206061 | 0.206061 |
| AP@0.7 | 0.090909 | 0.090909 | 0.090909 |
| 帧AUC | 0.792448 | 0.791552 | 0.792023 |
| 帧AUPR | 0.487493 | 0.486459 | 0.487016 |

23候选，18个进入搜索，7个采用修改（7/18=38.9%；7/23=30.4%），6个视频异常段对象发生变化，其中4个仅depth/visits元数据变化；实际最终时间区间变化为2个视频。Normal候选虽接受修改，但最后被过滤。接受率提高不等于精度提高：整体mAP未提升，帧AUC略降。与v213跨次mAP差异不能归因于MCTS，因为本次off/on mAP相同且上游/API输出存在变化。

| 接受候选 | 原边界（秒） | 新提案（秒） |
|---|---|---|
| Arson035_x264 / 0 | [23.0, 33.5] | [22.0, 33.5] |
| Explosion002_x264 / 0 | [51.5, 61.5] | [50.5, 61.5] |
| Normal_Videos_782_x264 / 0 | [105.75, 110.75] | [105.5, 110.75] |
| Shooting032_x264 / 0 | [295.5, 300.5] | [295.5, 301.0] |
| Shoplifting001_x264 / 2 | [48.0, 56.5] | [49.0, 57.5] |
| Stealing036_x264 / 0 | [34.0, 40.0] | [33.5, 40.0] |
| Vandalism015_x264 / 1 | [30.0, 51.5] | [28.0, 49.5] |

最终边界仍可能被后处理再次修改，完整对照保存在summary.json的acceptance_audit。

## 用量与时间

两次批次执行墙钟合计476.4秒（428.8+47.6，不含模型初始化及两次运行间隔）。共306次SDK请求，288成功、18失败；已知输入430,693、输出25,627、总456,320 token。18次失败请求无usage，不能按零实际消耗处理。此总数包含失败尝试及全部配对分支，不能当作单独on分支部署成本。逐请求总数已与两个历史run_id汇总核对。

metadata/token_usage.json及evaluation_report.json的timing属于最后一次续跑，完整实验统计请看summary.json，不能把latest与历史文件重复相加。

1000次配对bootstrap的mAP差值区间为[0,0.032969]；原始总体差值为0，区间包含0，不支持总体改善结论。重采样改变视频权重，重采样差值可非零，即使原始批次mAP相同。

## 复现命令

在项目根目录，已配置环境变量DASHSCOPE_API_KEY时：

```powershell
.\vad_v3\run_relaxed_search.ps1 -DatasetDir "F:\dataset\ucf_crime" -OutputDir "D:\Users\zzl\project\mcts_vad\vad_v3\experiments\v214_repeat\results" -Concurrency 2
```

新输出目录用于独立重测；指定已存在v214/results会校验manifest并跳过已有结果。实际完整参数保存在COMMAND.json；首次日志v214_console.log，续跑日志v214_resume1_console.log。原首次exit_status.json=1保留，resume1_exit_status.json=0表示恢复成功。


## 后续核验更正

之前将完整anomaly字典不相等计为“区间变化”，包含了depth/visits元数据，因此“6个视频最终区间变化”的表述不准确。真正时间边界改变的只有Shoplifting001与Stealing036，其他接受修改的候选被后处理拉回原边界或删除。逐视频字段比较见final_comparison.json；off/on/skip_only完整最终结果分别导出在comparison_arms目录。

18次失败全部为Burglary079的APITimeoutError，集中在2026-10-27 17:00:43发起、约6.6秒后失败。对应主异常堆栈是httpcore.ConnectTimeout → httpx.ConnectTimeout → openai.APITimeoutError，发生在http_proxy连接路径。可确认连接超时，不能仅凭此确定是代理故障、网络波动或并发压力。视频并发2不等于API并发2，视频内部用asyncio.gather同时扫描多个片段；此次18个请求是同视频片段请求，非18次MCTS拒绝。缺失视频随后独立补跑成功。

Shoplifting的一个最终区间由48–56.5秒变为49–57.5秒，与GT的IoU由0.258929升至0.330189，AP@0.3的真阳性总数从4增到5。但现有evaluate.py使用11点插值AP：召回率从4/13=0.3077升至5/13=0.3846，没有达到下一0.4采样点；新增TP置信度排名第17，也未改善已有采样点的精确率包络，因此AP及mAP数值不变。这不是所有定位都没有变化。

配对模式共享Stage5以前的上游数据，在深拷贝上独立执行完整off/skip_only/on尾流程；相同请求缓存用于减少随机差异。若要对比独立端到端成本，需要另外使用两个新输出目录分别运行mcts_mode=off和mcts_mode=refine，并关闭paired_eval；但上游API随机性会成为混杂因素，应多次重复，不能拿单次跨运行差异直接归因于MCTS。
