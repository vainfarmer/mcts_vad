# v219：Normal 作为第七个检测类别的离线重评

覆盖原批次 35 个视频，含 5 个真实 Normal 视频；未新增视频，未调用模型或 API。

Normal 预测来自原结果的 `is_normal=True`，并要求无异常检测，输出整段 `[0, 视频时长)`；置信度为 `1 - 源帧时间轴上的平均异常分数`。所有视频统一应用同一规则，构造预测的函数不接收视频名称、真实类别或 GT。
Normal GT 仅在真实 Normal 视频上标为完整视频；异常视频中的正常背景不额外标为 Normal GT。异常视频若被模型误判成 Normal，其预测会计为 Normal 类 FP。

| 组别 | 原六类 AVG % | Normal AP AVG % | 七类 AVG % | 七类 mAP@0.5 % |
|---|---:|---:|---:|---:|
| baseline_off | 18.018 | 100.000 | 29.730 | 17.328 |
| mcts | 19.133 | 100.000 | 30.685 | 18.563 |
| mcts_qscore | 13.824 | 100.000 | 26.135 | 17.545 |
| mcts_qscore_cbt | 12.861 | 100.000 | 25.310 | 17.349 |
| mcts_qscore_cbt_bue | 17.881 | 100.000 | 29.613 | 18.716 |
| mcts_cbt_bue | 21.624 | 100.000 | 32.821 | 20.484 |

## 指标解释

`七类 mAP = (六类 mAP × 6 + Normal AP) / 7`，每个 IoU 阈值分别计算，再对五个阈值取平均。七类指标包含了 Normal 的贡献；数值增加不意味着原六类异常的定位变好。
所有原六类的 AP、TP、FP 和预测数量已逐类校验完全不变。二分类帧级 AP/AUC 也不变。
本重评使用现有模型的视频级 Normal 判定，不是根据真实 Normal 视频名单补预测，也没有新增 Normal 视觉识别模型。

与 VadCLIP 默认七类均值相近的部分是类别集合、AP 公式和 IoU 阈值；本项目仍采用整视频 Normal 预测和源视频完整帧长，未复现其 Normal 类曲线阈值提案或特征长度×16的时间范围。35 视频子集的七类结果不能直接当作完整 XD 基准成绩。

## 文件

- `evaluation_report.json`：七类 mAP、Normal AP、原六类对照与逐类 TP/FP。
- `result/<视频名>.json`：五组及 off 对照的七类检测，字段 `arms.<组名>.detections`。
- `metadata/matching_<组名>.json`：包括 Normal 在内的逐预测 GT 匹配。
- `metadata/manifest.json`：原文件哈希、评分规则及源码哈希；原 v219 结果未覆盖。

复现命令：

```powershell
conda run -n common python -m evaluation_tools.reevaluate_xd_seven_class --experiment vad_v3/experiments/v219 --dataset_root F:/dataset/xd-violence
```
