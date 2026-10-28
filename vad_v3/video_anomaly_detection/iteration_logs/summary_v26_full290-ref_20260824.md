# v26 — full-290 committed-baseline 复现（REF290 重测）

**日期** 2026-08-24 · **运行** `results/batch_eval_v26_full290/` · **日志** `results/batch_eval_v26_full290.log`
**结论**（2026-08-24 21:16 已补齐续跑后更新）✅ **RESOLVED — 基线在噪声内复现，未检出漂移。完整覆盖 mAP = 0.16682（290/290，0 失败）。**
下文第 1–4 节保留补齐前的分析（含被抬高的 0.17673 与下界 0.16052）作为 §8.4 陷阱的实证记录；最终结论见第 7 节。

**补齐前的初始结论** ⚠️ INCONCLUSIVE — 不可作为基线数字引用。 24/290 视频因 `Connection error` 失败，且失败**高度集中在两个最难类别**（19 Shoplifting + 5 Shooting）。`evaluate()` 把无预测的视频从 GT 分母中剔除，因此 `evaluation_report.json` 里的 **0.17673 是被抬高的**，不可与任何完整覆盖的 290 基线比较。诚实口径见下。

---

## 1. 这一轮跑的是什么

不带任何实验 flag 的**已提交基线**（pipeline v182）：

```
python3.14 -u -m video_anomaly_detection.batch_evaluate \
  --dataset_dir /Users/zzl/Movies/ucf-crime \
  --output_dir  .../results/batch_eval_v26_full290 \
  --max_per_category 999 --concurrency 2
```

生效配置（全部来自 default-ON）：`qscore_rerank` · `bue_enabled` · `bue_closure_gate` · `cbt_enabled` · `cbt_recal` · `classify_skip_class_agnostic`。

**这构成一次对 v160_full 的合法复现**——v160_full 当时是显式传 `--bue_extension --qscore_rerank --cbt_tighten --cbt_recal`，而这些开关此后已晋升为默认值，两者有效机制配置等价。已用标记计数核实：

| 标记 | 计数 | 判读 |
|---|---:|---|
| `[CBT-VID]` | 266 | = succeeded，每视频恰好一次 → FM17 满足 |
| `[BUE]` | 407 | BUE 实际触发 |
| `[CGR-VID]` / `[CCR-VID]` / `[CNV-VID]` / `[SNH…]` / `[FSD-VID]` / `[ECE]` / `[EUBC…]` | **0** | 实验机制确认全部未参与 |

---

## 2. 运行统计

| 项 | 值 |
|---|---|
| 起止 | 02:11:50 → 10:42:43 |
| 墙钟 | 30562.4s = **8.49h** |
| 并发 | 2 |
| 选中 / 成功 / 失败 / 跳过 | 290 / **266** / **24** / 0 |
| 单视频均时 | 229.36s |
| 最慢视频 | `Normal_Videos_210_x264` 2115.2s（180s 视频）；前 5 慢**全部是 Normal** |

**失败构成**（24 条全部为 `Connection error`，DashScope 侧网络/接口）：

| 类别 | 失败数 | 该类总数 |
|---|---:|---:|
| Shoplifting | **19** | 21 |
| Shooting | **5** | 23 |

Shoplifting 只剩 2/21 进入评测。

---

## 3. 指标 —— 两个口径必须同时给出

`compute_temporal_map` 遍历**全部 GT**（`evaluate.py` 第 18 行 `for video_id, gt_entry in gt.items()`），缺失预测经 `pred.get(video_id)` 返回 `None` 后其 GT 段计为漏检。但 `evaluate()` 传入的 GT 已被限制到有预测的视频，于是失败视频**连带其 GT 一起消失**。

实测复现（`load_ground_truth` 全量 GT vs 报告值）：

| 口径 | mAP | 说明 |
|---|---:|---|
| **报告值 / 仅按已产出视频计分** | **0.17673** | = `evaluation_report.json`。分母已剔除 24 个最难视频 → **偏高，不可引用** |
| **按完整 290 GT 计分（失败=漏检）** | **0.16052** | 惩罚了从未尝试的视频 → **真实值的下界** |

真实的完整覆盖值落在 **[0.16052, 0.17673]** 区间内，本轮无法收窄。

其余指标（同样基于 266 视频，带同一偏置）：

| 指标 | 值 |
|---|---|
| AP@0.3 / @0.5 / @0.7 | 0.26313 / 0.15852 / 0.10855 |
| frame AUC / AUPR | 0.82309 / 0.37386 |
| 视频级 acc / prec / recall / f1 / auc | 0.71053 / 0.60209 / **0.99138** / 0.74919 / 0.95684 |
| 类别准确率（overall，取自日志） | 0.8190 |

视频级 recall 0.991 但 precision 0.602 —— Normal 误报仍是视频级主要失分项（与 SNH/CNV 两次关闭结论一致，不构成新信息）。

---

## 4. 与历史 290 基线的可比性核对

| 运行 | succ / fail | 缺失异常视频 | 报告 mAP | 全 GT 口径 | 可比？ |
|---|---|---:|---:|---:|---|
| `batch_eval_full290_bue`（BUE only） | 103 / 0 | 0 | 0.15114 | 0.15114 | ✅ |
| `batch_eval_v160_full`（BUE+CBT+recal） | 290 / 0 | 0 | **0.17254** | 0.17254 | ✅ 权威基线 |
| `batch_eval_v26_full290`（本轮，同配置） | 266 / 24 | **24** | 0.17673 | 0.16052 | ❌ |

- 若按报告值直读：0.17673 vs 0.17254 = **+0.0042「提升」——这是丢掉 19 个 Shoplifting 换来的假象**。
- 若按全 GT 下界：0.16052 vs 0.17254 = −0.0120。
- 而 v16 实测的同配置跨-run 离散度在 hard-88 上达 **0.025**（`v167_full` 0.12246 vs `v169_ab88` 0.09743，flag 串逐字相同）；290 尺度噪声较小但与这 0.012 的差同阶。

**因此本轮既不能确认、也不能否认基线漂移。**

---

## 5. 结论与动作

**结论**：本轮是一次**未完成**的基线复现。唯一可靠的产出是运行层面的：8.49h / 229s 每视频 / 并发 2 下的稳定性画像，以及 8.3% 的接口失败率。机制层面零新信息。

**必须做**（成本低）：续跑 24 个失败视频。`process_one_video` 对已存在的 JSON 直接 SKIP，因此重跑同一 `--output_dir` 只会补齐缺口：

```bash
cd /Users/zzl/PycharmProjects/vad
source video_anomaly_detection/.env
nohup python3.14 -u -m video_anomaly_detection.batch_evaluate \
  --dataset_dir /Users/zzl/Movies/ucf-crime \
  --output_dir  $PWD/results/batch_eval_v26_full290 \
  --max_per_category 999 --concurrency 2 \
  > results/batch_eval_v26_full290.resume.log 2>&1 &
```

预估 24 × 229s ÷ 2 ≈ **45 分钟**，之后 `evaluation_report.json` 会以 290/290 重写，届时方可与 0.17254 直接比较。

**协议缺口（本轮暴露，已写入 `PROJECT_DOCUMENTATION.md` §8.4）**：`evaluation_report.json` 不自带完整性字段，`failed>0` 时 mAP 被静默抬高。任何引用 290 数字前必须先查 `timing.failed == 0`。建议在 `batch_evaluate.py` 增加与 `[ECE-GATE]` 同形的硬门：`failed > 0` 时拒绝写 `evaluation_report.json`（或写入 `coverage_complete: false`）。**注意**：该改动会触及 `batch_evaluate.py`，属评测器变更，需按治理规则单独预注册，本轮未实施。

---

## 6. 台账行

见 `session_v97_ledger.md`：`| 0 | v26-REF290 | ... | kept=false |`（非机制 loop，记录为基线重测事件）。

---

## 7. 续跑后的最终结论（2026-08-24 21:16）

补齐 24 个缺口：`Succeeded: 24, Skipped: 266, Failed: 0`，17.9min 墙钟，89.5s/视频，2.00x 并行加速。**24 个此前 `Connection error` 的视频全部一次通过**，证实失败是瞬时接口问题而非视频本身不可处理。

**完整性门（§8.4）通过**：`failed == 0`，缺失异常视频 0，报告值与全 GT 重算值**逐位一致**（0.16682 == 0.16682）→ 无剔除偏置。

### 权威对比（双方均 290/290，机制配置等价）

| 指标 | `v160_full` | v26（补齐后） | delta |
|---|---:|---:|---:|
| **mAP** | **0.17254** | **0.16682** | **−0.00572** |
| AP@0.3 | 0.26497 | 0.25381 | −0.01116 |
| AP@0.5 | 0.14993 | 0.14429 | −0.00564 |
| AP@0.7 | 0.10273 | 0.10236 | −0.00037 |

其余层级：frame AUC 0.79058 / AUPR 0.33585；视频级 acc 0.73103 · prec 0.64486 · recall 0.98571 · f1 0.77966 · auc 0.94786。

### 判读

- **基线复现成立**。−0.00572 远小于 290 尺度预注册 VLM 方差告警（±0.019），也小于同配置 hard-88 实测离散度 0.025（§8.3.1）→ **未检出基线漂移**，pipeline v182 默认配置与 v160 时代等价。
- **三个数字的关系印证了 §8.4 的推理**：真实值 0.16682 确实落在补齐前给出的区间 [0.16052, 0.17673] 内。被抬高的报告值比真值高 **+0.00991**，而按"失败=漏检"的下界比真值低 0.00630。**若当初直读 0.17673，会得到 +0.0042 的假性提升，方向与真实的 −0.0057 相反。**
- **可引用基线不变**：`v160_full` 0.17254 仍为权威 290 基线；v26 作为其独立复现（0.16682）并存。
