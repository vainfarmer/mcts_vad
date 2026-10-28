# VAD 测试套件

## 运行单元测试

```bash
cd video_anomaly_detection
python3 -m pytest tests/ -v
```

单元测试覆盖**纯函数与轻量逻辑**，不依赖 CLIP/VideoMAE 模型或 DashScope API：

| 文件 | 覆盖范围 |
|------|----------|
| `test_post_processor.py` | `global_merge` / `unify_anomaly_types` / `narrow_short_events` 的合并、IoU、加权投票、收窄逻辑 |
| `test_config.py` | `VADConfig` 默认值对齐 v33（回归保护）、`from_args` 映射、`save/load` 往返 |
| `test_models.py` | `DetectionResult.to_dict()` 字段契约（与 batch_evaluate JSON schema 一致）、`AnomalySegment.from_dict` 往返 |
| `test_frame_scorer.py` | `FrameScorer` guard clauses（无 CLIP 特征/无模型/None 类别时正确返回 None）、缓存命中 |
| `test_token_compressor.py` | `SemanticFrameSampler.compute_target_frames` 帧数随时长/深度/shrink 缩放 |

## 设计原则

- **回归保护**：`test_config.py::TestDefaults` 锁定所有 v33 magic number
  （`clip_base_weight=0.35`、`confidence_gate=0.72`、`sigma_factor=1.5` 等），
  任何无意改动会被测试捕获。
- **字段契约**：`test_models.py` 确保 `DetectionResult.to_dict()` 产出的字段集
  与 `batch_evaluate.py` 写入 JSON 的 schema 一致，杜绝 v33 之前 `frame_scores=None`
  那类字段遗漏 bug 复现。
- **无模型依赖**：所有测试用 `unittest.mock.MagicMock` 桩化 CLIP/VideoMAE，
  因此可在无 GPU / 无权重的 CI 环境运行。

## 快速回归 harness（建议补充）

单元测试覆盖纯逻辑，但端到端 AUC 回归仍需全量 batch（~2h）。建议增加一个
**golden-file 回归**，把迭代验证从 ~2h 缩短到 ~2min：

```bash
# 思路（尚未实现）：
# 1. 选 5 个代表性视频（覆盖难类 Arson/Shoplifting + 易类 RoadAccidents + Normal）
# 2. 当前版本跑一遍，存 golden 结果到 tests/golden/
# 3. 改动后跑 regression_compare.py：对这 5 个视频重跑，对比
#    - anomaly 段数 / 时间 / 置信度（漂移阈值）
#    - frame_scores 曲线 L2 距离
# 4. 任何超出阈值的漂移 -> 失败，提示人工核查
```

关键点：固定 seed、固定 LLM temperature=0、固定帧采样，保证可复现。
LLM 的非确定性是回归 harness 的主要噪声源——对 LLM 依赖步骤用 mock 返回
固定 JSON，只回归纯算法部分（合并/评分/边界修正）。
