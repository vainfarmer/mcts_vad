# v212 接受阶段逐项诊断

最终比较共10个候选。注意：代码先检查变更区域引用，再检查语义条件；trace.reason只保留首次失败，不能把7个 missing_changed_region_citation 全部解释为“仅格式问题”。

| 视频/候选 | 首个失败 | 新边界获胜 | 完整无损 |
|---|---|---|
| Arson035_x264 / 0 | missing_changed_region_citation | False | True |
| Burglary079_x264 / 0 | comparison_rejected | True | False |
| Normal_Videos_782_x264 / 0 | missing_changed_region_citation | False | True |
| Robbery050_x264 / 1 | missing_changed_region_citation | False | False |
| Shooting032_x264 / 0 | missing_changed_region_citation | True | True |
| Shoplifting001_x264 / 1 | missing_changed_region_citation | False | False |
| Shoplifting001_x264 / 2 | comparison_rejected | True | False |
| Stealing036_x264 / 0 | missing_changed_region_citation | True | False |
| Vandalism015_x264 / 0 | missing_changed_region_citation | False | True |
| Vandalism015_x264 / 2 | comparison_rejected | True | False |

统计（问题可能重叠）：5个未选择新边界；6个 complete_without_loss=false；7个缺少删除区域引用。9/10至少一项语义条件不通过。只有 Shooting032_x264 候选0的所有最终语义条件通过，但 removed_evidence_indices=[]，因此仍不能接受。3个 comparison_rejected 都是选择了新边界但完整性为false。

代码/提示词层面的改进方向（本次仅注释，未实施行为修改）：

- 当前 A/B 提示列出原区间、新区间和图像角色，但没有显式给出逐段 changed_regions 与可用引用编号映射。可将 reference→proposal 的每段变化直接列出，减少推理负担。
- JSON 示例含空 removed/added 列表，虽然随后要求非空变化必须引用，模型仍多次输出空列表。可用清晰字段说明替代容易照抄的空值示例。
- winner、完整性与引用属于不同维度，需把所有失败项同时记录，避免first-failure日志误导诊断。
- 不应仅删除引用检查：即使去掉引用检查，9个候选仍有语义失败；只有1个可能新增接受，且这不证明其边界质量更好。
- 默认shrink动作只能收缩，无法通过外扩补全已经过窄的原候选；完整性判定不通过时，继续收缩未必能提供合适提案。

这些是根据现有回复和代码提出的解释方向，不是已通过对照实验确认的因果结论。
