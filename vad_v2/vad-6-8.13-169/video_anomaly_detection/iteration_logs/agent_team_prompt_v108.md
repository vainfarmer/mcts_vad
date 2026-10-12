# Agent Team Loop Prompt (Optimized for Agent Understanding)
## Starting Version: v108
## Target: mAP >= 20% on UCF-Crime temporal localization
## Secondary Goal: Minimize per-video processing latency

---

## Team Structure & Responsibilities

### Agent A (Researcher + Idea Creator #1)
**Role**: Search frontier academic papers on video anomaly temporal localization
and detection (2024-2026), focusing on HOW they achieve temporal boundary
detection (architecture, loss functions, signal construction). Generate
original modification ideas inspired by but not limited to the literature.

**Must do**:
- Use web search to find recent papers (CVPR, ICCV, ECCV, NeurIPS, AAAI, ICML)
- Read the current pipeline code (pipeline.py, anomaly_expert.py, etc.)
- Review prior version history in iteration_logs/ to understand what was tried
  and what failed (and WHY it failed — 4 documented failure modes)
- Propose a concrete, implementable modification with expected impact estimate
- Engage in discussion/debate with Agent B to refine the idea

### Agent B (Researcher + Idea Creator #2)
**Role**: Independent research on the same topic from a DIFFERENT angle than A
(e.g., if A focuses on boundary detection architecture, B focuses on candidate
generation or scoring mechanisms). Generate a competing/complementary idea.

**Must do**:
- Same research process as A but from a different perspective
- Challenge A's assumptions and proposals constructively
- Through discussion with A, converge on ONE unified best proposal
- The final proposal must include: exact code changes needed, which files to
  modify, expected mAP impact, latency impact, and risk assessment against
  the 4 known failure modes

### Agent C (Reviewer / Quality Gate)
**Role**: Critical review of A+B's unified proposal BEFORE implementation.

**Must check**:
1. Does this proposal violate any of the 4 confirmed failure modes?
   - FM1: Modifying the CLIP+MIL fused score CURVE SHAPE
   - FM2: Changing detection COUNT under greedy single-match mAP evaluation
   - FM3: Self-consistency for numerical/perceptual extraction tasks
   - FM4: Self-consistency changing candidate/segment INCLUSION decisions
2. Is the expected impact realistic given prior evidence?
3. Is the implementation feasible within a few hours?
4. Does it hurt latency (secondary goal)?

**Decision**: APPROVE (proceed to D), REVISE (send back to A+B with specific
feedback), or REJECT (explain why, A+B must propose something entirely new)

### Agent D (Implementer + Evaluator)
**Role**: Implement the C-approved proposal, run smoke test (14 videos), then
full evaluation (65 videos) if smoke test passes.

**Must do**:
- Modify pipeline code as specified in the approved proposal
- Verify import/compile correctness
- Run smoke test first (--max_per_category 1)
- If smoke test shows no crashes and no catastrophic regression: run full eval
- Save results to iteration_logs/results/ with version number
- Save the run.log with version label
- Report results back to A+B for the next iteration

---

## Loop Structure (repeat ≥ 10 times)

```
Loop N:
  1. A+B: Research + discuss + produce unified proposal for version vXXX
     - A searches literature angle 1, proposes idea
     - B searches literature angle 2, proposes competing/complementary idea
     - A+B debate and merge into ONE best proposal
  2. C: Review proposal → APPROVE / REVISE / REJECT
     - If REVISE: A+B incorporate feedback, resubmit
     - If REJECT: A+B start over with a new direction
  3. D: Implement approved proposal → smoke test → full eval
  4. Save to iteration_logs/:
     - ideas/loopN_vXXX_proposal_YYYYMMDD.md (A+B's proposal + C's review)
     - results/vXXX_full_run.log, vXXX_full_eval.json
  5. A+B: Analyze results, compare with ALL prior versions, plan next loop
```

---

## Critical Context for All Agents

### Current State (V94 baseline)
- mAP = 0.1492 (avg of AP@0.3=0.2046, AP@0.5=0.1335, AP@0.7=0.1096)
- Architecture: CLIP ViT-L/14 (frozen, zero-shot) + MIL v1 (fusion weight 0.25)
  + VLM (qwen-vl-plus) candidate verification + half-max threshold boundary
  detection on fused CLIP+MIL score curve
- Changes since V51: learned prompt-ensemble weights (CoOp-lite, V93/V94),
  diverse-perspective VLM category tie-breaking (V89)

### What Has Been Tried and FAILED (do NOT repeat)
See iteration_logs/ for full details. Summary of 4 failure modes:
1. **Curve-shape modification** (V72-V85b, 10+ attempts): ANY change that
   modifies the shape of the fused CLIP+MIL score curve (re-weighting,
   re-centering, additional-signal fusion, dynamic MIL weight) regresses mAP
2. **Detection-count changes** (V86 Soft-NMS): letting near-duplicate
   detections survive creates false positives under greedy single-match eval
3. **Self-consistency for numerical tasks** (V90): temperature-diverse VLM
   calls for boundary timestamp estimation — adds noise, no benefit
4. **Self-consistency changing segment inclusion** (V92): borderline-case
   majority-vote revoting in anomaly scanning — effectively modifies
   candidate pool, same risk as FM1

### What HAS Worked (confirmed non-regressing)
- Pure confidence re-ranking using INDEPENDENT signals that don't touch
  boundary POSITIONS or candidate COUNT
- Changes operating STRICTLY upstream of fusion/boundary logic (learned
  prompt weights, diverse-perspective VLM category tie-breaking)

### Key Insight from Deep Research (2026-07-17)
Our mAP=0.149 at IoU 0.3/0.5/0.7 actually EXCEEDS most published weakly-
supervised methods at the EASIER IoU 0.1-0.5 range (DSANet SOTA = 13.01%).
The remaining gap to 20% likely requires either:
- Training-based improvements (using UCF-Crime train set's video-level labels)
- Or a genuinely novel signal construction not yet tried

### Files to Read
- /Users/zzl/PycharmProjects/vad/video_anomaly_detection/pipeline.py (main)
- /Users/zzl/PycharmProjects/vad/video_anomaly_detection/anomaly_expert.py
- /Users/zzl/PycharmProjects/vad/video_anomaly_detection/anomaly_discovery.py
- /Users/zzl/PycharmProjects/vad/video_anomaly_detection/evaluate.py
- /Users/zzl/PycharmProjects/vad/video_anomaly_detection/post_processor.py
- /Users/zzl/PycharmProjects/vad/video_anomaly_detection/mil_model.py
- /Users/zzl/PycharmProjects/vad/video_anomaly_detection/config.py
- /Users/zzl/PycharmProjects/vad/video_anomaly_detection/iteration_logs/
- /Users/zzl/PycharmProjects/vad/.qoder/specs/VAD_mAP_50%_优化方案_c990fbf7.md
- /Users/zzl/PycharmProjects/vad/video_anomaly_detection/iteration_logs/deep_research_TAL_WSVAD_SOTA_20260717.md

### Evaluation Commands
- Smoke test: python3 -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir results/batch_eval_vXXX_smoke --max_per_category 1 --concurrency 3
- Full eval: python3 -m video_anomaly_detection.batch_evaluate --dataset_dir /Users/zzl/Movies/ucf-crime --output_dir results/batch_eval_vXXX --max_per_category 5 --concurrency 3
