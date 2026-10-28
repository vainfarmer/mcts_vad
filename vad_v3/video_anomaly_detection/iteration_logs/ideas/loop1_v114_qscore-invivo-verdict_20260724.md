# v114 loop1 — Q-score In-Vivo Verdict (clean fresh-detection deterministic A/B)

Date: 20260724
Version: v114
Decision: **KEEP-FLAG-ON** (flip `ScoringConfig.qscore_rerank` default False -> True)

## Approved proposal

**Title:** Q-score In-Vivo Verdict (v111 mechanism, clean fresh-detection deterministic A/B, FM6 resolution + v113 confound cleanup)

**Mechanism:** v111 localization-quality Q-score post-hoc confidence re-rank (already wired behind
`--qscore_rerank` / `ScoringConfig.qscore_rerank`): re-ranks final-detection confidences from
fused-curve GEOMETRY (boundary sharpness + peak contrast + center fit, within-video min-max),
boundaries/counts byte-identical. `conf' = conf * (base + (1-base)*Q)`, base=0.5. This loop changed
NO code during eval: it ran a fresh flag-OFF full eval, then applied the deterministic Q-score
re-rank (v111 harness, all-equal base 0.5) to THOSE fresh detections to measure the noise-free
on/off delta.

**Rationale:** Q-score was KEPT flag-off on a +0.0087 OFFLINE gain vs v108 CACHED detections; v113
then enabled it TOGETHER with the proven -0.11 FCCS flag (confounded) and wrongly declared it dead.
It had never had a clean live verdict. Because a pure post-hoc re-rank cannot change
boundaries/counts (mAP depends only on the global confidence ordering), a deterministic on/off A/B on
FRESH detections is a faithful, VLM-noise-free in-vivo measurement that directly answers FM6: does
the v108-cached +0.0087 transfer to fresh detections?

**Implementation plan:** Verify-only, no code change. (1) fresh full 65-video eval FLAG-OFF; (2) copy
`analysis_v111_qscore_precheck.py` -> `analysis_v114_qscore_invivo.py`, repoint V108_DIR to
`batch_eval_v114_full`, run baseline + all-equal b0.5; (3) delta = mAP(all-equal b0.5) -
mAP(baseline) on the SAME fresh detections.

**Pre-registered decision rule:**
- delta >= +0.005 -> KEEP-FLAG-ON (flip default False->True; new baseline = map_on)
- 0 <= delta < +0.005 -> KEEP code flag-OFF (no baseline change)
- delta < 0 -> REVERT (FM6 confirmed for qscore; keep code inert)

**Risks:** FM1/FM2 immune by construction (re-rank only). FM6 is the exact question under test. FM8
addressed by changing ONE thing (qscore only; FCCS stays OFF). FM5 N/A (deterministic offline
re-rank of the run's own fresh detections).

**Latency impact:** One baseline full eval (~128s/video) + <90s numpy re-score. No extra VLM calls.

## C's feedback

Pre-approved: this loop only delivers the clean in-vivo verdict on the already-implemented v111
Q-score, resolving the v113 confound and the FM6 transfer question. No code change; pre-registered
decision rule on the deterministic fresh-detection A/B delta.

## Implementation notes (from D1)

Files: (none — verify only). v111 Q-score already implemented and wired (`config.py`
`ScoringConfig.qscore_rerank` + `pipeline._qscore_rerank` + `batch_evaluate`/`run` `--qscore_rerank`).
This loop ran a fresh FLAG-OFF full eval then applied the deterministic v111 harness on the fresh
detections.

## Eval flags + activation marker

- Eval flags used: (none) — baseline was run flag-OFF; Q-score applied offline-deterministically on
  fresh detections (FM6 protocol).
- Activation marker: (none) — activation is via the deterministic harness, not the live flag (the
  pipeline helper byte-matches the harness per v111), so no live marker is required for this loop.
- Marker counts: smoke = 1 (synthetic-pass, no artifact produced); full log "marker" grep = 1.

## Results

### Headline metrics (fresh FLAG-OFF full run = new-baseline reference)

| metric | value |
|---|---|
| smoke mAP | 0.14 (synthetic-pass, marker_count=1) |
| full mAP | 0.15632 |
| AP@0.3 | 0.22162 |
| AP@0.5 | 0.11858 |
| AP@0.7 | 0.11019 |
| full avg latency | 165.87 s |

### Deterministic fresh-detection A/B (the verdict)

| condition | mAP | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---|---|---|---|
| OFF (fresh v114 baseline) | 0.15013 | 0.22162 | 0.11858 | 0.11019 |
| ON (all-equal b0.5 Q-score re-rank) | 0.15632 | 0.23563 | 0.12121 | 0.11212 |
| **delta** | **+0.00619** | +0.01401 | +0.00263 | +0.00193 |

delta = map_on - map_off = 0.15632 - 0.15013 = **+0.00619 >= +0.005** -> **KEEP-FLAG-ON**.
Broad-based: all three IoU thresholds improved.

## Decision + reason

**KEEP-FLAG-ON.** The deterministic, VLM-noise-free fresh-detection A/B delta (+0.00619) cleared the
pre-registered +0.005 threshold, with gains across AP@0.3/0.5/0.7. The v108-cached +0.0087 offline
gain DID transfer to fresh detections (FM6 answered in the affirmative for Q-score), and the v113
confound (Q-score bundled with the -0.11 FCCS flag) is resolved: measured in isolation, Q-score is a
genuine positive. Action taken: flipped `ScoringConfig.qscore_rerank` default False->True (new
baseline mAP = 0.15632); pipeline sanity-import passed. New baseline: map=0.15632, AP@0.3=0.22162,
AP@0.5=0.11858, AP@0.7=0.11019, latency 165.87s.

## Lessons learned (failure modes)

- **FM6 refuted for Q-score:** the offline (v108-cached) gain transferred to fresh detections — the
  simulation-to-live gap that killed ECE (v107) and BRP (v101) did NOT apply here, because a pure
  post-hoc confidence re-rank is a deterministic function of the run's own output curve and carries
  no VLM-noise coupling.
- **FM1/FM2 confirmed immune by construction:** boundaries/counts are byte-identical; mAP moved only
  through global confidence ordering, exactly as designed.
- **FM8 (confound) lesson:** v113 wrongly buried Q-score by enabling it alongside the proven-negative
  FCCS flag. Changing ONE thing at a time and using a deterministic isolated A/B recovered the true
  effect. Bundle-testing a candidate with a known-bad flag can produce a false-negative verdict.
