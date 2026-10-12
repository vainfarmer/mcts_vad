# Agent Team Loops v124+ Summary — LATENCY-REDUCTION SESSION
## Date: 2026-07-26
## Pipeline: video_anomaly_detection (UCF-Crime) — temporal localization, weak supervision
## Objective: reduce per-video execution time. HARD ACCURACY GUARD: full-eval temporal mAP >= 0.15132 (v114 committed baseline 0.15632 minus 0.005).
## Win metric (FM10): drift-free per-video VLM+CLIP CALL COUNT; wall-clock is API-drift-dominated corroboration only.

## Executive Summary
This session ran 7 loops (v124-v130), pivoting from the tapped accuracy-re-rank arc
to LATENCY. **Loop 1 (v124) is the only KEEP**: it completed the half-applied v114
Q-score CLI-default flip and added always-on `[COST]` per-video instrumentation,
establishing the **same-session reference** that all subsequent drift-free
reductions are measured against:

- **REF_vlm_calls = 31.969 / video**, REF_clip_calls = 22.308, **REF_wallclock = 183.11 s/video** (65 videos, concurrency 3).
- v124 full mAP = 0.17076 (>= floor 0.15132), reproducing the committed v114 Q-score baseline within VLM noise.

**Net per-video speedup COMMITTED this session: 0%.** Every latency mechanism
v125-v130 REVERTED. The final kept config is byte-identical to v124: 31.969
vlm_calls, 183.11 s/video. **The committed accuracy floor held throughout** — the
production baseline never dropped below 0.15132 (two experimental arms, v126/v129,
breached the floor at 0.125/0.120 but were reverted; nothing below-floor was ever
committed).

The session produced a **clean, structural negative map** of the latency space.
Two real call reductions were achieved (v126 -65%, v129 -69.4%) but **both broke
the accuracy floor**; every mechanism that was mAP-neutral by construction was
either a structural no-op or capped below the 15% gate. The two axes are in tension:
mechanisms that actually remove VLM calls do so by altering VLM payload
composition (which is not mAP-neutral), while mechanisms that preserve payloads
save nothing on the mean because `max_scan` already binds or the gate never fires.

## What Was Tried (v124-v130)

| loop | ver | mechanism | call reduction (drift-free) | wall-clock | full mAP | kept | verdict |
|---|---|---|---|---|---|---|---|
| 1 | v124 | Baseline calibration: Q-score CLI-default flip + always-on `[COST]` instrumentation | REFERENCE (0.0% by construction) | 183.11s (REF) | 0.17076 | **YES** | **KEEP — establishes REF; floor held; FM5-clean (14/14+65/65 markers)** |
| 2 | v125 | PREGATE adaptive CLIP-confidence cascade gate on Source-2 VLM padding | **+0.24% (0 skips)** | 275.5s | 0.16802 | no | REVERT — structural near-no-op; thresh below mean AND top-rank pad = empty intersection |
| 3 | v126 | TRLM+ temporal run-length merge of contiguous cosine>=0.97 near-dup segments | **-65.0%** (11.2 vs 31.97) | 134.7s | **0.12547** | no | REVERT — **breached floor**; FM2 verdict-propagation not mAP-neutral |
| 4 | v127 | CNG-DUAL CLIP+MIL dual-agreement whole-video Normal early-exit gate | **-0.3% (0 fires)** | 251.83s | 0.16152 | no | REVERT — 0-fire no-op; dual-AND too conservative for addressable pop. |
| 5 | v128 | VMW-SKIP prune saturated VideoMAE sliding-window Stage-2b (local-compute) | -0.6% call / **-12.7% wall-clock** | 164.9s | 0.17590 | no | REVERT — mAP-neutral but magnitude < 15%; local-compute lever capped |
| 6 | v129 | BDS batched directed scan (N calls -> ceil(N/B) contact-sheets, B=4) | **-69.4%** (9.77 vs 31.97) | 189.09s | **0.11980** | no | REVERT — **breached floor**; FM2/FM3 cross-segment context bleed flips verdicts |
| 7 | v130 | CASCADE early-exit: skip Source-2 padding when Source-1 fills top_k | **-3.1%** (high fire-rate) | 205.39s | 0.17708 | no | REVERT — efficacy anti-correlation; `max_scan` already binds -> savings capped |

## Which Latency Levers Worked vs Failed, and Why

**FAMILY A — VLM-call REMOVAL that actually cuts calls (v126 TRLM, v129 BDS): WORKED on latency, FAILED on accuracy.**
Both crushed the 15% call gate (-65%, -69.4%) with markers firing at scale. Both
breached the accuracy floor because **any change to VLM payload composition on the
common directed-scan path is not mAP-neutral in practice**: TRLM propagates one
verdict across a contiguous run (FM2); BDS packs multiple segments into one
contact-sheet, and cross-segment visual context bleed flips per-segment verdicts
(FM2/FM3). Together with v97 FCCS, the **VLM-call-batching/merge family is now
effectively REFUTED for accuracy** — per-call `scan_segment` INDEPENDENCE is
load-bearing for mAP, not incidental.

**FAMILY B — mAP-neutral GATING that preserves payloads (v125 PREGATE, v127 CNG, v130 CASCADE): SAFE on accuracy, FAILED to save.**
All held the floor by construction (independent single-segment payloads, no verdict
propagation), but none cleared 15%:
- v125 PREGATE: adaptive threshold (mean-0.5*std) sat below the mean while padding
  selected only top-20%-by-rank -> the "top-rank AND below-threshold" intersection was
  near-empty -> **0 skips**, AND `max_scan` already bound the candidate count.
- v127 CNG: conservative dual-AND (clip_peak<0.5 AND mil_max<0.3) **never jointly
  fired** on any video reaching the directed scan (truly-normal empty-region videos
  already bypass the scan pre-gate). Addressable population = only hard cases with
  >=1 elevated cheap signal, where the AND is false.
- v130 CASCADE: high fire-rate (48/58) but tiny savings because **`max_scan` already
  binds** — many firing videos logged `src2_skipped=0`. Fire-rate is the wrong
  headline; skipped-magnitude is the real lever.

**FAMILY C — LOCAL-compute pruning (v128 VMW-SKIP): SAFE, mAP-byte-identical, but structurally capped.**
Removing the saturated VideoMAE stage issued ZERO VLM/CLIP calls, so the call-count
headline was flat by design; wall-clock fell only -12.7%, below the 15% gate. A
local-compute lever is **structurally capped below the session's primary drift-free
metric** and clears 15% only on long compute-heavy CCTV videos, not on the mean.
The VideoMAE saturated-stage lever is now CLOSED with documentary value.

**Core structural finding:** on this pipeline, `max_scan` / `candidate_top_k`
already caps the common-path VLM cost, so Source-2 padding-pruning (v125/v130) has
almost nothing to remove, and the whole-video early-exit gates (v127) find no
addressable population. The only way to cut calls materially is to change payload
composition (v126/v129) — which the floor forbids. **The two goals are in direct
structural tension on the current architecture.**

## Updated Failure-Mode Taxonomy (FM1-FM11)
- **FM1 curve-shape modification regresses** — immune-by-construction in every
  v124+ loop (all latency levers were count/curve-frozen on the kept/neutral path).
- **FM2 detection-count/verdict changes under greedy match** — **CONFIRMED as the
  dominant accuracy killer this session** (v126, v129). Propagating or co-mingling
  VLM verdicts across segments turns near-duplicates into displaced FPs; NOT
  mAP-neutral even at tau=0.97 + contiguity (v126) or absolute per-segment prompts
  + B=4 (v129). Refuted-as-materialized where payloads stayed independent (v130 mAP rose).
- **FM3 self-consistency / packed-payload perception** — CONFIRMED for contact-sheet
  packing (v129): cross-segment context bleed inside one VLM call flips verdicts.
- **FM4 self-consistency on segment inclusion** — not exercised.
- **FM5 mechanism implemented but never activated** — **the always-on, default-ON,
  no-flag `[COST]` pattern (v124) is the cleanest possible FM5 activation** (fired
  100% of videos, no launch/harness gate to skip). Refinement this session: an
  activation MARKER is necessary but NOT sufficient — the deterministic **fire-COUNT**
  (skips/merges/src2_skipped) is what distinguishes a real firing from a 0-fire
  no-op (v125 0 skips, v127 0 fires) that a plain marker-grep would mask.
- **FM6 simulation-to-live gap** — N/A (latency validated by one full eval, not simulation).
- **FM7 video-level priors can't fix intra-video mislocation** — orthogonal to latency;
  not re-litigated.
- **FM8 offline precheck config-specific** — N/A (no offline harness used this session, by design — see FM11).
- **FM9 MPS device-placement** — N/A this session (only int counters / stage bypass /
  CPU-numpy keyframes; no on-device tensor->numpy ops), so no crash.
- **FM10 wall-clock is API-drift-dominated** — **CONFIRMED essential repeatedly.** The
  same config swung 134s->275s across loops from qwen-VL load alone. The drift-free
  call count correctly killed a spurious v125 -48.1% wall-clock "win" (calls actually
  +0.24%), read v127 as a true no-op (not a +37.5% "regression"), and reported v129's
  true -69.4% independent of its +3.3% wall noise. **Always headline the call count.**
- **FM11 offline harness must complete as a pollable upstream step** — **avoided
  entirely by design**: every latency lever was validated by ONE full eval (mAP +
  `[COST]`), so no bake-off harness existed to stall on. This is why v124-v130 had
  ZERO harness-skip deaths (contrast v117/v120/v121/v123 last session). **Lesson:
  latency mechanisms that need no offline harness are structurally FM11-immune —
  the preferred archetype.**

## Concrete Next Latency Directions
1. **Attack the `max_scan` ceiling itself, not the padding.** v125/v130 both died
   because `max_scan`/`candidate_top_k` already caps common-path VLM cost. The real
   lever is an **adaptive `max_scan`** that shrinks the scan budget on
   low-prior videos (few high-CLIP regions) — mAP-neutral if the dropped segments
   are provably below the detection threshold. Validate by [COST] on one full eval.
2. **Classify-path (qwen3-vl-plus) pruning — the banked NOCLS lever.** All 6 latency
   attempts targeted the DETECT path; the classify call is a second dominant cost
   never touched this session. Skip/short-circuit classify when the detect verdict is
   decisive (single unambiguous category), keeping payloads independent (avoids FM2/FM3).
3. **Whole-video early-exit with a CALIBRATED, less-conservative gate (v127 remedy).**
   The dual-AND fired 0 times; raise `clip_normal_gate_floor` / switch to a single
   calibrated CLIP-margin OR-gate so it fires on the addressable hard-case population,
   while keeping scanned-video curves byte-identical. Pre-register a fire-rate AND a
   src-skipped-magnitude target (fire-rate alone is the wrong headline — v130 lesson).
4. **fps=2 -> adaptive frame sampling.** Lever #2 in the driver list, untouched.
   Reduce keyframe volume on low-motion regions; validate detection-set stability.
5. **AVOID for accuracy:** any VLM-call batching/merge/verdict-propagation
   (v97/v126/v129 all breached the floor — family refuted). AVOID pure local-compute
   trims as headline levers (v128: structurally capped below the call metric).

## Files (this session)
- iteration_logs/ideas/loop1_v124 ... loop7_v130 (7 idea/verdict files, 20260725-26)
- iteration_logs/session_v97_ledger.md (extended with v124-v130 rows incl. [COST])
- Committed code change: v124 Q-score CLI-default flip (config.from_args False->True,
  `--no_qscore_rerank` opt-out) + always-on `[COST]` instrumentation
  (llm_client.call_count + anomaly_expert.clip_call_count + pipeline detect wrapper).
- All v125-v130 mechanisms restored flag-off/removed byte-identical from .qoder/backups/.
