# v108 — TSG Live Completion Test (v104 mechanism, first full in-vivo evaluation + pre-registered coverage adjudication)

## Approved Proposal

- **Title**: TSG Live Completion Test (v104 mechanism, first full in-vivo evaluation + pre-registered coverage adjudication)
- **Mechanism**: Temporal-Structure-Guided scan targeting (already implemented in the tree behind `--temporal_structure_scan`): replaces ONLY the flat-CLIP fallback branch segment-selection ranking with EventVAD-style temporal-gradient+motion peak picking; exact VLM-call-count parity with incumbent pre-screen; judgment/merge/boundary/fusion/NMS byte-identical; `[TSG-GATE]` hard gate in `batch_evaluate.py`.
- **Rationale**: 28.8% of wrong-location candidates concentrate in flat-CLIP categories (Shoplifting/Stealing/Arson/Burglary, ~20/65 videos take the Stage 4 fallback); incumbent generic CLIP pre-screen std is only 0.001-0.005 there, so scanned segments are near-random. Smoke passed (mAP 0.1443, 14/14 markers, fired 1/14). Full eval stalled at 5/65 in the previous session — this loop completes it and runs the pre-registered adjudication.
- **Implementation plan**: Verify-only: confirm `--temporal_structure_scan` in `batch_evaluate.py --help`, `[TSG]` markers in anomaly_discovery, then run the full 65-video eval with the flag ON.
- **Expected gain (pre-registered decision rule)**: coverage not improved -> REVERT regardless of headline; coverage improved + mAP within 0.1494+/-0.03 -> keep code flag-OFF (no new baseline); coverage improved + mAP > 0.1794 -> KEEP-FLAG-ON (new baseline).
- **Risks**: FM2 bounded by exact call-count parity + unchanged downstream; FM1/FM3/FM4 absent by construction; FM5 eliminated by `[TSG-GATE]` hard gate; FM6 n/a (no simulation evidence used).
- **Latency impact**: Zero extra CLIP/VLM/MIL calls (<20ms numpy); latency-neutral.
- **Literature**: EventVAD (arXiv 2504.13092); LAVAD (CVPR 2024); UBoCo (CVPR 2022)/GEBD; T3AL (CVPR 2024); LEC-VAD (ICML 2025).

## C's Feedback

Pre-approved as v104 (C verdict with 4 mandated items, previous session). This loop only completes the stalled full eval and runs the pre-registered adjudication.

## Implementation Notes (D1)

Verify-only loop — no files changed. v104 TSG already implemented and verified present in the tree; smoke already passed (0.1443, 14/14 markers, fired 1/14); full evaluated with `--temporal_structure_scan`.

## Eval Flags + Activation Marker

- Flags: `--temporal_structure_scan`
- Activation marker: `[TSG]`; hard gate: `[TSG-GATE]`
- Marker counts: smoke log = 15 raw `[TSG]` lines (14 video-level markers, fired 1/14) + 1 `[TSG-GATE]`; full log = 69 `[TSG]` lines + 1 `[TSG-GATE]` (65 videos, 64 succeeded, 1 failed). GATE PASSED — mechanism genuinely ran in vivo.

## Results

| Metric | Smoke (reused v104 smoke, flag-on) | Full (65 videos, flag-on) |
|---|---|---|
| mAP | 0.1443 | 0.15833597680401443 |
| AP@0.3 | — | 0.21232555106299725 |
| AP@0.5 | — | 0.1490460157126824 |
| AP@0.7 | — | 0.11363636363636363 |
| avg latency (s/video) | — | 302.99 |
| marker_count | 14 | 69 |

Note: the smoke run was reused from v104 (flag-on, TSG-GATE ok, fired 1/14); copied as v108 smoke artifacts with source noted.

## Pre-Registered TSG Adjudication (coverage)

GT from HF parquet test split (frames@30fps, same as batch_evaluate evaluate(fps=30)); segment times reconstructed exactly via VideoPreprocessor(fps=2, segment_duration=6.0, segment_overlap=0.25, motion_threshold=1.0, min_motion_interval=2.0) = the actual v108 run config (stride 4.5s; verified: Shoplifting001 dur=144.8s with segment idx 29 only valid at stride 4.5).

Per-video (selected vs baseline, GT-covered seconds / GT segments hit):

| Video | Selected segs | Sel coverage | Baseline segs | Base coverage | Delta |
|---|---|---|---|---|---|
| Shoplifting001_x264 | [3,7,8,24,28,29] | 0.00s, 0/1 | [4,5,7,11,14,15] | 7.50s, 1/1 | -7.50s / -1 |
| Shoplifting020_x264 | [19,20,33,34,39,40] | 0.00s, 0/1 | [14,17,18,19,41,42] | 4.00s, 1/1 | -4.00s / -1 |
| Vandalism007_x264 | [0,2,3,4,6,7] | 15.00s, 1/1 | [0,1,2,3,5,6] | 14.00s, 1/1 | +1.00s / 0 |
| Vandalism028_x264 | [13,14,19,20,27,28] | 8.50s, 2/2 | [12,13,14,18,19,30] | 13.00s, 2/2 | -4.50s / 0 |

- AGGREGATE (exact): selected 23.50s & 3/5 GT segments vs baseline 38.50s & 5/5 => delta -15.00s, -2 segments (strictly worse).
- Sensitivity under the protocol-literal nominal grid (segment i = [6i, 6i+6)): selected 24.33s & 3/5 vs baseline 25.67s & 3/5 => delta -1.33s, +0 (also not better).
- Verdict false under BOTH mappings. Fired=4, not-applicable=61 ([TSG] markers=65, succeeded=64).

## Decision: REVERT

Pre-registered adjudication: GT-overlap coverage NOT improved by selected vs baseline segment sets — revert regardless of headline mAP. Mechanism stays in the tree flag-OFF (`--temporal_structure_scan` default False in `batch_evaluate.py`, `run.py`, and `config.py` line 191 default False); no baseline change; code inert. No backup restore needed (verify-only loop; no v108 backup dir exists; tree confirmed default-OFF).

## Lessons Learned

- FM5 (flag never actually enabled) definitively refuted for TSG: `[TSG-GATE]` hard gate passed, 69 full-log markers, fired 4/64 — the mechanism genuinely ran live, so this is a true negative result, not an activation artifact.
- FM1 (mechanism does the wrong thing) confirmed: EventVAD-style temporal-gradient+motion peak picking selected segments with strictly WORSE GT coverage than the near-random flat-CLIP incumbent (-15.0s, -2/5 GT segments) — temporal-structure peaks in UCF-Crime flat categories do not align with anomaly location; the incumbent's apparent randomness was not the bottleneck.
- Headline mAP (0.1583 vs v97 baseline ~0.1616, within the +/-0.03 noise band) would have looked "neutral-acceptable" — the pre-registered coverage adjudication was essential to avoid keeping a mechanism that actively misdirects scan budget.
- Simulation-free, in-vivo completion plus exact segment-time reconstruction (stride 4.5s verification) is the right protocol; both grid mappings agreed, so the revert is robust to segment-indexing ambiguity.
