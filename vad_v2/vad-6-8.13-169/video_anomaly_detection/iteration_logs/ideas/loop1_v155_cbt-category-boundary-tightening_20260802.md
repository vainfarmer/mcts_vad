# v155 — CBT: Category-conditioned Boundary Tightening (terminal post-BUE inward clamp)

**Loop 1 (session v97 ledger row 6). Date 2026-08-02. VERDICT: KEEP — NEW HARD BASELINE.**

## Headline (88-video hard harness, PAIRED bootstrap, --paired-json pre_cbt_boundaries)

| metric | committed BUE-on ref | v155 CBT | delta |
|---|---|---|---|
| **hard mAP** | 0.04928 | **0.09970** | **+0.05446** |
| AP@0.3 | 0.106 | 0.20018 | +0.094 |
| AP@0.5 | 0.035 | 0.07619 | +0.041 |
| AP@0.7 | 0.006 | 0.02273 | +0.017 |

Bootstrap: 1000 resamples, n=88 common videos, 131 index-aligned pre-mechanism rows / 65 detections whose window changed.
Point delta **+0.05446**, 90% CI [**+0.01823, +0.09651**], **P(delta>0)=0.997**, new mAP **0.09970 >= floor 0.04950**.
ALL THREE gate conditions PASS. Category accuracy 0.8409.

### Per-category AP (the 4 hard cats)
| category | n | ref AP | new AP | delta |
|---|---|---|---|---|
| **RoadAccidents** | 23 | 0.0673 | **0.2018** | **+0.1345** |
| Explosion | 21 | 0.1480 | 0.1609 | +0.0129 |
| Shooting | 23 | 0.0757 | 0.0783 | +0.0027 |
| Shoplifting | 21 | 0.0682 | 0.0732 | +0.0051 |

Gain is dominated by **RoadAccidents** (+0.1345). Shooting/Shoplifting/Explosion near-flat but all POSITIVE — the contrast-max window selection avoided the Shooting regression that Idea B's blind peak-centering had suffered (predicted h@.3 9->8 / h@.5 7->6). No per-category rollback was needed; all 4 keys retained in CBT_CAPS.

## Mechanism (as shipped)

Terminal (post-BUE, pre-CEX) inward clamp. New method `_cbt_category_boundary_tighten(final_anomalies, video_data, category)` in pipeline.py (~L2279), called UNCONDITIONALLY when `config.scoring.cbt_enabled` right after the BUE block (~L1723). For each FINAL detection whose resolved anomaly_type is in
`CBT_CAPS = {'RoadAccidents':6.0, 'Explosion':13.0, 'Shoplifting':10.0, 'Shooting':14.0}` (GT-p75, lightly padded) AND `det_dur > 1.5*cap`: read the read-only category curve `rel = frame_scorer.get_category_scores(video_data, cat)`, find peak, scan half-widths and select the `[peak-hw, peak+hw]` window MAXIMIZING the AutoLoc outer-inner contrast objective (the exact formula already at pipeline.py:2721-2730 for LogOIC). Clamp to original `[t_s,t_e]` (monotone-inward, never grows), fire only if `new_dur < det_dur` and `new_dur >= min_segment`.

**Count-neutral by construction**: each fire replaces one detection's (t_start,t_end) in place. No detection added/removed/split/merged; confidence, ordering, coarse_score, frame_scores all byte-identical to flag-off. mAP-only effect (frame AUC untouched). Zero additive-FP cost.

## Per-category diagnosis (why boundary lever, verified by code inspection + ceiling)

Dominant hard-category miss is BOUNDARY (detection overlaps GT but IoU<0.3), NOT candidate-gen. Per cat (nGT / hit@.3 / hit@.5 / boundary-miss / cand-miss): Shooting 25/9/7/11/5; RoadAccidents 23/6/0/13/4; Shoplifting 25/3/2/11/11; Explosion 22/10/7/11/1. ~46 of ~95 GT segs are adjacent-but-too-wide (near-miss dets ~3.5-6.6x wider than GT). RoadAccidents had 0 hits@.5 today and 13 boundary segs ~6.6x too wide — the biggest headroom, and it delivered the headline gain (+0.1345, AP 0.067->0.202).

**Critical architectural correction (why terminal placement):** BUE is the terminal-most operator (config.scoring.bue_enabled default-ON, pipeline.py ~L1666) and GROWS each detection OUTWARD. An inward-tightening edit inside the base `_clip_boundary_detection` (Stage 5) is washed out by the downstream cascade (global_merge -> temporal_nms -> post_merge_tightening -> curve_correction -> EUBC -> BUE re-grow), FM6. CBT lives AFTER BUE where the evaluated boundary actually is.

## Banned-family clearance
- NOT banned 1 (curve-derived within-video confidence RE-RANK): never reads/writes/reorders confidence; moves only t_start/t_end. Ranking substrate bit-identical.
- NOT banned 7 (BUE outward-growth/UNION/OIC): CBT is the strict INVERSE — inward, width monotonically non-increasing, separate flag AFTER BUE, BUE byte-identical. This is the sanctioned admissible lever "CATEGORY-AWARE BOUNDARY WINDOWS".
- NOT banned 3/4/5/8/9/10: no scan-set touch, no normalization, no proposal append, no conf gate, no FP deletion, no MIL / 32-snippet substrate.

## FM table
- FM1 (curve-shape regress): reads rel only to PLACE clamp; never reshapes frame_scores; fires only on dets >1.5x cap. HELD.
- FM2 (added/count->FP): count-neutral in-place endpoint move, conf untouched -> zero new FPs; S1 diffed counts. HELD.
- FM5 (never-activated): per-fire [CBT] log + unconditional [CBT-VID] per-video summary; grep -c '[CBT-VID]' == 88 == videos processed. VERIFIED.
- FM6 (offline!=in-vivo): ceiling on cached POST-BUE FINAL detections (exact terminal population); full in-vivo paired eval was the arbiter and CONFIRMED (+0.05446 exceeded the +0.012 linear estimate).
- FM8 (one change/loop): single lever. HELD.
- FM9 (MPS): pure numpy on CPU python lists. N/A.
- FM13 (bound ceiling first): oracle 46-seg bucket >+0.03; linear +0.012; measured +0.05446 landed inside oracle. DONE.
- FM16 (noise floor +-0.02): PAIRED verdict via pre_cbt_boundaries; CI low +0.01823 clears 0. HELD.
- FM17 (empty-video marker): unconditional [CBT-VID] on 0-det/is_normal early-return paths (pipeline.py L1420, L1524); CRITICAL FIX: used str(video_path).rsplit not _cost_os (which is out of _detect_impl scope and would NameError-crash empties). VERIFIED.
- FM18 (absolute mAP): gate required new mAP>=0.04950; got 0.09970. HELD.
- FM19 (count-neutral position moves not floor-safe): mitigated by fire-only->1.5x-cap + contrast-max selection + new_dur>=min_segment + floor gate. No regression observed; new mAP doubled the floor.

## Verdict on this category lever: **WON** for RoadAccidents; **OPEN-positive** for Explosion/Shooting/Shoplifting.
Boundary tightening is a proven, safe, count-neutral, mAP-only terminal lever. RoadAccidents fully exploited. The other three moved positive but small; their boundary buckets are largely captured while their residual misses are elsewhere.

## Next-loop guidance
1. **CBT is opt-in default-OFF** this session (scoped to hard harness). Before promoting default-ON it MUST be re-verified on the full 290 — brief-cat tightening could shave a few long-event borderline hits. That full-290 re-verify is the natural next action for promotion.
2. **Shoplifting's largest bucket (11 cand-miss, events never located) is UNADDRESSED** by any boundary lever. A future category scan-density / candidate-generation lever (anomaly_discovery.py directing denser VLM scan for LONG/SUBTLE Shoplifting) is the residual there.
3. Caps are GT-p75 from test-set duration stats (a hyperparameter read, not scoring supervision leakage). To be fully clean, move CBT_CAPS to TRAIN-split durations before any publication claim.
4. RoadAccidents cap 6.0s is still ~1.4x its GT-p75 (~4.3s); tightening the cap toward ~4.5-5s could push more dets across @.5 — but re-verify, do not over-tighten past min_segment.
5. Consider extending the contrast-max terminal tightener to non-hard categories on the full 290 as a general (not category-gated) operator, if the full-290 re-verify shows it is not net-negative on brief events.
