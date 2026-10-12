# v187 — SNH-C (window-level CLIP-curve protection for the SNH veto): NO-GO (last non-MIL protection signal refuted; Normal-FP video-level-veto line EXHAUSTED)

Date: 2026-08-18. Precheck agent run (27 tool calls, ~26 min local CLIP). NO batch_evaluate, NO VLM, NO source edits. Artifacts: iteration_logs/analysis_v187_snhc_precheck.py, results/v187_snhc_precheck.json (+ .out).

## 1. What was proposed

After MIL-protection was permanently closed (v180 argmax: mislocalized; v186 threshold: signal ABSENT — video-level MIL max 0.039/0.00025 on misfires), SNH-C tested the last non-MIL protection signal: a fired detection survives iff the curve of ITS OWN anomaly_type category has peak z >= 2.0 inside the detection window (v18 median/MAD recipe). Documented prior: SNH's video-level features already include curve stats, so the prior was not high — the window-level measurement was the genuinely new information.

## 2. Phase-0 precheck (60 fired videos, pipeline-exact curve recompute)

**Validity clean**: recomputed kept-frame timestamps match the v179 eval arrays exactly (max abs diff = 0.0, counts equal, 0 errors).

Gate (a) >= 4/5 GT-harm survive: **FAIL — 3/5**:

| detection | window z | survives |
|---|---|---|
| RoadAccidents128 3.0-8.0 | 1.0260 | no |
| Explosion025 10.0-16.5 | **7.2242** | yes |
| Shooting048 46.0-52.0 | **12.3445** | yes |
| Shoplifting022 45.5-55.0 | 1.3618 | no |
| RoadAccidents002 9.0-11.5 | **3.4381** | yes |

Gate (b) FA rel reduction >= 25%: **FAIL — 0.2105**. 58/92 fires survive; 35/51 fully-vetoed Normal videos resurrect -> est_fa = 60, rel = (76-60)/76.

## 3. The precise failure structure (why this line is now exhausted)

- Unlike MIL (signal ABSENT on misfires), the curve signal EXISTS at the harm windows (z up to 12.3) — but the SAME signal fires on Normal-video false positives: 35/51 Normal videos have at least one detection window with curve z >= 2.0. The curve-window signal cannot separate "vetoed true positive" from "vetoed false positive" — both look like real peaks. And it is precisely this ambiguity the SNH head was exploiting at video level.
- Therefore: no cheap per-window protection of the video-level SNH veto exists in ANY of the pipeline's signal families (MIL: absent; curve: non-separating). The -67.1% FA win of v179 cannot be salvaged at video-gate granularity without accepting the 5 GT harms.

## 4. CLOSED directions (do not re-propose — FINAL)

- **ALL protection variants of the SNH video-level veto** (MIL-argmax v180, MIL-threshold+abstain v186, curve-window v187). The Normal-FP owner objective via video-level veto is EXHAUSTED: unsupervised cascade (v17 CNV) harmed, supervised head (v179 SNH) harmed, and no signal in the tree can repair the harm.
- Remaining Normal-FP paths, if the owner wants to continue, are qualitatively different builds, not loop tweaks: (i) a window-level SUPERVISED head needs window-level labels the train split lacks (pseudo-labeling via MIL was just shown empty on the critical videos); (ii) accept v179 --snh_veto as a documented opt-in (FA -67%, 5 GT harms, abs mAP held) for deployment profiles that tolerate the harm.

## 5. Pipeline state

UNCHANGED: committed = v137 + BUE + CBT + CBT-recal (all default-ON); snh_veto stays default-OFF opt-in; no new code.
