# v186 — SNH-W (window-level MIL-threshold protection + short-video abstention for the SNH veto): NO-GO at Phase-0 offline precheck (SNH-protection direction PERMANENTLY CLOSED, zero eval spend)

Date: 2026-08-18. Precheck agent run (19 tool calls, ~7 min). NO batch_evaluate, NO VLM, NO source edits. Artifacts: iteration_logs/analysis_v186_snhw_precheck.py, results/v186_snhw_precheck.json (+ .out).

## 1. What was proposed

A rule attacking EXACTLY the two failure modes the v180 precheck documented for MIL-argmax protection:
- argmax misplacement -> replace argmax-coverage with a window-level THRESHOLD: a fired detection survives iff max(mil_scores over covered snippets) >= 0.5;
- short-clip MIL blindness -> ABSTAIN carve-out: videos with no valid mil_scores (<32 kept frames) are not gated at all.

## 2. Phase-0 precheck (reused the v180 cache — full 32 mil_scores for all 54 MIL-valid fired videos; zero recompute)

The 5 GT-harm detections under the rule:

| detection | window MIL max | survives? |
|---|---|---|
| RoadAccidents128 3.00-8.00 | — (no MIL) | yes (abstain) |
| Explosion025 10.00-16.50 | — (no MIL) | yes (abstain) |
| Shooting048 46.00-52.00 | **0.007758** | NO |
| Shoplifting022 45.50-55.00 | **0.000079** | NO |
| RoadAccidents002 9.00-11.50 | — (no MIL) | yes (abstain) |

harm_saved = 3/5, all via abstention. Gate (a) needs >= 4/5 => **FAIL**.

Gate (b) FA estimate: only 6/92 fires survive (exactly the 6 abstain-video fires); **ZERO of 86 MIL-valid fires reach 0.5** (max window max = 0.0815, median 0.0005). Resurrected Normal videos = 1 (Normal_Videos_889, abstain) -> est_fa = 26, fa_rel_est = 0.658 >= 0.25 => PASS (irrelevant; both gates required).

## 3. The decisive structural finding

On the videos where the SNH head misfires, the MIL signal is not mislocalized — it is **ABSENT**: Shooting048's VIDEO-level MIL max is 0.039, Shoplifting022's is 0.00025. No function of mil_scores (argmax, window threshold, top-k) can protect what the signal never marks. This closes the MIL-protection direction harder than v180 did: v180 showed the peak is in the wrong place; v186 shows there is effectively no peak.

## 4. CLOSED directions (do not re-propose — FINAL)

- **MIL-based protection of the SNH video-level veto, in ANY form** (argmax coverage v180; window threshold + abstention v186; any top-k/quantile variant — the signal is absent on exactly the misfire videos).
- The SNH whole-video veto family is now closed at three points: v21 (G4 harm), v180 (argmax protection NO-GO), v186 (threshold+abstain protection NO-GO). Any future Normal-FP lever must NOT be an SNH-veto variant; it must either (i) use a protection signal from a DIFFERENT family (e.g. window-level CLIP-curve z — noting the correlation caveat that SNH's own video-level curve features already looked normal on these videos), or (ii) abandon the video-level gate granularity entirely.

## 5. Pipeline state

UNCHANGED: committed = v137 + BUE + CBT + CBT-recal (all default-ON); snh_veto stays default-OFF in-tree; snh_protect/snh_wind NEVER implemented. v179's FA win (-67.1%) remains available as an opt-in flag with its documented G4 harm.
