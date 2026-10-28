# Loop 2 — v98: Verified FCCS Baseline Establishment — GATE FAILED (second flag omission); decision: REVERT

**Date:** 2026-07-18
**Version:** v98
**Decision: REVERT (kept=false)** — enforced by D4: all three v98-touched files restored from `.qoder/backups/v98/` and verified byte-identical via diff; `pipeline.py` re-validated with `ast.parse`.

---

## 1. Approved proposal (summary)

**Title:** v98 Verified FCCS Baseline Establishment — execute the never-run v97 full-coverage comparative scan with a mandatory log-verification gate (zero code delta, one controlled architecture switch: directed partial-coverage independent judging -> full-coverage batched comparative judging); DGEA approved-in-principle as the NEXT loop with C's 2 mandated refinements.

**Premise (re-verified true by C and D1):** the recorded "v97 FCCS" baseline never ran FCCS — `results/batch_eval_v97_full/run.log` has 59 `[Directed scan]` / 0 `[FCCS]` lines (smoke 13/0), and the colocated `evaluation_report.json` matches the ledger to the last digit (0.16154893439350917 / 0.15034965034965034). Wiring was verified correct and complete (config.py:80/:153, batch_evaluate.py:189-191, pipeline.py:854-885/:692-705, anomaly_discovery.py:403-462); `--full_coverage_scan` was simply never passed at launch.

**Controlled change:** zero code edits; run smoke (14 videos) and full (65 videos) with `--full_coverage_scan` ON against the byte-identical post-v97 tree. Everything downstream untouched (0.3 threshold, `_merge_adjacent`, candidate_top_k=3, v51 half-max boundary on 0.75*CLIP+0.25*MIL, v95/v96 modulators, hard NMS, gates).

**Mandatory verification gate:** before accepting any number — `[FCCS] Full-coverage comparative scan` > 0 (one per video reaching Stage 4) AND `[Directed scan]` == 0 in the run log. Gate fails -> STOP, diagnose, record nothing.

**Decision rule:** FCCS mAP >= 0.16155 - 0.03 -> adopt FCCS as new baseline, DGEA measures delta against it; < 0.13155 -> the "FM2 refuted for wholesale replacement" lesson is itself refuted and DGEA is re-framed as the rescue mechanism.

## 2. C's feedback (summary)

APPROVE. C independently re-verified: (a) the v97 premise is TRUE (59/0 and 13/0 log counts, JSONs match ledger); (b) all five wiring points correct (`[Directed scan]` at pipeline.py:1273, not :1332 — immaterial drift); (c) the head-slice defect is real (`_comparative_call` truncates `keyframes[:2]` vs ~12 frames in directed `scan_segment`), so Scenario B (regression via ~6x per-segment evidence reduction) is genuinely live; (d) `clip_regions=[]` side-effect-free, and FCCS also skips the Stage-3 CLIP region loop (small latency saving); (e) FM1 untouched, FM2 correctly demoted VALIDATED -> UNDER TEST, FM3/FM4 absent; (f) full-coverage batched comparative scanning was never actually evaluated, so not a rejected alternative. **Key audit finding: `evaluation_report.json` persists NO run args/flags — mislabeling is structurally undetectable from the report alone; grep gate necessary but not sufficient -> mandate 1: persist the invocation (COMMAND.txt).** Mandates: (1) COMMAND.txt per run; (2) count-consistent gate cross-checked vs per-video JSONs; (3) record-correction banners in ledger + v97 idea file. Sequenced DGEA loop approved-in-principle with refinements (within-batch water-filling to exactly batch_size x k_base images/call; spike-robust k_base=1 placement), R1/R2 kept, >60-segment video in smoke recall check, per-video wrong-location conversion as primary criterion.

## 3. Implementation notes (D1)

Zero code delta to the evaluated system. Pre-flight wiring re-verified unmodified. Work done (procedural/docs only; backups in `.qoder/backups/v98/` before editing): pipeline.py docstring v98 entry (diff-verified docstring-only); session_v97_ledger.md correction banner + v97 row relabeled + v98 PENDING row; v97 idea file RECORD CORRECTION banner, KEEP annotated void-as-FCCS-validation, FM2 lesson demoted to UNDER TEST; new v98 idea file `loop1_v98_verified_fccs_baseline_20260718.md`; COMMAND.txt pre-written for both runs with `--full_coverage_scan` and all defaults explicit. Validation: ast.parse OK, import OK.

## 4. Results

| run | mAP | AP@0.3 | AP@0.5 | AP@0.7 | avg latency (s/video) | videos |
|---|---|---|---|---|---|---|
| v98 smoke | 0.15034965034965034 | 0.25524 | 0.10490 | 0.09091 | 144.51 | 14/14 ok |
| v98 full | 0.14503631011344897 | 0.20509 | 0.11858 | 0.11144 | 147.9 | 63 ok / 2 failed |

Reference: v97 (directed-scan, mislabeled as FCCS): smoke 0.15034965034965034, full 0.16154893439350917, latency 141.47s.

### GATE EVIDENCE (D4, post-hoc — the decisive finding)

| log | `[FCCS] Full-coverage comparative scan` | `[FCCS]` (any) | `[Directed scan]` |
|---|---|---|---|
| batch_eval_v98_smoke/run.log | **0** | **0** | **13** |
| batch_eval_v98_full/run.log | **0** | **0** | **59** |

**THE MANDATORY GATE FAILED — AGAIN.** Both v98 runs took the directed-scan branch on every video (13 and 59 directed lines, zero FCCS lines of any kind), exactly reproducing the v97 failure signature. The code wiring is confirmed correct in the restored tree (config.py:80/:153, batch_evaluate.py:189-191 + :102 `VADConfig.from_args`, pipeline.py:854 `self.config.discovery.full_coverage_comparative_scan` branch, anomaly_discovery.py:436/:461 log points), and COMMAND.txt contains `--full_coverage_scan` — so the flag was not honored at launch: the actually-executed invocation differed from COMMAND.txt (second consecutive operator/runner omission), and the pre-acceptance grep gate was not enforced before the numbers were recorded.

Consequences:
- The v98 numbers are **NOT FCCS measurements**. They are a **third directed-scan recording**: smoke mAP is bit-identical to v97 smoke (0.15034965034965034 — deterministic temp=0 directed path on the same 14 videos); full mAP 0.14504 vs v97's 0.16155 is a -0.0165 drift, inside the ±0.03 API-stochasticity band.
- **FCCS has still never been evaluated.** The "FM2 refuted for wholesale mechanism replacement" lesson remains UNDER TEST — neither confirmed nor refuted by v98.
- The latency numbers (144.51s smoke / 147.9s full) are directed-scan latencies, not FCCS latencies.

## 5. Keep/revert decision + reason

**REVERT.** Two independent grounds: (1) The experiment is void — its own mandatory activation gate failed (0 FCCS lines, 13/59 directed lines), so per the proposal's gate protocol no FCCS result exists to keep; the measured system was the v96/v97 directed configuration, which is already the running baseline and needs no adoption. (2) Even taken at face value as a directed re-run, full mAP 0.14504 < 0.16155 (within noise, no improvement), so there is no performance case for keeping anything. The v98 delta was documentation-only (docstring/ledger/idea banners + COMMAND.txt); D4 restored all three backed-up files byte-identical. The v98 idea file and this report remain as the historical record; COMMAND.txt files and eval artifacts persist under `results/batch_eval_v98_{smoke,full}/` with copies in `iteration_logs/results/v98_{smoke,full}_{run.log,eval.json}`.

## 6. Lessons learned (failure modes)

- **FM2 (wholesale Stage-4 mechanism replacement safety): still UNTESTED, not refuted and not validated.** FCCS never fired; no evidence about comparative full-coverage judging was produced. The sequenced DGEA loop must NOT treat v98 as a baseline for anything.
- **New confirmed failure mode — repeated flag omission + gate non-enforcement (FM5, process):** the same operator omission occurred twice in a row (v97, v98) despite correct wiring, a pre-written COMMAND.txt containing the flag, and an explicit mandatory grep gate. Root cause this time: the runner did not execute COMMAND.txt verbatim (or stripped the flag), and the accepting eval agent extracted metrics from `evaluation_report.json` — which persists no args/flags — without running the grep gate. **Standing methodology upgrade:** the gate must be executed and its counts pasted into the eval handoff BEFORE metrics are reported; a run whose log lacks mechanism-activation lines must be treated as failed, not merely suspect. Prefer launching via `bash COMMAND.txt`-style execution of the persisted file (or a wrapper that greps the log at exit) so the persisted invocation and the executed invocation cannot diverge.
- **FM1/FM3/FM4:** untouched/absent as designed (no code delta existed to affect them).
- **Process positive:** the backup/restore + diff-verify protocol worked cleanly — REVERT was enforceable in minutes with byte-identical verification, and the grep gate (run post-hoc by D4) unambiguously diagnosed the void experiment from the logs alone, validating C's mandate that log evidence, not the eval JSON, is the activation source of truth.

## 7. Next loop (unchanged, still blocked)

The v98 experiment must be **re-run correctly** (flag verified in the live process command line AND grep gate passed before accepting numbers) before DGEA proceeds; DGEA remains approved-in-principle with C's two refinements, measured against the first REAL FCCS number.
