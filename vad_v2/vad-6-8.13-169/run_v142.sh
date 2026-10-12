#!/bin/bash
# v142 BUE — Boundary-Union Extension. Launcher for the FIXED harness.
#
# Usage:
#   ./run_v142.sh smoke_on    # flag-ON 3-video smoke (FM5 activation + M4 materiality + M3 anti-stack)
#   ./run_v142.sh smoke_off   # flag-OFF 3-video smoke (code-path identity)
#   ./run_v142.sh full        # the ONE verification run (65 videos, concurrency 2)
#
# The harness is FIXED by owner instruction: --max_per_category 5 --concurrency 2.
# concurrency 3 exhausted system swap and hung a run for 70 min. If a run stalls
# >30 min with no log growth, kill it and relaunch — the runner SKIPS videos whose
# result JSON already exists, so resumption is cheap.
set -u
source /Users/zzl/PycharmProjects/vad/video_anomaly_detection/.env 2>/dev/null
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

MODE="${1:-full}"
case "$MODE" in
  full)      OUT=../results/batch_eval_v142_full;      MPC=5; BUE="--bue_extension" ;;
  smoke_on)  OUT=../results/batch_eval_v142_smoke_on;  MPC=1; BUE="--bue_extension" ;;
  smoke_off) OUT=../results/batch_eval_v142_smoke_off; MPC=1; BUE="--no_bue_extension" ;;
  *) echo "usage: $0 {full|smoke_on|smoke_off}"; exit 2 ;;
esac

cd /Users/zzl/PycharmProjects/vad/video_anomaly_detection
ABS_OUT="$(python3.14 -c "import os,sys;print(os.path.abspath(sys.argv[1]))" "$OUT")"
mkdir -p "$ABS_OUT"

# ---------------------------------------------------------------------------
# M2 — PRE-REGISTERED DECISION RULE, written into the run log BEFORE launch.
# vad_logger opens run.log with logging.FileHandler default mode='a' (append),
# so this block is preserved ahead of every logged line of the run itself.
# ---------------------------------------------------------------------------
cat >> "$ABS_OUT/run.log" <<'PREREG'
=== v142 BUE PRE-REGISTRATION (M2) — written BEFORE the run started ===
Mechanism: BUE, Boundary-Union Extension. Terminal-most slot, after _qscore_rerank,
before the v141 CEX block. UNION-ONLY outward growth of each FINAL detection to the
outer-inner-contrast (OIC) argmax window on the read-only category curve, length
capped at min(bue_lmax_sec=40 s, 0.8*duration), v51 half-max closure gate ON.
Only t_start/t_end are written; confidence, detection count, anomaly_type and rank
order are frozen bit-for-bit; zero new VLM/CLIP/MIL calls.

EXPECTED paired point delta: +0.030 to +0.040.

MEASURED VARIANCE FOR THIS OPERATOR (the sole justification for the threshold
relaxation below, stated in advance): P(delta>0) on 6 caches had mean 0.891,
range 0.835-0.952, with only 3/6 >= 0.90. The video-level bootstrap half-width for
a CONCENTRATED BOUNDARY operator is +/-0.055 — 2.2x the +/-0.025 quoted for the
re-rank deltas the +/-0.025 noise-floor figure was derived from. A 0.90 gate would
therefore FAIL roughly half the time on a genuine +0.04 effect.

DECISION RULE (fixed now, never to be chosen after seeing P):
  KEEP iff paired point delta >= 0.02 AND P(delta>0) >= 0.85 AND full mAP >= 0.15132.
  If P lands in [0.85, 0.90) the KEEP is recorded as PROVISIONAL-ON-VARIANCE with
  the bootstrap half-width quoted — never silently as a clean pass.
  If P < 0.85 or delta < 0.02 or mAP < 0.15132 -> REVERT (flip bue_enabled default
  back to False; it is already False).

ADJUDICATION COMMAND (M1 — exact paired path, no regex parser):
  python3.14 video_anomaly_detection/iteration_logs/analysis_bootstrap_ab.py \
      --new results/batch_eval_v142_full --paired-json pre_bue_boundaries
Secondary evidence only (cross-run, NOT the verdict):
  ... --new results/batch_eval_v142_full --ref results/batch_eval_v124_full

PRE-LAUNCH REPLAY OF THE VERDICT PATH ON CACHED RUNS (for honesty, recorded now):
  v124: delta +0.03719  P=0.831 -> REVERT under this rule
  v130: delta +0.02828  P=0.823 -> REVERT under this rule
  v132: delta +0.04727  P=0.946 -> KEEP
  i.e. the point-delta gate passes 3/3 but the P>=0.85 gate passes only 1/3 on
  cached replays. This is known BEFORE launch and the rule is not being changed
  for it.

M6 PARAMETER RECORD: bue_lmax_sec=40. L=40 and L=50 are BIT-IDENTICAL because the
0.8*duration constraint binds above 40. 24/24 arm-cache combinations over
Lmax in {15,20,25,30,35,40,50,60} x 3 caches are POSITIVE; L=90 collapses to
-0.00117 (1/3 positive). 12-cache out-of-sample read: 9 held-out caches, 9/9
positive, mean +0.0370 — that is the prior the next loop should inherit, not the
proposal's understated 3/3. The closure gate is a measured mAP NO-OP (gated ==
ungated bit-identical at Lmax 40 and 60 on 3/3 caches) and is kept only to reduce
blast radius (67 fires instead of 78).
UNION IS THE MECHANISM: closure-gated REPLACE measures -0.0065 mean vs UNION
+0.0378 mean on the same 3 caches.

M3 ANTI-STACK: cex_extend stays False; `grep -c CEX run.log` must be 0.
M4 MATERIALITY GATE (would have caught v107 ECE, which passed ACTIVATION with 160
markers while its median growth was 0.000 s): the flag-ON smoke requires median
per-fire growth > 2.0 s AND mean post/pre duration ratio > 1.5. Offline replay on
3 caches gives median growth 10.5-11.5 s and mean ratio 4.85-5.14.
=== END PRE-REGISTRATION ===
PREREG

echo "mode=$MODE out=$ABS_OUT bue=$BUE"
python3.14 -u batch_evaluate.py \
  --dataset_dir /Users/zzl/Movies/ucf-crime \
  --output_dir "$OUT" \
  --max_per_category "$MPC" \
  --concurrency 2 \
  --detect_model qwen-vl-plus \
  --classify_model qwen3-vl-plus \
  --fps 2 \
  --clip_model ViT-L-14 \
  --clip_pretrained openai \
  --device mps \
  --mcts_simulations 12 \
  --mcts_max_depth 4 \
  $BUE \
  2>&1 | tee -a "$ABS_OUT/../$(basename "$ABS_OUT").log"
