"""v202 ECE dual-arm rig self-test: ARM INDEPENDENCE + NON-ALIASING.

Complements the real 12-video smoke (results/batch_eval_v202_ece_selftest).
The smoke proves the *0-fire* direction empirically (arm-OFF byte-identical to
final on videos where ECE changed nothing that survived). This script proves the
*differs* direction and the deepcopy hazard directly, by calling the REAL rig
function `MultiAnomalyVAD._ab_ablate_arm_off_replay` on a synthetic but
fully-deterministic curve, with the downstream committed chain (BUE/CBT) turned
OFF **for the test only** so that ECE's growth is observable at the output.

WHY the downstream chain must be off for THIS test: the smoke established that
the committed chain ABSORBS ECE (BUE grows outward by up to lmax~36s and CBT
then clamps to the curve peak, so the final window is a function of the peak
location, not of the incoming boundary). Absorption is a property of BUE/CBT,
NOT of the rig; aliasing is a property of the deepcopy. Disabling BUE/CBT
isolates the rig property under test. No pipeline default is touched — the
ScoringConfig used here is constructed locally.

Run:  python3.14 -m video_anomaly_detection.iteration_logs.analysis_v202_ece_rig_selftest
"""
import copy
import numpy as np

from video_anomaly_detection.config import VADConfig, ScoringConfig
from video_anomaly_detection.pipeline import MultiAnomalyVAD, CATEGORY_DURATION_PRIORS


class FakeFrameScorer:
    """Deterministic stand-in for the cached CLIP category curve + frame scores.

    Only the two methods the terminal chain calls are implemented; both are pure
    functions of their arguments, exactly like the cached real ones.
    """

    def __init__(self, curve):
        self.curve = np.asarray(curve, dtype=float)

    def get_category_scores(self, video_data, category):
        return self.curve

    def compute_frame_scores(self, video_data, anomalies, category):
        # Shape-compatible output curve (qscore reads it); value irrelevant here.
        return [0.0] * len(self.curve)


def build_probe():
    """A curve with a WIDE plateau above half-max so ECE grows materially."""
    n = 121
    ts = np.arange(n, dtype=float) * 1.0          # 1s grid, 0..120s
    rel = np.full(n, 0.10)
    rel[40:61] = 0.90                             # 40..60s plateau (21s)
    rel[50] = 1.00                                # peak
    video_data = {'timestamps': ts, 'duration': 120.0}
    return video_data, rel


def main():
    cat = 'Shoplifting'                            # max_dur 20.0s -> 21s plateau
    max_dur = CATEGORY_DURATION_PRIORS[cat]['expected_duration'][1]
    video_data, rel = build_probe()

    # ECE ON + rig ON; BUE/CBT/qscore OFF so ECE's growth is observable.
    scoring = ScoringConfig(
        event_continuation_extension=True,
        ab_ablate='ece',
        qscore_rerank=False,
        bue_enabled=False,
        cbt_enabled=False,
    )
    cfg = VADConfig()
    cfg.scoring = scoring

    vad = object.__new__(MultiAnomalyVAD)          # no CLIP / no VLM needed
    vad.config = cfg
    vad.frame_scorer = FakeFrameScorer(rel)
    vad.anomaly_categories = list(CATEGORY_DURATION_PRIORS.keys())

    base = [{
        't_start': 46.0, 't_end': 48.0, 'confidence': 0.7,
        'anomaly_type': cat, 'depth': 1, 'visits': 3,
    }]
    base_before = copy.deepcopy(base)

    # --- arm-ON: the real path (ECE mutates IN PLACE, as in the pipeline) ---
    arm_on_in = copy.deepcopy(base)
    arm_on = vad._event_continuation_extension(arm_on_in, video_data, cat)
    on_win = [[float(a['t_start']), float(a['t_end'])] for a in arm_on]
    on_conf = [float(a['confidence']) for a in arm_on]

    # --- arm-OFF: the REAL rig function on the pre-ECE snapshot ---
    off_win, off_conf, changed = vad._ab_ablate_arm_off_replay(
        base, video_data, cat, '/tmp/probe_v202.mp4')

    print(f'category={cat} max_dur={max_dur}s  plateau=40-60s  det=[46.0,48.0]')
    print(f'cfg_changed        = {changed}')
    print(f'arm-ON  windows    = {on_win}   conf={on_conf}')
    print(f'arm-OFF windows    = {off_win}  conf={off_conf}')

    c1 = off_win is not None and len(off_win) == len(on_win) == len(base)
    print(f'[c-len]   FM2 count frozen across arms (len==1)            : {c1}')

    c2 = off_win != on_win
    print(f'[c-diff]  FIRING det -> arms DIFFER (no aliasing/no leak)  : {c2}')

    c3 = base == base_before
    print(f'[c-alias] caller snapshot UNMUTATED by the replay          : {c3}')

    c4 = changed == ('event_continuation_extension',)
    print(f'[c-cfg]   arms differ by EXACTLY the ECE flag              : {c4}')

    # 0-fire control: a detection already longer than max_dur -> ECE skips ->
    # the two arms must be byte-identical.
    base0 = [{
        't_start': 30.0, 't_end': 70.0, 'confidence': 0.7,
        'anomaly_type': cat, 'depth': 1, 'visits': 3,
    }]
    on0 = vad._event_continuation_extension(copy.deepcopy(base0), video_data, cat)
    on0_win = [[float(a['t_start']), float(a['t_end'])] for a in on0]
    off0_win, off0_conf, _ = vad._ab_ablate_arm_off_replay(
        base0, video_data, cat, '/tmp/probe_v202.mp4')
    c5 = off0_win == on0_win
    print(f'[c-zero]  0-FIRE det -> arms BYTE-IDENTICAL {on0_win}: {c5}')

    ok = all([c1, c2, c3, c4, c5])
    print(f'\nRIG SELF-TEST: {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
