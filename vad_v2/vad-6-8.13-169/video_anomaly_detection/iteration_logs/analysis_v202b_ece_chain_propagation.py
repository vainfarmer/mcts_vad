"""v202b ECE dual-arm rig probe: END-TO-END PROPAGATION through the COMMITTED chain.

WHY THIS EXISTS (and what it is NOT). The real 12-video smoke
(results/batch_eval_v202_ece_selftest) produced 5 MATERIAL [ECE] extensions but
`n_win_changed=0 / n_conf_changed=0` on 12/12 videos, i.e. the two arms were
byte-identical at the FINAL output. Two very different explanations fit that
observation:

  (H1) ABSORPTION — the committed qscore->BUE->CBT chain re-derives the same
       final window whether or not ECE nudged the input (traced on
       RoadAccidents019: ECE 40.5-42.5 -> 40.0-43.5, BUE -> 10.5-43.5,
       CBT -> 26.00-28.00, and the no-ECE arm lands on 26.00-28.00 TOO because
       CBT anchors on the curve PEAK, not on the incoming boundary).
  (H2) A BROKEN RIG that silently returns arm-ON as arm-OFF, which would make
       EVERY delta exactly zero and every future verdict a false null.

analysis_v202_ece_rig_selftest.py already refutes H2 with the downstream chain
OFF. This probe closes the remaining gap: it refutes H2 with the COMMITTED chain
fully ON (qscore + BUE + bue_closure_gate + CBT + cbt_recal), i.e. in exactly the
configuration the real eval runs.

THE TRICK (no rig logic is re-implemented here — that would defeat the point):
both arms are produced by the SAME real function
`MultiAnomalyVAD._ab_ablate_arm_off_replay`, from the SAME pre-ECE snapshot, and
differ ONLY in `ab_ablate`:

  arm-ON  : ab_ablate='cnv'  -> _ab_ablate_scoring resets the cnv_ family, which
                               is ALREADY at its committed default, so
                               cfg_changed==() and the replay runs the FULL
                               committed chain *including* ECE.
  arm-OFF : ab_ablate='ece'  -> the event_continuation_extension family is reset
                               to False, so the replay runs the SAME chain minus
                               exactly one function call.

So a difference between the two outputs is attributable to the ECE call alone,
and it is produced by the shipped rig code, not by a test-local mirror of it.

No pipeline default is touched: every ScoringConfig here is constructed locally.

Run: python3.14 -m video_anomaly_detection.iteration_logs.analysis_v202b_ece_chain_propagation
"""
import copy

import numpy as np

from video_anomaly_detection.config import VADConfig, ScoringConfig
from video_anomaly_detection.pipeline import (MultiAnomalyVAD,
                                              CATEGORY_DURATION_PRIORS)


class FakeFrameScorer:
    """Deterministic stand-in for the cached CLIP category curve.

    Both methods are pure functions of their arguments, exactly like the cached
    real ones (which is the property that makes the terminal chain replayable —
    PROJECT_DOCUMENTATION 8.3.2).
    """

    def __init__(self, curve):
        self.curve = np.asarray(curve, dtype=float)

    def get_category_scores(self, video_data, category):
        return self.curve

    def compute_frame_scores(self, video_data, anomalies, category):
        # qscore reads this; it is a function of the ARM's OWN windows, so it is
        # rebuilt per arm exactly as the real replay does.
        out = np.zeros(len(self.curve), dtype=float)
        ts = np.asarray(video_data['timestamps'], dtype=float)
        for a in anomalies or []:
            m = (ts >= float(a['t_start'])) & (ts <= float(a['t_end']))
            out[m] = np.maximum(out[m], float(a.get('confidence', 0.0)))
        return out.tolist()


def make_vad(curve, scoring):
    cfg = VADConfig()
    cfg.scoring = scoring
    vad = object.__new__(MultiAnomalyVAD)      # no CLIP / no VLM needed
    vad.config = cfg
    vad.frame_scorer = FakeFrameScorer(curve)
    vad.anomaly_categories = list(CATEGORY_DURATION_PRIORS.keys())
    return vad


def committed_scoring(mech):
    """The COMMITTED pipeline (v182 defaults) + ECE ON + the rig set to `mech`."""
    return ScoringConfig(
        event_continuation_extension=True,   # the mechanism under test
        ab_ablate=mech,
        # committed default-ON chain, stated explicitly so the probe is readable
        qscore_rerank=True,
        bue_enabled=True,
        bue_closure_gate=True,
        cbt_enabled=True,
        cbt_recal=True,
    )


def run_case(name, cat, curve, video_data, base):
    """Return (arm_on, arm_off) for one synthetic detection under the committed chain."""
    on_vad = make_vad(curve, committed_scoring('cnv'))    # ECE stays ON
    off_vad = make_vad(curve, committed_scoring('ece'))   # ECE ablated

    b1, b2 = copy.deepcopy(base), copy.deepcopy(base)
    on_win, on_conf, on_changed = on_vad._ab_ablate_arm_off_replay(
        b1, video_data, cat, '/tmp/probe_v202b.mp4')
    off_win, off_conf, off_changed = off_vad._ab_ablate_arm_off_replay(
        b2, video_data, cat, '/tmp/probe_v202b.mp4')

    print(f'--- {name} (cat={cat}, det={[[a["t_start"], a["t_end"]] for a in base]}) ---')
    print(f'  arm-ON  cfg_changed={on_changed}  win={on_win} conf={[round(c, 6) for c in on_conf]}')
    print(f'  arm-OFF cfg_changed={off_changed}  win={off_win} conf={[round(c, 6) for c in off_conf]}')
    return (on_win, on_conf, on_changed), (off_win, off_conf, off_changed)


def build_curve(n, dt, plateau=None, peak=None, floor=0.10):
    rel = np.full(n, floor)
    if plateau:
        lo, hi, val = plateau
        rel[int(lo / dt):int(hi / dt) + 1] = val
    if peak:
        pi, val = peak
        rel[int(pi / dt)] = val
    return rel


def main():
    dt = 0.5
    n = 241                                     # 0..120s on a 0.5s grid
    ts = np.arange(n, dtype=float) * dt
    video_data = {'timestamps': ts, 'duration': 120.0,
                  'video_path': '/tmp/probe_v202b.mp4'}

    results = []

    # CASE A — the ABSORPTION geometry (H1), i.e. what the real smoke saw.
    # A single sharp peak inside a wide half-max plateau: BUE grows both arms far
    # past the CBT over-wide trigger, then CBT re-anchors both on the SAME peak.
    cat_a = 'Shooting'
    curve_a = build_curve(n, dt, plateau=(30.0, 80.0, 0.90), peak=(55.0, 1.00))
    base_a = [{'t_start': 50.0, 't_end': 52.0, 'confidence': 0.70,
               'anomaly_type': cat_a, 'depth': 1, 'visits': 3}]
    on_a, off_a = run_case('CASE A  absorption geometry', cat_a, curve_a,
                           video_data, base_a)
    same_a = on_a[0] == off_a[0]
    print(f'  [A] arms identical at FINAL (absorption reproduced) : {same_a}')
    results.append(('A-absorption-reproduced', same_a))

    # CASE B — the PROPAGATION geometry. Two SEPARATED half-max islands: the
    # detection sits on the small island, ECE grows it across into the tall one,
    # so the two arms end on DIFFERENT CBT peak anchors. Same committed chain.
    cat_b = 'Shooting'
    curve_b = build_curve(n, dt, floor=0.05)
    curve_b[int(48.0 / dt):int(58.0 / dt) + 1] = 0.62      # island the det is on
    curve_b[int(58.5 / dt):int(76.0 / dt) + 1] = 0.99      # taller island
    curve_b[int(66.0 / dt)] = 1.00                          # its peak
    base_b = [{'t_start': 50.0, 't_end': 53.0, 'confidence': 0.70,
               'anomaly_type': cat_b, 'depth': 1, 'visits': 3}]
    on_b, off_b = run_case('CASE B  propagation geometry', cat_b, curve_b,
                           video_data, base_b)
    diff_b = on_b[0] != off_b[0]
    print(f'  [B] arms DIFFER at FINAL through the COMMITTED chain : {diff_b}')
    results.append(('B-propagates-through-committed-chain', diff_b))

    # CASE C — CBT-no-op geometry: keep the final duration under the over-wide
    # trigger (1.1*cap) so CBT cannot re-anchor, and check ECE survives.
    cat_c = 'RoadAccidents'                                 # recal cap 4.5s
    curve_c = build_curve(n, dt, floor=0.05)
    curve_c[int(50.0 / dt):int(53.0 / dt) + 1] = 0.95
    curve_c[int(51.0 / dt)] = 1.00
    base_c = [{'t_start': 50.5, 't_end': 51.5, 'confidence': 0.70,
               'anomaly_type': cat_c, 'depth': 1, 'visits': 3}]
    on_c, off_c = run_case('CASE C  CBT-no-op geometry', cat_c, curve_c,
                           video_data, base_c)
    diff_c = on_c[0] != off_c[0]
    print(f'  [C] arms differ (ECE survives a CBT no-op)           : {diff_c}')
    results.append(('C-survives-cbt-noop', diff_c))

    # STRUCTURAL invariants that must hold in EVERY case (FM2 + attribution).
    len_ok = all(len(on[0]) == len(off[0]) == 1
                 for on, off in [(on_a, off_a), (on_b, off_b), (on_c, off_c)])
    print(f'\n[c-len]  FM2 count frozen across arms in all 3 cases  : {len_ok}')
    results.append(('len-frozen', len_ok))

    attr_ok = all(on[2] == () and off[2] == ('event_continuation_extension',)
                  for on, off in [(on_a, off_a), (on_b, off_b), (on_c, off_c)])
    print(f'[c-attr] arms differ by EXACTLY the ECE call           : {attr_ok}')
    results.append(('attribution-exactly-ece', attr_ok))

    # The decisive claim: propagation must be demonstrated at least once with the
    # committed chain ON, otherwise a zero delta is not interpretable.
    propagates = diff_b or diff_c
    print(f'[c-prop] SOME committed-chain geometry propagates ECE  : {propagates}')
    results.append(('propagation-demonstrated', propagates))

    ok = all(v for _, v in results if _ != 'A-absorption-reproduced') and len_ok
    print('\n' + '\n'.join(f'  {k:38s} {v}' for k, v in results))
    print(f'\nV202B CHAIN-PROPAGATION PROBE: {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
