"""v202c ECE dual-arm: RANDOMIZED ABSORPTION SWEEP through the COMMITTED chain.

WHAT THIS ANSWERS. The real 12-video smoke
(results/batch_eval_v202_ece_selftest) produced 5 MATERIAL [ECE] extensions and
`n_win_changed=0 / n_conf_changed=0` on 12/12 videos: every ECE fire was ERASED
before the output. analysis_v202_ece_rig_selftest.py proves that is NOT a broken
rig (the same shipped replay function returns a genuinely different arm-OFF when
the downstream chain is off). analysis_v202b_ece_chain_propagation.py failed to
find a hand-built geometry that propagates ECE through the COMMITTED chain.
This script replaces hand-picked geometries with a deterministic RANDOMIZED sweep
so the absorption rate is MEASURED over a population rather than asserted from
n=5, and so any surviving geometry is found if one exists.

THE STRUCTURAL PREDICTION BEING TESTED (pipeline.py `_bue_boundary_union_
extension`, committed default-ON since v142): BUE maximizes inner-minus-outer
contrast over EVERY (start, length) window that contains `anchor`, where
`anchor = i0 + argmax(cs[i0:i1])` is the curve argmax INSIDE the incoming
detection, with length in [bue_lmin_sec, min(bue_lmax_sec=40, frac*duration)].
So BUE's output depends on the incoming window ONLY through
  (1) that in-window argmax, and
  (2) the closure gate `cs[i0] < hm and cs[i1-1] < hm` (skip BUE entirely).
ECE grows OUTWARD inside the half-max level region of that same peak, so it
cannot move the argmax (a level-set cannot cross a sub-half-max valley) and
usually cannot flip the gate => BUE re-derives the SAME window from both arms,
and CBT then re-anchors both on the same peak. Prediction: ECE is structurally
REDUNDANT with the committed BUE outward growth, and its end-to-end delta is ~0.

METHOD (no rig logic is re-implemented — that would defeat the purpose): both
arms come from the SAME shipped function
`MultiAnomalyVAD._ab_ablate_arm_off_replay`, from the SAME pre-ECE snapshot,
differing ONLY in `ab_ablate` (identical trick to v202b):
  arm-ON  : ab_ablate='cnv' -> resets the cnv_ family, already at its committed
                              default => cfg_changed==() => the FULL committed
                              chain runs, ECE INCLUDED.
  arm-OFF : ab_ablate='ece' -> event_continuation_extension reset to False =>
                              the SAME chain minus exactly one function call.
No pipeline default is touched: every ScoringConfig here is built locally.

Run: python3.14 -m video_anomaly_detection.iteration_logs.analysis_v202c_ece_absorption_sweep
"""
import copy

import numpy as np

from video_anomaly_detection.config import VADConfig, ScoringConfig
from video_anomaly_detection.pipeline import (MultiAnomalyVAD,
                                              CATEGORY_DURATION_PRIORS)

HARD88_CATS = ['Shooting', 'RoadAccidents', 'Shoplifting', 'Explosion']
DT = 0.5
N = 241                                   # 0..120s on the 0.5s grid
SEED = 20260825
N_CASES = 400


class FakeFrameScorer:
    """Deterministic stand-in for the cached CLIP category curve + frame scores.

    Both methods are pure functions of their arguments, exactly like the cached
    real ones — the property that makes the terminal chain replayable
    (PROJECT_DOCUMENTATION 8.3.2).
    """

    def __init__(self, curve):
        self.curve = np.asarray(curve, dtype=float)

    def get_category_scores(self, video_data, category):
        return self.curve

    def compute_frame_scores(self, video_data, anomalies, category):
        out = np.zeros(len(self.curve), dtype=float)
        ts = np.asarray(video_data['timestamps'], dtype=float)
        for a in anomalies or []:
            m = (ts >= float(a['t_start'])) & (ts <= float(a['t_end']))
            out[m] = np.maximum(out[m], float(a.get('confidence', 0.0)))
        return out.tolist()


def committed_scoring(mech):
    """The COMMITTED v182 terminal chain + ECE ON + the rig set to `mech`."""
    return ScoringConfig(
        event_continuation_extension=True,
        ab_ablate=mech,
        qscore_rerank=True,
        bue_enabled=True,
        bue_closure_gate=True,
        cbt_enabled=True,
        cbt_recal=True,
    )


def make_vad(curve, scoring):
    cfg = VADConfig()
    cfg.scoring = scoring
    vad = object.__new__(MultiAnomalyVAD)   # no CLIP / no VLM needed
    vad.config = cfg
    vad.frame_scorer = FakeFrameScorer(curve)
    vad.anomaly_categories = list(CATEGORY_DURATION_PRIORS.keys())
    return vad


def random_case(rng):
    """A random curve + one detection that overlaps one of its islands."""
    floor = rng.uniform(0.02, 0.18)
    rel = np.full(N, floor)
    islands = []
    for _ in range(rng.integers(1, 4)):
        lo = rng.uniform(2.0, 100.0)
        span = rng.uniform(2.0, 40.0)
        hi = min(118.0, lo + span)
        val = rng.uniform(max(0.25, floor + 0.1), 1.0)
        i0, i1 = int(lo / DT), int(hi / DT) + 1
        rel[i0:i1] = np.maximum(rel[i0:i1], val)
        pk = rng.integers(i0, max(i0 + 1, i1))
        rel[pk] = min(1.0, val + rng.uniform(0.0, 0.15))
        islands.append((lo, hi))
    # optional smoothing so some cases have soft (non-cliff) shoulders
    if rng.random() < 0.5:
        k = int(rng.integers(2, 7))
        ker = np.ones(k) / k
        rel = np.convolve(rel, ker, mode='same')
    lo, hi = islands[int(rng.integers(0, len(islands)))]
    ds = rng.uniform(max(0.0, lo - 3.0), max(lo + 0.5, hi - 1.0))
    dur = rng.uniform(1.5, 15.0)
    de = min(119.5, ds + dur)
    ds, de = round(ds * 2) / 2, round(de * 2) / 2
    if de <= ds:
        de = ds + 1.0
    cat = HARD88_CATS[int(rng.integers(0, len(HARD88_CATS)))]
    return rel, cat, ds, de


def main():
    rng = np.random.default_rng(SEED)
    ts = np.arange(N, dtype=float) * DT
    video_data = {'timestamps': ts, 'duration': 120.0,
                  'video_path': '/tmp/probe_v202c.mp4'}

    n_ece_fire = n_final_diff = n_fire_and_diff = n_len_bad = n_none = 0
    survivors = []
    for i in range(N_CASES):
        rel, cat, ds, de = random_case(rng)
        base = [{'t_start': ds, 't_end': de, 'confidence': 0.70,
                 'anomaly_type': cat, 'depth': 1, 'visits': 3}]

        # (a) did ECE itself move this window? (isolated call, own deepcopy)
        ece_vad = make_vad(rel, committed_scoring('cnv'))
        ece_out = ece_vad._event_continuation_extension(
            copy.deepcopy(base), video_data, cat)
        ece_win = [round(float(ece_out[0]['t_start']), 4),
                   round(float(ece_out[0]['t_end']), 4)]
        fired = ece_win != [round(ds, 4), round(de, 4)]

        # (b) the two arms, both from the SHIPPED replay function
        on_vad = make_vad(rel, committed_scoring('cnv'))    # ECE stays ON
        off_vad = make_vad(rel, committed_scoring('ece'))   # ECE ablated
        on_win, on_conf, on_chg = on_vad._ab_ablate_arm_off_replay(
            copy.deepcopy(base), video_data, cat, '/tmp/probe_v202c.mp4')
        off_win, off_conf, off_chg = off_vad._ab_ablate_arm_off_replay(
            copy.deepcopy(base), video_data, cat, '/tmp/probe_v202c.mp4')
        if on_win is None or off_win is None:
            n_none += 1
            continue
        if not (len(on_win) == len(off_win) == 1):
            n_len_bad += 1
            continue
        assert on_chg == () and off_chg == ('event_continuation_extension',), \
            (on_chg, off_chg)

        diff = ([round(v, 6) for v in on_win[0]] != [round(v, 6) for v in off_win[0]]
                or round(on_conf[0], 9) != round(off_conf[0], 9))
        n_ece_fire += int(fired)
        n_final_diff += int(diff)
        if fired and diff:
            n_fire_and_diff += 1
        if diff:
            shift = (abs(on_win[0][0] - off_win[0][0])
                     + abs(on_win[0][1] - off_win[0][1]))
            survivors.append((i, cat, [ds, de], ece_win,
                              [round(v, 2) for v in on_win[0]],
                              [round(v, 2) for v in off_win[0]], round(shift, 2)))

    print(f'cases={N_CASES} seed={SEED} cats={HARD88_CATS} '
          f'chain=COMMITTED(qscore+BUE+gate+CBT+recal)')
    print(f'  ECE moved the window (arm-ON, pre-chain)  : {n_ece_fire}'
          f'  ({100.0 * n_ece_fire / N_CASES:.1f}%)')
    print(f'  arms DIFFER at the FINAL output           : {n_final_diff}'
          f'  ({100.0 * n_final_diff / N_CASES:.1f}%)')
    print(f'  ECE fired AND survived to the output      : {n_fire_and_diff}'
          f'  -> absorption rate '
          f'{100.0 * (n_ece_fire - n_fire_and_diff) / max(1, n_ece_fire):.1f}% '
          f'of {n_ece_fire} fires')
    print(f'  replay returned None / len mismatch       : {n_none} / {n_len_bad}')
    if survivors:
        print(f'  SURVIVING geometries (first 10 of {len(survivors)}):')
        for s in survivors[:10]:
            print(f'    case={s[0]:3d} cat={s[1]:14s} det={s[2]} ece={s[3]} '
                  f'armON={s[4]} armOFF={s[5]} |shift|={s[6]}s')
        med = float(np.median([s[6] for s in survivors]))
        print(f'  median |shift| among survivors: {med:.2f}s')
    else:
        print('  SURVIVING geometries: NONE — ECE is fully absorbed by the '
              'committed BUE/CBT chain on this population')

    # Instrument validity: the sweep is only interpretable if the rig CAN show a
    # difference. That is what analysis_v202_ece_rig_selftest.py proves; here we
    # additionally require that the sweep exercised real ECE fires.
    ok = (n_none == 0 and n_len_bad == 0 and n_ece_fire > 0)
    print(f'\nV202C SWEEP: {"VALID" if ok else "INVALID"} '
          f'(replay clean on all cases, ECE fires observed)')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
