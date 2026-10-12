"""v186 SNH-W — window-level MIL survival rule pre-check (OFFLINE, zero eval spend).

Reuses the v180 MIL cache (results/v180_mil_argmax.json), which stores the FULL
32-snippet mil_scores per video for every one of the 60 videos that fired the
v179 SNH gate -> NO MIL recompute is needed (verified at runtime; the script
asserts full scores are present for every MIL-valid fired video).

RULE (pre-registered):
  When the video-level SNH gate fires, a detection [t_start, t_end] SURVIVES iff
      max(mil_scores[s] for covered snippets s) >= 0.5
  where snippet s covers (s/32*duration, (s+1)/32*duration) and the detection
  covers snippets floor(ts/dur*32) .. floor(te/dur*32), clamped to [0, 31].

  ABSTAIN carve-out: videos with NO valid mil_scores (too few kept frames for
  the 32-chunk MIL forward) are NOT gated at all -> every detection survives.
  Known no-MIL videos: RoadAccidents128, Explosion025, RoadAccidents002,
  Explosion016, RoadAccidents004, Normal_Videos_889.

GO GATE (pre-registered):
  (a) >= 4 of the 5 GT-harm detections survive (abstention counts as survive);
  AND
  (b) est Normal-FA relative reduction >= 25%:
      est_fa = 25 (v179 FA-on) + #resurrected fully-vetoed Normal videos
               (a fully-vetoed Normal video is resurrected if ANY of its fired
               detections survives, or it abstains);
      fa_rel = (76 - est_fa) / 76 >= 0.25.

Inputs (all cached):
  - results/batch_eval_v179_snh290.log   (92 [SNH-FIRE] lines)
  - results/v180_mil_argmax.json         (per-video 32 mil_scores / mil_ok flag)
  - results/batch_eval_v179_snh290/<stem>.json  (post-veto per-video results)

Usage:
  python3 iteration_logs/analysis_v186_snhw_precheck.py
Writes:
  results/v186_snhw_precheck.json
"""

import json
import math
import re
from pathlib import Path

V179_LOG = Path('/Users/zzl/PycharmProjects/vad/results/batch_eval_v179_snh290.log')
V179_DIR = Path('/Users/zzl/PycharmProjects/vad/results/batch_eval_v179_snh290')
CACHE = Path('/Users/zzl/PycharmProjects/vad/results/v180_mil_argmax.json')
OUT = Path('/Users/zzl/PycharmProjects/vad/results/v186_snhw_precheck.json')

NUM_SNIPPETS = 32
SURVIVE_THRESH = 0.5

# The 5 v179 GT-harm detections (v21 adjudication): (vid_stem, t_start, t_end, IoU)
GT_HARM = [
    ('RoadAccidents128_x264', 3.00, 8.00, 0.7333),
    ('Explosion025_x264', 10.00, 16.50, 0.5106),
    ('Shooting048_x264', 46.00, 52.00, 0.4286),
    ('Shoplifting022_x264', 45.50, 55.00, 0.4211),
    ('RoadAccidents002_x264', 9.00, 11.50, 0.2857),
]

V179_FA_ON = 25     # Normal videos with >=1 surviving det after v179 SNH veto
V179_FA_OFF = 76    # Normal videos with >=1 det with SNH off

FIRE_RX = re.compile(
    r'\[SNH-FIRE\] vid=(\S+) idx=(\d+) cat=(\S+) '
    r't=([\d.]+)-([\d.]+) conf=([\d.]+)->([\d.]+) p_normal=([\d.]+)')


def parse_fires():
    fires = []
    with open(V179_LOG) as f:
        for line in f:
            m = FIRE_RX.search(line)
            if m:
                fires.append({
                    'vid': m.group(1), 'idx': int(m.group(2)),
                    'cat': m.group(3),
                    't_start': float(m.group(4)), 't_end': float(m.group(5)),
                    'conf_old': float(m.group(6)),
                    'p_normal': float(m.group(8)),
                })
    return fires


def covered_snippets(ts, te, dur):
    lo = max(0, min(NUM_SNIPPETS - 1, math.floor(ts / dur * NUM_SNIPPETS)))
    hi = max(0, min(NUM_SNIPPETS - 1, math.floor(te / dur * NUM_SNIPPETS)))
    return lo, hi


def main():
    fires = parse_fires()
    cache = json.load(open(CACHE))
    assert len(fires) == 92, f'expected 92 fires, parsed {len(fires)}'

    fired_vids = sorted({f['vid'] for f in fires})
    missing = [v for v in fired_vids if v not in cache]
    assert not missing, f'fired videos missing from MIL cache: {missing}'

    # Verify every MIL-valid fired video really carries the full 32 scores
    # (this is what lets v186 run with ZERO recompute).
    for v in fired_vids:
        e = cache[v]
        if e.get('mil_ok'):
            assert isinstance(e.get('mil_scores'), list) and \
                len(e['mil_scores']) == NUM_SNIPPETS, \
                f'{v}: cache lacks full mil_scores -> would need recompute'

    # Fully-vetoed Normal videos in the v179 post-veto per-video JSONs
    # (Normal video, >=1 detection, all conf == 0 after veto).
    fully_vetoed_normal = set()
    for v in fired_vids:
        if not v.startswith('Normal_Videos'):
            continue
        d = json.load(open(V179_DIR / f'{v}.json'))
        anoms = d.get('anomalies', [])
        if anoms and all(a.get('confidence', 0.0) <= 0.0 for a in anoms):
            fully_vetoed_normal.add(v)

    # Per-fire evaluation of the SNH-W rule.
    table = []
    dur_mismatch = []
    for f in fires:
        v = f['vid']
        e = cache[v]
        # duration cross-check: per-video JSON vs cache
        jdur = json.load(open(V179_DIR / f'{v}.json'))['duration']
        dur = jdur
        if abs(jdur - e['duration']) > 0.05:
            dur_mismatch.append((v, jdur, e['duration']))
        abstain = not e.get('mil_ok')
        if abstain:
            row = {**f, 'abstain': True, 'snip_lo': None, 'snip_hi': None,
                   'window_mil_max': None, 'survives': True}
        else:
            lo, hi = covered_snippets(f['t_start'], f['t_end'], dur)
            wmax = max(e['mil_scores'][lo:hi + 1])
            row = {**f, 'abstain': False, 'snip_lo': lo, 'snip_hi': hi,
                   'window_mil_max': wmax,
                   'survives': bool(wmax >= SURVIVE_THRESH)}
        table.append(row)

    n_surv = sum(1 for r in table if r['survives'])

    # (a) GT-harm survival
    harm_rows = []
    for vid, ts, te, iou in GT_HARM:
        cand = [r for r in table if r['vid'] == vid
                and abs(r['t_start'] - ts) < 0.01
                and abs(r['t_end'] - te) < 0.01]
        assert cand, f'GT-harm fire not found: {vid} {ts}-{te}'
        r = cand[0]
        harm_rows.append({
            'vid': vid, 't_start': ts, 't_end': te, 'iou': iou,
            'abstain': r['abstain'],
            'snip_lo': r['snip_lo'], 'snip_hi': r['snip_hi'],
            'window_mil_max': r['window_mil_max'],
            'survives': r['survives'],
        })
    harm_saved = sum(1 for h in harm_rows if h['survives'])
    gate_a = harm_saved >= 4

    # (b) Normal-FA estimate: resurrected fully-vetoed Normal videos
    by_vid = {}
    for r in table:
        by_vid.setdefault(r['vid'], []).append(r)
    resurrected = sorted(v for v in fully_vetoed_normal
                         if any(r['survives'] for r in by_vid[v]))
    est_fa = V179_FA_ON + len(resurrected)
    fa_rel = (V179_FA_OFF - est_fa) / V179_FA_OFF
    gate_b = fa_rel >= 0.25

    go = bool(gate_a and gate_b)

    out = {
        'variant': 'v186_snhw',
        'rule': ('on SNH gate fire: detection survives iff '
                 'max(mil_scores[floor(ts/dur*32)..floor(te/dur*32)] clamped) '
                 f'>= {SURVIVE_THRESH}; videos with no valid mil_scores abstain '
                 '(all detections survive)'),
        'inputs': {'fires_log': str(V179_LOG), 'mil_cache': str(CACHE)},
        'go': go,
        'gate_a_harm_saved_ge_4': gate_a,
        'gate_b_fa_rel_ge_0.25': gate_b,
        'harm_saved': harm_saved,
        'harm_rows': harm_rows,
        'n_normal_fully_vetoed_v179': len(fully_vetoed_normal),
        'n_normal_resurrected': len(resurrected),
        'normal_resurrected': resurrected,
        'est_fa': est_fa,
        'fa_rel_est': fa_rel,
        'fires_surviving': n_surv,
        'fires_total': len(table),
        'n_abstain_videos': len({r['vid'] for r in table if r['abstain']}),
        'abstain_videos': sorted({r['vid'] for r in table if r['abstain']}),
        'duration_mismatches': dur_mismatch,
        'per_fire': table,
    }
    OUT.write_text(json.dumps(out, indent=2))

    # console summary
    print(f'GO: {go}  (gate_a harm_saved {harm_saved}/5 >= 4: {gate_a}; '
          f'gate_b fa_rel {fa_rel:.4f} >= 0.25: {gate_b})')
    print('GT-harm rows:')
    for h in harm_rows:
        wm = 'ABSTAIN' if h['abstain'] else f"{h['window_mil_max']:.6f}"
        print(f"  {h['vid']:<26} {h['t_start']:>6.2f}-{h['t_end']:>6.2f} "
              f"window_mil_max={wm:<10} survives={h['survives']}")
    print(f'fires surviving: {n_surv}/{len(table)}; '
          f'abstain videos: {out["n_abstain_videos"]}')
    print(f'normal resurrected: {len(resurrected)} -> est_fa={est_fa}, '
          f'fa_rel_est={fa_rel:.4f}')
    if dur_mismatch:
        print('DURATION MISMATCHES:', dur_mismatch)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
