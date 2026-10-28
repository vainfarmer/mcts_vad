"""v180 SNH-P — MIL-peak protection rule pre-check (OFFLINE, zero VLM, zero eval spend).

Recomputes, LOCALLY and pipeline-exactly, the 32-snippet MIL scores for every
video that had at least one [SNH-FIRE] line in the v179 snh290 run, then
evaluates the pre-registered protection rule:

  PROTECTED iff [t_start, t_end] covers t_peak = (argmax(mil_scores)+0.5)/32*duration

Pipeline-exact recipe (mirrors pipeline.py Stage-1, L1711-1741):
  - VideoPreprocessor(fps=2, segment_duration=6, motion_threshold=1.0,
    min_motion_interval=2.0, segment_overlap=0.25)  (config.py defaults used by
    the v179 eval; segment params do NOT affect all_frames/timestamps)
  - VideoAnomalyExpert CLIP ViT-L-14/openai on mps -> precompute_video_features
  - MILAnomalyScorer(model_mil.pth): 32 consecutive-chunk mean-pool of the KEPT
    frames' CLIP feats -> one forward -> 32 snippet scores

GO GATE (pre-registered):
  (a) >= 4 of the 5 GT-harm detections protected; AND
  (b) est Normal-FA relative reduction >= 25%:
      est_fa = 25 (v179 FA-on) + resurrected (fully-vetoed Normal videos with
      >= 1 protected fire);  fa_rel = (76 - est_fa) / 76.

Degenerate MIL arrays (all scores identical, e.g. all-zero -> argmax undefined)
are treated as NOT protected and counted in the report.

Usage:
  python3.14 iteration_logs/analysis_v180_snhp_precheck.py
Resumable: per-video results cached to results/v180_mil_argmax.json.
"""

import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from video_preprocessor import VideoPreprocessor          # noqa: E402
from anomaly_expert import VideoAnomalyExpert             # noqa: E402
from mil_model import MILAnomalyScorer                    # noqa: E402

V179_LOG = Path('/Users/zzl/PycharmProjects/vad/results/batch_eval_v179_snh290.log')
V179_DIR = Path('/Users/zzl/PycharmProjects/vad/results/batch_eval_v179_snh290')
VID_ROOT = Path('/Users/zzl/Movies/ucf-crime/videos')
CACHE = Path('/Users/zzl/PycharmProjects/vad/results/v180_mil_argmax.json')

CATS = ["Abuse", "Arrest", "Arson", "Assault", "Burglary",
        "Explosion", "Fighting", "RoadAccidents", "Robbery",
        "Shooting", "Shoplifting", "Stealing", "Vandalism"]

DEV = 'mps'
NUM_SNIPPETS = 32

# The 5 v179 GT-harm detections (G4 failure), (vid_stem, t_start, t_end)
GT_HARM = [
    ('RoadAccidents128_x264', 3.00, 8.00),
    ('Explosion025_x264', 10.00, 16.50),
    ('Shooting048_x264', 46.00, 52.00),
    ('Shoplifting022_x264', 45.50, 55.00),
    ('RoadAccidents002_x264', 9.00, 11.50),
]

V179_FA_ON = 25     # Normal videos with >=1 det after v179 SNH veto
V179_FA_OFF = 76    # Normal videos with >=1 det with SNH off


def parse_fires():
    """All [SNH-FIRE] lines -> list of dicts."""
    rx = re.compile(
        r'\[SNH-FIRE\] vid=(\S+) idx=(\d+) cat=(\S+) '
        r't=([\d.]+)-([\d.]+) conf=([\d.]+)->([\d.]+) p_normal=([\d.]+)')
    fires = []
    with open(V179_LOG) as f:
        for line in f:
            m = rx.search(line)
            if m:
                fires.append({
                    'vid': m.group(1), 'idx': int(m.group(2)),
                    'cat': m.group(3), 't_start': float(m.group(4)),
                    't_end': float(m.group(5)), 'conf': float(m.group(6)),
                    'p_normal': float(m.group(8)),
                })
    return fires


def parse_snh_vid_normal_fullveto():
    """[SNH-VID] gate=1 Normal lines with ndet==vetoed>0 -> detection-free set."""
    out = {}
    with open(V179_LOG) as f:
        for line in f:
            if '[SNH-VID]' not in line or 'Normal_Videos' not in line:
                continue
            mv = re.search(r'vid=(\S+)', line)
            mn = re.search(r'ndet=(\d+)', line)
            mve = re.search(r'vetoed=(\d+)', line)
            mg = re.search(r'gate=(\d)', line)
            if not (mv and mn and mve and mg):
                continue
            stem = mv.group(1).replace('.mp4', '')
            ndet, vetoed, gate = int(mn.group(1)), int(mve.group(1)), int(mg.group(1))
            out[stem] = {'ndet': ndet, 'vetoed': vetoed, 'gate': gate,
                         'detection_free': (vetoed == ndet and ndet > 0)}
    return out


def resolve_path(stem):
    if stem.startswith('Normal_Videos'):
        return VID_ROOT / 'Testing_Normal' / f'{stem}.mp4'
    cat = re.match(r'[A-Za-z]+', stem).group(0)
    return VID_ROOT / cat / f'{stem}.mp4'


def mil_snippet_scores(mil_model, clip_feats):
    """pipeline.py L1724-1741 verbatim recipe: 32 consecutive chunks, mean-pool."""
    n_frames = clip_feats.shape[0]
    if n_frames < NUM_SNIPPETS:
        return None
    seg_len = n_frames // NUM_SNIPPETS
    snip = []
    for si in range(NUM_SNIPPETS):
        s = si * seg_len
        e = s + seg_len if si < NUM_SNIPPETS - 1 else n_frames
        snip.append(clip_feats[s:e].mean(dim=0))
    tensor = torch.stack(snip).unsqueeze(0).to(DEV)  # (1, 32, 768)
    with torch.no_grad():
        scores = mil_model(tensor).squeeze(0).cpu().numpy()
    return scores


def load_cache():
    if CACHE.exists():
        with open(CACHE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    tmp = CACHE.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(cache, f)
    tmp.replace(CACHE)


def extract(todo, cache):
    pre = VideoPreprocessor(fps=2, segment_duration=6, motion_threshold=1.0,
                            min_motion_interval=2.0, segment_overlap=0.25)
    expert = VideoAnomalyExpert(CATS, clip_model_name='ViT-L-14',
                                clip_pretrained='openai', device=DEV)
    assert expert.model is not None, 'CLIP failed to load'
    mil_model = MILAnomalyScorer(feature_dim=768, hidden_dim=512, k=3,
                                 dropout=0.6)
    mil_path = Path(__file__).resolve().parent.parent / 'model_mil.pth'
    mil_model.load_state_dict(torch.load(str(mil_path), map_location=DEV,
                                         weights_only=True))
    mil_model.eval().to(DEV)
    print(f'[SNHP] models loaded (CLIP ViT-L-14/openai + {mil_path.name}) '
          f'on {DEV}', flush=True)

    for i, stem in enumerate(todo):
        t0 = time.time()
        vp = resolve_path(stem)
        try:
            vd = pre.process(str(vp))
            frames = vd['all_frames']
            duration = float(vd['duration'])
            feats = expert.precompute_video_features(frames)
            mil = mil_snippet_scores(mil_model, feats)
            rec = {
                'video_path': str(vp),
                'duration': duration,
                'n_frames': int(len(frames)),
                'mil_ok': mil is not None,
            }
            if mil is not None:
                mil = np.asarray(mil, dtype=np.float64).reshape(-1)
                am = int(np.argmax(mil))
                rec['mil_scores'] = [round(float(x), 6) for x in mil]
                rec['argmax'] = am
                rec['t_peak'] = (am + 0.5) / NUM_SNIPPETS * duration
                rec['mil_max'] = float(mil.max())
                rec['mil_min'] = float(mil.min())
                # degenerate: all scores identical -> argmax undefined
                rec['degenerate'] = bool(mil.max() == mil.min())
            else:
                rec['degenerate'] = True
            # cross-check vs eval JSON duration
            ej = V179_DIR / f'{stem}.json'
            if ej.exists():
                with open(ej) as f:
                    rec['eval_duration'] = json.load(f)['duration']
            cache[stem] = rec
            save_cache(cache)
            print(f'[SNHP] {i + 1}/{len(todo)} {stem} frames={len(frames)} '
                  f'dur={duration:.2f} mil={"ok" if mil is not None else "NA"} '
                  f'argmax={rec.get("argmax")} t_peak={rec.get("t_peak") and round(rec["t_peak"], 2)} '
                  f'degen={rec["degenerate"]} in {time.time() - t0:.1f}s',
                  flush=True)
        except Exception as e:
            cache[stem] = {'error': str(e), 'mil_ok': False, 'degenerate': True}
            save_cache(cache)
            print(f'[SNHP-ERROR] {stem}: {e}', flush=True)
            traceback.print_exc()


def analyze(fires, cache, normal_vids):
    print('\n[SNHP-ANALYSIS] ' + '=' * 60, flush=True)
    fires_total = len(fires)
    fires_protected = 0
    per_fire = []
    for fr in fires:
        rec = cache.get(fr['vid'], {})
        ok = rec.get('mil_ok', False) and not rec.get('degenerate', True)
        t_peak = rec.get('t_peak') if ok else None
        covered = bool(ok and t_peak is not None
                       and fr['t_start'] <= t_peak <= fr['t_end'])
        fires_protected += covered
        per_fire.append({**fr, 't_peak': t_peak, 'covered': covered})
        print(f"  fire vid={fr['vid']} idx={fr['idx']} "
              f"t={fr['t_start']:.2f}-{fr['t_end']:.2f} "
              f"t_peak={t_peak if t_peak is None else round(t_peak, 2)} "
              f"covered={covered}", flush=True)

    # (a) GT-harm coverage
    print('[SNHP-HARM]', flush=True)
    harm_lines = []
    harm_protected = 0
    for vid, ts, te in GT_HARM:
        fr = next(f for f in per_fire
                  if f['vid'] == vid and abs(f['t_start'] - ts) < 0.01
                  and abs(f['t_end'] - te) < 0.01)
        rec = cache.get(vid, {})
        line = (f"{vid} window={ts:.2f}-{te:.2f} "
                f"mil_argmax_t={fr['t_peak'] if fr['t_peak'] is None else round(fr['t_peak'], 2)} "
                f"covered={fr['covered']}")
        harm_lines.append(line)
        harm_protected += fr['covered']
        print(f'  {line}', flush=True)

    # (b) Normal FA resurrection estimate
    fullveto_normals = sorted(v for v, d in normal_vids.items()
                              if d['detection_free'])
    resurrected = []
    for v in fullveto_normals:
        vf = [f for f in per_fire if f['vid'] == v]
        if any(f['covered'] for f in vf):
            resurrected.append(v)
    est_fa = V179_FA_ON + len(resurrected)
    fa_rel = (V179_FA_OFF - est_fa) / V179_FA_OFF
    print(f'[SNHP-FA] fully_vetoed_normals={len(fullveto_normals)} '
          f'resurrected={len(resurrected)} {resurrected}', flush=True)
    print(f'[SNHP-FA] est_fa_videos={est_fa} fa_rel_est={fa_rel:.4f} '
          f'(need >= 0.25)', flush=True)

    degen = sorted(v for v, r in cache.items()
                   if r.get('degenerate', True) or not r.get('mil_ok', False))
    print(f'[SNHP-DEGEN] degenerate_or_missing_mil={len(degen)} {degen}',
          flush=True)

    go = (harm_protected >= 4) and (fa_rel >= 0.25)
    print(f'[SNHP-VERDICT] go={go} harm_protected={harm_protected}/5 '
          f'fires_total={fires_total} fires_protected={fires_protected} '
          f'fa_rel_est={fa_rel:.4f}', flush=True)
    return {
        'go': go, 'harm_protected': harm_protected,
        'fires_total': fires_total, 'fires_protected': fires_protected,
        'resurrected': resurrected, 'est_fa': est_fa, 'fa_rel_est': fa_rel,
        'harm_lines': harm_lines, 'degenerate': degen,
        'fullveto_normals': fullveto_normals,
    }


def main():
    fires = parse_fires()
    normal_vids = parse_snh_vid_normal_fullveto()
    vids = sorted({f['vid'] for f in fires})
    cache = load_cache()
    todo = [v for v in vids if v not in cache]
    print(f'[SNHP] fires={len(fires)} distinct_vids={len(vids)} '
          f'cached={len(cache)} todo={len(todo)} '
          f'fullveto_normals={sum(1 for d in normal_vids.values() if d["detection_free"])}',
          flush=True)
    if todo:
        extract(todo, cache)
    result = analyze(fires, load_cache(), normal_vids)
    out = Path('/Users/zzl/PycharmProjects/vad/results/v180_snhp_precheck_result.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[SNHP] result written to {out}', flush=True)


if __name__ == '__main__':
    main()
