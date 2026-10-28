"""v21 SNH Agent-C RECORDS verification (FM20): recompute every load-bearing number
from the PRODUCED artifacts only (results/batch_eval_v179_snh290 + its .log + GT).
ZERO model calls, ZERO eval runs — pure JSON/log/parquet analysis. Read-only.

Checks:
 1. marker counts [SNH-VID]=290, [SNH-FIRE]=92, [SNH-COLLISION]=4, [SNH-ERR]=0
 2. rounded (vid,t_start,t_end) keys among fires collapse 92 -> 87 (M4 VOID path)
 3. exact idx= keyed restore: 0/92 mismatches; arm-OFF mAP == 0.17120316
 4. arm-ON mAP reproduces evaluation_report.json (0.17324230348112937)
 5. point delta and video-level bootstrap CI/P (same protocol as
    analysis_bootstrap_ab.py: seed=0, n=1000, percentile 5/95) on idx-restored arms
 6. Normal-video FA objective both arms (150 videos): FA-video rate + fp dets/video
 7. GT-harm: vetoed detections on anomaly videos overlapping GT at IoU>=0.1 (fps=30)
 8. gate fire rates from [SNH-VID] lines: normal vs anomaly
"""
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import load_ground_truth, load_predictions, compute_temporal_map  # noqa: E402

RUN = ROOT / 'results' / 'batch_eval_v179_snh290'
LOG = ROOT / 'results' / 'batch_eval_v179_snh290.log'
LABEL_PATH = '/Users/zzl/Movies/ucf-crime/data'
FPS = 30

lines = LOG.read_text(errors='replace').splitlines()
n_vid = sum(1 for l in lines if '[SNH-VID]' in l)
n_fire = sum(1 for l in lines if '[SNH-FIRE]' in l)
n_coll = sum(1 for l in lines if '[SNH-COLLISION]' in l)
n_err = sum(1 for l in lines if '[SNH-ERR]' in l)
print(f'1. markers: SNH-VID={n_vid} SNH-FIRE={n_fire} SNH-COLLISION={n_coll} SNH-ERR={n_err}')

fire_re = re.compile(r'\[SNH-FIRE\] vid=(?P<vid>\S+) idx=(?P<idx>\d+) cat=(?P<cat>\S+) '
                     r't=(?P<ts>[0-9.]+)-(?P<te>[0-9.]+) conf=(?P<old>[0-9.]+)->(?P<new>[0-9.]+)')
fires = {}
for l in lines:
    m = fire_re.search(l)
    if m:
        fires.setdefault(m.group('vid'), []).append(
            (int(m.group('idx')), float(m.group('ts')), float(m.group('te')),
             float(m.group('old'))))
n_lines = sum(len(v) for v in fires.values())
keys = {(v, round(ts, 2), round(te, 2)) for v, fl in fires.items() for _, ts, te, _ in fl}
print(f'2. fire lines={n_lines} unique rounded keys={len(keys)} '
      f'(dup videos: {sorted(v for v, fl in fires.items() if len({(round(ts,2), round(te,2)) for _, ts, te, _ in fl}) < len(fl))})')

gt = load_ground_truth(LABEL_PATH, split='test')
pred_on = load_predictions(str(RUN))
report = json.loads((RUN / 'evaluation_report.json').read_text())
rep_map = report['evaluation']['temporal_map']['mAP']
rep_nv = report['timing']['num_videos']

# 3. exact idx= keyed restore
pred_off = copy.deepcopy(pred_on)
mism = 0
restored = 0
for vid, fl in fires.items():
    an = pred_off[vid]['anomalies']
    for idx, ts, te, conf in fl:
        a = an[idx]
        if abs(a['t_start'] - ts) > 1e-6 or abs(a['t_end'] - te) > 1e-6:
            mism += 1
            print(f'   MISMATCH {vid} idx={idx}: json {a["t_start"]}-{a["t_end"]} vs log {ts}-{te}')
        a['confidence'] = conf
        restored += 1
print(f'3. idx-restore: {restored} restored, {mism} mismatches')

gte = {k: v for k, v in gt.items() if k in pred_on}
print(f'   gt videos common to preds: {len(gte)}')
m_on = compute_temporal_map(gte, {k: pred_on[k] for k in gte}, fps=FPS)
m_off = compute_temporal_map(gte, {k: pred_off[k] for k in gte}, fps=FPS)
print(f'4. arm-ON  mAP={m_on["mAP"]:.14f} (report {rep_map:.14f}, num_videos={rep_nv}) '
      f'APs={[round(m_on[f"AP@{t}"], 8) for t in (0.3, 0.5, 0.7)]}')
print(f'   arm-OFF mAP={m_off["mAP"]:.14f} APs={[round(m_off[f"AP@{t}"], 8) for t in (0.3, 0.5, 0.7)]}')
point = m_on['mAP'] - m_off['mAP']
print(f'5. point delta (ON-OFF) = {point:+.8f}')

# bootstrap, same protocol as analysis_bootstrap_ab.py (seed=0, n=1000)
vids = sorted(gte)
rng = np.random.default_rng(0)
ds = []
for _ in range(1000):
    samp = rng.choice(vids, size=len(vids), replace=True)
    g, a, b = {}, {}, {}
    for i, v in enumerate(samp):
        k = f'{v}#{i}'
        g[k], a[k], b[k] = gt[v], pred_off[v], pred_on[v]
    ds.append(compute_temporal_map(g, b, fps=FPS)['mAP']
              - compute_temporal_map(g, a, fps=FPS)['mAP'])
ds = np.array(ds)
print(f'   bootstrap: mean {ds.mean():+.5f} 90% CI [{np.percentile(ds, 5):+.5f}, '
      f'{np.percentile(ds, 95):+.5f}] P(delta>0)={(ds > 0).mean():.3f} sd {ds.std():.5f}')

# 6. FA objective on the 150 Normal test videos
def fa(p):
    nv = [v for v in p if v.startswith('Normal_Videos')]
    n_fa = sum(1 for v in nv if any(a.get('confidence', 0) > 0 for a in p[v]['anomalies']))
    n_det = sum(sum(1 for a in p[v]['anomalies'] if a.get('confidence', 0) > 0) for v in nv)
    return len(nv), n_fa, n_fa / len(nv), n_det, n_det / len(nv)

print(f'6. FA OFF: {fa(pred_off)}')
print(f'   FA ON : {fa(pred_on)}')

# 7. GT harm on anomaly-video vetoes
rows = []
for vid, fl in fires.items():
    if vid.startswith('Normal_Videos'):
        continue
    segs = gt[vid]['segments']
    for idx, ts, te, conf in fl:
        best = 0.0
        for gs, ge in segs:
            ov = max(0.0, min(te * FPS, ge) - max(ts * FPS, gs))
            un = max(te * FPS, ge) - min(ts * FPS, gs)
            if un > 0:
                best = max(best, ov / un)
        rows.append((vid, ts, te, conf, best))
rows.sort(key=lambda r: -r[4])
for vid, ts, te, conf, iou in rows:
    if iou > 0.05:
        print(f'7. {vid} t={ts:.2f}-{te:.2f} conf={conf:.4f} IoU={iou:.4f}')
print(f'   gt_vetoed (IoU>=0.1): {sum(1 for r in rows if r[4] >= 0.1)} of '
      f'{len(rows)} anomaly-video vetoes')

# 8. gate fire rates from [SNH-VID]
gate_lines = [l for l in lines if '[SNH-VID]' in l]
g1 = [l for l in gate_lines if ' gate=1 ' in l]
g1n = sum(1 for l in g1 if 'vid=Normal_Videos_' in l)
nn = sum(1 for l in gate_lines if 'vid=Normal_Videos_' in l)
print(f'8. gate=1: {len(g1)}/{len(gate_lines)} total; normal {g1n}/{nn} '
      f'({g1n / nn:.1%}); anomaly {len(g1) - g1n}/{len(gate_lines) - nn} '
      f'({(len(g1) - g1n) / (len(gate_lines) - nn):.1%})')
