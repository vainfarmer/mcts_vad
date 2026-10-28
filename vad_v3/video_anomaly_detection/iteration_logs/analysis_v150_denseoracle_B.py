import json, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import load_ground_truth

REF = "/Users/zzl/PycharmProjects/vad/results/batch_eval_v144c_bue65ref"
LABEL = os.path.expanduser("~/Movies/ucf-crime/data")

gt = load_ground_truth(LABEL, split="test")  # {video_id: {segments:[(sf,ef)], ...}}

def iou(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0.0, e - s)
    uni = (a[1]-a[0]) + (b[1]-b[0]) - inter
    return inter/uni if uni > 0 else 0.0

def best_iou_over_gt(seg, gsegs):
    return max((iou(seg, g) for g in gsegs), default=0.0)

def build_runs(ts, fs, thr):
    """contiguous runs where fs>=thr; split when timestamp gap>2s."""
    ts = np.asarray(ts); fs = np.asarray(fs)
    runs = []
    i = 0; n = len(ts)
    active = fs >= thr
    j = 0
    while j < n:
        if not active[j]:
            j += 1; continue
        k = j
        while k+1 < n and active[k+1] and (ts[k+1]-ts[k]) <= 2.0:
            k += 1
        # run spans ts[j]..ts[k]; extend half-step
        st = ts[j]; en = ts[k] + 0.5
        peak = fs[j:k+1].max(); mean = fs[j:k+1].mean()
        runs.append((st, en, peak, mean))
        j = k+1
    return runs

anom = [(os.path.basename(f)[:-5], f) for f in sorted(glob.glob(REF+"/*.json"))]
rows = []
for vid, f in anom:
    d = json.load(open(f))
    if d.get("is_normal"): continue
    g = gt.get(vid)
    if not g or not g.get("segments"): continue
    gsegs = [(float(s)/30.0, float(e)/30.0) for s, e in g["segments"]]
    fs = d.get("frame_scores"); ts = d.get("timestamps")
    if not fs or not ts: continue
    fs = np.asarray(fs); ts = np.asarray(ts)
    mx = fs.max(); mn = fs.min()
    # sweep thresholds as fraction of dynamic range
    cand = []
    for frac in [0.3,0.4,0.5,0.6,0.7,0.8]:
        thr = mn + frac*(mx-mn)
        cand += build_runs(ts, fs, thr)
    if not cand:
        rows.append((vid, 0.0, 0.0, 0.0, len(cand))); continue
    # GT-aware: best possible IoU
    gt_aware = max(best_iou_over_gt((c[0],c[1]), gsegs) for c in cand)
    # verdict-independent selectors (no GT):
    # (a) highest-peak run
    peak_run = max(cand, key=lambda c: c[2])
    vi_peak = best_iou_over_gt((peak_run[0],peak_run[1]), gsegs)
    # (b) highest-mean run
    mean_run = max(cand, key=lambda c: c[3])
    vi_mean = best_iou_over_gt((mean_run[0],mean_run[1]), gsegs)
    rows.append((vid, gt_aware, vi_peak, vi_mean, len(cand)))

def rec(idx, thr=0.3):
    return sum(1 for r in rows if r[idx] >= thr)

n = len(rows)
print(f"anomaly videos with GT: {n}")
for T in [0.3, 0.5]:
    print(f"@IoU>={T}: GT-aware {rec(1,T)}/{n} | VI-peak {rec(2,T)}/{n} | VI-mean {rec(3,T)}/{n}")
# also report committed detection recall for reference
comm = []
for vid, f in anom:
    d = json.load(open(f))
    if d.get("is_normal"): continue
    g = gt.get(vid)
    if not g or not g.get("segments"): continue
    gsegs = [(float(s)/30.0,float(e)/30.0) for s,e in g["segments"]]
    dets = d.get("anomalies",[])
    bi = max((best_iou_over_gt((a["t_start"],a["t_end"]), gsegs) for a in dets), default=0.0)
    comm.append((vid,bi))
for T in [0.3,0.5]:
    print(f"COMMITTED dets @IoU>={T}: {sum(1 for _,b in comm if b>=T)}/{len(comm)}")
# gap: how many wrong-loc (committed<0.3) does VI-peak recover?
commd = dict(comm)
wrong = [r for r in rows if commd.get(r[0],0.0) < 0.3]
print(f"\nwrong-loc (committed<0.3): {len(wrong)}")
for T in [0.3,0.5]:
    print(f"  of those, VI-peak>={T}: {sum(1 for r in wrong if r[2]>=T)} | VI-mean>={T}: {sum(1 for r in wrong if r[3]>=T)} | GT-aware>={T}: {sum(1 for r in wrong if r[1]>=T)}")
