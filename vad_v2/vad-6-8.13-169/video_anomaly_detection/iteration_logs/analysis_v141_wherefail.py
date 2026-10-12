"""v141 WHERE-FAILURE diagnostic (offline, zero API). Splits the 50/99 zero-overlap
detections into candidate-generation failures vs boundary failures, and bounds the
mAP headroom of a COUNT-PRESERVING relocation mechanism."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import load_ground_truth, load_predictions, compute_temporal_map, _temporal_iou
LABEL_PATH = str(Path('/Users/zzl/Movies/ucf-crime') / 'data'); FPS = 30

def seg_sec(seg):
    # gt segments are frame pairs
    return (seg[0]/FPS, seg[1]/FPS)

def run(rd):
    gt = load_ground_truth(LABEL_PATH, split='test')
    pred = load_predictions(str(rd))
    gt_eval = {k: v for k, v in gt.items() if k in pred}
    base = compute_temporal_map(gt_eval, pred, fps=FPS)
    ndet = sum(len(p['anomalies']) for p in pred.values())
    ngt = sum(len(v['segments']) for v in gt_eval.values())
    print(f'\n=== {rd.name}: {len(pred)} vids, {ndet} det, {ngt} gt, mAP={base["mAP"]:.5f}')
    rows = []
    for vid, e in pred.items():
        if vid not in gt_eval: continue
        gts = [seg_sec(s) for s in gt_eval[vid]['segments']]
        if not gts: continue
        dur = e.get('duration') or 0.0
        dets = e['anomalies']
        for i, a in enumerate(dets):
            ds, de = a['t_start'], a['t_end']
            ious = [_temporal_iou(a, s, FPS) for s in gt_eval[vid]['segments']]
            bi = max(ious) if ious else 0.0
            # gap to nearest GT (0 if overlapping)
            gaps = [max(0.0, max(gs - de, ds - ge)) for gs, ge in gts]
            g = min(gaps)
            j = int(np.argmin(gaps))
            gs, ge = gts[j]
            rows.append(dict(vid=vid, i=i, conf=a['confidence'], ds=ds, de=de, iou=bi,
                             gap=g, dur=dur, gs=gs, ge=ge, gdur=ge-gs, ddur=de-ds,
                             ngt=len(gts), ndet=len(dets)))
    zr = [r for r in rows if r['iou'] <= 0.0]
    print(f'  zero-overlap: {len(zr)}/{len(rows)} ({100*len(zr)/len(rows):.0f}%)')
    gaps = np.array([r['gap'] for r in zr])
    for thr in (0, 2, 5, 10, 20, 30, 60):
        print(f'   gap<={thr:3d}s: {int((gaps<=thr).sum()):3d}  ({100*(gaps<=thr).mean():.0f}%)')
    print(f'   gap median={np.median(gaps):.1f}s mean={gaps.mean():.1f}s p90={np.percentile(gaps,90):.1f}s')
    # normalize gap by video duration
    ng = np.array([r['gap']/max(1e-6, r['dur']) for r in zr])
    print(f'   gap/videodur median={np.median(ng):.3f} mean={ng.mean():.3f}')
    # per-video: any hit?
    vids = sorted(set(r['vid'] for r in rows))
    novhit = [v for v in vids if max(r['iou'] for r in rows if r['vid']==v) <= 0.0]
    print(f'  videos with ZERO hitting detection: {len(novhit)}/{len(vids)}')
    # detection union coverage of video
    covs=[]
    for v in vids:
        rs=[r for r in rows if r['vid']==v]
        iv=sorted((r['ds'],r['de']) for r in rs); m=[]
        for s,eo in iv:
            if m and s<=m[-1][1]: m[-1]=(m[-1][0],max(m[-1][1],eo))
            else: m.append((s,eo))
        c=sum(b-a for a,b in m); covs.append(c/max(1e-6,rs[0]['dur']))
    covs=np.array(covs)
    print(f'  det union coverage of video: median={np.median(covs):.3f} mean={covs.mean():.3f}')
    # GT recall side: per GT segment, best IoU over dets in that video
    grecs=[]
    for v in vids:
        rs=[r for r in rows if r['vid']==v]
        for k,s in enumerate(gt_eval[v]['segments']):
            best=max(_temporal_iou({'t_start':r['ds'],'t_end':r['de']}, s, FPS) for r in rs)
            grecs.append(best)
    grecs=np.array(grecs)
    print(f'  GT segs (in det-videos) with best IoU=0: {int((grecs<=0).sum())}/{len(grecs)}; >=0.3: {int((grecs>=0.3).sum())}')
    return rows, gt_eval, pred, base

if __name__=='__main__':
    dirs=[Path(x) for x in (sys.argv[1:] or [str(ROOT/'results/batch_eval_v124_full')])]
    for d in dirs: run(d)
