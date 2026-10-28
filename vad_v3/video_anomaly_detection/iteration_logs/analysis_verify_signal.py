"""Analysis: can a zero-cost CLIP-curve signal separate wrong-location FPs from TPs?

Recomputes CLIP features for the 65 v94 eval videos (fps=2, no motion filter),
then measures separation (AUC) between TP detections (best-IoU>=0.3) and
wrong-location FPs (best-IoU<0.1) for a battery of read-only signals.
Caches per-video curves to verify_curves.pkl for fast iteration.
"""
import sys, json, pickle, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, '/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
from evaluate import load_ground_truth, load_predictions, _temporal_iou, compute_temporal_map

ROOT = Path('/Users/zzl/PycharmProjects/vad')
VID_ROOT = Path('/Users/zzl/Movies/ucf-crime/videos')
CACHE = ROOT / 'video_anomaly_detection/iteration_logs/analysis_verify_curves.pkl'

FPS_SAMPLE = 2

def sample_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / vfps
    interval = max(1, int(vfps / FPS_SAMPLE))
    frames, ts = [], []
    idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx % interval == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts.append(idx / vfps)
        idx += 1
    cap.release()
    return frames, np.array(ts), duration


def main():
    gt_all = load_ground_truth('/Users/zzl/Movies/ucf-crime/data', split='test')
    pred = {k: v for k, v in load_predictions(str(ROOT / 'results/batch_eval_v94')).items()
            if k in gt_all}
    gt = {k: v for k, v in gt_all.items() if k in pred}

    if CACHE.exists():
        curves = pickle.load(open(CACHE, 'rb'))
    else:
        curves = {}

    from anomaly_expert import VideoAnomalyExpert
    import torch
    cats = list(gt_all[k]['category'] for k in list(gt_all)[:0])  # placeholder
    CATS = ["Abuse","Arrest","Arson","Assault","Burglary","Explosion","Fighting",
            "RoadAccidents","Robbery","Shooting","Shoplifting","Stealing","Vandalism"]
    expert = VideoAnomalyExpert(CATS, device='mps')

    import os
    max_per_run = int(os.environ.get('MAX_PER_RUN', '8'))
    todo = [vid for vid in pred if vid not in curves][:max_per_run]
    print(f'{len(todo)} videos to encode this run')
    for i, vid in enumerate(todo):
        cat = gt[vid]['category']
        vpath = VID_ROOT / cat / f'{vid}.mp4'
        if not vpath.exists():
            # search all category dirs
            found = list(VID_ROOT.glob(f'*/{vid}.mp4'))
            vpath = found[0] if found else None
        if vpath is None:
            print('  MISSING', vid); continue
        frames, ts, dur = sample_video(vpath)
        feats = expert.precompute_video_features(np.stack(frames))
        with torch.no_grad():
            ls = expert.model.logit_scale.exp().detach().clamp(max=100.0)
            sim = feats @ expert.text_features.T * ls
            probs = torch.softmax(sim, dim=-1).cpu().numpy()
        rel = {}
        for ci, c in enumerate(CATS):
            rel[c] = probs[:, ci] / (probs[:, ci] + probs[:, -1] + 1e-8)
        top3 = np.sort(probs[:, :-1], axis=1)[:, -3:].sum(axis=1)
        agen = top3 / (top3 + probs[:, -1] + 1e-8)
        curves[vid] = {'ts': ts, 'dur': dur, 'rel': rel, 'agen': agen}
        pickle.dump(curves, open(CACHE, 'wb'))
        print(f'  {i+1}/{len(todo)} cached {vid}')
    pickle.dump(curves, open(CACHE, 'wb'))
    print('encoding done,', len(curves), 'videos total')
    if len([v for v in pred if v not in curves]) > 0:
        print('MORE VIDEOS REMAIN - run again')
        return

    # ---- signal extraction ----
    rows = []
    for vid, entry in pred.items():
        if vid not in curves:
            continue
        segs = gt[vid]['segments']
        gcat = gt[vid]['category']
        ts = curves[vid]['ts']
        for a in entry['anomalies']:
            bi = max([_temporal_iou(a, s, 30) for s in segs] + [0.0])
            pcat = a.get('anomaly_type', gcat)
            if pcat not in curves[vid]['rel']:
                pcat = gcat
            curve = curves[vid]['rel'].get(pcat, curves[vid]['agen'])
            agen = curves[vid]['agen']
            lo = np.searchsorted(ts, a['t_start'])
            hi = min(len(ts), np.searchsorted(ts, a['t_end']))
            if hi <= lo:
                hi = lo + 1
            in_mean = float(curve[lo:hi].mean())
            out = np.concatenate([curve[:lo], curve[hi:]])
            out_mean = float(out.mean()) if len(out) else 0.0
            # window rank: percentile of span mean among all same-length windows
            wlen = hi - lo
            if len(curve) > wlen:
                cs = np.cumsum(np.concatenate([[0.0], curve]))
                wmeans = (cs[wlen:] - cs[:-wlen]) / wlen
                wrank = float((wmeans < in_mean).mean())
            else:
                wrank = 1.0
            contains_peak = bool(lo <= int(np.argmax(curve)) <= hi)
            agen_in = float(agen[lo:hi].mean())
            # onset/offset (v96-style)
            w = max(2, int(1.0 * FPS_SAMPLE))
            ps = curve[max(0, lo - w):lo]; fs = curve[lo:lo + w]
            pe = curve[max(0, hi - w):hi]; fe = curve[hi:hi + w]
            on_off = 0.5
            if len(ps) and len(fs) and len(pe) and len(fe):
                on_off = float(np.clip(((fs.mean() - ps.mean()) + (pe.mean() - fe.mean())) / 2 + 0.5, 0, 1))
            rows.append({
                'vid': vid, 'iou': bi, 'conf': a['confidence'],
                'in_mean': in_mean, 'oic': in_mean - out_mean,
                'wrank': wrank, 'contains_peak': contains_peak,
                'agen_in': agen_in, 'on_off': on_off,
                'is_tp': bi >= 0.3, 'is_fp': bi < 0.1,
            })

    tp = [r for r in rows if r['is_tp']]
    fp = [r for r in rows if r['is_fp']]
    print(f'TP={len(tp)} FP={len(fp)}')

    def sep_auc(key):
        # P(signal_TP > signal_FP)
        vals_t = [r[key] for r in tp]
        vals_f = [r[key] for r in fp]
        cnt = sum(1 for t in vals_t for f in vals_f if t > f) + 0.5 * sum(1 for t in vals_t for f in vals_f if t == f)
        return cnt / (len(vals_t) * len(vals_f))

    for key in ['conf', 'in_mean', 'oic', 'wrank', 'agen_in', 'on_off']:
        print(f'  sep-AUC {key:14s}: {sep_auc(key):.3f}  (TP mean={np.mean([r[key] for r in tp]):.3f} FP mean={np.mean([r[key] for r in fp]):.3f})')
    print(f"  contains_peak: TP {np.mean([r['contains_peak'] for r in tp]):.2f} vs FP {np.mean([r['contains_peak'] for r in fp]):.2f}")

    json.dump(rows, open(ROOT / 'video_anomaly_detection/iteration_logs/analysis_verify_rows.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
