"""Pre-validation gate: does MIL v6 (anti-collapse) localize better than v1
on the 17 zero-IoU wrong-location videos from the v97 full run?

Mirrors the pipeline's MIL input path: sample frames at fps=2, CLIP-encode,
mean-pool into 32 snippet features, run MILAnomalyScorer (v1 and v6).
Measures, per video:
  - argmax_hit: does the model's argmax snippet fall inside a GT segment?
  - gt_lift: mean(scores[GT snippets]) - mean(scores[non-GT snippets])
  - hm_iou: best IoU of contiguous above-half-max runs (snippet space) vs GT
  - det_support: mean(scores inside current v97 detections) - mean(outside)
Caches 32x768 snippet features per video for zero-cost iteration.
"""
import sys, json, pickle, time
from pathlib import Path
import numpy as np
import cv2
import torch

sys.path.insert(0, '/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
from evaluate import load_ground_truth, load_predictions, _temporal_iou

ROOT = Path('/Users/zzl/PycharmProjects/vad')
VID_ROOT = Path('/Users/zzl/Movies/ucf-crime/videos')
FEAT_CACHE = ROOT / 'video_anomaly_detection/iteration_logs/analysis_milv6_feats.pkl'
FPS_SAMPLE = 2

WRONG = ['Abuse028_x264','Burglary005_x264','Burglary079_x264','Fighting003_x264',
         'Fighting018_x264','Fighting033_x264','RoadAccidents001_x264','RoadAccidents019_x264',
         'Robbery050_x264','Robbery102_x264','Shooting018_x264','Shooting032_x264',
         'Shoplifting004_x264','Shoplifting037_x264','Stealing036_x264','Stealing062_x264',
         'Vandalism015_x264']
CONTROLS = ['Arson009_x264','Burglary001_x264','Shoplifting001_x264','Stealing001_x264',
            'Fighting001_x264','RoadAccidents002_x264','Robbery001_x264','Abuse001_x264']

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


def pool32(feats):
    n = feats.shape[0]
    seg = n // 32
    out = []
    for i in range(32):
        s = i * seg
        e = s + seg if i < 31 else n
        out.append(feats[s:e].mean(dim=0))
    return torch.stack(out)


def main():
    import torch
    from anomaly_expert import VideoAnomalyExpert
    from mil_model import MILAnomalyScorer

    gt_all = load_ground_truth('/Users/zzl/Movies/ucf-crime/data', split='test')
    pred = {k: v for k, v in load_predictions(str(ROOT / 'results/batch_eval_v97_full')).items()
            if k in gt_all}

    feats_cache = pickle.load(open(FEAT_CACHE, 'rb')) if FEAT_CACHE.exists() else {}
    CATS = ["Abuse","Arrest","Arson","Assault","Burglary","Explosion","Fighting",
            "RoadAccidents","Robbery","Shooting","Shoplifting","Stealing","Vandalism"]
    targets = [v for v in WRONG + CONTROLS if v in pred]
    todo = [v for v in targets if v not in feats_cache]
    print(f'{len(todo)} videos to encode ({len(feats_cache)} cached)')
    if todo:
        expert = VideoAnomalyExpert(CATS, device='mps')
        for i, vid in enumerate(todo):
            found = list(VID_ROOT.glob(f'*/{vid}.mp4'))
            if not found:
                print('  MISSING', vid); continue
            frames, ts, dur = sample_video(found[0])
            t0 = time.time()
            feats = expert.precompute_video_features(np.stack(frames))
            feats_cache[vid] = {'snip': pool32(feats).cpu().numpy(), 'ts': ts, 'dur': dur}
            pickle.dump(feats_cache, open(FEAT_CACHE, 'wb'))
            print(f'  {i+1}/{len(todo)} {vid}: {len(frames)} frames, {time.time()-t0:.1f}s')

    def load_mil(path):
        m = MILAnomalyScorer(feature_dim=768, hidden_dim=512, k=3, dropout=0.6)
        m.load_state_dict(torch.load(path, map_location='cpu'))
        m.eval()
        return m

    mil1 = load_mil(ROOT / 'video_anomaly_detection/model_mil.pth')
    mil6 = load_mil(ROOT / 'video_anomaly_detection/model_mil_v6.pth')

    def analyze(model, vid):
        e = feats_cache[vid]
        x = torch.tensor(e['snip'], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            s = model(x).squeeze(0).numpy()
        dur = float(e['dur'])
        snip_dur = dur / 32.0
        segs = gt_all[vid]['segments']  # frame-based @30fps
        gt_t = [(a / 30.0, b / 30.0) for a, b in segs]
        gt_snip = set()
        for (a, b) in gt_t:
            for si in range(32):
                cs, ce = si * snip_dur, (si + 1) * snip_dur
                if ce > a and cs < b:
                    gt_snip.add(si)
        argmax_hit = int(np.argmax(s)) in gt_snip if gt_snip else None
        gt_mask = np.array([i in gt_snip for i in range(32)])
        gt_lift = float(s[gt_mask].mean() - s[~gt_mask].mean()) if gt_mask.any() and (~gt_mask).any() else 0.0
        # half-max contiguous runs in snippet space
        hm = (s.max() + np.percentile(s, 20)) / 2.0
        above = s >= hm
        runs = []
        st = None
        for i, a in enumerate(above):
            if a and st is None:
                st = i
            elif not a and st is not None:
                runs.append((st, i)); st = None
        if st is not None:
            runs.append((st, 32))
        best_iou = 0.0
        for (a, b) in runs:
            win = {'t_start': a * snip_dur, 't_end': b * snip_dur}
            for g in segs:
                best_iou = max(best_iou, _temporal_iou(win, g, 30))
        # support for current v97 detections
        det_in, det_out = [], []
        for d in pred[vid]['anomalies']:
            m = np.array([(i * snip_dur < d['t_end']) and ((i + 1) * snip_dur > d['t_start']) for i in range(32)])
            if m.any():
                det_in.append(float(s[m].mean()))
        det_support = (np.mean(det_in) - s.mean()) if det_in else None
        return dict(argmax_hit=argmax_hit, gt_lift=gt_lift, hm_iou=best_iou,
                    det_support=det_support, rng=float(s.max() - s.min()))

    rows = []
    for vid in targets:
        if vid not in feats_cache:
            continue
        r6 = analyze(mil6, vid)
        r1 = analyze(mil1, vid)
        rows.append(dict(vid=vid, wrong=vid in WRONG, cat=gt_all[vid]['category'],
                         **{f'v6_{k}': v for k, v in r6.items()},
                         **{f'v1_{k}': v for k, v in r1.items()}))

    for tag, subset in [('WRONG-LOCATION', [r for r in rows if r['wrong']]),
                        ('CONTROL-TP', [r for r in rows if not r['wrong']])]:
        n = len(subset)
        if not n:
            continue
        h6 = np.mean([r['v6_argmax_hit'] for r in subset if r['v6_argmax_hit'] is not None])
        h1 = np.mean([r['v1_argmax_hit'] for r in subset if r['v1_argmax_hit'] is not None])
        print(f"\n== {tag} (n={n}) ==")
        print(f"  argmax-in-GT rate:  v6={h6:.2f}  v1={h1:.2f}")
        print(f"  GT lift (in-out):   v6={np.mean([r['v6_gt_lift'] for r in subset]):+.3f}  v1={np.mean([r['v1_gt_lift'] for r in subset]):+.3f}")
        print(f"  half-max best IoU:  v6={np.mean([r['v6_hm_iou'] for r in subset]):.3f}  v1={np.mean([r['v1_hm_iou'] for r in subset]):.3f}")
        print(f"  score range:        v6={np.mean([r['v6_rng'] for r in subset]):.3f}  v1={np.mean([r['v1_rng'] for r in subset]):.3f}")
        ds = [r['v6_det_support'] for r in subset if r['v6_det_support'] is not None]
        print(f"  v6 det-support:     mean={np.mean(ds):+.3f}  frac-negative={np.mean([x < 0 for x in ds]):.2f} (n={len(ds)})")

    print('\nper-video (wrong-location set):')
    for r in rows:
        if r['wrong']:
            print(f"  {r['vid']:24s} {r['cat']:14s} v6_hit={r['v6_argmax_hit']} v6_lift={r['v6_gt_lift']:+.3f} v6_hmIoU={r['v6_hm_iou']:.3f} v1_hit={r['v1_argmax_hit']} v1_hmIoU={r['v1_hm_iou']:.3f}")
    json.dump(rows, open(ROOT / 'video_anomaly_detection/iteration_logs/analysis_milv6_rows.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
