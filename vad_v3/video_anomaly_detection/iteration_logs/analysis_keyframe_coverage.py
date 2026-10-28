"""Pre-validation: does the current VLM keyframe selection MISS the anomaly
cue window that a motion-peak selection would catch — on the 17 zero-IoU
wrong-location videos from the v97 full run?

Faithfully replicates the pipeline's Stage 1 (fps=2, motion filter thr=1.0,
min_motion_interval=2.0, 6s segments, overlap=0.25 -> stride 4.5s) and the
FCCS keyframe path: _sample_keyframes -> SemanticFrameSampler.select_keyframes
(uniform pre-sample to K=ceil(6*3)=18 -> CLIP semantic dedup tau=0.90)
-> keyframes[:2] shown to the VLM.

For every GT-overlapping segment, measures:
  - cur_hit: any of the 2 currently-shown frames falls inside GT(+/-0.5s)
  - peak_hit: any of the top-2 motion-energy frames falls inside GT(+/-0.5s)
Caches per-frame CLIP features so the analysis is re-runnable for free.
"""
import sys, json, pickle, time
from pathlib import Path
import numpy as np
import cv2
import torch

sys.path.insert(0, '/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
from evaluate import load_ground_truth

ROOT = Path('/Users/zzl/PycharmProjects/vad')
VID_ROOT = Path('/Users/zzl/Movies/ucf-crime/videos')
FEAT_CACHE = ROOT / 'video_anomaly_detection/iteration_logs/analysis_keyframe_feats.pkl'

WRONG = ['Abuse028_x264','Burglary005_x264','Burglary079_x264','Fighting003_x264',
         'Fighting018_x264','Fighting033_x264','RoadAccidents001_x264','RoadAccidents019_x264',
         'Robbery050_x264','Robbery102_x264','Shooting018_x264','Shooting032_x264',
         'Shoplifting004_x264','Shoplifting037_x264','Stealing036_x264','Stealing062_x264',
         'Vandalism015_x264']

FPS = 2
SEG_DUR = 6.0
OVERLAP = 0.25
MOTION_THR = 1.0
MIN_MOTION_INTERVAL = 2.0


def preprocess(video_path):
    """Exact replica of VideoPreprocessor.process (fps=2, motion filter, segments)."""
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / video_fps
    interval = max(1, int(video_fps / FPS))
    frames, timestamps = [], []
    idx = 0
    while True:
        if idx % interval == 0:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamps.append(idx / video_fps)
        else:
            if not cap.grab():
                break
        idx += 1
    cap.release()
    frames = np.array(frames)
    timestamps = np.array(timestamps)
    # motion filter (pipeline formula)
    small = np.stack([cv2.resize(f, (64, 64)) for f in frames])
    motion = np.zeros(len(frames))
    diffs = np.abs(small[1:].astype(np.float32) - small[:-1].astype(np.float32))
    motion[1:] = diffs.mean(axis=(1, 2, 3))
    keep = np.zeros(len(frames), dtype=bool)
    keep[0] = True
    last_kept = timestamps[0]
    for i in range(1, len(frames)):
        if motion[i] > MOTION_THR:
            keep[i] = True; last_kept = timestamps[i]
        elif timestamps[i] - last_kept >= MIN_MOTION_INTERVAL:
            keep[i] = True; last_kept = timestamps[i]
    frames = frames[keep]; timestamps = timestamps[keep]; motion = motion[keep]
    # segments
    stride = SEG_DUR * (1 - OVERLAP)
    segments = []
    seg_start = 0.0
    import bisect as _b
    while seg_start < duration:
        seg_end = min(seg_start + SEG_DUR, duration)
        if seg_end - seg_start < SEG_DUR * 0.3 and segments:
            break
        lo = _b.bisect_left(timestamps, seg_start)
        hi = _b.bisect_left(timestamps, seg_end)
        if lo < hi:
            segments.append({'t_start': float(seg_start), 't_end': float(seg_end),
                             'frame_indices': list(range(lo, hi))})
        seg_start += stride
    return frames, timestamps, motion, segments, duration


def main():
    from anomaly_expert import VideoAnomalyExpert
    from token_compressor import SemanticFrameSampler

    gt_all = load_ground_truth('/Users/zzl/Movies/ucf-crime/data', split='test')
    cache = pickle.load(open(FEAT_CACHE, 'rb')) if FEAT_CACHE.exists() else {}
    CATS = ["Abuse","Arrest","Arson","Assault","Burglary","Explosion","Fighting",
            "RoadAccidents","Robbery","Shooting","Shoplifting","Stealing","Vandalism"]

    todo = [v for v in WRONG if v not in cache]
    print(f'{len(todo)} videos to process ({len(cache)} cached)')
    if todo:
        expert = VideoAnomalyExpert(CATS, device='mps')
        for i, vid in enumerate(todo):
            found = list(VID_ROOT.glob(f'*/{vid}.mp4'))
            if not found:
                print('  MISSING', vid); continue
            t0 = time.time()
            frames, ts, motion, segments, dur = preprocess(found[0])
            feats = expert.precompute_video_features(frames)
            cache[vid] = {'feats': feats.cpu().numpy(), 'ts': ts, 'motion': motion,
                          'segments': segments, 'dur': dur}
            pickle.dump(cache, open(FEAT_CACHE, 'wb'))
            print(f'  {i+1}/{len(todo)} {vid}: {len(frames)} kept frames, {time.time()-t0:.1f}s')

    rows = []
    for vid in WRONG:
        if vid not in cache:
            continue
        e = cache[vid]
        ts = e['ts']; motion = e['motion']; feats = e['feats']
        # rebuild a torch feature tensor for the sampler's lookup path
        import torch as _t
        feats_t = _t.tensor(feats)
        gt_t = [(a / 30.0, b / 30.0) for a, b in gt_all[vid]['segments']]
        for seg in e['segments']:
            fi = seg['frame_indices']
            if not fi:
                continue
            # does this segment overlap GT?
            seg_gt = [g for g in gt_t if seg['t_end'] > g[0] and seg['t_start'] < g[1]]
            if not seg_gt:
                continue
            # --- current FCCS selection, exact SemanticFrameSampler logic ---
            n = len(fi)
            target = min(18, n)  # K = clamp(ceil(6s * 3.0 fps), 3, 200) = 18
            if n <= target:
                # shortcut: returns ALL frames -> FCCS shows the FIRST 2
                shown_local = list(range(min(2, n)))
                shown_times = [float(ts[fi[i]]) for i in shown_local]
            else:
                # uniform presample K -> CLIP semantic dedup (tau=0.90) -> first 2
                pre = np.linspace(0, n - 1, target, dtype=int)
                sub = [fi[p] for p in pre]
                f = feats_t[sub]
                sim = (f @ f.T).numpy()
                assigned = [False] * len(sub)
                clusters = []
                for a in range(len(sub)):
                    if assigned[a]:
                        continue
                    c = [a]; assigned[a] = True
                    for b2 in range(a + 1, len(sub)):
                        if not assigned[b2] and sim[a, b2] > 0.90:
                            c.append(b2); assigned[b2] = True
                    clusters.append(c)
                reps = []
                for c in clusters:
                    if len(c) == 1:
                        reps.append(c[0])
                    else:
                        ss = sim[np.ix_(c, c)]
                        reps.append(c[int(np.argmin(ss.mean(axis=1)))])
                if len(reps) < 3 and len(sub) >= 3:
                    remaining = [x for x in range(len(sub)) if x not in reps]
                    while len(reps) < 3 and remaining:
                        reps.append(remaining.pop(0))
                reps.sort()
                shown_times = [float(ts[sub[i]]) for i in reps[:2]]
            # --- motion-peak alternative: top-2 motion frames (greedy, +-2 exclusion) ---
            m_seg = [(motion[gi], gi) for gi in fi]
            order = sorted(m_seg, key=lambda x: -x[0])
            picks = []
            for mv, gi in order:
                if all(abs(gi - p) > 2 for p in picks):
                    picks.append(gi)
                if len(picks) == 2:
                    break
            peak_times = [float(ts[gi]) for gi in picks]
            def hit(times):
                for t in times:
                    for (a, b) in seg_gt:
                        if a - 0.5 <= t <= b + 0.5:
                            return True
                return False
            rows.append({'vid': vid, 'seg': (seg['t_start'], seg['t_end']),
                         'gt': seg_gt, 'cur_hit': hit(shown_times),
                         'peak_hit': hit(peak_times),
                         'shown_times': shown_times, 'peak_times': peak_times,
                         'n_frames': n})

    cur = np.mean([r['cur_hit'] for r in rows])
    pk = np.mean([r['peak_hit'] for r in rows])
    print(f'\nGT-overlapping segments analyzed: {len(rows)} across {len(set(r["vid"] for r in rows))} videos')
    print(f'  current-selection cue coverage: {cur:.2f}')
    print(f'  motion-peak   cue coverage: {pk:.2f}')
    both_miss = [r for r in rows if not r['cur_hit'] and not r['peak_hit']]
    only_peak = [r for r in rows if not r['cur_hit'] and r['peak_hit']]
    print(f'  segments where ONLY motion-peak hits: {len(only_peak)} ({len(only_peak)/max(1,len(rows)):.2f})')
    print(f'  segments where BOTH miss: {len(both_miss)} ({len(both_miss)/max(1,len(rows)):.2f})')
    per_vid = {}
    for r in rows:
        per_vid.setdefault(r['vid'], []).append(r)
    print('\nper-video (cur -> peak):')
    for vid, rs in per_vid.items():
        print(f"  {vid:24s} {np.mean([r['cur_hit'] for r in rs]):.2f} -> {np.mean([r['peak_hit'] for r in rs]):.2f}  (n={len(rs)})")
    json.dump(rows, open(ROOT / 'video_anomaly_detection/iteration_logs/analysis_keyframe_rows.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
