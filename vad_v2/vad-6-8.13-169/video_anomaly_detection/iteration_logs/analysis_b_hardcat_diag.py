import json, glob, sys
import numpy as np
import pandas as pd
from pathlib import Path

RES = Path('/Users/zzl/PycharmProjects/vad/results/batch_eval_full290_bue')
PARQ = '/Users/zzl/Movies/ucf-crime/data/test-00000-of-00001.parquet'
FPS = 30
HARD = ['Shooting', 'RoadAccidents', 'Shoplifting', 'Explosion']

def vid_id(p): return Path(str(p)).stem

df = pd.read_parquet(PARQ)
gt = {}
for _, r in df.iterrows():
    segs = []
    if r['annotation_start1'] >= 0: segs.append((r['annotation_start1']/FPS, r['annotation_end1']/FPS))
    if r['annotation_start2'] >= 0: segs.append((r['annotation_start2']/FPS, r['annotation_end2']/FPS))
    gt[vid_id(r['video_path'])] = {'cat': r['category'], 'segs': segs}

def iou(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0, e-s); uni = (a[1]-a[0])+(b[1]-b[0])-inter
    return inter/uni if uni > 0 else 0

def cover(a, b):  # fraction of GT b covered by det a
    s = max(a[0], b[0]); e = min(a[1], b[1])
    return max(0, e-s)/(b[1]-b[0]) if (b[1]-b[0])>0 else 0

for cat in HARD:
    files = sorted(glob.glob(str(RES/f'{cat}*.json')))
    n = 0; gt_durs=[]; det_durs=[]
    hit3=hit5=0
    miss_candidate=0; miss_boundary=0; miss_ranking=0
    for f in files:
        vfn = Path(f).stem
        if vfn not in gt or gt[vfn]['cat'] != cat: continue
        d = json.load(open(f))
        if d.get('is_normal'): continue
        segs = gt[vfn]['segs']
        if not segs: continue
        dets = d.get('anomalies', [])
        n += 1
        for s in segs: gt_durs.append(s[1]-s[0])
        # sort dets by confidence
        dets_s = sorted(dets, key=lambda x: -x.get('confidence',0))
        for dd in dets_s: det_durs.append(dd['t_end']-dd['t_start'])
        # best IoU over any det vs any GT seg
        best_iou = 0; best_cover=0; best_iou_topk=0
        for gseg in segs:
            for i, dd in enumerate(dets_s):
                dwin = (dd['t_start'], dd['t_end'])
                io = iou(dwin, gseg); cv = cover(dwin, gseg)
                best_iou = max(best_iou, io)
                best_cover = max(best_cover, cv)
        if best_iou >= 0.3: hit3 += 1
        if best_iou >= 0.5: hit5 += 1
        # classify miss at IoU 0.5
        if best_iou < 0.5:
            if best_cover >= 0.5 and best_iou < 0.5:
                miss_boundary += 1   # overlaps GT well but extent wrong (too wide/narrow)
            elif best_cover > 0.1:
                miss_boundary += 1
            else:
                miss_candidate += 1  # essentially no overlap: wrong place
    print(f'\n=== {cat}: {n} videos ===')
    print(f'  hit@0.3={hit3}/{n}  hit@0.5={hit5}/{n}')
    print(f'  GT dur: median={np.median(gt_durs):.1f}s p75={np.percentile(gt_durs,75):.1f}s max={np.max(gt_durs):.1f}s')
    print(f'  DET dur: median={np.median(det_durs):.1f}s p75={np.percentile(det_durs,75):.1f}s max={np.max(det_durs):.1f}s n_det={len(det_durs)}')
    print(f'  MISS@0.5 split: candidate(no-overlap)={miss_candidate}  boundary(overlap-wrong-extent)={miss_boundary}')
