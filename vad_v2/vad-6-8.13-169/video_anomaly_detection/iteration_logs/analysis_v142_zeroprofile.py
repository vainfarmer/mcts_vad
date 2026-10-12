import sys, json, glob, os
sys.path.insert(0,'/Users/zzl/PycharmProjects/vad')
sys.path.insert(0,'/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
from video_anomaly_detection.evaluate import load_ground_truth
import numpy as np

DS=None
DS='/Users/zzl/Movies/ucf-crime/data'
gt=load_ground_truth(DS, split='test')
FPS=30.0

def iou(a,b):
    i=max(0.0,min(a[1],b[1])-max(a[0],b[0])); u=(a[1]-a[0])+(b[1]-b[0])-i
    return i/u if u>0 else 0.0

def gap(a,b):  # temporal gap (0 if overlap)
    return max(0.0, max(a[0],b[0])-min(a[1],b[1]))

rows=[]
for run in sys.argv[1:]:
    for f in sorted(glob.glob(run+'/*.json')):
        d=json.load(open(f)); vid=os.path.basename(f)[:-5]
        g=gt.get(vid)
        if not g or not g['segments']: continue
        gts=[(s/FPS,e/FPS) for s,e in g['segments']]
        dur=d['duration']
        for a in d['anomalies']:
            det=(a['t_start'],a['t_end'])
            best=max(gts,key=lambda gg: iou(det,gg))
            bi=iou(det,best)
            mg=min(gap(det,gg) for gg in gts)
            gtcov=sum(e-s for s,e in gts)
            rows.append(dict(run=os.path.basename(run),vid=vid,dur=dur,ds=det[0],de=det[1],
                conf=a['confidence'],iou=bi,gap=mg,detlen=det[1]-det[0],
                gtlen=best[1]-best[0],gt_frac=gtcov/max(dur,1e-6),ngt=len(gts),
                gt0=best[0],gt1=best[1]))
print(json.dumps(rows))
