import json, glob, os
import pandas as pd
import numpy as np

df = pd.read_parquet('/Users/zzl/Movies/ucf-crime/data/test-00000-of-00001.parquet')
# build gt: video stem -> list of (s,e)
def stem(p): return os.path.splitext(os.path.basename(str(p)))[0]
gt={}
for _,r in df.iterrows():
    segs=[]
    for a,b in [('annotation_start1','annotation_end1'),('annotation_start2','annotation_end2')]:
        if r[a]>=0 and r[b]>=0: segs.append((float(r[a])/30.0,float(r[b])/30.0))
    gt[stem(r['video_path'])]=segs

def iou(a,b):
    s=max(a[0],b[0]); e=min(a[1],b[1]); inter=max(0,e-s)
    uni=(a[1]-a[0])+(b[1]-b[0])-inter
    return inter/uni if uni>0 else 0

files=glob.glob('/Users/zzl/PycharmProjects/vad/results/batch_eval_v124_full/*.json')
ndet=0; zero=0; adj_dist=[]; elsewhere=0; adjacent=0; det_durs=[]; gt_durs=[]
zero_gtpos=[]  # for zero-overlap: min distance from det to any gt seg, normalized by video dur
per_video_recall_gain=0
for f in files:
    d=json.load(open(f));
    if "video_path" not in d: continue
    v=stem(d['video_path']); dur=d['duration']
    if v not in gt or not gt[v]: continue
    segs=gt[v]
    for a in d['anomalies']:
        ndet+=1
        det=(a['t_start'],a['t_end'])
        best=max((iou(det,g) for g in segs),default=0)
        det_durs.append(det[1]-det[0])
        if best==0:
            zero+=1
            # distance from det center to nearest gt seg (gap)
            dc=(det[0]+det[1])/2
            gaps=[]
            for g in segs:
                if det[1]<g[0]: gaps.append(g[0]-det[1])
                elif det[0]>g[1]: gaps.append(det[0]-g[1])
                else: gaps.append(0)
            mg=min(gaps)
            zero_gtpos.append(mg/dur if dur>0 else 0)
            if mg < 5.0: adjacent+=1
            else: elsewhere+=1
print(f"videos={len(files)} ndet={ndet} zero_overlap={zero} ({100*zero/ndet:.0f}%)")
print(f"zero-overlap breakdown: adjacent(<5s gap)={adjacent}  elsewhere(>=5s)={elsewhere}")
za=np.array(zero_gtpos)
print(f"zero-overlap gap/dur: median={np.median(za):.3f} mean={np.mean(za):.3f}  >0.2 of video away: {np.mean(za>0.2)*100:.0f}%")
print(f"det dur median={np.median(det_durs):.1f}s")
