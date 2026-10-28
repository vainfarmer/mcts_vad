import sys, json, os, numpy as np
sys.path.insert(0,'/Users/zzl/PycharmProjects/vad')
sys.path.insert(0,'/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
import analysis_v140_livecurve_lib as LC
from video_anomaly_detection.evaluate import load_ground_truth
gt=load_ground_truth('/Users/zzl/Movies/ucf-crime/data', split='test'); FPS=30.0
rows=json.load(open('/tmp/zp124.json'))
byv={}
for r in rows: byv.setdefault(r['vid'],[]).append(r)
out=[]
for vid,dets in sorted(byv.items()):
    g=gt.get(vid)
    if not g or not g['segments']: continue
    cat=g['category']
    try:
        cur,ts=LC.live_curve(vid,cat)
    except Exception as e:
        out.append(dict(vid=vid,err=str(e)[:60])); continue
    cur=np.asarray(cur).ravel(); ts=np.asarray(ts).ravel()
    gts=[(s/FPS,e/FPS) for s,e in g['segments']]
    def maxin(a,b):
        m=(ts>=a)&(ts<=b)
        return float(cur[m].max()) if m.any() else float('nan')
    gmax=max(maxin(a,b) for a,b in gts)
    # rank of GT max within global curve
    grank=float((cur<gmax).mean())
    dmax=[maxin(d['ds'],d['de']) for d in dets]
    # top-6s window ranking: does GT contain a top-N 6s window?
    W=6.0; step=3.0; wins=[]
    t=0.0; T=float(ts[-1])
    while t+W<=T+1e-9:
        wins.append((t,t+W,maxin(t,t+W))); t+=step
    wins=[w for w in wins if not np.isnan(w[2])]
    wins.sort(key=lambda w:-w[2])
    def ov(a,b,c,d): return max(0.0,min(b,d)-max(a,c))>0
    gt_win_rank=None
    for i,w in enumerate(wins):
        if any(ov(w[0],w[1],a,b) for a,b in gts): gt_win_rank=i; break
    out.append(dict(vid=vid,cat=cat,ndet=len(dets),
        best_iou=max(d['iou'] for d in dets),
        gmax=round(gmax,4),grank=round(grank,3),
        dmax=[round(x,4) for x in dmax],
        dmax_best=round(max(dmax),4) if dmax else None,
        gt_win_rank=gt_win_rank,nwins=len(wins),
        gtdur=round(sum(b-a for a,b in gts),1),dur=round(dets[0]['dur'],1)))
json.dump(out,open('/tmp/wd124.json','w'))
print('done',len(out))
