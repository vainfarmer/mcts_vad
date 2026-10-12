"""A+B loop: measure candidate SHARPER localizers on the cached 32x768 snippet
features for the 17 zero-IoU wrong-loc videos. Compare against v1 MIL.
Signals (all verdict-independent, pre-calibratable, computed from features/test):
  MAG   = L2 norm of snippet feature (RTFM-style)
  NOV   = 1 - cos(snip[i], snip[i-1]) temporal novelty (frame-diff)
  CEN   = ||snip[i] - video_mean|| centroid distance (outlierness)
  NORMD = distance from a NORMAL prototype (mean of Normal-video snippet feats)
For each we report argmax_hit (argmax snippet inside GT), gt_lift, and hm_iou
(best above-half-max contiguous run IoU vs GT), and the per-video range (sharpness).
"""
import sys, pickle
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0,'/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
from evaluate import load_ground_truth

ROOT=Path('/Users/zzl/PycharmProjects/vad')
VID_ROOT=Path('/Users/zzl/Movies/ucf-crime/videos')
c=pickle.load(open(ROOT/'video_anomaly_detection/iteration_logs/analysis_milv6_feats.pkl','rb'))
gt=load_ground_truth('/Users/zzl/Movies/ucf-crime/data',split='test')

WRONG=['Abuse028_x264','Burglary005_x264','Burglary079_x264','Fighting003_x264',
 'Fighting018_x264','Fighting033_x264','RoadAccidents001_x264','RoadAccidents019_x264',
 'Robbery050_x264','Robbery102_x264','Shooting018_x264','Shooting032_x264',
 'Shoplifting004_x264','Shoplifting037_x264','Stealing036_x264','Stealing062_x264',
 'Vandalism015_x264']

def vfps(name):
    for cat_guess in [name.split('_')[0].rstrip('0123456789')]:
        pass
    # search video file
    import glob
    hits=glob.glob(str(VID_ROOT/'**'/(name+'*')),recursive=True)
    if not hits:
        hits=glob.glob(str(VID_ROOT/'**'/(name.replace('_x264','')+'*')),recursive=True)
    if not hits: return None
    cap=cv2.VideoCapture(hits[0]); f=cap.get(cv2.CAP_PROP_FPS); cap.release(); return f or 30.0

def gt_snip_mask(name, nsnip=32):
    d=c[name]; ts=d['ts']; dur=d['dur']
    fps=vfps(name) or 30.0
    segs=gt[name]['segments']
    # snippet i covers sampled-frame indices [i*seg:(i+1)*seg]; time = ts of those
    n=len(ts); seg=n//nsnip
    mask=np.zeros(nsnip,bool)
    for i in range(nsnip):
        s=i*seg; e=(s+seg) if i<nsnip-1 else n
        t0=ts[s]; t1=ts[min(e,n)-1]
        for (a,b) in segs:
            a_t,b_t=a/fps,b/fps
            if t1>=a_t and t0<=b_t: mask[i]=True
    return mask

def hm_iou(curve,mask):
    hm=0.5*(curve.max()+curve.min())
    on=curve>=hm
    # contiguous runs
    best=0.0; i=0; n=len(curve)
    runs=[]; j=0
    while j<n:
        if on[j]:
            k=j
            while k<n and on[k]: k+=1
            runs.append((j,k)); j=k
        else: j+=1
    for (s,e) in runs:
        r=np.zeros(n,bool); r[s:e]=True
        inter=(r&mask).sum(); uni=(r|mask).sum()
        if uni>0: best=max(best, inter/uni)
    return best

def stats(curve,mask):
    am=int(np.argmax(curve))
    hit=bool(mask[am])
    rng=float(curve.max()-curve.min())
    if mask.any() and (~mask).any():
        lift=float(curve[mask].mean()-curve[~mask].mean())
    else: lift=0.0
    return hit,round(lift,3),round(hm_iou(curve,mask),3),round(rng,4)

# NORMAL prototype from cached controls (approx: use all cached non-wrong as proxy pool)
normpool=[]
for k,v in c.items():
    if k not in WRONG:
        normpool.append(v['snip'])
normproto=np.mean(np.concatenate(normpool,0),0) if normpool else None

print(f"{'video':22s} {'sig':6s} {'hit':4s} {'lift':6s} {'hmIoU':6s} {'range':7s}")
agg={s:[0,0,[]] for s in ['MAG','NOV','CEN','NORMD']}  # hits, count, iou list
for name in WRONG:
    if name not in c: continue
    snip=c[name]['snip'].astype(np.float64)  # 32x768
    mask=gt_snip_mask(name)
    if not mask.any():
        print(name,'NO GT SNIP'); continue
    curves={}
    curves['MAG']=np.linalg.norm(snip,axis=1)
    nov=np.zeros(32);
    for i in range(1,32):
        a,b=snip[i],snip[i-1]
        nov[i]=1-float(a@b/((np.linalg.norm(a)*np.linalg.norm(b))+1e-9))
    nov[0]=nov[1]; curves['NOV']=nov
    vm=snip.mean(0); curves['CEN']=np.linalg.norm(snip-vm,axis=1)
    if normproto is not None:
        curves['NORMD']=np.linalg.norm(snip-normproto,axis=1)
    for s,cv in curves.items():
        hit,lift,iou,rng=stats(cv,mask)
        agg[s][1]+=1; agg[s][0]+=int(hit); agg[s][2].append(iou)
        print(f"{name:22s} {s:6s} {str(hit):4s} {lift:<6} {iou:<6} {rng:<7}")
    print()
print("=== AGGREGATE over wrong-loc videos ===")
for s,(h,n,ious) in agg.items():
    if n: print(f"{s:6s} argmax_hit {h}/{n}  hmIoU>=0.3: {sum(1 for x in ious if x>=0.3)}/{n}  hmIoU>=0.5: {sum(1 for x in ious if x>=0.5)}/{n}  mean_iou {round(float(np.mean(ious)),3)}")
