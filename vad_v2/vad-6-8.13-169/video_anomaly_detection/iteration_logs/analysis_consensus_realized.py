"""Realized full-65 dmAP for CEN+NOV z-consensus relocation (after-BUE scaffold).
Verdict-independent, count-neutral. Measures delta via evaluate.compute_temporal_map."""
import sys,glob,json,copy
from pathlib import Path
import numpy as np
sys.path.insert(0,'/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
from evaluate import load_ground_truth, load_predictions, compute_temporal_map
ROOT=Path('/Users/zzl/PycharmProjects/vad')
FEAT=ROOT/'features/test'
BASE=str(ROOT/'results/batch_eval_v144c_bue65ref')
gt=load_ground_truth('/Users/zzl/Movies/ucf-crime/data',split='test')
pred=load_predictions(BASE)
# restrict gt to the 65 eval videos present in pred
vids=set(pred.keys())
gt65={k:v for k,v in gt.items() if k in vids}

def znorm(x):
    sd=x.std(); return (x-x.mean())/sd if sd>1e-9 else x*0
def consensus(snip):
    vm=snip.mean(0); cen=np.linalg.norm(snip-vm,axis=1)
    nov=np.zeros(len(snip))
    for i in range(1,len(snip)):
        a,b=snip[i],snip[i-1]; nov[i]=1-a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
    nov[0]=nov[1]
    return znorm(cen)+znorm(nov)
def maxmean_run(s,frac):
    hm=frac*s.max()+(1-frac)*np.median(s); above=s>=hm
    runs=[];st=None
    for i,a in enumerate(above):
        if a and st is None:st=i
        elif not a and st is not None:runs.append((st,i));st=None
    if st is not None:runs.append((st,len(s)))
    if not runs:return None
    return max(runs,key=lambda r:s[r[0]:r[1]].mean())

def base_map():
    return compute_temporal_map(gt65,pred)

def relocate(peakz, gap_frac, frac):
    newpred=copy.deepcopy(pred)
    fires=0; vids_fired=0
    for vid,entry in newpred.items():
        fp=FEAT/(vid+'.npy')
        if not fp.exists(): continue
        dets=entry['anomalies']
        if not dets: continue
        snip=np.load(fp).astype(np.float64)
        ns=len(snip)
        cv=consensus(snip)
        if cv.max()<peakz:  # non-prominent -> skip (in-vivo gate)
            continue
        run=maxmean_run(cv,frac)
        if run is None: continue
        dur=float(entry.get('duration') or 0.0)
        if dur<=0: continue
        sd=dur/ns
        a,b=run; rs,re=a*sd,b*sd; rc=(rs+re)/2.0
        vf=False
        for d in dets:
            c=(d['t_start']+d['t_end'])/2.0
            if abs(c-rc) >= gap_frac*dur:  # far det -> relocate (count/conf/rank frozen)
                d['t_start']=rs; d['t_end']=re
                fires+=1; vf=True
        if vf: vids_fired+=1
    return newpred,fires,vids_fired

b=base_map()
print(f"BASE mAP={b['mAP']:.5f} AP@0.3={b['AP@0.3']:.5f} AP@0.5={b['AP@0.5']:.5f} AP@0.7={b['AP@0.7']:.5f}")
for peakz,gap,frac in [(1.5,0.15,0.25),(1.0,0.15,0.25),(2.0,0.20,0.25),(1.5,0.10,0.5),(1.5,0.15,0.5),(1.0,0.10,0.25)]:
    np_,fires,vf=relocate(peakz,gap,frac)
    m=compute_temporal_map(gt65,np_)
    print(f"peakz>={peakz} gap>={gap} frac={frac}: fires={fires} vids={vf} mAP={m['mAP']:.5f} (d={m['mAP']-b['mAP']:+.5f}) AP@0.3 d={m['AP@0.3']-b['AP@0.3']:+.5f} AP@0.5 d={m['AP@0.5']-b['AP@0.5']:+.5f} AP@0.7 d={m['AP@0.7']-b['AP@0.7']:+.5f}")
