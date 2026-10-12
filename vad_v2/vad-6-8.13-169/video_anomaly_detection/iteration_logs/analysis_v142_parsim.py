import sys, json, os, glob, numpy as np, random
sys.path.insert(0,'/Users/zzl/PycharmProjects/vad'); sys.path.insert(0,'/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
import analysis_v140_livecurve_lib as LC
from video_anomaly_detection.evaluate import load_ground_truth, compute_temporal_map
GT=load_ground_truth('/Users/zzl/Movies/ucf-crime/data', split='test'); FPS=30.0

def load_run(run):
    pred={}
    for f in sorted(glob.glob(run+'/*.json')):
        d=json.load(open(f)); vid=os.path.basename(f)[:-5]
        if 'anomalies' not in d: continue
        pred[vid]={'anomalies':[dict(t_start=a['t_start'],t_end=a['t_end'],confidence=a['confidence']) for a in d['anomalies']],'duration':d.get('duration',0.0)}
    return pred

def halfmax_extent(cur, ts, pk_i, frac=0.5):
    v=cur[pk_i]; base=float(np.percentile(cur,10)); thr=base+frac*(v-base)
    i=pk_i
    while i>0 and cur[i-1]>=thr: i-=1
    j=pk_i
    while j<len(cur)-1 and cur[j+1]>=thr: j+=1
    return float(ts[i]), float(ts[j])

def peak_proposal(vid, cat, mode, W, rng=None):
    cur,ts=LC.live_curve(vid,cat); cur=np.asarray(cur).ravel(); ts=np.asarray(ts).ravel()
    if mode=='rand': pk=int(rng.randrange(len(cur)))
    else: pk=int(np.argmax(cur))
    tpk=float(ts[pk])
    if W=='hm': a,b=halfmax_extent(cur,ts,pk)
    else: a,b=tpk-W/2.0, tpk+W/2.0
    a=max(0.0,a); b=min(float(ts[-1]),b)
    if b-a<4.0: a,b=max(0.0,tpk-2.0),min(float(ts[-1]),tpk+2.0)
    return tpk,a,b,float(cur[pk]),cur,ts

def run_variant(pred, mode, W, confpol, gate_iou=None, seed=0):
    rng=random.Random(seed); newpred={}; fired=0; stats=[]
    for vid,e in pred.items():
        g=GT.get(vid); an=[dict(x) for x in e['anomalies']]
        if g and g['segments']:
            try: tpk,a,b,pv,cur,ts=peak_proposal(vid,g['category'],mode,W,rng)
            except Exception: tpk=None
            if tpk is not None:
                covered=any(x['t_start']-1e-9<=tpk<=x['t_end']+1e-9 for x in an)
                if not covered:
                    confs=[x['confidence'] for x in an]
                    if confpol=='median': c=float(np.median(confs)) if confs else 0.5
                    elif confpol=='min': c=(min(confs)-1e-3) if confs else 0.3
                    elif confpol=='max': c=(max(confs)) if confs else 0.6
                    elif confpol=='bottom': c=0.001
                    else: c=float(confpol)
                    an.append(dict(t_start=a,t_end=b,confidence=c))
                    fired+=1
                    gts=[(s/FPS,t/FPS) for s,t in g['segments']]
                    def iou(p,q):
                        i=max(0.0,min(p[1],q[1])-max(p[0],q[0])); u=(p[1]-p[0])+(q[1]-q[0])-i
                        return i/u if u>0 else 0.0
                    stats.append(max(iou((a,b),gg) for gg in gts))
        newpred[vid]={'anomalies':an}
    r=compute_temporal_map(GT,newpred,fps=30)
    return r, fired, stats

BASE='/Users/zzl/PycharmProjects/vad/results/batch_eval_v124_full'
pred=load_run(BASE)
b=compute_temporal_map(GT,pred,fps=30)
print('BASE mAP',round(b['mAP'],5), {k:round(v,4) for k,v in b.items() if k.startswith('AP')})
res=[]
for W in ['hm',12.0,20.0,30.0,40.0]:
    for cp in ['median','max','min']:
        r,f,s=run_variant(pred,'peak',W,cp)
        hr=sum(1 for x in s if x>=0.3)/max(1,len(s))
        res.append((str(W),cp,round(r['mAP']-b['mAP'],5),round(r['mAP'],5),f,round(hr,3),round(float(np.mean(s)) if s else 0,3)))
print('W  conf  dmAP  mAP  fired hit@0.3 meanIoU')
for x in res: print(x)
# random control at best W
print('--- RANDOM-LOCATION CONTROL (signal-free) ---')
for seed in range(5):
    r,f,s=run_variant(pred,'rand',30.0,'median',seed=seed)
    print(' seed',seed,'dmAP',round(r['mAP']-b['mAP'],5),'fired',f,'hit@0.3',round(sum(1 for x in s if x>=0.3)/max(1,len(s)),3))
