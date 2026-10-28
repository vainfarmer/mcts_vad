import sys, json, glob, numpy as np, torch
sys.path.insert(0, '/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
from mil_model import MILAnomalyScorer
import pandas as pd

FEAT='/Users/zzl/PycharmProjects/vad/features/test'
RUN='/Users/zzl/PycharmProjects/vad/results/batch_eval_v124_full'
PARQ='/Users/zzl/Movies/ucf-crime/data/test-00000-of-00001.parquet'

def vid_id(p):
    import os; return os.path.splitext(os.path.basename(str(p)))[0]

# GT
df=pd.read_parquet(PARQ)
gt={}
for _,r in df.iterrows():
    segs=[]
    if r['annotation_start1']>=0 and r['annotation_end1']>=0: segs.append((r['annotation_start1']/30.0,r['annotation_end1']/30.0))
    if r['annotation_start2']>=0 and r['annotation_end2']>=0: segs.append((r['annotation_start2']/30.0,r['annotation_end2']/30.0))
    gt[vid_id(r['video_path'])]={'segs':segs}

# MIL
dev='cpu'
model=MILAnomalyScorer().to(dev)
sd=torch.load('/Users/zzl/PycharmProjects/vad/video_anomaly_detection/model_mil_v6.pth',map_location=dev)
if isinstance(sd,dict) and 'model' in sd: sd=sd['model']
if isinstance(sd,dict) and 'state_dict' in sd: sd=sd['state_dict']
model.load_state_dict(sd); model.eval()

def iou(a,b):
    s=max(a[0],b[0]); e=min(a[1],b[1]); inter=max(0,e-s)
    u=(a[1]-a[0])+(b[1]-b[0])-inter
    return inter/u if u>0 else 0.0

rows=[]  # (vlm_conf, mil_mean, mil_max, mil_gmax, best_iou, dur, gtmin_mil, has_gt)
missing=0
for jf in sorted(glob.glob(RUN+'/*.json')):
    d=json.load(open(jf))
    if 'video_path' not in d: continue
    v=vid_id(d['video_path']); dur=d['duration']
    anoms=d.get('anomalies',[])
    if not anoms: continue
    ff=f'{FEAT}/{v}.npy'
    try: feat=np.load(ff)
    except: missing+=1; continue
    T=feat.shape[0]
    with torch.no_grad():
        sc=model(torch.tensor(feat[None],dtype=torch.float32)).squeeze(0).cpu().numpy()  # (T,)
    gmax=float(sc.max())
    segs=gt.get(v,{}).get('segs',[])
    for a in anoms:
        ts,te=a['t_start'],a['t_end']
        # snippet indices overlapping [ts,te]
        i0=int(np.floor(ts/dur*T)); i1=int(np.ceil(te/dur*T))
        i0=max(0,min(T-1,i0)); i1=max(i0+1,min(T,i1))
        win=sc[i0:i1]
        bi=max([iou((ts,te),g) for g in segs],default=0.0)
        rows.append((a['confidence'],float(win.mean()),float(win.max()),gmax,bi,dur))

R=np.array(rows)
conf,mmean,mmax,gmax,biou,dur=R[:,0],R[:,1],R[:,2],R[:,3],R[:,4],R[:,5]
print(f'dets={len(R)} missing_feat={missing}')
for thr,name in [(1e-9,'IoU>0'),(0.5,'IoU>=0.5'),(0.3,'IoU>=0.3')]:
    pos=biou>=thr; neg=~pos
    print(f'\n== TP def {name}: TP={pos.sum()} FP={neg.sum()} ==')
    def auc(s):
        from itertools import product
        # rank-based AUC
        order=np.argsort(s); r=np.empty_like(order,dtype=float); r[order]=np.arange(len(s))
        return (r[pos].mean()-(pos.sum()-1)/2)/neg.sum() if pos.sum()>0 and neg.sum()>0 else float('nan')
    for nm,s in [('VLM_conf',conf),('MIL_mean',mmean),('MIL_max',mmax),('MIL_gmax',gmax)]:
        print(f'  AUC {nm:9s}={auc(s):.4f}  pos_mean={s[pos].mean():.3f} neg_mean={s[neg].mean():.3f}')
    # blend AUC over grid
    best=None
    cz=(conf-conf.mean())/conf.std(); mz=(mmean-mmean.mean())/mmean.std()
    for w in [0,.1,.2,.3,.4,.5,.7,1.0]:
        s=(1-w)*cz+w*mz
        order=np.argsort(s); r=np.empty_like(order,dtype=float); r[order]=np.arange(len(s))
        a=(r[pos].mean()-(pos.sum()-1)/2)/neg.sum()
        if best is None or a>best[1]: best=(w,a)
    print(f'  best blend w_mil={best[0]} AUC={best[1]:.4f}')
