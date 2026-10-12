import json, glob, os, sys
import numpy as np, torch
import pandas as pd
sys.path.insert(0, '/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
from mil_model import MILAnomalyScorer

FPS=30.0
DETDIR='/Users/zzl/PycharmProjects/vad/results/batch_eval_v145_full'
FEAT='/Users/zzl/PycharmProjects/vad/features/test'
PARQ='/Users/zzl/Movies/ucf-crime/data/test-00000-of-00001.parquet'

def extract_vid(p):
    b=os.path.basename(str(p));
    for ext in ['.mp4','.avi','']:
        if b.endswith(ext) and ext: b=b[:-len(ext)]
    return b

df=pd.read_parquet(PARQ)
gt={}
for _,r in df.iterrows():
    vid=extract_vid(r['video_path']); segs=[]
    if r['annotation_start1']>=0 and r['annotation_end1']>=0: segs.append((r['annotation_start1']/FPS,r['annotation_end1']/FPS))
    if r['annotation_start2']>=0 and r['annotation_end2']>=0: segs.append((r['annotation_start2']/FPS,r['annotation_end2']/FPS))
    gt[vid]={'segs':segs}

def load_mil(path):
    m=MILAnomalyScorer(); sd=torch.load(path,map_location='cpu'); m.load_state_dict(sd); m.eval(); return m
mil6=load_mil('/Users/zzl/PycharmProjects/vad/video_anomaly_detection/model_mil_v6.pth')

def snippet_scores(vid):
    f=os.path.join(FEAT,vid+'.npy')
    if not os.path.exists(f): return None
    x=np.load(f); x=torch.from_numpy(x).float().unsqueeze(0)
    with torch.no_grad(): s=mil6(x).squeeze(0).cpu().numpy()  # (32,)
    return s

def iou(a,b):
    s=max(a[0],b[0]); e=min(a[1],b[1]); inter=max(0,e-s)
    uni=(a[1]-a[0])+(b[1]-b[0])-inter
    return inter/uni if uni>0 else 0.0

def win_mil(scores,dur,t0,t1,mode='mean'):
    # snippet i covers [i/32*dur,(i+1)/32*dur]
    edges=np.linspace(0,dur,33)
    ov=[]
    for i in range(32):
        s=max(t0,edges[i]); e=min(t1,edges[i+1])
        if e>s: ov.append(scores[i])
    if not ov:
        # nearest snippet to center
        c=(t0+t1)/2; idx=min(31,max(0,int(c/dur*32))); return scores[idx]
    return float(np.mean(ov)) if mode=='mean' else float(np.max(ov))

JFILES=[]
for jf in glob.glob(os.path.join(DETDIR,'*.json')):
    try:
        dd=json.load(open(jf))
    except Exception: continue
    if isinstance(dd,dict) and 'video_path' in dd and 'duration' in dd: JFILES.append(jf)
print('valid video jsons=',len(JFILES))

dets=[]  # (vid, t0,t1, conf, milmean, milmax, ongt, best_iou)
for jf in JFILES:
    d=json.load(open(jf)); vid=extract_vid(d['video_path'])
    if vid not in gt: continue
    dur=d['duration']; ss=snippet_scores(vid)
    if ss is None: continue
    segs=gt[vid]['segs']
    for a in d.get('anomalies',[]):
        t0,t1=a['t_start'],a['t_end']; conf=a['confidence']
        bi=max([iou((t0,t1),g) for g in segs],default=0.0)
        mm=win_mil(ss,dur,t0,t1,'mean'); mx=win_mil(ss,dur,t0,t1,'max')
        dets.append((vid,t0,t1,conf,mm,mx,int(bi>0),bi))

dets=np.array([(c,mm,mx,og,bi) for (_,_,_,c,mm,mx,og,bi) in dets])
conf=dets[:,0]; milm=dets[:,1]; milx=dets[:,2]; ong=dets[:,3].astype(int); biou=dets[:,4]
print(f'N dets={len(dets)}  on-GT(IoU>0)={ong.sum()}  off-GT={ (ong==0).sum() }')
def sep(name,v):
    on=v[ong==1]; off=v[ong==0]
    print(f'{name:10s} on-GT mean={on.mean():.4f} med={np.median(on):.4f} | off-GT mean={off.mean():.4f} med={np.median(off):.4f} | gap={on.mean()-off.mean():+.4f}')
sep('scan_conf',conf); sep('mil_mean',milm); sep('mil_max',milx)
# AUC (rank-based separation): P(on>off)
def auc(v):
    on=v[ong==1]; off=v[ong==0]
    if len(on)==0 or len(off)==0: return float('nan')
    c=0;
    for o in on: c+=(o>off).sum()+0.5*(o==off).sum()
    return c/(len(on)*len(off))
print(f'AUC on-vs-off: scan_conf={auc(conf):.3f} mil_mean={auc(milm):.3f} mil_max={auc(milx):.3f}')
# corr of conf with mil (are they redundant?)
print(f'corr(conf,mil_mean)={np.corrcoef(conf,milm)[0,1]:.3f}')

# ---- AP simulation: greedy single-match global AP at IoU thr, 11-pt interp ----
# rebuild det list with vid to match GT per video
detl=[]
for jf in JFILES:
    d=json.load(open(jf)); vid=extract_vid(d['video_path'])
    if vid not in gt: continue
    dur=d['duration']; ss=snippet_scores(vid)
    if ss is None: continue
    for a in d.get('anomalies',[]):
        detl.append({'vid':vid,'w':(a['t_start'],a['t_end']),'conf':a['confidence'],
                     'milm':win_mil(ss,dur,a['t_start'],a['t_end'],'mean')})
alljson_vids=set(extract_vid(json.load(open(jf))['video_path']) for jf in JFILES)
ngt=sum(len(gt[v]['segs']) for v in alljson_vids if v in gt)
# AUC with on-GT defined at IoU>=0.3 (the harness hit threshold)
on3=(biou>=0.3).astype(int)
def auc3(v):
    on=v[on3==1]; off=v[on3==0]
    if len(on)==0 or len(off)==0: return float('nan')
    c=sum((o>off).sum()+0.5*(o==off).sum() for o in on); return c/(len(on)*len(off))
print(f'IoU>=0.3 hits={on3.sum()}  AUC@.3: scan_conf={auc3(conf):.3f} mil_mean={auc3(milm):.3f} mil_max={auc3(milx):.3f}')

def ap_at(thr, key):
    order=sorted(detl,key=lambda x:-x[key])
    matched={}
    tp=np.zeros(len(order)); fp=np.zeros(len(order))
    for i,dd in enumerate(order):
        segs=gt[dd['vid']]['segs']; best=-1; bj=0
        for j,g in enumerate(segs):
            ii=iou(dd['w'],g)
            if ii>=thr and ii>bj and (dd['vid'],j) not in matched: bj=ii; best=j
        if best>=0: matched[(dd['vid'],best)]=1; tp[i]=1
        else: fp[i]=1
    ctp=np.cumsum(tp); cfp=np.cumsum(fp)
    rec=ctp/max(ngt,1); prec=ctp/np.maximum(ctp+cfp,1)
    ap=0.0
    for t in np.linspace(0,1,11):
        p=prec[rec>=t]; ap+=(p.max() if len(p) else 0)/11
    return ap
for key in ['conf','milm']:
    aps=[ap_at(t,key) for t in [0.3,0.5,0.7]]
    print(f'rank_by={key:5s} AP@.3={aps[0]:.4f} AP@.5={aps[1]:.4f} AP@.7={aps[2]:.4f} mAP={np.mean(aps):.4f}')
# blend sweep
for w in [0.2,0.35,0.5,0.65]:
    cmin,cmax=conf.min(),conf.max(); mmin,mmax=milm.min(),milm.max()
    for dd in detl:
        cn=(dd['conf']-cmin)/(cmax-cmin+1e-9); mn=(dd['milm']-mmin)/(mmax-mmin+1e-9)
        dd['blend']=(1-w)*cn+w*mn
    aps=[ap_at(t,'blend') for t in [0.3,0.5,0.7]]
    print(f'blend w_mil={w:.2f} AP@.3={aps[0]:.4f} AP@.5={aps[1]:.4f} AP@.7={aps[2]:.4f} mAP={np.mean(aps):.4f}')
print('ngt=',ngt,'ndet=',len(detl))
