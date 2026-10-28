import numpy as np, glob, os, json, torch, sys
sys.path.insert(0,'/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
import pandas as pd
from mil_model import MILAnomalyScorer
def stem(p): return os.path.splitext(os.path.basename(str(p)))[0]
df=pd.read_parquet('/Users/zzl/Movies/ucf-crime/data/test-00000-of-00001.parquet')
gt={}
for _,r in df.iterrows():
    segs=[]
    for a,b in [('annotation_start1','annotation_end1'),('annotation_start2','annotation_end2')]:
        if r[a]>=0 and r[b]>=0: segs.append((float(r[a])/30.0,float(r[b])/30.0))
    gt[stem(r['video_path'])]=segs
# durations from v124 cache
dur={}
for f in glob.glob('/Users/zzl/PycharmProjects/vad/results/batch_eval_v124_full/*.json'):
    d=json.load(open(f))
    if 'video_path' in d: dur[stem(d['video_path'])]=d['duration']
m=MILAnomalyScorer(feature_dim=768); sd=torch.load('model_mil_v6.pth',map_location='cpu')
sd=sd.get('model',sd) if isinstance(sd,dict) and 'model' in sd else sd
try: m.load_state_dict(sd)
except Exception as e:
    print('load warn',str(e)[:100]); m.load_state_dict(sd,strict=False)
m.eval()
def milscores(vid):
    a=np.load(f'/Users/zzl/PycharmProjects/vad/features/test/{vid}.npy'); x=torch.tensor(a,dtype=torch.float32).unsqueeze(0)
    with torch.no_grad(): s=m(x)
    return s.squeeze(0).numpy()
Ks=[3,6,10]; recall={k:0 for k in Ks}; top1=0; tot=0; clip_ref=0
for vid,segs in gt.items():
    if not segs or vid not in dur: continue
    if not os.path.exists(f'/Users/zzl/PycharmProjects/vad/features/test/{vid}.npy'): continue
    D=dur[vid]; s=milscores(vid)
    gtsnip=set()
    for (a,b) in segs:
        i0=max(0,int(a/D*32)); i1=min(31,int(b/D*32))
        for i in range(i0,i1+1): gtsnip.add(i)
    if not gtsnip: continue
    tot+=1
    order=np.argsort(-s)
    if order[0] in gtsnip: top1+=1
    for k in Ks:
        if any(i in gtsnip for i in order[:k]): recall[k]+=1
print(f'anomaly videos with GT+feat: {tot}')
print(f'MIL v6 top-1 in GT: {top1}/{tot} = {top1/tot:.3f}')
for k in Ks: print(f'MIL v6 recall@{k} (any top-{k} snippet in GT): {recall[k]}/{tot} = {recall[k]/tot:.3f}')
