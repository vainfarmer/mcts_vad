"""C-mandated realized-ceiling recompute for the MIL relocation lever.
Uses cached 32-snippet features (analysis_milv6_feats.pkl), runs v1 and v6 MIL,
computes the EXACT implemented run def (median-half-max, max-MEAN run), realized
IoU vs GT, and tests source-gates for zero-overlap detection separation.
"""
import sys, json, pickle
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
from evaluate import load_ground_truth, load_predictions, _temporal_iou
from mil_model import MILAnomalyScorer

ROOT = Path('/Users/zzl/PycharmProjects/vad')
FEAT = ROOT / 'video_anomaly_detection/iteration_logs/analysis_milv6_feats.pkl'
feats = pickle.load(open(FEAT, 'rb'))
gt = load_ground_truth('/Users/zzl/Movies/ucf-crime/data', split='test')
pred = load_predictions(str(ROOT / 'results/batch_eval_v144c_bue65ref'))

def load_mil(p):
    m = MILAnomalyScorer(feature_dim=768, hidden_dim=512, k=3, dropout=0.6)
    m.load_state_dict(torch.load(p, map_location='cpu')); m.eval(); return m
mil1 = load_mil(ROOT/'video_anomaly_detection/model_mil.pth')
mil6 = load_mil(ROOT/'video_anomaly_detection/model_mil_v6.pth')

def scores(model, snip):
    x = torch.tensor(snip, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return model(x).squeeze(0).numpy()

def maxmean_run(s, frac):
    """median half-max, pick max-MEAN contiguous above-threshold run. returns (a,b) snippet idx."""
    hm = frac*s.max() + (1-frac)*np.median(s)
    above = s >= hm
    runs=[]; st=None
    for i,a in enumerate(above):
        if a and st is None: st=i
        elif not a and st is not None: runs.append((st,i)); st=None
    if st is not None: runs.append((st,32))
    if not runs: return None
    best=max(runs, key=lambda r: s[r[0]:r[1]].mean())
    return best

WRONG = ['Abuse028_x264','Burglary005_x264','Burglary079_x264','Fighting003_x264',
         'Fighting018_x264','Fighting033_x264','RoadAccidents001_x264','RoadAccidents019_x264',
         'Robbery050_x264','Robbery102_x264','Shooting018_x264','Shooting032_x264',
         'Shoplifting004_x264','Shoplifting037_x264','Stealing036_x264','Stealing062_x264','Vandalism015_x264']
CONTROLS = ['Arson009_x264','Shoplifting001_x264']

def analyze(vid):
    if vid not in feats or vid not in gt or vid not in pred: return None
    e=feats[vid]; dur=float(e['dur']); snip_dur=dur/32.0
    s1=scores(mil1, e['snip']); s6=scores(mil6, e['snip'])
    segs=gt[vid]['segments']  # frames @30
    out={'vid':vid, 'dur':dur, 'rng1':float(s1.max()-s1.min()), 'rng6':float(s6.max()-s6.min())}
    for tag,s in [('1',s1),('6',s6)]:
        run=maxmean_run(s,0.5)
        if run is None:
            out['iou'+tag]=0.0; out['runmean'+tag]=0.0; out['runc'+tag]=None; continue
        a,b=run
        win={'t_start':a*snip_dur,'t_end':b*snip_dur}
        iou=max([_temporal_iou(win,g,30) for g in segs]+[0.0])
        out['iou'+tag]=iou
        out['runmean'+tag]=float(s[a:b].mean())
        out['runc'+tag]=(a+b)/2.0*snip_dur
        out['runspan'+tag]=(a*snip_dur, b*snip_dur)
    # per current-baseline det: true IoU, gap to run, det-window MIL mean
    dets=[]
    for d in pred[vid]['anomalies']:
        ts_,te_=d['t_start'],d['t_end']
        true_iou=max([_temporal_iou(d,g,30) for g in segs]+[0.0])
        c=(ts_+te_)/2.0
        m=np.array([(i*snip_dur<te_) and ((i+1)*snip_dur>ts_) for i in range(32)])
        dwm1=float(s1[m].mean()) if m.any() else float(s1.mean())
        dwm6=float(s6[m].mean()) if m.any() else float(s6.mean())
        gap1=abs(c-out['runc1']) if out['runc1'] is not None else 0.0
        gap6=abs(c-out['runc6']) if out['runc6'] is not None else 0.0
        dets.append(dict(iou=true_iou,center=c,dwm1=dwm1,dwm6=dwm6,gap1=gap1,gap6=gap6,conf=d.get('confidence',0)))
    out['dets']=dets
    return out

rows=[analyze(v) for v in WRONG]; rows=[r for r in rows if r]
crows=[analyze(v) for v in CONTROLS]; crows=[r for r in crows if r]

print('=== REALIZED max-mean run IoU (median half-max, implemented def) ===')
for tag in ['1','6']:
    ge3=sum(1 for r in rows if r['iou'+tag]>=0.3); ge5=sum(1 for r in rows if r['iou'+tag]>=0.5)
    print(f'  v{tag}: realized-run IoU>=0.3: {ge3}/{len(rows)}   >=0.5: {ge5}/{len(rows)}')

print('\n=== per wrong-loc video (v6, the non-flat model) ===')
for r in rows:
    nd=len(r['dets']); zeros=sum(1 for d in r['dets'] if d['iou']==0)
    print(f"  {r['vid'][:20]:20s} rng6={r['rng6']:.2f} run6IoU={r['iou6']:.3f} runmean6={r['runmean6']:.2f} ndet={nd} zero={zeros}")

# proxy test: fire iff det is zero-overlap candidate. C's proxy: large gap + low det MIL + high run mean
print('\n=== PROXY: relocate det onto v6 run iff gap6>GAP and runmean6-dwm6>MARGIN and rng6>=RNG ===')
for GAP,MARG,RNG in [(20,0.15,0.15),(15,0.10,0.10),(30,0.20,0.20)]:
    fired=0; fired_zero=0; fired_tp=0; conv3=0; conv5=0; ctrl_fire=0
    for r in rows:
        if r['rng6']<RNG or r['runc6'] is None: continue
        for d in r['dets']:
            if d['gap6']>GAP and (r['runmean6']-d['dwm6'])>MARG:
                fired+=1
                if d['iou']==0: fired_zero+=1
                else: fired_tp+=1
        # would this video's relocation convert? only if a fired det exists and run overlaps GT
        anyfire=any(d['gap6']>GAP and (r['runmean6']-d['dwm6'])>MARG for d in r['dets'])
        if anyfire and r['iou6']>=0.3: conv3+=1
        if anyfire and r['iou6']>=0.5: conv5+=1
    for r in crows:
        if r['rng6']<RNG or r['runc6'] is None: continue
        for d in r['dets']:
            if d['gap6']>GAP and (r['runmean6']-d['dwm6'])>MARG:
                ctrl_fire+=1
    print(f'  GAP>{GAP} MARG>{MARG} RNG>={RNG}: fired={fired} (zero={fired_zero} TP={fired_tp}) conv@0.3={conv3} conv@0.5={conv5} CTRL_misfire={ctrl_fire}')
