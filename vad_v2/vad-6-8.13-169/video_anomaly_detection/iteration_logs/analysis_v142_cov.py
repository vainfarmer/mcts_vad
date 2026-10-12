import sys,json,glob,os,numpy as np
sys.path.insert(0,'/Users/zzl/PycharmProjects/vad'); sys.path.insert(0,'/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
import analysis_v140_livecurve_lib as LC
from video_anomaly_detection.evaluate import load_ground_truth
GT=load_ground_truth('/Users/zzl/Movies/ucf-crime/data',split='test'); FPS=30.0
rows=json.load(open('/tmp/zp124.json')); vids=sorted(set(r['vid'] for r in rows))
tot=dict(nv=0,nseg=[],nscan=[],cov_scan=[],cov_all=[],gt_reach_scan=0,gt_reach_all=0,ngt=0,run_needed=[])
for vid in vids:
    g=GT.get(vid)
    if not g or not g['segments']: continue
    try: cur,ts=LC.live_curve(vid,g['category'])
    except Exception: continue
    cur=np.asarray(cur).ravel(); ts=np.asarray(ts).ravel(); T=float(ts[-1])
    # reproduce pipeline segmentation grid: 6s window, 25% overlap => step 4.5s
    segs=[]; t=0.0
    while t < T-1.0:
        segs.append((t,min(t+6.0,T))); t+=4.5
    def segscore(a,b):
        m=(ts>=a)&(ts<=b); return float(cur[m].mean()) if m.any() else -1.0
    sc=[segscore(a,b) for a,b in segs]
    order=sorted(range(len(segs)),key=lambda i:-sc[i])
    max_scan=min(max(10,len(segs)//5),18)
    scan=set(order[:max_scan])
    gts=[(s/FPS,e/FPS) for s,e in g['segments']]
    def ov(a,b,c,d): return max(0.0,min(b,d)-max(a,c))
    tot['nv']+=1; tot['nseg'].append(len(segs)); tot['nscan'].append(min(max_scan,len(segs)))
    for (a,b) in gts:
        tot['ngt']+=1
        cs=sum(ov(segs[i][0],segs[i][1],a,b) for i in scan)
        ca=sum(ov(s0,s1,a,b) for s0,s1 in segs)
        L=max(b-a,1e-6)
        tot['cov_scan'].append(min(1.0,cs/L)); tot['cov_all'].append(min(1.0,ca/L))
        if any(ov(segs[i][0],segs[i][1],a,b)>0 for i in scan): tot['gt_reach_scan']+=1
        if any(ov(s0,s1,a,b)>0 for s0,s1 in segs): tot['gt_reach_all']+=1
        # how many contiguous 6s segs are needed to cover this GT at IoU>=0.3
        tot['run_needed'].append(max(1,int(np.ceil(0.3*(b-a)/6.0))))
print('videos',tot['nv'],'ngt',tot['ngt'])
print('mean segs/video',round(np.mean(tot['nseg']),1),'mean scanned',round(np.mean(tot['nscan']),1),
      'coverage frac',round(np.mean(tot['nscan'])/np.mean(tot['nseg']),3))
print('GT reachable by SCANNED set (any overlap):',tot['gt_reach_scan'],'/',tot['ngt'],'=',round(tot['gt_reach_scan']/tot['ngt'],3))
print('GT reachable by ALL segments:',tot['gt_reach_all'],'/',tot['ngt'])
cs=np.array(tot['cov_scan']); ca=np.array(tot['cov_all'])
print('mean GT time-fraction covered by SCANNED segs',round(cs.mean(),3),'| by ALL segs',round(ca.mean(),3))
print('GT with scanned-coverage >=0.5:',int((cs>=0.5).sum()),' >=0.3:',int((cs>=0.3).sum()),' ==0:',int((cs==0).sum()))
print('so GT segs whose scanned coverage < 0.3 (cannot yield IoU>=0.3 candidate):',int((cs<0.3).sum()))
