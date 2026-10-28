from pathlib import Path
import sys,json,collections
here=Path(__file__).resolve().parent;code=here.parents[1]/'video_anomaly_detection';sys.path.insert(0,str(code))
from evaluate import load_ground_truth,compute_temporal_map,compute_category_accuracy
report=json.loads((here/'comparison.json').read_text(encoding='utf-8'))
records={p.stem:json.loads(p.read_text(encoding='utf-8')) for p in (here/'results').glob('*.json') if p.name!='evaluation_report.json'}
labels=load_ground_truth(str(here/'dataset/data'));gt={k:labels[k] for k in records}
usage=json.loads((here/'results/metadata/token_usage.json').read_text(encoding='utf-8'))
evaluation=json.loads((here/'results/evaluation_report.json').read_text(encoding='utf-8'))
rows=[];candidates=[]
for name,r in records.items():
 t=r.get('mcts_trace') or {};cs=list(t.get('candidates',{}).values());candidates.extend(cs)
 off=r['paired_results']['original_off'];on=r
 rows.append(dict(video=name,category=gt[name]['category'],off_mAP=compute_temporal_map({name:gt[name]},{name:off},30)['mAP'],on_mAP=compute_temporal_map({name:gt[name]},{name:on},30)['mAP'],off_count=len(off['anomalies']),on_count=len(on['anomalies']),tokens=r['token_usage']['total_tokens'],mcts_requests=t.get('logical_requests',0),applied=sum(bool(c.get('applied')) for c in cs),reasons=[c.get('reason') for c in cs]))
report['per_video']=rows;report['token_usage']=usage['usage'];report['timing']=evaluation['timing']
for arm in ('off','on','skip_only'):
 pred=records if arm=='on' else {k:r['paired_results']['original_off' if arm=='off' else 'skip_only'] for k,r in records.items()}
 report['arms'][arm]['category_accuracy']=compute_category_accuracy(gt,pred)
(here/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['per_video','arms']},ensure_ascii=False,indent=2))
for arm,a in report['arms'].items():print(arm,{k:v for k,v in a.items() if k!='per_category'})
