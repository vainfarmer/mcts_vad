from pathlib import Path
import json,sys
here=Path(__file__).resolve().parent;sys.path.insert(0,str(here.parents[1]/'video_anomaly_detection'))
from evaluate import load_ground_truth,compute_temporal_map,compute_frame_auc,compute_video_level
labels=load_ground_truth('F:/dataset/ucf_crime/data_parquet');arms={k:{} for k in ('off','on','skip_only')};rows=[]
def canonical(items,keys):return sorted([tuple(a.get(k) for k in keys) for a in items],key=repr)
def best_iou(items,segments):
 values=[]
 for a in items:
  x,y=a['t_start'],a['t_end'];scores=[]
  for sf,ef in segments:
   s,e=sf/30,ef/30;overlap=max(0,min(y,e)-max(x,s));union=max(0,y-x)+max(0,e-s)-overlap;scores.append(overlap/union if union else 0.)
  values.append(max(scores,default=0.))
 return values
for file in sorted((here/'results').glob('*.json')):
 r=json.loads(file.read_text(encoding='utf-8'))
 if 'paired_results' not in r:continue
 off=r['paired_results']['original_off'];on={k:v for k,v in r.items() if k not in ('paired_results','mcts_trace')}
 for arm,value in [('off',off),('on',on),('skip_only',r['paired_results']['skip_only'])]:
  # 导出的文件仅供离线比较；用量归整次配对执行，不能分配给某一个分支。
  value={k:v for k,v in value.items() if k!='token_usage'};arms[arm][file.stem]=value
  dest=here/'comparison_arms'/arm;dest.mkdir(parents=True,exist_ok=True);(dest/file.name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
 boundary_keys=('t_start','t_end');pred_keys=boundary_keys+('confidence','anomaly_type')
 rows.append({'video':file.stem,'category':labels[file.stem]['category'],
  'boundary_changed':canonical(off['anomalies'],boundary_keys)!=canonical(on['anomalies'],boundary_keys),
  'prediction_changed':canonical(off['anomalies'],pred_keys)!=canonical(on['anomalies'],pred_keys),
  'metadata_or_prediction_changed':off['anomalies']!=on['anomalies'],
  'off':off['anomalies'],'on':on['anomalies'],
  'off_best_gt_iou':best_iou(off['anomalies'],labels[file.stem]['segments']),
  'on_best_gt_iou':best_iou(on['anomalies'],labels[file.stem]['segments'])})
gt={k:labels[k] for k in arms['on']}
report={'comparison':'shared upstream; independent complete tails; identical requests cached','videos':rows,
 'boundary_changed_count':sum(x['boundary_changed'] for x in rows),
 'prediction_changed_count':sum(x['prediction_changed'] for x in rows),
 'object_changed_count':sum(x['metadata_or_prediction_changed'] for x in rows),
 'metrics':{k:{'temporal':compute_temporal_map(gt,v,30),'frame':compute_frame_auc(gt,v,30),'video':compute_video_level(gt,v)} for k,v in arms.items()}}
(here/'final_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print({k:v for k,v in report.items() if k not in ('videos','metrics')})
for row in rows:
 if row['boundary_changed']:print(row['video'],row['off_best_gt_iou'],row['on_best_gt_iou'])
