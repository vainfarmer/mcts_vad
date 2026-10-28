"""用v215冻结结果验证最终NMS：不重跑模型，不修改v215文件。"""
from pathlib import Path
import copy
import json
import sys

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parents[1]/'video_anomaly_detection'))
from evaluate import load_ground_truth, compute_temporal_map
from post_processor import temporal_nms

arms={k:{} for k in ('off','on','off_final_nms','on_final_nms')}
changes=[]
for f in sorted((here.parent/'v215/results').glob('*.json')):
    raw=json.loads(f.read_text(encoding='utf-8'))
    if 'paired_results' not in raw:continue
    for name,r in [('off',raw['paired_results']['original_off']),('on',raw)]:
        arms[name][f.stem]=r
        updated=copy.deepcopy(r)
        updated['anomalies']=temporal_nms(updated['anomalies'],iou_threshold=.3)
        arms[name+'_final_nms'][f.stem]=updated
        if len(r['anomalies'])!=len(updated['anomalies']):
            changes.append(dict(video=f.stem,arm=name,before=len(r['anomalies']),after=len(updated['anomalies'])))
assert len(arms['on'])==65
labels=load_ground_truth('F:/dataset/ucf_crime/data_parquet')
gt={k:labels[k] for k in arms['on']}
report=dict(videos=65,source='v215 frozen final predictions; offline final NMS only; no API',
            metrics={k:compute_temporal_map(gt,r,30) for k,r in arms.items()},changes=changes)
(here/'offline_nms_65.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
