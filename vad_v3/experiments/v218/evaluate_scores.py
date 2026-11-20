"""GT is read only here, after the specified inference scope is complete."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

HERE=Path(__file__).resolve().parent


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def write(p,x):
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(p)


def interpolate(row, values, timestamps):
    return np.interp(np.arange(row['frame_count']),np.asarray(timestamps)*row['fps'],
                     np.asarray(values,dtype=np.float32)).astype(np.float32)


def evaluate(rows,manifest,scope,selected_arm=None):
    gtpath=Path('F:/dataset/ucf_crime/data_parquet/test-00000-of-00001.parquet')
    gt={Path(r['video_path']).stem:r for r in pq.read_table(gtpath).to_pylist()}
    predictions={k:[] for k in ['baseline','yesno_probability','fusion_50','yesno_hard_diagnostic']}
    labels=[]; categories=[]; per_video=[]
    totals=dict(videos=len(rows),normal_videos=0,windows=0,exact_pairs=0,bounded_pairs=0,prompt_tokens=0,completion_tokens=0)
    for row in rows:
        n=row['video_id'];result=read(HERE/'results'/(n+'.json'))
        assert result['fingerprint']==manifest['fingerprint']
        source=Path(manifest['source_results'][n]['path'])
        assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source_results'][n]['sha256']
        historical=read(source)
        baseline=historical['arms']['force_best_fixed'] if 'arms' in historical else historical['paired_results']['force_best']
        base_scores=interpolate(row,baseline['frame_scores'],baseline['timestamps'])
        new_scores=interpolate(row,result['frame_scores'],result['timestamps'])
        hard=interpolate(row,result['hard_scores'],result['timestamps'])
        weight=manifest['settings']['fusion_weight']
        arrays=dict(baseline=base_scores,yesno_probability=new_scores,
                    fusion_50=((1-weight)*base_scores+weight*new_scores).astype(np.float32),yesno_hard_diagnostic=hard)
        annotation=gt[n];y=np.zeros(row['frame_count'],dtype=np.uint8)
        for j in (1,2):
            lo,hi=annotation[f'annotation_start{j}'],annotation[f'annotation_end{j}']
            if lo>=0 and hi>=lo:y[lo:min(hi+1,len(y))]=1
        normal=annotation['category']=='Normal'
        if normal:assert not y.any()
        totals['normal_videos']+=normal
        totals['windows']+=len(result['timestamps'])
        for k in ['exact_pairs','bounded_pairs','prompt_tokens','completion_tokens']:totals[k]+=result[k]
        labels.append(y);categories.append(annotation['category'])
        for name,values in arrays.items():
            assert len(values)==len(y) and np.isfinite(values).all()
            predictions[name].append(values)
        per_video.append(dict(video_id=n,category=annotation['category'],positive_frames=int(y.sum()),frames=len(y),
                              auc={k:float(roc_auc_score(y,s)) if 0<y.sum()<len(y) else None for k,s in arrays.items()}))
    all_y=np.concatenate(labels)
    abnormal=np.concatenate([np.full(len(y),c!='Normal') for y,c in zip(labels,categories)])
    metrics={}
    for name,parts in predictions.items():
        all_s=np.concatenate(parts)
        precision,recall,_=precision_recall_curve(all_y,all_s)
        per_category={}
        for cat in sorted(set(categories)-{'Normal'}):
            ys=np.concatenate([y for y,c in zip(labels,categories) if c==cat])
            ss=np.concatenate([s for s,c in zip(parts,categories) if c==cat])
            per_category[cat]=float(roc_auc_score(ys,ss)) if 0<ys.sum()<len(ys) else None
        metrics[name]=dict(global_auc=float(roc_auc_score(all_y,all_s)),
                           ano_auc=float(roc_auc_score(all_y[abnormal],all_s[abnormal])),
                           aupr=float(auc(recall,precision)),per_category_auc=per_category)
    candidates=['yesno_probability','fusion_50']
    chosen=selected_arm or max(candidates,key=lambda x:metrics[x]['global_auc'])
    gate=dict(selected_arm=chosen,selected_auc=metrics[chosen]['global_auc'],threshold=.8,
              passed=metrics[chosen]['global_auc']>.8,selection_scope='pilot65',
              rule='v218 full test explicitly authorized by user; primary fusion_50 frozen before full evaluation. Threshold is descriptive only.')
    report=dict(version='v218',scope=scope,counts=totals,metrics=metrics,gate=gate,per_video=per_video,
                fingerprint=manifest['fingerprint'],gt_sha256=hashlib.sha256(gtpath.read_bytes()).hexdigest(),
                limitations='Exploratory previously used inputs; full290 includes pilot65. Standalone new scorer has no upstream VideoMAE dependency; baseline and fusion retain its provenance limitations. Top-five censoring explicitly counted.')
    write(HERE/(scope+'_metrics.json'),report)
    print(json.dumps(dict(evaluation_scope=scope,counts=totals,
                          auc_percent={k:100*v['global_auc'] for k,v in metrics.items()},gate=gate)),flush=True)
    if scope=='full290':
        previous=read(HERE.parent/'stage4_six_groups_290/six_group_comparison.json')
        assert abs(metrics['baseline']['global_auc']-previous['summary']['force_best_fixed']['global_auc_roc'])<1e-12
    return report
