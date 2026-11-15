"""Four-level XD reports with explicit multilabel and Normal semantics.

Loads only compact predictions, not all videos' MCTS trees or frame images.
Normal is a seventh detection class; the six-anomaly-class reference is retained.
Undefined metrics (e.g. ROC on Normal-only frames) are None, logged as N/A.
"""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from xd_config import XD_CLASSES
from xd_evaluate import temporal_metrics
import vad_logger

log=vad_logger.get('vad.xd.metrics')
ALL_CLASSES=(*XD_CLASSES,'Normal')
ARMS=('baseline_off','mcts','mcts_qscore','mcts_qscore_cbt','mcts_qscore_cbt_bue','mcts_cbt_bue')


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def safe_ratio(a,b):
    return float(a/b) if b else None


def score_metrics(labels,scores):
    labels=np.asarray(labels);scores=np.asarray(scores)
    return dict(AUC=float(roc_auc_score(labels,scores)) if len(np.unique(labels))==2 else None,
                AP=float(average_precision_score(labels,scores)) if np.any(labels) else None)


def binary_metrics(truth,prediction,scores):
    truth=np.asarray(truth,dtype=bool);prediction=np.asarray(prediction,dtype=bool)
    tp=int(np.sum(truth & prediction));fp=int(np.sum(~truth & prediction))
    fn=int(np.sum(truth & ~prediction));tn=int(np.sum(~truth & ~prediction))
    return dict(accuracy=float(np.mean(truth==prediction)),precision=safe_ratio(tp,tp+fp),
                recall=safe_ratio(tp,tp+fn),f1=safe_ratio(2*tp,2*tp+fp+fn),
                auc=score_metrics(truth,scores)['AUC'],tp=tp,fp=fp,fn=fn,tn=tn,
                score_definition='maximum source-frame anomaly score',
                decision_definition='at least one final anomalous detection')


def category_metrics(expected,predicted):
    """Set accuracy handles simultaneous labels without arbitrary top-1 mapping."""
    per={};tp_sum=fp_sum=fn_sum=0
    for c in ALL_CLASSES:
        tp=sum(c in a and c in b for a,b in zip(expected,predicted))
        fp=sum(c not in a and c in b for a,b in zip(expected,predicted))
        fn=sum(c in a and c not in b for a,b in zip(expected,predicted))
        per[c]=dict(correct=tp,total=tp+fn,tp=tp,fp=fp,fn=fn,
                    precision=safe_ratio(tp,tp+fp),recall=safe_ratio(tp,tp+fn),f1=safe_ratio(2*tp,2*tp+fp+fn))
        tp_sum+=tp;fp_sum+=fp;fn_sum+=fn
    return dict(exact_set_accuracy=float(np.mean([a==b for a,b in zip(expected,predicted)])),
                micro_precision=safe_ratio(tp_sum,tp_sum+fp_sum),micro_recall=safe_ratio(tp_sum,tp_sum+fn_sum),
                micro_f1=safe_ratio(2*tp_sum,2*tp_sum+fp_sum+fn_sum),per_class=per)


def fmt(value):
    return 'N/A' if value is None else f'{value:.4f}'


def log_report(arm,report):
    log.info('='*60)
    log.info('Evaluation arm: %s',arm)
    log.info('--- Level 1: Frame-level AUC-ROC / AP ---')
    log.info('  Global AUC-ROC: %s',fmt(report['frame_AUC']))
    log.info('  Global AP:      %s (average_precision_score)',fmt(report['frame_AP']))
    log.info('  Per-category AUC-ROC / AP (GT-category video subsets; multilabel subsets overlap):')
    for c,r in report['frame_per_category'].items():
        log.info('    %-20s: AUC=%s AP=%s videos=%d',c,fmt(r['AUC']),fmt(r['AP']),r['video_count'])
    log.info('--- Level 2: Temporal Segment mAP (7 classes, includes Normal) ---')
    values=report['temporal_mAP_7class']['values']
    log.info('  mAP:    %.4f',values['AVG'])
    for t in (.1,.2,.3,.4,.5):log.info('  mAP@%.1f: %.4f',t,values[f'mAP@{t:.1f}'])
    log.info('  Reference 6-anomaly-class mAP: %.4f',report['temporal_mAP_full']['values']['AVG'])
    log.info('  Normal AP@0.5: %.4f',report['temporal_mAP_7class']['per_class']['Normal']['AP@0.5'])
    log.info('--- Level 3: Video-level Metrics ---')
    for label,key in [('Accuracy','accuracy'),('Precision','precision'),('Recall','recall'),('F1','f1'),('AUC','auc')]:
        log.info('  %-10s: %s',label,fmt(report['video_metrics'][key]))
    log.info('--- Level 4: Category Accuracy (multilabel) ---')
    cats=report['category_metrics']
    log.info('  Overall exact-set accuracy: %s',fmt(cats['exact_set_accuracy']))
    log.info('  Micro precision=%s recall=%s F1=%s',fmt(cats['micro_precision']),fmt(cats['micro_recall']),fmt(cats['micro_f1']))
    for c,r in cats['per_class'].items():
        log.info('  %-20s: %d/%d | precision=%s recall=%s F1=%s',c,r['correct'],r['total'],fmt(r['precision']),fmt(r['recall']),fmt(r['f1']))


def evaluate_full(output_dir,records,metadata,fingerprint,annotation_hashes):
    output=Path(output_dir);names=sorted(records)
    paths={p.stem:p for p in (output/'result').glob('*.json')}
    matched=set(paths)&set(names)
    log.info('GT videos: %d, Predicted videos: %d, Matched: %d',len(names),len(paths),len(matched))
    log.info('  GT only (missing predictions): %d',len(set(names)-set(paths)))
    log.info('FPS source: per-video manifest (probed from source videos)')
    log.info('  Abnormal: %d, Normal: %d',sum(bool(r['events']) for r in records.values()),sum(r['category']=='Normal' for r in records.values()))
    if set(paths)!=set(names):raise ValueError('Full evaluation requires exact selected-video coverage')
    predictions={a:{} for a in ARMS};timestamps={};processing={};traces=[]
    for name in names:
        data=json.loads(paths[name].read_text(encoding='utf-8'))
        if data['experiment_fingerprint']!=fingerprint or Path(data['video_path']).stem!=name:
            raise ValueError('Result identity/fingerprint mismatch: '+name)
        timestamps[name]=np.asarray(data['timestamps'],dtype=float)
        processing[name]=data['processing_time']
        if len(timestamps[name])==0 or np.any(np.diff(timestamps[name])<=0):raise ValueError('Invalid timestamps')
        for arm in ARMS:
            row=data['baseline_off'] if arm=='baseline_off' else data['arms'][arm]
            if row['is_normal'] != (not row['anomalies']):raise ValueError('Inconsistent video Normal decision')
            scores=np.asarray(row['frame_scores'],dtype=float)
            if scores.shape!=timestamps[name].shape or not np.isfinite(scores).all() or np.any(scores<0) or np.any(scores>1):
                raise ValueError('Invalid frame scores')
            predictions[arm][name]=dict(anomalies=row['anomalies'],detections=row['detections'],
                                        is_normal=row['is_normal'],frame_scores=scores)
        traces.append(data['shared']['mcts_trace'])
    labels=[];offsets={};pos=0;expected=[]
    for name in names:
        count=metadata[name]['frame_count'];y=np.zeros(count,dtype=np.uint8)
        for start,end in records[name]['segments']:
            if not 0<=start<end<=count:raise ValueError('Invalid binary GT: '+name)
            y[start:end]=1
        for e in records[name]['events']:
            if not 0<=e['start_frame']<e['end_frame']<=count:raise ValueError('Invalid typed GT: '+name)
        labels.append(y);offsets[name]=(pos,pos+count);pos+=count
        expected.append({e['category'] for e in records[name]['events']} or {'Normal'})
    y=np.concatenate(labels);del labels
    groups={c:[i for i,s in enumerate(expected) if c in s] for c in ALL_CLASSES}
    reports={}
    for arm in ARMS:
        preds=predictions[arm]
        scores=np.concatenate([np.interp(np.arange(metadata[n]['frame_count'])/metadata[n]['fps'],timestamps[n],preds[n]['frame_scores']) for n in names])
        global_scores=score_metrics(y,scores)
        per={}
        for c,indices in groups.items():
            spans=[offsets[names[i]] for i in indices]
            yy=np.concatenate([y[a:b] for a,b in spans]) if spans else np.array([],dtype=np.uint8)
            ss=np.concatenate([scores[a:b] for a,b in spans]) if spans else np.array([])
            per[c]=dict(score_metrics(yy,ss),video_count=len(indices))
        full,audit6=temporal_metrics(records,preds,metadata,True)
        abnormal,_=temporal_metrics(records,preds,metadata,False)
        seven_preds={n:dict(anomalies=p['detections']) for n,p in preds.items()}
        seven,audit7=temporal_metrics(records,seven_preds,metadata,True,normal_as_class=True)
        for c in XD_CLASSES:
            if full['per_class'][c]!=seven['per_class'][c]:raise AssertionError('Six-class metrics changed')
        predicted=[{e['anomaly_type'] for e in preds[n]['anomalies']} or {'Normal'} for n in names]
        video=binary_metrics([s!={'Normal'} for s in expected],[s!={'Normal'} for s in predicted],
                             [float(np.max(scores[slice(*offsets[n])])) for n in names])
        cats=category_metrics(expected,predicted)
        reports[arm]=dict(frame_AP=global_scores['AP'],frame_AUC=global_scores['AUC'],frame_per_category=per,
            temporal_mAP_full=full,temporal_mAP_abnormal=abnormal,temporal_mAP_7class=seven,
            video_metrics=video,category_metrics=cats,
            normal_false_alarm_count=video['fp'],normal_video_count=sum(s=={'Normal'} for s in expected),
            total_predictions=sum(len(p['anomalies']) for p in preds.values()),
            per_video=[dict(video_id=n,gt_categories=sorted(expected[i]),predicted_categories=sorted(predicted[i]),
                            category_set_correct=expected[i]==predicted[i],predicted_event_count=len(preds[n]['anomalies']),
                            processing_time=processing[n]) for i,n in enumerate(names)])
        save(output/'metadata'/('matching_6class_'+arm+'.json'),audit6)
        save(output/'metadata'/('matching_7class_'+arm+'.json'),audit7)
        log_report(arm,reports[arm])
    reasons={}
    for trace in traces:
        for row in trace['candidates'].values():
            reason=row.get('reason','unknown');reasons[reason]=reasons.get(reason,0)+1
    return dict(dataset='xd_violence',video_count=len(names),arms=reports,
        mcts=dict(searched=sum(t['searched_candidates'] for t in traces),applied=sum(t['applied_candidates'] for t in traces),reasons=reasons),
        protocol=dict(scope='full local XD test split',video_count=len(names),native_fps=True,vfr_verified=False,
                      shared_upstream=True,shared_mcts=True,thresholds=[.1,.2,.3,.4,.5],
                      frame_AP='sklearn.average_precision_score on concatenated source frames',
                      primary_temporal_classes=list(ALL_CLASSES),
                      normal_prediction='saved is_normal -> whole video; confidence 1 - mean source-frame anomaly score',
                      normal_gt='whole source video only for GT-Normal videos',
                      frame_category_subsets='binary anomaly metrics on videos containing each GT class; subsets overlap',
                      category_accuracy='exact predicted/GT label-set match; per-class counts are TP/GT',
                      annotation_sha256=annotation_hashes))
