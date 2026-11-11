"""Immutable experiment inputs and explicit class-aware evaluation protocols."""
from __future__ import annotations
import json
import os
import csv
import io
import time
import uuid
from pathlib import Path
import numpy as np
from .settings import ROOT, V3_DIR, DATASET_ROOT, CATEGORIES, atomic_json, digest, dumps, file_hash
from .results import validate_result
from evaluation_tools.video_fps import probe_video_metadata, source_video_metadata

PROTOCOLS = {
    'vadclip_native_fps': dict(ap='rank_ap',thresholds=[.1,.2,.3,.4,.5],
        prediction_interval='[int(start*fps),int(end*fps))',gt_end='exclusive',
        frame_auc_gt_end='exclusive',frame_interpolation='linear_edge_hold',
        frame_dtype='float64',aupr='trapezoidal_precision_recall',
        frame_count='source_metadata',fps_source='original_video_files',
        macro_classes=list(CATEGORIES),scope='abnormal_and_all_separately'),
    'report_historical_v1': dict(ap='rank_ap',thresholds=[.1,.2,.3,.4,.5],
        prediction_interval='[int(start*30),int(end*30))',gt_end='exclusive',
        frame_auc_gt_end='inclusive',frame_interpolation='linear_edge_hold',
        frame_dtype='float32',aupr='trapezoidal_precision_recall',
        frame_count='floor_prediction_duration_times_30',
        macro_classes=list(CATEGORIES),scope='abnormal_and_all_separately'),
}


def validate_bundle(bundle):
    if not isinstance(bundle,dict):
        raise ValueError('实验包必须是 JSON 对象。')
    gt,groups=bundle.get('ground_truth'),bundle.get('groups')
    if not isinstance(gt,dict) or not gt or not isinstance(groups,dict) or not groups:
        raise ValueError('实验需要非空 ground_truth 和 groups 对象。')
    if len(gt)>10000 or len(groups)>20:
        raise ValueError('实验规模超过本机导入上限。')
    for vid,row in gt.items():
        if not isinstance(row,dict):
            raise ValueError(f'{vid}: GT 必须是对象。')
        if row.get('category') not in (*CATEGORIES,'Normal'):
            raise ValueError(f'{vid}: GT 类别无效。')
        fps=row.get('fps')
        if type(fps) not in (float,int) or not np.isfinite(fps) or fps<=0:
            raise ValueError(f'{vid}: 必须提供实际 FPS。')
        if type(row.get('frame_count')) is not int or row['frame_count']<=0:
            raise ValueError(f'{vid}: 必须提供原始帧数。')
        if not isinstance(row.get('segments'),list):
            raise ValueError(f'{vid}: GT 必须提供 segments。')
        for seg in row['segments']:
            if (not isinstance(seg,list) or len(seg)!=2
                or any(type(x) is not int for x in seg) or not 0<=seg[0]<seg[1]):
                raise ValueError('GT 区间须为 0 起始、右端不包含的整数帧区间。')
        if row['category']=='Normal' and row['segments']:
            raise ValueError('Normal GT 不应包含异常区间。')
    for group,predictions in groups.items():
        if not isinstance(predictions,dict):
            raise ValueError('每组预测必须是以视频标识为键的对象。')
        if set(predictions)!=set(gt):
            missing=sorted(set(gt)-set(predictions))
            extra=sorted(set(predictions)-set(gt))
            raise ValueError(f'{group}: 预测覆盖不一致，缺失 {missing[:10]}，额外 {extra[:10]}。')
        for vid,result in predictions.items():
            # 输入包的 FPS 仅是历史记录；原生协议执行时会重新探测原视频并校验时长。
            validate_result(result)
    return bundle


def create_experiment(store,bundle):
    validate_bundle(bundle)
    eid=uuid.uuid4().hex
    relative=f'experiments/{eid}/input.json'
    atomic_json(store.path(relative),bundle)
    manifest=dict(input_path=relative,input_sha256=file_hash(store.path(relative)),
                  video_count=len(bundle['ground_truth']),groups=list(bundle['groups']),
                  provenance=bundle.get('provenance',{}),
                  annotation_unit='frame',frame_index_origin=0,end_convention='exclusive',
                  gt_out_of_bounds=[vid for vid,row in bundle['ground_truth'].items()
                      if any(end>row['frame_count'] for _,end in row['segments'])],
                  gt_policy='AP 保留原标注；帧级评价在实际视频帧范围截取。越界项单独列出。')
    with store.connect() as db:
        db.execute('INSERT INTO experiments VALUES(?,?,?,?,?)',
                   (eid,str(bundle.get('name','导入实验'))[:200],time.time(),dumps(manifest),'ready'))
    return dict(id=eid,manifest=manifest)


def from_jobs(store,data):
    """Join completed predictions to separately supplied GT, with explicit input identity."""
    if not isinstance(data,dict):
        raise ValueError('实验请求必须是 JSON 对象。')
    mappings=data.get('groups')
    supplied=data.get('ground_truth')
    if not isinstance(mappings,dict) or not isinstance(supplied,dict):
        raise ValueError('请选择任务组并上传 GT JSON。')
    gt,groups,identities,provenance={},{},{},{}
    for group,job_ids in mappings.items():
        if not isinstance(job_ids,list) or not job_ids:
            raise ValueError('每个配置组至少选择一个已完成任务。')
        groups[group]={};provenance[group]={}
        for jid in job_ids:
            job=store.job(jid);video=store.video(job['video_id'],deleted=True)
            if job['status']!='completed':
                raise ValueError('实验只接受已完成任务。')
            key=Path(video['filename']).stem
            if key in groups[group]:
                raise ValueError(f'{group}: 同名视频只能选择一个任务。')
            if key in identities and identities[key]!=video['sha256']:
                raise ValueError('同名视频内容不一致，请先修正输入身份。')
            identities[key]=video['sha256']
            source=supplied.get(key)
            if not isinstance(source,dict):
                raise ValueError(f'GT 缺少视频 {key}。')
            metadata=probe_video_metadata(store.path(video['path']))
            gt[key]=dict(source,fps=metadata['fps'],frame_count=metadata.get('frame_count'))
            groups[group][key]=store.result(jid)
            provenance[group][key]=dict(job_id=jid,attempt_id=job['result_attempt_id'],
                config_hash=job['config_hash'],code_hash=job['code_hash'],video_sha256=video['sha256'])
    if set(supplied)!=set(gt):
        raise ValueError('GT 和所选视频范围不一致。')
    return create_experiment(store,dict(name=data.get('name','任务对比实验'),ground_truth=gt,
        groups=groups,provenance=dict(kind='completed_system_jobs',jobs=provenance)))


def list_experiments(store):
    with store.connect() as db:
        experiments=[store.unpack(r) for r in db.execute('SELECT * FROM experiments ORDER BY created_at DESC')]
        for e in experiments:
            e['evaluations']=[store.unpack(r) for r in db.execute(
                'SELECT * FROM evaluations WHERE experiment_id=? ORDER BY created_at DESC',(e['id'],))]
    return experiments


def experiment(store,eid):
    with store.connect() as db:
        return store.unpack(db.execute('SELECT * FROM experiments WHERE id=?',(eid,)).fetchone())


def create_evaluation(store,eid,protocol_name='vadclip_native_fps'):
    exp=experiment(store,eid)
    if protocol_name not in PROTOCOLS:
        raise ValueError('不支持的评价协议。')
    protocol=dict(PROTOCOLS[protocol_name],name=protocol_name,
        evaluator_hash=digest({p.name:file_hash(p) for p in
            [Path(__file__),ROOT/'evaluation_tools/temporal_ap.py',ROOT/'evaluation_tools/video_fps.py']}))
    rid=uuid.uuid4().hex
    with store.connect() as db:
        db.execute("""INSERT INTO evaluations(id,experiment_id,protocol_json,input_hash,created_at,status)
            VALUES(?,?,?,?,?,'queued')""",(rid,eid,dumps(protocol),exp['manifest']['input_sha256'],time.time()))
    return rid


def _evaluation_video_metadata(store, bundle):
    """从受管理的视频或本机数据集读取 FPS，禁止把旧 JSON 的 FPS 当作原生值。"""
    provenance = bundle.get('provenance', {})
    paths = {}
    if provenance.get('kind') == 'completed_system_jobs':
        sources = next(iter(provenance['jobs'].values()))
        for name in bundle['ground_truth']:
            job = store.job(sources[name]['job_id'], deleted=True)
            video = store.video(job['video_id'], deleted=True)
            if video['sha256'] != sources[name]['video_sha256']:
                raise ValueError(f'{name}: 原视频身份与实验记录不一致')
            paths[name] = store.path(video['path'])
    else:
        from vad_v3.video_anomaly_detection.dataset_index import resolve_video
        # 在请求时读取，确保 Web 启动时载入的 .env 配置也生效。
        dataset_root = Path(os.getenv('VAD_DATASET_DIR', str(DATASET_ROOT)))
        predictions = next(iter(bundle['groups'].values()))
        allowed_roots = (dataset_root.resolve(), (store.root / 'videos').resolve())
        for name, gt in bundle['ground_truth'].items():
            saved = predictions[name].get('video_path')
            if not saved:
                if Path(name).name != name:
                    raise ValueError('视频标识不能包含目录')
                saved = ('videos/Testing_Normal/' if gt['category'] == 'Normal' else '') + name + '.mp4'
            path = Path(saved).resolve()
            if not path.is_file():
                path = Path(resolve_video(dataset_root, saved)).resolve()
            # JSON 导入只允许引用本机数据集及已上传视频，不能借由路径读取任意本地文件。
            if not any(path.is_relative_to(root) for root in allowed_roots):
                raise ValueError(f'{name}: 原视频须位于本机数据集或系统视频目录')
            paths[name] = path
    return source_video_metadata(paths)


def evaluate(store,rid):
    with store.connect() as db:
        r=store.unpack(db.execute('SELECT * FROM evaluations WHERE id=?',(rid,)).fetchone())
        db.execute("UPDATE evaluations SET status='running' WHERE id=?",(rid,))
    try:
        exp=experiment(store,r['experiment_id'])
        path=store.path(exp['manifest']['input_path'])
        if file_hash(path)!=r['input_hash']:
            raise ValueError('实验输入已改变，拒绝覆盖原实验。')
        bundle=validate_bundle(json.loads(path.read_text(encoding='utf-8')))
        gt=bundle['ground_truth']
        metadata = {}
        if r['protocol']['name']=='report_historical_v1':
            fps=30.
        else:
            metadata=_evaluation_video_metadata(store,bundle)
            fps={v:x['fps'] for v,x in metadata.items()}
            # 只更新本次计算的副本，原始实验包和过去的指标始终保留。
            gt={v:dict(row,fps=metadata[v]['fps'],frame_count=metadata[v]['frame_count']) for v,row in gt.items()}
            for predictions in bundle['groups'].values():
                for v,result in predictions.items():
                    validate_result(result,metadata[v]['duration'])
        from evaluation_tools.temporal_ap import compute_vadclip_map
        from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, confusion_matrix, classification_report
        output=dict(schema_version=1,evaluation_id=rid,protocol=r['protocol'],
                    input_sha256=r['input_hash'],groups={},coverage=dict(expected=len(gt),completed=len(gt)),
                    source_video_metadata=metadata)
        audits={}
        for group,pred in bundle['groups'].items():
            abnormal,audit=compute_vadclip_map(gt,pred,fps)
            all_videos,all_audit=compute_vadclip_map(gt,pred,fps,include_normal_videos=True)
            audits[group]=dict(abnormal=audit,all_videos=all_audit)
            labels=list(CATEGORIES)+['Normal','unknown']
            actual,predicted=[],[]
            y_parts,s_parts=[],[]
            missing=[]
            per_video={}
            for vid,row in gt.items():
                actual.append(row['category'])
                detections=pred[vid]['anomalies']
                votes={}
                for d in detections:
                    votes[d['anomaly_type']]=votes.get(d['anomaly_type'],0)+(d['t_end']-d['t_start'])*d['confidence']
                predicted.append(max(votes,key=votes.get) if votes else 'Normal')
                per_video[vid]=dict(actual=row['category'],predicted=predicted[-1],
                    correct=row['category']==predicted[-1],detection_count=len(detections),auc=None)
                scores,times=pred[vid].get('frame_scores'),pred[vid].get('timestamps')
                if not scores or not times:
                    missing.append(vid)
                    continue
                frame_fps=30. if isinstance(fps,float) else fps[vid]
                total_frames=(int(pred[vid]['duration']*frame_fps)
                    if r['protocol'].get('frame_count')=='floor_prediction_duration_times_30'
                    else row['frame_count'])
                if r['protocol'].get('frame_dtype')=='float32':
                    values=np.interp(np.arange(total_frames),np.asarray(times)*frame_fps,
                                     np.asarray(scores,dtype=np.float32)).astype(np.float32)
                else:
                    axis=np.arange(total_frames)/frame_fps
                    values=np.interp(axis,np.asarray(times),np.asarray(scores))
                target=np.zeros(total_frames,dtype=np.uint8)
                for start,end in row['segments']:
                    if r['protocol']['frame_auc_gt_end']=='inclusive': end+=1
                    target[max(0,start):min(end,len(target))]=1
                y_parts.append(target);s_parts.append(values)
                if len(np.unique(target))==2:
                    per_video[vid]['auc']=float(roc_auc_score(target,values))
            frame=dict(auc=None,aupr=None,missing_videos=missing,
                reason='曲线覆盖不完整，未计算汇总指标' if missing else None)
            if not missing:
                yy,ss=np.concatenate(y_parts),np.concatenate(s_parts)
                if len(np.unique(yy))==2:
                    precision,recall,_=precision_recall_curve(yy,ss)
                    frame.update(auc=float(roc_auc_score(yy,ss)),aupr=float(auc(recall,precision)),
                                 average_precision=float(average_precision_score(yy,ss)))
                else:
                    frame['reason']='帧标签仅含一个类别，ROC-AUC 无定义'
            normal=[i for i,x in enumerate(actual) if x=='Normal']
            output['groups'][group]=dict(abnormal=abnormal,all_videos=all_videos,frame=frame,
                per_video=per_video,
                classification=dict(labels=labels,confusion_matrix=confusion_matrix(actual,predicted,labels=labels).tolist(),
                    report=classification_report(actual,predicted,labels=labels,output_dict=True,zero_division=0),
                    accuracy=sum(a==b for a,b in zip(actual,predicted))/len(actual),
                    normal_false_positive_rate=(sum(predicted[i]!='Normal' for i in normal)/len(normal)) if normal else None))
            timings=[float(p['processing_time']) for p in pred.values()
                     if type(p.get('processing_time')) in (int,float) and np.isfinite(p['processing_time'])]
            usages=[p.get('token_usage') for p in pred.values() if isinstance(p.get('token_usage'),dict)]
            output['groups'][group]['cost']=dict(
                timing_coverage=len(timings),video_count=len(pred),
                median_seconds=float(np.median(timings)) if timings else None,
                p95_seconds=float(np.percentile(timings,95)) if timings else None,
                usage_coverage=len(usages),
                total_tokens=(sum(u['total_tokens'] for u in usages if type(u.get('total_tokens')) is int)
                    if any(type(u.get('total_tokens')) is int for u in usages) else None),
                requests=(sum(u['requests'] for u in usages if type(u.get('requests')) is int)
                    if any(type(u.get('requests')) is int for u in usages) else None),
                note='配对实验共享上游；各分支用量如未单独计量，不可解释为独立运行成本。')
        relative=f"experiments/{exp['id']}/evaluations/{rid}/metrics.json"
        atomic_json(store.path(relative),output)
        atomic_json(store.path(relative).with_name('matching_audit.json'),audits)
        with store.connect() as db:
            db.execute("UPDATE evaluations SET status='completed',metrics_path=? WHERE id=?",(relative,rid))
    except Exception as exc:
        with store.connect() as db:
            db.execute("UPDATE evaluations SET status='failed',error=? WHERE id=?",(str(exc),rid))


def metrics_csv(metrics):
    stream=io.StringIO(newline='')
    writer=csv.writer(stream)
    writer.writerow(['group','scope','metric','value'])
    for name,group in metrics['groups'].items():
        safe_name="'"+name if name.startswith(('=','+','-','@')) else name
        for area in ('abnormal','all_videos'):
            for key,value in group[area]['values'].items():
                writer.writerow([safe_name,area,key,value])
        for key in ('auc','aupr','average_precision'):
            writer.writerow([safe_name,'frame',key,group['frame'].get(key)])
        for key in ('accuracy','normal_false_positive_rate'):
            writer.writerow([safe_name,'video_classification',key,group['classification'].get(key)])
        for category,values in group['abnormal']['per_class'].items():
            for key,value in values.items():
                if isinstance(value,(int,float)):
                    writer.writerow([safe_name,'class:'+category,key,value])
    return '\ufeff'+stream.getvalue()


def historical_bundle():
    """Only the repository's known archive; never accepts a browser-supplied path."""
    from vad_v3.video_anomaly_detection.dataset_index import resolve_video
    dataset_root=Path(os.getenv('VAD_DATASET_DIR', str(DATASET_ROOT)))
    labels_path=dataset_root/'data_parquet/test-00000-of-00001.parquet'
    if not labels_path.exists():
        raise FileNotFoundError('历史 GT 文件未挂载。可使用 JSON 实验包导入。')
    import pandas as pd
    gt={}
    for row in pd.read_parquet(labels_path).to_dict('records'):
        name=Path(row['video_path']).stem
        meta=probe_video_metadata(resolve_video(dataset_root,row['video_path']))
        gt[name]=dict(category=row['category'],fps=meta['fps'],frame_count=meta['frame_count'],
            segments=[[int(row[f'annotation_start{i}']),int(row[f'annotation_end{i}'])]
                      for i in (1,2) if row[f'annotation_start{i}']>=0 and row[f'annotation_end{i}']>=0])
    groups={'MCTS on':{},'MCTS off':{}}
    hashes={}
    for path in sorted((V3_DIR/'experiments/v217/results').glob('*.json')):
        obj=json.loads(path.read_text(encoding='utf-8'))
        if 'video_path' not in obj or 'anomalies' not in obj:
            continue
        name=Path(obj['video_path']).stem
        groups['MCTS on'][name]=obj
        groups['MCTS off'][name]=obj['paired_results']['original_off']
        hashes[path.name]=file_hash(path)
    return dict(name='v217 · 历史配对实验',ground_truth=gt,groups=groups,
        provenance=dict(kind='historical_predictions',source='vad_v3/experiments/v217/results',
                        prediction_hashes=hashes,gt_sha256=file_hash(labels_path)))
