"""Reproduce frozen XD discoveries and test independent event review in pairs.

Samples videos per anomaly class or selects the complete abnormal test split,
and compares new inference with the exact v220 control on the same videos.
"""
import argparse
import asyncio
import copy
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

SOURCE=Path(__file__).resolve().parent
PROJECT=SOURCE.parents[1]
for p in (SOURCE,PROJECT):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from config import VADConfig
from xd_config import XD_CLASSES
from xd_dataset import load_xd_index,select_xd_videos,annotation_fingerprint
from xd_candidate_verifier import verify_candidates,rescore_verified,VerificationResponseError
from xd_mcts_ablation import shared_mcts,save_json,read_json,sha
from xd_postprocess import make_arms,XDTailConfig
from xd_evaluate import temporal_metrics
from xd_batch_metrics import ARMS
from xd_full_ablation import batch_summary
from video_preprocessor import VideoPreprocessor
from llm_client import DashScopeLLMClient
from llm_usage import track_batch_usage,track_video_usage,video_usage_all_attempts
from api_recovery import ensure_api_success,run_video_jobs,is_retryable
from experiment import prepare_manifest,request_cache
from evaluation_tools.reevaluate_xd_seven_class import add_normal_prediction
import vad_logger

log=vad_logger.get('vad.xd.verification_eval')


def select_records(dataset_dir,per_class,seed,all_abnormal=False,expected_abnormal=500):
    """GT determines evaluation membership only, never any inference decision."""
    all_records=load_xd_index(dataset_dir)
    if all_abnormal:
        records={n:r for n,r in sorted(all_records.items()) if r['events']}
        if len(records)!=expected_abnormal:
            raise ValueError(f'Expected {expected_abnormal} abnormal videos, found {len(records)}')
        slots={c:[n for n,r in records.items() if any(e['category']==c for e in r['events'])]
               for c in XD_CLASSES}
        selection=dict(mode='all_abnormal',unique_video_count=len(records),slots=slots,
            actual_event_class_coverage={c:len(v) for c,v in slots.items()},
            include_normal_videos=False,normal_as_class=False)
    else:
        table=select_xd_videos(dataset_dir,per_class,seed=seed,categories=XD_CLASSES)
        selection=table.attrs['selection']
        records={n:all_records[n] for n in sorted(table['video_id'])}
    return records,selection


class VerificationReplay:
    def __init__(self,cfg,client,source,output,fingerprint,metadata):
        self.config=cfg;self.client=client;self.source=source;self.output=output
        self.fingerprint=fingerprint;self.metadata=metadata;self.caches={}
        self.decode_pool=asyncio.Semaphore(2)

    def decode(self,path):
        p=VideoPreprocessor(fps=self.config.fps,segment_duration=self.config.segment_duration,
            motion_threshold=0.,min_motion_interval=self.config.min_motion_interval,
            segment_overlap=self.config.segment_overlap)
        p.max_frame_side=self.config.xd.max_frame_side
        return p.process(path)

    @track_video_usage
    @ensure_api_success
    async def detect(self,video_path):
        started=time.perf_counter();name=Path(video_path).stem
        original=read_json(self.source/'result'/f'{name}.json')
        async with self.decode_pool:
            data=await asyncio.to_thread(self.decode,video_path)
        times=np.asarray(data['timestamps'],dtype=float)
        if (times.shape!=np.asarray(original['timestamps']).shape or
                not np.allclose(times,original['timestamps'],rtol=0,atol=1e-8) or
                abs(data['duration']-original['duration'])>1e-8 or
                data['video_fps']!=self.metadata[name]['fps']):
            raise ValueError('Frozen feature/evidence timebase mismatch')
        curves={c:np.asarray(original['shared']['class_curves'][c]) for c in XD_CLASSES}
        prior=original['category_prior'];seeds=original['shared']['seed_events']
        scorer=SimpleNamespace(get_category_scores=lambda video,c:curves[c])
        token=request_cache.set(self.caches.setdefault(video_path,{}))
        checkpoint=self.output/'metadata/verification'/f'{name}.json'
        try:
            verify_started=time.perf_counter()
            if checkpoint.exists():
                saved=read_json(checkpoint)
                if saved['experiment_fingerprint']!=self.fingerprint:raise ValueError('Verification checkpoint mismatch')
                reviewed,audit=saved['events'],saved['audit']
                log.info('[XD-VERIFY] reused completed review: %s',name)
            else:
                reviewed,audit=await verify_candidates(seeds,data,self.client,self.config.xd)
                save_json(checkpoint,dict(events=reviewed,audit=audit,experiment_fingerprint=self.fingerprint))
            reviewed=rescore_verified(reviewed,times,curves,prior,data['duration'])
            verify_seconds=time.perf_counter()-verify_started
            search_started=time.perf_counter()
            searched,trace=await shared_mcts(reviewed,data,self.config.mcts,self.client,scorer)
            search_seconds=time.perf_counter()-search_started
            arms=make_arms(searched,times,curves,data['duration'],XDTailConfig(nms_iou=self.config.xd.nms_iou))
            baseline=dict(anomalies=reviewed,is_normal=not reviewed)
            base=np.max(np.stack(list(curves.values())),axis=0)*self.config.scoring.clip_base_weight
            for row in [baseline,*arms.values()]:
                row['anomalies']=rescore_verified(row['anomalies'],times,curves,prior,data['duration'])
                scores=base.copy()
                for e in row['anomalies']:
                    mask=(times>=e['t_start'])&(times<e['t_end'])
                    scores[mask]=np.maximum(scores[mask],e['confidence'])
                row['frame_scores']=scores.tolist()
                detections,normal_audit=add_normal_prediction(row,times,self.metadata[name]['fps'],self.metadata[name]['frame_count'])
                row.update(detections=detections,normal_prediction_audit=normal_audit)
            return dict(schema='xd-verified-paired-v1',dataset='xd_violence',video_path=video_path,
                duration=data['duration'],source_fps=data['video_fps'],timestamps=times.tolist(),
                baseline_off=baseline,arms=arms,category_prior=prior,
                shared=dict(preverification_events=seeds,seed_events=reviewed,mcts_events=searched,
                    mcts_trace=trace,verification=audit,class_curves=original['shared']['class_curves']),
                source_result_sha256=sha(self.source/'result'/f'{name}.json'),
                experiment_fingerprint=self.fingerprint,processing_time=time.perf_counter()-started,
                timing=dict(verification_seconds=verify_seconds,mcts_seconds=search_seconds),
                token_scope='New review + new shared MCTS only; frozen v220 upstream not billed again')
        finally:
            request_cache.reset(token)


def evaluate(source,output,records,metadata):
    """Same GT, FPS, ranking AP and six classes for both sides; no Normal class."""
    names=sorted(records);predictions={side:{a:{} for a in ARMS} for side in ('control','verified')}
    temporal_scores={side:{a:[] for a in ARMS} for side in predictions}
    labels=[];review_counts=Counter();mcts=Counter()
    for name in names:
        gt=np.zeros(metadata[name]['frame_count'],dtype=np.uint8)
        for start,end in records[name]['segments']:gt[start:end]=1
        labels.append(gt)
        for side,folder in (('control',source),('verified',output)):
            row=read_json(folder/'result'/f'{name}.json')
            if side=='verified':
                review_counts.update(row['shared']['verification']['summary'])
                for k in ('searched_candidates','applied_candidates','logical_requests'):
                    mcts[k]+=row['shared']['mcts_trace'][k]
            for arm in ARMS:
                part=row['baseline_off'] if arm=='baseline_off' else row['arms'][arm]
                predictions[side][arm][name]={'anomalies':part['anomalies']}
                temporal_scores[side][arm].append(np.interp(np.arange(len(gt))/metadata[name]['fps'],row['timestamps'],part['frame_scores']))
    y=np.concatenate(labels);report={}
    for side in predictions:
        report[side]={}
        for arm,preds in predictions[side].items():
            metric,audit=temporal_metrics(records,preds,metadata,False)
            scores=np.concatenate(temporal_scores[side][arm])
            value=dict(temporal_mAP=metric,frame_AP=float(average_precision_score(y,scores)),
                frame_AUC=float(roc_auc_score(y,scores)),
                predictions=sum(len(p['anomalies']) for p in preds.values()),
                TP_iou05=sum(p['TP@0.5'] for p in metric['per_class'].values()),
                FP_iou05=sum(p['FP@0.5'] for p in metric['per_class'].values()))
            report[side][arm]=value
            save_json(output/'metadata/matching'/side/f'{arm}.json',audit)
            log.info('[XD-COMPARE] side=%s arm=%s videos=%d mAP=%.4f%% frame_AP=%.4f%% predictions=%d TP@.5=%d FP@.5=%d',
                side,arm,len(names),metric['percent']['AVG'],100*value['frame_AP'],value['predictions'],value['TP_iou05'],value['FP_iou05'])
    deltas={a:report['verified'][a]['temporal_mAP']['percent']['AVG']-report['control'][a]['temporal_mAP']['percent']['AVG'] for a in ARMS}
    return dict(video_count=len(names),results=report,delta_mAP_percentage_points=deltas,
        verification=review_counts,mcts=mcts,protocol=dict(classes=list(XD_CLASSES),thresholds=[.1,.2,.3,.4,.5],
        include_normal_videos=False,source_fps=True,source_experiment=str(source),
        comparison='Frozen same initial candidates, priors and curves; review and MCTS rerun only for verified side',
        limitation='No Normal videos or Normal AP; frozen-upstream paired comparison, not a fresh end-to-end benchmark'))


@track_batch_usage
async def run(args):
    source=Path(args.source).resolve();output=Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True);vad_logger.setup(output/'run.log')
    log.info('VAD batch evaluation started | output_dir=%s | log=%s',output,output/'run.log')
    records,selection=select_records(args.dataset_dir,args.per_class,args.seed,
        args.all_abnormal,args.expected_abnormal)
    names=sorted(records)
    log.info('Selected %d unique videos from 6 categories (mode=%s):',len(names),
             'all_abnormal' if args.all_abnormal else f'{args.per_class}_per_class')
    for c in XD_CLASSES:log.info('  %-20s: slots=%d actual_multilabel_coverage=%d',c,len(selection['slots'][c]),selection['actual_event_class_coverage'][c])
    source_manifest=read_json(source/'metadata/manifest.json')
    annotations=annotation_fingerprint(args.dataset_dir)
    if annotations!=source_manifest['runtime_settings']['annotation_sha256']:raise ValueError('GT changed')
    source_fps=read_json(source/'metadata/video_fps.json')['videos'];metadata={n:source_fps[n] for n in names}
    inputs={n:sha(source/'result'/f'{n}.json') for n in names}
    for n in names:
        stat=Path(records[n]['video_path']).stat()
        if dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)!=source_manifest['runtime_settings']['source_video_stat'][n]:
            raise ValueError('Source video changed since frozen feature extraction: '+n)
    cfg=VADConfig.load(source/'metadata/config.json');cfg.xd.verify_candidates=True
    cfg.mcts.acceptance_policy='relaxed_node';cfg.mcts.comparison_gate='best_valid'
    reuse=Path(args.reuse_validated).resolve() if args.reuse_validated else None
    reusable={n:sha(reuse/'result'/f'{n}.json') for n in names
              if reuse and (reuse/'result'/f'{n}.json').exists()}
    pool=asyncio.Semaphore(args.api_concurrency)
    client=DashScopeLLMClient(model=args.model,max_tokens=4096,max_frames=32,
        api_semaphore=pool,api_concurrency=args.api_concurrency,api_timeout_seconds=120.)
    try:
        manifest=prepare_manifest(output/'metadata/manifest.json',cfg,[client],names,runtime_settings=dict(
            concurrency=args.concurrency,api_concurrency=args.api_concurrency,selection=selection,
            annotation_sha256=annotations,source_experiment=str(source),input_results_sha256=inputs,
            test_protocol='same frozen v220 discoveries, six anomaly classes, no Normal videos',
            ranking_weights=dict(base=.6,clip=.25,local_mae=.15),uncertainty_weight=.5,
            schema_exhaustion='Retry broader context once, then retain unresolved group as uncertain; API errors still fail',
            reused_validated_source=str(reuse) if reuse else None,reused_validated_sha256=reusable))
        if reusable:
            previous=read_json(reuse/'metadata/manifest.json')
            if previous['config']!=cfg.to_dict():raise ValueError('Reuse configuration mismatch')
            changed={n for n,h in manifest['source_sha256'].items() if previous['source_sha256'].get(n)!=h}
            if changed-{'xd_candidate_verifier.py','xd_verification_eval.py'}:
                raise ValueError('Reuse changes more than exhausted-schema handling')
            same_verifier='xd_candidate_verifier.py' not in changed
            for n,h in reusable.items():
                row=read_json(reuse/'result'/f'{n}.json')
                if row['source_result_sha256']!=inputs[n]:raise ValueError('Reuse upstream mismatch')
                if not same_verifier and row['shared']['verification']['summary'].get('schema_fallback_groups',0):
                    raise ValueError('Only successfully parsed reviews may be reused')
                # Valid review parsing, scores and MCTS are unchanged; the new
                # branch executes only after exhausted structural repairs.
                for group in row['shared']['verification']['groups']:
                    allowed=('ok','unresolved_schema') if same_verifier else ('ok',)
                    if group['decision']['status'] not in allowed:raise ValueError('Cannot reuse failed review')
                prior_reuse=row.get('reused_validated_result')
                row['reused_validated_result']=dict(path=str(reuse/'result'/f'{n}.json'),sha256=h,
                    previous_fingerprint=row['experiment_fingerprint'],prior_reuse=prior_reuse,
                    reason=('Identical verifier, MCTS, postprocessing and configuration; expanded evaluation scope'
                            if same_verifier else 'Valid-response path unchanged; only schema-exhaustion fallback added'))
                row['experiment_fingerprint']=manifest['fingerprint']
                dest=output/'result'/f'{n}.json'
                if not dest.exists():save_json(dest,row)
            log.info('Reused %d validated results; additional videos require new inference',len(reusable))
        for name in manifest['source_sha256']:
            dest=output/'metadata/source_snapshot/video_anomaly_detection'/name
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/name,dest)
        save_json(output/'metadata/config.json',cfg.to_dict());save_json(output/'metadata/selection.json',selection)
        save_json(output/'metadata/video_fps.json',dict(videos=metadata))
        save_json(output/'metadata/command.json',dict(argv=sys.argv,python=sys.executable))
        replay=VerificationReplay(cfg,client,source,output,manifest['fingerprint'],metadata)
        jobs=[dict(video_id=n,video_path=records[n]['video_path'],idx=i+1) for i,n in enumerate(names)]
        video_pool=asyncio.Semaphore(args.concurrency);started=time.perf_counter();latest={}
        log.info('Starting batch processing: %d videos, concurrency=%d, API concurrency=%d',len(jobs),args.concurrency,args.api_concurrency)
        def record(e):
            latest[e['video_id']]=e
            with (output/'metadata/video_attempts.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(e,ensure_ascii=False)+'\n')
            save_json(output/'metadata/run_status.json',dict(status='running',total=len(jobs),finished=len(latest),
                succeeded=sum(x['status']=='succeeded' for x in latest.values()),
                skipped=sum(x['status']=='skipped' for x in latest.values()),
                failed=sum(x['status']=='failed' for x in latest.values()),wall_seconds=time.perf_counter()-started))
        async def execute(job,attempt):
            async with video_pool:
                n=job['video_id'];path=job['video_path'];dest=output/'result'/f'{n}.json'
                if dest.exists():
                    if read_json(dest)['experiment_fingerprint']!=manifest['fingerprint']:raise ValueError('Result fingerprint mismatch')
                    return dict(status='skipped',elapsed=0.,retryable=False,error_type=None)
                log.info('[%d/%d] START %s attempt=%d',job['idx'],len(jobs),n,attempt)
                t=time.perf_counter()
                try:
                    result=await replay.detect(video_path=path)
                    result.update(diagnosis_attempt=attempt,token_usage_all_attempts=video_usage_all_attempts(path))
                    save_json(dest,result);replay.caches.pop(path,None)
                    log.info('[%d/%d] DONE %s elapsed=%.1fs verification=%s',job['idx'],len(jobs),n,time.perf_counter()-t,result['shared']['verification']['summary'])
                    return dict(status='succeeded',elapsed=time.perf_counter()-t,retryable=False,error_type=None)
                except Exception as error:
                    if hasattr(error,'diagnostics'):save_json(output/'metadata/failed_responses'/f'{n}_{attempt}.json',error.diagnostics)
                    retryable=is_retryable(error) or isinstance(error,VerificationResponseError)
                    if isinstance(error,VerificationResponseError):replay.caches.pop(path,None)
                    log.exception('[%d/%d] FAIL %s error=%s retryable=%s',job['idx'],len(jobs),n,type(error).__name__,retryable)
                    return dict(status='failed',elapsed=time.perf_counter()-t,retryable=retryable,error_type=type(error).__name__)
        status,history=await run_video_jobs(jobs,execute,1,5.,15.,record)
        timing=batch_summary(status,history,time.perf_counter()-started)
        save_json(output/'metadata/run_status.json',dict(timing,status='failed' if timing['failed'] else 'evaluating'))
        if timing['failed']:raise RuntimeError(f"{timing['failed']} video(s) failed; no incomplete-scope success report")
        if inputs!={n:sha(source/'result'/f'{n}.json') for n in names}:raise ValueError('Control inputs changed')
        report=evaluate(source,output,records,metadata);report['timing']=timing
        report['protocol']['scope']=f'all {len(records)} abnormal test videos' if args.all_abnormal else 'sampled abnormal videos'
        report['protocol']['reused_verified_videos']=len(reusable)
        save_json(output/'evaluation_report.json',report)
        save_json(output/'metadata/run_status.json',dict(timing,status='completed'))
        log.info('Full report saved to: %s',output/'evaluation_report.json')
        return report
    finally:
        await client.client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=str(PROJECT/'vad_v3/experiments/v220'))
    parser.add_argument('--output_dir',default=str(PROJECT/'vad_v3/experiments/v221'))
    parser.add_argument('--dataset_dir',default='F:/dataset/xd-violence')
    parser.add_argument('--per_class',type=int,default=10)
    parser.add_argument('--all_abnormal',action='store_true',help='Evaluate every abnormal test video, excluding Normal videos and Normal AP')
    parser.add_argument('--expected_abnormal',type=int,default=500)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--concurrency',type=int,default=4)
    parser.add_argument('--api_concurrency',type=int,default=8)
    parser.add_argument('--model',default='qwen3-vl-plus')
    parser.add_argument('--reuse_validated',help='Reuse compatible completed results with matching inference code and configuration')
    args=parser.parse_args()
    if min(args.per_class,args.concurrency,args.api_concurrency,args.expected_abnormal)<1:parser.error('Invalid budget')
    asyncio.run(run(args))
