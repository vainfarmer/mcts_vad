"""Complete XD test split: one upstream pass + one MCTS search + five tails.

Writes resumable per-video results and UCF-style lifecycle/metric summaries.
GT is used only for selection, logging and evaluation, never inference prompts.
"""
import argparse
import asyncio
import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time
import uuid
import numpy as np

SOURCE=Path(__file__).resolve().parent
PROJECT=SOURCE.parents[1]
for path in (SOURCE,PROJECT):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from config import VADConfig
from xd_config import XD_CLASSES
from xd_dataset import load_xd_index,annotation_fingerprint
from xd_pipeline import XDMultiAnomalyVAD,XDResponseError,choose_windows,refine_clip_boundary,classwise_postprocess,PROMPT_VERSION
from xd_postprocess import ARMS,XDTailConfig,make_arms
from xd_candidate_verifier import verify_candidates,rescore_verified,VerificationResponseError
from xd_mcts_ablation import shared_mcts,save_json,read_json,sha
from xd_batch_metrics import evaluate_full
from llm_client import DashScopeLLMClient
from llm_usage import track_batch_usage,track_video_usage,video_usage_all_attempts
from api_recovery import ensure_api_success,gather_video_tasks,run_video_jobs,is_retryable
from experiment import request_cache,prepare_manifest
from evaluation_tools.video_fps import source_video_metadata
from evaluation_tools.reevaluate_xd_seven_class import add_normal_prediction,POLICY
import vad_logger

log=vad_logger.get('vad.xd.full')


def selection_summary(records):
    slots={c:[n for n,r in sorted(records.items()) if
              (r['category']=='Normal' if c=='Normal' else any(e['category']==c for e in r['events']))]
           for c in (*XD_CLASSES,'Normal')}
    return dict(mode='full_test_split',unique_video_count=len(records),slots=slots,
                actual_event_class_coverage={c:len(v) for c,v in slots.items()},
                note='A multilabel video appears in multiple category counts but is diagnosed once')


class XDFullPipeline:
    def __init__(self,config,detect_client,classify_client,output_dir,fingerprint,metadata):
        self.config=config
        self.vad=XDMultiAnomalyVAD(detect_client,config,classify_client)
        self.feature_lock=asyncio.Lock()
        self.output=Path(output_dir)
        self.fingerprint=fingerprint
        self.metadata=metadata
        self.tail=XDTailConfig(nms_iou=config.xd.nms_iou)
        self.request_caches={}
        log.info('[VideoMAE] XD multilabel model loaded on %s',config.device)
        log.info('[XD-MODELS] UCF learned prompt weights=False, UCF MIL=False; six XD class prompts + Normal prompts')

    def prepare_data(self,video_path,need_prior):
        data=self.vad.preprocessor.process(video_path)
        if len(data['timestamps'])<2:raise ValueError('Too few decoded frames')
        meta=self.metadata[Path(video_path).stem]
        if abs(data['duration']-meta['duration'])>1e-6 or abs(data['video_fps']-meta['fps'])>1e-6:
            raise ValueError('Source video timebase changed after preflight')
        data['clip_features']=self.vad.anomaly_expert.precompute_video_features(data['all_frames'])
        curves={c:self.vad.frame_scorer.get_category_scores(data,c) for c in XD_CLASSES}
        prior=self.vad.videomae.predict(data['all_frames'],data['timestamps'],data['duration']) if need_prior else None
        return data,curves,prior

    @track_video_usage
    @ensure_api_success
    async def detect(self,video_path):
        started=time.perf_counter();name=Path(video_path).stem
        checkpoint_path=self.output/'metadata/upstream'/f'{name}.json'
        checkpoint=read_json(checkpoint_path) if checkpoint_path.exists() else None
        if checkpoint and checkpoint.get('experiment_fingerprint')!=self.fingerprint:
            raise ValueError('Upstream checkpoint fingerprint mismatch')
        token=request_cache.set(self.request_caches.setdefault(video_path,{}))
        try:
            # Only the shared GPU preprocessing/encoding/prior pass is serialized.
            # CPU/API work for another video continues while it runs in a thread.
            async with self.feature_lock:
                data,curves,prior=await asyncio.to_thread(self.prepare_data,video_path,checkpoint is None)
            times=np.asarray(data['timestamps'],dtype=float)
            feature_seconds=time.perf_counter()-started
            if checkpoint:
                if checkpoint['timestamps']!=times.tolist():raise ValueError('Checkpoint timebase mismatch')
                prior=checkpoint['category_prior'];inspections=checkpoint['inspections']
                raw=checkpoint['raw_events'];refined=checkpoint['clip_events'];off=checkpoint['seed_events']
                log.info('[XD-UPSTREAM] reused completed checkpoint video=%s',name)
            else:
                windows=choose_windows(prior,curves,times,self.config.xd)
                log.info('[XD-DISCOVERY] video=%s windows=%d duration=%.2fs',name,len(windows),data['duration'])
                inspected=await gather_video_tasks(*(self.vad.inspect_window(w,data) for w in windows))
                inspections=[audit for _,audit in inspected]
                raw=[e for events,_ in inspected for e in events]
                refined=[refine_clip_boundary(e,times,curves[e['anomaly_type']],self.config.xd,data['duration']) for e in raw]
                off=classwise_postprocess(refined,self.config.xd.nms_iou)
                for i,e in enumerate(off):e['source_candidate_ids']=[i]
                # Successful upstream diagnosis persists before MCTS, so a later
                # transient API failure does not repeat all classification calls.
                save_json(checkpoint_path,dict(experiment_fingerprint=self.fingerprint,video_path=video_path,
                    timestamps=times.tolist(),category_prior=prior,inspections=inspections,
                    raw_events=raw,clip_events=refined,seed_events=off))
            preverification=copy.deepcopy(off);verification=None
            if self.config.xd.verify_candidates:
                off,verification=await verify_candidates(off,data,self.vad.classify_llm_client,self.config.xd)
                off=rescore_verified(off,times,curves,prior,data['duration'])
            search_started=time.perf_counter()
            searched,trace=await shared_mcts(off,data,self.config.mcts,self.vad.llm_client,self.vad.frame_scorer)
            search_seconds=time.perf_counter()-search_started
            tail_started=time.perf_counter()
            arms=make_arms(searched,times,curves,data['duration'],self.tail)
            if self.config.xd.verify_candidates:
                for row in arms.values():
                    row['anomalies']=rescore_verified(row['anomalies'],times,curves,prior,data['duration'])
            base=np.max(np.stack(list(curves.values())),axis=0)*self.config.scoring.clip_base_weight
            def decorate(row):
                scores=base.copy()
                for e in row['anomalies']:
                    mask=(times>=e['t_start']) & (times<e['t_end'])
                    scores[mask]=np.maximum(scores[mask],e['confidence'])
                row['frame_scores']=scores.tolist()
                meta=self.metadata[name]
                detections,audit=add_normal_prediction(row,times,meta['fps'],meta['frame_count'])
                row.update(detections=detections,normal_prediction_audit=audit)
            baseline=dict(anomalies=copy.deepcopy(off),is_normal=not off)
            for row in [baseline,*arms.values()]:decorate(row)
            return dict(schema='xd-paired-full-v1',dataset='xd_violence',video_path=video_path,
                duration=data['duration'],source_fps=data['video_fps'],timestamps=times.tolist(),
                baseline_off=baseline,arms=arms,category_prior=prior,
                shared=dict(seed_events=off,mcts_events=searched,mcts_trace=trace,
                            preverification_events=preverification,verification=verification,
                            class_curves={c:v.tolist() for c,v in curves.items()},raw_events=raw,
                            clip_events=refined,inspections=inspections,prompt_version=PROMPT_VERSION),
                processing_time=time.perf_counter()-started,
                timing=dict(features_and_queue_seconds=feature_seconds,mcts_seconds=search_seconds,
                            tail_seconds=time.perf_counter()-tail_started),
                token_scope='Shared upstream + one MCTS search; all five tail arms share calls')
        finally:
            request_cache.reset(token)


def batch_summary(status,history,wall):
    succeeded=sum(r['status']=='succeeded' for r in status)
    skipped=sum(r['status']=='skipped' for r in status)
    failed=sum(r['status']=='failed' for r in status)
    processing=sum(r['elapsed'] for r in history)
    successful_time=sum(r['elapsed'] for r in history if r['status']=='succeeded')
    log.info('='*60);log.info('Batch complete:')
    log.info('  Succeeded: %d, Skipped: %d, Failed: %d',succeeded,skipped,failed)
    log.info('  Total wall-clock time:   %.1fs (%.1fmin)',wall,wall/60)
    log.info('  Total processing time:   %.1fs (%.1fmin)',processing,processing/60)
    log.info('  Avg per video:           %.1fs',successful_time/succeeded if succeeded else 0.)
    log.info('  Parallelism speedup:     %.2fx',processing/wall if wall else 0.)
    log.info('='*60)
    return dict(succeeded=succeeded,skipped=skipped,failed=failed,total=len(status),
                wall_clock_seconds=wall,total_processing_seconds=processing,
                avg_successful_video_seconds=successful_time/succeeded if succeeded else None,
                attempts=len(history),recovered_videos=sum(r['status']=='succeeded' and r['attempt']>1 for r in history))


def write_summary(output,report):
    lines=['# v220：完整 XD 测试集五组配对消融','',
        f"已评测 {report['video_count']} 个视频。每个视频共享一次上游检测和一次 MCTS；原生 FPS 时间轴。",'',
        '| 组别 | 帧 AP % | 六异常类 mAP AVG % | 七类 mAP AVG % | 视频准确率 % | 类别集合准确率 % |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in report['arms'].items():
        lines.append(f"| {name} | {100*r['frame_AP']:.3f} | {r['temporal_mAP_full']['percent']['AVG']:.3f} | {r['temporal_mAP_7class']['percent']['AVG']:.3f} | {100*r['video_metrics']['accuracy']:.3f} | {100*r['category_metrics']['exact_set_accuracy']:.3f} |")
    lines+=['',f"MCTS 搜索 {report['mcts']['searched']} 个候选，采用 {report['mcts']['applied']} 个。",'',
        'Normal 预测来自模型的 is_normal 判定，输出整段；置信度为 1 减源帧平均异常分数。Normal GT 仅覆盖真实 Normal 视频整段，异常视频背景不额外标为 Normal。七类指标与六类指标分别保留。',
        '帧级类别指标按真实类别的视频子集分组，使用二分类异常分数；多标签子集重叠。Normal-only 子集无正样本，AUC/AP 记 N/A。视频类别准确率使用预测集合与真实集合完全相等，而非单标签 top-1。',
        '本实验仍不是 VadCLIP 的 Normal 曲线提案复现；数据范围为本地完整 800 视频源时间轴。', '',
        '`result/<视频名>.json` 中 `arms.<组名>.anomalies` 为异常区间，`detections` 同时包含可选的 Normal 检测。`baseline_off` 是共享上游的 MCTS 关闭对照。',
        '`run.log` 含启动、类别数量、逐视频 START/DONE/FAIL/SKIP、MCTS Root/Sim 与批次四层指标；metadata 保存 FPS、配置、源码快照、请求 token 明细和匹配审计。']
    (Path(output)/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


@track_batch_usage
async def run(args):
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    log_file=output/'run.log';vad_logger.setup(log_file)
    log.info('VAD batch evaluation started | output_dir=%s | log=%s',output,log_file)
    records=load_xd_index(args.dataset_dir)
    if len(records)!=args.expected_videos:raise ValueError(f'Expected {args.expected_videos} videos, found {len(records)}')
    selection=selection_summary(records)
    log.info('Selected %d videos from %d categories:',len(records),len(selection['slots']))
    for c,rows in selection['slots'].items():log.info('  %-20s: %3d / %d',c,len(rows),len(rows))
    log.info('  Counts are multilabel coverage; unique abnormal=%d, Normal=%d',
             sum(bool(r['events']) for r in records.values()),sum(r['category']=='Normal' for r in records.values()))
    fps_path=output/'metadata/video_fps.json'
    if fps_path.exists():
        metadata=read_json(fps_path)['videos']
        if set(metadata)!=set(records):raise ValueError('FPS manifest video coverage mismatch')
    else:
        metadata=source_video_metadata({n:r['video_path'] for n,r in records.items()})
        save_json(fps_path,dict(schema='video-fps-v1',videos=metadata))
    for n,r in records.items():
        count=metadata[n]['frame_count']
        if any(not 0<=e['start_frame']<e['end_frame']<=count for e in r['events']):raise ValueError('Typed GT out of bounds')
        if any(not 0<=s<e<=count for s,e in r['segments']):raise ValueError('Binary GT out of bounds')
    if args.evaluate_only:
        manifest=read_json(output/'metadata/manifest.json')
        report=evaluate_full(output,records,metadata,manifest['fingerprint'],annotation_fingerprint(args.dataset_dir))
        save_json(output/'evaluation_report.json',report);write_summary(output,report)
        return report
    cfg=VADConfig.load(args.config)
    if cfg.dataset!='xd_violence' or cfg.mcts.mode!='refine' or cfg.mcts.acceptance_policy!='force_best' or cfg.mcts.action_set!='shrink_expand_shift':
        raise ValueError('Expected the fixed v219 XD force_best expand/shift configuration')
    api_pool=asyncio.Semaphore(args.api_concurrency)
    clients=[DashScopeLLMClient(model=model,max_tokens=2048,max_frames=32,api_semaphore=api_pool,
                api_concurrency=args.api_concurrency,api_timeout_seconds=120.) for model in (args.detect_model,args.classify_model)]
    extra_files=[PROJECT/'evaluation_tools/reevaluate_xd_seven_class.py',PROJECT/'evaluation_tools/temporal_ap.py',
                 PROJECT/'evaluation_tools/video_fps.py']
    source_stat={n:dict(size=Path(r['video_path']).stat().st_size,mtime_ns=Path(r['video_path']).stat().st_mtime_ns) for n,r in records.items()}
    manifest=prepare_manifest(output/'metadata/manifest.json',cfg,clients,list(records),
        runtime_settings=dict(concurrency=args.concurrency,api_concurrency=args.api_concurrency,
            retries=args.video_retry_rounds,selection=selection,source_video_stat=source_stat,
            fps_manifest_sha256=sha(fps_path),annotation_sha256=annotation_fingerprint(args.dataset_dir),
            normal_policy=POLICY,arms=ARMS,tail_config=asdict(XDTailConfig(nms_iou=cfg.xd.nms_iou)),
            extra_source_sha256={str(p.relative_to(PROJECT)):sha(p) for p in extra_files}))
    fingerprint=manifest['fingerprint']
    for name in manifest['source_sha256']:
        dest=output/'metadata/source_snapshot/video_anomaly_detection'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(SOURCE/name,dest)
    for p in extra_files:
        dest=output/'metadata/source_snapshot'/p.relative_to(PROJECT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    save_json(output/'metadata/config.json',cfg.to_dict());save_json(output/'metadata/selection.json',selection)
    save_json(output/'metadata/command.json',dict(argv=sys.argv,python=sys.executable))
    vad=XDFullPipeline(cfg,clients[0],clients[1],output,fingerprint,metadata)
    log.info('Starting batch processing: %d videos, concurrency=%d',len(records),args.concurrency)
    log.info('Detect model: %s, Classify model: %s',args.detect_model,args.classify_model)
    log.info('API concurrency: %d | action_set=%s acceptance=%s | five shared-search arms',args.api_concurrency,cfg.mcts.action_set,cfg.mcts.acceptance_policy)
    log.info('='*60)
    jobs=[dict(video_id=n,video_path=r['video_path'],category=','.join(r['video_labels']),idx=i+1) for i,(n,r) in enumerate(sorted(records.items()))]
    pool=asyncio.Semaphore(args.concurrency);started=time.perf_counter()
    attempt_path=output/'metadata'/('video_attempts_'+uuid.uuid4().hex+'.jsonl')
    latest={}
    def record(entry):
        latest[entry['video_id']]=entry
        with attempt_path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
        save_json(output/'metadata/run_status.json',dict(status='running',total=len(jobs),finished=len(latest),
            succeeded=sum(r['status']=='succeeded' for r in latest.values()),skipped=sum(r['status']=='skipped' for r in latest.values()),
            failed=sum(r['status']=='failed' for r in latest.values()),wall_seconds=time.perf_counter()-started,
            last_video=entry['video_id'],last_state=entry['status'],attempt_file=attempt_path.name))
    async def execute(job,attempt):
        async with pool:
            n,path,idx=job['video_id'],job['video_path'],job['idx'];dest=output/'result'/f'{n}.json'
            if dest.exists():
                if read_json(dest).get('experiment_fingerprint')!=fingerprint:raise ValueError('Existing result fingerprint mismatch')
                log.info('[%d/%d] SKIP %s (result exists)',idx,len(jobs),n)
                return dict(status='skipped',elapsed=0.,retryable=False,error_type=None)
            log.info('[%d/%d] START %s (%s) attempt=%d',idx,len(jobs),n,job['category'],attempt)
            t0=time.perf_counter()
            try:
                result=await vad.detect(video_path=path)
                elapsed=time.perf_counter()-t0
                result.update(experiment_fingerprint=fingerprint,diagnosis_attempt=attempt,
                              processing_time=elapsed,token_usage_all_attempts=video_usage_all_attempts(path))
                save_json(dest,result);vad.request_caches.pop(path,None)
                trace=result['shared']['mcts_trace']
                log.info('[%d/%d] DONE  %s: %.1fs video, %s anomalies by arm, %.1fs processing | MCTS searched=%d applied=%d',
                    idx,len(jobs),n,result['duration'],{a:len(r['anomalies']) for a,r in result['arms'].items()},elapsed,
                    trace['searched_candidates'],trace['applied_candidates'])
                return dict(status='succeeded',elapsed=elapsed,retryable=False,error_type=None)
            except Exception as error:
                elapsed=time.perf_counter()-t0
                if hasattr(error,'diagnostics'):
                    save_json(output/'metadata/failed_responses'/f'{n}_attempt{attempt}.json',error.diagnostics)
                # Schema exhaustion gets one final diagnosis retry as well as
                # transient API failures; cached successful requests are reused.
                retryable=is_retryable(error) or isinstance(error,(XDResponseError,VerificationResponseError))
                if isinstance(error,(XDResponseError,VerificationResponseError)):vad.request_caches.pop(path,None)
                log.error('[%d/%d] FAIL  %s: %s (%.1fs), retryable=%s',idx,len(jobs),n,type(error).__name__,elapsed,retryable,exc_info=True)
                return dict(status='failed',elapsed=elapsed,retryable=retryable,error_type=type(error).__name__)
    try:
        status,history=await run_video_jobs(jobs,execute,args.video_retry_rounds,5.,15.,record)
        timing=batch_summary(status,history,time.perf_counter()-started)
        save_json(output/'metadata/run_status.json',dict(timing,status='evaluating' if not timing['failed'] else 'failed',attempt_file=attempt_path.name))
        if timing['failed']:raise RuntimeError(f"{timing['failed']} videos failed; full metrics not computed")
        log.info('Running evaluation...')
        report=evaluate_full(output,records,metadata,fingerprint,annotation_fingerprint(args.dataset_dir))
        report['timing']=timing;save_json(output/'evaluation_report.json',report);write_summary(output,report)
        save_json(output/'metadata/run_status.json',dict(timing,status='completed',attempt_file=attempt_path.name))
        log.info('Full report saved to: %s',output/'evaluation_report.json')
        return report
    finally:
        for client in clients:await client.client.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_dir',default='F:/dataset/xd-violence')
    parser.add_argument('--output_dir',default=str(PROJECT/'vad_v3/experiments/v220'))
    parser.add_argument('--config',default=str(PROJECT/'vad_v3/experiments/v219/metadata/config.json'))
    parser.add_argument('--expected_videos',type=int,default=800)
    parser.add_argument('--concurrency',type=int,default=2)
    parser.add_argument('--api_concurrency',type=int,default=4)
    parser.add_argument('--detect_model',default='qwen3-vl-plus')
    parser.add_argument('--classify_model',default='qwen3-vl-plus')
    parser.add_argument('--video_retry_rounds',type=int,default=1)
    parser.add_argument('--evaluate_only',action='store_true')
    args=parser.parse_args()
    if min(args.concurrency,args.api_concurrency,args.expected_videos)<1 or args.video_retry_rounds<0:parser.error('Invalid budget')
    if args.evaluate_only:
        # Do not overwrite the API ledger for an offline reevaluation.
        asyncio.run(run.__wrapped__(args))
    else:
        asyncio.run(run(args))


if __name__=='__main__':main()
