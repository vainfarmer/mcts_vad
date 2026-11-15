"""Replay frozen XD discoveries, search once, evaluate five paired tail arms.

The source experiment is read-only. Each result/<video_id>.json contains ALL
arms; a result never mixes five independent stochastic MCTS executions.
API credentials come only from DASHSCOPE_API_KEY, as in the normal runner.
"""
import argparse
import asyncio
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

SOURCE = Path(__file__).resolve().parent
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from config import VADConfig
from mcts_config import MCTSConfig
from xd_config import XD_CLASSES
from xd_pipeline import XD_PROMPTS, NORMAL_PROMPTS
from xd_postprocess import ARMS, XDTailConfig, make_arms
from video_preprocessor import VideoPreprocessor
from anomaly_expert import VideoAnomalyExpert
from frame_scorer import FrameScorer
from mcts_refiner import MCTSBoundaryRefiner
from mcts_boundary_scorer import BoundaryEvidenceScorer
from llm_client import DashScopeLLMClient
from llm_usage import track_video_usage, track_batch_usage, video_usage_all_attempts
from api_recovery import ensure_api_success, run_video_jobs, is_retryable
from experiment import request_cache, prepare_manifest
from xd_dataset import load_xd_index, annotation_fingerprint
from xd_evaluate import temporal_metrics, source_video_metadata
import vad_logger

log = vad_logger.get('vad.xd.ablation')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prediction_signature(events, geometry_only=False):
    # Ignore metadata/provenance additions when counting changed predictions.
    fields = ('anomaly_type', 't_start', 't_end') + (() if geometry_only else ('confidence',))
    return sorted(tuple(round(e[k], 6) if isinstance(e[k], (float, int)) else e[k] for k in fields) for e in events)


def candidate_order(events, data, scorer, cfg):
    """Round-robin predicted classes, ranked by raw boundary contrast within class.

    Class priority uses prediction confidence only. Never use filename labels,
    annotation membership or GT duration for candidate allocation.
    """
    groups = {}
    for i, event in enumerate(events):
        category = event['anomaly_type']
        try:
            curve = scorer.get_category_scores(data, category)
            contrast = BoundaryEvidenceScorer(data, curve, cfg).quality(
                (event['t_start'], event['t_end']))['contrast_raw']
        except ValueError:
            contrast = float('inf')
        groups.setdefault(category, []).append((contrast, event['t_start'], i))
    classes = sorted(groups, key=lambda c: (-max(events[x[2]]['confidence'] for x in groups[c]), c))
    queues = {c: [x[2] for x in sorted(groups[c])] for c in classes}
    return [queues[c][rank] for rank in range(max(map(len, queues.values()), default=0))
            for c in classes if rank < len(queues[c])]


async def shared_mcts(events, data, cfg, client, scorer):
    if any('verification' in e for e in events) and cfg.acceptance_policy == 'force_best':
        # Verification decides whether a candidate exists; search may only
        # replace its boundary with a node that still contains the same target.
        cfg = copy.deepcopy(cfg)
        cfg.acceptance_policy = 'relaxed_node'
    output = copy.deepcopy(events)
    order = candidate_order(events, data, scorer, cfg)
    trace = dict(mode=cfg.mode, action_set=cfg.action_set, acceptance_policy=cfg.acceptance_policy,
                 trigger_policy=cfg.trigger_policy, allocation='predicted_class_round_robin',
                 candidate_order=order, logical_requests=0, searched_candidates=0, candidates={})
    for i in order:
        event = events[i]
        category = event['anomaly_type']
        if event.get('verification', {}).get('status') == 'uncertain':
            trace['candidates'][str(i)] = dict(candidate_id=i, category=category,
                baseline=[event['t_start'], event['t_end']], applied=False,
                logical_requests=0, reason='verification_uncertain')
            continue
        remaining = cfg.max_vlm_evaluations_per_video-trace['logical_requests']
        if trace['searched_candidates'] >= cfg.max_candidates_per_video or remaining < 2:
            trace['candidates'][str(i)] = dict(candidate_id=i, category=category,
                baseline=[event['t_start'], event['t_end']], applied=False, logical_requests=0,
                reason='candidate_or_video_budget')
            continue
        local = copy.deepcopy(cfg)
        local.max_candidates_per_video = 1
        local.max_vlm_evaluations_per_video = remaining
        refiner = MCTSBoundaryRefiner(local, client, category_priors={category: {
            'expected_duration': (local.min_interval_seconds, data['duration'])}})
        outcome = await refiner.refine_candidates({i: event}, {i: event}, data, category, scorer)
        output[i] = outcome.candidates_by_id[i]
        sub = outcome.trace or {}
        used = sub.get('logical_requests', 0)
        record = sub.get('candidates', {}).get(str(i), dict(candidate_id=i, applied=False,
                     baseline=[event['t_start'], event['t_end']], reason=sub.get('skipped_reason', 'no_trace')))
        record.update(category=category, logical_requests=used)
        trace['candidates'][str(i)] = record
        trace['logical_requests'] += used
        trace['searched_candidates'] += int(used > 0)
    trace['applied_candidates'] = sum(r.get('applied', False) for r in trace['candidates'].values())
    return output, trace


class XDReplay:
    def __init__(self, cfg, client, tail_cfg):
        self.config, self.client, self.tail_cfg = cfg, client, tail_cfg
        self.preprocessor = VideoPreprocessor(fps=cfg.fps, segment_duration=cfg.segment_duration,
            motion_threshold=0., min_motion_interval=cfg.min_motion_interval, segment_overlap=cfg.segment_overlap)
        self.preprocessor.max_frame_side = cfg.xd.max_frame_side
        self.expert = VideoAnomalyExpert(list(XD_CLASSES), cfg.clip.model_name, cfg.clip.pretrained,
            cfg.device, prompt_overrides=XD_PROMPTS, normal_prompts=NORMAL_PROMPTS, use_learned_prompt_weights=False)
        if self.expert.model is None:
            raise RuntimeError('CLIP weights unavailable')
        self.scorer = FrameScorer(self.expert, cfg, list(XD_CLASSES))
        self.feature_lock = asyncio.Lock()
        self.caches = {}  # Successful exact requests survive a video API retry.

    def prepare_data(self, video_path):
        data = self.preprocessor.process(video_path)
        data['clip_features'] = self.expert.precompute_video_features(data['all_frames'])
        curves = {c: self.scorer.get_category_scores(data, c) for c in XD_CLASSES}
        return data, curves

    @track_video_usage
    @ensure_api_success
    async def detect(self, video_path, baseline):
        started = time.perf_counter()
        async with self.feature_lock:
            data, curves = await asyncio.to_thread(self.prepare_data, video_path)
        times = np.asarray(data['timestamps'], dtype=float)
        if (abs(data['duration']-baseline['duration']) > 1e-5 or
            len(times) != len(baseline['timestamps']) or
            not np.allclose(times, baseline['timestamps'], atol=1e-6, rtol=0)):
            raise ValueError('Upstream timebase changed')
        seeds = copy.deepcopy(baseline['diagnostics']['postprocess_off'])
        for i, seed in enumerate(seeds):
            seed['source_candidate_ids'] = [i]
        base = np.max(np.stack(list(curves.values())), axis=0)*self.config.scoring.clip_base_weight
        def frame_scores(events):
            scores = base.copy()
            for e in events:
                mask = (times >= e['t_start']) & (times < e['t_end'])
                scores[mask] = np.maximum(scores[mask], e['confidence'])
            return scores.tolist()
        max_error = float(np.max(np.abs(np.asarray(frame_scores(seeds))-baseline['frame_scores'])))
        if max_error > 1e-4:
            raise ValueError(f'Upstream CLIP/frame-score replay drift: {max_error}')
        pre_seconds = time.perf_counter()-started
        token = request_cache.set(self.caches.setdefault(video_path, {}))
        try:
            search_started = time.perf_counter()
            searched, trace = await shared_mcts(seeds, data, self.config.mcts, self.client, self.scorer)
            search_seconds = time.perf_counter()-search_started
        finally:
            request_cache.reset(token)
        tail_started = time.perf_counter()
        arms = make_arms(searched, times, curves, data['duration'], self.tail_cfg)
        for arm in arms.values():
            arm['frame_scores'] = frame_scores(arm['anomalies'])
        return dict(schema='xd-paired-tail-v1', dataset='xd_violence', video_path=video_path,
                    duration=data['duration'], timestamps=times.tolist(),
                    source_fps=data['video_fps'], upstream_frame_score_max_error=max_error,
                    baseline_off=dict(anomalies=baseline['anomalies'], frame_scores=baseline['frame_scores'],
                                      is_normal=baseline['is_normal']),
                    shared=dict(seed_events=seeds, mcts_events=searched, mcts_trace=trace,
                                class_curves={c: v.tolist() for c, v in curves.items()}),
                    arms=arms, processing_time=time.perf_counter()-started,
                    timing=dict(preprocess_and_queue_seconds=pre_seconds, mcts_seconds=search_seconds,
                                tail_seconds=time.perf_counter()-tail_started),
                    token_scope='Shared MCTS only; upstream v218 calls reused, not billed again')


def evaluate(args, names):
    """Read GT only after predictions; native-FPS tIoU and full-source-frame AP."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    root = Path(args.output_dir)
    all_records = load_xd_index(args.dataset_dir)
    records = {n: all_records[n] for n in names}
    metadata = source_video_metadata({n: r['video_path'] for n, r in records.items()})
    save_json(root/'metadata/video_fps.json', dict(schema='video-fps-v1', videos=metadata))
    results = {n: read_json(root/'result'/(n+'.json')) for n in names}
    labels = []
    for n, r in records.items():
        count = metadata[n]['frame_count']
        y = np.zeros(count, dtype=np.uint8)
        for start, end in r['segments']:
            if not 0 <= start < end <= count:
                raise ValueError('Binary GT exceeds video: '+n)
            y[start:end] = 1
        if any(not 0 <= e['start_frame'] < e['end_frame'] <= count for e in r['events']):
            raise ValueError('Typed GT exceeds video: '+n)
        labels.append(y)
    labels = np.concatenate(labels)
    reports = {}
    for arm in ('baseline_off', *ARMS):
        predictions = {n: (r['baseline_off'] if arm == 'baseline_off' else r['arms'][arm]) for n, r in results.items()}
        full, audit = temporal_metrics(records, predictions, metadata, True)
        abnormal, _ = temporal_metrics(records, predictions, metadata, False)
        frame_scores = np.concatenate([np.interp(np.arange(metadata[n]['frame_count'])/metadata[n]['fps'],
            results[n]['timestamps'], predictions[n]['frame_scores']) for n in names])
        operators = {}
        for n in names:
            for entry in predictions[n].get('audit', []):
                counts = operators.setdefault(entry['operator'], dict(checked=0, changed=0))
                counts['checked'] += 1; counts['changed'] += int(entry['applied'])
        reports[arm] = dict(frame_AP=float(average_precision_score(labels, frame_scores)),
            frame_AUC=float(roc_auc_score(labels, frame_scores)), temporal_mAP_full=full,
            temporal_mAP_abnormal=abnormal, total_predictions=sum(len(p['anomalies']) for p in predictions.values()),
            normal_false_alarm_count=sum(bool(predictions[n]['anomalies']) for n in names if not records[n]['events']),
            operators=operators,
            per_video=[dict(video_id=n, predicted_event_count=len(predictions[n]['anomalies']),
                changed_from_off=prediction_signature(predictions[n]['anomalies']) != prediction_signature(results[n]['baseline_off']['anomalies']),
                geometry_changed_from_off=prediction_signature(predictions[n]['anomalies'], True) != prediction_signature(results[n]['baseline_off']['anomalies'], True)) for n in names])
        save_json(root/'metadata'/('matching_'+arm+'.json'), audit)
    trace_records = [item for r in results.values() for item in r['shared']['mcts_trace']['candidates'].values()]
    reasons = {}
    for item in trace_records:
        reasons[item.get('reason', 'unknown')] = reasons.get(item.get('reason', 'unknown'), 0)+1
    report = dict(dataset='xd_violence', video_count=len(names), arms=reports,
                  mcts=dict(total_candidates=len(trace_records),
                            searched=sum(r.get('logical_requests', 0)>0 for r in trace_records),
                            applied=sum(r.get('applied', False) for r in trace_records), reasons=reasons),
                  protocol=dict(source_experiment=str(Path(args.source_dir).resolve()),
                      shared_upstream=True, shared_mcts=True, independent_tail_arms=True,
                      native_fps=True, vfr_verified=False, thresholds=[.1,.2,.3,.4,.5],
                      temporal_ap='sum(precision_at_rank * TP) / N_GT; six-class macro mean',
                      annotation_sha256=annotation_fingerprint(args.dataset_dir),
                      scope='35-video exploratory paired ablation, not the full XD benchmark',
                      qscore='XD raw class curve, fixed scale; after geometry and merge, before NMS',
                      cbt_bue='XD support-based adaptation; no UCF GT-derived duration caps'))
    save_json(root/'evaluation_report.json', report)
    return report


@track_batch_usage
async def run(args):
    out, upstream = Path(args.output_dir), Path(args.source_dir)
    selection = read_json(upstream/'metadata/selection.json')
    names = sorted(set(n for group in selection['slots'].values() for n in group))
    if args.evaluate_only:
        return evaluate(args, names)
    source_manifest = read_json(upstream/'metadata/manifest.json')
    if source_manifest['config']['dataset'] != 'xd_violence' or source_manifest['config']['mcts']['mode'] != 'off':
        raise ValueError('Source must be an XD MCTS-off experiment')
    save_json(out/'metadata/upstream_config.json', source_manifest['config'])
    cfg = VADConfig.load(out/'metadata/upstream_config.json')
    cfg.mcts = MCTSConfig(mode='refine', action_set='shrink_expand_shift', acceptance_policy='force_best',
        trigger_policy='all_valid', trace_enabled=True, max_candidates_per_video=args.max_candidates,
        max_vlm_evaluations_per_video=args.max_candidates*10, max_vlm_evaluations_per_candidate=10,
        max_images_per_evaluation=32, max_boundary_gap_seconds=1.5, max_evidence_gap_seconds=3.,
        timeout_seconds_per_candidate=120.)
    tail_cfg = XDTailConfig(nms_iou=cfg.xd.nms_iou)
    client = DashScopeLLMClient(model=args.model, max_tokens=2048, max_frames=32,
                               api_concurrency=args.api_concurrency, api_timeout_seconds=120.)
    inputs = {n: sha(upstream/(n+'.json')) for n in names}
    manifest = prepare_manifest(out/'metadata/manifest.json', cfg, [client], names,
        dict(source_dir=str(upstream.resolve()), upstream_manifest_sha256=sha(upstream/'metadata/manifest.json'),
             upstream_results_sha256=inputs, selection_sha256=sha(upstream/'metadata/selection.json'),
             arms=ARMS, tail_config=asdict(tail_cfg), concurrency=args.concurrency,
             candidate_allocation='predicted_class_round_robin', annotation_sha256=annotation_fingerprint(args.dataset_dir)))
    save_json(out/'metadata/selection.json', selection)
    save_json(out/'metadata/config.json', cfg.to_dict())
    save_json(out/'metadata/command.json', dict(argv=sys.argv, python=sys.executable))
    snap = out/'metadata/source_snapshot'
    snap.mkdir(exist_ok=True)
    for filename in manifest['source_sha256']:
        shutil.copy2(SOURCE/filename, snap/filename)
    baseline = {n: read_json(upstream/(n+'.json')) for n in names}
    for n, b in baseline.items():
        if b.get('dataset') != 'xd_violence' or Path(b['video_path']).stem != n or b.get('experiment_fingerprint') != source_manifest['fingerprint']:
            raise ValueError('Mismatched source video: '+n)
    replay = XDReplay(cfg, client, tail_cfg)
    semaphore = asyncio.Semaphore(args.concurrency)
    jobs = [dict(video_id=n, video_path=baseline[n]['video_path']) for n in names]
    started = time.perf_counter()
    async def execute(job, attempt):
        async with semaphore:
            n, path = job['video_id'], job['video_path']
            dest = out/'result'/(n+'.json')
            if dest.exists():
                if read_json(dest).get('experiment_fingerprint') != manifest['fingerprint']:
                    raise ValueError('Existing result fingerprint mismatch')
                return dict(status='skipped', retryable=False)
            try:
                result = await replay.detect(path, baseline[n])
                result.update(experiment_fingerprint=manifest['fingerprint'],
                              upstream_result_sha256=inputs[n], diagnosis_attempt=attempt,
                              token_usage_all_attempts=video_usage_all_attempts(path))
                save_json(dest, result)
                replay.caches.pop(path, None)
                t = result['shared']['mcts_trace']
                log.info('[XD-ABLATION] complete=%s/%s video=%s searched=%s applied=%s events=%s elapsed=%.2fs',
                    len(list((out/'result').glob('*.json'))), len(names), n, t['searched_candidates'],
                    t['applied_candidates'], {a: len(r['anomalies']) for a,r in result['arms'].items()}, result['processing_time'])
                return dict(status='success', retryable=False)
            except Exception as error:
                log.error('[XD-ABLATION] failed video=%s attempt=%s error=%s', n, attempt, type(error).__name__)
                return dict(status='failed', retryable=is_retryable(error), error_type=type(error).__name__, error=str(error))
    try:
        status, history = await run_video_jobs(jobs, execute, rounds=1,
            record=lambda item: save_json(out/'metadata/latest_job.json', item))
        save_json(out/'metadata/jobs.json', dict(results=status, history=history, wall_seconds=time.perf_counter()-started))
        failures = [s for s in status if s['status']=='failed']
        if failures:
            raise RuntimeError(f'{len(failures)} failed videos; inspect metadata/jobs.json')
        report = evaluate(args, names)
        log.info('[XD-ABLATION] DONE videos=%s searched=%s applied=%s AVG=%s', len(names),
                 report['mcts']['searched'], report['mcts']['applied'],
                 {a: round(r['temporal_mAP_full']['percent']['AVG'], 4) for a,r in report['arms'].items()})
        return report
    finally:
        await client.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset_dir', default='F:/dataset/xd-violence')
    parser.add_argument('--model', default='qwen3-vl-plus')
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--api_concurrency', type=int, default=4)
    parser.add_argument('--max_candidates', type=int, default=4)
    parser.add_argument('--evaluate_only', action='store_true')
    args = parser.parse_args()
    if min(args.concurrency, args.api_concurrency, args.max_candidates) <= 0:
        parser.error('Concurrency and candidate budgets must be positive')
    vad_logger.setup(Path(args.output_dir)/'run.log')
    # Evaluation-only never overwrites the actual API usage ledger.
    if args.evaluate_only:
        selection = read_json(Path(args.source_dir)/'metadata/selection.json')
        names = sorted(set(n for group in selection['slots'].values() for n in group))
        evaluate(args, names)
    else:
        asyncio.run(run(args))


if __name__ == '__main__':
    main()
