"""Freeze previous MCTS trees; change acceptance only; rerun real BUE -> CBT.

No VLM requests or GT access. This is a paired acceptance ablation on saved
search results, not a fresh end-to-end run or an increased search-budget test.
"""
import copy
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODE = ROOT / 'vad_v3/video_anomaly_detection'
SOURCE = HERE.parent / 'stage4_fixed_vs_mcts'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(CODE))
os.environ.setdefault('HF_HUB_OFFLINE', '1')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def gpu_slot():
    from vad_v3.webapp.store import Store
    from vad_v3.webapp.worker import InstanceLock
    from vad_v3.webapp.service import read_runtime, alive
    from vad_v3.webapp.settings import atomic_json
    import psutil
    store = Store(ROOT / 'vad_v3/web_data'); runtime = read_runtime(store)
    resume = alive(runtime.get('worker'))
    with store.connect() as db:
        for table, statuses in (('jobs', "'queued','running','cancelling'"),
                                ('evaluations', "'queued','running'")):
            if db.execute(f'SELECT COUNT(*) FROM {table} WHERE status IN ({statuses})').fetchone()[0]:
                raise RuntimeError('Active web work; GPU replay must wait')
    try:
        if resume:
            store.service('control', dict(stop=True))
            deadline = time.monotonic() + 15
            while alive(runtime.get('worker')) and time.monotonic() < deadline:
                time.sleep(.2)
            if alive(runtime.get('worker')):
                raise RuntimeError('Worker did not release GPU lock')
        with InstanceLock(store.root / '.worker.lock'):
            yield
    finally:
        if resume:
            store.service('control', dict(stop=False))
        if resume and not alive(runtime.get('worker')):
            flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == 'nt' else 0
            with (store.root / 'worker.service.log').open('a', encoding='utf-8') as log:
                process = subprocess.Popen([sys.executable, '-u', '-m', 'vad_v3.webapp.worker',
                    '--data-dir', str(store.root)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8'), creationflags=flags)
            runtime['worker'] = dict(pid=process.pid, created=psutil.Process(process.pid).create_time())
            atomic_json(store.root / 'runtime.json', runtime)


def geometry(rows):
    return [(r['t_start'], r['t_end'], r['confidence'], r['anomaly_type']) for r in rows]


def execute():
    import numpy as np
    from config import VADConfig
    from pipeline import MultiAnomalyVAD, VideoAnomalyExpert, CATEGORY_DURATION_PRIORS
    from video_preprocessor import VideoPreprocessor
    from frame_scorer import FrameScorer
    from mcts_acceptance import NODE_ONLY_POLICIES, select_node_proposal
    from mcts_boundary_scorer import BoundaryEvidenceScorer
    from boundary_search import LocalBoundarySearch
    from token_compressor import SemanticFrameSampler
    from video_state import fork_video_state
    import vad_logger

    vad_logger.setup(log_file=str(HERE / 'console.log'))
    selected = json.loads((SOURCE / 'selected_videos.json').read_text(encoding='utf-8'))
    source_results = {v['video_id']: json.loads((SOURCE / 'results' / (v['video_id'] + '.json')).read_text(encoding='utf-8'))
                      for v in selected}
    configs = [r['mcts_trace']['effective_config'] for r in source_results.values()
               if (r.get('mcts_trace') or {}).get('route') == 'stage4_direct']
    assert configs and all(c == configs[0] for c in configs)
    write(HERE / 'config.json', configs[0]); config = VADConfig.load(HERE / 'config.json')
    assert config.device == 'cuda' and config.mcts.dense_fps == 0
    write(HERE / 'selected_videos.json', selected)
    # Postprocessing and feature extraction must be the actual previous code.
    with zipfile.ZipFile(SOURCE / 'executed_source.zip') as archive:
        for name in ('pipeline.py', 'video_preprocessor.py', 'anomaly_expert.py', 'frame_scorer.py',
                     'mcts_boundary_scorer.py', 'boundary_search.py', 'token_compressor.py'):
            assert archive.read('video_anomaly_detection/' + name) == (CODE / name).read_bytes(), name
    for video in selected:
        assert digest(Path(video['path'])) == video['sha256'], video['video_id']
    write(HERE / 'protocol.json', dict(
        kind='frozen_search_tree_acceptance_replay', source=str(SOURCE),
        policies=['fixed', 'strict', *NODE_ONLY_POLICIES],
        relaxed_node='Keep target presence, same event, CLIP quality floor; drop completeness, cut-boundary rejection and final A/B comparison.',
        force_best='Select the highest-reward evaluated non-root node, even if below root; drop semantic/CLIP acceptance and final A/B gates.',
        preserved='Same 14 videos, Stage 4 candidates, predictions, scores, search tree, rewards, geometry/evidence validation, budgets and fixed BUE -> CBT.',
        candidate_scope='One best proposal per searched candidate. No emitting all tree nodes; no invented proposal for unsearched/failed candidates.',
        inference='Recompute native video frames and actual CLIP category curves on CUDA; execute original BUE/CBT. Verify baseline boundaries exactly against source run.',
        costs='Zero new VLM requests. Replay runtime includes feature rebuilding and multiple tails, not per-policy fresh-run latency.',
        limitations='One video per class; fixed shrink trees; no new exploration. Existing CBT priors unchanged, historical training provenance unverified.',
        source_results_sha256={v['video_id']: digest(SOURCE / 'results' / (v['video_id'] + '.json')) for v in selected}))
    with zipfile.ZipFile(HERE / 'executed_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(CODE.glob('*.py')):
            archive.write(path, 'video_anomaly_detection/' + path.name)
        for path in sorted((ROOT / 'evaluation_tools').glob('*.py')):
            archive.write(path, 'evaluation_tools/' + path.name)
        archive.write(Path(__file__), 'experiment/run_replay.py')

    # Only the feature extractor and fixed tail are needed. No API client,
    # discovery, classification or GT labels are constructed in this replay.
    vad = MultiAnomalyVAD.__new__(MultiAnomalyVAD)
    vad.config = config; vad.anomaly_categories = config.anomaly_categories
    vad.preprocessor = VideoPreprocessor(**asdict(config.preprocess))
    vad.anomaly_expert = VideoAnomalyExpert(anomaly_categories=vad.anomaly_categories,
        clip_model_name=config.clip.model_name, clip_pretrained=config.clip.pretrained, device=config.device)
    if vad.anomaly_expert.model is None:
        raise RuntimeError('CLIP must load for real fixed-tail replay')
    vad.frame_scorer = FrameScorer(vad.anomaly_expert, config, vad.anomaly_categories)

    for index, video in enumerate(selected, 1):
        started = time.perf_counter(); name = video['video_id']; source = source_results[name]
        trace = source.get('mcts_trace') or {}; arms = {}; decisions = {}
        if trace.get('route') != 'stage4_direct':
            assert not source['anomalies']
            arms = {policy: dict(anomalies=[], post_search=[], post_bue=[])
                    for policy in ('fixed', 'strict', *NODE_ONLY_POLICIES)}
        else:
            data = vad.preprocessor.process(video['path'])
            data['clip_features'] = vad.anomaly_expert.precompute_video_features(data['all_frames'])
            category = trace['search_category']
            curve = vad.frame_scorer.get_category_scores(data, category)
            assert curve is not None and np.isfinite(curve).all()
            seeds = trace['stages']['post_stage4']
            assert np.allclose(data['timestamps'], source['timestamps'], atol=1e-9, rtol=0)
            scorer = BoundaryEvidenceScorer(data, curve, config.mcts,
                CATEGORY_DURATION_PRIORS[category]['expected_duration'])
            np.savez_compressed(HERE / 'curves' / (name + '.npz'), timestamps=data['timestamps'], curve=curve)

            def finish(rows):
                rows = copy.deepcopy(rows); state = fork_video_state(data)
                result = dict(post_search=copy.deepcopy(rows))
                rows = vad._bue_boundary_union_extension(rows, state, category)
                result['post_bue'] = copy.deepcopy(rows)
                rows = vad._cbt_category_boundary_tighten(rows, state, category)
                # Keep original descriptions and metadata while changing boundaries only.
                result['anomalies'] = copy.deepcopy(source['paired_results']['original_off']['anomalies'])
                for target, row in zip(result['anomalies'], rows):
                    target['t_start'], target['t_end'] = row['t_start'], row['t_end']
                assert len(result['anomalies']) == len(rows) == len(seeds)
                return result

            arms['fixed'] = finish(seeds)
            arms['strict'] = finish(trace['stages']['post_mcts'])
            assert geometry(arms['fixed']['anomalies']) == geometry(source['paired_results']['original_off']['anomalies']), name
            assert geometry(arms['strict']['anomalies']) == geometry(source['anomalies']), name
            assert geometry(arms['fixed']['post_bue']) == geometry(source['paired_results']['original_off']['mcts_trace']['stages']['post_bue']), name
            for policy in NODE_ONLY_POLICIES:
                rows = copy.deepcopy(seeds); decisions[policy] = {}
                for cid, candidate in trace['candidates'].items():
                    record = dict(applied=False, source_reason=candidate.get('reason'))
                    decisions[policy][cid] = record
                    nodes = [SimpleNamespace(interval=tuple(n['interval']), depth=n['depth'], evaluation=n['evaluation'])
                             for n in candidate.get('nodes', [])]
                    if not nodes or not nodes[0].evaluation:
                        record['reason'] = 'no_evaluated_search_tree'; continue
                    best = select_node_proposal(nodes, candidate['baseline_curve_quality'], policy,
                                                config.mcts.max_curve_quality_drop)
                    if best is None:
                        record['reason'] = 'no_valid_proposal'; continue
                    seed = seeds[int(cid)]; interval = (seed['t_start'], seed['t_end'])
                    search = LocalBoundarySearch(interval, scorer, config.mcts, int(cid))
                    assert search.legal(best.interval), (name, cid, policy)
                    SemanticFrameSampler.select_boundary_evidence(scorer, interval, [best.interval],
                        candidate['core_evidence_ids'], config.mcts.max_images_per_evaluation)
                    assert abs(scorer.quality(interval)['quality'] - candidate['baseline_curve_quality']) < 1e-6
                    rows[int(cid)]['t_start'], rows[int(cid)]['t_end'] = best.interval
                    record.update(applied=True, reason='accepted', proposed=list(best.interval),
                        root_reward=nodes[0].evaluation['reward'], selected_reward=best.evaluation['reward'],
                        selected_answer=best.evaluation['answer'], selected_depth=best.depth)
                arms[policy] = finish(rows)
                assert [(x['confidence'], x['anomaly_type']) for x in arms[policy]['anomalies']] == [
                    (x['confidence'], x['anomaly_type']) for x in arms['fixed']['anomalies']]
        out = dict(video=video, arms=arms, decisions=decisions, new_vlm_requests=0,
            baseline_reproduced=True, replay_seconds=time.perf_counter()-started)
        write(HERE / 'results' / (name + '.json'), out)
        print(json.dumps(dict(progress=f'{index}/{len(selected)}', video=name,
            accepted={p: sum(c['applied'] for c in ds.values()) for p, ds in decisions.items()},
            elapsed_seconds=round(out['replay_seconds'], 2))), flush=True)
    write(HERE / 'exit_status.json', dict(returncode=0, completed=len(selected), new_vlm_requests=0))


if __name__ == '__main__':
    (HERE / 'curves').mkdir(parents=True, exist_ok=True)
    with gpu_slot():
        execute()
