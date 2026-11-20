"""Independently cross-check completed v218_ucf with the repository evaluator."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'video_anomaly_detection'))
from evaluate import compute_frame_auc, load_ground_truth
from scoring import windows


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    manifest = read(HERE / 'manifest.json')
    rows = read(HERE / 'selected_290.json')
    report = read(HERE / 'full290_metrics.json')
    progress = read(HERE / 'progress.json')
    assert read(HERE / 'exit_status.json')['status'] == 'completed'
    assert len(rows) == len(list((HERE / 'results').glob('*.json'))) == 290
    assert sum(r['category'] == 'Normal' for r in rows) == 150
    assert progress['counts']['max_observed_video_workers'] == 2
    assert progress['counts']['max_observed_api_requests'] <= 4
    assert not progress['failed_videos']
    assert manifest['primary_arm'] == 'fusion_50'
    payload = {k: v for k, v in manifest.items() if k != 'fingerprint'}
    assert hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() == manifest['fingerprint']
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
    gt_all = load_ground_truth('F:/dataset/ucf_crime/data_parquet/test-00000-of-00001.parquet')
    gt = {r['video_id']: gt_all[r['video_id']] for r in rows}
    gt_ano = {k: v for k, v in gt.items() if v['category'] != 'Normal'}
    fps = {r['video_id']: r['fps'] for r in rows}
    predictions = {k: {} for k in report['metrics']}
    expected_windows = 0
    new_windows = 0
    for row in rows:
        name = row['video_id']
        result = read(HERE / 'results' / (name + '.json'))
        assert result['fingerprint'] == manifest['fingerprint']
        count = len(windows(row['duration'], manifest['settings']['stride_seconds'], manifest['settings']['context_seconds']))
        assert len(result['timestamps']) == len(result['frame_scores']) == count
        assert result['exact_pairs'] + result['bounded_pairs'] == count
        expected_windows += count
        if name in manifest['pilot_ids']:
            origin = manifest['reused_pilot_results'][name]
            origin_path = Path(origin['path'])
            assert hashlib.sha256(origin_path.read_bytes()).hexdigest() == origin['sha256']
            old = read(origin_path)
            for key in ['timestamps', 'frame_scores', 'hard_scores', 'exact_pairs', 'bounded_pairs']:
                assert result[key] == old[key]
        else:
            assert len(list((HERE / 'requests' / name).glob('*.json'))) == count
            new_windows += count
        source = manifest['source_results'][name]
        source_path = Path(source['path'])
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source['sha256']
        historical = read(source_path)
        baseline = historical['arms']['force_best_fixed'] if 'arms' in historical else historical['paired_results']['force_best']
        common = dict(video_path=row['path'], duration=row['duration'], source_frame_count=row['frame_count'], anomalies=[])
        predictions['baseline'][name] = dict(common, frame_scores=baseline['frame_scores'], timestamps=baseline['timestamps'])
        predictions['yesno_probability'][name] = dict(common, frame_scores=result['frame_scores'], timestamps=result['timestamps'])
        predictions['yesno_hard_diagnostic'][name] = dict(common, frame_scores=result['hard_scores'], timestamps=result['timestamps'])
        # Construct the frozen fusion at native frame resolution independently.
        frame = np.arange(row['frame_count'])
        b = np.interp(frame, np.asarray(baseline['timestamps']) * row['fps'], np.asarray(baseline['frame_scores'], dtype=np.float32)).astype(np.float32)
        p = np.interp(frame, np.asarray(result['timestamps']) * row['fps'], np.asarray(result['frame_scores'], dtype=np.float32)).astype(np.float32)
        weight = manifest['settings']['fusion_weight']
        predictions['fusion_50'][name] = dict(common, frame_scores=((1-weight)*b+weight*p).astype(np.float32), timestamps=None)
    assert expected_windows == report['counts']['windows']
    assert new_windows == progress['counts']['windows_completed']
    comparisons = {}
    for arm, pred in predictions.items():
        actual = compute_frame_auc(gt, pred, fps=fps)
        actual_ano = compute_frame_auc(gt_ano, pred, fps=fps)
        expected = report['metrics'][arm]
        deltas = dict(global_auc=actual['auc']-expected['global_auc'],
                      aupr=actual['aupr']-expected['aupr'],
                      ano_auc=actual_ano['auc']-expected['ano_auc'])
        for delta in deltas.values():
            assert abs(delta) < 1e-12, (arm, deltas)
        for category, value in expected['per_category_auc'].items():
            assert abs(actual['per_category_auc'][category] - value) < 1e-12
        comparisons[arm] = deltas
        print(json.dumps(dict(verified_arm=arm, deltas=deltas)), flush=True)
    output = dict(status='passed', videos=290, abnormal_videos=140, normal_videos=150,
                  reused_videos=65, new_videos=225, total_windows=expected_windows,
                  new_windows=new_windows, source_hashes_unchanged=True,
                  imported_scores_unchanged=True, video_concurrency_peak=2,
                  evaluator='vad_v3/video_anomaly_detection/evaluate.py:compute_frame_auc',
                  metric_tolerance=1e-12, metric_deltas=comparisons,
                  fingerprint=manifest['fingerprint'])
    (HERE / 'evaluation_validation.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
