"""Replay frozen scores through the production adapter; no new model inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]/'video_anomaly_detection'
sys.path.insert(0, str(SOURCE))
from config import ScoringConfig
from evaluate import compute_frame_auc, load_ground_truth
from models import DetectionResult
from yesno_frame_scorer import YesNoFrameScorer


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class NoInference:
    async def score_yesno(self, *args, **kwargs):
        raise AssertionError('Cached replay must not call a model')


def main():
    rows = read(HERE/'selected_290.json')
    manifest = read(HERE/'manifest.json')
    expected_metrics = read(HERE/'full290_metrics.json')['metrics']['fusion_50']
    spec = importlib.util.spec_from_file_location('frozen_v218_evaluator', HERE/'evaluate_scores.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    scorer = YesNoFrameScorer(NoInference(), ScoringConfig(yesno_enabled=True))
    predictions = {}
    for row in rows:
        name = row['video_id']
        old = read(Path(manifest['source_results'][name]['path']))
        base = old['arms']['force_best_fixed'] if 'arms' in old else old['paired_results']['force_best']
        cached = read(HERE/'results'/(name+'.json'))
        result = DetectionResult(row['path'], row['duration'], True,
                                 frame_scores=base['frame_scores'], timestamps=base['timestamps'])
        visual = dict(source={k: row[k] for k in ['fps', 'frame_count', 'duration']},
                      settings=manifest['settings'], timestamps=cached['timestamps'],
                      frame_scores=cached['frame_scores'], exact_pairs=cached['exact_pairs'],
                      bounded_pairs=cached['bounded_pairs'])
        scorer.apply_scores(result, visual)
        expected_curve = (.5*reference.interpolate(row, base['frame_scores'], base['timestamps'])
                          +.5*reference.interpolate(row, cached['frame_scores'], cached['timestamps'])).astype(np.float32)
        assert np.array_equal(result.frame_scores, expected_curve), name
        predictions[name] = result.to_dict()
    all_gt = load_ground_truth('F:/dataset/ucf_crime/data_parquet/test-00000-of-00001.parquet')
    gt = {r['video_id']: all_gt[r['video_id']] for r in rows}
    fps = {r['video_id']: r['fps'] for r in rows}
    actual = compute_frame_auc(gt, predictions, fps)
    abnormal = {k: v for k, v in gt.items() if v['category'] != 'Normal'}
    ano = compute_frame_auc(abnormal, predictions, fps)
    assert abs(actual['auc']-expected_metrics['global_auc']) < 1e-12
    assert abs(actual['aupr']-expected_metrics['aupr']) < 1e-12
    assert abs(ano['auc']-expected_metrics['ano_auc']) < 1e-12
    for category, expected in expected_metrics['per_category_auc'].items():
        assert abs(actual['per_category_auc'][category]-expected) < 1e-12
    output = dict(status='passed', mode='cached_score_replay_not_fresh_inference', new_api_calls=0,
                  videos=290, normal_videos=150, identical_native_curves=290,
                  global_auc=actual['auc'], ano_auc=ano['auc'], aupr=actual['aupr'],
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in SOURCE.glob('*.py')},
                  note='Validates production fusion and evaluation against v218_ucf; does not rerun upstream detection.')
    (HERE/'mainflow_integration_validation.json').write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in output.items() if k != 'source_sha256'}), flush=True)


if __name__ == '__main__':
    main()
