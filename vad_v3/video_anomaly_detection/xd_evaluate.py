"""XD frame AP and temporal mAP with explicit video/class evaluation scopes."""
from pathlib import Path
from collections import defaultdict
import json
import sys
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from .xd_config import XD_CLASSES
    from .xd_dataset import load_xd_index, annotation_fingerprint
except ImportError:
    from xd_config import XD_CLASSES
    from xd_dataset import load_xd_index, annotation_fingerprint

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path: sys.path.insert(0, str(PROJECT))
from evaluation_tools.temporal_ap import ranked_ap, prediction_frames, THRESHOLDS
from evaluation_tools.video_fps import source_video_metadata


def temporal_metrics(records, predictions, metadata, include_normal, *, normal_as_class=False):
    """Optionally evaluate Normal as a seventh temporal detection class.

    include_normal controls which VIDEOS are evaluated, while normal_as_class
    controls which CLASSES enter the macro mean. These are distinct decisions.
    Normal GT covers the entire source video only for a genuinely Normal video;
    background gaps of abnormal videos do not become Normal GT. Predictions
    must be supplied independently; this evaluator never invents detections.
    The default six-class protocol remains unchanged.
    """
    if normal_as_class and not include_normal:
        raise ValueError('Normal-as-class requires inclusion of Normal videos')
    classes = ('Normal', *XD_CLASSES) if normal_as_class else XD_CLASSES
    ground = {c: defaultdict(list) for c in classes}
    detected = {c: [] for c in classes}
    included = [name for name, r in records.items() if include_normal or r['events']]
    for name in included:
        if normal_as_class:
            # Explicit source labels are required, so missing anomaly annotation
            # is never mistaken for a verified Normal ground-truth video.
            category = records[name]['category']
            if category == 'Normal':
                if records[name]['events']:
                    raise ValueError('Normal GT video contains anomaly annotations')
                count = metadata[name]['frame_count']
                if not isinstance(count, int) or count <= 0:
                    raise ValueError('Normal GT requires positive source frame count')
                ground['Normal'][name].append((0, count))
        for e in records[name]['events']:
            ground[e['category']][name].append((e['start_frame'], e['end_frame']))
        for i, e in enumerate(predictions[name]['anomalies']):
            c = e['anomaly_type']
            if c not in classes: raise ValueError('Unknown predicted XD category: ' + c)
            detected[c].append(dict(video_id=name, category=c, prediction_index=i,
                                    confidence=float(e['confidence']),
                                    frames=prediction_frames(e, metadata[name]['fps'])))
    per_class = {};
    audit = {}
    for c in classes:
        per_class[c] = {};
        audit[c] = {}
        for t in THRESHOLDS:
            metric, rows = ranked_ap(detected[c], ground[c], t)
            per_class[c][f'AP@{t:.1f}'] = metric['ap']
            per_class[c][f'TP@{t:.1f}'] = metric['tp']
            per_class[c][f'FP@{t:.1f}'] = metric['fp']
            audit[c][str(t)] = rows
        per_class[c].update(gt_count=metric['gt_count'], prediction_count=metric['prediction_count'])
    values = {f'mAP@{t:.1f}': float(np.mean([per_class[c][f'AP@{t:.1f}'] for c in classes])) for t in THRESHOLDS}
    values['AVG'] = float(np.mean(list(values.values())))
    return dict(values=values, percent={k: 100 * v for k, v in values.items()}, per_class=per_class,
                video_count=len(included), class_count=len(classes), include_normal_videos=include_normal), audit


def evaluate_xd(root, result_dir, selected_ids, annotations=None):
    records_all = load_xd_index(root, annotations)
    records = {name: records_all[name] for name in selected_ids}
    predictions = {}
    for name in selected_ids:
        p = Path(result_dir) / (name + '.json')
        d = json.loads(p.read_text(encoding='utf-8'))
        if d.get('dataset') != 'xd_violence' or Path(d['video_path']).stem != name:
            raise ValueError('Mismatched XD prediction identity: ' + name)
        predictions[name] = d
    metadata = source_video_metadata({n: r['video_path'] for n, r in records.items()})
    directory = Path(result_dir) / 'metadata'
    directory.mkdir(exist_ok=True)
    (directory / 'video_fps.json').write_text(json.dumps(dict(schema='video-fps-v1', videos=metadata), indent=2),
                                              encoding='utf-8')
    labels = [];
    scores = [];
    per_video = []
    for name, r in records.items():
        n = metadata[name]['frame_count'];
        fps = metadata[name]['fps']
        y = np.zeros(n, dtype=np.uint8)
        for start, end in r['segments']:
            if not 0 <= start < end <= n: raise ValueError('Binary GT exceeds decoded video: ' + name)
            y[start:end] = 1
        for e in r['events']:
            if not 0 <= e['start_frame'] < e['end_frame'] <= n:
                raise ValueError('Typed GT exceeds decoded video: ' + name)
        pred = predictions[name]
        ts = np.asarray(pred['timestamps'], dtype=float)
        fs = np.asarray(pred['frame_scores'], dtype=float)
        if len(ts) != len(fs) or not len(ts) or not np.isfinite(ts).all() or not np.isfinite(fs).all() or np.any(
                np.diff(ts) <= 0):
            raise ValueError('Invalid prediction time series: ' + name)
        labels.append(y)
        if pred.get('frame_score_grid') == 'native_frames':
            if len(fs) != n:
                raise ValueError('Native frame score count mismatch: ' + name)
            scores.append(fs)
        else:
            scores.append(np.interp(np.arange(n) / fps, ts, fs))
        expected = sorted({e['category'] for e in r['events']})
        got = sorted({e['anomaly_type'] for e in pred['anomalies']})
        if set(got) - set(XD_CLASSES): raise ValueError('Invalid category')
        per_video.append(dict(video_id=name, gt_categories=expected, predicted_categories=got,
                              gt_event_count=len(r['events']), predicted_event_count=len(pred['anomalies']),
                              category_set_correct=expected == got, is_normal=not expected,
                              normal_false_alarm=not expected and bool(got),
                              processing_time=pred.get('processing_time'),
                              prior_scores=pred.get('category_prior', {}).get('scores', {})))
    y = np.concatenate(labels);
    score = np.concatenate(scores)
    full, audit = temporal_metrics(records, predictions, metadata, True)
    abnormal, _ = temporal_metrics(records, predictions, metadata, False)
    normal = [r for r in per_video if r['is_normal']]
    report = dict(dataset='xd_violence', video_count=len(records), frame_AP=float(average_precision_score(y, score)),
                  frame_AUC=float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
                  temporal_mAP_full=full, temporal_mAP_abnormal=abnormal,
                  normal_video_count=len(normal), normal_false_alarm_count=sum(r['normal_false_alarm'] for r in normal),
                  total_predictions=sum(r['predicted_event_count'] for r in per_video), per_video=per_video,
                  protocol=dict(frame_metric='sklearn.average_precision_score on concatenated source frames',
                                temporal_ap='sum(precision_at_rank * TP) / N_GT', classes=list(XD_CLASSES),
                                thresholds=list(THRESHOLDS), interval='source frames [start,end)',
                                timebase='per-video OpenCV native FPS; variable frame rate not verified',
                                annotation_sha256=annotation_fingerprint(root, annotations),
                                scope='full test split' if set(records)==set(records_all)
                                else 'selected videos only; not the full 800-video benchmark'))
    (directory / 'temporal_matching_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                                                            encoding='utf-8')
    return report
