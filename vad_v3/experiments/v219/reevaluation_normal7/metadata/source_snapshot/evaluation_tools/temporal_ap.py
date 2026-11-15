"""VadCLIP论文的异常视频/异常类别评测范围 + 官方代码的AP积分公式。

保留调用者的最终检测区间和置信度，不重做VadCLIP的逐帧曲线提议/NMS。
预测秒数乘各视频FPS，再int截断，区间为[start,end)，对应官方range集合IoU。
AP_c = sum_k(precision_at_rank_k * TP_k) / N_GT_c；不是VOC 11点插值。
官方当前ucf_test.py默认包含Normal，与论文文字不一致；本模块明确选择论文范围。
"""
from collections import defaultdict
import math
import numpy as np
from .video_fps import fps_for_video

ANOMALY_CLASSES = ('Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
                   'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting',
                   'Stealing', 'Vandalism')
THRESHOLDS = (.1, .2, .3, .4, .5)


def interval_iou(left, right):
    intersection = max(0, min(left[1],right[1])-max(left[0],right[0]))
    union = max(0,left[1]-left[0])+max(0,right[1]-right[0])-intersection
    return intersection / union if union > 0 else 0.


def prediction_frames(row, fps, integer=True):
    start, end, score = (float(row[k]) for k in ('t_start','t_end','confidence'))
    if not all(math.isfinite(x) for x in (start,end,score)) or not 0 <= start < end:
        raise ValueError('Non-finite/invalid prediction interval or confidence')
    result = (start*fps, end*fps)
    # A sub-frame detection may become empty. Keep it as a false positive.
    return tuple(int(x) for x in result) if integer else result


def ranked_ap(detections, gt_by_video, threshold, formula='vadclip'):
    """Class filtering happens before this function. Duplicates count as FPs."""
    total_gt = sum(len(rows) for rows in gt_by_video.values())
    used = defaultdict(set)
    tp, records = [], []
    # 与 VadCLIP 官方一致：仅按置信度调用 NumPy 默认 argsort。
    # 不再额外按视频/边界打破并列；并列顺序取决于输入和 NumPy 版本。
    order = np.argsort(-np.asarray([x['confidence'] for x in detections], dtype=float))
    ranked = [detections[int(i)] for i in order]
    for rank, pred in enumerate(ranked,1):
        name = pred['video_id']
        best_iou, best_id = 0., None
        for gid, segment in enumerate(gt_by_video.get(name, [])):
            if gid in used[name]:
                continue
            score = interval_iou(pred['frames'], segment)
            if score > best_iou:
                best_iou, best_id = score, gid
        hit = best_id is not None and best_iou >= threshold
        if hit:
            used[name].add(best_id)
        tp.append(int(hit))
        records.append(dict(pred, rank=rank, true_positive=hit, best_unmatched_iou=best_iou,
                            matched_gt_index=best_id if hit else None))
    cumulative = np.cumsum(tp,dtype=float)
    precision = cumulative / np.arange(1,len(tp)+1) if tp else np.array([])
    recall = cumulative / total_gt if total_gt else np.zeros(len(tp))
    if not total_gt:
        ap = 0.
    elif formula == 'vadclip':
        ap = float(np.sum(precision*np.asarray(tp))/total_gt)
    elif formula == 'voc11':
        ap = float(sum(precision[recall>=t].max() if np.any(recall>=t) else 0.
                       for t in np.arange(0.,1.1,.1))/11.)
    else:
        raise ValueError('Unknown AP formula: '+formula)
    return dict(ap=ap,gt_count=total_gt,prediction_count=len(tp),tp=sum(tp),fp=len(tp)-sum(tp)), records


def compute_vadclip_map(gt, predictions, fps, *, include_normal_videos=False,
                       thresholds=THRESHOLDS, classes=ANOMALY_CLASSES):
    if set(gt) != set(predictions):
        raise ValueError('Ground truth and predictions must have identical video coverage')
    included = sorted(k for k,v in gt.items() if include_normal_videos or v['category']!='Normal')
    detections, ground = defaultdict(list), {c:defaultdict(list) for c in classes}
    for name in included:
        video_fps = fps_for_video(fps,name)
        category = gt[name]['category']
        if category != 'Normal' and category not in classes:
            raise ValueError('Unknown GT category: '+category)
        if category != 'Normal':
            for segment in gt[name]['segments']:
                if len(segment)!=2 or not 0 <= segment[0] < segment[1]:
                    raise ValueError('Invalid GT interval')
                ground[category][name].append(tuple(int(x) for x in segment))
        for index,row in enumerate(predictions[name]['anomalies']):
            category = row['anomaly_type']
            if category not in classes:
                raise ValueError('Unknown predicted anomaly category: '+category)
            detections[category].append(dict(video_id=name,category=category,prediction_index=index,
                confidence=float(row['confidence']),frames=prediction_frames(row,video_fps),
                seconds=[row['t_start'],row['t_end']],fps=video_fps))
    per_class, audit = {}, {}
    for category in classes:
        metrics, audit[category] = {}, {}
        for threshold in thresholds:
            value, records = ranked_ap(detections[category],ground[category],threshold)
            metrics[f'AP@{threshold:.1f}'] = value['ap']
            metrics[f'TP@{threshold:.1f}'] = value['tp']
            audit[category][f'{threshold:.1f}'] = records
        metrics.update(gt_count=value['gt_count'],prediction_count=value['prediction_count'])
        per_class[category] = metrics
    values = {f'mAP@{t:.1f}':float(np.mean([per_class[c][f'AP@{t:.1f}'] for c in classes])) for t in thresholds}
    values['AVG'] = float(np.mean(list(values.values())))
    return dict(values=values,percent={k:v*100 for k,v in values.items()},per_class=per_class,
        video_count=len(included),class_count=len(classes),
        gt_count=sum(x['gt_count'] for x in per_class.values()),
        prediction_count=sum(x['prediction_count'] for x in per_class.values())), audit


def compute_legacy_map(gt,predictions,fps,thresholds=(.3,.5,.7)):
    """旧类别无关/全视频/连续帧边界/VOC11公式，用于仅FPS变更对照。"""
    rows = []
    for name in sorted(gt):
        for index,row in enumerate(predictions[name]['anomalies']):
            rows.append(dict(video_id=name,category=row['anomaly_type'],prediction_index=index,
                confidence=float(row['confidence']),frames=prediction_frames(row,fps_for_video(fps,name),False)))
    ground={k:v['segments'] for k,v in gt.items()}
    scores={f'AP@{t:.1f}':ranked_ap(rows,ground,t,'voc11')[0]['ap'] for t in thresholds}
    scores['mAP']=float(np.mean(list(scores.values())))
    return scores
