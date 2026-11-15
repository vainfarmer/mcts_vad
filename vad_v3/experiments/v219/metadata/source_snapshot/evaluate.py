"""
UCF-Crime 评估模块

提供四级评估指标，全面衡量异常检测与定位的性能：

  Level 1 — 帧级 AUC-ROC / AUPR（UCF-Crime 标准指标）
    将所有视频的帧级标签和预测分数拼接后计算全局 AUC 和 AUPR。
    AUC-ROC 是 UCF-Crime benchmark 的官方评估方式。
    AUPR 在正负样本极不均衡时（异常帧远少于正常帧）比 AUC-ROC 更灵敏。
    同时报告 per-category AUC，暴露不同异常类别的检测难度差异。

  Level 2 — VadCLIP 方式的时间段级 mAP（IoU 0.1—0.5）
    按类别匹配，AP = sum(Precision@k * TP_k) / N_GT，不做 11 点插值。
    使用原视频 FPS、整数半开帧区间；主指标对 13 类异常取均值，正常视频单独评价。
    同时保留类别无关的 0.3/0.5/0.7 辅助指标，也使用上述非插值 AP。

  Level 3 — 视频级 accuracy / precision / recall / F1 / AUC
    二分类指标：视频是否包含异常。用最高置信度异常段的 confidence 作为视频级分数。

  Level 4 — 异常类别分类准确率
    检查预测的异常类型是否匹配 GT 类别。只评估异常视频（Normal 视频跳过）。

用法：
  # parquet 目录（HuggingFace 数据集格式，自动读取 test split）
  python evaluate.py \
    --label_path data/ucf-crime/data/ \
    --result_dir data/results/ \
    --video_root F:/dataset/ucf_crime

  # 单个 parquet 文件
  python evaluate.py \
    --label_path data/test-00000-of-00001.parquet \
    --result_dir data/results/

  # 传统纯文本格式（向后兼容）
  python evaluate.py \
    --label_path data/ucf_crime_labels.txt \
    --result_dir data/results/

注意：默认直接打开预测结果 video_path 指向的原视频读取 FPS/帧数。
视频搬迁后可用 --video_root 定位；--legacy_fixed_fps 仅用于显式历史复现。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 共享视频探测工具位于项目根目录，兼容直接运行本文件。
import sys
for _project_root in Path(__file__).resolve().parents:
    if (_project_root / 'evaluation_tools' / 'video_fps.py').is_file():
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))
        break
from evaluation_tools.video_fps import source_video_metadata, fps_for_video
from evaluation_tools.temporal_ap import compute_vadclip_map, ANOMALY_CLASSES, THRESHOLDS

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

try:
    import vad_logger as _vl
    _log = _vl.get("vad.eval")
except Exception:
    import logging
    _log = logging.getLogger("vad.eval")


# ---------------------------------------------------------------------------
# GT 加载（支持 parquet / 纯文本）
# ---------------------------------------------------------------------------

def _extract_video_id(video_path: str) -> str:
    return Path(video_path).stem


def load_ground_truth(label_path: str, split: str = "test") -> Dict[str, Dict[str, Any]]:
    """
    解析 GT 标注，自动识别 parquet 目录、parquet 文件或纯文本格式。

    Parquet 格式（HuggingFace UCF-Crime 数据集）：
      列: video_path, category, annotation_start1, annotation_end1,
          annotation_start2, annotation_end2
      -1 表示该段无标注。

    纯文本格式（每行一个视频）：
      video_id category [start_frame1 end_frame1 start_frame2 end_frame2 ...]

    Returns:
        {video_id: {'category': str, 'segments': [(start_frame, end_frame), ...]}}
    """
    p = Path(label_path)

    if p.suffix == '.parquet':
        return _load_gt_from_parquet(str(p))

    if p.is_dir():
        parquet_files = sorted(p.glob(f'{split}-*.parquet'))
        if not parquet_files:
            parquet_files = sorted(p.glob('*.parquet'))
        if parquet_files:
            dfs = [pd.read_parquet(f) for f in parquet_files]
            df = pd.concat(dfs, ignore_index=True)
            return _parse_gt_dataframe(df)

    return _load_gt_from_text(str(p))


def _load_gt_from_parquet(parquet_path: str) -> Dict[str, Dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    return _parse_gt_dataframe(df)


def _parse_gt_dataframe(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    gt = {}
    for _, row in df.iterrows():
        video_id = _extract_video_id(row['video_path'])
        category = row['category']
        segments = []
        if row['annotation_start1'] >= 0 and row['annotation_end1'] >= 0:
            segments.append((int(row['annotation_start1']), int(row['annotation_end1'])))
        if row['annotation_start2'] >= 0 and row['annotation_end2'] >= 0:
            segments.append((int(row['annotation_start2']), int(row['annotation_end2'])))
        gt[video_id] = {'category': category, 'segments': segments}
    return gt


def _load_gt_from_text(label_file: str) -> Dict[str, Dict[str, Any]]:
    gt = {}
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            video_id = parts[0]
            category = parts[1]
            segments = []
            frame_nums = parts[2:]
            for i in range(0, len(frame_nums) - 1, 2):
                sf = int(frame_nums[i])
                ef = int(frame_nums[i + 1])
                if sf >= 0 and ef >= 0:
                    segments.append((sf, ef))
            gt[video_id] = {'category': category, 'segments': segments}
    return gt


# ---------------------------------------------------------------------------
# 预测结果加载
# ---------------------------------------------------------------------------

def load_predictions(result_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    加载预测结果目录下的所有 JSON 文件

    每个 JSON 文件对应一个视频的检测结果（由 run.py 的 --output_file 生成）。
    video_id 从文件中的 video_path 字段提取 stem。

    Returns:
        {video_id: {'anomalies': [{'t_start', 't_end', 'confidence', 'anomaly_type'}],
                    'duration': float}}
    """
    preds = {}
    result_path = Path(result_dir)
    for json_file in result_path.glob('*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'video_path' not in data or 'anomalies' not in data:
            continue
        video_path = data.get('video_path', json_file.stem)
        video_id = _extract_video_id(video_path)
        preds[video_id] = {
            'video_path': video_path,
            'anomalies': data.get('anomalies', []),
            'duration': data.get('duration', 0),
            'frame_scores': data.get('frame_scores'),
            'timestamps': data.get('timestamps'),
        }
    return preds


# ---------------------------------------------------------------------------
# Level 1 — 帧级 AUC-ROC / AUPR / per-category AUC
# ---------------------------------------------------------------------------

def _get_total_frames(video_id, gt_entry, pred_entry, fps):
    # 原生评估使用解码器读取的帧数，避免 duration * fps 浮点取整丢失末帧。
    if pred_entry and pred_entry.get('source_frame_count'):
        return pred_entry['source_frame_count']
    if pred_entry and pred_entry['duration'] > 0:
        return int(pred_entry['duration'] * fps)
    if gt_entry['segments']:
        max_frame = max(ef for _, ef in gt_entry['segments'])
        return max_frame + fps
    return 0


def compute_frame_auc(
    gt: Dict[str, Dict],
    pred: Dict[str, Dict],
    fps: int = 30,
) -> Dict[str, Any]:
    """
    帧级 AUC-ROC + AUPR + per-category AUC

    AUC-ROC 是 UCF-Crime 的标准指标。
    AUPR 补充衡量极不均衡场景（异常帧 << 正常帧）下的检测能力。
    per-category AUC 暴露不同异常类别的检测难度差异。
    """
    all_gt_labels = []
    all_pred_scores = []
    per_video_auc = {}

    category_gt = {}
    category_pred = {}

    for video_id, gt_entry in gt.items():
        pred_entry = pred.get(video_id)
        video_fps = fps_for_video(fps, video_id)
        total_frames = _get_total_frames(video_id, gt_entry, pred_entry, video_fps)
        if total_frames == 0:
            continue

        gt_labels = np.zeros(total_frames, dtype=np.float32)
        for sf, ef in gt_entry['segments']:
            gt_labels[sf:min(ef + 1, total_frames)] = 1.0

        pred_scores = np.zeros(total_frames, dtype=np.float32)
        if pred_entry:
            frame_scores = pred_entry.get('frame_scores')
            pred_timestamps = pred_entry.get('timestamps')
            if frame_scores is not None and len(frame_scores) > 1:
                fs = np.array(frame_scores, dtype=np.float32)
                if pred_timestamps is not None and len(pred_timestamps) == len(fs):
                    sample_frames = np.array(pred_timestamps) * video_fps
                else:
                    sample_frames = np.linspace(0, total_frames - 1, len(fs))
                pred_scores = np.interp(
                    np.arange(total_frames), sample_frames, fs
                ).astype(np.float32)
            else:
                for anomaly in pred_entry['anomalies']:
                    start_f = int(anomaly['t_start'] * video_fps)
                    end_f = int(anomaly['t_end'] * video_fps)
                    conf = anomaly.get('confidence', 1.0)
                    start_f = max(0, min(start_f, total_frames - 1))
                    end_f = max(0, min(end_f, total_frames))
                    pred_scores[start_f:end_f] = np.maximum(
                        pred_scores[start_f:end_f], conf
                    )

        all_gt_labels.append(gt_labels)
        all_pred_scores.append(pred_scores)

        cat = gt_entry['category']
        if cat not in category_gt:
            category_gt[cat] = []
            category_pred[cat] = []
        category_gt[cat].append(gt_labels)
        category_pred[cat].append(pred_scores)

        if gt_labels.sum() > 0 and gt_labels.sum() < len(gt_labels):
            try:
                per_video_auc[video_id] = float(
                    roc_auc_score(gt_labels, pred_scores)
                )
            except ValueError:
                per_video_auc[video_id] = 0.0

    all_gt = np.concatenate(all_gt_labels)
    all_pred = np.concatenate(all_pred_scores)

    if all_gt.sum() == 0 or all_gt.sum() == len(all_gt):
        global_auc = 0.0
        global_aupr = 0.0
    else:
        global_auc = float(roc_auc_score(all_gt, all_pred))
        prec, rec, _ = precision_recall_curve(all_gt, all_pred)
        global_aupr = float(auc(rec, prec))

    per_category_auc = {}
    for cat in sorted(category_gt.keys()):
        if cat == 'Normal':
            continue
        cat_gt = np.concatenate(category_gt[cat])
        cat_pred = np.concatenate(category_pred[cat])
        if cat_gt.sum() > 0 and cat_gt.sum() < len(cat_gt):
            try:
                per_category_auc[cat] = float(roc_auc_score(cat_gt, cat_pred))
            except ValueError:
                per_category_auc[cat] = 0.0

    return {
        'auc': global_auc,
        'aupr': global_aupr,
        'per_video': per_video_auc,
        'per_category_auc': per_category_auc,
    }


# ---------------------------------------------------------------------------
# Level 2 — 时间段级 mAP
# ---------------------------------------------------------------------------

def _temporal_iou(pred_seg, gt_seg, fps):
    pred_start = pred_seg['t_start'] * fps
    pred_end = pred_seg['t_end'] * fps
    gt_start, gt_end = gt_seg

    overlap_start = max(pred_start, gt_start)
    overlap_end = min(pred_end, gt_end)
    overlap = max(0, overlap_end - overlap_start)

    union = (max(pred_end, gt_end) - min(pred_start, gt_start))
    if union <= 0:
        return 0.0
    return overlap / union


def compute_temporal_map(
    gt: Dict[str, Dict],
    pred: Dict[str, Dict],
    fps: int = 30,
    iou_thresholds: List[float] = None,
) -> Dict[str, Any]:
    """
    类别无关的辅助定位指标；保留既有匹配范围，只替换为 VadCLIP 非插值 AP。
    主评估入口使用 compute_vadclip_temporal_map 的类别敏感结果。
    """
    if iou_thresholds is None:
        iou_thresholds = [0.3, 0.5, 0.7]

    all_predictions = []
    all_gt_segments = []

    for video_id in sorted(gt):
        gt_entry = gt[video_id]
        for seg in gt_entry['segments']:
            all_gt_segments.append({'video_id': video_id, 'segment': seg})

        pred_entry = pred.get(video_id)
        if pred_entry:
            for anomaly in pred_entry['anomalies']:
                all_predictions.append({
                    'video_id': video_id,
                    'segment': anomaly,
                    'confidence': anomaly.get('confidence', 0.0),
                })

    if not all_predictions or not all_gt_segments:
        return {
            'mAP': 0.0,
            **{f'AP@{t}': 0.0 for t in iou_thresholds},
        }

    # 使用与主指标、VadCLIP 一致的置信度排序，不追加并列排序键。
    order = np.argsort(-np.asarray([x['confidence'] for x in all_predictions], dtype=float))
    all_predictions = [all_predictions[int(i)] for i in order]
    n_gt = len(all_gt_segments)
    results = {}

    for thresh in iou_thresholds:
        tp = np.zeros(len(all_predictions))
        fp = np.zeros(len(all_predictions))
        matched_gt = set()

        for i, p in enumerate(all_predictions):
            best_iou = 0.0
            best_gt_idx = -1

            for j, g in enumerate(all_gt_segments):
                if g['video_id'] != p['video_id']:
                    continue
                if j in matched_gt:
                    continue
                iou = _temporal_iou(p['segment'], g['segment'], fps_for_video(fps, p['video_id']))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= thresh and best_gt_idx >= 0:
                tp[i] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[i] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / n_gt
        precision = cum_tp / (cum_tp + cum_fp)

        ap = _compute_ap(precision, recall)
        results[f'AP@{thresh}'] = float(ap)

    results['mAP'] = float(np.mean([results[f'AP@{t}'] for t in iou_thresholds]))
    return results


def _compute_ap(precision, recall):
    """VadCLIP rank AP: sum(P_k * TP_k) / N_GT = sum(P_k * delta_recall_k).

    No precision envelope, trapezoidal integration or 11-point interpolation.
    Unrecalled GT remains in the recall denominator; duplicate detections add FP.
    """
    precision=np.asarray(precision,dtype=float)
    recall=np.asarray(recall,dtype=float)
    if precision.ndim!=1 or precision.shape!=recall.shape:
        raise ValueError('precision and recall must be equal-length vectors')
    if (not np.isfinite(precision).all() or not np.isfinite(recall).all()
            or np.any((precision<0)|(precision>1)) or np.any((recall<0)|(recall>1))
            or np.any(np.diff(recall)<0)):
        raise ValueError('Invalid precision/recall curve')
    return float(np.sum(precision*np.diff(np.r_[0.,recall])))


def compute_vadclip_temporal_map(gt, pred, fps=30, iou_thresholds=None,
                                 include_normal_videos=False, classes=ANOMALY_CLASSES):
    """Class-aware primary metric, sharing the web evaluator's exact implementation.

    Only the AP/matching protocol is adopted. Existing final detections and
    confidence scores are retained; VadCLIP's proposal generation is not rerun.
    """
    thresholds=tuple(THRESHOLDS if iou_thresholds is None else iou_thresholds)
    metrics,_=compute_vadclip_map(gt,pred,fps,thresholds=thresholds,classes=classes,
                                include_normal_videos=include_normal_videos)
    return dict(metrics,mAP=metrics['values']['AVG'],
        **{f'AP@{t:g}':metrics['values'][f'mAP@{t:.1f}'] for t in thresholds},
        protocol=dict(ap_formula='sum(precision_at_rank * TP) / N_GT',
            interpolation='none',sorting='numpy.argsort(-confidence)',
            class_aware=True,classes=list(classes),
            iou_thresholds=list(thresholds),include_normal_videos=include_normal_videos,
            interval='integer source frames [start,end)',proposal_source='existing final detections'))


# ---------------------------------------------------------------------------
# Level 3 — 视频级二分类指标
# ---------------------------------------------------------------------------

def compute_video_level(
    gt: Dict[str, Dict],
    pred: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    视频级二分类指标：accuracy / precision / recall / F1 / AUC
    """
    gt_labels = []
    pred_labels = []
    pred_scores = []

    for video_id, gt_entry in gt.items():
        is_abnormal = len(gt_entry['segments']) > 0
        gt_labels.append(1 if is_abnormal else 0)

        pred_entry = pred.get(video_id)
        if pred_entry and pred_entry['anomalies']:
            pred_labels.append(1)
            max_conf = max(a['confidence'] for a in pred_entry['anomalies'])
            pred_scores.append(max_conf)
        else:
            pred_labels.append(0)
            pred_scores.append(0.0)

    gt_arr = np.array(gt_labels)
    pred_arr = np.array(pred_labels)
    score_arr = np.array(pred_scores)

    tp = ((gt_arr == 1) & (pred_arr == 1)).sum()
    fp = ((gt_arr == 0) & (pred_arr == 1)).sum()
    fn = ((gt_arr == 1) & (pred_arr == 0)).sum()
    tn = ((gt_arr == 0) & (pred_arr == 0)).sum()

    accuracy = (tp + tn) / len(gt_arr) if len(gt_arr) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if gt_arr.sum() > 0 and gt_arr.sum() < len(gt_arr):
        video_auc = float(roc_auc_score(gt_arr, score_arr))
    else:
        video_auc = 0.0

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(video_auc),
    }


# ---------------------------------------------------------------------------
# Level 4 — 异常类别分类准确率
# ---------------------------------------------------------------------------

def compute_category_accuracy(
    gt: Dict[str, Dict],
    pred: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    异常类别分类准确率（只评估异常视频）
    """
    correct = 0
    total = 0
    per_category = {}

    for video_id, gt_entry in gt.items():
        if gt_entry['category'] == 'Normal':
            continue
        gt_cat = gt_entry['category']
        if gt_cat not in per_category:
            per_category[gt_cat] = {'correct': 0, 'total': 0}
        per_category[gt_cat]['total'] += 1
        total += 1

        pred_entry = pred.get(video_id)
        if pred_entry and pred_entry['anomalies']:
            pred_types = [a.get('anomaly_type', '') for a in pred_entry['anomalies']]
            if gt_cat in pred_types:
                correct += 1
                per_category[gt_cat]['correct'] += 1

    accuracy = correct / total if total > 0 else 0.0
    return {'accuracy': float(accuracy), 'per_category': per_category}


# ---------------------------------------------------------------------------
# 主评估入口
# ---------------------------------------------------------------------------

def evaluate(label_path, result_dir, fps=30, split="test", *, video_root=None, legacy_fixed_fps=False):
    gt = load_ground_truth(label_path, split=split)
    pred = load_predictions(result_dir)

    matched = set(gt.keys()) & set(pred.keys())
    gt_only = set(gt.keys()) - set(pred.keys())
    pred_only = set(pred.keys()) - set(gt.keys())
    _log.info(f"GT videos: {len(gt)}, Predicted videos: {len(pred)}, Matched: {len(matched)}")
    if gt_only:
        _log.info(f"  GT only (missing predictions): {len(gt_only)}")
    if pred_only:
        _log.info(f"  Pred only (no GT):             {len(pred_only)}")

    if not matched:
        _log.warning("No matching videos found between GT and predictions.")
        return {}

    metadata = {}
    if not legacy_fixed_fps:
        # 每次评估重新读取原视频；缺失或无法解码时失败，不回退 JSON 清单或 30 FPS。
        metadata = source_video_metadata({k: pred[k]['video_path'] for k in sorted(matched)},
                                         video_root=video_root, relative_to=result_dir)
        fps = {k: row['fps'] for k, row in metadata.items()}
        for k, row in metadata.items():
            pred[k]['source_frame_count'] = row['frame_count']
    _log.info('FPS source: %s', 'original video files' if metadata else f'explicit fixed {fps}')
    gt_matched = {k: gt[k] for k in matched}
    pred_matched = {k: pred[k] for k in matched}

    n_abnormal = sum(1 for v in gt_matched.values() if v['segments'])
    n_normal = len(gt_matched) - n_abnormal
    _log.info(f"  Abnormal: {n_abnormal}, Normal: {n_normal}")

    _log.info("--- Level 1: Frame-level AUC-ROC / AUPR ---")
    frame_results = compute_frame_auc(gt_matched, pred_matched, fps)
    _log.info(f"  Global AUC-ROC: {frame_results['auc']:.4f}")
    _log.info(f"  Global AUPR:    {frame_results['aupr']:.4f}")
    if frame_results['per_category_auc']:
        _log.info("  Per-category AUC-ROC:")
        for cat, cat_auc in sorted(frame_results['per_category_auc'].items()):
            _log.info(f"    {cat:20s}: {cat_auc:.4f}")

    _log.info("--- Level 2: Temporal Segment mAP ---")
    map_results = compute_vadclip_temporal_map(gt_matched, pred_matched, fps)
    global_map_results = compute_temporal_map(gt_matched, pred_matched, fps)
    _log.info(f"  mAP:    {map_results['mAP']:.4f}")
    for threshold in THRESHOLDS:
        _log.info('  mAP@%.1f: %.4f',threshold,map_results[f'AP@{threshold:g}'])

    _log.info("--- Level 3: Video-level Metrics ---")
    video_results = compute_video_level(gt_matched, pred_matched)
    _log.info(f"  Accuracy:  {video_results['accuracy']:.4f}")
    _log.info(f"  Precision: {video_results['precision']:.4f}")
    _log.info(f"  Recall:    {video_results['recall']:.4f}")
    _log.info(f"  F1:        {video_results['f1']:.4f}")
    _log.info(f"  AUC:       {video_results['auc']:.4f}")

    _log.info("--- Level 4: Category Accuracy ---")
    cat_results = compute_category_accuracy(gt_matched, pred_matched)
    _log.info(f"  Overall: {cat_results['accuracy']:.4f}")
    for cat, stats in sorted(cat_results['per_category'].items()):
        _log.info(f"  {cat:20s}: {stats['correct']}/{stats['total']}")

    return {
        'video_metadata': dict(source='original_video_files' if metadata else 'legacy_fixed_fps',
                               videos=metadata),
        'frame_auc': frame_results,
        'temporal_map': map_results,
        'temporal_map_class_agnostic': dict(global_map_results,ap_method='vadclip_rank_noninterpolated',
            iou_thresholds=[.3,.5,.7],class_aware=False,interval='continuous source frames'),
        'video_level': video_results,
        'category': cat_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCF-Crime Evaluation")
    parser.add_argument("--label_path", type=str, required=True,
                        help="Path to GT: parquet file, parquet directory, or text file")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Directory containing prediction JSON files")
    parser.add_argument("--fps", type=float, default=30,
                        help="Fixed FPS used only with --legacy_fixed_fps")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use when label_path is a directory")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save evaluation report to JSON")
    parser.add_argument("--video_root", default=None,
                        help="Original video root, used to resolve relocated result video_path values")
    parser.add_argument("--legacy_fixed_fps", action="store_true",
                        help="Explicitly reproduce historical fixed-FPS scores without probing videos")
    args = parser.parse_args()

    results = evaluate(args.label_path, args.result_dir, args.fps, args.split,
                       video_root=args.video_root, legacy_fixed_fps=args.legacy_fixed_fps)

    if args.output_file and results:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {args.output_file}")
