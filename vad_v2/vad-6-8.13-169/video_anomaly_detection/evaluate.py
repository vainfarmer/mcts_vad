"""
UCF-Crime 评估模块

提供四级评估指标，全面衡量异常检测与定位的性能：

  Level 1 — 帧级 AUC-ROC / AUPR（UCF-Crime 标准指标）
    将所有视频的帧级标签和预测分数拼接后计算全局 AUC 和 AUPR。
    AUC-ROC 是 UCF-Crime benchmark 的官方评估方式。
    AUPR 在正负样本极不均衡时（异常帧远少于正常帧）比 AUC-ROC 更灵敏。
    同时报告 per-category AUC，暴露不同异常类别的检测难度差异。

  Level 2 — 时间段级 mAP（temporal IoU @ 0.3/0.5/0.7）
    评估时间定位的精度。IoU 计算在帧空间进行（预测秒数 × fps → 帧号）。
    三个阈值从松到严，反映不同精度要求下的性能。

  Level 3 — 视频级 accuracy / precision / recall / F1 / AUC
    二分类指标：视频是否包含异常。用最高置信度异常段的 confidence 作为视频级分数。

  Level 4 — 异常类别分类准确率
    检查预测的异常类型是否匹配 GT 类别。只评估异常视频（Normal 视频跳过）。

用法：
  # parquet 目录（HuggingFace 数据集格式，自动读取 test split）
  python evaluate.py \
    --label_path data/ucf-crime/data/ \
    --result_dir data/results/ \
    --fps 30

  # 单个 parquet 文件
  python evaluate.py \
    --label_path data/test-00000-of-00001.parquet \
    --result_dir data/results/

  # 传统纯文本格式（向后兼容）
  python evaluate.py \
    --label_path data/ucf_crime_labels.txt \
    --result_dir data/results/

注意：--fps 应设为 GT 标注使用的原始视频帧率（通常 30fps），
不是 Pipeline 的采样帧率。GT 标注的帧号基于原始视频帧率。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

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
        total_frames = _get_total_frames(video_id, gt_entry, pred_entry, fps)
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
                    sample_frames = np.array(pred_timestamps) * fps
                else:
                    sample_frames = np.linspace(0, total_frames - 1, len(fs))
                pred_scores = np.interp(
                    np.arange(total_frames), sample_frames, fs
                ).astype(np.float32)
            else:
                for anomaly in pred_entry['anomalies']:
                    start_f = int(anomaly['t_start'] * fps)
                    end_f = int(anomaly['t_end'] * fps)
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
    时间段级 mAP（Mean Average Precision）

    在不同 IoU 阈值下计算 AP，衡量时间定位精度。
    """
    if iou_thresholds is None:
        iou_thresholds = [0.3, 0.5, 0.7]

    all_predictions = []
    all_gt_segments = []

    for video_id, gt_entry in gt.items():
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

    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
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
                iou = _temporal_iou(p['segment'], g['segment'], fps)
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
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max()
    return ap / 11.0


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

def evaluate(label_path, result_dir, fps=30, split="test"):
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
    map_results = compute_temporal_map(gt_matched, pred_matched, fps)
    _log.info(f"  mAP:    {map_results['mAP']:.4f}")
    _log.info(f"  AP@0.3: {map_results['AP@0.3']:.4f}")
    _log.info(f"  AP@0.5: {map_results['AP@0.5']:.4f}")
    _log.info(f"  AP@0.7: {map_results['AP@0.7']:.4f}")

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
        'frame_auc': frame_results,
        'temporal_map': map_results,
        'video_level': video_results,
        'category': cat_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCF-Crime Evaluation")
    parser.add_argument("--label_path", type=str, required=True,
                        help="Path to GT: parquet file, parquet directory, or text file")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Directory containing prediction JSON files")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video frame rate for frame-level evaluation")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use when label_path is a directory")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save evaluation report to JSON")
    args = parser.parse_args()

    results = evaluate(args.label_path, args.result_dir, args.fps, args.split)

    if args.output_file and results:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {args.output_file}")
