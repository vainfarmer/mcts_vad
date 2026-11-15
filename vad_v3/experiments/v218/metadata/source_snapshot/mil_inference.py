"""
MIL-based 异常检测推理

替代 VLM scan + MCTS，直接用训练好的 MIL 模型产出 snippet-level 异常分数
→ 阈值化 → NMS → 段生成

用法:
  python mil_inference.py --video /path/to/video.mp4 --model model_mil.pth
"""
import argparse
import json
import sys
import time
import cv2
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mil_model import MILAnomalyScorer


def extract_clip_features(video_path, model, preprocess, device, num_segments=32):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps

    if total_frames < num_segments:
        cap.release()
        return None, duration, fps

    segment_len = total_frames // num_segments
    from PIL import Image
    frames_list = []
    for seg_idx in range(num_segments):
        mid = seg_idx * segment_len + segment_len // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        if not ret:
            frames_list.append(np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    images = [preprocess(Image.fromarray(f)) for f in frames_list]
    batch = torch.stack(images).to(device)
    with torch.no_grad():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy(), duration, fps


def scores_to_segments(scores, duration, threshold=0.5, min_gap=1.0, min_dur=0.5):
    T = len(scores)
    snippet_dur = duration / T

    segments = []
    start = None
    seg_scores = []
    for i in range(T):
        if scores[i] >= threshold:
            if start is None:
                start = i
            seg_scores.append(scores[i])
        else:
            if start is not None:
                segments.append({
                    't_start': start * snippet_dur,
                    't_end': (i) * snippet_dur,
                    'confidence': float(np.mean(seg_scores)),
                })
                start = None
                seg_scores = []
    if start is not None:
        segments.append({
            't_start': start * snippet_dur,
            't_end': T * snippet_dur,
            'confidence': float(np.mean(seg_scores)),
        })

    # Merge close segments
    merged = []
    for seg in segments:
        if merged and (seg['t_start'] - merged[-1]['t_end']) < min_gap:
            merged[-1]['t_end'] = seg['t_end']
            merged[-1]['confidence'] = max(merged[-1]['confidence'], seg['confidence'])
        else:
            merged.append(seg)

    # Filter short
    merged = [s for s in merged if (s['t_end'] - s['t_start']) >= min_dur]
    return merged


def classify_with_videomae(video_path, device='mps'):
    try:
        from videomae_classifier import VideoMAEClassifier
        vmae = VideoMAEClassifier(device=device)
        if vmae.model is None:
            return 'unknown', 0.0

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if len(frames) < 16:
            return 'unknown', 0.0

        all_frames = np.array(frames[::max(1, len(frames)//100)])  # sample ~100 frames
        result = vmae.classify(all_frames, tta=True)
        return result['category'], result['confidence']
    except Exception:
        return 'unknown', 0.0


def detect_video(video_path, mil_model, clip_model, clip_preprocess,
                 device='mps', threshold=0.5, num_segments=32):
    t0 = time.time()

    # 1. Extract CLIP features
    features, duration, fps = extract_clip_features(
        video_path, clip_model, clip_preprocess, device, num_segments)
    if features is None:
        return {'video_path': str(video_path), 'duration': 0, 'is_normal': True,
                'anomalies': [], 'frame_scores': None, 'timestamps': None}

    # 2. MIL scoring
    feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    mil_model.eval()
    with torch.no_grad():
        scores = mil_model(feat_tensor).squeeze(0).cpu().numpy()

    # 3. Adaptive thresholding
    mean_s = np.mean(scores)
    std_s = np.std(scores)
    adaptive_thresh = max(threshold, mean_s + 1.0 * std_s)

    # 4. Segment generation
    segments = scores_to_segments(scores, duration, threshold=adaptive_thresh,
                                  min_gap=duration/num_segments, min_dur=0.5)

    # 5. Classification (VideoMAE)
    is_normal = len(segments) == 0
    category = 'Normal'
    cat_confidence = 0.0
    if not is_normal:
        category, cat_confidence = classify_with_videomae(video_path, device)
        if category == 'Normal':
            is_normal = True
            segments = []

    # 6. Frame scores (interpolate snippet scores to timestamps)
    snippet_dur = duration / num_segments
    timestamps = [i * snippet_dur for i in range(num_segments)]
    frame_scores = scores.tolist()

    anomalies = []
    for seg in segments:
        anomalies.append({
            't_start': round(seg['t_start'], 2),
            't_end': round(seg['t_end'], 2),
            'confidence': round(seg['confidence'], 3),
            'anomaly_type': category if category != 'Normal' else 'unknown',
        })

    elapsed = time.time() - t0
    return {
        'video_path': str(video_path),
        'duration': round(duration, 2),
        'is_normal': is_normal,
        'anomalies': anomalies,
        'frame_scores': frame_scores,
        'timestamps': timestamps,
        'processing_time': round(elapsed, 2),
    }
