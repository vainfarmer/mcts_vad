"""
帧级评分模块 (FrameScorer)

从 pipeline 抽离的 CLIP 依赖评分逻辑 — 生成逐帧连续异常分数 + 边界曲线修正：
  - get_category_scores        : 类别相关的相对 CLIP 分数（per video+category 缓存）
  - compute_frame_scores       : 段置信度 overlay + CLIP base + 高斯平滑 -> 平滑分数曲线
  - clip_score_curve_correction: 用半峰值阈值对合并后段做边界微调

依赖 anomaly_expert（CLIP 模型 + 文本特征）和 VADConfig（评分参数）。
这是 v33 帧级 AUC=0.694 的核心，行为严格保持不变。
"""

from __future__ import annotations
from bisect import bisect_left

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import vad_logger as _vl
    _log = _vl.get("vad.scorer")
except Exception:
    import logging
    _log = logging.getLogger("vad.scorer")


class FrameScorer:
    """CLIP-based frame-level anomaly scoring + boundary correction."""

    def __init__(self, anomaly_expert, config, anomaly_categories):
        self.anomaly_expert = anomaly_expert
        self.config = config
        self.anomaly_categories = anomaly_categories

    def get_category_scores(self, video_data, category):
        """计算类别相关的相对 CLIP 分数（per video+category 缓存到 video_data）。"""
        cache_key = f'_cat_scores_{category}'
        cached = video_data.get(cache_key)
        if cached is not None:
            return cached
        precomputed = video_data.get('clip_features')
        if (precomputed is None or self.anomaly_expert.model is None
                or category is None or category == 'Normal'
                or category not in self.anomaly_categories or not HAS_TORCH):
            return None
        cat_idx = self.anomaly_categories.index(category)
        with torch.no_grad():
            logit_scale = self.anomaly_expert.model.logit_scale.exp().detach().clamp(max=100.0)
            similarity = precomputed @ self.anomaly_expert.text_features.T * logit_scale
            probs = torch.softmax(similarity, dim=-1).cpu().numpy()
            cat_scores = probs[:, cat_idx]
            normal_scores = probs[:, -1]
            result = cat_scores / (cat_scores + normal_scores + 1e-8)
        video_data[cache_key] = result
        return result

    def compute_frame_scores(self, video_data, anomalies, category):
        """段置信度为主 + CLIP base 为辅的混合评分，高斯平滑后返回。

        v33 方法：clip_base = cat_scores * clip_base_weight(0.35)，
        段区间 overlay 置信度，sigma = 1.5 * fps_est 平滑。
        """
        timestamps = video_data['timestamps']
        n_frames = len(timestamps)
        if n_frames == 0:
            return None

        precomputed = video_data.get('clip_features')
        if precomputed is None or self.anomaly_expert.model is None or not HAS_TORCH:
            return None

        from scipy.ndimage import gaussian_filter1d

        # CLIP base signal (low amplitude — provides discrimination for undetected frames)
        cat_scores = self.get_category_scores(video_data, category)
        if cat_scores is None:
            with torch.no_grad():
                logit_scale = self.anomaly_expert.model.logit_scale.exp().clamp(max=100.0)
                sim = precomputed @ self.anomaly_expert.text_features.T * logit_scale
                probs = torch.softmax(sim, dim=-1).cpu().numpy()
                cat_scores = 1.0 - probs[:, -1]

        sc = self.config.scoring
        base_signal = cat_scores * sc.clip_base_weight

        # Channel 2: VideoMAE sliding window signal (additive)
        mae_centers = video_data.get('videomae_window_centers')
        mae_scores = video_data.get('videomae_window_scores')
        if mae_centers and mae_scores and sc.videomae_weight > 0 and len(mae_centers) >= 2:
            mae_interp = np.interp(np.arange(n_frames), mae_centers, mae_scores)
            base_signal = base_signal + mae_interp * sc.videomae_weight

        # Channel 3: Motion energy signal (additive)
        motion = video_data.get('motion_energy')
        if motion is not None and sc.motion_weight > 0 and len(motion) == n_frames:
            motion_max = np.max(motion)
            if motion_max > 0:
                base_signal = base_signal + (motion / motion_max) * sc.motion_weight

        # Overlay segment detections (high amplitude, unchanged)
        for a in anomalies:
            lo = bisect_left(timestamps, a['t_start'])
            hi = bisect_left(timestamps, a['t_end'])
            base_signal[lo:min(hi, n_frames)] = np.maximum(
                base_signal[lo:min(hi, n_frames)], a['confidence'])

        # Gaussian smoothing (unchanged)
        fps_est = max(1, n_frames / max(1, video_data['duration']))
        sigma = max(sc.gaussian_sigma_min, sc.gaussian_sigma_factor * fps_est)
        smoothed = gaussian_filter1d(base_signal, sigma=sigma)

        return smoothed.tolist()

    def clip_score_curve_correction(self, anomalies, video_data, category, max_dur=None):
        """合并后的边界修正：用半峰值阈值扫描收缩段边界（v52起只收缩不扩张）。

        v53: 新增 max_dur 参数——若某段时长已 <= max_dur（已被上游阶段正确校准），
        直接跳过此修正，不再对已校准好的候选段做二次收缩。这是修复 v52 回归的
        关键一环：v52 对所有非Normal候选无条件应用此步骤，导致本已校准好的边界
        被过度收缩（如 Arrest007: 0.90x -> 0.64x 时长比）。
        """
        relative_scores = self.get_category_scores(video_data, category)
        if relative_scores is None or len(anomalies) == 0:
            return anomalies

        timestamps = video_data['timestamps']
        duration = video_data['duration']
        fps = max(1, len(timestamps) / max(1, duration))
        smooth_w = max(3, int(self.config.scoring.curve_smooth_seconds * fps))

        if len(relative_scores) >= smooth_w:
            kernel = np.ones(smooth_w) / smooth_w
            smoothed = np.convolve(relative_scores, kernel, mode='same')
        else:
            smoothed = relative_scores.copy()

        global_mean = np.mean(smoothed)
        global_std = np.std(smoothed)

        corrected = []
        for a in anomalies:
            a = a.copy()

            # v53: already within the category's expected duration — leave it alone.
            if max_dur is not None and (a['t_end'] - a['t_start']) <= max_dur:
                corrected.append(a)
                continue

            lo = bisect_left(timestamps, a['t_start'])
            hi = bisect_left(timestamps, a['t_end'])
            if hi <= lo + 2:
                corrected.append(a)
                continue

            core = smoothed[lo:hi]
            core_mean = np.mean(core)
            core_max = np.max(core)

            # Only correct if the core has meaningful signal
            if core_max < global_mean + self.config.scoring.curve_core_signal_floor * global_std:
                corrected.append(a)
                continue

            # Half-max threshold: find where scores cross 50% of core peak
            half_max = (core_max + global_mean) * self.config.scoring.curve_half_max_weight

            # v52: Contract-only correction. The previous version scanned outward
            # past the original boundary and could EXPAND it whenever noise stayed
            # above half_max beyond the edge — this compounded the ~4x over-wide
            # boundary bug found in V51 analysis. Now we only scan inward from the
            # original edges, so this stage can shrink a segment but never widen it.
            new_lo = lo
            for i in range(lo, hi):
                if smoothed[i] >= half_max:
                    new_lo = i
                    break
            else:
                new_lo = lo

            new_hi = hi
            for i in range(hi - 1, lo - 1, -1):
                if smoothed[i] >= half_max:
                    new_hi = i + 1
                    break
            else:
                new_hi = hi

            new_start = timestamps[max(0, min(new_lo, len(timestamps) - 1))]
            new_end = timestamps[max(0, min(new_hi - 1, len(timestamps) - 1))]

            if new_end - new_start < 1.0:
                corrected.append(a)
                continue

            if abs(new_start - a['t_start']) > 1.0 or abs(new_end - a['t_end']) > 1.0:
                _log.info(f"    [ScoreCurve] {a['t_start']:.1f}-{a['t_end']:.1f}s "
                          f"-> {new_start:.1f}-{new_end:.1f}s "
                          f"(half_max={half_max:.3f})")

            a['t_start'] = max(0, new_start)
            a['t_end'] = min(new_end, duration)
            corrected.append(a)

        return corrected


__all__ = ["FrameScorer"]
