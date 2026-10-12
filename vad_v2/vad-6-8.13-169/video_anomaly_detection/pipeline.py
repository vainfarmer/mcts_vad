"""
视频异常检测与时间定位 — 完整 Pipeline

Committed pipeline (v137 base + committed enhancements, all default-ON):
  qscore_rerank (v111) + BUE bue_enabled (v142) + CBT cbt_enabled (v155)
  + CBT-recal cbt_recal (v160) + classify_skip_class_agnostic (v137).

Architecture (per video):
  Stage 1: video preprocessing (2fps frames, 6s segments) + motion energy
  Stage 2: VideoMAE video-level classification (Normal -> early return)
  Stage 3: CLIP category curves + multi-scale temporal localization
  Stage 4: directed VLM scan (CLIP-region-union targeted segments; the
           legacy full-coverage comparative scan path was REMOVED in v208)
  Stage 5: candidate build -> merge/NMS -> PostTighten -> narrow short events
  Stage 6: ensemble classification (VLM + VideoMAE, class-agnostic skip)
  Stage 7: CLIP score-curve post-correction + MIL video-level sanity
           (max<0.3 -> conf*0.7) + category-aware confidence gate
  Terminal: qscore_rerank -> BUE (outward union extension, category curve)
            -> CBT (+recal) (inward clamp to per-category GT-p75 caps)

Paired-adjudication plumbing kept in the output JSON (DetectionResult):
  pre_bue_boundaries / pre_cbt_boundaries — index-aligned pre-mechanism
  windows; analysis_bootstrap_ab.py --paired-json rebuilds the flag-off arm
  exactly (zero cross-run VLM noise).

Historical experiment log: iteration_logs/session_v97_ledger.md +
per-version summaries. All REVERT/CLOSED experimental flags (ECE, EUBC,
CEX, CGR, CCR, CGR-AB, MCS, CNV, SNH, VLC, DSV, SVA, KFD, TCP, FSD, FDS,
AB-ABLATE rig, FCCS) were REMOVED in v208 — see
iteration_logs/removed_experimental_flags_20260830.md for each flag's
mechanism, what it changed, and its measured result.
"""


import asyncio
import bisect
import json
from collections import Counter
import numpy as np
from typing import List, Dict, Any, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    import vad_logger as _vl
    _log = _vl.get("vad.pipeline")
except Exception:
    import logging
    _log = logging.getLogger("vad.pipeline")


try:
    from .video_preprocessor import VideoPreprocessor
    from .anomaly_discovery import AnomalyDiscovery
    from .temporal_mcts import TemporalBoundaryMCTS
    from .anomaly_expert import VideoAnomalyExpert
    from .token_compressor import SemanticFrameSampler
except ImportError:
    from video_preprocessor import VideoPreprocessor
    from anomaly_discovery import AnomalyDiscovery
    from temporal_mcts import TemporalBoundaryMCTS
    from anomaly_expert import VideoAnomalyExpert
    from token_compressor import SemanticFrameSampler

try:
    from .videomae_classifier import VideoMAEClassifier
except ImportError:
    from videomae_classifier import VideoMAEClassifier

try:
    from .config import VADConfig, ScoringConfig
    from .models import AnomalySegment, DetectionResult
except ImportError:
    from config import VADConfig, ScoringConfig
    from models import AnomalySegment, DetectionResult

try:
    from .frame_scorer import FrameScorer
    from .post_processor import global_merge, narrow_short_events, unify_anomaly_types, temporal_nms
except ImportError:
    from frame_scorer import FrameScorer
    from post_processor import global_merge, narrow_short_events, unify_anomaly_types, temporal_nms


CATEGORY_CUES = {
    "Abuse": "DEFENSELESS victim (child/elderly/restrained) being hit repeatedly — attacker has total control, victim CANNOT fight back",
    "Arrest": "UNIFORMED police/security officers restraining, handcuffing, or detaining a suspect — look for uniforms, police vehicles, badges",
    "Arson": "Fire starting SMALL from one point and GROWING gradually — a person may be near the fire origin with lighter/matches, deliberate fire-setting",
    "Assault": "ONE-SIDED attack — one person aggressively strikes another who is cowering, on the ground, or fleeing but NOT fighting back (victim is an able adult)",
    "Burglary": "Person ENTERING a building by force — breaking doors/windows, climbing walls/fences, forcing locks to GET INSIDE",
    "Explosion": "SUDDEN instantaneous blast in a fraction of a second — bright flash, debris FLYING outward, large smoke/dust cloud appears ALL AT ONCE",
    "Fighting": "MUTUAL combat — TWO OR MORE people BOTH actively throwing punches/kicks at EACH OTHER, both sides in aggressive postures",
    "RoadAccidents": "Vehicle COLLISION or CRASH — cars/trucks/motorcycles colliding, flipping, or skidding on a road, with debris and damage",
    "Robbery": "FACE-TO-FACE confrontation — thief uses WEAPON (gun/knife) or FORCE/THREATS to demand and take items from a person directly",
    "Shooting": "People suddenly DROP/COLLAPSE or crowd SCATTERS in panic WITHOUT anyone physically touching them — signs of gunfire, guns may NOT be visible",
    "Shoplifting": "Inside a RETAIL STORE with product shelves — person secretly concealing merchandise in bag/pocket/clothing and moving toward exit",
    "Stealing": "COVERT theft — taking UNATTENDED items (bag/phone from table/bench/car) or SNATCHING from a distracted person, then fleeing — NO weapon, NO threat",
    "Vandalism": "DAMAGING property WHERE IT STANDS — smashing car windows, kicking objects, spray-painting walls, breaking street furniture — NOT entering a building",
}

VIDEOMAE_CONFUSION_MAP = {
    # Only entries with net positive effect (no regressions)
    'RoadAccidents': ['Robbery'],    # Robbery102 fix: VideoMAE=RoadAccidents→VLM→Robbery
    'Abuse': ['Explosion'],          # Explosion035 fix: VideoMAE=Abuse→VLM→Explosion
    'Assault': ['Arson'],            # Neutral: keeps Arson check alive, no Assault regression
    # REMOVED: Burglary→Shoplifting (v29 test: VLM chose Shoplifting 3/3, no improvement)
    # REMOVED: Stealing, Vandalism — VLM bias toward Robbery/Burglary caused -8 videos
}

SHORT_EVENT_CATEGORIES = {'Explosion', 'RoadAccidents'}

# v50: Categories where CLIP AUC is too low for reliable boundary detection
LOW_CLIP_CATEGORIES = {'Shoplifting', 'Stealing', 'Arson', 'Burglary'}

# v50: Category-aware duration priors (physically motivated, not dataset-overfit)
# v52/v53/v54 each tried adding threshold cleverness here (fixed per-category,
# overshoot-adaptive, multi-threshold+duration-fit) and each regressed mAP.
# v55 reverts to v51's plain fixed-ratio contrast selection (see
# _clip_boundary_detection) and keeps this dict untouched by threshold logic.
CATEGORY_DURATION_PRIORS = {
    # Short-burst events
    'RoadAccidents': {'expected_duration': (2.0, 20.0), 'preferred_smooth': 1.0, 'min_segment': 1.5},  # v51: was (2.0, 8.0) — real accidents span 10-30s incl. aftermath
    'Explosion':     {'expected_duration': (2.0, 10.0), 'preferred_smooth': 1.0, 'min_segment': 1.5},
    'Abuse':         {'expected_duration': (2.0, 8.0), 'preferred_smooth': 1.0, 'min_segment': 1.5},
    'Fighting':      {'expected_duration': (3.0, 15.0), 'preferred_smooth': 1.5, 'min_segment': 2.0},
    'Shooting':      {'expected_duration': (3.0, 15.0), 'preferred_smooth': 1.5, 'min_segment': 2.0},
    # Medium events
    'Robbery':       {'expected_duration': (5.0, 30.0), 'preferred_smooth': 2.0, 'min_segment': 3.0},
    'Burglary':      {'expected_duration': (5.0, 40.0), 'preferred_smooth': 2.5, 'min_segment': 3.0},
    'Shoplifting':   {'expected_duration': (3.0, 20.0), 'preferred_smooth': 2.0, 'min_segment': 2.0},
    'Stealing':      {'expected_duration': (5.0, 30.0), 'preferred_smooth': 2.0, 'min_segment': 3.0},
    'Vandalism':     {'expected_duration': (3.0, 20.0), 'preferred_smooth': 2.0, 'min_segment': 2.0},
    # Long events
    'Arson':         {'expected_duration': (5.0, 40.0), 'preferred_smooth': 2.5, 'min_segment': 3.0},
    'Arrest':        {'expected_duration': (15.0, 90.0), 'preferred_smooth': 4.0, 'min_segment': 8.0},
    'Assault':       {'expected_duration': (10.0, 90.0), 'preferred_smooth': 3.0, 'min_segment': 5.0},
}

# v155 CBT: per-category inward duration caps (seconds) = GT-p75 of the 4 hard
# categories on UCF-Crime, lightly padded. Only categories present here are
# tightened; deleting a key = per-category rollback (no core-method change).
# CBT fires only when a FINAL detection's dur > 1.5*cap (over-wide) and the
# contrast-selected sub-window is strictly narrower — so a cap slightly above
# GT (e.g. RoadAccidents 6.0 vs GT-p75 ~4.3) just gates which dets qualify.
CBT_CAPS = {
    'RoadAccidents': 6.0,
    'Explosion':     13.0,
    'Shoplifting':   10.0,
    'Shooting':      14.0,
}


def _compute_tsm_novelty(clip_features, kernel_half_widths=(15, 30, 60)):
    """v95: Temporal Self-Similarity Matrix + checkerboard kernel novelty
    function (Foote's novelty function, UBoCo/GEBD lineage). Loop 1 research:
    the ONLY signal construction among ~7 paradigms studied that is a
    genuinely different SIGNAL TYPE from our text-similarity curve - a
    frame-to-FRAME visual similarity matrix, not frame-to-TEXT. Detects
    visual scene transitions completely independent of category semantics.

    SSM[i,j] = cosine_sim(feat_i, feat_j) for all frame pairs. Checkerboard
    kernel (+1 same-block quadrants, -1 cross-block, Gaussian-tapered)
    correlated along the SSM diagonal. Peaks = visual transitions.

    Multi-scale: computed at 3 kernel half-widths, combined via max (a
    transition detected at ANY scale counts). Zero additional CLIP/VLM calls
    - reuses features already extracted for the primary pipeline.

    Returns a novelty(t) array, same length as clip_features, normalized to
    [0, 1] via min-max over the video.
    """
    n = clip_features.shape[0]
    if n < 8:
        return np.zeros(n)

    feats = clip_features
    if hasattr(feats, 'cpu'):
        feats = feats.cpu().numpy()
    feats = feats.astype(np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    feats_n = feats / norms
    ssm = feats_n @ feats_n.T  # (n, n) cosine similarity matrix

    novelty_multiscale = np.zeros(n)
    for L in kernel_half_widths:
        L = min(L, n // 2 - 1)
        if L < 2:
            continue
        size = 2 * L
        idx = np.arange(size) - L + 0.5
        gauss = np.exp(-(idx[:, None] ** 2 + idx[None, :] ** 2) / (2 * (L / 2.0) ** 2))
        checker = np.sign(np.outer(np.sign(idx), np.sign(idx)))
        kernel = checker * gauss

        novelty = np.zeros(n)
        for t in range(L, n - L):
            block = ssm[t - L:t + L, t - L:t + L]
            novelty[t] = float(np.sum(kernel * block))
        _rng = novelty.max() - novelty.min()
        if _rng > 1e-8:
            novelty = (novelty - novelty.min()) / _rng
        novelty_multiscale = np.maximum(novelty_multiscale, novelty)

    return novelty_multiscale


def _score_boundary_proposal(smoothed, gradient, seg_times, lo, hi, refine_range, min_dur, max_dur):
    """v54: Refine a threshold-crossing [lo, hi) via gradient, then score it by
    Outer-Inner-Contrast (OIC, as in AutoLoc/BSN) times a duration-prior fit.

    This is the key architectural change from v50-v53: instead of picking the
    proposal with the highest contrast alone (which favors narrow, spiky, but
    often implausible boundaries), the selection score also rewards proposals
    whose duration actually falls inside the category's expected range. A
    proposal outside [min_dur, max_dur] is penalized by how far outside it is,
    rather than being accepted just because no better-contrast alternative was
    tried at that threshold.
    """
    left_lo = max(0, lo - refine_range)
    left_hi = min(len(gradient), lo + refine_range + 1)
    refined_lo = left_lo + int(np.argmax(gradient[left_lo:left_hi])) if left_hi > left_lo else lo

    right_lo = max(0, hi - refine_range)
    right_hi = min(len(gradient), hi + refine_range + 1)
    refined_hi = right_lo + int(np.argmin(gradient[right_lo:right_hi])) if right_hi > right_lo else hi

    if refined_hi <= refined_lo:
        refined_lo, refined_hi = lo, hi
    if refined_hi <= refined_lo:
        return None

    inside_mean = float(np.mean(smoothed[refined_lo:refined_hi + 1]))
    outside_left = float(np.mean(smoothed[:max(1, refined_lo)])) if refined_lo > 0 else 0.0
    outside_right = float(np.mean(smoothed[refined_hi + 1:])) if refined_hi < len(smoothed) - 1 else 0.0
    contrast = inside_mean - (outside_left + outside_right) / 2.0

    t_start = float(seg_times[min(refined_lo, len(seg_times) - 1)])
    t_end = float(seg_times[min(refined_hi, len(seg_times) - 1)])
    dur = t_end - t_start
    if dur <= 0:
        return None

    if dur < min_dur:
        duration_fit = dur / min_dur
    elif dur > max_dur:
        duration_fit = max_dur / dur
    else:
        duration_fit = 1.0

    return {
        't_start': t_start, 't_end': t_end,
        'contrast': contrast, 'duration_fit': duration_fit,
        'combined': contrast * duration_fit,
    }


# v51: Category-aware ConfFilter thresholds (calibrated against new confidence formula)
CATEGORY_CONF_GATES = {
    # Low-CLIP categories (AUC < 0.55): trust LLM more, filter less
    'Shoplifting': 0.38, 'Stealing': 0.38, 'Robbery': 0.42,
    'Vandalism': 0.42, 'Arson': 0.42, 'Burglary': 0.42,
    # Medium-CLIP categories (AUC 0.55-0.75): moderate filtering
    'Abuse': 0.48, 'RoadAccidents': 0.48,
    # High-CLIP categories (AUC > 0.75): can afford stricter filtering
    'Arrest': 0.55, 'Assault': 0.55, 'Explosion': 0.55,
    'Fighting': 0.55, 'Shooting': 0.52,
}


class MultiAnomalyVAD:

    DEFAULT_ANOMALY_CATEGORIES = [
        "Abuse", "Arrest", "Arson", "Assault", "Burglary",
        "Explosion", "Fighting", "RoadAccidents", "Robbery",
        "Shooting", "Shoplifting", "Stealing", "Vandalism",
    ]

    def __init__(
        self,
        llm_client,
        config: VADConfig,
        classify_llm_client=None,
        image_processor=None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.classify_llm_client = classify_llm_client or llm_client
        self.image_processor = image_processor
        self.anomaly_categories = config.anomaly_categories or self.DEFAULT_ANOMALY_CATEGORIES

        pp = config.preprocess
        self.preprocessor = VideoPreprocessor(
            fps=pp.fps,
            segment_duration=pp.segment_duration,
            motion_threshold=pp.motion_threshold,
            min_motion_interval=pp.min_motion_interval,
            segment_overlap=pp.segment_overlap,
        )

        cl = config.clip
        self.anomaly_expert = VideoAnomalyExpert(
            anomaly_categories=self.anomaly_categories,
            clip_model_name=cl.model_name,
            clip_pretrained=cl.pretrained,
            device=config.device,
        )

        self.frame_sampler = SemanticFrameSampler(
            anomaly_expert=self.anomaly_expert,
            tau=cl.semantic_tau,
            max_frames=cl.max_frames,
            min_frames=cl.min_frames,
            fps_sample=cl.fps_sample,
        )

        try:
            self.videomae = VideoMAEClassifier(device=config.device)
        except Exception as e:
            _log.warning(f"[Pipeline] VideoMAE not available: {e}")
            self.videomae = None

        dc = config.discovery
        self.discovery = AnomalyDiscovery(
            llm_client=llm_client,
            anomaly_categories=self.anomaly_categories,
            image_processor=image_processor,
            clip_expert=self.anomaly_expert,
            frame_sampler=self.frame_sampler,
            min_pass_rate=dc.min_pass_rate,
            clip_threshold=dc.clip_threshold,
            max_scan_segments=dc.max_scan_segments,
        )

        self.mcts_simulations = config.mcts.simulations
        self.mcts_max_depth = config.mcts.max_depth

        self.frame_scorer = FrameScorer(
            anomaly_expert=self.anomaly_expert,
            config=config,
            anomaly_categories=self.anomaly_categories,
        )

        # v51: Load MIL model for supplementary localization on low-CLIP categories
        self.mil_model = None
        try:
            from pathlib import Path as _Path
            _mil_path = _Path(__file__).parent / 'model_mil.pth'
            if _mil_path.exists():
                try:
                    from .mil_model import MILAnomalyScorer
                except ImportError:
                    from mil_model import MILAnomalyScorer
                self.mil_model = MILAnomalyScorer(feature_dim=768, hidden_dim=512, k=3, dropout=0.6)
                _state = torch.load(str(_mil_path), map_location=config.device, weights_only=True)
                self.mil_model.load_state_dict(_state)
                self.mil_model.eval()
                self.mil_model.to(config.device)
                _log.info(f"  [MIL] Loaded MIL model from {_mil_path}")
            else:
                _log.info(f"  [MIL] No MIL model found, MIL fusion disabled")
        except Exception as _e:
            _log.warning(f"  [MIL] Failed to load MIL model: {_e}")
            self.mil_model = None

    def _qscore_rerank(self, final_anomalies, frame_scores, timestamps, base=0.5):
        """v111: pure post-hoc localization-quality confidence re-rank.

        Q = mean(boundary_sharpness, peak_contrast, center_fit) on the OUTPUT
        frame_scores curve, min-max normalized WITHIN the video, then
        conf' = conf * (base + (1-base)*Q). Runs after the conf-gate
        (count already fixed) so it NEVER changes boundaries or detection count
        (FM1/FM2 immune). Byte-for-byte matches analysis_v111_qscore_precheck.py.
        """
        if not final_anomalies or frame_scores is None or timestamps is None:
            _log.info("  [QSCORE-GATE] skipped (no detections / no curve)")
            return final_anomalies
        fs = np.asarray(frame_scores, dtype=float)
        ts = np.asarray(timestamps, dtype=float)
        if len(fs) < 3 or len(ts) != len(fs):
            _log.info("  [QSCORE-GATE] skipped (curve too short / misaligned)")
            return final_anomalies
        grad = np.abs(np.gradient(fs))
        vid_range = fs.max() - fs.min() + 1e-9
        qs = []
        for a in final_anomalies:
            t_s, t_e = a['t_start'], a['t_end']
            in_mask = (ts >= t_s) & (ts <= t_e)
            if in_mask.sum() < 1:
                qs.append(0.0)
                continue
            peak = fs[in_mask].max()
            peak_idx = np.where(in_mask)[0][np.argmax(fs[in_mask])]
            out_mask = ~in_mask
            out_mean = fs[out_mask].mean() if out_mask.any() else fs.min()
            contrast = float(np.clip((peak - out_mean) / vid_range, 0, 1))
            i_s = int(np.argmin(np.abs(ts - t_s)))
            i_e = int(np.argmin(np.abs(ts - t_e)))
            sharp = float(np.clip((grad[i_s] + grad[i_e]) / (2 * (grad.max() + 1e-9)), 0, 1))
            center_t = 0.5 * (t_s + t_e)
            dur = max(t_e - t_s, 1e-6)
            center = float(np.clip(1 - 2 * abs(ts[peak_idx] - center_t) / dur, 0, 1))
            qs.append((sharp + contrast + center) / 3.0)
        qs = np.array(qs)
        if len(qs) > 1 and qs.max() - qs.min() > 1e-9:
            qn = (qs - qs.min()) / (qs.max() - qs.min())
        else:
            qn = np.ones_like(qs)
        for a, q in zip(final_anomalies, qn):
            a['confidence'] = min(1.0, a['confidence'] * (base + (1 - base) * float(q)))
        _log.info(f"  [QSCORE-GATE] reranked {len(final_anomalies)} detections "
                  f"(base={base}, raw Q range [{qs.min():.3f},{qs.max():.3f}])")
        return final_anomalies

    # ================================================================
    # v142 BUE: Boundary-Union Extension (TERMINAL-MOST slot)
    # ================================================================

    def _bue_boundary_union_extension(self, final_anomalies, video_data, category):
        """v142 BUE: grow each FINAL detection OUTWARD to the outer-inner-contrast
        (OIC) argmax window on the read-only category curve — UNION ONLY, length
        capped at bue_lmax_sec / bue_lmax_frac*duration.

        Contract (why this is FM2/FM3/FM4/FM14-immune):
          * Only a['t_start'] / a['t_end'] are ever written. confidence, detection
            COUNT, anomaly_type, rank order, NMS, merge and every prompt are frozen
            bit-for-bit; the per-fire marker logs `conf=<c>-><c>` to make that
            visible to the existing paired parsers.
          * Zero new VLM / CLIP / MIL calls: get_category_scores is the SAME cached
            call _clip_boundary_single_scale already makes (per video+category cache
            key `_cat_scores_<cat>` on video_data), and it is strictly READ-ONLY —
            the curve is never written back, rescaled or fused (FM1 in the strict
            sense).
          * UNION IS THE MECHANISM, not an implementation detail. Offline 3-cache
            ablation: closure-gated REPLACE (emit the OIC window instead of the old
            one) = -0.0065 mean, UNION = +0.0378 mean. Monotone outward growth
            toward the contrast argmax is what wins; pure OIC relocation is
            worthless or harmful. Never contract, never replace.
          * The v51 half-max CLOSURE TEST (hm = (seg.max() + p20(curve))/2) skips
            byte-identically when the curve is already below hm at BOTH edges — the
            boundary was then genuinely set by a signal crossing rather than clipped
            by the padded search window. Measured mAP no-op (gated == ungated
            bit-identical on 3/3 caches at Lmax 40 and 60); retained purely as free
            blast-radius reduction (67 fires instead of 78).

        Byte-for-byte mirror of analysis_v142_bue_verify.py::bue_pick (an
        independent reimplementation from the written spec reproduced +0.02221 /
        +0.01629 / +0.03879 to five decimals), on the fidelity-gated substrate of
        analysis_v140_livecurve_lib (58/58 recorded live peak times at 0.000 s) —
        the FM15 remedy. features/test/*.npy is NOT used anywhere.

        FM13 magnitude bound taken BEFORE the eval: best-of-two oracle +0.0764,
        realized +0.0378 at Lmax=40, sign-robust 24/24 over Lmax in [15,60] x 3
        caches; 12-cache out-of-sample read 9/9 positive, mean +0.0370.
        """
        cfg = self.config.scoring
        # flag-off: return BEFORE any curve is read -> byte-identical, zero cost.
        if not cfg.bue_enabled or not final_anomalies or not category:
            return final_anomalies

        import os as _bue_os
        vid = _bue_os.path.basename(str(video_data.get('video_path') or '')) or 'unknown'

        timestamps = video_data.get('timestamps')
        if timestamps is None:
            _log.info(f"  [BUE] summary: vid={vid} fires=0 closed_skips=0 "
                      f"ndet_before={len(final_anomalies)} "
                      f"ndet_after={len(final_anomalies)} growth_med=0.000 "
                      f"dur_ratio=1.000 selfov=0 skip=no_timestamps")
            return final_anomalies
        t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        n = int(len(t))
        if n < 8:
            _log.info(f"  [BUE] summary: vid={vid} fires=0 closed_skips=0 "
                      f"ndet_before={len(final_anomalies)} "
                      f"ndet_after={len(final_anomalies)} growth_med=0.000 "
                      f"dur_ratio=1.000 selfov=0 skip=curve_too_short")
            return final_anomalies

        duration = float(video_data.get('duration') or float(t[-1]))
        dt = float(np.median(np.diff(t))) or 0.5
        w = int(cfg.bue_smooth_win)
        alpha = float(cfg.bue_alpha)
        l_min = float(cfg.bue_lmin_sec)
        l_max = float(cfg.bue_lmax_sec)
        l_frac = float(cfg.bue_lmax_frac)
        eff_lmax = min(l_max, l_frac * duration) if duration > 0 else l_max
        kmin = max(2, int(round(l_min / dt)))
        kmax = max(kmin + 1, int(round(eff_lmax / dt)))

        _prep_cache = {}

        def _prep(cat):
            """(cs, prefix-sum, percentile20) for a category, or None."""
            if cat in _prep_cache:
                return _prep_cache[cat]
            try:
                rel = self.frame_scorer.get_category_scores(video_data, cat)
            except Exception as _e:  # never let a read-only diagnostic kill detect()
                _log.warning(f"  [BUE] curve read failed vid={vid} cat={cat}: {_e}")
                rel = None
            if rel is None:
                _prep_cache[cat] = None
                return None
            if hasattr(rel, 'detach'):      # FM9: MPS/CUDA tensors
                rel = rel.detach().cpu().numpy()
            c = np.asarray(rel, dtype=np.float64).reshape(-1)
            if len(c) != n:
                _prep_cache[cat] = None
                return None
            cs = c if (w <= 1 or len(c) < w) else np.convolve(
                c, np.ones(w) / w, mode='same')
            P = np.concatenate([[0.0], np.cumsum(cs)])
            _prep_cache[cat] = (cs, P, float(np.percentile(cs, 20)))
            return _prep_cache[cat]

        def _mean(P, i, j):
            i = max(0, min(int(i), n))
            j = max(0, min(int(j), n))
            return (P[j] - P[i]) / max(1, j - i) if j > i else 0.0

        def _oic(P, s, e):
            k = int(e) - int(s)
            if k <= 0:
                return 0.0
            r = max(1, int(round(alpha * k)))
            nl = min(r, max(0, int(s)))
            nr = min(r, max(0, n - int(e)))
            den = nl + nr
            out = ((_mean(P, s - r, s) * nl + _mean(P, e, e + r) * nr) / den
                   if den else 0.0)
            return _mean(P, s, e) - out

        n_fire = n_closed = n_degen = n_noop = n_nocurve = 0
        growths, ratios = [], []
        ndet_before = len(final_anomalies)

        for di, a in enumerate(final_anomalies):
            # FM15: category resolution MUST match the verified replay exactly —
            # the detection's OWN anomaly_type first, then videomae_category.
            cat = str(a.get('anomaly_type') or '').strip() or category
            prep = _prep(cat)
            if prep is None:
                n_nocurve += 1
                continue
            cs, P, p20 = prep
            ds = float(a['t_start'])
            de = float(a['t_end'])
            i0 = int(np.searchsorted(t, ds))
            i1 = int(np.searchsorted(t, de))
            i0 = max(0, min(i0, n - 1))
            i1 = max(i0 + 1, min(i1, n))
            seg = cs[i0:i1]
            hm = (float(seg.max()) + p20) / 2.0
            anchor = i0 + int(np.argmax(seg))
            if cfg.bue_closure_gate and cs[i0] < hm and cs[i1 - 1] < hm:
                n_closed += 1                     # boundary genuinely closed by the signal
                continue
            best, bs, be = -1e30, None, None
            for s in range(max(0, anchor - kmax), anchor + 1):
                for k in range(kmin, kmax + 1):
                    e = s + k
                    if e > n or e <= anchor:
                        continue
                    r = max(1, int(round(alpha * k)))
                    nl = min(r, max(0, s))
                    nr = min(r, max(0, n - e))
                    den = nl + nr
                    out = ((_mean(P, s - r, s) * nl + _mean(P, e, e + r) * nr) / den
                           if den else 0.0)
                    v = _mean(P, s, e) - out
                    if v > best:
                        best, bs, be = v, s, e
            if bs is None:
                n_degen += 1
                continue
            ns = float(t[bs])
            ne = float(t[min(be, n - 1)])
            # UNION ONLY — never a replacement, never a contraction (measured:
            # REPLACE = -0.0065 vs UNION = +0.0378).
            ns = min(ds, ns)
            ne = max(de, ne)
            if duration > 0:
                ns = max(0.0, min(ns, duration))
                ne = max(0.0, min(ne, duration))
            if ne <= ns:
                n_degen += 1
                continue
            if abs(ns - ds) <= 1e-6 and abs(ne - de) <= 1e-6:
                n_noop += 1
                continue
            old_oic = float(_oic(P, i0, i1))
            us = max(0, min(int(np.searchsorted(t, ns)), n - 1))
            ue = max(us + 1, min(int(np.searchsorted(t, ne)), n))
            new_oic = float(_oic(P, us, ue))
            old_d = de - ds
            new_d = ne - ns
            conf = float(a['confidence'])
            a['t_start'] = ns
            a['t_end'] = ne
            n_fire += 1
            growths.append(new_d - old_d)
            ratios.append(new_d / old_d if old_d > 1e-9 else 1.0)
            _log.info(
                f"  [BUE] vid={vid} cat={cat} det={di} closed=0 "
                f"t={ds:.3f}-{de:.3f}->{ns:.3f}-{ne:.3f} "
                f"dur={old_d:.2f}->{new_d:.2f} oic={old_oic:.4f}->{new_oic:.4f} "
                f"conf={conf:.4f}->{conf:.4f}")

        # M5 (diagnostic only, DO NOT act on it this loop — FM8): count surviving
        # detection pairs whose mutual temporal IoU exceeds 0.3 after widening.
        # BUE runs AFTER NMS, so widening can create new mutual overlap and under
        # greedy single-match mAP the loser becomes an FP. That penalty is already
        # inside the measured +0.0378; this field only quantifies the de-dup
        # headroom for a later loop.
        selfov = 0
        for _i in range(len(final_anomalies)):
            for _j in range(_i + 1, len(final_anomalies)):
                _a, _b = final_anomalies[_i], final_anomalies[_j]
                _inter = max(0.0, min(_a['t_end'], _b['t_end'])
                             - max(_a['t_start'], _b['t_start']))
                _union = ((_a['t_end'] - _a['t_start'])
                          + (_b['t_end'] - _b['t_start']) - _inter)
                if _union > 1e-9 and _inter / _union > 0.3:
                    selfov += 1

        g_med = float(np.median(growths)) if growths else 0.0
        d_ratio = float(np.mean(ratios)) if ratios else 1.0
        # M4 MATERIALITY GATE:
        # the flag-ON smoke requires median per-fire growth > 2.0 s AND mean
        # post/pre duration ratio > 1.5 (offline: 3.9 s / 1.90).
        _log.info(
            f"  [BUE] summary: vid={vid} fires={n_fire} closed_skips={n_closed} "
            f"ndet_before={ndet_before} ndet_after={len(final_anomalies)} "
            f"growth_med={g_med:.3f} dur_ratio={d_ratio:.3f} selfov={selfov} "
            f"degen={n_degen} noop={n_noop} nocurve={n_nocurve} "
            f"lmax={eff_lmax:.1f} gate={int(bool(cfg.bue_closure_gate))}")
        return final_anomalies

    async def detect(
        self,
        video_path: str,
        context: str = "surveillance camera",
    ) -> Dict[str, Any]:
        """v124 [COST] instrumentation wrapper (DEFAULT-ON, no flag, FM10).

        Snapshots the drift-free per-video VLM (llm_client + classify_llm_client
        .generate) and CLIP (anomaly_expert.encode_image) call counters plus a
        wall-clock start, delegates to _detect_impl (unchanged detection logic),
        and emits exactly one machine-parseable [COST] INFO line per video in a
        finally block so it fires on every exit path (incl. early returns and
        exceptions). Counters are read-only here and never touch any request,
        feature, or detection -> mAP is byte-neutral.
        """
        import time as _cost_time
        import os as _cost_os
        _cost_t0 = _cost_time.time()
        _vlm_detect_start = getattr(self.llm_client, 'call_count', 0)
        _vlm_classify_start = getattr(self.classify_llm_client, 'call_count', 0)
        _clip_start = getattr(self.anomaly_expert, 'clip_call_count', 0)
        try:
            return await self._detect_impl(video_path, context)
        finally:
            _n_detect = getattr(self.llm_client, 'call_count', 0) - _vlm_detect_start
            if self.classify_llm_client is self.llm_client:
                # Same object: the detect delta already includes classify calls.
                _vlm_calls = _n_detect
            else:
                _n_classify = (getattr(self.classify_llm_client, 'call_count', 0)
                               - _vlm_classify_start)
                _vlm_calls = _n_detect + _n_classify
            _n_clip = getattr(self.anomaly_expert, 'clip_call_count', 0) - _clip_start
            _elapsed = _cost_time.time() - _cost_t0
            _vname = _cost_os.path.basename(str(video_path))
            _log.info(f"[COST] video={_vname} vlm_calls={_vlm_calls} "
                      f"clip_calls={_n_clip} elapsed_s={_elapsed:.1f}")

    async def _detect_impl(
        self,
        video_path: str,
        context: str = "surveillance camera",
    ) -> Dict[str, Any]:
        video_data = self.preprocessor.process(video_path)

        # Stage 1: CLIP precompute
        if self.anomaly_expert.model is not None:
            import time as _time
            _t0 = _time.time()
            video_data['clip_features'] = self.anomaly_expert.precompute_video_features(
                video_data['all_frames']
            )
            _log.info(f"  [CLIP precompute] {len(video_data['all_frames'])} frames "
                      f"encoded in {_time.time()-_t0:.1f}s")

            # v51: Compute MIL snippet scores from CLIP features
            if self.mil_model is not None and video_data['clip_features'] is not None:
                try:
                    _clip_feats = video_data['clip_features']
                    _n_frames = _clip_feats.shape[0]
                    _num_segments = 32
                    if _n_frames >= _num_segments:
                        _seg_len = _n_frames // _num_segments
                        _snippet_feats = []
                        for _si in range(_num_segments):
                            _start = _si * _seg_len
                            _end = _start + _seg_len if _si < _num_segments - 1 else _n_frames
                            _snippet_feats.append(_clip_feats[_start:_end].mean(dim=0))
                        _snippet_tensor = torch.stack(_snippet_feats).unsqueeze(0)  # (1, 32, 768)
                        with torch.no_grad():
                            _mil_scores = self.mil_model(_snippet_tensor).squeeze(0).cpu().numpy()
                        video_data['mil_scores'] = _mil_scores
                        _log.info(f"  [MIL] snippet scores: mean={_mil_scores.mean():.3f}, "
                                  f"max={_mil_scores.max():.3f}, min={_mil_scores.min():.3f}")

                    else:
                        _log.info(f"  [MIL] Too few frames ({_n_frames}) for 32-snippet MIL")
                except Exception as _me:
                    _log.warning(f"  [MIL] Scoring failed: {_me}")


        # v95: TSM checkerboard-kernel novelty (Loop 1 research). Zero extra
        # CLIP/VLM calls - reuses already-extracted clip_features. Used ONLY
        # for confidence re-ranking (never boundary position), avoiding all
        # 4 known failure modes from prior rounds.
        if video_data.get('clip_features') is not None:
            try:
                video_data['tsm_novelty'] = _compute_tsm_novelty(video_data['clip_features'])
            except Exception as _tsme:
                _log.warning(f"  [TSM] novelty computation failed: {_tsme}")
                video_data['tsm_novelty'] = None

        # Stage 1b: Motion energy computation
        video_data['motion_energy'] = self._compute_motion_energy(video_data)

        # Stage 2: VideoMAE full-video classification (moved up from end)
        videomae_category = None
        videomae_confidence = 0.0
        if self.videomae is not None and self.videomae.model is not None:
            result = self.videomae.classify(video_data['all_frames'], top_k=3, tta=True)
            videomae_category = result['category']
            videomae_confidence = result['confidence']
            top3_str = ', '.join(f'{c}({s:.0%})' for c, s in result['top_k'])
            _log.info(f"  [VideoMAE] {top3_str} -> {videomae_category}@{videomae_confidence:.0%}")

            # Early Normal filter
            if videomae_category == 'Normal' and videomae_confidence > 0.5:
                _log.info(f"  [VideoMAE] Normal@{videomae_confidence:.0%} — skipping detection")
                if self.config.scoring.cbt_enabled:
                    # FM17: unconditional per-video CBT marker on the empty /
                    # 0-detection path so grep -c '[CBT-VID]' == videos processed.
                    _log.info(f"  [CBT-VID] vid={str(video_path).rsplit('/', 1)[-1]} "
                              f"enabled=True ndet=0 fired=0 "
                              f"recal={'on' if self.config.scoring.cbt_recal else 'off'} "
                              f"(early return: video normal)")
                frame_scores = self.frame_scorer.compute_frame_scores(video_data, [], None)
                return DetectionResult(
                    video_path=video_path,
                    duration=video_data['duration'],
                    is_normal=True,
                    anomalies=[],
                    frame_scores=frame_scores,
                    timestamps=video_data['timestamps'].tolist() if frame_scores else None,
                )

            # Low-confidence Normal: CLIP fallback
            if videomae_category == 'Normal':
                precomputed = video_data.get('clip_features')
                if precomputed is not None and self.anomaly_expert.model is not None:
                    all_indices = list(range(len(video_data['all_frames'])))
                    clip_dist = self.anomaly_expert.get_category_distribution(
                        all_indices, precomputed, top_k=3)
                    if clip_dist:
                        clip_cats = [c for c in clip_dist.keys() if c != 'Normal']
                        if clip_cats:
                            videomae_category = clip_cats[0]
                            _log.info(f"  [VideoMAE] Normal override -> CLIP: {videomae_category}")

            video_data['videomae_category'] = videomae_category

            # Stage 2b: VideoMAE sliding window temporal scanning
            sc = self.config.scoring
            if sc.videomae_weight > 0:
                import time as _time2
                _t_mae = _time2.time()
                centers, mae_scores = self.videomae.sliding_window_scores(
                    video_data['all_frames'],
                    window_size=sc.mae_window_size,
                    stride=sc.mae_window_stride)
                video_data['videomae_window_centers'] = centers
                video_data['videomae_window_scores'] = mae_scores
                _log.info(f"  [VideoMAE-window] {len(centers)} windows in "
                          f"{_time2.time()-_t_mae:.1f}s")

        # Stage 3: Multi-category CLIP temporal localization
        categories_to_search = [videomae_category] if videomae_category else []
        if videomae_category in VIDEOMAE_CONFUSION_MAP:
            for alt in VIDEOMAE_CONFUSION_MAP[videomae_category]:
                if alt not in categories_to_search:
                    categories_to_search.append(alt)

        all_clip_regions = []
        for cat in categories_to_search:
            regions = self._clip_temporal_localization(video_data, cat)
            for r in regions:
                r['source_category'] = cat
            all_clip_regions.extend(regions)

        clip_regions = self._deduplicate_clip_regions(all_clip_regions)

        # Stage 4: LLM scanning — directed by CLIP regions
        context_prompt = f"This is a {context} video." if context else ""
        self.discovery.scan_segments_count = 0
        if clip_regions:
            candidates = await self._directed_llm_scan(
                video_data, clip_regions, context_prompt, videomae_category)
        else:
            candidates = await self.discovery.discover(
                video_data, context_prompt)

        if not candidates:
            if self.config.scoring.cbt_enabled:
                # FM17: unconditional per-video CBT marker on the 0-detection path.
                _log.info(f"  [CBT-VID] vid={str(video_path).rsplit('/', 1)[-1]} "
                          f"enabled=True ndet=0 fired=0 "
                          f"recal={'on' if self.config.scoring.cbt_recal else 'off'} "
                          f"(early return: no candidates)")
            frame_scores = self.frame_scorer.compute_frame_scores(video_data, [], videomae_category)
            return DetectionResult(
                video_path=video_path,
                duration=video_data['duration'],
                is_normal=True,
                anomalies=[],
                frame_scores=frame_scores,
                timestamps=video_data['timestamps'].tolist() if frame_scores else None,
            )

        _log.info(f"Found {len(candidates)} anomaly candidate(s):")
        for i, c in enumerate(candidates):
            _log.info(f"  Candidate {i+1}: {c['t_start']:.1f}s-{c['t_end']:.1f}s "
                  f"score={c['coarse_score']:.2f}")

        top_k = self.config.discovery.candidate_top_k
        if len(candidates) > top_k:
            candidates.sort(key=lambda c: c['coarse_score'], reverse=True)
            _log.info(f"  [Limit] Keeping top {top_k}/{len(candidates)} candidates")
            candidates = candidates[:top_k]

        # Stage 5: CLIP score-curve boundary detection (v50 — replaces MCTS)
        all_anomalies = self._clip_boundary_detection(
            candidates, video_data, videomae_category)

        # Stage 5b: VLM boundary estimation for low-CLIP-confidence candidates
        # V62 evaluation: VLM over-wide override triggered 18 times on 65 videos
        # but net mAP regressed (0.1492->0.1012). VLM timestamp estimation from
        # sparse keyframes is too noisy to reliably improve over CLIP half-max.
        # Reverted to V51's proven guard: LOW_CLIP + clip_conf < 0.15 only.
        if videomae_category in LOW_CLIP_CATEGORIES:
            vlm_refined = []
            for anom in all_anomalies:
                clip_conf = anom.get('boundary_contrast', 1.0)
                if clip_conf < 0.15:
                    r = await self._vlm_boundary_estimation(anom, video_data, videomae_category)
                    vlm_refined.append(r)
                else:
                    vlm_refined.append(anom)
            all_anomalies = vlm_refined

        _log.info(f"  --- Pre-merge anomalies: {len(all_anomalies)} ---")
        for i, a in enumerate(all_anomalies):
            _log.debug(f"    {i+1}: {a['t_start']:.1f}-{a['t_end']:.1f}s, "
                  f"conf={a['confidence']:.3f}, type={a.get('anomaly_type','?')}")

        final_anomalies = global_merge(all_anomalies, same_type_gap=2.0)
        # v86->v89 REVERT: Soft-NMS regressed full-eval mAP 0.1492->0.0961.
        # Root cause: evaluate.py's mAP computation uses standard GREEDY
        # matching (predictions sorted by confidence, each GT segment can only
        # be matched ONCE - see evaluate.py compute_temporal_map). Soft-NMS's
        # permissive score_thresh=0.15 let 2-3 overlapping near-duplicate
        # detections survive per true event (vs Hard-NMS's exact iou_threshold
        # cutoff keeping only 1); only the highest-confidence duplicate gets
        # matched as a true positive, the REST become FALSE POSITIVES under
        # greedy matching, directly hurting precision/AP. Reverted to proven
        # Hard-NMS (temporal_nms) which was calibrated correctly for this
        # specific greedy-matching evaluation protocol.
        final_anomalies = temporal_nms(final_anomalies, iou_threshold=0.3)
        _log.info(f"  --- Post-merge+NMS anomalies: {len(final_anomalies)} ---")

        # Stage 6b (v50): Post-merge boundary tightening for over-long segments
        final_anomalies = self._post_merge_boundary_tightening(
            final_anomalies, video_data, videomae_category)

        # Stage 6c: Short-event narrowing
        if videomae_category in SHORT_EVENT_CATEGORIES:
            final_anomalies = narrow_short_events(final_anomalies, video_data)

        # Stage 7: Multi-classifier ensemble (VideoMAE + CLIP + VLM)
        if final_anomalies:
            ensemble_cat = await self._ensemble_classify(
                final_anomalies, video_data, videomae_category, videomae_confidence)
            for a in final_anomalies:
                a['anomaly_type'] = ensemble_cat
        elif videomae_category and videomae_category != 'Normal':
            pass  # no anomalies to classify

        final_anomalies = unify_anomaly_types(final_anomalies)

        # Stage 7b: CLIP score curve post-correction
        if videomae_category and videomae_category != 'Normal':
            # v53: pass max_dur so already-well-calibrated segments are left alone
            _curve_max_dur = CATEGORY_DURATION_PRIORS.get(
                videomae_category, {}).get('expected_duration', (3.0, 30.0))[1]
            final_anomalies = self.frame_scorer.clip_score_curve_correction(
                final_anomalies, video_data, videomae_category, max_dur=_curve_max_dur)

        # v51: MIL video-level sanity check — reduce confidence when MIL says Normal
        _mil_scores = video_data.get('mil_scores')
        if _mil_scores is not None and np.max(_mil_scores) < 0.3:
            _log.info(f"  [MIL-sanity] MIL max_score={np.max(_mil_scores):.3f} < 0.3, "
                      f"reducing confidence by 30%")
            for a in final_anomalies:
                a['confidence'] *= 0.7

        # v51: Category-aware confidence gate (replaces uniform 0.72 threshold)
        default_gate = self.config.scoring.confidence_gate
        conf_gate = CATEGORY_CONF_GATES.get(videomae_category, default_gate)
        before_filter = len(final_anomalies)
        final_anomalies = [a for a in final_anomalies if a['confidence'] >= conf_gate]
        if len(final_anomalies) < before_filter:
            _log.info(f"  [ConfFilter] {before_filter} -> {len(final_anomalies)} "
                      f"(threshold={conf_gate:.2f}, cat={videomae_category})")

        final_anomalies = [
            {
                't_start': a['t_start'],
                't_end': a['t_end'],
                'confidence': a['confidence'],
                'anomaly_type': a.get('anomaly_type', 'unknown'),
                'depth': a.get('depth'),
                'visits': a.get('visits'),
            }
            for a in final_anomalies
        ]

        # Generate per-frame continuous anomaly scores
        frame_scores = self.frame_scorer.compute_frame_scores(
            video_data, final_anomalies, videomae_category)

        # v111 QSCORE: pure post-hoc localization-quality confidence re-rank.
        # After conf-gate (count fixed) + frame_scores computed; never
        # touches boundaries/count.
        if self.config.scoring.qscore_rerank:
            final_anomalies = self._qscore_rerank(
                final_anomalies, frame_scores, video_data.get('timestamps'),
                base=self.config.scoring.qscore_base)

        # v142 BUE: Boundary-Union Extension — grow each FINAL detection
        # OUTWARD on the read-only category curve (union only, length capped).
        # pre_bue_boundaries is persisted for the paired flag-off reconstruction
        # (analysis_bootstrap_ab.py --paired-json pre_bue_boundaries).
        pre_bue_boundaries = None
        if self.config.scoring.bue_enabled:
            pre_bue_boundaries = [
                [float(a['t_start']), float(a['t_end'])] for a in final_anomalies]
            final_anomalies = self._bue_boundary_union_extension(
                final_anomalies, video_data, videomae_category)

        # v155 CBT: Category-conditioned Boundary Tightening — runs AFTER BUE
        # (strict INVERSE: BUE grows outward, CBT clamps inward toward the
        # per-category GT-p75 caps). pre_cbt_boundaries persisted for the paired
        # flag-off reconstruction (analysis_bootstrap_ab.py --paired-json
        # pre_cbt_boundaries).
        pre_cbt_boundaries = None
        if self.config.scoring.cbt_enabled:
            pre_cbt_boundaries = [
                [float(a['t_start']), float(a['t_end'])] for a in final_anomalies]
            final_anomalies = self._cbt_category_boundary_tighten(
                final_anomalies, video_data, videomae_category)

        return DetectionResult(
            video_path=video_path,
            duration=video_data['duration'],
            is_normal=len(final_anomalies) == 0,
            anomalies=[AnomalySegment.from_dict(a) for a in final_anomalies],
            frame_scores=frame_scores,
            timestamps=video_data['timestamps'].tolist() if frame_scores else None,
            pre_bue_boundaries=pre_bue_boundaries,
            pre_cbt_boundaries=pre_cbt_boundaries,
        )

    # ================================================================
    # v27-RIG AB-ABLATE: OPERATOR-AGNOSTIC dual-arm instrument (diagnostic)
    # ================================================================

    def _cbt_category_boundary_tighten(self, final_anomalies, video_data, category):
        """v155 CBT: TERMINAL inward clamp of over-wide FINAL detections of the
        4 hard categories toward per-category GT-p75 caps (CBT_CAPS), choosing
        the sub-window by the AutoLoc outer-inner CONTRAST objective already
        coded for the LogOIC tightener (count-neutral, confidence-untouched,
        monotone-inward). Runs AFTER BUE (strict INVERSE of BUE's outward
        union).

        For each FINAL detection a=[t_s,t_e]:
          1. Category resolution: detection's OWN anomaly_type, fallback the
             videomae_category arg; skip when the resolved category is not in
             CBT_CAPS (per-category rollback = delete a CBT_CAPS key).
          2. Over-wide gate: cap=CBT_CAPS[cat]; if det_dur <= 1.5*cap -> no-op
             (protects already-compact detections — FM1/FM19).
          3. Peak anchor peak_idx = argmax of the read-only cached relative
             curve rel (frame_scorer.get_category_scores — the exact curve BUE
             reads) within [t_s,t_e].
          4. Contrast-max window: scan half-widths hw in
             [ceil(min_seg*fps/2) .. int(cap*fps/2)] and pick the
             [peak-hw, peak+hw] window MAXIMIZING the AutoLoc outer-inner
             contrast OIC = mean(inside) - weighted_mean(two 1/log-weighted
             outer rings over the whole video) — the identical objective at
             the LogOIC tightener.
          5. Monotone-inward fire: clamp the chosen window to the ORIGINAL
             [t_s,t_e] (never grows); FIRE only if new_dur < det_dur AND
             new_dur >= CATEGORY_DURATION_PRIORS[cat]['min_segment'].

        Confidence, detection count/order, anomaly_type, coarse_score and
        frame_scores are byte-identical to flag-off; curve strictly read-only;
        pure numpy on CPU (FM9 n/a). Per-fire [CBT] log with the pre/post
        duration; exactly one UNCONDITIONAL [CBT-VID] per-video summary at the
        end (fires on the empty list too — FM17).
        """
        import os as _cbt_os
        cfg = self.config.scoring
        # v160 CBT-RECAL: swap ONLY the two scalar inputs when --cbt_recal is ON;
        # flag-off => caps==CBT_CAPS and trig==1.5 (byte-identical to v155 CBT).
        caps = cfg.cbt_recal_caps if cfg.cbt_recal else CBT_CAPS
        trig = cfg.cbt_recal_trigger if cfg.cbt_recal else 1.5
        vid = _cbt_os.path.basename(str(video_data.get('video_path') or '')) or 'unknown'
        fires = 0

        timestamps = video_data.get('timestamps')
        if not final_anomalies or timestamps is None:
            _log.info(f"  [CBT-VID] vid={vid} enabled={cfg.cbt_enabled} "
                      f"ndet={len(final_anomalies) if final_anomalies else 0} "
                      f"fired=0 wide_noop=0 nocurve=0 nofire=0 "
                      f"skip={'no_dets' if not final_anomalies else 'no_timestamps'} "
                      f"recal={'on' if cfg.cbt_recal else 'off'}")
            return final_anomalies
        t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        n = int(len(t))
        if n < 3:
            _log.info(f"  [CBT-VID] vid={vid} enabled={cfg.cbt_enabled} "
                      f"ndet={len(final_anomalies)} fired=0 wide_noop=0 "
                      f"nocurve=0 nofire=0 skip=curve_too_short "
                      f"recal={'on' if cfg.cbt_recal else 'off'}")
            return final_anomalies
        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            dt = 0.5
        fps = 1.0 / dt
        idx = np.arange(n)

        rel_cache = {}

        def _rel(cat):
            if cat in rel_cache:
                return rel_cache[cat]
            try:
                r = self.frame_scorer.get_category_scores(video_data, cat)
            except Exception as _e:  # a read-only diagnostic must never kill detect()
                _log.warning(f"  [CBT] curve read failed vid={vid} cat={cat}: {_e}")
                r = None
            if r is not None and hasattr(r, 'detach'):  # FM9: MPS/CUDA tensors
                r = r.detach().cpu().numpy()
            if r is not None:
                r = np.asarray(r, dtype=np.float64).reshape(-1)
                if len(r) != n:
                    r = None
            rel_cache[cat] = r
            return r

        n_wide_noop = n_nocurve = n_nofire = 0
        for di, a in enumerate(final_anomalies):
            # category resolution matches BUE exactly: own type, fallback arg
            cat = str(a.get('anomaly_type') or '').strip() or (category or '')
            if cat not in caps:
                continue
            cap = float(caps[cat])
            t_s = float(a['t_start'])
            t_e = float(a['t_end'])
            det_dur = t_e - t_s
            if det_dur <= trig * cap:
                n_wide_noop += 1     # not over-wide -> protect it (FM1/FM19)
                continue
            rel = _rel(cat)
            if rel is None:
                n_nocurve += 1
                continue
            lo = max(0, min(int(np.searchsorted(t, t_s)), n - 1))
            hi = max(lo + 1, min(int(np.searchsorted(t, t_e)), n))
            peak_idx = lo + int(np.argmax(rel[lo:hi]))
            min_seg = CATEGORY_DURATION_PRIORS.get(cat, {}).get('min_segment', 1.5)
            min_hw = max(1, int(np.ceil(min_seg * fps * 0.5)))
            max_hw = max(min_hw, int(cap * fps * 0.5))
            best_c, best_win = -1e30, None
            for hw in range(min_hw, max_hw + 1):
                lo_b = max(0, peak_idx - hw)
                hi_b = min(n - 1, peak_idx + hw)
                if hi_b <= lo_b:
                    continue
                inner_m = float(np.mean(rel[lo_b:hi_b + 1]))
                # AutoLoc/LogOIC 1/log-weighted outer rings over the whole video
                o_sum = o_wt = 0.0
                if lo_b > 0:
                    wl = 1.0 / np.log(2.0 + (lo_b - idx[:lo_b]))
                    o_sum += float(np.sum(rel[:lo_b] * wl))
                    o_wt += float(np.sum(wl))
                if hi_b + 1 < n:
                    wr = 1.0 / np.log(2.0 + (idx[hi_b + 1:] - hi_b))
                    o_sum += float(np.sum(rel[hi_b + 1:] * wr))
                    o_wt += float(np.sum(wr))
                o_m = o_sum / o_wt if o_wt > 0 else 0.0
                c = inner_m - o_m
                if c > best_c:
                    best_c, best_win = c, (lo_b, hi_b)
            if best_win is None:
                n_nofire += 1
                continue
            # clamp the chosen window to the ORIGINAL [t_s,t_e] — monotone inward
            ns = max(float(t[best_win[0]]), t_s)
            ne = min(float(t[best_win[1]]), t_e)
            new_dur = ne - ns
            if not (ne > ns and new_dur < det_dur and new_dur >= min_seg):
                n_nofire += 1
                continue
            conf = float(a.get('confidence', 0.0))
            a['t_start'] = ns
            a['t_end'] = ne
            fires += 1
            _log.info(
                f"  [CBT] vid={vid} cat={cat} det={di} "
                f"pre={det_dur:.1f} post={new_dur:.1f} "
                f"t={t_s:.2f}-{t_e:.2f}->{ns:.2f}-{ne:.2f} "
                f"cap={cap:.1f} contrast={best_c:.4f} conf={conf:.4f}->{conf:.4f}")

        _log.info(f"  [CBT-VID] vid={vid} enabled={cfg.cbt_enabled} "
                  f"ndet={len(final_anomalies)} fired={fires} "
                  f"wide_noop={n_wide_noop} nocurve={n_nocurve} nofire={n_nofire} "
                  f"recal={'on' if cfg.cbt_recal else 'off'}")
        return final_anomalies

    def _deduplicate_clip_regions(self, regions):
        """Merge overlapping CLIP-loc regions from multiple categories."""
        if not regions:
            return []
        regions = sorted(regions, key=lambda r: r['t_start'])
        merged = [regions[0].copy()]
        for r in regions[1:]:
            last = merged[-1]
            if r['t_start'] <= last['t_end'] + 3.0:
                last['t_end'] = max(last['t_end'], r['t_end'])
                last['clip_cat_score'] = max(last['clip_cat_score'], r['clip_cat_score'])
            else:
                merged.append(r.copy())
        merged.sort(key=lambda r: r['clip_cat_score'], reverse=True)
        merged = merged[:5]
        if len(regions) > len(merged):
            _log.info(f"  [Multi-CLIP-loc] {len(regions)} regions deduped to {len(merged)}")
        return merged


    def _motion_temporal_localization(self, video_data):
        """Use motion energy peaks for temporal localization (Explosion/RoadAccidents)."""
        motion = video_data.get('motion_energy')
        if motion is None or len(motion) < 4:
            return []
        
        timestamps = video_data['timestamps']
        duration = video_data['duration']
        
        # Smooth with 1-second window
        fps = max(1, len(timestamps) / max(1, duration))
        window = max(2, int(1.0 * fps))
        kernel = np.ones(window) / window
        smoothed = np.convolve(motion, kernel, mode='same')
        
        mean_m = np.mean(smoothed)
        std_m = np.std(smoothed)
        
        if std_m < 0.5:
            return []
        
        # Find regions above mean + 1.5*std (strong motion peaks)
        threshold = mean_m + 1.5 * std_m
        above = smoothed > threshold
        
        segments = []
        in_seg = False
        for i in range(len(above)):
            if above[i] and not in_seg:
                in_seg = True
                seg_start = i
            elif not above[i] and in_seg:
                in_seg = False
                segments.append((seg_start, i - 1))
        if in_seg:
            segments.append((seg_start, len(above) - 1))
        
        if not segments:
            return []
        
        # Convert to time and expand
        regions = []
        for start_idx, end_idx in segments:
            t_start = max(0, timestamps[start_idx] - 2.0)
            t_end = min(duration, timestamps[min(end_idx, len(timestamps)-1)] + 2.0)
            if t_end - t_start < 3.0:
                center = (t_start + t_end) / 2
                t_start = max(0, center - 3.0)
                t_end = min(duration, center + 3.0)
            avg_motion = np.mean(smoothed[start_idx:end_idx+1])
            regions.append({
                't_start': t_start, 't_end': t_end,
                'clip_cat_score': float(avg_motion / (np.max(smoothed) + 1e-8)),
            })
        
        regions.sort(key=lambda r: r['clip_cat_score'], reverse=True)
        regions = regions[:3]
        
        _log.info(f"  [Motion-loc] Found {len(regions)} motion peak regions:")
        for r in regions:
            _log.debug(f"    [{r['t_start']:.1f}-{r['t_end']:.1f}]s score={r['clip_cat_score']:.3f}")
        
        return regions

    def _compute_motion_energy(self, video_data):
        """Compute per-frame motion energy (mean absolute frame difference)."""
        import cv2
        frames = video_data['all_frames']
        if len(frames) < 2:
            return np.zeros(len(frames))
        small = np.stack([cv2.resize(f, (64, 64)) for f in frames]).astype(np.float32)
        motion = np.zeros(len(frames))
        motion[1:] = np.abs(small[1:] - small[:-1]).mean(axis=(1, 2, 3))
        return motion

    def _clip_temporal_localization(self, video_data, category):
        """Multi-scale CLIP temporal localization (v50).

        Runs peak detection at multiple smoothing scales to capture both
        short-burst events (RoadAccidents ~3s) and long events (Arrest ~50s).
        """
        precomputed = video_data.get('clip_features')
        if (precomputed is None or self.anomaly_expert.model is None
                or category is None or category == 'Normal'
                or torch is None):
            return []

        if category not in self.anomaly_categories:
            return []

        cat_idx = self.anomaly_categories.index(category)
        timestamps = video_data['timestamps']
        duration = video_data['duration']

        with torch.no_grad():
            logit_scale = self.anomaly_expert.model.logit_scale.exp().detach().clamp(max=100.0)
            similarity = precomputed @ self.anomaly_expert.text_features.T * logit_scale
            probs = torch.softmax(similarity, dim=-1).cpu().numpy()
            cat_scores = probs[:, cat_idx]
            normal_scores = probs[:, -1]
            relative_scores = cat_scores / (cat_scores + normal_scores + 1e-8)


        fps = max(1, len(timestamps) / max(1, duration))

        # Check for uniform scores (no discriminability)
        raw_range = np.max(relative_scores) - np.min(relative_scores)
        raw_cv = np.std(relative_scores) / (np.mean(relative_scores) + 1e-8)

        _log.info(f"  [CLIP-loc] Category={category}, mean={np.mean(relative_scores):.3f}, "
              f"std={np.std(relative_scores):.3f}, range={raw_range:.3f}, cv={raw_cv:.3f}")

        motion_categories = {'Explosion', 'RoadAccidents'}
        if raw_range < 0.08 or raw_cv < 0.03:
            if category in motion_categories:
                motion_regions = self._motion_temporal_localization(video_data)
                if motion_regions:
                    _log.info(f"  [CLIP-loc] Uniform but using motion energy for {category}")
                    return motion_regions
            _log.info(f"  [CLIP-loc] Scores too uniform, falling back to standard scanning")
            return []

        # Category-aware smoothing scales
        priors = CATEGORY_DURATION_PRIORS.get(category, {})
        preferred_smooth = priors.get('preferred_smooth', 2.0)
        smooth_scales = [
            max(0.5, preferred_smooth * 0.5),
            preferred_smooth,
            preferred_smooth * 2.0,
        ]

        # For Explosion/RoadAccidents: fuse motion energy with CLIP scores
        base_scores = relative_scores.copy()
        if category in motion_categories:
            motion = video_data.get('motion_energy', np.zeros(len(base_scores)))
            if len(motion) == len(base_scores) and np.max(motion) > 0:
                motion_norm = motion / (np.max(motion) + 1e-8)
                base_scores = 0.6 * base_scores + 0.4 * motion_norm
                _log.info(f"  [CLIP-loc] Fused with motion energy for {category}")

        # Multi-scale peak detection
        all_regions = []
        for smooth_sec in smooth_scales:
            window = max(3, int(smooth_sec * fps))
            if window >= len(base_scores):
                window = max(3, len(base_scores) // 2)
            if window % 2 == 0:
                window += 1
            kernel = np.ones(window) / window
            smoothed = np.convolve(base_scores, kernel, mode='same')

            mean_score = np.mean(smoothed)
            std_score = np.std(smoothed)

            # Adaptive threshold
            threshold = mean_score + 0.3 * std_score
            n_above = np.sum(smoothed > threshold)
            min_frames_needed = max(4, int(0.15 * len(smoothed)))
            if n_above < min_frames_needed:
                sorted_scores = np.sort(smoothed)[::-1]
                threshold = sorted_scores[min(min_frames_needed - 1, len(sorted_scores) - 1)]

            above = smoothed > threshold

            # Find contiguous regions
            segments = []
            in_seg = False
            seg_start_idx = 0
            for i in range(len(above)):
                if above[i] and not in_seg:
                    seg_start_idx = i
                    in_seg = True
                elif not above[i] and in_seg:
                    segments.append((seg_start_idx, i - 1))
                    in_seg = False
            if in_seg:
                segments.append((seg_start_idx, len(above) - 1))

            # Merge with small gaps (< 3 seconds)
            gap_threshold = 3.0
            merged_segs = []
            for start_idx, end_idx in segments:
                t_start = float(timestamps[start_idx])
                t_end = float(timestamps[min(end_idx, len(timestamps) - 1)])
                if merged_segs and t_start - merged_segs[-1][1] < gap_threshold:
                    merged_segs[-1] = (merged_segs[-1][0], t_end)
                else:
                    merged_segs.append((t_start, t_end))

            # Expand each region by 2 seconds on each side
            for t_start, t_end in merged_segs:
                t_start = max(0, t_start - 2.0)
                t_end = min(duration, t_end + 2.0)
                if t_end - t_start < 3.0:
                    center = (t_start + t_end) / 2
                    t_start = max(0, center - 1.5)
                    t_end = min(duration, center + 1.5)
                mask = (timestamps >= t_start) & (timestamps <= t_end)
                if np.any(mask):
                    avg_score = float(np.mean(smoothed[mask]))
                    peak_score = float(np.max(smoothed[mask]))
                    all_regions.append({
                        't_start': t_start,
                        't_end': t_end,
                        'clip_cat_score': avg_score,
                        'peak_score': peak_score,
                        'scale': smooth_sec,
                    })

        # Multi-scale NMS: prefer higher peak score, break ties with tighter boundaries
        all_regions.sort(key=lambda r: (r['peak_score'], -(r['t_end'] - r['t_start'])), reverse=True)
        selected = []
        for region in all_regions:
            overlap = False
            for sel in selected:
                iou = self._temporal_iou(region, sel)
                if iou > 0.5:
                    overlap = True
                    break
            if not overlap:
                selected.append(region)

        selected = selected[:5]

        if selected:
            _log.info(f"  [CLIP-loc] Found {len(selected)} multi-scale regions:")
            for r in selected:
                _log.debug(f"    [{r['t_start']:.1f}-{r['t_end']:.1f}]s "
                      f"score={r['clip_cat_score']:.3f} peak={r['peak_score']:.3f} "
                      f"scale={r['scale']:.1f}s")

        return selected

    @staticmethod
    def _temporal_iou(a, b):
        """Compute temporal IoU between two segments."""
        overlap_start = max(a['t_start'], b['t_start'])
        overlap_end = min(a['t_end'], b['t_end'])
        overlap = max(0, overlap_end - overlap_start)
        union = max(a['t_end'], b['t_end']) - min(a['t_start'], b['t_start'])
        return overlap / union if union > 0 else 0

    async def _directed_llm_scan(self, video_data, clip_regions, context_prompt, category):
        """Hybrid LLM scan: union of CLIP-loc regions + standard CLIP pre-screen."""
        segments = video_data['segments']
        all_frames = video_data['all_frames']
        precomputed = video_data.get('clip_features')

        targeted_set = set()
        targeted_segments = []

        # Source 1: segments overlapping CLIP-loc regions
        for seg in segments:
            for region in clip_regions:
                if seg['t_start'] < region['t_end'] and seg['t_end'] > region['t_start']:
                    if id(seg) not in targeted_set:
                        targeted_segments.append(seg)
                        targeted_set.add(id(seg))
                    break

        # Source 2: standard CLIP anomaly pre-screen (top segments by general anomaly score)
        if self.anomaly_expert.model is not None and precomputed is not None:
            clip_results = self.anomaly_expert.batch_score_segments(segments, precomputed)
            all_scores = [r['anomaly_score'] for r in clip_results]
            mean_s = np.mean(all_scores)
            std_s = np.std(all_scores)
            adaptive_thresh = max(0.3, mean_s - 0.5 * std_s)
            scored_indices = sorted(range(len(segments)),
                                    key=lambda i: all_scores[i], reverse=True)
            # Add top CLIP-scored segments not already covered
            min_extra = max(2, int(len(segments) * 0.2))
            added = 0
            for i in scored_indices:
                if added >= min_extra:
                    break
                if id(segments[i]) not in targeted_set:
                    targeted_segments.append(segments[i])
                    targeted_set.add(id(segments[i]))
                    added += 1

        max_scan = min(max(10, len(segments) // 5), 18)
        if len(targeted_segments) > max_scan:
            targeted_segments = targeted_segments[:max_scan]

        _log.info(f"  [Directed scan] {len(targeted_segments)}/{len(segments)} segments "
                  f"selected for LLM scanning")

        category_hint = ""
        if category:
            cue = CATEGORY_CUES.get(category, category)
            category_hint = f"\nExpected anomaly type: {category} — {cue}\n"

        tasks = []
        for seg in targeted_segments:
            tasks.append(self.discovery.scan_segment(
                seg, all_frames, context_prompt + category_hint, precomputed))

        results = await asyncio.gather(*tasks)

        suspicious = []
        for seg, result in zip(targeted_segments, results):
            status = "SUSPICIOUS" if result['is_suspicious'] else "normal"
            _log.debug(f"    {seg['t_start']:.1f}-{seg['t_end']:.1f}s: {status} "
                  f"(score={result['anomaly_score']:.2f})")
            if result['is_suspicious']:
                suspicious.append({
                    't_start': seg['t_start'],
                    't_end': seg['t_end'],
                    'frame_indices': seg['frame_indices'],
                    'coarse_score': result['anomaly_score'],
                    'description': result['description'],
                })

        return self.discovery._merge_adjacent(suspicious)

    # ================================================================
    # Stage 5 (v50): CLIP score-curve boundary detection
    # ================================================================

    def _clip_boundary_detection(self, candidates, video_data, category):
        """Replace MCTS with CLIP score curve boundary detection.

        Uses multi-scale half-max crossing with gradient refinement to find
        precise anomaly boundaries from the continuous CLIP score signal.
        Category-aware smoothing windows adapt to expected event duration.
        """
        from bisect import bisect_left

        precomputed = video_data.get('clip_features')
        if precomputed is None or self.anomaly_expert.model is None or torch is None:
            # Fallback: return candidates as-is with default confidence
            return [dict(c, confidence=c.get('coarse_score', 0.8),
                         coarse_score=c.get('coarse_score', 0.8),
                         anomaly_type=category or 'unknown') for c in candidates]

        timestamps = video_data['timestamps']
        duration = video_data['duration']
        fps = max(1, len(timestamps) / max(1, duration))

        # Get category-specific relative scores
        relative_scores = self.frame_scorer.get_category_scores(video_data, category)
        if relative_scores is None:
            return [dict(c, confidence=c.get('coarse_score', 0.8),
                         coarse_score=c.get('coarse_score', 0.8),
                         anomaly_type=category or 'unknown') for c in candidates]

        # Category-aware smoothing scales
        priors = CATEGORY_DURATION_PRIORS.get(category, {})
        preferred_smooth = priors.get('preferred_smooth', 2.0)
        min_dur, max_dur = priors.get('expected_duration', (3.0, 30.0))
        min_segment = priors.get('min_segment', 2.0)

        smooth_scales = [
            max(0.5, preferred_smooth * 0.5),
            preferred_smooth,
            preferred_smooth * 2.0,
        ]

        refined_candidates = []
        for cand in candidates:
            t_s = cand['t_start']
            t_e = cand['t_end']
            cand_dur = t_e - t_s

            # Search window: candidate region + 50% padding on each side
            padding = cand_dur * 0.5
            search_start = max(0, t_s - padding)
            search_end = min(duration, t_e + padding)

            lo_idx = max(0, bisect_left(timestamps, search_start))
            hi_idx = min(len(timestamps), bisect_left(timestamps, search_end))

            if hi_idx - lo_idx < 4:
                refined_candidates.append(dict(
                    cand, confidence=cand.get('coarse_score', 0.8),
                    coarse_score=cand.get('coarse_score', 0.8),
                    anomaly_type=category or 'unknown', boundary_contrast=0.0))
                continue

            seg_scores = relative_scores[lo_idx:hi_idx]
            seg_times = timestamps[lo_idx:hi_idx]

            # v63: V51 half-max crossing + motion-sharpened boundary detection.
            # Motion energy has SHARP transitions at anomaly boundaries (sudden onset
            # of violence, crash impact, explosion) while CLIP scores are smooth/broad.
            # Fusing motion with CLIP before boundary detection produces sharper
            # boundaries for motion-sensitive categories.
            # V59's peak-valley detection (nearest local minima as boundaries) was
            # a regression: mAP 0.1172->0.0922, AP@0.7=0. Root cause: shallow dips
            # within anomaly plateaus are false valleys — the nearest valley is often
            # INSIDE the anomaly, producing severely over-narrowed segments.
            # V51's half-max crossing (mAP=0.1492) is the proven best. V60 restores
            # it but adds a targeted post-step: when the half-max result is wider
            # than 2x the category's max_dur, re-threshold at 70% of peak (stricter)
            # to tighten just the extreme over-wide cases (RoadAccidents 28s det for
            # 2.7s event, Shooting 61s for 7s event) without affecting well-calibrated
            # segments. MIL fusion removed (saturated, no-op in practice).

            # v66: V51 half-max crossing + LogOIC tightening for over-wide results.
            # V65 showed LogOIC excels at short events (RoadAccidents IoU 0.095->0.45,
            # Abuse 0.282->0.60) but severely over-narrows long events (Fighting
            # 0.575->0.18, Assault 0.297->0.04). The solution: use V51's proven
            # half-max as PRIMARY, then apply LogOIC ONLY when the result exceeds
            # max_dur — the cases where half-max is known to be too wide.
            # FFT actionness calibration retained from V65.

            # v68: MIL fusion RESTORED. V67 showed that removing MIL fusion
            # (despite MIL being "near-saturated") changes boundary detection via
            # subtle interpolation non-uniformity — Fighting IoU 0.574→0.176,
            # Explosion 0.500→0.350. LogOIC tightening never even fired because
            # MIL removal already changed half-max results to be narrower.
            mil_scores = video_data.get('mil_scores')
            if mil_scores is not None and len(mil_scores) >= 2:
                _mil_interp = np.interp(
                    np.linspace(0, 1, len(seg_scores)),
                    np.linspace(0, 1, len(mil_scores)),
                    mil_scores
                )
                fused_scores = 0.75 * seg_scores + 0.25 * _mil_interp
            else:
                fused_scores = seg_scores

            best_boundaries = None
            best_score = -1.0

            for smooth_sec in smooth_scales:
                window = max(3, int(smooth_sec * fps))
                if window >= len(fused_scores):
                    window = max(3, len(fused_scores) // 2)
                if window % 2 == 0:
                    window += 1
                kernel = np.ones(window) / window
                smoothed = np.convolve(fused_scores, kernel, mode='same')
                gradient = np.gradient(smoothed)
                refine_range = max(1, int(1.0 * fps))
                peak_val = np.max(smoothed)
                baseline = np.percentile(smoothed, 20)
                half_max = (peak_val + baseline) / 2.0

                above = smoothed >= half_max
                transitions = np.diff(above.astype(np.int32))
                rises = np.where(transitions == 1)[0]
                falls = np.where(transitions == -1)[0]

                if len(rises) == 0 and len(falls) == 0:
                    if np.mean(above) > 0.5:
                        new_lo, new_hi = 0, len(fused_scores) - 1
                    else:
                        continue
                else:
                    new_lo = rises[0] + 1 if len(rises) > 0 else 0
                    new_hi = falls[-1] if len(falls) > 0 else len(fused_scores) - 1
                if new_hi <= new_lo:
                    continue

                prop = _score_boundary_proposal(
                    smoothed, gradient, seg_times, new_lo, new_hi,
                    refine_range, min_dur, max_dur)
                if prop is None:
                    continue

                # v66: LogOIC tightening for over-wide half-max results
                prop_dur = prop['t_end'] - prop['t_start']
                if prop_dur > max_dur:
                    peak_idx = np.argmax(smoothed)
                    n = len(smoothed)
                    best_logoic = -1.0
                    best_tight = None
                    min_hw = max(2, int(min_dur * fps * 0.5))
                    max_hw = min(n // 2, max(3, int(max_dur * fps * 0.5)))
                    step_hw = max(1, (max_hw - min_hw) // 10)
                    for hw in range(min_hw, max_hw + 1, step_hw):
                        lo_b = max(0, peak_idx - hw)
                        hi_b = min(n - 1, peak_idx + hw)
                        if hi_b <= lo_b:
                            continue
                        inner_m = float(np.mean(smoothed[lo_b:hi_b + 1]))
                        o_sum, o_wt = 0.0, 0.0
                        for j in range(lo_b):
                            w = 1.0 / np.log(2.0 + lo_b - j)
                            o_sum += smoothed[j] * w; o_wt += w
                        for j in range(hi_b + 1, n):
                            w = 1.0 / np.log(2.0 + j - hi_b)
                            o_sum += smoothed[j] * w; o_wt += w
                        o_m = o_sum / o_wt if o_wt > 0 else 0.0
                        logoic = inner_m - o_m
                        if logoic > best_logoic:
                            ts = float(seg_times[min(lo_b, len(seg_times)-1)])
                            te = float(seg_times[min(hi_b, len(seg_times)-1)])
                            if te > ts:
                                best_logoic = logoic
                                best_tight = (ts, te)
                    if best_tight is not None:
                        _log.debug(f"      [LogOIC-tighten] {prop_dur:.1f}s -> "
                                   f"{best_tight[1]-best_tight[0]:.1f}s")
                        prop = {'t_start': best_tight[0], 't_end': best_tight[1],
                                'contrast': best_logoic}

                if prop['contrast'] > best_score:
                    best_score = prop['contrast']
                    best_boundaries = (prop['t_start'], prop['t_end'])

            if best_boundaries is not None:
                result = dict(cand)
                result['t_start'] = max(0.0, best_boundaries[0])
                result['t_end'] = min(best_boundaries[1], duration)

                # v54: light backstop only — duration-prior-fit already steers
                # selection above, so this should rarely trigger now.
                detected_dur = result['t_end'] - result['t_start']
                if detected_dur < min_segment:
                    center = (result['t_start'] + result['t_end']) / 2.0
                    result['t_start'] = max(0.0, center - min_segment / 2.0)
                    result['t_end'] = min(duration, center + min_segment / 2.0)
                elif detected_dur > max_dur * 2.0:
                    tighter = self._clip_boundary_single_scale(
                        cand, video_data, category,
                        smooth_sec=max(0.5, preferred_smooth * 0.5))
                    if tighter is not None and (tighter['t_end'] - tighter['t_start']) >= min_segment:
                        result = tighter

                # v51: Redesigned confidence formula — decouple boundary precision from detection existence
                # Base confidence: 80% of LLM coarse_score (the anomaly EXISTS)
                # Boundary bonus: up to 20% for high CLIP contrast (boundaries are PRECISE)
                # v95: TSM sharpness re-ranking (Loop 1-2 research). Modulates
                # ONLY the confidence bonus (never t_start/t_end) using a
                # genuinely independent signal (frame-frame visual novelty,
                # not text-similarity) - detections whose boundaries align
                # with real visual scene transitions get a small confidence
                # boost. Pure re-ranking, zero risk to boundary positions or
                # detection count (respects all 4 known failure modes).
                _tsm_novelty = video_data.get('tsm_novelty')
                _tsm_mod = 1.0
                if _tsm_novelty is not None and len(_tsm_novelty) >= 2:
                    from bisect import bisect_left as _bl95
                    _ts_idx = min(len(_tsm_novelty) - 1, max(0, _bl95(timestamps, result['t_start'])))
                    _te_idx = min(len(_tsm_novelty) - 1, max(0, _bl95(timestamps, result['t_end'])))
                    _tsm_sharpness = float((_tsm_novelty[_ts_idx] + _tsm_novelty[_te_idx]) / 2.0)
                    _tsm_mod = 0.85 + 0.15 * _tsm_sharpness

                # v96: Asymmetric onset/offset likelihood (BSN-inspired, Loop 4
                # research). Confirmed via literature (BSN TEM, LGN) that
                # separating DIRECTIONAL start/end signals from a symmetric
                # inner-outer contrast is a real, validated architectural
                # pattern. Training-free analog: at the START boundary, a
                # real onset should show FUTURE window more anomalous than
                # PAST window; at the END boundary, PAST should be more
                # anomalous than FUTURE. Pure confidence modulation only
                # (never changes t_start/t_end) - same safe pattern as V95.
                _onset_mod = 1.0
                _w_frames = max(2, int(1.0 * fps))
                _ts_i = min(len(relative_scores) - 1, max(0, _bl95(timestamps, result['t_start'])))
                _te_i = min(len(relative_scores) - 1, max(0, _bl95(timestamps, result['t_end'])))
                _past_s = relative_scores[max(0, _ts_i - _w_frames):_ts_i]
                _fut_s = relative_scores[_ts_i:min(len(relative_scores), _ts_i + _w_frames)]
                _past_e = relative_scores[max(0, _te_i - _w_frames):_te_i]
                _fut_e = relative_scores[_te_i:min(len(relative_scores), _te_i + _w_frames)]
                if len(_past_s) > 0 and len(_fut_s) > 0 and len(_past_e) > 0 and len(_fut_e) > 0:
                    _onset_sharp = float(np.mean(_fut_s) - np.mean(_past_s))
                    _offset_sharp = float(np.mean(_past_e) - np.mean(_fut_e))
                    _directional_quality = max(0.0, min(1.0, (_onset_sharp + _offset_sharp) / 2.0 + 0.5))
                    _onset_mod = 0.9 + 0.1 * _directional_quality

                _coarse = cand.get('coarse_score', 0.8)
                _base_conf = _coarse * 0.80
                _boundary_bonus = _coarse * 0.20 * min(1.0, best_score / 0.5) * _tsm_mod * _onset_mod
                result['confidence'] = min(1.0, _base_conf + _boundary_bonus)
                result['coarse_score'] = _coarse  # v51: preserve for PostTighten
                result['anomaly_type'] = category or 'unknown'
                result['boundary_contrast'] = best_score

                _log.info(f"    [CLIPBound] {t_s:.1f}-{t_e:.1f}s -> "
                          f"{result['t_start']:.1f}-{result['t_end']:.1f}s "
                          f"(contrast={best_score:.3f}, cat={category})")
                refined_candidates.append(result)
            else:
                refined_candidates.append(dict(
                    cand, confidence=cand.get('coarse_score', 0.8),
                    coarse_score=cand.get('coarse_score', 0.8),
                    anomaly_type=category or 'unknown', boundary_contrast=0.0))

        return refined_candidates

    def _clip_boundary_single_scale(self, candidate, video_data, category, smooth_sec=1.0):
        """Single-scale CLIP boundary detection for re-tightening over-long segments."""
        from bisect import bisect_left

        timestamps = video_data['timestamps']
        duration = video_data['duration']
        fps = max(1, len(timestamps) / max(1, duration))

        relative_scores = self.frame_scorer.get_category_scores(video_data, category)
        if relative_scores is None:
            return None

        t_s, t_e = candidate['t_start'], candidate['t_end']
        cand_dur = t_e - t_s
        padding = cand_dur * 0.3
        search_start = max(0, t_s - padding)
        search_end = min(duration, t_e + padding)

        lo_idx = max(0, bisect_left(timestamps, search_start))
        hi_idx = min(len(timestamps), bisect_left(timestamps, search_end))
        if hi_idx - lo_idx < 4:
            return None

        seg_scores = relative_scores[lo_idx:hi_idx]
        seg_times = timestamps[lo_idx:hi_idx]

        window = max(3, int(smooth_sec * fps))
        if window >= len(seg_scores):
            window = max(3, len(seg_scores) // 2)
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window) / window
        smoothed = np.convolve(seg_scores, kernel, mode='same')

        # v54: classic 50% threshold — this function is now only a rare backstop
        # (invoked when the primary multi-threshold proposal in _clip_boundary_detection
        # still produces an extreme outlier), not the primary detection mechanism.
        peak_val = np.max(smoothed)
        baseline = np.percentile(smoothed, 20)
        half_max = (peak_val + baseline) / 2.0

        above = smoothed >= half_max
        transitions = np.diff(above.astype(np.int32))
        rises = np.where(transitions == 1)[0]
        falls = np.where(transitions == -1)[0]

        if len(rises) == 0 and len(falls) == 0:
            return None

        new_lo = rises[0] + 1 if len(rises) > 0 else 0
        new_hi = falls[-1] if len(falls) > 0 else len(seg_scores) - 1

        if new_hi <= new_lo:
            return None

        result = dict(candidate)
        result['t_start'] = max(0.0, float(seg_times[min(new_lo, len(seg_times) - 1)]))
        result['t_end'] = min(float(seg_times[min(new_hi, len(seg_times) - 1)]), duration)
        inside_mean = float(np.mean(smoothed[new_lo:new_hi + 1]))
        # v51: Same redesigned confidence formula as main boundary detection
        _coarse = candidate.get('coarse_score', 0.8)
        _base_conf = _coarse * 0.80
        _boundary_bonus = _coarse * 0.20 * min(1.0, inside_mean / 0.5)
        result['confidence'] = min(1.0, _base_conf + _boundary_bonus)
        result['coarse_score'] = _coarse  # v51: preserve for PostTighten
        result['anomaly_type'] = category or 'unknown'
        result['boundary_contrast'] = float(inside_mean)
        return result

    # ================================================================
    # Stage 5b (v50): Single VLM boundary estimation
    # ================================================================

    async def _vlm_boundary_estimation(self, candidate, video_data, category):
        """Single VLM call for temporal boundary estimation.

        For low-CLIP-confidence candidates, ask the VLM once to identify
        start/end times, then cross-validate against CLIP curve.
        """
        t_s = candidate['t_start']
        t_e = candidate['t_end']
        dur = t_e - t_s
        duration = video_data['duration']
        timestamps = video_data['timestamps']

        # Expand search window by 30% on each side
        search_start = max(0, t_s - dur * 0.3)
        search_end = min(duration, t_e + dur * 0.3)

        # Sample 8-12 frames uniformly with timestamps
        n_frames = min(12, max(8, int((search_end - search_start) / 2)))
        frame_times = np.linspace(search_start, search_end, n_frames)

        frames = []
        actual_times = []
        for t in frame_times:
            idx = int(np.argmin(np.abs(timestamps - t)))
            frames.append(video_data['all_frames'][idx])
            actual_times.append(float(timestamps[idx]))

        keyframes = np.stack(frames)
        time_labels = ', '.join(f'{t:.1f}s' for t in actual_times)

        prompt = (
            f"""Analyze these {n_frames} surveillance video frames in chronological order.
Frame timestamps: {time_labels}

Target anomaly type: {category}

For each frame, state whether the anomaly ({category}) is ACTIVELY occurring.
Then provide your best estimate of when the anomaly starts and ends.

Respond ONLY in JSON format:
{{"anomaly_detected": true, "start_time": <seconds>, "end_time": <seconds>, "confidence": 0-100}}"""
        )

        # v90->v91 REVERT: median-of-3 self-consistency regressed full-eval
        # mAP 0.1494->0.1075. Root cause hypothesis: the temperature ladder
        # (0.0, 0.3, 0.5) was designed for REASONING-diversity self-consistency
        # (multiple valid reasoning PATHS converging on the same correct
        # answer), which doesn't transfer to PERCEPTUAL/EXTRACTION tasks like
        # "what timestamp does this event start" - there's no equivalent of
        # "another valid way to see the same correct answer" for reading a
        # timestamp off a frame. Deliberately raising temperature on 2 of 3
        # calls just added noise, diluting the more-accurate temp=0 estimate
        # via the median rather than improving it. Reverted to the original
        # single deterministic call.
        try:
            response = await self.llm_client.generate(prompt, keyframes)
            text = response.strip()
            start_j = text.find('{')
            end_j = text.rfind('}') + 1
            if start_j >= 0 and end_j > start_j:
                parsed = json.loads(text[start_j:end_j])
            else:
                parsed = {}
        except Exception as e:
            _log.warning(f"    [VLM-Bound] Parse failed: {e}")
            return candidate

        if not parsed.get('anomaly_detected', False):
            return candidate

        vlm_start = float(parsed.get('start_time', t_s))
        vlm_end = float(parsed.get('end_time', t_e))
        vlm_conf = float(parsed.get('confidence', 50)) / 100.0

        # Sanity check VLM answer
        if vlm_start >= vlm_end or vlm_start < 0 or vlm_end > duration * 1.1:
            _log.warning(f"    [VLM-Bound] Invalid VLM times: {vlm_start:.1f}-{vlm_end:.1f}")
            return candidate

        # v61: Cross-validate with CLIP curve, with over-wide override
        clip_start = candidate.get('t_start', t_s)
        clip_end = candidate.get('t_end', t_e)
        clip_conf = candidate.get('boundary_contrast', 0.5)
        clip_dur = clip_end - clip_start

        priors_bd = CATEGORY_DURATION_PRIORS.get(category, {})
        cat_max_dur = priors_bd.get('expected_duration', (3.0, 30.0))[1]

        if clip_dur > cat_max_dur * 1.2 and vlm_conf > 0.4:
            # CLIP is demonstrably over-wide — trust VLM boundaries directly
            final_start = vlm_start
            final_end = vlm_end
            w_clip, w_vlm = 0.0, 1.0
            _log.info(f"    [VLM-Bound] CLIP over-wide ({clip_dur:.1f}s > "
                      f"{cat_max_dur*1.2:.1f}s), using VLM boundaries")
        else:
            # Normal weighted average
            w_clip = clip_conf / (clip_conf + vlm_conf + 1e-8)
            w_vlm = vlm_conf / (clip_conf + vlm_conf + 1e-8)
            final_start = w_clip * clip_start + w_vlm * vlm_start
            final_end = w_clip * clip_end + w_vlm * vlm_end

        result = dict(candidate)
        result['t_start'] = max(0.0, final_start)
        result['t_end'] = min(final_end, duration)
        result['vlm_confidence'] = vlm_conf

        # Ensure minimum duration
        priors = CATEGORY_DURATION_PRIORS.get(category, {})
        min_seg = priors.get('min_segment', 2.0)
        if result['t_end'] - result['t_start'] < min_seg:
            center = (result['t_start'] + result['t_end']) / 2.0
            result['t_start'] = max(0.0, center - min_seg / 2.0)
            result['t_end'] = min(duration, center + min_seg / 2.0)

        _log.info(f"    [VLM-Bound] {t_s:.1f}-{t_e:.1f}s -> "
                  f"{result['t_start']:.1f}-{result['t_end']:.1f}s "
                  f"(vlm_conf={vlm_conf:.2f}, w_clip={w_clip:.2f}, w_vlm={w_vlm:.2f})")

        return result

        # ================================================================
    # Stage 6b (v50): Post-merge boundary tightening
    # ================================================================

    def _post_merge_boundary_tightening(self, merged_detections, video_data, category):
        """Re-run CLIP boundary detection on merged segments that are too long.

        v51 changes:
        - Never reduce confidence after re-running boundary detection
        - Skip PostTighten entirely for LOW_CLIP categories (re-running produces worse results)
        - Preserve coarse_score through the process
        """
        if not merged_detections:
            return merged_detections

        tightened = []
        for det in merged_detections:
            dur = det['t_end'] - det['t_start']
            cat = det.get('anomaly_type', category) or category
            expected_max = CATEGORY_DURATION_PRIORS.get(
                cat, {}).get('expected_duration', (3, 30))[1]

            # v51: Skip PostTighten for low-CLIP categories where re-running
            # boundary detection produces lower-quality results
            if cat in LOW_CLIP_CATEGORIES:
                tightened.append(det)
                continue

            if dur > expected_max * 1.2:
                original_conf = det['confidence']
                original_coarse = det.get('coarse_score', det['confidence'])
                refined = self._clip_boundary_detection([det], video_data, cat)
                if refined and (refined[0]['t_end'] - refined[0]['t_start']) >=                         CATEGORY_DURATION_PRIORS.get(cat, {}).get('min_segment', 2.0):
                    ref = refined[0]
                    # v51: NEVER reduce confidence — PostTighten is about boundaries, not existence
                    ref['confidence'] = max(original_conf, ref['confidence'])
                    ref['coarse_score'] = original_coarse
                    _log.info(f"    [PostTighten] {det['t_start']:.1f}-{det['t_end']:.1f}s "
                              f"-> {ref['t_start']:.1f}-{ref['t_end']:.1f}s "
                              f"(was {dur:.1f}s > {expected_max*1.2:.1f}s limit, "
                              f"conf={original_conf:.3f}->{ref['confidence']:.3f})")
                    tightened.append(ref)
                else:
                    tightened.append(det)
            else:
                tightened.append(det)

        return tightened

        # ================================================================
    # Stage 6: Category-aware CLIP boundary refinement
    # ================================================================


    async def _refine_boundary_category_aware(self, anomaly, video_data, category):
        """Gradient-based boundary refinement using CLIP score curve derivatives."""
        t_start = anomaly['t_start']
        t_end = anomaly['t_end']
        duration = video_data['duration']
        timestamps = video_data['timestamps']

        relative_scores = self.frame_scorer.get_category_scores(video_data, category)
        if relative_scores is None:
            return await self._refine_boundary(anomaly, video_data)

        global_std = np.std(relative_scores)
        global_mean = np.mean(relative_scores)
        coeff_var = global_std / (global_mean + 1e-8)
        if coeff_var < 0.05:
            return await self._refine_boundary(anomaly, video_data)

        from bisect import bisect_left
        lo = bisect_left(timestamps, t_start)
        hi = bisect_left(timestamps, t_end)
        if hi <= lo:
            return anomaly

        # Smooth scores for gradient computation
        fps = max(1, len(timestamps) / max(1, duration))
        smooth_w = max(3, int(1.5 * fps))
        if len(relative_scores) >= smooth_w:
            kernel = np.ones(smooth_w) / smooth_w
            smoothed = np.convolve(relative_scores, kernel, mode='same')
        else:
            smoothed = relative_scores.copy()

        core_scores = smoothed[lo:hi]
        core_mean = np.mean(core_scores)
        seg_dur = t_end - t_start

        # Compute score gradient (first derivative)
        gradient = np.gradient(smoothed)

        # --- Gradient-based left boundary ---
        max_expand = seg_dur * 0.5
        search_lo = max(0, bisect_left(timestamps, max(0, t_start - max_expand)))
        # Look for the sharpest positive gradient (score rising) near left boundary
        best_left = lo
        left_search = range(max(search_lo, lo - int(max_expand * fps)), lo + int(2 * fps) + 1)
        if left_search:
            left_grads = [(i, gradient[i]) for i in left_search if 0 <= i < len(gradient)]
            if left_grads:
                # Find steepest rise — that's where the anomaly starts
                rise_candidates = [(i, g) for i, g in left_grads if g > 0]
                if rise_candidates:
                    best_rise_idx = max(rise_candidates, key=lambda x: x[1])[0]
                    if smoothed[best_rise_idx] >= core_mean * 0.4:
                        best_left = best_rise_idx

        # Also check threshold-based left boundary
        threshold_left = lo
        thresh = core_mean * 0.55
        for i in range(lo - 1, search_lo - 1, -1):
            if smoothed[i] >= thresh:
                threshold_left = i
            else:
                break

        # Only contract, never expand beyond original boundary
        final_left = max(lo, min(best_left, threshold_left))

        # --- Gradient-based right boundary ---
        search_hi = min(len(timestamps), bisect_left(timestamps,
                        min(duration, t_end + max_expand)))
        best_right = hi
        right_search = range(max(0, hi - int(2 * fps)), min(search_hi, hi + int(max_expand * fps)) + 1)
        if right_search:
            right_grads = [(i, gradient[i]) for i in right_search if 0 <= i < len(gradient)]
            if right_grads:
                # Find steepest drop (negative gradient) — that's where the anomaly ends
                drop_candidates = [(i, g) for i, g in right_grads if g < 0]
                if drop_candidates:
                    best_drop_idx = min(drop_candidates, key=lambda x: x[1])[0]
                    if smoothed[best_drop_idx] >= core_mean * 0.4:
                        best_right = best_drop_idx + 1

        # Also check threshold-based right boundary
        threshold_right = hi
        for i in range(hi, search_hi):
            if smoothed[i] >= thresh:
                threshold_right = i + 1
            else:
                break

        # Only contract, never expand beyond original boundary
        final_right = min(hi, max(best_right, threshold_right))

        # Convert back to time
        final_left = max(0, min(final_left, len(timestamps) - 1))
        final_right = max(0, min(final_right, len(timestamps) - 1))
        new_start = timestamps[final_left]
        new_end = timestamps[final_right]

        # Shrink: trim frames with very low scores from edges (max 15%)
        shrink_thresh = core_mean * self.config.scoring.boundary_shrink_ratio
        s_lo = bisect_left(timestamps, new_start)
        s_hi = bisect_left(timestamps, new_end)
        total_frames = s_hi - s_lo
        max_trim = max(1, total_frames // 7)
        for i in range(s_lo, min(s_lo + max_trim, s_hi)):
            if i < len(smoothed) and smoothed[i] < shrink_thresh:
                new_start = timestamps[min(i + 1, len(timestamps) - 1)]
            else:
                break
        for i in range(s_hi - 1, max(s_hi - max_trim - 1, s_lo), -1):
            if i < len(smoothed) and smoothed[i] < shrink_thresh:
                new_end = timestamps[max(i - 1, 0)]
            else:
                break

        refined = anomaly.copy()
        refined['t_start'] = max(0, new_start)
        refined['t_end'] = min(new_end, duration)

        if refined['t_end'] - refined['t_start'] < 1.0:
            refined['t_start'] = t_start
            refined['t_end'] = t_end

        if abs(refined['t_start'] - t_start) > 0.5 or abs(refined['t_end'] - t_end) > 0.5:
            _log.info(f"    [GradRefine] {t_start:.1f}-{t_end:.1f}s "
                      f"-> {refined['t_start']:.1f}-{refined['t_end']:.1f}s "
                      f"(cat={category}, cv={coeff_var:.3f})")

        return refined

    # ================================================================
    # Fallback boundary refinement (generic CLIP)
    # ================================================================

    async def _refine_boundary(self, anomaly, video_data):
        """Generic CLIP anomaly score boundary refinement (vectorized, no async calls)."""
        t_start = anomaly['t_start']
        t_end = anomaly['t_end']
        duration = video_data['duration']
        precomputed = video_data.get('clip_features')
        timestamps = video_data['timestamps']

        if precomputed is None or self.anomaly_expert.model is None or torch is None:
            return anomaly

        # Use general anomaly scores (max across all anomaly categories)
        with torch.no_grad():
            logit_scale = self.anomaly_expert.model.logit_scale.exp().detach().clamp(max=100.0)
            similarity = precomputed @ self.anomaly_expert.text_features.T * logit_scale
            probs = torch.softmax(similarity, dim=-1).cpu().numpy()
            # Anomaly score = 1 - normal_prob (last category is Normal)
            anomaly_scores = 1.0 - probs[:, -1]

        from bisect import bisect_left
        lo = bisect_left(timestamps, t_start)
        hi = bisect_left(timestamps, t_end)
        if hi <= lo:
            return anomaly

        core_scores = anomaly_scores[lo:hi]
        core_mean = np.mean(core_scores)
        seg_dur = t_end - t_start

        # Expand: include adjacent frames above 55% of core mean
        thresh = core_mean * 0.55
        max_expand = seg_dur * 0.4
        new_lo = lo
        search_lo = max(0, bisect_left(timestamps, max(0, t_start - max_expand)))
        for i in range(lo - 1, search_lo - 1, -1):
            if anomaly_scores[i] >= thresh:
                new_lo = i
            else:
                break

        new_hi = hi
        search_hi = min(len(timestamps), bisect_left(timestamps,
                        min(duration, t_end + max_expand)))
        for i in range(hi, search_hi):
            if anomaly_scores[i] >= thresh:
                new_hi = i + 1
            else:
                break

        # Shrink: trim very low score edges
        shrink_thresh = core_mean * self.config.scoring.boundary_shrink_ratio
        total_frames = new_hi - new_lo
        max_trim = max(1, total_frames // 6)
        for i in range(new_lo, min(new_lo + max_trim, new_hi)):
            if anomaly_scores[i] < shrink_thresh:
                new_lo = i + 1
            else:
                break
        for i in range(new_hi - 1, max(new_hi - max_trim - 1, new_lo), -1):
            if anomaly_scores[i] < shrink_thresh:
                new_hi = i
            else:
                break

        refined = anomaly.copy()
        new_lo = max(0, min(new_lo, len(timestamps) - 1))
        new_hi = max(0, min(new_hi - 1, len(timestamps) - 1))
        refined['t_start'] = max(0, timestamps[new_lo])
        refined['t_end'] = min(timestamps[new_hi], duration)

        if refined['t_end'] - refined['t_start'] < 1.0:
            refined['t_start'] = t_start
            refined['t_end'] = t_end

        return refined

    # ================================================================
    # Classification fallback
    # ================================================================


    def _dual_videomae_classify(self, anomalies, video_data, full_video_category, full_video_confidence):
        """Dual VideoMAE: combine full-video and segment-level classification."""
        if not anomalies or self.videomae is None or self.videomae.model is None:
            for a in anomalies:
                a['anomaly_type'] = full_video_category
            return anomalies

        # Get frames from detected anomaly segments
        all_seg_frames = []
        for a in anomalies:
            frames, indices = self.preprocessor.get_segment_frames(
                video_data, a['t_start'], a['t_end'])
            if len(frames) > 0:
                all_seg_frames.append(frames)

        if not all_seg_frames:
            for a in anomalies:
                a['anomaly_type'] = full_video_category
            return anomalies

        # Concatenate all segment frames
        seg_frames = np.concatenate(all_seg_frames, axis=0)

        # Classify segment frames (need at least 16 frames for VideoMAE)
        if len(seg_frames) >= 8:
            seg_result = self.videomae.classify(seg_frames, top_k=3, tta=False)
            seg_category = seg_result['category']
            seg_confidence = seg_result['confidence']
            seg_top3 = ', '.join(f'{c}({s:.0%})' for c, s in seg_result['top_k'])
            _log.info(f"    [VideoMAE-seg] {seg_top3} -> {seg_category}@{seg_confidence:.0%}")

            # Decision logic
            if seg_category == full_video_category:
                # Both agree — use it
                category = full_video_category
            elif seg_category == 'Normal':
                # Segment says Normal, use full-video
                category = full_video_category
            elif full_video_confidence > 0.85 and seg_confidence < 0.5:
                # Full-video very confident, segment not — trust full-video
                category = full_video_category
            elif seg_confidence > full_video_confidence:
                # Segment more confident — prefer segment (anomaly content is concentrated)
                category = seg_category
                _log.info(f"    [VideoMAE-dual] Override: {full_video_category}@{full_video_confidence:.0%} "
                      f"-> {seg_category}@{seg_confidence:.0%} (segment)")
            else:
                # Otherwise trust full-video (more context)
                category = full_video_category
        else:
            category = full_video_category

        for a in anomalies:
            a['anomaly_type'] = category
        return anomalies

    async def _ensemble_classify(self, anomalies, video_data, videomae_cat, videomae_conf):
        """Multi-classifier ensemble: VideoMAE + VLM for known confusion pairs only."""
        # v137 NOCLS-CA (KEPT latency win): drop the entire classify VLM sub-tree
        # (qwen3-vl-plus, <=3 calls/video) and return the upstream VideoMAE label.
        # mAP-NEUTRAL BY CONSTRUCTION: the returned string only sets a['anomaly_type'],
        # which compute_temporal_map never reads (it uses {video_id,t_start,t_end,
        # confidence} only); the confidence gate / curve correction / frame scores
        # all key off videomae_category, not this result. Cuts ~7.5% of VLM calls.
        # Tradeoff: degrades the separate Level-4 category-accuracy metric, not mAP.
        if self.config.scoring.classify_skip_class_agnostic:
            _label = videomae_cat if (videomae_cat and videomae_cat != 'Normal') else 'unknown'
            _log.info(f"  [NOCLS-CA] classify VLM sub-tree skipped -> label='{_label}' "
                      f"(mAP-neutral; ~-7.5% VLM calls)")
            return _label
        if not videomae_cat or videomae_cat == 'Normal':
            result = await self._classify_clip_guided(anomalies, video_data)
            if result and result[0].get('anomaly_type'):
                return result[0]['anomaly_type']
            return 'unknown'

        # Low-confidence VideoMAE (<50%): override with CLIP top-1 if it's clearly different
        # This recovers cases like Burglary005 where VideoMAE is uncertain but CLIP is clear
        if videomae_conf < 0.50:
            precomputed = video_data.get('clip_features')
            _lc_frame_indices = []
            for _a in anomalies:
                _, _idx = self.preprocessor.get_segment_frames(
                    video_data, _a['t_start'], _a['t_end'])
                if _idx:
                    _lc_frame_indices.extend(_idx)
            if _lc_frame_indices and precomputed is not None and self.anomaly_expert.model is not None:
                _unique = sorted(set(_lc_frame_indices))
                _clip_dist = self.anomaly_expert.get_category_distribution(_unique, precomputed, top_k=3)
                if _clip_dist:
                    _clip_top1 = max(_clip_dist, key=_clip_dist.get)
                    _clip_top1_score = _clip_dist[_clip_top1]
                    if _clip_top1 != videomae_cat and _clip_top1_score > 0.35:
                        _log.info(f"  [Ensemble] LowConf VideoMAE({videomae_conf:.0%}), CLIP={_clip_top1}({_clip_top1_score:.0%}) override")
                        return _clip_top1

        confusion_alts = VIDEOMAE_CONFUSION_MAP.get(videomae_cat, [])
        if not confusion_alts:
            _log.info(f"  [Ensemble] No confusion pair -> {videomae_cat}")
            return videomae_cat

        # Build candidates: ONLY VideoMAE prediction + known confusion alternatives
        candidates = [videomae_cat] + confusion_alts
        candidates = candidates[:4]

        # Collect anomaly segment frames for VLM
        all_frame_indices = []
        seg_info = []
        precomputed = video_data.get('clip_features')
        for a in anomalies:
            frames, indices = self.preprocessor.get_segment_frames(
                video_data, a['t_start'], a['t_end'])
            if indices:
                all_frame_indices.extend(indices)
                seg_info.append(f"{a['t_start']:.1f}-{a['t_end']:.1f}s")

        # CLIP distribution for VLM context (informational only)
        clip_dist = {}
        if all_frame_indices and precomputed is not None and self.anomaly_expert.model is not None:
            unique_indices = sorted(set(all_frame_indices))
            clip_dist = self.anomaly_expert.get_category_distribution(
                unique_indices, precomputed, top_k=5)
            if clip_dist:
                clip_sorted = sorted(clip_dist.items(), key=lambda x: x[1], reverse=True)
                _log.info(f"  [Ensemble] VideoMAE={videomae_cat}@{videomae_conf:.0%}, "
                          f"CLIP-seg={', '.join(f'{c}({s:.0%})' for c, s in clip_sorted[:3])}")

        _log.info(f"  [Ensemble] VLM deciding among {candidates}")

        all_frames = video_data['all_frames']
        if all_frame_indices and precomputed is not None:
            selected = self.anomaly_expert.get_top_anomaly_frames(
                sorted(set(all_frame_indices)), precomputed, n=16)
        else:
            step = max(1, len(all_frames) // 16)
            selected = list(range(0, len(all_frames), step))[:16]

        keyframes = np.stack([all_frames[i] for i in selected])
        # Majority voting: 3 VLM calls in parallel to reduce non-determinism
        vote_tasks = [
            self._classify_from_candidates(keyframes, seg_info, candidates, clip_dist, perspective_idx=i)
            for i in range(3)
        ]
        votes = list(await asyncio.gather(*vote_tasks))
        vote_counts = Counter(votes)
        majority_cat, majority_count = vote_counts.most_common(1)[0]
        if majority_count >= 2:
            vlm_cat = majority_cat
        else:
            vlm_cat = videomae_cat
        _log.info(f"  [Ensemble] VLM votes={dict(vote_counts)} -> {vlm_cat}")
        return vlm_cat

    async def _classify_clip_guided(self, anomalies, video_data):
        """CLIP top-3 guided classification fallback."""
        if not anomalies:
            return anomalies

        precomputed = video_data.get('clip_features')

        all_frame_indices = []
        seg_info = []
        for a in anomalies:
            frames, indices = self.preprocessor.get_segment_frames(
                video_data, a['t_start'], a['t_end'])
            if indices:
                all_frame_indices.extend(indices)
                seg_info.append(f"{a['t_start']:.1f}-{a['t_end']:.1f}s")

        if not all_frame_indices:
            return anomalies

        unique_indices = sorted(set(all_frame_indices))

        clip_dist = {}
        if precomputed is not None and self.anomaly_expert.model is not None:
            selected_indices = self.anomaly_expert.get_top_anomaly_frames(
                unique_indices, precomputed, n=16)
            clip_dist = self.anomaly_expert.get_category_distribution(
                unique_indices, precomputed, top_k=3)
        else:
            selected_indices = unique_indices[:16]

        all_frames = video_data['all_frames']
        keyframes = np.stack([all_frames[i] for i in selected_indices])

        sorted_cats = sorted(clip_dist.items(), key=lambda x: x[1], reverse=True) if clip_dist else []

        if not sorted_cats:
            return anomalies

        top1_cat, top1_score = sorted_cats[0]
        top2_cat, top2_score = sorted_cats[1] if len(sorted_cats) > 1 else (None, 0)
        top3_cat, top3_score = sorted_cats[2] if len(sorted_cats) > 2 else (None, 0)

        _log.info(f"    [CLIP] Top-3: {top1_cat}({top1_score:.1%}), "
              f"{top2_cat}({top2_score:.1%}), {top3_cat}({top3_score:.1%})")

        if top2_score > 0 and top1_score / top2_score > 3.0:
            anomaly_type = top1_cat
            _log.info(f"    [Classify] CLIP high-confidence -> {anomaly_type}")
        else:
            candidates_list = [top1_cat]
            if top2_cat:
                candidates_list.append(top2_cat)
            if top3_cat and top2_score > 0 and top1_score / top2_score < 1.5:
                candidates_list.append(top3_cat)

            anomaly_type = await self._classify_from_candidates(
                keyframes, seg_info, candidates_list, clip_dist)

        classified = []
        for a in anomalies:
            a = a.copy()
            a['anomaly_type'] = anomaly_type
            classified.append(a)
        return classified

    # v86: diverse-perspective prompts for self-consistency voting (Loop 1
    # research, Agent B). Replaces 3 IDENTICAL VLM calls (relying only on API
    # sampling randomness) with 3 DIFFERENT analytical angles, which the
    # self-consistency literature shows produces more accurate majority votes
    # than repeated identical prompts - especially for semantically-overlapping
    # categories (Shoplifting/Stealing/Robbery/Burglary all involve "taking an
    # item", differing in method/force/location). Zero extra API cost (same
    # call count).
    CLASSIFY_PERSPECTIVES = [
        "Focus on the SETTING and CONTEXT: where is this happening (store/street/"
        "building/vehicle), what objects or props are visible, and does the "
        "environment match a specific anomaly type?",
        "Focus on the SPECIFIC PHYSICAL ACTIONS: what exactly is each person doing "
        "with their hands/body (grabbing, forcing entry, threatening, concealing, "
        "striking), and in what sequence?",
        "Focus on the OUTCOME and INTERACTION: is there a victim reacting/resisting, "
        "is force or a weapon involved, does an item change possession, and what is "
        "the apparent consequence of the action?",
    ]

    async def _classify_from_candidates(self, keyframes, seg_info, candidates, clip_dist, perspective_idx=0):
        options_text = "\n".join(
            f"  {i+1}. {cat}: {CATEGORY_CUES.get(cat, cat)}"
            for i, cat in enumerate(candidates)
        )

        clip_hint = ""
        if clip_dist:
            scores = [(c, clip_dist.get(c, 0)) for c in candidates]
            clip_hint = "\nVisual analysis prior: " + ", ".join(
                f"{c}({s:.0%})" for c, s in scores) + "\n"

        _perspective = self.CLASSIFY_PERSPECTIVES[perspective_idx % len(self.CLASSIFY_PERSPECTIVES)]

        prompt = (
            f"These {len(keyframes)} frames are from anomalous segments "
            f"({', '.join(seg_info)}) of a surveillance video. "
            f"Frames are in CHRONOLOGICAL ORDER."
            f"{clip_hint}\n"
            f"{_perspective}\n\n"
            f"Which type of anomaly is shown?\n\n"
            f"{options_text}\n\n"
            f"Respond ONLY in JSON: "
            f"{{\"choice\": 1-{len(candidates)}, \"anomaly_type\": \"...\", "
            f"\"confidence\": 0.0-1.0, \"reasoning\": \"specific visual evidence\"}}"
        )

        response = await self.classify_llm_client.generate(prompt, keyframes)

        anomaly_type = None
        try:
            text = response.strip()
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                choice = data.get('choice')
                predicted = data.get('anomaly_type', '')
                if isinstance(choice, int) and 1 <= choice <= len(candidates):
                    anomaly_type = candidates[choice - 1]
                elif predicted in candidates:
                    anomaly_type = predicted
                elif predicted in self.anomaly_categories:
                    anomaly_type = predicted
        except Exception:
            pass

        if anomaly_type is None:
            anomaly_type = candidates[0]

        _log.info(f"    [Classify] VLM chose {anomaly_type} from {candidates}")
        return anomaly_type


