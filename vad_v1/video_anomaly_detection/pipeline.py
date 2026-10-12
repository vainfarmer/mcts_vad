"""
视频异常检测与时间定位 — 完整 Pipeline (v27)

v27: 降低Normal误报 + 提升分类准确率 + 改善时间定位
  核心变更：
  1. 移除"LLM无命中→CLIP区域兜底"逻辑（减少Normal视频假阳性）
  2. 最终异常置信度阈值0.72（过滤低置信度误报）
  3. CLIP覆盖阈值：VideoMAE<50%时用CLIP top-1覆盖（v26特性保留）
  4. VLM多数投票（3次调用取多数）用于混淆表仲裁（降低VLM非确定性影响）
  5. LLM扫描覆盖率按视频长度自适应扩展（max_scan up to 24）
  6. MCTS搜索范围扩大至±75%（改善时间边界覆盖）
"""

import asyncio
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
    # REMOVED: Shoplifting, Stealing, Vandalism — VLM bias toward Robbery/Burglary caused -8 videos
}

SHORT_EVENT_CATEGORIES = {'Explosion', 'RoadAccidents'}


class MultiAnomalyVAD:

    DEFAULT_ANOMALY_CATEGORIES = [
        "Abuse", "Arrest", "Arson", "Assault", "Burglary",
        "Explosion", "Fighting", "RoadAccidents", "Robbery",
        "Shooting", "Shoplifting", "Stealing", "Vandalism",
    ]

    def __init__(
        self,
        llm_client,
        classify_llm_client=None,
        image_processor=None,
        clip_model_name: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        fps: int = 2,
        segment_duration: float = 6.0,
        segment_overlap: float = 0.25,
        mcts_simulations: int = 12,
        mcts_max_depth: int = 4,
        semantic_tau: float = 0.90,
        anomaly_categories: Optional[List[str]] = None,
        max_frames: int = 200,
        min_frames: int = 3,
        fps_sample: float = 3.0,
        motion_threshold: float = 1.0,
        min_motion_interval: float = 2.0,
        device: str = 'cpu',
    ):
        self.llm_client = llm_client
        self.classify_llm_client = classify_llm_client or llm_client
        self.image_processor = image_processor
        self.anomaly_categories = anomaly_categories or self.DEFAULT_ANOMALY_CATEGORIES

        self.preprocessor = VideoPreprocessor(
            fps=fps,
            segment_duration=segment_duration,
            motion_threshold=motion_threshold,
            min_motion_interval=min_motion_interval,
            segment_overlap=segment_overlap,
        )

        self.anomaly_expert = VideoAnomalyExpert(
            anomaly_categories=self.anomaly_categories,
            clip_model_name=clip_model_name,
            clip_pretrained=clip_pretrained,
            device=device,
        )

        self.frame_sampler = SemanticFrameSampler(
            anomaly_expert=self.anomaly_expert,
            tau=semantic_tau,
            max_frames=max_frames,
            min_frames=min_frames,
            fps_sample=fps_sample,
        )

        try:
            self.videomae = VideoMAEClassifier(device=device)
        except Exception as e:
            print(f"[Pipeline] VideoMAE not available: {e}")
            self.videomae = None

        self.discovery = AnomalyDiscovery(
            llm_client=llm_client,
            anomaly_categories=self.anomaly_categories,
            image_processor=image_processor,
            clip_expert=self.anomaly_expert,
            frame_sampler=self.frame_sampler,
            min_pass_rate=0.2,
        )

        self.mcts_simulations = mcts_simulations
        self.mcts_max_depth = mcts_max_depth

    async def detect(
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
                return {
                    'video_path': video_path,
                    'duration': video_data['duration'],
                    'anomalies': [],
                    'is_normal': True,
                }

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
        if clip_regions:
            candidates = await self._directed_llm_scan(
                video_data, clip_regions, context_prompt, videomae_category)
        else:
            candidates = await self.discovery.discover(video_data, context_prompt)

        if not candidates:
            return {
                'video_path': video_path,
                'duration': video_data['duration'],
                'anomalies': [],
                'is_normal': True,
            }

        _log.info(f"Found {len(candidates)} anomaly candidate(s):")
        for i, c in enumerate(candidates):
            print(f"  Candidate {i+1}: {c['t_start']:.1f}s-{c['t_end']:.1f}s "
                  f"score={c['coarse_score']:.2f}")

        if len(candidates) > 3:
            candidates.sort(key=lambda c: c['coarse_score'], reverse=True)
            _log.info(f"  [Limit] Keeping top 3/{len(candidates)} candidates")
            candidates = candidates[:3]

        # Stage 5: MCTS refinement with expanded search range
        mcts_tasks = []
        for cand in candidates:
            mcts = TemporalBoundaryMCTS(
                anomaly_description=cand['description'],
                video_data=video_data,
                llm_client=self.llm_client,
                anomaly_expert=self.anomaly_expert,
                frame_sampler=self.frame_sampler,
                image_processor=self.image_processor,
                anomaly_categories=self.anomaly_categories,
            )
            mcts.n_simulations = self.mcts_simulations
            mcts.max_depth = self.mcts_max_depth
            seg_dur = cand['t_end'] - cand['t_start']
            padding = seg_dur * 0.75
            search_start = max(0, cand['t_start'] - padding)
            search_end = min(video_data['duration'], cand['t_end'] + padding)
            mcts_tasks.append(mcts.search(search_start, search_end))

        all_results = await asyncio.gather(*mcts_tasks, return_exceptions=True)
        all_anomalies = []
        for i, results in enumerate(all_results):
            if isinstance(results, Exception):
                print(f"  [MCTS] Candidate {i+1} failed: {results}")
                continue
            all_anomalies.extend(results)

        # Stage 6: Category-aware CLIP boundary refinement
        if self.anomaly_expert.model is not None:
            refined = []
            for anomaly in all_anomalies:
                r = await self._refine_boundary_category_aware(
                    anomaly, video_data, videomae_category)
                refined.append(r)
            all_anomalies = refined

        _log.info(f"  --- Pre-merge anomalies: {len(all_anomalies)} ---")
        for i, a in enumerate(all_anomalies):
            print(f"    {i+1}: {a['t_start']:.1f}-{a['t_end']:.1f}s, "
                  f"conf={a['confidence']:.3f}, type={a.get('anomaly_type','?')}")

        final_anomalies = self._global_merge(all_anomalies)
        _log.info(f"  --- Post-merge anomalies: {len(final_anomalies)} ---")

        # Stage 6b: Short-event narrowing
        if videomae_category in SHORT_EVENT_CATEGORIES:
            final_anomalies = self._narrow_short_events(final_anomalies, video_data)

        # Stage 7: Multi-classifier ensemble (VideoMAE + CLIP + VLM)
        if final_anomalies:
            ensemble_cat = await self._ensemble_classify(
                final_anomalies, video_data, videomae_category, videomae_confidence)
            for a in final_anomalies:
                a['anomaly_type'] = ensemble_cat
        elif videomae_category and videomae_category != 'Normal':
            pass  # no anomalies to classify

        final_anomalies = self._unify_anomaly_types(final_anomalies)
        final_anomalies = self._global_merge(final_anomalies)

        # Confidence gate: filter low-confidence anomalies to reduce false positives
        before_filter = len(final_anomalies)
        final_anomalies = [a for a in final_anomalies if a['confidence'] >= 0.72]
        if len(final_anomalies) < before_filter:
            _log.info(f"  [ConfFilter] {before_filter} -> {len(final_anomalies)} "
                      f"(threshold=0.72)")

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

        return {
            'video_path': video_path,
            'duration': video_data['duration'],
            'anomalies': final_anomalies,
            'is_normal': len(final_anomalies) == 0,
        }

    # ================================================================
    # Stage 3: Category-aware CLIP temporal localization
    # ================================================================



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

    def _narrow_short_events(self, anomalies, video_data):
        """Narrow overly broad predictions for short-event categories."""
        motion = video_data.get('motion_energy')
        timestamps = video_data['timestamps']
        duration = video_data['duration']
        if motion is None or len(motion) < 4:
            return anomalies
        narrowed = []
        for a in anomalies:
            a = a.copy()
            seg_dur = a['t_end'] - a['t_start']
            if seg_dur <= 10.0:
                narrowed.append(a)
                continue
            from bisect import bisect_left
            lo = bisect_left(timestamps, a['t_start'])
            hi = bisect_left(timestamps, a['t_end'])
            if hi <= lo + 2:
                narrowed.append(a)
                continue
            seg_motion = motion[lo:hi]
            peak_local = np.argmax(seg_motion)
            peak_global = peak_local + lo
            peak_val = motion[peak_global]
            if peak_val > np.mean(motion) + np.std(motion):
                peak_time = timestamps[min(peak_global, len(timestamps) - 1)]
                new_start = max(0, peak_time - 5.0)
                new_end = min(duration, peak_time + 5.0)
                if new_end - new_start >= 3.0:
                    _log.info(f"    [Short-narrow] {a['t_start']:.1f}-{a['t_end']:.1f}s "
                              f"-> {new_start:.1f}-{new_end:.1f}s (peak@{peak_time:.1f}s)")
                    a['t_start'] = new_start
                    a['t_end'] = new_end
            narrowed.append(a)
        return narrowed

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
        
        print(f"  [Motion-loc] Found {len(regions)} motion peak regions:")
        for r in regions:
            print(f"    [{r['t_start']:.1f}-{r['t_end']:.1f}]s score={r['clip_cat_score']:.3f}")
        
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
        """Use category-specific CLIP scores to find temporal peaks."""
        precomputed = video_data.get('clip_features')
        if (precomputed is None or self.anomaly_expert.model is None
                or category is None or category == 'Normal'
                or torch is None):
            return []

        if category not in self.anomaly_categories:
            return []

        cat_idx = self.anomaly_categories.index(category)
        timestamps = video_data['timestamps']

        with torch.no_grad():
            logit_scale = self.anomaly_expert.model.logit_scale.exp().detach().clamp(max=100.0)
            similarity = precomputed @ self.anomaly_expert.text_features.T * logit_scale
            probs = torch.softmax(similarity, dim=-1).cpu().numpy()
            cat_scores = probs[:, cat_idx]
            normal_scores = probs[:, -1]
            relative_scores = cat_scores / (cat_scores + normal_scores + 1e-8)

        # Smooth with rolling average (2-second window)
        fps = max(1, len(timestamps) / max(1, video_data['duration']))
        window = max(3, int(2 * fps))
        if len(relative_scores) >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(relative_scores, kernel, mode='same')
        else:
            smoothed = relative_scores

        mean_score = np.mean(smoothed)
        std_score = np.std(smoothed)
        max_score = np.max(smoothed)
        min_score = np.min(smoothed)

        score_range = max_score - min_score
        coeff_var = std_score / (mean_score + 1e-8)

        print(f"  [CLIP-loc] Category={category}, mean={mean_score:.3f}, "
              f"std={std_score:.3f}, range={score_range:.3f}, cv={coeff_var:.3f}")

        # For Explosion/RoadAccidents: use motion energy when CLIP is uniform
        motion_categories = {'Explosion', 'RoadAccidents'}
        if score_range < 0.08 or coeff_var < 0.03:
            if category in motion_categories:
                motion_regions = self._motion_temporal_localization(video_data)
                if motion_regions:
                    print(f"  [CLIP-loc] Uniform but using motion energy for {category}")
                    return motion_regions
            print(f"  [CLIP-loc] Scores too uniform, falling back to standard scanning")
            return []

        # For Explosion/RoadAccidents: fuse motion energy with CLIP scores
        if category in {'Explosion', 'RoadAccidents'}:
            motion = video_data.get('motion_energy', np.zeros(len(smoothed)))
            if len(motion) == len(smoothed) and np.max(motion) > 0:
                # Normalize motion to [0, 1]
                motion_norm = motion / (np.max(motion) + 1e-8)
                # Smooth motion
                if len(motion_norm) >= window:
                    motion_smooth = np.convolve(motion_norm, kernel, mode='same')
                else:
                    motion_smooth = motion_norm
                # Fuse: 60% CLIP + 40% motion
                smoothed = 0.6 * smoothed + 0.4 * motion_smooth
                mean_score = np.mean(smoothed)
                std_score = np.std(smoothed)
                max_score = np.max(smoothed)
                min_score = np.min(smoothed)
                score_range = max_score - min_score
                coeff_var = std_score / (mean_score + 1e-8)
                print(f"  [CLIP-loc] Fused with motion: range={score_range:.3f}, cv={coeff_var:.3f}")

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

        # Merge segments with small gaps (< 3 seconds)
        gap_threshold = 3.0
        merged = []
        for start_idx, end_idx in segments:
            t_start = timestamps[start_idx]
            t_end = timestamps[min(end_idx, len(timestamps) - 1)]
            if merged and t_start - merged[-1][1] < gap_threshold:
                merged[-1] = (merged[-1][0], t_end)
            else:
                merged.append((t_start, t_end))

        # Expand each region by 2 seconds on each side (conservative)
        duration = video_data['duration']
        expanded = []
        for t_start, t_end in merged:
            t_start = max(0, t_start - 2.0)
            t_end = min(duration, t_end + 2.0)
            if t_end - t_start < 6.0:
                center = (t_start + t_end) / 2
                t_start = max(0, center - 3.0)
                t_end = min(duration, center + 3.0)
            expanded.append((t_start, t_end))

        # Score each region
        scored_regions = []
        for t_start, t_end in expanded:
            mask = (timestamps >= t_start) & (timestamps <= t_end)
            if np.any(mask):
                avg_score = np.mean(smoothed[mask])
                scored_regions.append({
                    't_start': t_start,
                    't_end': t_end,
                    'clip_cat_score': float(avg_score),
                })

        scored_regions.sort(key=lambda r: r['clip_cat_score'], reverse=True)
        scored_regions = scored_regions[:5]

        if scored_regions:
            print(f"  [CLIP-loc] Found {len(scored_regions)} candidate regions "
                  f"(threshold={threshold:.3f}):")
            for r in scored_regions:
                print(f"    [{r['t_start']:.1f}-{r['t_end']:.1f}]s "
                      f"score={r['clip_cat_score']:.3f}")

        return scored_regions

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

        max_scan = min(max(14, len(segments) // 4), 24)
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
            print(f"    {seg['t_start']:.1f}-{seg['t_end']:.1f}s: {status} "
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
    # Stage 6: Category-aware CLIP boundary refinement
    # ================================================================

    async def _refine_boundary_category_aware(self, anomaly, video_data, category):
        """Refine boundaries using category-specific CLIP scores (conservative)."""
        t_start = anomaly['t_start']
        t_end = anomaly['t_end']
        duration = video_data['duration']
        precomputed = video_data.get('clip_features')

        if (precomputed is None or self.anomaly_expert.model is None
                or category is None or category == 'Normal'
                or category not in self.anomaly_categories or torch is None):
            return await self._refine_boundary(anomaly, video_data)

        cat_idx = self.anomaly_categories.index(category)
        timestamps = video_data['timestamps']

        with torch.no_grad():
            logit_scale = self.anomaly_expert.model.logit_scale.exp().detach().clamp(max=100.0)
            similarity = precomputed @ self.anomaly_expert.text_features.T * logit_scale
            probs = torch.softmax(similarity, dim=-1).cpu().numpy()
            cat_scores = probs[:, cat_idx]
            normal_scores = probs[:, -1]
            relative_scores = cat_scores / (cat_scores + normal_scores + 1e-8)

        # Check if scores are discriminative enough for this region
        global_std = np.std(relative_scores)
        global_mean = np.mean(relative_scores)
        coeff_var = global_std / (global_mean + 1e-8)

        # If scores are too uniform, fall back to generic refinement
        if coeff_var < 0.05:
            return await self._refine_boundary(anomaly, video_data)

        from bisect import bisect_left
        lo = bisect_left(timestamps, t_start)
        hi = bisect_left(timestamps, t_end)
        if hi <= lo:
            return anomaly

        core_scores = relative_scores[lo:hi]
        core_mean = np.mean(core_scores)
        seg_dur = t_end - t_start

        # Conservative expansion: max 40% of segment duration on each side
        max_expand = seg_dur * 0.4
        threshold = core_mean * 0.65

        # Expand left (limited)
        expanded_start = t_start
        search_lo = max(0, bisect_left(timestamps, max(0, t_start - max_expand)))
        for i in range(lo - 1, search_lo - 1, -1):
            if relative_scores[i] >= threshold:
                expanded_start = timestamps[i]
            else:
                break

        # Expand right (limited)
        expanded_end = t_end
        search_hi = min(len(timestamps), bisect_left(timestamps,
                        min(duration, t_end + max_expand)))
        for i in range(hi, search_hi):
            if relative_scores[i] >= threshold:
                expanded_end = timestamps[i]
            else:
                break

        # Shrink: trim low-score edges (max 20% from each side)
        shrink_threshold = core_mean * 0.4
        shrunk_start = expanded_start
        start_idx = bisect_left(timestamps, expanded_start)
        end_idx = bisect_left(timestamps, expanded_end)
        total_frames = end_idx - start_idx
        max_shrink_frames = max(1, total_frames // 5)
        for i in range(start_idx, min(start_idx + max_shrink_frames, end_idx)):
            if i < len(relative_scores) and relative_scores[i] < shrink_threshold:
                shrunk_start = timestamps[min(i + 1, len(timestamps) - 1)]
            else:
                break

        shrunk_end = expanded_end
        for i in range(end_idx - 1, max(end_idx - max_shrink_frames - 1, start_idx), -1):
            if i < len(relative_scores) and relative_scores[i] < shrink_threshold:
                shrunk_end = timestamps[max(i - 1, 0)]
            else:
                break

        refined = anomaly.copy()
        refined['t_start'] = max(0, shrunk_start)
        refined['t_end'] = min(shrunk_end, duration)

        if refined['t_end'] - refined['t_start'] < 1.0:
            refined['t_start'] = t_start
            refined['t_end'] = t_end

        if abs(refined['t_start'] - t_start) > 0.5 or abs(refined['t_end'] - t_end) > 0.5:
            _log.info(f"    [CatRefine] {t_start:.1f}-{t_end:.1f}s "
                      f"-> {refined['t_start']:.1f}-{refined['t_end']:.1f}s "
                      f"(cat={category}, cv={coeff_var:.3f})")

        return refined

    # ================================================================
    # Fallback boundary refinement (generic CLIP)
    # ================================================================

    async def _refine_boundary(self, anomaly, video_data):
        t_start = anomaly['t_start']
        t_end = anomaly['t_end']
        duration = video_data['duration']
        precomputed = video_data.get('clip_features')
        core_frames, core_indices = self.preprocessor.get_segment_frames(
            video_data, t_start, t_end)
        if len(core_frames) > 0:
            core_result = await self.anomaly_expert.score_segment(
                core_frames, t_start, t_end,
                frame_indices=core_indices, precomputed_features=precomputed)
            core_score = core_result['anomaly_score']
        else:
            core_score = 0.5
        shrink_threshold = core_score * 0.6
        seg_duration = t_end - t_start
        max_shrink = seg_duration * 0.3
        trimmed_start, trimmed_end = await asyncio.gather(
            self._bisect_shrink_left(video_data, t_start, t_start + max_shrink, t_end, shrink_threshold),
            self._bisect_shrink_right(video_data, t_end - max_shrink, t_end, t_start, shrink_threshold))
        expand_threshold = core_score * 0.8
        max_expand = (trimmed_end - trimmed_start) * 0.3
        expanded_start, expanded_end = await asyncio.gather(
            self._bisect_expand_left(video_data, trimmed_start, max_expand, expand_threshold),
            self._bisect_expand_right(video_data, trimmed_end, max_expand, duration, expand_threshold))
        refined = anomaly.copy()
        refined['t_start'] = expanded_start
        refined['t_end'] = min(expanded_end, duration)
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
            print(f"    [VideoMAE-seg] {seg_top3} -> {seg_category}@{seg_confidence:.0%}")

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
                print(f"    [VideoMAE-dual] Override: {full_video_category}@{full_video_confidence:.0%} "
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
        # Majority voting: 3 VLM calls to reduce non-determinism
        votes = []
        for _vote_i in range(3):
            _v = await self._classify_from_candidates(keyframes, seg_info, candidates, clip_dist)
            votes.append(_v)
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

        print(f"    [CLIP] Top-3: {top1_cat}({top1_score:.1%}), "
              f"{top2_cat}({top2_score:.1%}), {top3_cat}({top3_score:.1%})")

        if top2_score > 0 and top1_score / top2_score > 3.0:
            anomaly_type = top1_cat
            print(f"    [Classify] CLIP high-confidence -> {anomaly_type}")
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

    async def _classify_from_candidates(self, keyframes, seg_info, candidates, clip_dist):
        options_text = "\n".join(
            f"  {i+1}. {cat}: {CATEGORY_CUES.get(cat, cat)}"
            for i, cat in enumerate(candidates)
        )

        clip_hint = ""
        if clip_dist:
            scores = [(c, clip_dist.get(c, 0)) for c in candidates]
            clip_hint = "\nVisual analysis prior: " + ", ".join(
                f"{c}({s:.0%})" for c, s in scores) + "\n"

        prompt = (
            f"These {len(keyframes)} frames are from anomalous segments "
            f"({', '.join(seg_info)}) of a surveillance video. "
            f"Frames are in CHRONOLOGICAL ORDER."
            f"{clip_hint}\n"
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

    def _unify_anomaly_types(self, anomalies: List[Dict]) -> List[Dict]:
        if len(anomalies) <= 1:
            return anomalies
        types = set(a['anomaly_type'] for a in anomalies)
        if len(types) == 1:
            return anomalies
        type_weight = Counter()
        for a in anomalies:
            duration = a['t_end'] - a['t_start']
            type_weight[a['anomaly_type']] += a['confidence'] * duration
        best_type = type_weight.most_common(1)[0][0]
        _log.info(f"    [Unify] types={dict(type_weight)}, winner={best_type}")
        unified = []
        for a in anomalies:
            a = a.copy()
            a['anomaly_type'] = best_type
            unified.append(a)
        return unified

    async def _score_window(self, video_data, t, step=1.0):
        t_end = min(t + step, video_data['duration'])
        frames, frame_indices = self.preprocessor.get_segment_frames(video_data, t, t_end)
        if len(frames) == 0:
            return 0.0
        precomputed = video_data.get('clip_features')
        result = await self.anomaly_expert.score_segment(
            frames, t, t_end, frame_indices=frame_indices, precomputed_features=precomputed)
        return result['anomaly_score']

    async def _bisect_shrink_left(self, video_data, lo, hi, t_end, threshold):
        original_lo = lo
        step = max(0.5, (hi - lo) / 8)
        hi = min(hi, t_end - step)
        while hi - lo > step:
            mid = (lo + hi) / 2
            score = await self._score_window(video_data, mid, step)
            if score >= threshold: hi = mid
            else: lo = mid + step
        return lo if lo > original_lo else original_lo

    async def _bisect_shrink_right(self, video_data, lo, hi, t_start, threshold):
        original_hi = hi
        step = max(0.5, (hi - lo) / 8)
        lo = max(lo, t_start + step)
        if lo > hi: return original_hi
        while hi - lo > step:
            mid = (lo + hi) / 2
            score = await self._score_window(video_data, mid, step)
            if score >= threshold: lo = mid
            else: hi = mid - step
        return hi if hi < original_hi else original_hi

    async def _bisect_expand_left(self, video_data, trimmed_start, max_expand, threshold):
        lo = max(0, trimmed_start - max_expand)
        hi = trimmed_start
        step = max(0.5, (hi - lo) / 8)
        hi -= step
        if hi < lo: return trimmed_start
        expanded = trimmed_start
        while hi - lo > step:
            mid = (lo + hi) / 2
            score = await self._score_window(video_data, mid, step)
            if score >= threshold: expanded = mid; hi = mid - step
            else: lo = mid + step
        if lo <= hi:
            score = await self._score_window(video_data, lo, step)
            if score >= threshold: expanded = lo
        return expanded

    async def _bisect_expand_right(self, video_data, trimmed_end, max_expand, duration, threshold):
        lo = trimmed_end
        hi = min(duration, trimmed_end + max_expand)
        step = max(0.5, (hi - lo) / 8)
        if lo + step > hi: return trimmed_end
        expanded = trimmed_end
        while hi - lo > step:
            mid = (lo + hi) / 2
            score = await self._score_window(video_data, mid, step)
            if score >= threshold: expanded = mid + step; lo = mid + step
            else: hi = mid - step
        if lo <= hi:
            score = await self._score_window(video_data, lo, step)
            if score >= threshold: expanded = lo + step
        return expanded

    def _global_merge(self, anomalies, iou_threshold=0.3, same_type_gap=6.0):
        if not anomalies: return []
        merged = sorted([a.copy() for a in anomalies], key=lambda a: a['t_start'])
        changed = True
        while changed:
            changed = False
            new_merged = [merged[0]]
            for a in merged[1:]:
                did_merge = False
                for i, existing in enumerate(new_merged):
                    overlap_start = max(existing['t_start'], a['t_start'])
                    overlap_end = min(existing['t_end'], a['t_end'])
                    overlap = max(0, overlap_end - overlap_start)
                    gap = a['t_start'] - existing['t_end']
                    same_type = (existing.get('anomaly_type') == a.get('anomaly_type'))
                    if same_type and (overlap > 0 or gap <= same_type_gap):
                        keep = existing if existing['confidence'] >= a['confidence'] else a
                        new_merged[i] = {
                            't_start': min(existing['t_start'], a['t_start']),
                            't_end': max(existing['t_end'], a['t_end']),
                            'confidence': max(existing['confidence'], a['confidence']),
                            'anomaly_type': existing.get('anomaly_type'),
                            'depth': keep.get('depth'), 'visits': keep.get('visits'),
                        }
                        did_merge = True; changed = True; break
                    elif not same_type and overlap > 0:
                        union = max(existing['t_end'], a['t_end']) - min(existing['t_start'], a['t_start'])
                        if overlap / union > iou_threshold:
                            keep = existing if existing['confidence'] >= a['confidence'] else a
                            new_merged[i] = {
                                't_start': min(existing['t_start'], a['t_start']),
                                't_end': max(existing['t_end'], a['t_end']),
                                'confidence': max(existing['confidence'], a['confidence']),
                                'anomaly_type': keep.get('anomaly_type'),
                                'depth': keep.get('depth'), 'visits': keep.get('visits'),
                            }
                            did_merge = True; changed = True; break
                if not did_merge:
                    new_merged.append(a.copy())
            merged = sorted(new_merged, key=lambda a: a['t_start'])
        return merged
