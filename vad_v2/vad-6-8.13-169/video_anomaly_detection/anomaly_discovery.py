"""
粗粒度异常发现模块（Pipeline Stage 2）(v4)

v4 改进：
  - 检测 prompt 增加类别线索（不分类，但提高检测灵敏度）
  - 降低置信度阈值 0.3 → 0.2（减少漏检）
  - 降低 min_pass_rate 0.5 → 0.4（减少无效 LLM 调用）
"""

import asyncio
import json
import numpy as np
from typing import List, Dict, Any

try:
    import vad_logger as _vl
    _log = _vl.get("vad.discovery")
except Exception:
    import logging
    _log = logging.getLogger("vad.discovery")


class AnomalyDiscovery:

    def __init__(self, llm_client, anomaly_categories=None, image_processor=None,
                 clip_expert=None, clip_threshold=0.3, frame_sampler=None,
                 min_pass_rate=0.4, max_scan_segments=6):
        self.llm = llm_client
        self.categories = anomaly_categories or []
        self.image_processor = image_processor
        self.clip_expert = clip_expert
        self.clip_threshold = clip_threshold
        self.frame_sampler = frame_sampler
        self.min_pass_rate = min_pass_rate
        self.max_scan_segments = max_scan_segments
        # Per-video scan accounting (reset by the pipeline before Stage 4 each
        # video; internal instrumentation only).
        self.scan_segments_count = 0

    def _sample_keyframes(self, segment, all_frames, precomputed_features=None):
        duration = segment['t_end'] - segment['t_start']
        frames = all_frames[segment['frame_indices']]
        if self.frame_sampler is not None:
            return self.frame_sampler.select_keyframes(
                frames, duration=duration,
                frame_indices=segment['frame_indices'],
                precomputed_features=precomputed_features,
            )
        n = min(4, len(frames))
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        return frames[indices]

    async def scan_segment(self, segment, all_frames, context_prompt="",
                           precomputed_features=None):
        keyframes = self._sample_keyframes(segment, all_frames, precomputed_features)

        self.scan_segments_count += 1

        prompt = f"""Analyze these frames from a surveillance video segment (time: {segment['t_start']:.1f}s - {segment['t_end']:.1f}s).
{context_prompt}

Task: Determine if this segment contains any abnormal or suspicious activity.

Look carefully for ANY of these types of anomalies:
- Violence: fighting, assault, abuse, shooting (people falling without contact, crowds scattering)
- Crime: robbery (weapon threats), burglary (breaking into buildings), stealing, shoplifting (concealing items in a store)
- Property: vandalism (smashing/damaging objects), arson (deliberately starting fire)
- Incidents: explosions (sudden blasts), road accidents (vehicle collisions)
- Law enforcement: arrests (uniformed officers restraining someone)

Even subtle signs count: nervous behavior, concealing items, someone cowering, people suddenly running, smoke or fire starting, a person breaking a window or lock.

Respond ONLY in JSON format (no other text):
{{"is_anomalous": true or false, "confidence": 0.0 to 1.0, "description": "what abnormal activity you see, or null if normal"}}"""

        # v92->v94 REVERT: borderline-case majority-vote revoting regressed
        # full-eval mAP 0.1494->0.1183. Root cause hypothesis (refined from
        # V90's lesson): self-consistency doesn't just depend on task TYPE
        # (categorical vs numerical) - it also depends on WHERE in the
        # pipeline it acts. This changed WHICH SEGMENTS get included as
        # candidates in the first place (a segment's is_suspicious flip
        # determines whether it enters the candidate pool at all), which is
        # functionally similar to the already-confirmed-harmful "candidate
        # pool modification" risk category, just triggered by revoting
        # instead of a different model. Reverted to the single original call.
        response = await self.llm.generate(prompt, keyframes)
        result = self._parse_response(response)
        return result

    def _parse_response(self, response):
        try:
            text = response.strip()
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
            else:
                data = {}
        except (json.JSONDecodeError, ValueError):
            data = {}

        is_anomalous = data.get('is_anomalous', False)
        confidence = float(data.get('confidence', 0.0))
        description = data.get('description', None)

        return {
            'is_suspicious': is_anomalous and confidence > 0.3,
            'anomaly_score': confidence,
            'description': description or 'unknown anomaly',
        }

    async def discover(self, video_data, context_prompt="") -> List[Dict[str, Any]]:
        segments = video_data['segments']
        total = len(segments)

        all_frames = video_data['all_frames']
        precomputed = video_data.get('clip_features')
        if self.clip_expert is not None and self.clip_expert.model is not None:
            _log.info(f"  [CLIP pre-screen] Scanning {total} segments...")
            if precomputed is not None:
                clip_results = self.clip_expert.batch_score_segments(
                    segments, precomputed)
            else:
                clip_results = []
                for seg in segments:
                    seg_frames = all_frames[seg['frame_indices']]
                    score = await self.clip_expert.score_segment(
                        seg_frames, seg['t_start'], seg['t_end'],
                    )
                    clip_results.append(score)

            all_scores = [r['anomaly_score'] for r in clip_results]
            mean_score = np.mean(all_scores)
            std_score = np.std(all_scores)
            adaptive_thresh = max(self.clip_threshold, mean_score - 0.5 * std_score)
            _log.info(f"  [CLIP pre-screen] Adaptive threshold: {adaptive_thresh:.3f} "
                  f"(mean={mean_score:.3f}, std={std_score:.3f}, "
                  f"range: {min(all_scores):.2f}-{max(all_scores):.2f})")

            scored_indices = sorted(range(total), key=lambda i: all_scores[i], reverse=True)
            min_pass = max(1, int(total * self.min_pass_rate))

            filtered_indices = set()
            for i in range(total):
                if all_scores[i] >= adaptive_thresh:
                    filtered_indices.add(i)
            for i in scored_indices:
                if len(filtered_indices) >= min_pass:
                    break
                filtered_indices.add(i)

            filtered_indices = sorted(filtered_indices)

            for i in range(total):
                seg = segments[i]
                status = "KEEP" if i in filtered_indices else "SKIP"
                _log.debug(f"    Seg {i+1} ({seg['t_start']:.1f}-{seg['t_end']:.1f}s): "
                      f"CLIP={all_scores[i]:.2f} -> {status}")

            segments_to_scan = [segments[i] for i in filtered_indices]
            if len(segments_to_scan) > self.max_scan_segments:
                ranked = sorted(filtered_indices, key=lambda i: all_scores[i], reverse=True)
                filtered_indices = sorted(ranked[:self.max_scan_segments])
                segments_to_scan = [segments[i] for i in filtered_indices]
                _log.info(f"  [CLIP pre-screen] Limited to top {self.max_scan_segments} segments")
            _log.info(f"  [CLIP pre-screen] {len(segments_to_scan)}/{total} segments passed")
        else:
            segments_to_scan = segments

        if not segments_to_scan:
            return []

        _log.info(f"  [LLM scan] Scanning {len(segments_to_scan)} segments in parallel...")
        tasks = [self.scan_segment(seg, all_frames, context_prompt, precomputed)
                 for seg in segments_to_scan]
        results = await asyncio.gather(*tasks)

        suspicious = []
        for seg, result in zip(segments_to_scan, results):
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

        return self._merge_adjacent(suspicious)

    def _merge_adjacent(self, candidates, gap_threshold=2.0):
        if len(candidates) <= 1:
            return candidates

        candidates.sort(key=lambda c: c['t_start'])
        merged = [candidates[0]]

        for cand in candidates[1:]:
            last = merged[-1]
            if cand['t_start'] - last['t_end'] <= gap_threshold:
                merged[-1] = {
                    't_start': last['t_start'],
                    't_end': cand['t_end'],
                    'frame_indices': last['frame_indices'] + cand['frame_indices'],
                    'coarse_score': max(last['coarse_score'], cand['coarse_score']),
                    'description': last['description'],
                }
            else:
                merged.append(cand)

        return merged
