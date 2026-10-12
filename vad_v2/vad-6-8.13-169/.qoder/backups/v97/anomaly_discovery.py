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
from typing import List, Dict, Any, Optional

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

    async def scan_segments_comparative(self, segments, all_frames, context_prompt="",
                                         precomputed_features=None):
        """v56: Compare multiple candidate segments in ONE VLM call, instead of
        judging each in isolation.

        Root-cause finding (per-video debug trace, Shoplifting001_x264): when the
        category-specific CLIP curve is near-uniform (the common case for
        LOW_CLIP_CATEGORIES — Shoplifting/Stealing/Arson/Burglary — where the
        anomaly looks visually similar to normal activity), candidate selection
        falls back to a GENERIC CLIP pre-screen that is often equally
        uninformative, so the ~6 segments handed to the VLM are close to a random
        sample of the video. The existing scan_segment() prompt then judges each
        segment independently with a "look for even subtle signs" prompt that has
        no way to calibrate absolute suspicion — result: most/all of an arbitrary
        6-segment sample get flagged SUSPICIOUS at 0.90+ regardless of whether
        they contain the true event, because the model has no other segments to
        compare against.

        This mirrors recent video-LLM temporal grounding work (TimeChat,
        VTimeLLM, Momentor): grounding improves when the model reasons over
        broader temporal context rather than classifying isolated clips. Here we
        pack keyframes from ALL candidate segments into a single call, label each
        by index and time range, and ask the model to pick the segment(s) that
        stand out RELATIVE to the others in the SAME video — a comparative/
        relative judgment instead of an absolute per-clip threshold. This also
        cuts VLM calls from up to 6 down to 1 for this fallback path, which helps
        per-video latency (a standing concern alongside mAP).
        """
        n = len(segments)
        labels = []
        packed_frames = []
        frame_ranges = []
        for i, seg in enumerate(segments):
            keyframes = self._sample_keyframes(seg, all_frames, precomputed_features)
            keyframes = keyframes[:2] if len(keyframes) > 2 else keyframes
            start_idx = len(packed_frames)
            packed_frames.extend(keyframes)
            frame_ranges.append((start_idx, len(packed_frames)))
            labels.append(f"Segment {i+1} ({seg['t_start']:.1f}s-{seg['t_end']:.1f}s): "
                          f"images {start_idx+1}-{len(packed_frames)}")

        prompt = f"""You are looking at keyframes from {n} DIFFERENT time segments of the
SAME surveillance video, shown together in this order:
{chr(10).join(labels)}
{context_prompt}

Task: Rate EVERY ONE of these {n} segments for abnormal or suspicious activity
— violence, crime (robbery/burglary/stealing/shoplifting), vandalism, arson,
explosions, road accidents, or arrests. Even subtle signs count (nervous
behavior, concealing items, someone cowering, sudden running, smoke/fire,
breaking a window or lock). Use the other segments as your reference for what
"normal" looks like in THIS video, so your confidence scores are well
calibrated relative to each other — but rate every segment on its own merit;
do not suppress a segment just because another one looks more dramatic.

Respond ONLY in JSON format (no other text), with exactly {n} entries, one per
segment in order:
{{"segment_scores": [{{"segment": <1-based segment number>, "confidence": 0.0 to 1.0, "description": "what you see, or null if normal"}}, ...]}}"""

        import numpy as _np
        images = _np.stack(packed_frames) if packed_frames else None
        response = await self.llm.generate(prompt, images)
        return self._parse_comparative_response(response, n)

    def _parse_comparative_response(self, response, n):
        try:
            text = response.strip()
            start = text.find('{')
            end = text.rfind('}') + 1
            data = json.loads(text[start:end]) if (start >= 0 and end > start) else {}
        except (json.JSONDecodeError, ValueError):
            data = {}

        # v57: score EVERY segment (not just an exclusive "standouts" pick-list —
        # see scan_segments_comparative docstring) then threshold >0.3 downstream,
        # matching the original per-segment scan_segment() cutoff. v56's
        # "only list the ones that clearly stand out" framing measurably
        # suppressed real-but-moderate anomalies relative to other segments in
        # the same comparison (3 videos lost recall vs v51 that this framing
        # would have caught, per v56 evaluation diagnostics).
        results = [{'is_suspicious': False, 'anomaly_score': 0.0, 'description': 'unknown anomaly'}
                   for _ in range(n)]
        for entry in data.get('segment_scores', []) or []:
            try:
                idx = int(entry.get('segment', 0)) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < n:
                conf = float(entry.get('confidence', 0.0))
                results[idx] = {
                    # v59: reverted to 0.3 threshold (V58's 0.5 killed recall
                    # without precision gain — mAP 0.1284->0.1172). Boundary
                    # precision now handled by peak-valley detection instead.
                    'is_suspicious': conf > 0.3,
                    'anomaly_score': conf,
                    'description': entry.get('description') or 'unknown anomaly',
                }
        return results

    async def scan_segment(self, segment, all_frames, context_prompt="",
                           precomputed_features=None):
        keyframes = self._sample_keyframes(segment, all_frames, precomputed_features)

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
