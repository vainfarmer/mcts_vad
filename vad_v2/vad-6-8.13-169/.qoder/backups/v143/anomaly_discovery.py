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
                 min_pass_rate=0.4, max_scan_segments=6,
                 comparative_batch_segments=8, comparative_keyframes=2,
                 comparative_long_segments=60, comparative_long_batch=16):
        self.llm = llm_client
        self.categories = anomaly_categories or []
        self.image_processor = image_processor
        self.clip_expert = clip_expert
        self.clip_threshold = clip_threshold
        self.frame_sampler = frame_sampler
        self.min_pass_rate = min_pass_rate
        self.max_scan_segments = max_scan_segments
        # v97 FCCS batching knobs (only used when the pipeline enables
        # full_coverage_comparative_scan; defaults match the v97 proposal)
        self.comparative_batch_segments = comparative_batch_segments
        self.comparative_keyframes = comparative_keyframes
        self.comparative_long_segments = comparative_long_segments
        self.comparative_long_batch = comparative_long_batch

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
                                         precomputed_features=None, category_hint="",
                                         keyframes_per_segment=2):
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

        v97: optional category_hint (same CATEGORY_CUES text the directed scan
        uses) and keyframes_per_segment (FCCS long-video rule drops to 1).
        """
        results, _ok = await self._comparative_call(
            segments, all_frames, context_prompt, category_hint,
            keyframes_per_segment, precomputed_features)
        return results

    async def _comparative_call(self, segments, all_frames, context_prompt="",
                                category_hint="", keyframes_per_segment=2,
                                precomputed_features=None):
        """One comparative VLM call over a batch of segments.

        Returns (results, ok): results is the per-segment parsed list (same
        schema as scan_segment); ok is False when the response was unparseable
        or returned fewer entries than segments (v97 truncation guard — the
        caller decides whether to retry/fall back, instead of silently
        zero-scoring the whole batch).
        """
        n = len(segments)
        labels = []
        packed_frames = []
        frame_ranges = []
        for i, seg in enumerate(segments):
            keyframes = self._sample_keyframes(seg, all_frames, precomputed_features)
            keyframes = (keyframes[:keyframes_per_segment]
                         if len(keyframes) > keyframes_per_segment else keyframes)
            start_idx = len(packed_frames)
            packed_frames.extend(keyframes)
            frame_ranges.append((start_idx, len(packed_frames)))
            labels.append(f"Segment {i+1} ({seg['t_start']:.1f}s-{seg['t_end']:.1f}s): "
                          f"images {start_idx+1}-{len(packed_frames)}")

        prompt = f"""You are looking at keyframes from {n} DIFFERENT time segments of the
SAME surveillance video, shown together in this order:
{chr(10).join(labels)}
{context_prompt}
{category_hint}

Task: Rate EVERY ONE of these {n} segments for abnormal or suspicious activity
— violence, crime (robbery/burglary/stealing/shoplifting), vandalism, arson,
explosions, road accidents, or arrests. Even subtle signs count (nervous
behavior, concealing items, someone cowering, sudden running, smoke/fire,
breaking a window or lock). Use the other segments as your reference for what
"normal" looks like in THIS video, so your confidence scores are well
calibrated relative to each other — but rate every segment on its own merit;
do not suppress a segment just because another one looks more dramatic.

Respond ONLY in JSON format (no other text), with exactly {n} entries, one per
segment in order. Keep each description terse (max 8 words, or null) so the
full JSON fits in the output budget:
{{"segment_scores": [{{"segment": <1-based segment number>, "confidence": 0.0 to 1.0, "description": "max 8 words, or null"}}, ...]}}"""

        import numpy as _np
        images = _np.stack(packed_frames) if packed_frames else None
        response = await self.llm.generate(prompt, images)
        return self._parse_comparative_response_checked(response, n)

    def _parse_comparative_response(self, response, n):
        results, _ok = self._parse_comparative_response_checked(response, n)
        return results

    def _parse_comparative_response_checked(self, response, n):
        """Parse a comparative-scan response; returns (results, ok).

        ok=False means the JSON was unparseable or contained fewer than n
        segment entries (e.g. truncated by the max_tokens budget) — v97 FCCS
        treats this as a batch failure to retry/fall back on, instead of
        silently zero-scoring every segment in the batch (recall loss).
        """
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
        entries = data.get('segment_scores', []) or []
        for entry in entries:
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
        ok = isinstance(entries, list) and len(entries) >= n
        return results, ok

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

    async def probe_continuation(self, clip_segment, all_frames, event_description):
        """v141 CEX: BINARY, NUMBER-FREE continuation vote on ONE adjacent clip.

        Asked once per side per round by pipeline._continuation_extension, for a
        single short clip immediately adjacent to an ALREADY-CONFIRMED final
        detection. The only question is whether the SAME event is still ongoing
        in this clip; the caller (code) imposes all window geometry in fixed
        steps, so this prompt deliberately:
          - asks for NO timestamps / times / durations (the v62 failure mode was
            VLM absolute-timestamp regression from sparse keyframes, which
            regressed full mAP 0.1492 -> 0.1012);
          - does NOT reuse the scan_segment prompt or its is_anomalous field
            (that would re-open segment INCLUSION, the v92 self-consistency
            failure / FM4 risk) — CEX never revisits candidate membership;
          - is a SINGLE independent call over ONE clip: nothing merged,
            batched, contact-sheeted, and no other verdict is shown to it
            (FM3/FM12; v97 FCCS / v126 TRLM / v129 BDS all cratered mAP by
            breaking per-call independence).

        Returns {'ongoing': bool, 'confidence': float}; a parse/API failure
        returns ongoing=False so the caller closes that side (fail-closed: an
        unparseable probe can never grow a window).
        """
        frames = all_frames[clip_segment['frame_indices']]
        if len(frames) == 0:
            return {'ongoing': False, 'confidence': 0.0, 'ok': False}
        n = min(4, len(frames))
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        keyframes = frames[indices]

        desc = (event_description or '').strip() or 'the confirmed anomalous event'

        prompt = f"""These frames all come from ONE short clip of a surveillance video.

An event has ALREADY been confirmed in the immediately adjacent part of this same video. That event is described as:
"{desc}"

Task: decide whether that SAME event is STILL ONGOING inside THIS clip.

Answer true only if the same event is actively happening in these frames.
Answer false if this clip shows only ordinary/normal activity, only the calm
aftermath or the calm build-up, an empty scene, or a different event.

Do NOT report any time, timestamp, duration or frame number. Answer only the
yes/no question above.

Respond ONLY in JSON format (no other text):
{{"ongoing": true or false, "confidence": 0.0 to 1.0}}"""

        try:
            response = await self.llm.generate(prompt, keyframes)
        except Exception as e:  # network/API failure -> fail-closed
            _log.warning(f"    [CEX-PROBE] call failed: {e}")
            return {'ongoing': False, 'confidence': 0.0, 'ok': False}
        return self._parse_continuation(response)

    def _parse_continuation(self, response):
        """v141 CEX: strict parse of the binary continuation vote (fail-closed)."""
        try:
            text = (response or '').strip()
            start = text.find('{')
            end = text.rfind('}') + 1
            data = json.loads(text[start:end]) if (start >= 0 and end > start) else {}
        except (json.JSONDecodeError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        ongoing = data.get('ongoing', False)
        if isinstance(ongoing, str):
            ongoing = ongoing.strip().lower() in ('true', 'yes', '1')
        try:
            conf = float(data.get('confidence', 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = min(1.0, max(0.0, conf))
        return {'ongoing': bool(ongoing), 'confidence': conf, 'ok': True}

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

    # ================================================================
    # v97 FCCS: Full-Coverage Comparative Scanning
    # ================================================================

    def _comparative_batch_ranges(self, n, batch_size):
        """Contiguous index ranges [i, j) over n segments, ~batch_size each.

        Review mandate (v97): avoid tiny final batches — when n % batch_size
        is 1 or 2, fold the remainder by redistributing into floor(n/batch_size)
        near-equal batches (e.g. 26 segments at batch 8 -> 9/9/8, not
        8/8/8/2) so every comparative batch retains enough in-video
        calibration context.
        """
        if n <= 0:
            return []
        if n <= batch_size:
            return [(0, n)]
        rem = n % batch_size
        if 0 < rem < 3:
            n_batches = n // batch_size
            base, extra = divmod(n, n_batches)
            sizes = [base + 1] * extra + [base] * (n_batches - extra)
        else:
            sizes = [batch_size] * (n // batch_size) + ([rem] if rem else [])
        ranges = []
        i = 0
        for s in sizes:
            ranges.append((i, i + s))
            i += s
        return ranges

    async def _scan_comparative_batch_guarded(self, batch_segments, all_frames,
                                              context_prompt, category_hint,
                                              keyframes_per_segment,
                                              precomputed_features):
        """One comparative batch with the v97 truncation guard.

        Failure handling only (not a second mechanism): if the batch response
        is unparseable or missing entries, split the batch in half and retry
        once; if a half still fails, rescan just those segments individually
        via the existing independent scan_segment() path. Returns a result
        list aligned 1:1 with batch_segments (order preserved).
        """
        results, ok = await self._comparative_call(
            batch_segments, all_frames, context_prompt, category_hint,
            keyframes_per_segment, precomputed_features)
        if ok:
            return results

        _log.warning(f"  [FCCS] Comparative batch of {len(batch_segments)} segments "
                     f"unparseable/truncated; splitting in half and retrying once")
        mid = max(1, len(batch_segments) // 2)
        out = []
        for half in (batch_segments[:mid], batch_segments[mid:]):
            if not half:
                continue
            r2, ok2 = await self._comparative_call(
                half, all_frames, context_prompt, category_hint,
                keyframes_per_segment, precomputed_features)
            if ok2:
                out.extend(r2)
            else:
                _log.warning(f"  [FCCS] Retry still failing for {len(half)} segments; "
                             f"falling back to independent per-segment scan")
                tasks = [self.scan_segment(seg, all_frames,
                                           context_prompt + category_hint,
                                           precomputed_features)
                         for seg in half]
                out.extend(await asyncio.gather(*tasks))
        return out

    async def scan_full_coverage(self, video_data, context_prompt="", category_hint=""):
        """v97 FCCS: scan ALL segments of the video with batched comparative
        VLM calls over CONTIGUOUS time ranges.

        Replaces both partial-coverage paths (directed scan: 10-18 of 23-120
        segments; fallback pre-screen: ~6) with 100% coverage: every segment
        is judged, and judged RELATIVE to its temporal neighbors (the correct
        in-video normal baseline) rather than by an uncalibrated absolute
        threshold. Batching keeps this latency-feasible (~4 calls/video vs
        26-50 for full-coverage independent judging): batch=8 segments x 2
        keyframes; for >comparative_long_segments videos, batch=16 x 1
        keyframe. Downstream (0.3 threshold, _merge_adjacent, candidate
        schema) is identical to the scan_segment path.
        """
        segments = video_data['segments']
        all_frames = video_data['all_frames']
        precomputed = video_data.get('clip_features')
        n = len(segments)
        if n == 0:
            return []

        # Contiguous time order — each batch must be a contiguous time range
        # so segments calibrate against their temporal neighbors.
        segments = sorted(segments, key=lambda s: s['t_start'])

        if n > self.comparative_long_segments:
            batch_size = self.comparative_long_batch
            keyframes = 1
        else:
            batch_size = self.comparative_batch_segments
            keyframes = self.comparative_keyframes

        ranges = self._comparative_batch_ranges(n, batch_size)
        _log.info(f"  [FCCS] Full-coverage comparative scan: {n} segments in "
                  f"{len(ranges)} batch(es) (batch~{batch_size}, "
                  f"{keyframes} keyframe(s)/segment)")

        tasks = [self._scan_comparative_batch_guarded(
                     segments[i:j], all_frames, context_prompt, category_hint,
                     keyframes, precomputed)
                 for (i, j) in ranges]
        batch_results = await asyncio.gather(*tasks)

        suspicious = []
        for (i, j), results in zip(ranges, batch_results):
            for seg, result in zip(segments[i:j], results):
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

        _log.info(f"  [FCCS] {len(suspicious)}/{n} segments flagged before merge")
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
