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


# v177 TCP (Temporal-Contrast Prompt): the EXACT pre-registered block that
# scan_segment appends immediately BEFORE the "Respond ONLY in JSON format"
# line when DiscoveryConfig.tcp_prompt is ON. Prompt-side ONLY — same frames,
# same call count, same directed path, same JSON response schema
# (is_anomalous/confidence/description); ONLY prompt text changes (zero
# latency cost; NOT an abs-conf gate / merge / batch / scan-density family).
# SINGLE SOURCE OF TRUTH: pipeline.py's [TCP-BLOCK] marker imports this
# constant so the logged block is bit-identical to the appended one (FM5/FM21).
TCP_PROMPT_BLOCK = """Temporal reasoning instructions:
- The frames are shown in time order and span the whole segment uniformly.
- A real anomaly may be BRIEF (1-3 seconds) and occupy only a few frames in the middle of the clip. Do NOT average the clip: judge by the single most anomalous moment you can find.
- Compare the early frames against the later frames: note any sudden change (impact, fall, crowd scatter, smoke or flame onset, concealment motion). A change that starts and ends inside the clip still counts fully."""


class AnomalyDiscovery:

    def __init__(self, llm_client, anomaly_categories=None, image_processor=None,
                 clip_expert=None, clip_threshold=0.3, frame_sampler=None,
                 min_pass_rate=0.4, max_scan_segments=6,
                 comparative_batch_segments=8, comparative_keyframes=2,
                 comparative_long_segments=60, comparative_long_batch=16,
                 kfd_uniform_keyframes=False, kfd_k=2,
                 tcp_prompt=False):
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
        # v173 KFD-U (default OFF = byte-identical first-k truncation): when ON,
        # _comparative_call replaces the keyframes[:k] truncation with a
        # deterministic uniform spread over the SAME sampled pool (see
        # _kfd_spread_select). kfd_k: 2 = {t, t+3.0s} (pre-registered),
        # 3 = {t, t+2.5s, t+5.0s} (fallback), 1 = segment midpoint.
        self.kfd_uniform_keyframes = kfd_uniform_keyframes
        self.kfd_k = kfd_k
        # v177 TCP (default OFF = prompt bit-identical to committed): when ON,
        # scan_segment appends TCP_PROMPT_BLOCK immediately BEFORE the
        # "Respond ONLY in JSON format" line. scan_segments_count is the
        # per-video scan accounting reported by [TCP-VID] (reset by the
        # pipeline before Stage 4 each video; internal instrumentation only —
        # never read when the flag is off, zero observable effect).
        self.tcp_prompt = tcp_prompt
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

    # ================================================================
    # v173 KFD-U: Uniform-spread keyframe selection (pre-registered)
    # ================================================================

    def _sample_keyframes_with_positions(self, segment, all_frames,
                                         precomputed_features=None):
        """v173 KFD-U: the SAME sampled pool _sample_keyframes returns, plus
        the segment-local integer position of each pooled frame (ascending,
        time-ordered) so the spread can look up exact timestamps. Only called
        when kfd_uniform_keyframes is ON."""
        duration = segment['t_end'] - segment['t_start']
        frames = all_frames[segment['frame_indices']]
        if self.frame_sampler is not None:
            return self.frame_sampler.select_keyframes(
                frames, duration=duration,
                frame_indices=segment['frame_indices'],
                precomputed_features=precomputed_features,
                return_positions=True,
            )
        n = min(4, len(frames))
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        return frames[indices], indices

    def _kfd_pool_times(self, segment, pool_pos, timestamps):
        """v173 KFD-U: exact timestamp of each pooled frame. pool_pos indexes
        into segment['frame_indices'] (which indexes all_frames/timestamps).
        Falls back to a deterministic linear estimate over the segment bounds
        when the timestamps array was not threaded through (never happens on
        the FCCS path, which always passes video_data['timestamps'])."""
        fi = segment['frame_indices']
        if timestamps is not None:
            return [float(timestamps[fi[int(p)]]) for p in pool_pos]
        n = len(fi)
        t0 = float(segment['t_start'])
        t1 = float(segment['t_end'])
        if n <= 1:
            return [t0 for _ in pool_pos]
        return [t0 + (t1 - t0) * (int(p) / (n - 1)) for p in pool_pos]

    def _kfd_spread_select(self, segment, pool, pool_pos,
                           keyframes_per_segment, timestamps):
        """v173 KFD-U deterministic uniform spread — REPLACES the first-k
        truncation (keyframes[:k]) on the SAME sampled pool.

        Pre-registered placements (no substitution; the offline ablation
        ordered {t,t+3.0s} 43/44 > {t+2,t+4} 41/44 > {t+1.5,t+4.5} 40/44 >
        {t+1,t+5} 38/44 > linspace-to-edges 37/44):
          * keyframes_per_segment == 1 (the >60-segment long-video rule):
            the single frame nearest the segment MIDPOINT;
          * k_eff == 2: the two frames whose timestamps are nearest
            {t_start, t_start + 3.0s};
          * k_eff == 3 (fallback arm, --kfd_k 3): nearest
            {t_start, t_start+2.5s, t_start+5.0s}.
        k_eff = 1 when the long-video rule is in force (1 frame/segment),
        else self.kfd_k. Deterministic (np.argmin first-minimum on ties),
        index-safe (picks deduped + clamped inside the pool), order preserved
        by timestamp (the pool is time-ordered; picks are sorted ascending).

        Returns (keyframes, old_positions, new_positions): old_positions is
        what the committed first-k truncation would have selected on the same
        pool (for the [KFD-FIRE] old->new record). When the pool fits within
        the budget there is nothing to spread: returns the whole pool and
        old == new (matches the committed `len(keyframes) <= k` keep-all
        branch exactly).
        """
        kps = int(keyframes_per_segment)
        n = len(pool)
        pool_pos = [int(p) for p in pool_pos]
        # Committed selection on this pool (first-k truncation semantics).
        old_pos = pool_pos[:kps] if n > kps else list(pool_pos)
        k_eff = 1 if kps <= 1 else max(1, int(self.kfd_k))
        if n <= k_eff:
            # Nothing to spread: keep the whole pool. Note old_pos still
            # records what the committed first-k truncation WOULD have kept
            # (pool_pos[:kps] when n > kps), so a kfd_k > kps widening from
            # the truncated set to the full pool is correctly reported as a
            # change; keyframes and new_positions stay consistent (both the
            # full pool).
            return pool, old_pos, list(pool_pos)
        times = self._kfd_pool_times(segment, pool_pos, timestamps)
        t0 = float(segment['t_start'])
        t1 = float(segment['t_end'])
        if k_eff == 1:
            targets = [(t0 + t1) / 2.0]
        elif k_eff == 3:
            targets = [t0, t0 + 2.5, t0 + 5.0]
        else:  # k_eff == 2 (pre-registered default)
            targets = [t0, t0 + 3.0]
        picks = []
        for tt in targets:
            j = int(np.argmin([abs(x - tt) for x in times]))
            if j not in picks:
                picks.append(j)
        picks.sort()
        new_pos = [pool_pos[j] for j in picks]
        return pool[picks], old_pos, new_pos

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
                                precomputed_features=None, kfd_ctx=None):
        """One comparative VLM call over a batch of segments.

        Returns (results, ok): results is the per-segment parsed list (same
        schema as scan_segment); ok is False when the response was unparseable
        or returned fewer entries than segments (v97 truncation guard — the
        caller decides whether to retry/fall back, instead of silently
        zero-scoring the whole batch).

        v173 KFD-U: when self.kfd_uniform_keyframes is ON, the first-k
        truncation below is replaced by the deterministic uniform spread
        (_kfd_spread_select) over the SAME sampled pool — same batch sizes,
        same frames/call budget (k_eff = keyframes_per_segment under the
        pre-registered default kfd_k=2), same prompt, same call count. When
        the spread changed the selected indices vs the committed first-k
        truncation on at least one segment, exactly ONE [KFD-FIRE] line is
        emitted per batch carrying vid + segment indices + old->new position
        and timestamp lists (FM21). kfd_ctx (built by scan_full_coverage)
        carries {'vid', 'timestamps', 'stats'}; stats['fires'] /
        stats['segments_changed'] are incremented here (synchronously, before
        any await) so per-video counts are exact under batch concurrency.
        Flag OFF: the original first-k truncation lines run UNTOUCHED and no
        marker is emitted (byte-identical committed behavior).
        """
        n = len(segments)
        labels = []
        packed_frames = []
        frame_ranges = []
        kfd_changed = []
        for i, seg in enumerate(segments):
            if self.kfd_uniform_keyframes:
                # v173 KFD-U: same pool, deterministic uniform spread instead
                # of the first-k truncation.
                pool, pool_pos = self._sample_keyframes_with_positions(
                    seg, all_frames, precomputed_features)
                keyframes, old_pos, new_pos = self._kfd_spread_select(
                    seg, pool, pool_pos, keyframes_per_segment,
                    (kfd_ctx or {}).get('timestamps'))
                if new_pos != old_pos:
                    fi = seg['frame_indices']
                    ts = (kfd_ctx or {}).get('timestamps')

                    def _glob(pos):
                        return [int(fi[p]) for p in pos]

                    def _times(pos):
                        if ts is None:
                            return None
                        return [round(float(ts[fi[p]]), 2) for p in pos]

                    kfd_changed.append((i, seg, old_pos, new_pos,
                                        _glob(old_pos), _glob(new_pos),
                                        _times(old_pos), _times(new_pos)))
            else:
                keyframes = self._sample_keyframes(seg, all_frames, precomputed_features)
                keyframes = (keyframes[:keyframes_per_segment]
                             if len(keyframes) > keyframes_per_segment else keyframes)
            start_idx = len(packed_frames)
            packed_frames.extend(keyframes)
            frame_ranges.append((start_idx, len(packed_frames)))
            labels.append(f"Segment {i+1} ({seg['t_start']:.1f}s-{seg['t_end']:.1f}s): "
                          f"images {start_idx+1}-{len(packed_frames)}")

        if kfd_changed:
            # FM5/FM21: exactly ONE [KFD-FIRE] line per batch where the spread
            # changed the selected indices vs the first-k truncation.
            stats = (kfd_ctx or {}).get('stats')
            if stats is not None:
                stats['fires'] += 1
                stats['segments_changed'] += len(kfd_changed)
            vid = (kfd_ctx or {}).get('vid', 'unknown')
            k_eff = 1 if int(keyframes_per_segment) <= 1 else max(1, int(self.kfd_k))
            parts = []
            for (i, seg, old_pos, new_pos, old_g, new_g, old_t, new_t) in kfd_changed:
                piece = (f"seg{i}@{seg['t_start']:.1f}-{seg['t_end']:.1f}s "
                         f"pos {old_pos}->{new_pos} idx {old_g}->{new_g}")
                if old_t is not None:
                    piece += f" t {old_t}->{new_t}"
                parts.append(piece)
            _log.info(f"  [KFD-FIRE] vid={vid} kps={int(keyframes_per_segment)} "
                      f"k_eff={k_eff} nseg={n} changed={len(kfd_changed)} "
                      + "; ".join(parts))

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

        # v177 TCP: per-video scan accounting for [TCP-VID] (the pipeline
        # resets this before Stage 4 each video; internal instrumentation
        # only — never read when the flag is off, zero observable effect).
        self.scan_segments_count += 1
        _tcp_first_this_video = self.tcp_prompt and self.scan_segments_count == 1

        # v177 TCP: when ON, the pre-registered TCP_PROMPT_BLOCK (+ a blank
        # line, matching the prompt's paragraph style) is appended immediately
        # BEFORE the "Respond ONLY in JSON format" line; when OFF the
        # placeholder expands to "" so the prompt is BYTE-IDENTICAL to the
        # committed prompt. Same keyframes, same schema, same parser — ONLY
        # prompt text changes (zero latency cost).
        tcp_block = (TCP_PROMPT_BLOCK + "\n\n") if self.tcp_prompt else ""
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

{tcp_block}Respond ONLY in JSON format (no other text):
{{"is_anomalous": true or false, "confidence": 0.0 to 1.0, "description": "what abnormal activity you see, or null if normal"}}"""

        if _tcp_first_this_video:
            # v177 TCP FM5/FM21: log the FIRST scanned segment's FULL assembled
            # prompt once per video — the actual string sent to the VLM — so
            # the produced log shows the TCP block inside a real scan prompt
            # (bounded: exactly one prompt dump per video). Under
            # asyncio.gather the synchronous preambles run in task-creation
            # order, so scan_segments_count == 1 fires exactly once per video.
            _log.info(f"  [TCP-PROMPT] first scan_segment prompt this video "
                      f"(segment {segment['t_start']:.1f}s-{segment['t_end']:.1f}s) "
                      f"follows verbatim:\n{prompt}")

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
                                              precomputed_features,
                                              kfd_ctx=None):
        """One comparative batch with the v97 truncation guard.

        Failure handling only (not a second mechanism): if the batch response
        is unparseable or missing entries, split the batch in half and retry
        once; if a half still fails, rescan just those segments individually
        via the existing independent scan_segment() path. Returns a result
        list aligned 1:1 with batch_segments (order preserved).
        """
        results, ok = await self._comparative_call(
            batch_segments, all_frames, context_prompt, category_hint,
            keyframes_per_segment, precomputed_features, kfd_ctx)
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
                keyframes_per_segment, precomputed_features, kfd_ctx)
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

        v173 KFD-U: returns (candidates, kfd_stats). kfd_stats is None unless
        kfd_uniform_keyframes is ON, in which case it carries
        {'segments', 'batches', 'fires', 'segments_changed'} for the
        unconditional per-video [KFD-VID] marker the pipeline emits (FM17).
        The single call site (pipeline Stage 4) unpacks the tuple.
        """
        segments = video_data['segments']
        all_frames = video_data['all_frames']
        precomputed = video_data.get('clip_features')
        n = len(segments)
        if n == 0:
            return [], None

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

        # v173 KFD-U: per-video context for the spread + markers. Built once
        # per scan_full_coverage call and passed down the batch chain, so
        # per-video [KFD-FIRE]/stats accounting stays exact even when the
        # runner processes videos concurrently on a shared discovery instance.
        kfd_ctx = None
        if self.kfd_uniform_keyframes:
            import os as _kfd_os
            _vp = _kfd_os.path.basename(str(video_data.get('video_path') or ''))
            kfd_ctx = {
                'vid': _kfd_os.path.splitext(_vp)[0] or 'unknown',
                'timestamps': video_data.get('timestamps'),
                'stats': {'segments': n, 'batches': len(ranges),
                          'fires': 0, 'segments_changed': 0},
            }

        tasks = [self._scan_comparative_batch_guarded(
                     segments[i:j], all_frames, context_prompt, category_hint,
                     keyframes, precomputed, kfd_ctx)
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
        return (self._merge_adjacent(suspicious),
                kfd_ctx['stats'] if kfd_ctx is not None else None)

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
