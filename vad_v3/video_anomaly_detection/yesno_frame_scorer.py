"""Production adapter for the frozen v218 visual likelihood scorer.

Localization finishes first. Every video then receives the same category-blind
visual pass, including videos with no detected segments. Fusion is performed
at native frame resolution, exactly as in the v218 experiment.
"""
import asyncio
import base64
import hashlib
import math

import cv2
import numpy as np

try:
    from .yesno_scoring import PROMPT, extract_score, windows
except ImportError:
    from yesno_scoring import PROMPT, extract_score, windows


SETTINGS = dict(temperature=0.0, max_tokens=16, logprobs=True, top_logprobs=5,
                enable_thinking=False, stride_seconds=5.0, context_seconds=10.0,
                frames_per_clip=10, jpeg_quality=80, max_censor_width=0.001)


def sample_content(cap, row, win, settings=SETTINGS):
    """Keep OpenCV frame selection, JPEG bytes and prompt identical to v218."""
    positions = np.linspace(win['context_start'], win['context_end'], settings['frames_per_clip'])
    indices = np.unique(np.clip(np.rint(positions*row['fps']).astype(int), 0, row['frame_count']-1))
    content = []
    digest = hashlib.sha256()
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Source frame could not be read')
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, settings['jpeg_quality']])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        data = encoded.tobytes()
        digest.update(data)
        content.append({'type': 'text', 'text': f'Frame time {index/row["fps"]:.3f} seconds.'})
        content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'+base64.b64encode(data).decode()}})
    prompt = PROMPT+f'\nTarget interval: {win["start"]:.3f} to {win["end"]:.3f} seconds.'
    content.append({'type': 'text', 'text': prompt})
    digest.update(prompt.encode())
    return content, digest.hexdigest(), indices.tolist()


def native_curve(scores, timestamps, row):
    if scores is None or not len(scores):
        raise ValueError('Yes/No fusion requires the original frame scores')
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Frame scores must be a finite one-dimensional curve')
    if timestamps is None:
        positions = np.linspace(0, row['frame_count']-1, len(values))
    else:
        positions = np.asarray(timestamps, dtype=float)*row['fps']
        if positions.ndim != 1 or len(positions) != len(values) or not np.isfinite(positions).all() or np.any(np.diff(positions) <= 0):
            raise ValueError('Score timestamps must be finite, increasing and aligned')
    return np.interp(np.arange(row['frame_count']), positions, values).astype(np.float32)


class YesNoFrameScorer:
    def __init__(self, client, config):
        config.validate_yesno()
        if not callable(getattr(client, 'score_yesno', None)):
            raise TypeError('Yes/No scoring requires a client supporting actual token log probabilities')
        self.client = client
        self.model = config.yesno_model
        self.weight = config.yesno_weight

    async def score_video(self, video_path):
        # Decoder and all mutable state belong to this video, not the shared pipeline.
        cap = cv2.VideoCapture(str(video_path))
        pending = set()
        sample_task = None
        try:
            if not cap.isOpened():
                raise RuntimeError('Cannot open video for Yes/No scoring')
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
                raise ValueError('Video has invalid native FPS or frame count')
            row = dict(fps=fps, frame_count=frame_count, duration=frame_count/fps)
            clips = windows(row['duration'], SETTINGS['stride_seconds'], SETTINGS['context_seconds'])
            items = []

            async def request(win, content, digest, indices):
                raw = await self.client.score_yesno(content, model=self.model)
                score = extract_score(raw, SETTINGS['max_censor_width'])
                return dict(window=win, request_sha256=digest, frame_indices=indices,
                            score=score, response=raw)

            for win in clips:
                sample_task = asyncio.create_task(asyncio.to_thread(sample_content, cap, row, win))
                content, digest, indices = await asyncio.shield(sample_task)
                pending.add(asyncio.create_task(request(win, content, digest, indices)))
                if len(pending) >= 4:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    items.extend(await asyncio.gather(*done))
            if pending:
                items.extend(await asyncio.gather(*pending))
            items.sort(key=lambda item: item['window']['index'])
            if len(items) != len(clips):
                raise RuntimeError('Incomplete Yes/No window coverage')
            return dict(source=row, settings=dict(SETTINGS, model=self.model, fusion_weight=self.weight),
                        prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
                        timestamps=[item['window']['center'] for item in items],
                        frame_scores=[item['score']['score'] for item in items],
                        exact_pairs=sum(item['score']['kind'] == 'exact_pair' for item in items),
                        bounded_pairs=sum(item['score']['kind'] != 'exact_pair' for item in items),
                        windows=items)
        finally:
            for task in pending:
                if not task.done():
                    task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            # A cancelled coroutine must not close a decoder still used by a thread.
            if sample_task is not None and not sample_task.done():
                await asyncio.gather(sample_task, return_exceptions=True)
            cap.release()

    def apply_scores(self, result, visual):
        """Apply one visual pass to the main result and each paired baseline."""
        row = visual['source']
        probability = native_curve(visual['frame_scores'], visual['timestamps'], row)

        def attach(target, main=False):
            get = target.get if isinstance(target, dict) else lambda key: getattr(target, key)
            baseline_scores, baseline_times = get('frame_scores'), get('timestamps')
            if self.weight == 1:
                fused = probability
            else:
                baseline = native_curve(baseline_scores, baseline_times, row)
                fused = ((1-self.weight)*baseline+self.weight*probability).astype(np.float32)
            metadata = dict(method='v218_yesno_fusion',
                            baseline_frame_scores=baseline_scores, baseline_timestamps=baseline_times,
                            output_grid='native_frames', settings=visual['settings'])
            if main:
                metadata.update(visual)
            else:
                metadata['visual_scores_reference'] = 'parent.yesno_scoring'
            fields = dict(frame_scores=fused.tolist(),
                          timestamps=(np.arange(row['frame_count'])/row['fps']).tolist(),
                          frame_score_grid='native_frames',
                          source_frame_count=row['frame_count'], source_fps=row['fps'],
                          yesno_scoring=metadata)
            if isinstance(target, dict):
                target.update(fields)
            else:
                for key, value in fields.items():
                    setattr(target, key, value)

        # Paired arms use their own original curves and the same visual evidence.
        for branch in (result.paired_results or {}).values():
            attach(branch)
        attach(result, main=True)
        return result

    async def apply(self, result, video_path):
        visual = await self.score_video(video_path)
        return self.apply_scores(result, visual)
