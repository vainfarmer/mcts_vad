"""Opt-in evidence recorder. Observations do not participate in inference."""
from __future__ import annotations
import base64
import contextvars
import hashlib
import json
from pathlib import Path
import threading
import time

active = None
scope = contextvars.ContextVar('vad_research_scope', default={})


def serial(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(type(value).__name__)


class Recorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.sequence = 0
        self.requests = []
        self.stages = {}
        self.frame_times = {}
        self.data = {}
        self.classification = None
        self.observations = {}
        self.curves = []
        self.request_count = 0
        self.max_requests = 1000
        self.exit_reason = None
        self.event('initializing', 'start')

    def save(self, name, value):
        target = self.directory / name
        temp = target.with_suffix(target.suffix + '.tmp')
        temp.write_text(json.dumps(value, ensure_ascii=False, default=serial, allow_nan=False), encoding='utf-8')
        temp.replace(target)

    def event(self, stage, event_type='progress', **payload):
        with self.lock:
            self.sequence += 1
            with (self.directory / 'stage_events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(sequence=self.sequence, stage=stage, type=event_type,
                                             timestamp=time.time(), **payload), ensure_ascii=False,
                                        default=serial) + '\n')

    def preprocessed(self, data):
        self.data = data
        for frame, timestamp in zip(data['all_frames'], data['timestamps']):
            key = hashlib.sha256(frame.tobytes()).hexdigest()
            self.frame_times.setdefault(key, []).append(float(timestamp))
        self.event('preprocessing', 'completed', sampled_frames=len(data['timestamps']))

    def request(self, prompt, selected, content):
        with self.lock:
            self.request_count += 1
            if self.request_count > self.max_requests:
                raise RuntimeError('VLM request budget exceeded')
            rid = f'request-{self.request_count:05d}'
            row = dict(id=rid, scope=scope.get().copy(), prompt=prompt, images=[], started_at=time.time())
            uris = []
            for item in content:
                if item['type'] == 'image_url':
                    uris.append(item['image_url']['url'])
                elif item['type'] == 'video':
                    uris.extend(item['video'])
            evidence = self.directory / 'evidence'
            evidence.mkdir(exist_ok=True)
            for i, uri in enumerate(uris):
                binary = base64.b64decode(uri.split(',', 1)[1])
                relative = f'evidence/{rid}-{i:03d}.jpg'
                (self.directory / relative).write_bytes(binary)
                times = []
                explicit_times = row['scope'].get('evidence_times')
                if selected is not None and i < len(selected):
                    times = self.frame_times.get(hashlib.sha256(selected[i].tobytes()).hexdigest(), [])
                    interval = row['scope'].get('interval')
                    if interval:
                        times = [t for t in times if interval[0] <= t <= interval[1]]
                if explicit_times is not None and len(explicit_times) == len(uris):
                    times = [float(explicit_times[i])]
                row['images'].append(dict(file=relative, sha256=hashlib.sha256(binary).hexdigest(),
                                          source_times=times,
                                          time_status='unique' if len(times) == 1 else 'ambiguous_or_unrecorded'))
            self.requests.append(row)
            self.event(row['scope'].get('stage', 'vlm'), 'request', request_id=rid, image_count=len(uris))
            self.save('research.json',
                      dict(evidence=self.requests, stages=self.stages, classification=self.classification))
            return row

    def response(self, row, response=None, error=None):
        with self.lock:
            row.update(response=response, error=error, finished_at=time.time())
            self.save('research.json',
                      dict(evidence=self.requests, stages=self.stages, classification=self.classification))

    def snapshot(self, name, items):
        with self.lock:
            self.stages[name] = json.loads(json.dumps(items, default=serial))
            self.event(name, 'completed', count=len(items))

    def observe(self, name, value):
        with self.lock:
            self.observations[name] = json.loads(json.dumps(value, default=serial))

    def curve(self, name, timestamps, values, **metadata):
        with self.lock:
            self.curves.append(dict(name=name, timestamps=timestamps, values=values, **metadata))

    def finish(self, result, scorer=None):
        curves = list(self.curves)
        data = self.data
        if data:
            times = data['timestamps']
            motion = data.get('motion_energy')
            if motion is not None and len(motion) == len(times):
                curves.append(dict(name='motion_energy', timestamps=times, values=motion))
            category = result.videomae_category
            if scorer is not None and category:
                curve = scorer.get_category_scores(data, category)
                if curve is not None:
                    curves.append(dict(name='clip_relative:' + category, timestamps=times, values=curve))
            centers = data.get('videomae_window_centers')
            scores = data.get('videomae_window_scores')
            if centers is not None and scores is not None and len(centers):
                import numpy as np
                ts = np.interp(centers, np.arange(len(times)), times)
                curves.append(dict(name='videomae_window_scores', timestamps=ts, values=scores))
        self.save('curves.json', curves)
        self.save('research.json', dict(schema_version=1, evidence=self.requests, stages=self.stages,
                                        stage_order=list(self.stages), classification=self.classification,
                                        observations=self.observations,
                                        exit_reason=self.exit_reason or (
                                            'completed_with_detections' if result.anomalies else 'completed_without_detections')))
        self.event('completed', 'completed', anomaly_count=len(result.anomalies))


def initialize(directory):
    global active
    active = Recorder(directory) if directory else None
    return active
