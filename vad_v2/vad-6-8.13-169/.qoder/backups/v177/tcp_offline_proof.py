"""v177 TCP offline prompt proof (no VLM calls):
1. new scan_segment with tcp_prompt=False produces a prompt BYTE-IDENTICAL
   to the committed backup's scan_segment (same stub inputs).
2. new scan_segment with tcp_prompt=True differs ONLY by the pre-registered
   TCP block (+ trailing blank line) inserted immediately BEFORE the
   "Respond ONLY in JSON format" line.
3. [TCP-PROMPT] full-prompt log fires exactly once per video when ON and
   never when OFF; scan_segments_count accounting works.
"""
import asyncio
import importlib.util
import sys

import numpy as np

sys.path.insert(0, '/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
import anomaly_discovery as ad_new


def load_committed():
    spec = importlib.util.spec_from_file_location(
        'ad_committed',
        '/Users/zzl/PycharmProjects/vad/.qoder/backups/v177/anomaly_discovery.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ad_committed = load_committed()


class StubLLM:
    def __init__(self):
        self.prompts = []

    async def generate(self, prompt, images=None):
        self.prompts.append(prompt)
        return '{"is_anomalous": false, "confidence": 0.1, "description": null}'


class RecLog:
    def __init__(self):
        self.infos = []

    def info(self, msg, *a, **k):
        self.infos.append(str(msg))

    def debug(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def make_seg():
    return {'t_start': 4.5, 't_end': 10.5,
            'frame_indices': np.arange(0, 12)}


def frames():
    return np.zeros((12, 8, 8, 3), dtype=np.uint8)


def run_one(mod, tcp, ctx='This is a RoadAccidents video.'):
    llm = StubLLM()
    kw = dict(llm_client=llm)
    if 'tcp_prompt' in mod.AnomalyDiscovery.__init__.__code__.co_varnames:
        kw['tcp_prompt'] = tcp
    d = mod.AnomalyDiscovery(**kw)
    seg = make_seg()
    asyncio.run(d.scan_segment(seg, frames(), ctx))
    return llm.prompts[0], d


p_committed, _ = run_one(ad_committed, False)
p_off, d_off = run_one(ad_new, False)
p_on, d_on = run_one(ad_new, True)

print('1) flag-OFF prompt == committed backup prompt (byte-identical):',
      p_off == p_committed)

marker = 'Respond ONLY in JSON format'
expected_on = p_committed.replace(
    marker, ad_new.TCP_PROMPT_BLOCK + '\n\n' + marker)
print('2) flag-ON prompt == committed + TCP block before JSON line:',
      p_on == expected_on)
print('   block position check (immediately before Respond line):',
      ('counts fully.\n\n' + marker) in p_on)
print('   context_prompt unchanged in ON prompt:',
      'This is a RoadAccidents video.' in p_on)
print('   JSON schema line unchanged in ON prompt:',
      '"is_anomalous": true or false' in p_on)
print('   "Temporal reasoning instructions" in ON prompt:',
      'Temporal reasoning instructions' in p_on)
print('   "Temporal reasoning instructions" in OFF prompt:',
      'Temporal reasoning instructions' in p_off)

# 3) [TCP-PROMPT] once-per-video log behavior + counter
rec = RecLog()
old_log = ad_new._log
ad_new._log = rec
try:
    llm = StubLLM()
    d = ad_new.AnomalyDiscovery(llm_client=llm, tcp_prompt=True)
    d.scan_segments_count = 0
    segs = [dict(make_seg(), **{'t_start': float(i * 6),
                                't_end': float(i * 6 + 6)}) for i in range(3)]

    async def _three():
        await asyncio.gather(*[d.scan_segment(s, frames(), '') for s in segs])

    asyncio.run(_three())
    n_tcp_prompt_logs = sum('[TCP-PROMPT]' in m for m in rec.infos)
    print('3a) ON: [TCP-PROMPT] log count for 3 gathered scans (expect 1):',
          n_tcp_prompt_logs)
    print('3b) ON: scan_segments_count after 3 scans (expect 3):',
          d.scan_segments_count)
    tcp_prompt_shows_block = any(
        '[TCP-PROMPT]' in m and 'Temporal reasoning instructions' in m
        and 'Respond ONLY in JSON format' in m for m in rec.infos)
    print('3c) ON: [TCP-PROMPT] shows TCP block inside actual prompt:',
          tcp_prompt_shows_block)

    rec2 = RecLog()
    ad_new._log = rec2
    llm2 = StubLLM()
    d2 = ad_new.AnomalyDiscovery(llm_client=llm2, tcp_prompt=False)
    d2.scan_segments_count = 0
    asyncio.run(d2.scan_segment(make_seg(), frames(), ''))
    print('3d) OFF: [TCP-PROMPT] log count (expect 0):',
          sum('[TCP-PROMPT]' in m for m in rec2.infos))
finally:
    ad_new._log = old_log
