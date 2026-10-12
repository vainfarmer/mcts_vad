"""v177 TCP smoke-OFF bit-identity proof (b):
The OFF run logs no prompts (committed behavior — zero [TCP-PROMPT] lines),
so the bit-identity diff is done against the COMMITTED BACKUP module:
  1. extract the actual scan prompt from the ON run's [TCP-PROMPT] dump
     (Explosion002, segment 4.5s-10.5s — the same video/segment);
  2. strip the exact TCP block (+ its trailing blank line) -> the OFF-arm
     prompt as reconstructed from the PRODUCED LOG;
  3. generate the prompt with the committed backup anomaly_discovery.py
     (stub LLM, same segment, same context_prompt parsed out of the log);
  4. assert byte-identity; also assert ON == committed + block (position).
"""
import asyncio
import importlib.util
import sys

import numpy as np

sys.path.insert(0, '/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
from anomaly_discovery import TCP_PROMPT_BLOCK

ONLOG = '/Users/zzl/PycharmProjects/vad/results/v177_selftest_tcpON.out'

lines = open(ONLOG).read().splitlines()
start = None
for i, ln in enumerate(lines):
    if '[TCP-PROMPT]' in ln:
        start = i + 1
        break
assert start is not None, 'no [TCP-PROMPT] in ON log'
body = []
for ln in lines[start:]:
    if ln.startswith('2026-'):  # next timestamped log line ends the dump
        break
    body.append(ln)
prompt_on = '\n'.join(body)

print('A) ON-log prompt contains TCP block:',
      'Temporal reasoning instructions' in prompt_on)
print('   ON-log prompt segment line:', body[0])

stripped = prompt_on.replace(TCP_PROMPT_BLOCK + '\n\n', '')
print('B) strip removed exactly one block:',
      prompt_on.count(TCP_PROMPT_BLOCK) == 1 and
      'Temporal reasoning instructions' not in stripped)

# parse context_prompt out of the actual prompt (between line 1 and \n\nTask:)
first_nl = prompt_on.index('\n')
task_at = prompt_on.index('\n\nTask:')
context_prompt = prompt_on[first_nl + 1:task_at]

spec = importlib.util.spec_from_file_location(
    'ad_committed',
    '/Users/zzl/PycharmProjects/vad/.qoder/backups/v177/anomaly_discovery.py')
ad_committed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ad_committed)


class StubLLM:
    def __init__(self):
        self.prompts = []

    async def generate(self, prompt, images=None):
        self.prompts.append(prompt)
        return '{"is_anomalous": false, "confidence": 0.1, "description": null}'


llm = StubLLM()
d = ad_committed.AnomalyDiscovery(llm_client=llm)
seg = {'t_start': 4.5, 't_end': 10.5, 'frame_indices': np.arange(0, 12)}
frames = np.zeros((12, 8, 8, 3), dtype=np.uint8)
asyncio.run(d.scan_segment(seg, frames, context_prompt))
prompt_committed = llm.prompts[0]

print('C) OFF-arm prompt (ON-log minus TCP block) == committed backup '
      'prompt, same video/segment/context (bit-identical):',
      stripped == prompt_committed)

marker = 'Respond ONLY in JSON format'
print('D) ON-log prompt == committed + TCP block immediately before '
      '"Respond ONLY in JSON format":',
      prompt_on == prompt_committed.replace(
          marker, TCP_PROMPT_BLOCK + '\n\n' + marker))

if stripped == prompt_committed:
    import difflib
    diff = list(difflib.unified_diff(
        prompt_committed.splitlines(), prompt_on.splitlines(),
        'committed(OFF)', 'tcpON', lineterm=''))
    print('E) unified diff committed -> ON (only the +TCP block lines):')
    for dl in diff:
        print('   ' + dl)
