import argparse
import asyncio
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

from config import ScoringConfig, VADConfig, add_yesno_arguments
from evaluate import compute_frame_auc, load_predictions
from llm_client import DashScopeLLMClient
from llm_usage import track_video_usage
from models import DetectionResult
from yesno_frame_scorer import YesNoFrameScorer, sample_content
from yesno_scoring import PROMPT, extract_score, windows


def response(yes=.2, no=.6):
    rows = [dict(token='Yes', logprob=math.log(yes)), dict(token='No', logprob=math.log(no))]
    return dict(choices=[dict(logprobs=dict(content=[dict(token='No', logprob=math.log(no), top_logprobs=rows)]))],
                usage=dict(prompt_tokens=10, completion_tokens=2, total_tokens=12))


def visual():
    return dict(source=dict(fps=2., frame_count=24, duration=12.),
                settings=dict(model='qwen3-vl-plus', fusion_weight=.5),
                timestamps=[2.5, 7.5, 11.], frame_scores=[.1, .9, .2],
                exact_pairs=3, bounded_pairs=0, windows=[])


class Client:
    async def score_yesno(self, content, model):
        return response()


def test_cli_config_roundtrip_and_explicit_enable(tmp_path):
    parser = argparse.ArgumentParser()
    add_yesno_arguments(parser)
    assert VADConfig.from_args(parser.parse_args([])).scoring.yesno_enabled
    assert not VADConfig.from_args(parser.parse_args(['--no_yesno_scoring'])).scoring.yesno_enabled
    assert not VADConfig().scoring.yesno_enabled  # legacy programmatic/saved configs
    cfg = VADConfig.from_args(parser.parse_args(['--yesno_scoring', '--yesno_weight', '1']))
    assert cfg.scoring.yesno_enabled and cfg.scoring.yesno_weight == 1
    path = tmp_path/'config.json'
    cfg.save(path)
    assert VADConfig.load(path).to_dict() == cfg.to_dict()


@pytest.mark.parametrize('weight', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_weight_rejected(weight):
    with pytest.raises(ValueError):
        ScoringConfig(yesno_weight=weight)


def test_same_token_probability_and_missing_logits():
    assert extract_score(response())['score'] == pytest.approx(.25)
    with pytest.raises(ValueError, match='Missing actual'):
        extract_score({'choices': [{}]})
    raw = response()
    raw['choices'][0]['logprobs']['content'][0]['token'] = 'Maybe'
    with pytest.raises(ValueError, match='canonical'):
        extract_score(raw)


def test_frozen_prompt_and_sampling_match_v218(tmp_path):
    # Use the experiment code as an independent sampling reference, with no API.
    root = Path(__file__).resolve().parents[2]/'experiments/v218_ucf'
    spec = importlib.util.spec_from_file_location('frozen_v218_scoring', root/'scoring.py')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    assert PROMPT == frozen.PROMPT
    assert windows(12) == frozen.windows(12)
    import ast
    import base64
    import hashlib
    tree = ast.parse((root/'run_experiment.py').read_text(encoding='utf-8'))
    sample_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'sample_content')
    namespace = dict(np=np, cv2=cv2, base64=base64, hashlib=hashlib, PROMPT=frozen.PROMPT)
    exec(compile(ast.Module(body=[sample_fn], type_ignores=[]), '<frozen sampler>', 'exec'), namespace)
    path = tmp_path/'sample.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 2., (32, 32))
    assert writer.isOpened()
    for i in range(24):
        writer.write(np.full((32, 32, 3), i*10, dtype=np.uint8))
    writer.release()
    first, second = cv2.VideoCapture(str(path)), cv2.VideoCapture(str(path))
    row = dict(fps=2., frame_count=24, duration=12.)
    try:
        for win in windows(12):
            actual = sample_content(first, row, win)
            expected = namespace['sample_content'](second, row, win, dict(frames_per_clip=10, jpeg_quality=80))
            assert actual == expected
    finally:
        first.release()
        second.release()
    result = asyncio.run(YesNoFrameScorer(Client(), ScoringConfig()).score_video(path))
    assert result['source'] == row and result['exact_pairs'] == 3
    assert result['frame_scores'] == pytest.approx([.25]*3)
    assert [(w['window']['start'], w['window']['end']) for w in result['windows']] == [(0, 5), (5, 10), (10, 12)]


def test_native_fusion_preserves_segments_and_scores_every_paired_arm(tmp_path):
    scorer = YesNoFrameScorer(Client(), ScoringConfig())
    result = DetectionResult('video.mp4', 12., True, frame_scores=[.2, .6], timestamps=[0., 11.5])
    branch = result.to_dict()
    branch['frame_scores'] = [.4, .1]
    result.paired_results = dict(original_off=branch)
    before = copy.deepcopy(result.to_dict())
    v = visual()
    scorer.apply_scores(result, v)
    new = np.interp(np.arange(24), np.array(v['timestamps'])*2, np.array(v['frame_scores'], dtype=np.float32)).astype(np.float32)
    for actual, base in [(result.to_dict(), before), (branch, before['paired_results']['original_off'])]:
        old = np.interp(np.arange(24), np.array(base['timestamps'])*2, np.array(base['frame_scores'], dtype=np.float32)).astype(np.float32)
        assert np.array_equal(actual['frame_scores'], (.5*old+.5*new).astype(np.float32))
        assert actual['anomalies'] == base['anomalies'] and actual['is_normal'] == base['is_normal']
    path = tmp_path/'video.json'
    path.write_text(json.dumps(result.to_dict()), encoding='utf-8')
    loaded = load_predictions(str(tmp_path))['video']
    assert loaded['source_frame_count'] == 24 and loaded['frame_score_grid'] == 'native_frames'
    gt = dict(video=dict(category='Abuse', segments=[(8, 15)]))
    native = compute_frame_auc(gt, dict(video=loaded), fps=2.)
    # Native timestamps may be rounded for presentation: evaluation must use frame indices.
    loaded['timestamps'] = [round(t) for t in loaded['timestamps']]
    assert compute_frame_auc(gt, dict(video=loaded), fps=2.) == native
    loaded['frame_scores'].pop()
    with pytest.raises(ValueError, match='count mismatch'):
        compute_frame_auc(gt, dict(video=loaded), fps=2.)


@pytest.mark.parametrize('normal', [False, True])
def test_main_detect_hook_includes_normal_early_return(normal):
    from test_mcts_fusion import mock_pipeline
    vad = mock_pipeline('refine', True)
    vad.frame_scorer.compute_frame_scores = lambda vd, *args: [.2]*len(vd['timestamps'])
    if normal:
        vad.videomae.classify = lambda *a, **kw: dict(category='Normal', confidence=.9, top_k=[('Normal', .9)])
    scorer = YesNoFrameScorer(Client(), ScoringConfig(yesno_enabled=True))
    called = []
    async def score_video(path):
        called.append(path)
        return visual()
    scorer.score_video = score_video
    vad.yesno_scorer = scorer
    result = asyncio.run(vad.detect('test'))
    assert called == ['test']
    assert len(result.frame_scores) == 24
    assert result.is_normal == normal
    assert result.yesno_scoring['exact_pairs'] == 3
    assert all(b['frame_score_grid'] == 'native_frames' for b in result.paired_results.values())


def test_probability_failure_does_not_return_success():
    from test_mcts_fusion import mock_pipeline
    vad = mock_pipeline()
    class FailingScorer:
        async def apply(self, *args):
            raise ValueError('Missing actual token log probabilities')
    vad.yesno_scorer = FailingScorer()
    with pytest.raises(ValueError, match='Missing actual'):
        asyncio.run(vad.detect('test'))


def test_client_uses_fixed_logprobs_and_shared_limit_and_counts_tokens():
    async def run():
        client = DashScopeLLMClient(api_key='unit-test-only', model='detector-model', api_concurrency=2)
        sdk = client.client
        calls = []
        active = peak = 0
        async def create(**kwargs):
            nonlocal active, peak
            calls.append(kwargs)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(.005)
            active -= 1
            raw = response()
            return SimpleNamespace(usage=SimpleNamespace(**raw['usage']), model_dump=lambda: raw)
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        @track_video_usage
        async def video(video_path):
            await asyncio.gather(*(client.score_yesno([{'type': 'text', 'text': PROMPT}]) for _ in range(5)))
            return DetectionResult(video_path, 1., True)
        try:
            result = await video('normal.mp4')
        finally:
            await sdk.close()
        assert peak == 2 and client.call_count == 5
        assert result.token_usage['total_tokens'] == 60
        assert result.token_usage['by_model']['qwen3-vl-plus']['requests'] == 5
        for call in calls:
            assert call['model'] == 'qwen3-vl-plus' and call['temperature'] == 0
            assert call['logprobs'] and call['top_logprobs'] == 5 and call['max_tokens'] == 16
            assert call['extra_body'] == {'enable_thinking': False}
    asyncio.run(run())


def test_full_selection_contains_every_normal_video(tmp_path):
    from batch_evaluate import select_videos
    rows = []
    for category, size in [('Abuse', 2), ('Normal', 8)]:
        for i in range(size):
            path = tmp_path/f'{category}_{i}.mp4'
            path.touch()
            rows.append(dict(category=category, video_path=str(path)))
    pd.DataFrame(rows).to_parquet(tmp_path/'test-00000.parquet')
    selected = select_videos(str(tmp_path), None)
    assert len(selected) == 10 and (selected['category'] == 'Normal').sum() == 8
    with pytest.raises(ValueError):
        select_videos(str(tmp_path), 0)


def test_one_window_video_retains_its_probability():
    # A one-point curve is constant evidence, not an absent detector output.
    gt = dict(a=dict(category='Abuse', segments=[(0, 3)]), n=dict(category='Normal', segments=[]))
    pred = {k: dict(duration=2., source_frame_count=4, anomalies=[], frame_scores=[score], timestamps=[1.])
            for k, score in [('a', .9), ('n', .1)]}
    assert compute_frame_auc(gt, pred, fps=2.)['auc'] == 1.
