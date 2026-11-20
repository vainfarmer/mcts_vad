"""Regressions for merging main's AUC scorer with the independent XD route."""
import asyncio
import json
import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from config import VADConfig
from models import DetectionResult
from yesno_frame_scorer import YesNoFrameScorer
from yesno_scoring import PROMPT, XD_PROMPT, prompt_for_dataset
from xd_config import XD_CLASSES
import xd_dataset
import xd_evaluate
import xd_pipeline


class Client:
    max_frames = 16

    def __init__(self, normal=False, invalid=False):
        self.normal, self.invalid = normal, invalid
        self.probability_calls = []

    async def generate(self, prompt, images):
        events = [] if self.normal else [dict(category=c, t_start=1, t_end=3,
            confidence=.8, evidence_frame_ids=['f2']) for c in ('Riot', 'Fighting')]
        return json.dumps(dict(status='ok', events=events))

    async def score_yesno(self, content, model):
        self.probability_calls.append(content)
        if self.invalid:
            return {'choices': [{}]}
        rows = [dict(token='Yes', logprob=math.log(.6)), dict(token='No', logprob=math.log(.2))]
        return dict(choices=[dict(logprobs=dict(content=[dict(token='Yes',
            logprob=math.log(.6), top_logprobs=rows)]))])


def pipeline(client):
    vad = xd_pipeline.XDMultiAnomalyVAD.__new__(xd_pipeline.XDMultiAnomalyVAD)
    vad.config = VADConfig(dataset='xd_violence')
    frames = np.zeros((8, 32, 32, 3), dtype=np.uint8)
    data = dict(all_frames=frames, timestamps=np.arange(8)*.5, duration=4., video_fps=2.)
    vad.preprocessor = SimpleNamespace(process=lambda path: data)
    vad.anomaly_expert = SimpleNamespace(precompute_video_features=lambda images: None)
    vad.frame_scorer = SimpleNamespace(get_category_scores=lambda data, category: np.full(8, .5))
    window = dict(t_start=0., t_end=4., scores={c: .1 for c in XD_CLASSES})
    vad.videomae = SimpleNamespace(predict=lambda *args: dict(windows=[window], scores={}))
    vad.classify_llm_client = vad.llm_client = client
    return vad


@pytest.fixture
def video(tmp_path):
    path = tmp_path/'hidden_title_label_B4-0-0.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 2., (32, 32))
    assert writer.isOpened()
    for _ in range(8):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    return str(path)


def test_xd_constructor_wires_dataset_specific_scorer(monkeypatch):
    monkeypatch.setattr(xd_pipeline, 'VideoPreprocessor', lambda **kw: SimpleNamespace())
    monkeypatch.setattr(xd_pipeline, 'VideoAnomalyExpert', lambda *a, **kw: SimpleNamespace(model=object()))
    monkeypatch.setattr(xd_pipeline, 'XDVideoMAEPrior', lambda *a: object())
    cfg = VADConfig(dataset='xd_violence')
    cfg.scoring.yesno_enabled = True
    vad = xd_pipeline.XDMultiAnomalyVAD(Client(), cfg)
    assert vad.yesno_scorer.dataset == 'xd_violence'
    assert vad.yesno_scorer.prompt == XD_PROMPT
    assert prompt_for_dataset('ucf_crime') == PROMPT
    assert all(c in XD_PROMPT for c in XD_CLASSES)
    assert 'shoplifting' not in XD_PROMPT.lower()
    with pytest.raises(ValueError):
        prompt_for_dataset('unknown')


@pytest.mark.parametrize('normal', [False, True])
def test_xd_score_pass_preserves_events_and_covers_normal(video, normal):
    client = Client(normal=normal)
    vad = pipeline(client)
    baseline = asyncio.run(vad.detect(video))
    assert not client.probability_calls
    vad.yesno_scorer = YesNoFrameScorer(client, vad.config.scoring, dataset='xd_violence')
    result = asyncio.run(vad.detect(video))
    assert result.is_normal == baseline.is_normal == normal
    assert result.to_dict()['anomalies'] == baseline.to_dict()['anomalies']
    assert result.category_prior == baseline.category_prior
    assert result.dataset == 'xd_violence'
    assert result.source_frame_count == len(result.frame_scores) == 8
    assert result.yesno_scoring['method'] == 'xd_yesno_fusion'
    assert result.yesno_scoring['settings']['dataset'] == 'xd_violence'
    assert np.allclose(result.frame_scores, .5*np.array(baseline.frame_scores, dtype=np.float32)+.375)
    assert len(client.probability_calls) == 1
    text = '\n'.join(row['text'] for row in client.probability_calls[0] if row['type'] == 'text')
    assert XD_PROMPT in text and 'Target interval: 0.000 to 4.000 seconds.' in text
    assert 'hidden_title' not in text and '_label_B4' not in text
    assert 'diagnostics' in result.to_dict() and 'yesno_scoring' in result.to_dict()


def test_xd_missing_logprobs_fails_video(video):
    client = Client(normal=True, invalid=True)
    vad = pipeline(client)
    vad.yesno_scorer = YesNoFrameScorer(client, vad.config.scoring, dataset='xd_violence')
    with pytest.raises(ValueError, match='Missing actual'):
        asyncio.run(vad.detect(video))


def test_full_xd_selection_deduplicates_multilabel_and_keeps_all_normal(monkeypatch):
    records = {
        'multi': dict(video_id='multi', video_path='multi.mp4', category='Riot',
                      video_labels=['Riot', 'Fighting'], events=[dict(category=c) for c in ('Riot', 'Fighting')]),
        **{f'n{i}': dict(video_id=f'n{i}', video_path=f'n{i}.mp4', category='Normal',
                         video_labels=['Normal'], events=[]) for i in range(7)},
    }
    monkeypatch.setattr(xd_dataset, 'load_xd_index', lambda *args: records)
    rows = xd_dataset.select_xd_videos('unused', None)
    assert len(rows) == 8 and rows['video_id'].is_unique
    assert (rows['category'] == 'Normal').sum() == 7
    assert rows.attrs['selection']['mode'] == 'full_test_split'
    assert rows.attrs['selection']['slots']['Riot'] == rows.attrs['selection']['slots']['Fighting'] == ['multi']


def test_xd_evaluation_preserves_ap_and_half_open_gt_on_native_scores(tmp_path, monkeypatch):
    records = {
        'a': dict(video_path='a.mp4', category='Fighting', segments=[(1, 3)],
                  events=[dict(category='Fighting', start_frame=1, end_frame=3)]),
        'n': dict(video_path='n.mp4', category='Normal', segments=[], events=[]),
    }
    metadata = {k: dict(frame_count=4, fps=2.) for k in records}
    monkeypatch.setattr(xd_evaluate, 'load_xd_index', lambda *args: records)
    monkeypatch.setattr(xd_evaluate, 'source_video_metadata', lambda *args: metadata)
    monkeypatch.setattr(xd_evaluate, 'annotation_fingerprint', lambda *args: {})
    curves = {'a': [.1, .9, .8, .2], 'n': [.3, .4, .1, .1]}
    for name, scores in curves.items():
        value = DetectionResult(f'{name}.mp4', 2., True, dataset='xd_violence',
                                frame_scores=scores, timestamps=[0., .1, .2, .3],
                                source_frame_count=4, frame_score_grid='native_frames').to_dict()
        (tmp_path/f'{name}.json').write_text(json.dumps(value), encoding='utf-8')
    result = xd_evaluate.evaluate_xd('unused', tmp_path, ['a', 'n'])
    y = [0, 1, 1, 0, 0, 0, 0, 0]
    values = curves['a']+curves['n']
    assert result['frame_AP'] == average_precision_score(y, values)
    assert result['frame_AUC'] == roc_auc_score(y, values)
    assert result['normal_video_count'] == 1
    assert result['protocol']['interval'] == 'source frames [start,end)'
    assert result['protocol']['scope'] == 'full test split'
    bad = json.loads((tmp_path/'n.json').read_text())
    bad['frame_scores'].pop()
    bad['timestamps'].pop()
    (tmp_path/'n.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='count mismatch'):
        xd_evaluate.evaluate_xd('unused', tmp_path, ['a', 'n'])
