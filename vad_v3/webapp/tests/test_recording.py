"""Evidence must describe real inputs; response errors must not become Normal."""
import base64
import json
from types import SimpleNamespace
import numpy as np
import pytest
from vad_v3.video_anomaly_detection.anomaly_discovery import AnomalyDiscovery
from vad_v3.video_anomaly_detection.research_recording import Recorder,scope


@pytest.mark.parametrize('response',[
    'not JSON','{}','{"is_anomalous":"false","confidence":0.9}',
    '{"is_anomalous":false,"confidence":NaN}',
    '{"is_anomalous":false,"confidence":2}',
])
def test_invalid_discovery_response_never_becomes_normal(response):
    discovery=AnomalyDiscovery(None);discovery.strict_responses=True
    with pytest.raises(ValueError,match='VLM_DISCOVERY_INVALID_RESPONSE'):
        discovery._parse_response(response)


def test_description_preserved_in_original_language():
    discovery=AnomalyDiscovery(None);discovery.strict_responses=True
    row=discovery._parse_response('{"is_anomalous":true,"confidence":0.8,"description":"A person is running."}')
    assert row['original_description']=='A person is running.'
    normal=discovery._parse_response('{"is_anomalous":false,"confidence":0.9,"description":null}')
    assert not normal['is_suspicious'] and normal['original_description'] is None


def test_evidence_bytes_and_duplicate_frame_times_are_not_fabricated(tmp_path):
    recorder=Recorder(tmp_path)
    frames=np.zeros((2,3,3,3),dtype=np.uint8)
    recorder.preprocessed(dict(all_frames=frames,timestamps=[1.,7.]))
    original=b'exact encoded image sent to model'
    content=[dict(type='image_url',image_url={'url':'data:image/jpeg;base64,'+base64.b64encode(original).decode()})]
    row=recorder.request('original prompt',frames[:1],content)
    assert row['images'][0]['source_times']==[1.,7.]
    assert row['images'][0]['time_status']=='ambiguous_or_unrecorded'
    assert (tmp_path/row['images'][0]['file']).read_bytes()==original
    token=scope.set(dict(stage='mcts',evidence_times=[7.]))
    try: other=recorder.request('same pixels, explicit sampling time',frames[:1],content)
    finally: scope.reset(token)
    assert other['images'][0]['source_times']==[7.]
    recorder.response(row,'raw response')
    recorder.snapshot('stage',[])
    recorder.finish(SimpleNamespace(videomae_category=None,anomalies=[]))
    data=json.loads((tmp_path/'research.json').read_text(encoding='utf-8'))
    assert data['evidence'][0]['response']=='raw response'
    assert data['exit_reason']=='completed_without_detections'
    assert data['stage_order']==['stage']
