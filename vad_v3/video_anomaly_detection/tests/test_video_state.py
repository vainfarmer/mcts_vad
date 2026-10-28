import asyncio
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import pipeline
from frame_scorer import FrameScorer
from video_state import fork_video_state
from test_mcts_fusion import mock_pipeline


def video():
    times=np.arange(0.,20.,.5)
    return dict(video_path='mock.mp4',duration=20.,timestamps=times,
                all_frames=np.zeros((len(times),8,8,3),dtype=np.uint8),
                clip_features=torch.randn((len(times),3),generator=torch.Generator().manual_seed(42)),
                motion_energy=np.linspace(0.,1.,len(times)),
                segments=[dict(frame_indices=[0,1,2])],
                branch_cache={'values':[1]}, work_array=np.array([1.,2.]),
                _cat_scores_Existing=np.linspace(0.,1.,len(times)))


def test_shared_read_only_payload_and_isolated_mutable_state():
    original=video();a=fork_video_state(original);b=fork_video_state(original)
    for key in ('all_frames','timestamps','motion_energy','_cat_scores_Existing'):
        assert np.shares_memory(a[key],original[key]) and np.shares_memory(a[key],b[key])
        assert not a[key].flags.writeable and original[key].flags.writeable
        with pytest.raises(ValueError): a[key].flat[0]=9
    assert a['clip_features'].data_ptr()==b['clip_features'].data_ptr()==original['clip_features'].data_ptr()
    a['segments'][0]['frame_indices'].append(3)
    a['branch_cache']['values'].append(2)
    a['work_array'][0]=9
    a['_cat_scores_New']=np.ones(40)
    assert b['segments']==original['segments']==[dict(frame_indices=[0,1,2])]
    assert b['branch_cache']==original['branch_cache']=={'values':[1]}
    assert b['work_array'][0]==original['work_array'][0]==1
    assert '_cat_scores_New' not in b and '_cat_scores_New' not in original


def test_object_arrays_are_not_shared_even_under_known_key():
    original={'all_frames':np.array([{'items':[]}],dtype=object)}
    branch=fork_video_state(original)
    branch['all_frames'][0]['items'].append(1)
    assert original['all_frames'][0]['items']==[]


def test_real_frame_scoring_and_full_tail_match_deepcopy():
    original=video();before=copy.deepcopy(original)
    vad=mock_pipeline()
    vad.config.scoring.qscore_rerank=True
    vad.config.scoring.bue_enabled=True
    vad.config.scoring.cbt_enabled=True
    vad.config.mcts.final_nms_enabled=True
    expert=SimpleNamespace(model=SimpleNamespace(logit_scale=torch.tensor(1.)),
                           text_features=torch.tensor([[1.,0.,0.],[0.,1.,0.]]))
    vad.frame_scorer=FrameScorer(expert,vad.config,['Shoplifting','Normal'])
    seeds=[dict(t_start=2.,t_end=8.,confidence=.9,anomaly_type='Shoplifting',source_candidate_ids=[0]),
           dict(t_start=12.,t_end=18.,confidence=.8,anomaly_type='Shoplifting',source_candidate_ids=[1])]
    async def run(state):
        return await vad._finish_from_stage5(copy.deepcopy(seeds),state,'mock.mp4','Shoplifting',.9,set(),set())
    old=asyncio.run(run(copy.deepcopy(original))).to_dict()
    state=fork_video_state(original)
    new=asyncio.run(run(state)).to_dict()
    assert old==new
    assert '_cat_scores_Shoplifting' in state and '_cat_scores_Shoplifting' not in original
    torch.testing.assert_close(original['clip_features'],before['clip_features'],rtol=0,atol=0)
    for key in ('all_frames','timestamps','motion_energy','_cat_scores_Existing'):
        np.testing.assert_array_equal(original[key],before[key])


def without_timings(value):
    if isinstance(value,dict):
        return {k:without_timings(v) for k,v in value.items()
                if k not in ('elapsed_seconds','stage_seconds','tail_seconds')}
    if isinstance(value,list):return [without_timings(v) for v in value]
    return value


def test_all_paired_arms_share_payload_but_keep_predictions(monkeypatch):
    def build():
        vad=mock_pipeline('refine',True)
        cfg=vad.config.mcts
        cfg.postprocess_ablation=cfg.final_nms_enabled=cfg.boundary_guard_enabled=True
        return vad
    monkeypatch.setattr(pipeline,'fork_video_state',copy.deepcopy)
    old=asyncio.run(build().detect('x')).to_dict()
    branches=[];originals=[]
    def checked_fork(data):
        branch=fork_video_state(data);branches.append(branch);originals.append(data)
        assert np.shares_memory(branch['all_frames'],data['all_frames'])
        return branch
    monkeypatch.setattr(pipeline,'fork_video_state',checked_fork)
    new=asyncio.run(build().detect('x')).to_dict()
    assert len(branches)==6 and len({id(x) for x in branches})==6
    assert all(source is originals[0] for source in originals)
    assert without_timings(old)==without_timings(new)


def test_concurrent_forks_do_not_share_new_cache_entries():
    original=video()
    async def exercise():
        async def worker(value):
            state=fork_video_state(original)
            state['branch_cache']['values'].append(value)
            state['_cat_scores_New']=np.full(2,value)
            await asyncio.sleep(0)
            return state
        a,b=await asyncio.gather(worker(2),worker(3))
        assert a['branch_cache']['values']==[1,2] and b['branch_cache']['values']==[1,3]
        np.testing.assert_array_equal(a['_cat_scores_New'],[2,2])
        np.testing.assert_array_equal(b['_cat_scores_New'],[3,3])
    asyncio.run(exercise())
    assert original['branch_cache']=={'values':[1]}
